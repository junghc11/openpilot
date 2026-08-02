from __future__ import annotations

import io
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from openpilot.selfdrive.carrot.external_ai.frame_protocol import VideoFrame, encode_video_frame


DEFAULT_FRAME_PORT = 7724
DEFAULT_FRAME_FPS = 5
DEFAULT_JPEG_QUALITY = 75
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360


@dataclass(frozen=True)
class FrameSenderStats:
  connected: bool
  frames_queued: int
  frames_sent: int
  frames_replaced: int
  clients_rejected: int


class LatestFrameSlot:
  """A one-item queue that replaces old video instead of accumulating latency."""

  def __init__(self) -> None:
    self._condition = threading.Condition()
    self._generation = 0
    self._packet = b""
    self.frames_queued = 0
    self.frames_replaced = 0

  def put(self, packet: bytes) -> None:
    with self._condition:
      if self._packet:
        self.frames_replaced += 1
      self._packet = packet
      self._generation += 1
      self.frames_queued += 1
      self._condition.notify_all()

  def wait_after(self, generation: int, timeout_s: float) -> tuple[int, bytes] | None:
    with self._condition:
      self._condition.wait_for(lambda: self._generation > generation and bool(self._packet), timeout=timeout_s)
      if self._generation <= generation or not self._packet:
        return None
      packet = self._packet
      self._packet = b""
      return self._generation, packet

  def wake(self) -> None:
    with self._condition:
      self._condition.notify_all()


class VideoFrameTcpServer:
  def __init__(self, *, port: int = DEFAULT_FRAME_PORT, allowed_phone_ip: str = "", host: str = "0.0.0.0") -> None:
    self.port = port
    self.allowed_phone_ip = allowed_phone_ip.strip()
    self.host = host
    self.slot = LatestFrameSlot()
    self.client_connected = threading.Event()
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._server: socket.socket | None = None
    self._client: socket.socket | None = None
    self._frames_sent = 0
    self._clients_rejected = 0

  @property
  def bound_port(self) -> int:
    return self._server.getsockname()[1] if self._server is not None else self.port

  def start(self) -> None:
    if self._thread is not None:
      return
    self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._server.bind((self.host, self.port))
    self._server.listen(1)
    self._server.settimeout(0.5)
    self._thread = threading.Thread(target=self._run, name="external-ai-frame-server", daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    self.slot.wake()
    for sock in (self._client, self._server):
      if sock is not None:
        try:
          sock.close()
        except OSError:
          pass
    if self._thread is not None:
      self._thread.join(timeout=2.0)
    self._thread = None
    self._server = None
    self._client = None
    self.client_connected.clear()

  def stats(self) -> FrameSenderStats:
    return FrameSenderStats(
      connected=self.client_connected.is_set(),
      frames_queued=self.slot.frames_queued,
      frames_sent=self._frames_sent,
      frames_replaced=self.slot.frames_replaced,
      clients_rejected=self._clients_rejected,
    )

  def _run(self) -> None:
    assert self._server is not None
    while not self._stop.is_set():
      try:
        client, address = self._server.accept()
      except TimeoutError:
        continue
      except OSError:
        break
      if self.allowed_phone_ip and address[0] != self.allowed_phone_ip:
        self._clients_rejected += 1
        client.close()
        continue
      self._serve_client(client)

  def _serve_client(self, client: socket.socket) -> None:
    self._client = client
    client.settimeout(2.0)
    self.client_connected.set()
    generation = 0
    try:
      while not self._stop.is_set():
        next_frame = self.slot.wait_after(generation, timeout_s=0.5)
        if next_frame is None:
          continue
        generation, packet = next_frame
        client.sendall(packet)
        self._frames_sent += 1
    except OSError:
      pass
    finally:
      self.client_connected.clear()
      try:
        client.close()
      except OSError:
        pass
      self._client = None


def nv12_to_jpeg(buf: Any, *, width: int = OUTPUT_WIDTH, height: int = OUTPUT_HEIGHT,
                 quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
  import numpy as np
  from PIL import Image

  uv_height = ((buf.height // 2) + 15) // 16 * 16
  uv_plane_size = buf.stride * uv_height
  y = np.frombuffer(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
  uv_data = buf.data[buf.uv_offset:buf.uv_offset + uv_plane_size]
  uv = np.frombuffer(uv_data, dtype=np.uint8).reshape((-1, buf.stride))
  u = uv[:, 0::2][:buf.height // 2, :buf.width // 2]
  v = uv[:, 1::2][:buf.height // 2, :buf.width // 2]
  y_small = np.asarray(Image.fromarray(y).resize((width, height), Image.Resampling.BILINEAR))
  uv_size = (width // 2, height // 2)
  u_small = np.asarray(Image.fromarray(u).resize(uv_size, Image.Resampling.BILINEAR))
  v_small = np.asarray(Image.fromarray(v).resize(uv_size, Image.Resampling.BILINEAR))
  u_full = np.repeat(np.repeat(u_small, 2, axis=0), 2, axis=1)
  v_full = np.repeat(np.repeat(v_small, 2, axis=0), 2, axis=1)
  image = Image.merge("YCbCr", (
    Image.fromarray(y_small),
    Image.fromarray(u_full),
    Image.fromarray(v_full),
  ))
  output = io.BytesIO()
  image.save(output, "JPEG", quality=quality, optimize=False)
  return output.getvalue()


class RoadFrameCapture:
  def __init__(self, server: VideoFrameTcpServer, *, fps: int = DEFAULT_FRAME_FPS,
               jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> None:
    self.server = server
    self.fps = fps
    self.jpeg_quality = jpeg_quality
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    if self._thread is None:
      self._thread = threading.Thread(target=self._run, name="external-ai-road-capture", daemon=True)
      self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
    self._thread = None

  def _run(self) -> None:
    from msgq.visionipc import VisionIpcClient, VisionStreamType

    client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, conflate=True)
    next_frame_at = 0.0
    while not self._stop.is_set():
      if not self.server.client_connected.wait(timeout=0.25):
        continue
      if not client.is_connected():
        if not client.connect(False):
          self._stop.wait(0.5)
          continue
      frame = client.recv(timeout_ms=100)
      if frame is None:
        continue
      now = time.monotonic()
      if now < next_frame_at:
        continue
      try:
        source_timestamp_ns = time.monotonic_ns()
        jpeg = nv12_to_jpeg(frame, quality=self.jpeg_quality)
        packet = encode_video_frame(VideoFrame(
          frame_id=max(1, int(client.frame_id)),
          source_timestamp_monotonic_ns=source_timestamp_ns,
          width=OUTPUT_WIDTH,
          height=OUTPUT_HEIGHT,
          jpeg=jpeg,
        ))
      except Exception as exc:
        print(f"External AI frame encode failed: {exc}", flush=True)
        self._stop.wait(0.5)
        continue
      self.server.slot.put(packet)
      next_frame_at = now + 1.0 / self.fps
