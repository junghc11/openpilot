from __future__ import annotations

import io
import socket
import threading
import time
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from typing import Any

from openpilot.selfdrive.carrot.external_ai.frame_protocol import H264Frame, VideoFrame, encode_h264_frame, encode_video_frame


DEFAULT_FRAME_PORT = 7724
DEFAULT_FRAME_FPS = 5
DEFAULT_JPEG_QUALITY = 75
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360
# qRoadEncodeData is produced by the normal on-road encoder and is therefore the
# most predictable H.264 source on C3/C3X/C4. YouTube/external_ai_encoderd remains
# a fallback, but it must not prevent the phone from using the always-on qRoad feed.
H264_SOURCE = "qRoadEncodeData"
H264_FALLBACK_SOURCE = "youtubeRoadEncodeData"
H264_FALLBACK_TIMEOUT_S = 3.0
H264_SOURCE_STALE_TIMEOUT_S = 1.0
H264_QUEUE_MAX_FRAMES = 60
H264_QUEUE_MAX_BYTES = 2 * 1024 * 1024


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

  def reset(self) -> None:
    with self._condition:
      self._packet = b""


class AdaptiveFrameQueue:
  """Latest-only JPEG queue and ordered H.264 queue with keyframe recovery."""

  def __init__(self, *, max_h264_frames: int = H264_QUEUE_MAX_FRAMES,
               max_h264_bytes: int = H264_QUEUE_MAX_BYTES) -> None:
    self._condition = threading.Condition()
    self._packets: deque[tuple[int, bytes]] = deque()
    self._generation = 0
    self._queued_bytes = 0
    self._mode = ""
    self._waiting_for_keyframe = True
    self._max_h264_frames = max_h264_frames
    self._max_h264_bytes = max_h264_bytes
    self.frames_queued = 0
    self.frames_replaced = 0

  def put(self, packet: bytes, *, encoding: str = "jpeg", keyframe: bool = False) -> None:
    if encoding not in ("jpeg", "h264"):
      raise ValueError(f"unsupported queue encoding {encoding}")
    with self._condition:
      self.frames_queued += 1
      self._generation += 1
      generation = self._generation

      if encoding == "jpeg":
        self.frames_replaced += len(self._packets)
        self._packets.clear()
        self._queued_bytes = 0
        self._mode = "jpeg"
        self._waiting_for_keyframe = True
        self._packets.append((generation, packet))
        self._queued_bytes = len(packet)
        self._condition.notify_all()
        return

      if self._mode != "h264":
        self.frames_replaced += len(self._packets)
        self._packets.clear()
        self._queued_bytes = 0
        self._mode = "h264"
        self._waiting_for_keyframe = True

      if self._waiting_for_keyframe:
        if not keyframe:
          self.frames_replaced += 1
          return
        self._waiting_for_keyframe = False

      if len(self._packets) >= self._max_h264_frames or self._queued_bytes + len(packet) > self._max_h264_bytes:
        self.frames_replaced += len(self._packets)
        self._packets.clear()
        self._queued_bytes = 0
        self._waiting_for_keyframe = not keyframe
        if not keyframe:
          self.frames_replaced += 1
          return

      self._packets.append((generation, packet))
      self._queued_bytes += len(packet)
      self._condition.notify_all()

  def wait_after(self, generation: int, timeout_s: float) -> tuple[int, bytes] | None:
    del generation  # Queue order, rather than generation comparison, is authoritative for H.264.
    with self._condition:
      self._condition.wait_for(lambda: bool(self._packets), timeout=timeout_s)
      if not self._packets:
        return None
      packet_generation, packet = self._packets.popleft()
      self._queued_bytes -= len(packet)
      return packet_generation, packet

  def reset(self) -> None:
    with self._condition:
      self.frames_replaced += len(self._packets)
      self._packets.clear()
      self._queued_bytes = 0
      self._mode = ""
      self._waiting_for_keyframe = True

  def wake(self) -> None:
    with self._condition:
      self._condition.notify_all()


class VideoFrameTcpServer:
  def __init__(self, *, port: int = DEFAULT_FRAME_PORT, allowed_phone_ip: str = "", host: str = "0.0.0.0",
               slot: LatestFrameSlot | AdaptiveFrameQueue | None = None) -> None:
    self.port = port
    self.allowed_phone_ip = allowed_phone_ip.strip()
    self.host = host
    self.slot = slot or LatestFrameSlot()
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
    self.slot.reset()
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


def h264_packet_from_encode_data(encoded: Any) -> tuple[bytes, bool] | None:
  data = bytes(getattr(encoded, "data", b"") or b"")
  if not data:
    return None
  header = bytes(getattr(encoded, "header", b"") or b"")
  idx = getattr(encoded, "idx", None)
  frame_id = int(getattr(idx, "frameId", 0) or 0)
  timestamp_ns = int(getattr(idx, "timestampEof", 0) or 0)
  flags = int(getattr(idx, "flags", 0) or 0)
  width = int(getattr(encoded, "width", 0) or 0)
  height = int(getattr(encoded, "height", 0) or 0)
  if frame_id <= 0 or timestamp_ns <= 0 or width <= 0 or height <= 0:
    return None
  keyframe = bool(header) or bool(flags & 0x8)
  packet = encode_h264_frame(H264Frame(
    frame_id=frame_id,
    source_timestamp_monotonic_ns=timestamp_ns,
    width=width,
    height=height,
    keyframe=keyframe,
    codec_config=header if keyframe else b"",
    data=data,
  ))
  return packet, keyframe


class RoadFrameCapture:
  def __init__(self, server: VideoFrameTcpServer, *, fps: int = DEFAULT_FRAME_FPS,
               jpeg_quality: int = DEFAULT_JPEG_QUALITY,
               should_encode: Callable[[], bool] | None = None) -> None:
    self.server = server
    self.fps = fps
    self.jpeg_quality = jpeg_quality
    self.should_encode = should_encode
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
      if self.should_encode is not None and not self.should_encode():
        self._stop.wait(0.02)
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
      # H.264 may have recovered while the comparatively expensive JPEG was
      # being encoded. Do not let that stale fallback displace the new stream.
      if self.should_encode is not None and not self.should_encode():
        continue
      self.server.slot.put(packet)
      next_frame_at = now + 1.0 / self.fps


class H264FrameCapture:
  """Relays the best live H.264 source and changes sources only at an IDR."""

  def __init__(self, server: VideoFrameTcpServer, messaging_module: Any | None = None,
               *, source: str = H264_SOURCE,
               fallback_source: str = H264_FALLBACK_SOURCE) -> None:
    self.server = server
    self.messaging = messaging_module
    self.source = source
    self.sources = tuple(dict.fromkeys((source, fallback_source)))
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None
    self._state_lock = threading.Lock()
    self._last_frame_at = 0.0
    self._connected_at = 0.0
    self._active_source = ""
    self._last_source_frame_at: dict[str, float] = {}

  @property
  def is_recent(self) -> bool:
    with self._state_lock:
      last_frame_at = self._last_frame_at
    return last_frame_at > 0.0 and (time.monotonic() - last_frame_at) < H264_FALLBACK_TIMEOUT_S

  @property
  def active_source(self) -> str:
    with self._state_lock:
      return self._active_source

  def should_fallback_to_jpeg(self) -> bool:
    """Hold JPEG during H.264 startup/recovery instead of racing the first IDR."""
    now = time.monotonic()
    with self._state_lock:
      if self._last_frame_at > 0.0 and now - self._last_frame_at < H264_FALLBACK_TIMEOUT_S:
        return False
      return self._connected_at > 0.0 and now - self._connected_at >= H264_FALLBACK_TIMEOUT_S

  def start(self) -> None:
    if self._thread is None:
      self._thread = threading.Thread(target=self._run, name="external-ai-h264-capture", daemon=True)
      self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
    self._thread = None

  def _set_connected(self, connected: bool, *, now: float | None = None) -> None:
    with self._state_lock:
      if connected:
        if self._connected_at <= 0.0:
          self._connected_at = time.monotonic() if now is None else now
      else:
        self._connected_at = 0.0
        self._last_frame_at = 0.0
        self._active_source = ""
        self._last_source_frame_at.clear()

  def _accept_source(self, source: str, *, keyframe: bool, now: float) -> tuple[bool, bool]:
    """Return (accept, switched); a decoder-changing source switch requires an IDR."""
    with self._state_lock:
      self._last_source_frame_at[source] = now
      active_source = self._active_source
      if source == active_source:
        return True, False

      if not keyframe:
        return False, False

      if not active_source:
        self._active_source = source
        return True, True

      source_priority = self.sources.index(source)
      active_priority = self.sources.index(active_source)
      active_last_frame_at = self._last_source_frame_at.get(active_source, 0.0)
      if source_priority < active_priority or now - active_last_frame_at >= H264_SOURCE_STALE_TIMEOUT_S:
        self._active_source = source
        return True, True
      return False, False

  def _mark_frame_sent(self, now: float) -> None:
    with self._state_lock:
      self._last_frame_at = now

  def _run(self) -> None:
    if self.messaging is None:
      from openpilot.cereal import messaging as messaging_module
      self.messaging = messaging_module
    socks: dict[str, Any] = {}
    try:
      while not self._stop.is_set():
        if not self.server.client_connected.wait(timeout=0.25):
          for sock in socks.values():
            sock.close()
          socks.clear()
          self._set_connected(False)
          continue
        self._set_connected(True)
        if not socks:
          socks = {source: self.messaging.sub_sock(source, conflate=False) for source in self.sources}

        received_message = False
        for source, sock in socks.items():
          message = self.messaging.recv_one_or_none(sock)
          if message is None:
            continue
          received_message = True
          encoded = getattr(message, source, None)
          if encoded is None:
            continue
          try:
            encoded_packet = h264_packet_from_encode_data(encoded)
          except Exception as exc:
            print(f"External AI H.264 packet rejected from {source}: {exc}", flush=True)
            continue
          if encoded_packet is None:
            continue
          packet, keyframe = encoded_packet
          now = time.monotonic()
          accept, switched = self._accept_source(source, keyframe=keyframe, now=now)
          if not accept:
            continue
          slot = self.server.slot
          if switched:
            slot.reset()
            print(f"External AI H.264 source: {source}", flush=True)
          if isinstance(slot, AdaptiveFrameQueue):
            slot.put(packet, encoding="h264", keyframe=keyframe)
          else:
            slot.put(packet)
          self._mark_frame_sent(now)
        if not received_message:
          self._stop.wait(0.005)
    finally:
      for sock in socks.values():
        sock.close()
      self._set_connected(False)
