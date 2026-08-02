import io
import socket
import time
from types import SimpleNamespace

from PIL import Image

from openpilot.selfdrive.carrot.external_ai.frame_protocol import VideoFrame, encode_video_frame, receive_video_frame
from openpilot.selfdrive.carrot.external_ai.frame_sender import LatestFrameSlot, VideoFrameTcpServer, nv12_to_jpeg


def frame(frame_id: int) -> VideoFrame:
  return VideoFrame(frame_id, 8_000_000_000 + frame_id, 640, 360, b"\xff\xd8jpeg\xff\xd9")


def test_latest_frame_slot_replaces_pending_frame() -> None:
  slot = LatestFrameSlot()
  slot.put(b"old")
  slot.put(b"new")
  assert slot.wait_after(0, 0.0) == (2, b"new")
  assert slot.frames_replaced == 1


def test_tcp_server_sends_framed_video_to_phone_client() -> None:
  server = VideoFrameTcpServer(port=0, host="127.0.0.1")
  server.start()
  try:
    with socket.create_connection(("127.0.0.1", server.bound_port), timeout=1.0) as client:
      deadline = time.monotonic() + 1.0
      while not server.client_connected.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
      server.slot.put(encode_video_frame(frame(7)))
      assert receive_video_frame(client) == frame(7)
      deadline = time.monotonic() + 1.0
      while server.stats().frames_sent != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
      assert server.stats().frames_sent == 1
  finally:
    server.stop()


def test_tcp_server_rejects_non_matching_phone_ip() -> None:
  server = VideoFrameTcpServer(port=0, host="127.0.0.1", allowed_phone_ip="192.0.2.1")
  server.start()
  try:
    with socket.create_connection(("127.0.0.1", server.bound_port), timeout=1.0) as client:
      client.settimeout(1.0)
      assert client.recv(1) == b""
    assert server.stats().clients_rejected == 1
  finally:
    server.stop()


def test_nv12_frame_is_scaled_and_encoded_as_jpeg() -> None:
  width, height, stride = 4, 4, 4
  y_plane = bytes([128] * (stride * height))
  uv_plane = bytes([128] * (stride * 16))
  buf = SimpleNamespace(
    data=y_plane + uv_plane,
    width=width,
    height=height,
    stride=stride,
    uv_offset=len(y_plane),
  )
  jpeg = nv12_to_jpeg(buf, width=8, height=4, quality=70)
  with Image.open(io.BytesIO(jpeg)) as image:
    assert image.format == "JPEG"
    assert image.size == (8, 4)
