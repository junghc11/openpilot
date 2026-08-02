from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any


FRAME_MAGIC = b"CAI1"
FRAME_PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 4_096
MAX_JPEG_BYTES = 2 * 1024 * 1024
_PREFIX = struct.Struct("!4sII")


class FrameProtocolError(ValueError):
  pass


@dataclass(frozen=True)
class VideoFrame:
  frame_id: int
  source_timestamp_monotonic_ns: int
  width: int
  height: int
  jpeg: bytes


def _positive_int(value: Any, name: str, maximum: int) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
    raise FrameProtocolError(f"{name} must be an integer between 1 and {maximum}")
  return value


def encode_video_frame(frame: VideoFrame) -> bytes:
  _positive_int(frame.frame_id, "frame_id", (1 << 63) - 1)
  _positive_int(frame.source_timestamp_monotonic_ns, "source_timestamp_monotonic_ns", (1 << 63) - 1)
  _positive_int(frame.width, "width", 8_192)
  _positive_int(frame.height, "height", 8_192)
  if not frame.jpeg or len(frame.jpeg) > MAX_JPEG_BYTES:
    raise FrameProtocolError(f"jpeg must contain between 1 and {MAX_JPEG_BYTES} bytes")

  header = json.dumps({
    "protocol_version": FRAME_PROTOCOL_VERSION,
    "frame_id": frame.frame_id,
    "source_timestamp_monotonic_ns": frame.source_timestamp_monotonic_ns,
    "width": frame.width,
    "height": frame.height,
    "encoding": "jpeg",
  }, separators=(",", ":")).encode("utf-8")
  if len(header) > MAX_HEADER_BYTES:
    raise FrameProtocolError("frame header is too large")
  return _PREFIX.pack(FRAME_MAGIC, len(header), len(frame.jpeg)) + header + frame.jpeg


def recv_exact(sock: socket.socket, size: int) -> bytes:
  data = bytearray()
  while len(data) < size:
    chunk = sock.recv(size - len(data))
    if not chunk:
      raise EOFError("video frame connection closed")
    data.extend(chunk)
  return bytes(data)


def receive_video_frame(sock: socket.socket) -> VideoFrame:
  magic, header_size, jpeg_size = _PREFIX.unpack(recv_exact(sock, _PREFIX.size))
  if magic != FRAME_MAGIC:
    raise FrameProtocolError("invalid video frame magic")
  if not 1 <= header_size <= MAX_HEADER_BYTES:
    raise FrameProtocolError("invalid video frame header size")
  if not 1 <= jpeg_size <= MAX_JPEG_BYTES:
    raise FrameProtocolError("invalid video frame JPEG size")
  try:
    header = json.loads(recv_exact(sock, header_size).decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise FrameProtocolError("invalid video frame header JSON") from exc
  if not isinstance(header, dict):
    raise FrameProtocolError("video frame header must be an object")
  if header.get("protocol_version") != FRAME_PROTOCOL_VERSION:
    raise FrameProtocolError("unsupported video frame protocol version")
  if header.get("encoding") != "jpeg":
    raise FrameProtocolError("unsupported video frame encoding")

  return VideoFrame(
    frame_id=_positive_int(header.get("frame_id"), "frame_id", (1 << 63) - 1),
    source_timestamp_monotonic_ns=_positive_int(
      header.get("source_timestamp_monotonic_ns"), "source_timestamp_monotonic_ns", (1 << 63) - 1,
    ),
    width=_positive_int(header.get("width"), "width", 8_192),
    height=_positive_int(header.get("height"), "height", 8_192),
    jpeg=recv_exact(sock, jpeg_size),
  )
