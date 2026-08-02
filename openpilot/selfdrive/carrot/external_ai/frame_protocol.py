from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any


FRAME_MAGIC = b"CAI1"
H264_FRAME_MAGIC = b"CAI2"
FRAME_PROTOCOL_VERSION = 1
H264_FRAME_PROTOCOL_VERSION = 2
MAX_HEADER_BYTES = 4_096
MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_H264_BYTES = 4 * 1024 * 1024
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


@dataclass(frozen=True)
class H264Frame:
  frame_id: int
  source_timestamp_monotonic_ns: int
  width: int
  height: int
  keyframe: bool
  codec_config: bytes
  data: bytes


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


def encode_h264_frame(frame: H264Frame) -> bytes:
  _positive_int(frame.frame_id, "frame_id", (1 << 63) - 1)
  _positive_int(frame.source_timestamp_monotonic_ns, "source_timestamp_monotonic_ns", (1 << 63) - 1)
  _positive_int(frame.width, "width", 8_192)
  _positive_int(frame.height, "height", 8_192)
  if not isinstance(frame.keyframe, bool):
    raise FrameProtocolError("keyframe must be boolean")
  if not frame.data:
    raise FrameProtocolError("H.264 data must not be empty")
  if frame.codec_config and not frame.keyframe:
    raise FrameProtocolError("H.264 codec config is only valid on a keyframe")
  payload = frame.codec_config + frame.data
  if len(payload) > MAX_H264_BYTES:
    raise FrameProtocolError(f"H.264 payload must be at most {MAX_H264_BYTES} bytes")

  header = json.dumps({
    "protocol_version": H264_FRAME_PROTOCOL_VERSION,
    "frame_id": frame.frame_id,
    "source_timestamp_monotonic_ns": frame.source_timestamp_monotonic_ns,
    "width": frame.width,
    "height": frame.height,
    "encoding": "h264",
    "keyframe": frame.keyframe,
    "codec_config_size": len(frame.codec_config),
  }, separators=(",", ":")).encode("utf-8")
  if len(header) > MAX_HEADER_BYTES:
    raise FrameProtocolError("frame header is too large")
  return _PREFIX.pack(H264_FRAME_MAGIC, len(header), len(payload)) + header + payload


def recv_exact(sock: socket.socket, size: int) -> bytes:
  data = bytearray()
  while len(data) < size:
    chunk = sock.recv(size - len(data))
    if not chunk:
      raise EOFError("video frame connection closed")
    data.extend(chunk)
  return bytes(data)


def receive_encoded_video_frame(sock: socket.socket) -> VideoFrame | H264Frame:
  magic, header_size, payload_size = _PREFIX.unpack(recv_exact(sock, _PREFIX.size))
  if magic not in (FRAME_MAGIC, H264_FRAME_MAGIC):
    raise FrameProtocolError("invalid video frame magic")
  if not 1 <= header_size <= MAX_HEADER_BYTES:
    raise FrameProtocolError("invalid video frame header size")
  maximum_payload = MAX_JPEG_BYTES if magic == FRAME_MAGIC else MAX_H264_BYTES
  if not 1 <= payload_size <= maximum_payload:
    raise FrameProtocolError("invalid video frame payload size")
  try:
    header = json.loads(recv_exact(sock, header_size).decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise FrameProtocolError("invalid video frame header JSON") from exc
  if not isinstance(header, dict):
    raise FrameProtocolError("video frame header must be an object")
  frame_id = _positive_int(header.get("frame_id"), "frame_id", (1 << 63) - 1)
  timestamp_ns = _positive_int(
    header.get("source_timestamp_monotonic_ns"), "source_timestamp_monotonic_ns", (1 << 63) - 1,
  )
  width = _positive_int(header.get("width"), "width", 8_192)
  height = _positive_int(header.get("height"), "height", 8_192)
  payload = recv_exact(sock, payload_size)

  if magic == FRAME_MAGIC:
    if header.get("protocol_version") != FRAME_PROTOCOL_VERSION or header.get("encoding") != "jpeg":
      raise FrameProtocolError("unsupported JPEG frame protocol")
    return VideoFrame(frame_id, timestamp_ns, width, height, payload)

  if header.get("protocol_version") != H264_FRAME_PROTOCOL_VERSION or header.get("encoding") != "h264":
    raise FrameProtocolError("unsupported H.264 frame protocol")
  keyframe = header.get("keyframe")
  if not isinstance(keyframe, bool):
    raise FrameProtocolError("keyframe must be boolean")
  codec_config_size = header.get("codec_config_size")
  if isinstance(codec_config_size, bool) or not isinstance(codec_config_size, int):
    raise FrameProtocolError("codec_config_size must be an integer")
  if not 0 <= codec_config_size < len(payload):
    raise FrameProtocolError("invalid H.264 codec config size")
  if codec_config_size and not keyframe:
    raise FrameProtocolError("H.264 codec config requires a keyframe")
  return H264Frame(
    frame_id=frame_id,
    source_timestamp_monotonic_ns=timestamp_ns,
    width=width,
    height=height,
    keyframe=keyframe,
    codec_config=payload[:codec_config_size],
    data=payload[codec_config_size:],
  )


def receive_video_frame(sock: socket.socket) -> VideoFrame:
  frame = receive_encoded_video_frame(sock)
  if not isinstance(frame, VideoFrame):
    raise FrameProtocolError("expected a JPEG video frame")
  return frame
