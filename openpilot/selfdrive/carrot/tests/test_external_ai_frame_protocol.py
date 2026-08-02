from __future__ import annotations

import socket
import struct

import pytest

from openpilot.selfdrive.carrot.external_ai.frame_protocol import (
  FRAME_MAGIC,
  FrameProtocolError,
  VideoFrame,
  encode_video_frame,
  receive_video_frame,
)


def sample_frame() -> VideoFrame:
  return VideoFrame(42, 9_000_000_000, 640, 360, b"\xff\xd8mock-jpeg\xff\xd9")


def receive_packet(packet: bytes) -> VideoFrame:
  reader, writer = socket.socketpair()
  try:
    for offset in range(0, len(packet), 3):
      writer.sendall(packet[offset:offset + 3])
    writer.shutdown(socket.SHUT_WR)
    return receive_video_frame(reader)
  finally:
    reader.close()
    writer.close()


def test_video_frame_round_trip_survives_fragmented_tcp_reads() -> None:
  assert receive_packet(encode_video_frame(sample_frame())) == sample_frame()


@pytest.mark.parametrize("packet", [
  struct.pack("!4sII", b"NOPE", 1, 1) + b"{}x",
  struct.pack("!4sII", FRAME_MAGIC, 0, 1) + b"x",
  struct.pack("!4sII", FRAME_MAGIC, 2, 0) + b"{}",
])
def test_video_frame_rejects_invalid_prefix(packet: bytes) -> None:
  with pytest.raises(FrameProtocolError):
    receive_packet(packet)


def test_video_frame_rejects_truncated_connection() -> None:
  packet = encode_video_frame(sample_frame())
  with pytest.raises(EOFError):
    receive_packet(packet[:-1])
