#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time


DEFAULT_RESULT_PORT = 7725
DEFAULT_FPS = 5.0
MOCK_OBJECTS = (
  (2, "car", 0.95, 0.38, 0.42, 0.59, 0.74),
  (7, "truck", 0.90, 0.63, 0.34, 0.82, 0.70),
  (5, "bus", 0.88, 0.08, 0.28, 0.28, 0.72),
  (3, "motorcycle", 0.86, 0.52, 0.50, 0.61, 0.76),
  (1, "bicycle", 0.84, 0.27, 0.47, 0.36, 0.76),
  (0, "person", 0.92, 0.84, 0.38, 0.91, 0.77),
  (9, "traffic light", 0.89, 0.70, 0.08, 0.74, 0.23),
  (11, "stop sign", 0.87, 0.19, 0.17, 0.25, 0.31),
)


def build_mock_result(frame_id: int, source_timestamp_monotonic_ns: int) -> bytes:
  phone_receive_ns = time.monotonic_ns()
  inference_start_ns = phone_receive_ns + 2_000_000
  inference_end_ns = inference_start_ns + 18_000_000
  payload = {
    "protocol_version": 1,
    "frame_id": frame_id,
    "source_timestamp_monotonic_ns": source_timestamp_monotonic_ns,
    "phone_receive_timestamp_ns": phone_receive_ns,
    "inference_start_timestamp_ns": inference_start_ns,
    "inference_end_timestamp_ns": inference_end_ns,
    "model": "mock-yolo11n",
    "backend": "python-mock",
    "objects": [
      {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
      }
      for class_id, class_name, confidence, x1, y1, x2, y2 in MOCK_OBJECTS
    ],
  }
  return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def run(host: str, port: int, fps: float, count: int) -> None:
  interval_s = 1.0 / fps
  frame_id = 1
  next_send = time.monotonic()
  started = next_send
  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
    while count <= 0 or frame_id <= count:
      now = time.monotonic()
      if now < next_send:
        time.sleep(next_send - now)
      # This standalone mock runs on the C3X/PC so both ends share the same
      # monotonic clock. The Android client will echo the timestamp received
      # in the C3X video-frame header instead.
      source_ns = time.monotonic_ns() - 40_000_000
      payload = build_mock_result(frame_id, source_ns)
      sender.sendto(payload, (host, port))
      if frame_id == 1 or frame_id % max(1, round(fps)) == 0:
        if frame_id == 1:
          print(f"mock phone AI sent frame=1 objects={len(MOCK_OBJECTS)} target={fps:.1f}Hz")
        else:
          elapsed = max(0.001, time.monotonic() - started)
          print(f"mock phone AI sent frame={frame_id} objects={len(MOCK_OBJECTS)} rate={frame_id / elapsed:.1f}Hz")
      frame_id += 1
      next_send += interval_s


def main() -> None:
  parser = argparse.ArgumentParser(description="Send visualization-only mock phone AI detections to phoneaid")
  parser.add_argument("--host", default="127.0.0.1", help="C3X address. Default: 127.0.0.1")
  parser.add_argument("--port", type=int, default=DEFAULT_RESULT_PORT, help="phoneaid UDP result port")
  parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="result rate. Default: 5")
  parser.add_argument("--count", type=int, default=0, help="packets to send; 0 runs until Ctrl-C")
  args = parser.parse_args()
  if not 1 <= args.port <= 65_535:
    parser.error("--port must be between 1 and 65535")
  if not 0.1 <= args.fps <= 30.0:
    parser.error("--fps must be between 0.1 and 30")
  if args.count < 0:
    parser.error("--count must be non-negative")
  try:
    run(args.host, args.port, args.fps, args.count)
  except KeyboardInterrupt:
    print("mock phone AI stopped")


if __name__ == "__main__":
  main()
