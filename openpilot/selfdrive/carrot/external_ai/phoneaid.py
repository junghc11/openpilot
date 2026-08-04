from __future__ import annotations

import time
from typing import Any

from openpilot.selfdrive.carrot.external_ai.protocol import (
  DEFAULT_CONNECTION_TIMEOUT_MS,
  DEFAULT_MAX_LATENCY_MS,
)
from openpilot.selfdrive.carrot.external_ai.frame_sender import (
  AdaptiveFrameQueue,
  DEFAULT_FRAME_FPS,
  DEFAULT_FRAME_PORT,
  DEFAULT_JPEG_QUALITY,
  H264FrameCapture,
  RoadFrameCapture,
  VideoFrameTcpServer,
)
from openpilot.selfdrive.carrot.external_ai.receiver import DEFAULT_RESULT_PORT, ExternalAIUdpReceiver
from openpilot.selfdrive.carrot.external_ai.state import build_phone_ai_payload


PUBLISH_INTERVAL_S = 0.1
POLL_TIMEOUT_S = 0.05
TRANSPORT_JPEG = 0
TRANSPORT_H264 = 1
DEFAULT_TRANSPORT = TRANSPORT_H264


def _clamped_param_int(params: Any, name: str, default: int, minimum: int, maximum: int) -> int:
  try:
    value = int(params.get_int(name))
  except Exception:
    value = default
  if value == 0 and minimum > 0:
    value = default
  return max(minimum, min(maximum, value))


def _param_text(params: Any, name: str) -> str:
  try:
    value = params.get(name)
  except Exception:
    return ""
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace").strip()
  return str(value or "").strip()


class PhoneAIDaemon:
  def __init__(self, params: Any | None = None, messaging_module: Any | None = None) -> None:
    if params is None:
      from openpilot.common.params import Params
      params = Params()
    if messaging_module is None:
      from openpilot.cereal import messaging as messaging_module
    self.params = params
    self.messaging = messaging_module
    result_port = _clamped_param_int(self.params, "ExternalAIResultPort", DEFAULT_RESULT_PORT, 1, 65_535)
    max_latency_ms = _clamped_param_int(
      self.params,
      "ExternalAIMaxLatencyMs",
      int(DEFAULT_MAX_LATENCY_MS),
      20,
      5_000,
    )
    self.receiver = ExternalAIUdpReceiver(
      port=result_port,
      allowed_phone_ip=_param_text(self.params, "ExternalAIPhoneIP"),
      max_latency_ms=float(max_latency_ms),
      connection_timeout_ms=DEFAULT_CONNECTION_TIMEOUT_MS,
    )
    allowed_phone_ip = _param_text(self.params, "ExternalAIPhoneIP")
    frame_port = _clamped_param_int(self.params, "ExternalAIFramePort", DEFAULT_FRAME_PORT, 1, 65_535)
    frame_fps = _clamped_param_int(self.params, "ExternalAIFrameFPS", DEFAULT_FRAME_FPS, 1, 15)
    jpeg_quality = _clamped_param_int(self.params, "ExternalAIJpegQuality", DEFAULT_JPEG_QUALITY, 30, 95)
    self.transport = _clamped_param_int(self.params, "ExternalAITransport", DEFAULT_TRANSPORT, TRANSPORT_JPEG, TRANSPORT_H264)
    youtube_live = _clamped_param_int(self.params, "CarrotYouTubeLive", 0, 0, 1)
    youtube_quality = _clamped_param_int(self.params, "CarrotYouTubeQuality", 0, 0, 3)
    h264_source_compatible = youtube_live == 0 or youtube_quality == 0
    use_h264 = self.transport == TRANSPORT_H264 and h264_source_compatible
    queue = AdaptiveFrameQueue() if use_h264 else None
    self.frame_server = VideoFrameTcpServer(port=frame_port, allowed_phone_ip=allowed_phone_ip, slot=queue)
    self.h264_capture = H264FrameCapture(self.frame_server, messaging_module) if use_h264 else None
    self.frame_capture = RoadFrameCapture(
      self.frame_server,
      fps=frame_fps,
      jpeg_quality=jpeg_quality,
      should_encode=(lambda: not self.h264_capture.is_recent) if self.h264_capture is not None else None,
    )
    self.pm = messaging_module.PubMaster(["phoneAIState"])

  def publish_once(self, *, now_monotonic_ns: int | None = None) -> dict[str, object]:
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    result = self.receiver.tracker.fresh_result(now_monotonic_ns=now_ns)
    stats = self.receiver.stats(now_monotonic_ns=now_ns)
    payload = build_phone_ai_payload(
      result,
      stats,
      frame_connected=self.frame_server.stats().connected,
    )
    message = self.messaging.new_message("phoneAIState", valid=True)
    message.phoneAIState = payload
    self.pm.send("phoneAIState", message)
    return payload

  def run(self) -> None:
    self.receiver.open()
    try:
      self.frame_server.start()
      if self.h264_capture is not None:
        self.h264_capture.start()
      self.frame_capture.start()
      next_publish = 0.0
      while True:
        self.receiver.poll(timeout_s=POLL_TIMEOUT_S)
        now = time.monotonic()
        if now >= next_publish:
          self.publish_once()
          next_publish = now + PUBLISH_INTERVAL_S
    finally:
      self.frame_capture.stop()
      if self.h264_capture is not None:
        self.h264_capture.stop()
      self.frame_server.stop()
      self.receiver.close()


def main() -> None:
  PhoneAIDaemon().run()


if __name__ == "__main__":
  main()
