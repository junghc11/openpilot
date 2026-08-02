from __future__ import annotations

import time
from typing import Any

from openpilot.selfdrive.carrot.external_ai.protocol import (
  DEFAULT_CONNECTION_TIMEOUT_MS,
  DEFAULT_MAX_LATENCY_MS,
)
from openpilot.selfdrive.carrot.external_ai.receiver import DEFAULT_RESULT_PORT, ExternalAIUdpReceiver
from openpilot.selfdrive.carrot.external_ai.state import build_phone_ai_payload


PUBLISH_INTERVAL_S = 0.1
POLL_TIMEOUT_S = 0.05


def _clamped_param_int(params: Any, name: str, default: int, minimum: int, maximum: int) -> int:
  try:
    value = int(params.get_int(name))
  except Exception:
    value = default
  return max(minimum, min(maximum, value or default))


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
    self.pm = messaging_module.PubMaster(["phoneAIState"])

  def publish_once(self, *, now_monotonic_ns: int | None = None) -> dict[str, object]:
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    result = self.receiver.tracker.fresh_result(now_monotonic_ns=now_ns)
    stats = self.receiver.stats(now_monotonic_ns=now_ns)
    payload = build_phone_ai_payload(result, stats)
    message = self.messaging.new_message("phoneAIState", valid=True)
    message.phoneAIState = payload
    self.pm.send("phoneAIState", message)
    return payload

  def run(self) -> None:
    self.receiver.open()
    next_publish = 0.0
    try:
      while True:
        self.receiver.poll(timeout_s=POLL_TIMEOUT_S)
        now = time.monotonic()
        if now >= next_publish:
          self.publish_once()
          next_publish = now + PUBLISH_INTERVAL_S
    finally:
      self.receiver.close()


def main() -> None:
  PhoneAIDaemon().run()


if __name__ == "__main__":
  main()
