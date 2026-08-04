from __future__ import annotations

from openpilot.selfdrive.carrot.external_ai.protocol import ExternalAIResult
from openpilot.selfdrive.carrot.external_ai.receiver import ExternalAIReceiverStats


def build_phone_ai_payload(
    result: ExternalAIResult | None,
    stats: ExternalAIReceiverStats,
    *,
    frame_connected: bool = False,
) -> dict[str, object]:
  valid = bool(result is not None and stats.connected)
  connected = bool(frame_connected or stats.connected)
  return {
    "valid": valid,
    "connected": connected,
    "protocolVersion": result.protocol_version if valid else 0,
    "frameId": result.frame_id if valid else 0,
    "sourceTimestampMonotonicNanos": result.source_timestamp_monotonic_ns if valid else 0,
    "receiveTimestampMonotonicNanos": result.c3x_receive_timestamp_ns if valid else 0,
    "latencyMs": result.latency_ms if valid else 0.0,
    "inferenceMs": result.inference_ms if valid else 0.0,
    "decodeMs": result.decode_ms if valid else 0.0,
    "preprocessMs": result.preprocess_ms if valid else 0.0,
    "runtimeMs": result.runtime_ms if valid else 0.0,
    "postprocessMs": result.postprocess_ms if valid else 0.0,
    "phoneTotalMs": result.phone_total_ms if valid else 0.0,
    "inputWidth": result.input_width if valid else 0,
    "inputHeight": result.input_height if valid else 0,
    "modelName": result.model if valid else "",
    "backend": result.backend if valid else "",
    "trafficLightState": result.traffic_light_state if valid else "unknown",
    "trafficLightConfidence": result.traffic_light_confidence if valid else 0.0,
    "objects": [
      {
        "classId": item.class_id,
        "className": item.class_name,
        "confidence": item.confidence,
        "x1": item.x1,
        "y1": item.y1,
        "x2": item.x2,
        "y2": item.y2,
      }
      for item in (result.objects if valid else ())
    ],
    "acceptedPackets": stats.accepted_packets,
    "rejectedPackets": stats.rejected_packets,
    "rejectedSenders": stats.rejected_senders,
    "lastSenderIp": stats.last_sender_ip or "",
    "lastError": stats.last_error or "",
  }
