import json

import pytest

from openpilot.selfdrive.carrot.external_ai.protocol import (
  ExternalAIProtocolError,
  ExternalAIResultTracker,
  parse_external_ai_result,
)


NOW_NS = 10_000_000_000


def valid_result(**changes) -> dict:
  result = {
    "protocol_version": 1,
    "frame_id": 42,
    "source_timestamp_monotonic_ns": NOW_NS - 120_000_000,
    "phone_receive_timestamp_ns": 5_000_000_000,
    "inference_start_timestamp_ns": 5_010_000_000,
    "inference_end_timestamp_ns": 5_045_000_000,
    "model": "yolo11n-int8",
    "backend": "qnn",
    "decode_ms": 3.0,
    "preprocess_ms": 5.0,
    "runtime_ms": 24.0,
    "postprocess_ms": 6.0,
    "phone_total_ms": 41.0,
    "input_width": 320,
    "input_height": 320,
    "traffic_light_state": "green",
    "traffic_light_confidence": 0.86,
    "objects": [{
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.91,
      "x1": 0.25,
      "y1": 0.30,
      "x2": 0.65,
      "y2": 0.82,
    }],
  }
  result.update(changes)
  return result


def encode(result: dict) -> bytes:
  return json.dumps(result, separators=(",", ":")).encode()


def test_valid_result_is_parsed_with_measured_timing() -> None:
  result = parse_external_ai_result(encode(valid_result()), now_monotonic_ns=NOW_NS)

  assert result.frame_id == 42
  assert result.latency_ms == pytest.approx(120.0)
  assert result.inference_ms == pytest.approx(35.0)
  assert result.decode_ms == pytest.approx(3.0)
  assert result.runtime_ms == pytest.approx(24.0)
  assert result.phone_total_ms == pytest.approx(41.0)
  assert (result.input_width, result.input_height) == (320, 320)
  assert result.traffic_light_state == "green"
  assert result.traffic_light_confidence == pytest.approx(0.86)
  assert result.objects[0].class_name == "car"
  assert result.objects[0].center_x == pytest.approx(0.45)


@pytest.mark.parametrize("protocol_version", (0, 2))
def test_protocol_version_mismatch_is_rejected(protocol_version: int) -> None:
  with pytest.raises(ExternalAIProtocolError, match="protocol_version"):
    parse_external_ai_result(encode(valid_result(protocol_version=protocol_version)), now_monotonic_ns=NOW_NS)


@pytest.mark.parametrize(
  "box",
  (
    {"x1": -0.1},
    {"x2": 1.1},
    {"x1": 0.8, "x2": 0.2},
    {"y1": 0.9, "y2": 0.9},
    {"confidence": float("nan")},
  ),
)
def test_invalid_object_coordinates_or_confidence_are_rejected(box: dict) -> None:
  result = valid_result()
  result["objects"][0].update(box)

  with pytest.raises(ExternalAIProtocolError):
    parse_external_ai_result(encode(result), now_monotonic_ns=NOW_NS)


def test_stale_result_is_rejected() -> None:
  result = valid_result(source_timestamp_monotonic_ns=NOW_NS - 301_000_000)

  with pytest.raises(ExternalAIProtocolError, match="maximum latency"):
    parse_external_ai_result(encode(result), now_monotonic_ns=NOW_NS, max_latency_ms=300.0)


def test_future_source_timestamp_is_rejected() -> None:
  result = valid_result(source_timestamp_monotonic_ns=NOW_NS + 51_000_000)

  with pytest.raises(ExternalAIProtocolError, match="future"):
    parse_external_ai_result(encode(result), now_monotonic_ns=NOW_NS)


@pytest.mark.parametrize("frame_id", (40, 41))
def test_replayed_or_reversed_frame_is_rejected(frame_id: int) -> None:
  with pytest.raises(ExternalAIProtocolError, match="did not advance"):
    parse_external_ai_result(encode(valid_result(frame_id=frame_id)), now_monotonic_ns=NOW_NS, previous_frame_id=41)


def test_excess_object_count_is_rejected() -> None:
  item = valid_result()["objects"][0]
  result = valid_result(objects=[dict(item) for _ in range(3)])

  with pytest.raises(ExternalAIProtocolError, match="object count"):
    parse_external_ai_result(encode(result), now_monotonic_ns=NOW_NS, max_objects=2)


def test_unsupported_class_is_rejected() -> None:
  result = valid_result()
  result["objects"][0]["class_name"] = "dog"

  with pytest.raises(ExternalAIProtocolError, match="unsupported"):
    parse_external_ai_result(encode(result), now_monotonic_ns=NOW_NS)


def test_malformed_json_and_utf8_are_rejected() -> None:
  with pytest.raises(ExternalAIProtocolError, match="JSON"):
    parse_external_ai_result(b"{", now_monotonic_ns=NOW_NS)
  with pytest.raises(ExternalAIProtocolError, match="UTF-8"):
    parse_external_ai_result(b"\xff", now_monotonic_ns=NOW_NS)


def test_phone_timestamp_order_is_checked_without_comparing_phone_and_c3x_clocks() -> None:
  result = valid_result(inference_start_timestamp_ns=4_000_000_000)

  with pytest.raises(ExternalAIProtocolError, match="out of order"):
    parse_external_ai_result(encode(result), now_monotonic_ns=NOW_NS)


def test_optional_performance_metrics_remain_compatible_and_are_bounded() -> None:
  legacy = valid_result()
  for field in ("decode_ms", "preprocess_ms", "runtime_ms", "postprocess_ms", "phone_total_ms", "input_width", "input_height"):
    legacy.pop(field)
  parsed = parse_external_ai_result(encode(legacy), now_monotonic_ns=NOW_NS)
  assert parsed.runtime_ms == 0.0
  assert parsed.input_width == 0

  legacy.pop("traffic_light_state", None)
  legacy.pop("traffic_light_confidence", None)
  parsed = parse_external_ai_result(encode(legacy), now_monotonic_ns=NOW_NS)
  assert parsed.traffic_light_state == "unknown"
  assert parsed.traffic_light_confidence == 0.0

  with pytest.raises(ExternalAIProtocolError, match="runtime_ms"):
    parse_external_ai_result(encode(valid_result(runtime_ms=60_001.0)), now_monotonic_ns=NOW_NS)
  with pytest.raises(ExternalAIProtocolError, match="provided together"):
    parse_external_ai_result(encode(valid_result(input_height=None)), now_monotonic_ns=NOW_NS)


@pytest.mark.parametrize("state", ("blue", "", 1, None))
def test_invalid_traffic_light_state_is_rejected(state) -> None:
  with pytest.raises(ExternalAIProtocolError, match="traffic_light_state"):
    parse_external_ai_result(encode(valid_result(traffic_light_state=state)), now_monotonic_ns=NOW_NS)


def test_tracker_reports_disconnect_and_hides_stale_result() -> None:
  tracker = ExternalAIResultTracker(connection_timeout_ms=2_000.0)

  assert not tracker.connected(now_monotonic_ns=NOW_NS)
  accepted = tracker.accept(encode(valid_result()), now_monotonic_ns=NOW_NS)
  assert tracker.connected(now_monotonic_ns=NOW_NS + 1_999_000_000)
  assert tracker.fresh_result(now_monotonic_ns=NOW_NS + 1_999_000_000) == accepted
  assert not tracker.connected(now_monotonic_ns=NOW_NS + 2_001_000_000)
  assert tracker.fresh_result(now_monotonic_ns=NOW_NS + 2_001_000_000) is None


def test_tracker_preserves_last_good_result_after_invalid_packet() -> None:
  tracker = ExternalAIResultTracker()
  accepted = tracker.accept(encode(valid_result()), now_monotonic_ns=NOW_NS)

  with pytest.raises(ExternalAIProtocolError):
    tracker.accept(b"bad", now_monotonic_ns=NOW_NS + 10_000_000)

  assert tracker.last_result == accepted
  assert tracker.last_receive_timestamp_ns == NOW_NS
