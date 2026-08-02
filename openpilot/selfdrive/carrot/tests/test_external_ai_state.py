import json
from pathlib import Path
from types import SimpleNamespace

from openpilot.selfdrive.carrot.external_ai.protocol import parse_external_ai_result
from openpilot.selfdrive.carrot.external_ai.receiver import ExternalAIReceiverStats
from openpilot.selfdrive.carrot.external_ai.state import build_phone_ai_payload
from openpilot.selfdrive.carrot.external_ai.phoneaid import PhoneAIDaemon
from openpilot.selfdrive.carrot.external_ai.frame_sender import AdaptiveFrameQueue, LatestFrameSlot


OPENPILOT_ROOT = Path(__file__).resolve().parents[3]


def parsed_result():
  now_ns = 9_000_000_000
  return parse_external_ai_result(json.dumps({
    "protocol_version": 1,
    "frame_id": 9,
    "source_timestamp_monotonic_ns": now_ns - 80_000_000,
    "phone_receive_timestamp_ns": 2_000_000_000,
    "inference_start_timestamp_ns": 2_010_000_000,
    "inference_end_timestamp_ns": 2_040_000_000,
    "model": "yolo11n",
    "backend": "mock",
    "decode_ms": 2.0,
    "preprocess_ms": 4.0,
    "runtime_ms": 18.0,
    "postprocess_ms": 3.0,
    "phone_total_ms": 30.0,
    "input_width": 320,
    "input_height": 320,
    "objects": [{
      "class_id": 5,
      "class_name": "bus",
      "confidence": 0.88,
      "x1": 0.1,
      "y1": 0.2,
      "x2": 0.4,
      "y2": 0.9,
    }],
  }), now_monotonic_ns=now_ns)


def test_phone_ai_payload_contains_fresh_detection_and_health() -> None:
  result = parsed_result()
  stats = ExternalAIReceiverStats(3, 1, 2, "192.0.2.10", None, True)

  payload = build_phone_ai_payload(result, stats)

  assert payload["valid"] is True
  assert payload["connected"] is True
  assert payload["frameId"] == 9
  assert payload["latencyMs"] == 80.0
  assert payload["runtimeMs"] == 18.0
  assert payload["phoneTotalMs"] == 30.0
  assert payload["inputWidth"] == 320
  assert payload["objects"] == [{
    "classId": 5,
    "className": "bus",
    "confidence": 0.88,
    "x1": 0.1,
    "y1": 0.2,
    "x2": 0.4,
    "y2": 0.9,
  }]
  assert payload["acceptedPackets"] == 3
  assert payload["rejectedSenders"] == 2


def test_disconnected_payload_hides_old_objects_but_keeps_diagnostics() -> None:
  stats = ExternalAIReceiverStats(3, 4, 0, "192.0.2.10", "timeout", False)

  payload = build_phone_ai_payload(parsed_result(), stats)

  assert payload["valid"] is False
  assert payload["connected"] is False
  assert payload["frameId"] == 0
  assert payload["objects"] == []
  assert payload["rejectedPackets"] == 4
  assert payload["lastError"] == "timeout"


def test_phone_ai_cereal_service_uses_reserved_fork_slot() -> None:
  log_schema = (OPENPILOT_ROOT / "cereal" / "log.capnp").read_text(encoding="utf-8")
  custom_schema = (OPENPILOT_ROOT / "cereal" / "custom.capnp").read_text(encoding="utf-8")
  services = (OPENPILOT_ROOT / "cereal" / "services.py").read_text(encoding="utf-8")

  assert "phoneAIState @110 :Custom.PhoneAIState;" in log_schema
  assert "struct PhoneAIState @0xda96579883444c35" in custom_schema
  assert '"phoneAIState": (True, 10., 10)' in services


def test_external_ai_is_disabled_by_default_and_manager_gated() -> None:
  params_keys = (OPENPILOT_ROOT / "common" / "params_keys.h").read_text(encoding="utf-8")
  process_config = (OPENPILOT_ROOT / "system" / "manager" / "process_config.py").read_text(encoding="utf-8")
  loggerd = (OPENPILOT_ROOT / "system" / "loggerd" / "loggerd.h").read_text(encoding="utf-8")
  encoderd = (OPENPILOT_ROOT / "system" / "loggerd" / "encoderd.cc").read_text(encoding="utf-8")

  assert '{"ExternalAIEnabled", {PERSISTENT, BOOL, "0"}}' in params_keys
  assert '{"ExternalAITransport", {PERSISTENT, INT, "1"}}' in params_keys
  assert 'return started and params.get_bool("ExternalAIEnabled")' in process_config
  assert 'NativeProcess("external_ai_encoderd"' in process_config
  assert '["./encoderd", "--external-ai"]' in process_config
  external_profile = loggerd.split("static EncoderSettings ExternalAIEncoderSettings", 1)[1].split("static EncoderSettings", 1)[0]
  assert ".bitrate = 750'000" in external_profile
  assert ".gop_size = 15" in external_profile
  assert ".frame_width = 854" in external_profile
  assert ".frame_height = 480" in external_profile
  assert 'mode == "--external-ai"' in encoderd
  assert 'PythonProcess("phoneaid", "openpilot.selfdrive.carrot.external_ai.phoneaid"' in process_config


def test_phoneaid_publishes_cereal_payload_with_injected_runtime() -> None:
  class FakeParams:
    def get_int(self, name):
      return {"ExternalAIResultPort": 17725, "ExternalAIMaxLatencyMs": 250}[name]

    def get(self, name):
      assert name == "ExternalAIPhoneIP"
      return b"192.0.2.10"

  class FakePubMaster:
    def __init__(self, services):
      assert services == ["phoneAIState"]
      self.sent = []

    def send(self, service, message):
      self.sent.append((service, message))

  class FakeMessaging:
    PubMaster = FakePubMaster

    @staticmethod
    def new_message(service, valid):
      assert service == "phoneAIState"
      assert valid is True
      return SimpleNamespace(phoneAIState=None)

  daemon = PhoneAIDaemon(params=FakeParams(), messaging_module=FakeMessaging)
  payload = daemon.publish_once(now_monotonic_ns=9_000_000_000)

  assert daemon.receiver.port == 17725
  assert daemon.receiver.allowed_phone_ip == "192.0.2.10"
  assert payload["valid"] is False
  assert daemon.pm.sent[0][0] == "phoneAIState"
  assert daemon.pm.sent[0][1].phoneAIState == payload


def test_phoneaid_selects_h264_queue_and_avoids_incompatible_youtube_profiles() -> None:
  class FakeParams:
    def __init__(self, transport, youtube_live, youtube_quality):
      self.values = {
        "ExternalAITransport": transport,
        "CarrotYouTubeLive": youtube_live,
        "CarrotYouTubeQuality": youtube_quality,
      }

    def get_int(self, name):
      return self.values.get(name, 0)

    def get(self, name):
      return b""

  class FakeMessaging:
    class PubMaster:
      def __init__(self, services):
        assert services == ["phoneAIState"]

  h264 = PhoneAIDaemon(params=FakeParams(1, 0, 0), messaging_module=FakeMessaging)
  assert isinstance(h264.frame_server.slot, AdaptiveFrameQueue)
  assert h264.h264_capture is not None

  youtube_high = PhoneAIDaemon(params=FakeParams(1, 1, 2), messaging_module=FakeMessaging)
  assert isinstance(youtube_high.frame_server.slot, LatestFrameSlot)
  assert youtube_high.h264_capture is None

  jpeg = PhoneAIDaemon(params=FakeParams(0, 0, 0), messaging_module=FakeMessaging)
  assert isinstance(jpeg.frame_server.slot, LatestFrameSlot)
  assert jpeg.h264_capture is None
