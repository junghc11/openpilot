import importlib.util
from pathlib import Path
import socket
import time

from openpilot.selfdrive.carrot.external_ai.overlay import phone_ai_overlay_objects
from openpilot.selfdrive.carrot.external_ai.projection import project_phone_ai_state
from openpilot.selfdrive.carrot.external_ai.receiver import ExternalAIReceiverStats, ExternalAIUdpReceiver
from openpilot.selfdrive.carrot.external_ai.state import build_phone_ai_payload


OPENPILOT_ROOT = Path(__file__).resolve().parents[3]
MOCK_PATH = OPENPILOT_ROOT.parent / "tools" / "external_ai" / "mock_phone_ai.py"


def load_mock_module():
  spec = importlib.util.spec_from_file_location("external_ai_mock_phone", MOCK_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_mock_packet_reaches_cereal_projection_and_device_overlay() -> None:
  mock = load_mock_module()
  now_ns = time.monotonic_ns()
  with ExternalAIUdpReceiver(host="127.0.0.1", port=0) as receiver:
    assert receiver.bound_port is not None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
      sender.sendto(mock.build_mock_result(1, now_ns - 40_000_000), ("127.0.0.1", receiver.bound_port))
    result = receiver.poll(timeout_s=0.5, now_monotonic_ns=now_ns)

  assert result is not None
  stats = ExternalAIReceiverStats(1, 0, 0, "127.0.0.1", None, True)
  phone_ai_state = build_phone_ai_payload(result, stats)
  cluster_objects = project_phone_ai_state(phone_ai_state)
  device_objects = phone_ai_overlay_objects(
    phone_ai_state,
    screen_x=0.0,
    screen_y=0.0,
    screen_width=1920.0,
    screen_height=1080.0,
  )

  assert {item.class_name for item in cluster_objects} == {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
  assert {item.class_name for item in device_objects} == {
    "car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "stop sign",
  }
