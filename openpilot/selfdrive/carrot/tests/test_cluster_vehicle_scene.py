from dataclasses import replace
from pathlib import Path
import sys

import pytest


CLUSTER_DIR = Path(__file__).resolve().parents[1] / "cluster"
sys.path.insert(0, str(CLUSTER_DIR))

from cluster_models import ClusterUiState, DetectedVehicle, LaneMarking
from cluster_scene import build_cluster_scene


def cluster_state(**changes) -> ClusterUiState:
  state = ClusterUiState(
    speed_kph=60.0,
    accel_mps2=0.0,
    steering=0.0,
    speed_limit_kph=None,
    speed_limit_source=None,
    cruise_kph=None,
    cruise_display_state="off",
    gear_text=None,
    cruise_gap=None,
    lfa_active=None,
    left_signal=False,
    right_signal=False,
    left_blindspot=False,
    right_blindspot=False,
    lane_change=None,
    lane_change_phase="off",
    lane_change_progress=0.0,
    highlight_lane=None,
    highlight_lane_offset=None,
    ego_lane_offset=0.0,
    road_view_lane_position=0.0,
    camera_lane_center_offset_m=None,
    lane_width_m=3.6,
    steering_angle_deg=0.0,
    surround_yaw_deg=0.0,
    surround_pitch_deg=0.0,
    surround_view_active=False,
    lanes=(LaneMarking(-1.8), LaneMarking(1.8)),
  )
  return replace(state, **changes)


def test_external_ai_classes_select_real_shapes_and_dimensions() -> None:
  detections = (
    DetectedVehicle("car", 22.0, -1.2, source="externalAI", object_class="car", object_track_id=10),
    DetectedVehicle("truck", 35.0, 1.3, source="externalAI", object_class="truck", object_track_id=11),
    DetectedVehicle("person", 15.0, 2.8, source="externalAI", object_class="person", object_track_id=12),
  )

  scene = build_cluster_scene(cluster_state(detected_vehicles=detections))
  boxes = {box.label: box for box in scene.vehicles if box.label in {"car", "truck", "person"}}

  assert boxes["car"].model_key == "car"
  assert boxes["car"].width_m == pytest.approx(1.85)
  assert boxes["truck"].model_key == "truck"
  assert boxes["truck"].length_m == pytest.approx(7.50)
  assert boxes["person"].model_key == "person"
  assert boxes["person"].height_m == pytest.approx(1.75)


def test_unknown_external_ai_class_preserves_marker_fallback() -> None:
  detection = DetectedVehicle("cone", 18.0, 0.4, source="externalAI", object_class="traffic cone")

  scene = build_cluster_scene(cluster_state(detected_vehicles=(detection,)))
  box = next(box for box in scene.vehicles if box.label == "cone")

  assert box.model_key == ""
