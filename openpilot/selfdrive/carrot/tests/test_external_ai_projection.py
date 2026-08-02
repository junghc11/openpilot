from types import SimpleNamespace

import pytest

from openpilot.selfdrive.carrot.external_ai.projection import project_normalized_bbox, project_phone_ai_state


def test_larger_bbox_projects_object_closer() -> None:
  near = project_normalized_bbox("car", 0.3, 0.2, 0.7, 0.9)
  far = project_normalized_bbox("car", 0.4, 0.4, 0.6, 0.65)

  assert near is not None and far is not None
  assert near[0] < far[0]


def test_horizontal_bbox_position_projects_to_lateral_position() -> None:
  left = project_normalized_bbox("truck", 0.05, 0.3, 0.25, 0.8)
  center = project_normalized_bbox("truck", 0.4, 0.3, 0.6, 0.8)
  right = project_normalized_bbox("truck", 0.75, 0.3, 0.95, 0.8)

  assert left is not None and center is not None and right is not None
  assert left[1] < 0.0
  assert center[1] == pytest.approx(0.0)
  assert right[1] > 0.0


def test_unsupported_sign_class_remains_available_in_cereal_but_not_3d_scene() -> None:
  assert project_normalized_bbox("traffic light", 0.4, 0.1, 0.6, 0.4) is None
  assert project_normalized_bbox("stop sign", 0.4, 0.1, 0.6, 0.4) is None


def test_phone_ai_state_projects_supported_objects_and_stable_ids() -> None:
  state = SimpleNamespace(
    valid=True,
    connected=True,
    frameId=123,
    objects=(
      SimpleNamespace(className="car", confidence=0.92, x1=0.3, y1=0.3, x2=0.7, y2=0.8),
      SimpleNamespace(className="person", confidence=0.81, x1=0.7, y1=0.3, x2=0.8, y2=0.9),
      SimpleNamespace(className="traffic light", confidence=0.75, x1=0.1, y1=0.1, x2=0.2, y2=0.3),
    ),
  )

  projected = project_phone_ai_state(state)

  assert [item.class_name for item in projected] == ["car", "person"]
  assert [item.object_track_id for item in projected] == [(123 << 8), (123 << 8) | 1]
  assert projected[1].lateral_m > 0.0


@pytest.mark.parametrize(
  "state",
  (
    SimpleNamespace(valid=False, connected=True, frameId=1, objects=()),
    SimpleNamespace(valid=True, connected=False, frameId=1, objects=()),
  ),
)
def test_invalid_or_disconnected_state_has_no_projected_objects(state) -> None:
  assert project_phone_ai_state(state) == ()
