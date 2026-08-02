from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.carrot.external_ai.overlay import phone_ai_overlay_objects


OPENPILOT_ROOT = Path(__file__).resolve().parents[3]


def test_normalized_phone_bbox_maps_to_content_rectangle() -> None:
  state = SimpleNamespace(
    valid=True,
    connected=True,
    objects=(SimpleNamespace(
      className="car",
      confidence=0.9,
      x1=0.25,
      y1=0.20,
      x2=0.75,
      y2=0.80,
    ),),
  )

  objects = phone_ai_overlay_objects(
    state,
    screen_x=100.0,
    screen_y=50.0,
    screen_width=1000.0,
    screen_height=500.0,
  )

  assert len(objects) == 1
  assert objects[0].class_name == "car"
  assert objects[0].x == pytest.approx(350.0)
  assert objects[0].y == pytest.approx(150.0)
  assert objects[0].width == pytest.approx(500.0)
  assert objects[0].height == pytest.approx(300.0)


def test_invalid_phone_state_or_bbox_is_not_drawn() -> None:
  invalid_state = SimpleNamespace(valid=False, connected=True, objects=())
  bad_box_state = SimpleNamespace(
    valid=True,
    connected=True,
    objects=(SimpleNamespace(className="car", confidence=0.9, x1=0.8, y1=0.2, x2=0.2, y2=0.8),),
  )

  assert phone_ai_overlay_objects(invalid_state, screen_x=0, screen_y=0, screen_width=100, screen_height=100) == ()
  assert phone_ai_overlay_objects(bad_box_state, screen_x=0, screen_y=0, screen_width=100, screen_height=100) == ()


def test_regular_and_mici_ui_share_external_ai_shape_renderer() -> None:
  ui_state_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "ui_state.py").read_text(encoding="utf-8")
  regular_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "onroad" / "augmented_road_view.py").read_text(encoding="utf-8")
  mici_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "mici" / "onroad" / "augmented_road_view.py").read_text(encoding="utf-8")
  renderer_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "onroad" / "external_ai_overlay.py").read_text(encoding="utf-8")

  assert '"phoneAIState"' in ui_state_source
  assert "ExternalAIOverlayRenderer" in regular_source
  assert "ExternalAIOverlayRenderer" in mici_source
  for method in ("_draw_car", "_draw_truck", "_draw_bus", "_draw_motorcycle", "_draw_bicycle", "_draw_person"):
    assert f"def {method}" in renderer_source
