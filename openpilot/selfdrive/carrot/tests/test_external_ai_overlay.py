import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.carrot.external_ai.overlay import (
  phone_ai_compute_badge,
  phone_ai_overlay_objects,
  phone_ai_status_text,
)
from openpilot.selfdrive.ui.onroad.external_ai_labels import (
  DISPLAY_NAME_KEYS,
  SUPPORTED_DISPLAY_LANGUAGES,
  external_ai_display_name,
  external_ai_display_name_for_language,
  normalize_external_ai_language,
)
from openpilot.selfdrive.ui.translations.potools import parse_po


OPENPILOT_ROOT = Path(__file__).resolve().parents[3]
TRANSLATIONS_DIR = OPENPILOT_ROOT / "selfdrive" / "ui" / "translations"
SUPPORTED_LANGUAGES = tuple(json.loads((TRANSLATIONS_DIR / "languages.json").read_text(encoding="utf-8")).values())


def _translations(language: str) -> dict[str, str]:
  _, entries = parse_po(TRANSLATIONS_DIR / f"app_{language}.po")
  return {entry.msgid: entry.msgstr for entry in entries}


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


def test_c3x_overlay_defaults_to_english_object_names() -> None:
  assert external_ai_display_name("car", translate=lambda text: text) == "Car"
  assert external_ai_display_name("person", translate=lambda text: text) == "Pedestrian"
  assert external_ai_display_name("traffic light", translate=lambda text: text) == "Traffic Light"
  assert external_ai_display_name("custom") == "CUSTOM"


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_c3x_object_names_are_translated_for_every_supported_language(language: str) -> None:
  translations = _translations(language)
  translate = lambda text: translations.get(text, text) or text
  localized_names = {external_ai_display_name(class_name, translate=translate) for class_name in DISPLAY_NAME_KEYS}
  assert len(localized_names) == len(DISPLAY_NAME_KEYS)
  assert all(translations.get(key) for key in DISPLAY_NAME_KEYS.values())


def test_external_ai_translation_catalog_matches_configured_languages() -> None:
  _, template_entries = parse_po(TRANSLATIONS_DIR / "app.pot")
  template_keys = {entry.msgid for entry in template_entries}
  assert SUPPORTED_DISPLAY_LANGUAGES == frozenset(SUPPORTED_LANGUAGES)
  assert set(DISPLAY_NAME_KEYS.values()) <= template_keys


def test_c3x_object_name_examples_follow_selected_language() -> None:
  korean = _translations("ko")
  japanese = _translations("ja")
  assert external_ai_display_name("car", translate=korean.get) == "승용차"
  assert external_ai_display_name("person", translate=korean.get) == "보행자"
  assert external_ai_display_name("traffic light", translate=japanese.get) == "信号機"


def test_shared_object_labels_accept_language_setting_values() -> None:
  assert normalize_external_ai_language(b"main_ko") == "ko"
  assert normalize_external_ai_language("main_zh-CHT") == "zh-CHT"
  assert normalize_external_ai_language("unsupported") == "en"
  assert external_ai_display_name_for_language("truck", "main_ko") == "트럭"
  assert external_ai_display_name_for_language("person", "main_ja") == "歩行者"


def test_c3x_overlay_status_reports_connection_backend_and_latency() -> None:
  waiting, waiting_connected = phone_ai_status_text(None, service_alive=False, service_valid=False)
  assert waiting == "외부 AI · 시작 대기"
  assert not waiting_connected

  state = SimpleNamespace(
    valid=True,
    connected=True,
    backend="onnxruntime-nnapi",
    latencyMs=84.4,
    objects=(object(), object(), object()),
  )
  connected, is_connected = phone_ai_status_text(state, service_alive=True, service_valid=True)
  assert connected == "외부 AI · NNAPI · 84ms · 3개"
  assert is_connected


@pytest.mark.parametrize("backend", ("onnxruntime-nnapi", "onnxruntime-qnn", "qnn", "qnn-htp"))
def test_c3x_enpu_badge_is_active_for_external_accelerators(backend: str) -> None:
  state = SimpleNamespace(valid=True, connected=True, backend=backend)
  assert phone_ai_compute_badge(state) == "eNPU"


@pytest.mark.parametrize("backend", ("onnxruntime-cpu", "onnxruntime-cpu-fallback", "cpu", "cpu-fallback"))
def test_c3x_ecpu_badge_is_active_for_cpu_backends(backend: str) -> None:
  state = SimpleNamespace(valid=True, connected=True, backend=backend)
  assert phone_ai_compute_badge(state) == "eCPU"


@pytest.mark.parametrize("state", (
  SimpleNamespace(valid=True, connected=True, backend="python-mock"),
  SimpleNamespace(valid=True, connected=False, backend="onnxruntime-nnapi"),
  SimpleNamespace(valid=False, connected=True, backend="onnxruntime-nnapi"),
  None,
))
def test_c3x_compute_badge_is_hidden_without_live_known_backend(state) -> None:
  assert phone_ai_compute_badge(state) == ""


def test_regular_and_mici_ui_share_external_ai_shape_renderer() -> None:
  ui_state_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "ui_state.py").read_text(encoding="utf-8")
  regular_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "onroad" / "augmented_road_view.py").read_text(encoding="utf-8")
  mici_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "mici" / "onroad" / "augmented_road_view.py").read_text(encoding="utf-8")
  renderer_source = (OPENPILOT_ROOT / "selfdrive" / "ui" / "onroad" / "external_ai_overlay.py").read_text(encoding="utf-8")
  overlay_state_source = (OPENPILOT_ROOT / "selfdrive" / "carrot" / "external_ai" / "overlay.py").read_text(encoding="utf-8")

  assert '"phoneAIState"' in ui_state_source
  assert "ExternalAIOverlayRenderer" in regular_source
  assert "ExternalAIOverlayRenderer" in mici_source
  for method in ("_draw_car", "_draw_truck", "_draw_bus", "_draw_motorcycle", "_draw_bicycle", "_draw_person"):
    assert f"def {method}" in renderer_source
  assert "phone_ai_status_text" in renderer_source
  assert "phone_ai_compute_badge" in renderer_source
  assert "def _draw_compute_badge" in renderer_source
  assert '"eNPU"' in renderer_source
  assert 'return "eCPU"' in overlay_state_source
  assert "external_ai_display_name" in renderer_source
  assert "translate=tr" in renderer_source
  assert "font_fallback" in renderer_source
