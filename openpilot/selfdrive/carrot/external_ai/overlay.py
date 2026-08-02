from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


DISPLAY_NAMES_KO = {
  "car": "승용차",
  "truck": "트럭",
  "bus": "버스",
  "motorcycle": "오토바이",
  "bicycle": "자전거",
  "person": "보행자",
  "traffic light": "신호등",
  "stop sign": "정지표지",
}

BACKEND_DISPLAY_NAMES = {
  "onnxruntime-nnapi": "NNAPI",
  "onnxruntime-cpu-fallback": "CPU",
  "onnxruntime-cpu": "CPU",
  "onnxruntime-qnn": "QNN",
  "qnn": "QNN",
}

NPU_BADGE_BACKENDS = frozenset((
  "onnxruntime-nnapi",
  "onnxruntime-qnn",
  "qnn",
))


@dataclass(frozen=True, slots=True)
class ExternalAIOverlayObject:
  class_name: str
  confidence: float
  x: float
  y: float
  width: float
  height: float


def _field(value: Any, name: str, default: Any = None) -> Any:
  if isinstance(value, dict):
    return value.get(name, default)
  try:
    return getattr(value, name)
  except Exception:
    return default


def external_ai_display_name(class_name: str) -> str:
  normalized = str(class_name or "").strip().lower()
  return DISPLAY_NAMES_KO.get(normalized, normalized.upper())


def phone_ai_npu_badge_active(state: Any) -> bool:
  """Report whether the phone selected an external accelerated inference backend."""
  if not bool(_field(state, "valid", False)) or not bool(_field(state, "connected", False)):
    return False
  backend = str(_field(state, "backend", "") or "").strip().lower()
  return backend in NPU_BADGE_BACKENDS or backend.startswith("qnn-")


def phone_ai_status_text(
    state: Any,
    *,
    service_alive: bool,
    service_valid: bool,
) -> tuple[str, bool]:
  if not service_alive:
    return "외부 AI · 시작 대기", False
  if not service_valid or state is None:
    return "외부 AI · 상태 확인 중", False
  if not bool(_field(state, "connected", False)) or not bool(_field(state, "valid", False)):
    return "외부 AI · 스마트폰 연결 대기", False

  backend_value = str(_field(state, "backend", "") or "").strip().lower()
  backend = BACKEND_DISPLAY_NAMES.get(backend_value, backend_value.upper() or "연산 중")
  try:
    latency_ms = float(_field(state, "latencyMs", 0.0))
  except (TypeError, ValueError):
    latency_ms = 0.0
  if not math.isfinite(latency_ms) or latency_ms < 0.0:
    latency_ms = 0.0
  objects = _field(state, "objects", ()) or ()
  try:
    object_count = len(objects)
  except TypeError:
    object_count = 0
  return f"외부 AI · {backend} · {latency_ms:.0f}ms · {object_count}개", True


def phone_ai_overlay_objects(
    state: Any,
    *,
    screen_x: float,
    screen_y: float,
    screen_width: float,
    screen_height: float,
) -> tuple[ExternalAIOverlayObject, ...]:
  if screen_width <= 0.0 or screen_height <= 0.0:
    return ()
  if not bool(_field(state, "valid", False)) or not bool(_field(state, "connected", False)):
    return ()

  output: list[ExternalAIOverlayObject] = []
  for item in _field(state, "objects", ()) or ():
    try:
      class_name = str(_field(item, "className", "") or "").strip().lower()
      confidence = float(_field(item, "confidence", 0.0))
      x1 = float(_field(item, "x1"))
      y1 = float(_field(item, "y1"))
      x2 = float(_field(item, "x2"))
      y2 = float(_field(item, "y2"))
    except (TypeError, ValueError):
      continue
    if not class_name or not all(math.isfinite(value) for value in (confidence, x1, y1, x2, y2)):
      continue
    if not (0.0 <= confidence <= 1.0 and 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
      continue
    output.append(ExternalAIOverlayObject(
      class_name=class_name,
      confidence=confidence,
      x=screen_x + x1 * screen_width,
      y=screen_y + y1 * screen_height,
      width=(x2 - x1) * screen_width,
      height=(y2 - y1) * screen_height,
    ))
  return tuple(output)
