from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


BACKEND_DISPLAY_NAMES = {
  "onnxruntime-nnapi": "NNAPI",
  "onnxruntime-cpu-fallback": "CPU",
  "onnxruntime-cpu": "CPU",
  "onnxruntime-qnn": "QNN",
  "onnxruntime-qnn-mixed": "QNN+CPU",
  "onnxruntime-qnn-mixed-unverified": "QNN?",
  "qnn": "QNN",
}

NPU_BADGE_BACKENDS = frozenset((
  "onnxruntime-qnn",
  "qnn",
  "qnn-htp",
))
MIXED_NPU_BADGE_BACKENDS = frozenset(("onnxruntime-qnn-mixed",))
UNVERIFIED_QNN_BADGE_BACKENDS = frozenset(("onnxruntime-qnn-mixed-unverified",))
GENERIC_ACCEL_BADGE_BACKENDS = frozenset(("onnxruntime-nnapi",))
CPU_BADGE_BACKENDS = frozenset((
  "onnxruntime-cpu",
  "onnxruntime-cpu-fallback",
  "cpu",
  "cpu-fallback",
))
TRAFFIC_LIGHT_STATES = frozenset(("red", "yellow", "green"))


@dataclass(frozen=True, slots=True)
class ExternalAIOverlayObject:
  class_name: str
  confidence: float
  x: float
  y: float
  width: float
  height: float


class TrafficLightStateStabilizer:
  """Debounce phone/model signal observations for a display-only traffic light."""

  def __init__(self, *, required_samples: int = 3, hold_seconds: float = 1.5) -> None:
    if required_samples < 1:
      raise ValueError("required_samples must be positive")
    if not math.isfinite(hold_seconds) or hold_seconds <= 0.0:
      raise ValueError("hold_seconds must be positive")
    self.required_samples = required_samples
    self.hold_seconds = hold_seconds
    self.state = "unknown"
    self._candidate = "unknown"
    self._candidate_samples = 0
    self._last_sample_id: int | None = None
    self._last_detection_time = 0.0
    self._last_supported_time = 0.0

  def update(
      self,
      *,
      detected: bool,
      phone_state: str,
      model_state: int,
      sample_id: int,
      now: float | None = None,
  ) -> str:
    now_value = time.monotonic() if now is None else float(now)
    if detected:
      self._last_detection_time = now_value

    candidate = resolve_traffic_light_state(phone_state, model_state)
    if detected and candidate in TRAFFIC_LIGHT_STATES and sample_id != self._last_sample_id:
      self._last_sample_id = sample_id
      self._last_supported_time = now_value
      if candidate == self._candidate:
        self._candidate_samples += 1
      else:
        self._candidate = candidate
        self._candidate_samples = 1
      if self._candidate_samples >= self.required_samples:
        self.state = candidate
    elif sample_id != self._last_sample_id:
      self._last_sample_id = sample_id
      self._candidate = "unknown"
      self._candidate_samples = 0

    last_evidence = max(self._last_detection_time, self._last_supported_time)
    if not detected and now_value - last_evidence > self.hold_seconds:
      self.reset()
    elif detected and candidate == "unknown" and now_value - self._last_supported_time > self.hold_seconds:
      self.state = "unknown"
    return self.state

  def reset(self) -> None:
    self.state = "unknown"
    self._candidate = "unknown"
    self._candidate_samples = 0
    self._last_sample_id = None
    self._last_detection_time = 0.0
    self._last_supported_time = 0.0


def _field(value: Any, name: str, default: Any = None) -> Any:
  if isinstance(value, dict):
    return value.get(name, default)
  try:
    return getattr(value, name)
  except Exception:
    return default


def phone_ai_compute_badge(state: Any) -> str:
  """Return the active external compute badge, or an empty string while disconnected."""
  if not bool(_field(state, "valid", False)) or not bool(_field(state, "connected", False)):
    return ""
  backend = str(_field(state, "backend", "") or "").strip().lower()
  if backend in NPU_BADGE_BACKENDS or backend.startswith("qnn-"):
    return "eNPU"
  if backend in MIXED_NPU_BADGE_BACKENDS:
    return "eNPU+CPU"
  if backend in UNVERIFIED_QNN_BADGE_BACKENDS:
    return "eQNN?"
  if backend in GENERIC_ACCEL_BADGE_BACKENDS:
    return "eACCEL"
  if backend in CPU_BADGE_BACKENDS or backend.startswith("onnxruntime-cpu-"):
    return "eCPU"
  return ""


def resolve_traffic_light_state(phone_state: Any, model_state: Any) -> str:
  """Fuse phone color analysis with the existing C3X traffic state for display only."""
  normalized_phone = str(phone_state or "unknown").strip().lower()
  try:
    normalized_model = int(model_state)
  except (TypeError, ValueError):
    normalized_model = 0
  # The stock C3X planner is preferred for red/green. It currently has no yellow state,
  # so a confident phone crop analysis supplies yellow and acts as the fallback.
  if normalized_phone == "yellow":
    return "yellow"
  if normalized_model == 1:
    return "red"
  if normalized_model == 2:
    return "green"
  return normalized_phone if normalized_phone in TRAFFIC_LIGHT_STATES else "unknown"


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
  try:
    inference_ms = float(_field(state, "inferenceMs", 0.0))
  except (TypeError, ValueError):
    inference_ms = 0.0
  if not math.isfinite(inference_ms) or inference_ms < 0.0:
    inference_ms = 0.0
  try:
    input_width = int(_field(state, "inputWidth", 0))
  except (TypeError, ValueError):
    input_width = 0
  objects = _field(state, "objects", ()) or ()
  try:
    object_count = len(objects)
  except TypeError:
    object_count = 0
  if input_width > 0 and inference_ms > 0.0:
    return f"외부 AI · {backend} · {input_width} · 총{latency_ms:.0f}/AI{inference_ms:.0f}ms · {object_count}개", True
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
