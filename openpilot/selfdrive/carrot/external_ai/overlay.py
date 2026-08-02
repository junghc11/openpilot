from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


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
