from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


DEFAULT_FRAME_ASPECT = 640.0 / 384.0
DEFAULT_VERTICAL_FOV_DEG = 50.0
MIN_PROJECTED_DISTANCE_M = 3.0
MAX_PROJECTED_DISTANCE_M = 100.0
MAX_PROJECTED_LATERAL_M = 12.0
CLASS_HEIGHT_M = {
  "car": 1.50,
  "truck": 3.20,
  "bus": 3.40,
  "motorcycle": 1.35,
  "bicycle": 1.25,
  "person": 1.75,
}


@dataclass(frozen=True, slots=True)
class ProjectedExternalAIObject:
  class_name: str
  confidence: float
  longitudinal_m: float
  lateral_m: float
  object_track_id: int


def _field(value: Any, name: str, default: Any = None) -> Any:
  if isinstance(value, dict):
    return value.get(name, default)
  try:
    return getattr(value, name)
  except Exception:
    return default


def project_normalized_bbox(
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    vertical_fov_deg: float = DEFAULT_VERTICAL_FOV_DEG,
    frame_aspect: float = DEFAULT_FRAME_ASPECT,
) -> tuple[float, float] | None:
  normalized_class = str(class_name or "").strip().lower()
  object_height_m = CLASS_HEIGHT_M.get(normalized_class)
  values = (x1, y1, x2, y2, vertical_fov_deg, frame_aspect)
  if object_height_m is None or not all(math.isfinite(value) for value in values):
    return None
  if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
    return None
  if not 10.0 <= vertical_fov_deg <= 140.0 or frame_aspect <= 0.0:
    return None

  box_height = y2 - y1
  tan_vertical = math.tan(math.radians(vertical_fov_deg) * 0.5)
  focal_y_normalized = 1.0 / (2.0 * tan_vertical)
  distance_m = object_height_m * focal_y_normalized / box_height
  distance_m = max(MIN_PROJECTED_DISTANCE_M, min(MAX_PROJECTED_DISTANCE_M, distance_m))

  center_x = (x1 + x2) * 0.5
  tan_horizontal = tan_vertical * frame_aspect
  lateral_m = (center_x - 0.5) * 2.0 * tan_horizontal * distance_m
  lateral_m = max(-MAX_PROJECTED_LATERAL_M, min(MAX_PROJECTED_LATERAL_M, lateral_m))
  return distance_m, lateral_m


def project_phone_ai_state(
    state: Any,
    *,
    vertical_fov_deg: float = DEFAULT_VERTICAL_FOV_DEG,
    frame_aspect: float = DEFAULT_FRAME_ASPECT,
) -> tuple[ProjectedExternalAIObject, ...]:
  if not bool(_field(state, "valid", False)) or not bool(_field(state, "connected", False)):
    return ()
  try:
    frame_id = int(_field(state, "frameId", 0))
  except (TypeError, ValueError):
    return ()
  if frame_id < 0:
    return ()

  projected: list[ProjectedExternalAIObject] = []
  for index, item in enumerate(_field(state, "objects", ()) or ()):
    class_name = str(_field(item, "className", "") or "").strip().lower()
    try:
      confidence = float(_field(item, "confidence", 0.0))
      coordinates = tuple(float(_field(item, name)) for name in ("x1", "y1", "x2", "y2"))
    except (TypeError, ValueError):
      continue
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
      continue
    location = project_normalized_bbox(
      class_name,
      *coordinates,
      vertical_fov_deg=vertical_fov_deg,
      frame_aspect=frame_aspect,
    )
    if location is None:
      continue
    projected.append(ProjectedExternalAIObject(
      class_name=class_name,
      confidence=confidence,
      longitudinal_m=location[0],
      lateral_m=location[1],
      object_track_id=(frame_id << 8) | index,
    ))
  return tuple(projected)
