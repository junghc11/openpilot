from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass


PROTOCOL_VERSION = 1
DEFAULT_MAX_LATENCY_MS = 300.0
DEFAULT_CONNECTION_TIMEOUT_MS = 2_000.0
DEFAULT_MAX_OBJECTS = 64
MAX_RESULT_BYTES = 64 * 1024
MAX_CLOCK_LEAD_MS = 50.0
SUPPORTED_OBJECT_CLASSES = frozenset((
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "bus",
  "truck",
  "traffic light",
  "stop sign",
))
SUPPORTED_TRAFFIC_LIGHT_STATES = frozenset(("unknown", "red", "yellow", "green"))
SUPPORTED_SCENE_MODES = frozenset(("day", "night"))
SUPPORTED_PERFORMANCE_MODES = frozenset(("normal", "reduced", "thermal"))


class ExternalAIProtocolError(ValueError):
  pass


@dataclass(frozen=True, slots=True)
class ExternalAIObject:
  class_id: int
  class_name: str
  confidence: float
  x1: float
  y1: float
  x2: float
  y2: float
  track_id: int = 0

  @property
  def center_x(self) -> float:
    return (self.x1 + self.x2) * 0.5

  @property
  def center_y(self) -> float:
    return (self.y1 + self.y2) * 0.5

  @property
  def width(self) -> float:
    return self.x2 - self.x1

  @property
  def height(self) -> float:
    return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class ExternalAIResult:
  protocol_version: int
  frame_id: int
  source_timestamp_monotonic_ns: int
  phone_receive_timestamp_ns: int
  inference_start_timestamp_ns: int
  inference_end_timestamp_ns: int
  model: str
  backend: str
  objects: tuple[ExternalAIObject, ...]
  c3x_receive_timestamp_ns: int
  latency_ms: float
  inference_ms: float
  decode_ms: float
  preprocess_ms: float
  runtime_ms: float
  postprocess_ms: float
  phone_total_ms: float
  input_width: int
  input_height: int
  traffic_light_state: str
  traffic_light_confidence: float
  scene_mode: str = "day"
  scene_brightness: float = 1.0
  effective_fps: int = 0
  performance_mode: str = "normal"


def _mapping(value: object, field: str) -> dict:
  if not isinstance(value, dict):
    raise ExternalAIProtocolError(f"{field} must be an object")
  return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise ExternalAIProtocolError(f"{field} must be an integer >= {minimum}")
  return value


def _number(value: object, field: str, *, minimum: float, maximum: float) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ExternalAIProtocolError(f"{field} must be numeric")
  parsed = float(value)
  if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
    raise ExternalAIProtocolError(f"{field} must be between {minimum} and {maximum}")
  return parsed


def _optional_number(root: dict, field: str, *, minimum: float, maximum: float, default: float = 0.0) -> float:
  value = root.get(field)
  return default if value is None else _number(value, field, minimum=minimum, maximum=maximum)


def _optional_integer(root: dict, field: str, *, minimum: int, maximum: int, default: int = 0) -> int:
  value = root.get(field)
  if value is None:
    return default
  parsed = _integer(value, field, minimum=minimum)
  if parsed > maximum:
    raise ExternalAIProtocolError(f"{field} must be <= {maximum}")
  return parsed


def _short_text(value: object, field: str, *, maximum_length: int = 64) -> str:
  if not isinstance(value, str):
    raise ExternalAIProtocolError(f"{field} must be text")
  parsed = value.strip()
  if not parsed or len(parsed) > maximum_length:
    raise ExternalAIProtocolError(f"{field} must contain 1..{maximum_length} characters")
  return parsed


def _optional_traffic_light_state(root: dict) -> str:
  value = root.get("traffic_light_state", "unknown")
  if not isinstance(value, str):
    raise ExternalAIProtocolError("traffic_light_state must be text")
  parsed = value.strip().lower()
  if parsed not in SUPPORTED_TRAFFIC_LIGHT_STATES:
    raise ExternalAIProtocolError("traffic_light_state is unsupported")
  return parsed


def _optional_enum_text(root: dict, field: str, supported: frozenset[str], default: str) -> str:
  value = root.get(field, default)
  if not isinstance(value, str):
    raise ExternalAIProtocolError(f"{field} must be text")
  parsed = value.strip().lower()
  if parsed not in supported:
    raise ExternalAIProtocolError(f"{field} is unsupported")
  return parsed


def _parse_object(value: object, index: int) -> ExternalAIObject:
  item = _mapping(value, f"objects[{index}]")
  class_name = _short_text(item.get("class_name"), f"objects[{index}].class_name").lower()
  if class_name not in SUPPORTED_OBJECT_CLASSES:
    raise ExternalAIProtocolError(f"objects[{index}].class_name is unsupported")
  x1 = _number(item.get("x1"), f"objects[{index}].x1", minimum=0.0, maximum=1.0)
  y1 = _number(item.get("y1"), f"objects[{index}].y1", minimum=0.0, maximum=1.0)
  x2 = _number(item.get("x2"), f"objects[{index}].x2", minimum=0.0, maximum=1.0)
  y2 = _number(item.get("y2"), f"objects[{index}].y2", minimum=0.0, maximum=1.0)
  if x2 <= x1 or y2 <= y1:
    raise ExternalAIProtocolError(f"objects[{index}] bounding box is empty or reversed")
  return ExternalAIObject(
    class_id=_integer(item.get("class_id"), f"objects[{index}].class_id"),
    class_name=class_name,
    confidence=_number(item.get("confidence"), f"objects[{index}].confidence", minimum=0.0, maximum=1.0),
    x1=x1,
    y1=y1,
    x2=x2,
    y2=y2,
    track_id=_optional_integer(item, "track_id", minimum=0, maximum=1_000_000),
  )


def parse_external_ai_result(
    payload: bytes | bytearray | memoryview | str,
    *,
    now_monotonic_ns: int | None = None,
    previous_frame_id: int | None = None,
    max_latency_ms: float = DEFAULT_MAX_LATENCY_MS,
    max_objects: int = DEFAULT_MAX_OBJECTS,
) -> ExternalAIResult:
  if isinstance(payload, str):
    encoded_size = len(payload.encode("utf-8"))
    raw_text = payload
  elif isinstance(payload, (bytes, bytearray, memoryview)):
    encoded_size = len(payload)
    try:
      raw_text = bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
      raise ExternalAIProtocolError("result is not valid UTF-8") from exc
  else:
    raise ExternalAIProtocolError("result payload must be bytes or text")
  if encoded_size > MAX_RESULT_BYTES:
    raise ExternalAIProtocolError(f"result exceeds {MAX_RESULT_BYTES} bytes")

  try:
    document = json.loads(raw_text)
  except (json.JSONDecodeError, RecursionError) as exc:
    raise ExternalAIProtocolError("result is not valid JSON") from exc
  root = _mapping(document, "result")

  protocol_version = _integer(root.get("protocol_version"), "protocol_version", minimum=1)
  if protocol_version != PROTOCOL_VERSION:
    raise ExternalAIProtocolError(f"unsupported protocol_version {protocol_version}")
  frame_id = _integer(root.get("frame_id"), "frame_id")
  if previous_frame_id is not None and frame_id <= previous_frame_id:
    raise ExternalAIProtocolError("frame_id did not advance")

  source_ns = _integer(root.get("source_timestamp_monotonic_ns"), "source_timestamp_monotonic_ns", minimum=1)
  phone_receive_ns = _integer(root.get("phone_receive_timestamp_ns"), "phone_receive_timestamp_ns", minimum=1)
  inference_start_ns = _integer(root.get("inference_start_timestamp_ns"), "inference_start_timestamp_ns", minimum=1)
  inference_end_ns = _integer(root.get("inference_end_timestamp_ns"), "inference_end_timestamp_ns", minimum=1)
  if not phone_receive_ns <= inference_start_ns <= inference_end_ns:
    raise ExternalAIProtocolError("phone inference timestamps are out of order")

  now_ns = time.monotonic_ns() if now_monotonic_ns is None else _integer(now_monotonic_ns, "now_monotonic_ns", minimum=1)
  latency_ms = (now_ns - source_ns) / 1_000_000.0
  if latency_ms < -MAX_CLOCK_LEAD_MS:
    raise ExternalAIProtocolError("source timestamp is in the future")
  if not math.isfinite(max_latency_ms) or max_latency_ms <= 0.0:
    raise ValueError("max_latency_ms must be positive")
  if latency_ms > max_latency_ms:
    raise ExternalAIProtocolError("result exceeded maximum latency")

  if isinstance(max_objects, bool) or not isinstance(max_objects, int) or max_objects < 0:
    raise ValueError("max_objects must be a non-negative integer")
  object_values = root.get("objects")
  if not isinstance(object_values, list):
    raise ExternalAIProtocolError("objects must be an array")
  if len(object_values) > max_objects:
    raise ExternalAIProtocolError("result exceeded maximum object count")
  objects = tuple(_parse_object(value, index) for index, value in enumerate(object_values))

  decode_ms = _optional_number(root, "decode_ms", minimum=0.0, maximum=60_000.0)
  preprocess_ms = _optional_number(root, "preprocess_ms", minimum=0.0, maximum=60_000.0)
  runtime_ms = _optional_number(root, "runtime_ms", minimum=0.0, maximum=60_000.0)
  postprocess_ms = _optional_number(root, "postprocess_ms", minimum=0.0, maximum=60_000.0)
  phone_total_ms = _optional_number(root, "phone_total_ms", minimum=0.0, maximum=60_000.0)
  input_width = _optional_integer(root, "input_width", minimum=32, maximum=8192)
  input_height = _optional_integer(root, "input_height", minimum=32, maximum=8192)
  if (input_width == 0) != (input_height == 0):
    raise ExternalAIProtocolError("input_width and input_height must be provided together")
  traffic_light_state = _optional_traffic_light_state(root)
  traffic_light_confidence = _optional_number(root, "traffic_light_confidence", minimum=0.0, maximum=1.0)
  scene_mode = _optional_enum_text(root, "scene_mode", SUPPORTED_SCENE_MODES, "day")
  scene_brightness = _optional_number(root, "scene_brightness", minimum=0.0, maximum=1.0, default=1.0)
  effective_fps = _optional_integer(root, "effective_fps", minimum=0, maximum=120)
  performance_mode = _optional_enum_text(root, "performance_mode", SUPPORTED_PERFORMANCE_MODES, "normal")

  return ExternalAIResult(
    protocol_version=protocol_version,
    frame_id=frame_id,
    source_timestamp_monotonic_ns=source_ns,
    phone_receive_timestamp_ns=phone_receive_ns,
    inference_start_timestamp_ns=inference_start_ns,
    inference_end_timestamp_ns=inference_end_ns,
    model=_short_text(root.get("model"), "model"),
    backend=_short_text(root.get("backend"), "backend"),
    objects=objects,
    c3x_receive_timestamp_ns=now_ns,
    latency_ms=latency_ms,
    inference_ms=(inference_end_ns - inference_start_ns) / 1_000_000.0,
    decode_ms=decode_ms,
    preprocess_ms=preprocess_ms,
    runtime_ms=runtime_ms,
    postprocess_ms=postprocess_ms,
    phone_total_ms=phone_total_ms,
    input_width=input_width,
    input_height=input_height,
    traffic_light_state=traffic_light_state,
    traffic_light_confidence=traffic_light_confidence,
    scene_mode=scene_mode,
    scene_brightness=scene_brightness,
    effective_fps=effective_fps,
    performance_mode=performance_mode,
  )


class ExternalAIResultTracker:
  def __init__(
      self,
      *,
      max_latency_ms: float = DEFAULT_MAX_LATENCY_MS,
      connection_timeout_ms: float = DEFAULT_CONNECTION_TIMEOUT_MS,
      max_objects: int = DEFAULT_MAX_OBJECTS,
  ) -> None:
    if not math.isfinite(connection_timeout_ms) or connection_timeout_ms <= 0.0:
      raise ValueError("connection_timeout_ms must be positive")
    self.max_latency_ms = max_latency_ms
    self.connection_timeout_ns = int(connection_timeout_ms * 1_000_000.0)
    self.max_objects = max_objects
    self.last_result: ExternalAIResult | None = None
    self.last_receive_timestamp_ns: int | None = None

  def accept(self, payload: bytes | str, *, now_monotonic_ns: int | None = None) -> ExternalAIResult:
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    previous_frame_id = self.last_result.frame_id if self.last_result is not None else None
    result = parse_external_ai_result(
      payload,
      now_monotonic_ns=now_ns,
      previous_frame_id=previous_frame_id,
      max_latency_ms=self.max_latency_ms,
      max_objects=self.max_objects,
    )
    self.last_result = result
    self.last_receive_timestamp_ns = now_ns
    return result

  def connected(self, *, now_monotonic_ns: int | None = None) -> bool:
    if self.last_receive_timestamp_ns is None:
      return False
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    return 0 <= now_ns - self.last_receive_timestamp_ns <= self.connection_timeout_ns

  def fresh_result(self, *, now_monotonic_ns: int | None = None) -> ExternalAIResult | None:
    return self.last_result if self.connected(now_monotonic_ns=now_monotonic_ns) else None
