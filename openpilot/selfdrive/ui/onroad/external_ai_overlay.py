from __future__ import annotations

import time

import pyray as rl

from openpilot.selfdrive.carrot.external_ai.overlay import ExternalAIOverlayObject, phone_ai_overlay_objects
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_draw import draw_text_ui_style


PARAM_REFRESH_INTERVAL_S = 1.0
VEHICLE_CLASSES = frozenset(("car", "truck", "bus", "motorcycle", "bicycle"))


class ExternalAIOverlayRenderer:
  def __init__(self) -> None:
    self._enabled = False
    self._next_param_refresh = 0.0
    self._font = gui_app.font(FontWeight.SEMI_BOLD)

  def _refresh_enabled(self) -> None:
    now = time.monotonic()
    if now < self._next_param_refresh:
      return
    self._next_param_refresh = now + PARAM_REFRESH_INTERVAL_S
    try:
      self._enabled = ui_state.params.get_bool("ExternalAIShowOverlay")
    except Exception:
      self._enabled = False

  def render(self, rect: rl.Rectangle) -> None:
    self._refresh_enabled()
    if not self._enabled:
      return
    try:
      if not ui_state.sm.alive["phoneAIState"] or not ui_state.sm.valid["phoneAIState"]:
        return
      state = ui_state.sm["phoneAIState"]
    except Exception:
      return
    objects = phone_ai_overlay_objects(
      state,
      screen_x=float(rect.x),
      screen_y=float(rect.y),
      screen_width=float(rect.width),
      screen_height=float(rect.height),
    )
    for item in objects:
      self._draw_object(item)

  @staticmethod
  def _color(item: ExternalAIOverlayObject) -> rl.Color:
    alpha = int(110 + 110 * item.confidence)
    if item.class_name in ("person", "bicycle", "motorcycle"):
      return rl.Color(255, 174, 52, alpha)
    if item.class_name in ("traffic light", "stop sign"):
      return rl.Color(255, 72, 72, alpha)
    return rl.Color(76, 205, 255, alpha)

  def _draw_object(self, item: ExternalAIOverlayObject) -> None:
    color = self._color(item)
    x, y, width, height = item.x, item.y, item.width, item.height
    if item.class_name == "car":
      self._draw_car(x, y, width, height, color)
    elif item.class_name == "truck":
      self._draw_truck(x, y, width, height, color)
    elif item.class_name == "bus":
      self._draw_bus(x, y, width, height, color)
    elif item.class_name == "motorcycle":
      self._draw_motorcycle(x, y, width, height, color)
    elif item.class_name == "bicycle":
      self._draw_bicycle(x, y, width, height, color)
    elif item.class_name == "person":
      self._draw_person(x, y, width, height, color)
    elif item.class_name == "traffic light":
      self._draw_traffic_light(x, y, width, height, color)
    elif item.class_name == "stop sign":
      self._draw_stop_sign(x, y, width, height, color)
    else:
      return

    outline = rl.Color(color.r, color.g, color.b, 230)
    frame = rl.Rectangle(x, y, width, height)
    rl.draw_rectangle_rounded_lines_ex(frame, 0.10, 6, max(2.0, min(width, height) * 0.018), outline)
    label = f"{item.class_name.upper()} {item.confidence * 100.0:.0f}%"
    font_size = max(18, min(34, int(height * 0.13)))
    draw_text_ui_style(
      label,
      x + 4.0,
      y + 2.0,
      font_size,
      rl.WHITE,
      font=self._font,
      border_width=1.0,
      shadow_offset=2.0,
      align="left_top",
      y_offset=0.0,
    )

  @staticmethod
  def _draw_car(x, y, width, height, color) -> None:
    body = rl.Rectangle(x + width * 0.08, y + height * 0.48, width * 0.84, height * 0.36)
    roof = (rl.Vector2(x + width * 0.25, y + height * 0.50),
            rl.Vector2(x + width * 0.37, y + height * 0.28),
            rl.Vector2(x + width * 0.68, y + height * 0.28),
            rl.Vector2(x + width * 0.82, y + height * 0.50))
    rl.draw_rectangle_rounded(body, 0.28, 8, color)
    rl.draw_triangle(roof[0], roof[1], roof[2], color)
    rl.draw_triangle(roof[0], roof[2], roof[3], color)
    radius = max(2.0, min(width, height) * 0.09)
    rl.draw_circle(int(x + width * 0.24), int(y + height * 0.84), radius, rl.BLACK)
    rl.draw_circle(int(x + width * 0.76), int(y + height * 0.84), radius, rl.BLACK)

  @staticmethod
  def _draw_truck(x, y, width, height, color) -> None:
    rl.draw_rectangle_rounded(rl.Rectangle(x + width * 0.05, y + height * 0.18, width * 0.58, height * 0.62), 0.08, 6, color)
    rl.draw_rectangle_rounded(rl.Rectangle(x + width * 0.61, y + height * 0.43, width * 0.34, height * 0.37), 0.16, 6, color)
    radius = max(2.0, min(width, height) * 0.09)
    for center in (0.22, 0.54, 0.80):
      rl.draw_circle(int(x + width * center), int(y + height * 0.82), radius, rl.BLACK)

  @staticmethod
  def _draw_bus(x, y, width, height, color) -> None:
    rl.draw_rectangle_rounded(rl.Rectangle(x + width * 0.12, y + height * 0.08, width * 0.76, height * 0.78), 0.12, 8, color)
    glass = rl.Color(20, 48, 70, 210)
    rl.draw_rectangle_rounded(rl.Rectangle(x + width * 0.22, y + height * 0.18, width * 0.56, height * 0.25), 0.08, 5, glass)
    radius = max(2.0, min(width, height) * 0.08)
    rl.draw_circle(int(x + width * 0.27), int(y + height * 0.86), radius, rl.BLACK)
    rl.draw_circle(int(x + width * 0.73), int(y + height * 0.86), radius, rl.BLACK)

  @staticmethod
  def _draw_motorcycle(x, y, width, height, color) -> None:
    radius = max(3.0, min(width * 0.22, height * 0.18))
    left = rl.Vector2(x + width * 0.25, y + height * 0.78)
    right = rl.Vector2(x + width * 0.75, y + height * 0.78)
    rl.draw_circle_lines(int(left.x), int(left.y), radius, color)
    rl.draw_circle_lines(int(right.x), int(right.y), radius, color)
    rl.draw_line_ex(left, rl.Vector2(x + width * 0.52, y + height * 0.50), max(3.0, width * 0.05), color)
    rl.draw_line_ex(rl.Vector2(x + width * 0.52, y + height * 0.50), right, max(3.0, width * 0.05), color)
    rl.draw_circle(int(x + width * 0.53), int(y + height * 0.30), max(3.0, width * 0.08), color)

  @staticmethod
  def _draw_bicycle(x, y, width, height, color) -> None:
    radius = max(3.0, min(width * 0.23, height * 0.20))
    left = rl.Vector2(x + width * 0.24, y + height * 0.76)
    right = rl.Vector2(x + width * 0.76, y + height * 0.76)
    middle = rl.Vector2(x + width * 0.50, y + height * 0.52)
    rl.draw_circle_lines(int(left.x), int(left.y), radius, color)
    rl.draw_circle_lines(int(right.x), int(right.y), radius, color)
    thickness = max(2.0, width * 0.035)
    rl.draw_line_ex(left, middle, thickness, color)
    rl.draw_line_ex(middle, right, thickness, color)
    rl.draw_line_ex(right, rl.Vector2(x + width * 0.62, y + height * 0.38), thickness, color)
    rl.draw_line_ex(middle, rl.Vector2(x + width * 0.42, y + height * 0.38), thickness, color)

  @staticmethod
  def _draw_person(x, y, width, height, color) -> None:
    center_x = x + width * 0.5
    head_y = y + height * 0.20
    rl.draw_circle(int(center_x), int(head_y), max(3.0, min(width, height) * 0.10), color)
    thickness = max(3.0, width * 0.08)
    shoulder_y = y + height * 0.36
    hip_y = y + height * 0.62
    rl.draw_line_ex(rl.Vector2(center_x, shoulder_y), rl.Vector2(center_x, hip_y), thickness, color)
    rl.draw_line_ex(rl.Vector2(center_x, shoulder_y), rl.Vector2(x + width * 0.18, y + height * 0.52), thickness, color)
    rl.draw_line_ex(rl.Vector2(center_x, shoulder_y), rl.Vector2(x + width * 0.82, y + height * 0.52), thickness, color)
    rl.draw_line_ex(rl.Vector2(center_x, hip_y), rl.Vector2(x + width * 0.24, y + height * 0.90), thickness, color)
    rl.draw_line_ex(rl.Vector2(center_x, hip_y), rl.Vector2(x + width * 0.76, y + height * 0.90), thickness, color)

  @staticmethod
  def _draw_traffic_light(x, y, width, height, color) -> None:
    body = rl.Rectangle(x + width * 0.28, y + height * 0.08, width * 0.44, height * 0.76)
    rl.draw_rectangle_rounded(body, 0.25, 8, rl.Color(20, 24, 28, 220))
    radius = max(3.0, min(width * 0.14, height * 0.11))
    for fraction, light in ((0.27, color), (0.47, rl.Color(255, 190, 40, color.a)), (0.67, rl.Color(60, 220, 100, color.a))):
      rl.draw_circle(int(x + width * 0.5), int(y + height * fraction), radius, light)

  @staticmethod
  def _draw_stop_sign(x, y, width, height, color) -> None:
    rl.draw_poly(
      rl.Vector2(x + width * 0.5, y + height * 0.48),
      8,
      max(4.0, min(width, height) * 0.30),
      22.5,
      color,
    )
