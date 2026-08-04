from __future__ import annotations

import time

import pyray as rl

from openpilot.selfdrive.carrot.external_ai.overlay import (
  ExternalAIOverlayObject,
  TrafficLightStateStabilizer,
  phone_ai_compute_badge,
  phone_ai_overlay_objects,
  phone_ai_status_text,
)
from openpilot.selfdrive.ui.onroad.external_ai_labels import external_ai_display_name
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, font_fallback, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_draw import draw_text_ui_style
from openpilot.system.ui.lib.text_measure import measure_text_cached


PARAM_REFRESH_INTERVAL_S = 1.0
VEHICLE_CLASSES = frozenset(("car", "truck", "bus", "motorcycle", "bicycle"))


class ExternalAIOverlayRenderer:
  def __init__(self) -> None:
    self._external_ai_enabled = False
    self._show_overlay = False
    self._next_param_refresh = 0.0
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_display = gui_app.font(FontWeight.DISPLAY)
    self._traffic_light = TrafficLightStateStabilizer(required_samples=3, hold_seconds=1.5)

  def _refresh_enabled(self) -> None:
    now = time.monotonic()
    if now < self._next_param_refresh:
      return
    self._next_param_refresh = now + PARAM_REFRESH_INTERVAL_S
    try:
      self._external_ai_enabled = ui_state.params.get_bool("ExternalAIEnabled")
      self._show_overlay = ui_state.params.get_bool("ExternalAIShowOverlay")
    except Exception:
      self._external_ai_enabled = False
      self._show_overlay = False

  def render(self, rect: rl.Rectangle) -> None:
    self._refresh_enabled()
    if not self._external_ai_enabled:
      return
    try:
      service_alive = bool(ui_state.sm.alive["phoneAIState"])
      service_valid = bool(ui_state.sm.valid["phoneAIState"])
      state = ui_state.sm["phoneAIState"] if service_alive and service_valid else None
    except Exception:
      service_alive = False
      service_valid = False
      state = None
    compute_badge = phone_ai_compute_badge(state)
    if not self._show_overlay:
      if compute_badge:
        self._draw_compute_badge(rect, compute_badge, below_status=False)
      return
    status_text, connected = phone_ai_status_text(
      state,
      service_alive=service_alive,
      service_valid=service_valid,
    )
    if state is None:
      self._draw_status(rect, status_text, connected)
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
    traffic_detected = any(item.class_name == "traffic light" for item in objects)
    try:
      phone_traffic_state = str(state.trafficLightState)
    except Exception:
      phone_traffic_state = "unknown"
    try:
      model_traffic_state = int(ui_state.sm["longitudinalPlan"].trafficState)
    except Exception:
      try:
        model_traffic_state = int(ui_state.sm["carrotMan"].trafficState)
      except Exception:
        model_traffic_state = 0
    try:
      sample_id = int(state.frameId)
    except Exception:
      sample_id = 0
    stable_traffic_state = self._traffic_light.update(
      detected=traffic_detected,
      phone_state=phone_traffic_state,
      model_state=model_traffic_state,
      sample_id=sample_id,
    )
    if stable_traffic_state != "unknown":
      self._draw_signal_indicator(rect, stable_traffic_state)
    self._draw_status(rect, status_text, connected)
    if compute_badge:
      self._draw_compute_badge(rect, compute_badge, below_status=True)

  def _draw_status(self, rect: rl.Rectangle, text: str, connected: bool) -> None:
    font_size = max(22, min(32, int(rect.height * 0.032)))
    measured = measure_text_cached(self._font, text, font_size, 0.0)
    padding_x = 18.0
    padding_y = 10.0
    width = measured.x + padding_x * 2.0
    height = measured.y + padding_y * 2.0
    x = rect.x + (rect.width - width) * 0.5
    # Clear the stock/debug text occupying the top edge of C3/C3X/C4 displays.
    status_top = max(84.0, min(132.0, rect.height * 0.10))
    y = rect.y + status_top
    panel = rl.Rectangle(x, y, width, height)
    accent = rl.Color(80, 220, 140, 235) if connected else rl.Color(255, 184, 64, 235)
    rl.draw_rectangle_rounded(panel, 0.45, 10, rl.Color(8, 12, 16, 205))
    rl.draw_rectangle_rounded_lines_ex(panel, 0.45, 10, 2.0, accent)
    draw_text_ui_style(
      text,
      x + width * 0.5,
      y + padding_y,
      font_size,
      rl.WHITE,
      font=self._font,
      border_width=1.0,
      shadow_offset=2.0,
      align="center_top",
      y_offset=0.0,
    )

  def _draw_compute_badge(self, rect: rl.Rectangle, text: str, *, below_status: bool) -> None:
    del below_status
    width = 112.0
    height = 42.0
    x = rect.x + 14.0
    y = rect.y + rect.height - height - 14.0
    badge = rl.Rectangle(x, y, width, height)
    fill = {
      "eNPU": rl.GREEN,
    }.get(text, rl.Color(0, 122, 255, 230))
    rl.draw_rectangle_rounded(badge, 0.25, 8, fill)
    rl.draw_rectangle_rounded_lines_ex(badge, 0.25, 8, 2.0, rl.WHITE)
    draw_text_ui_style(
      text,
      x + width * 0.5,
      y + height - 10.0,
      27,
      rl.WHITE,
      font=self._font_display,
      border_width=2.0,
      shadow_offset=4.0,
      align="center_bottom",
      y_offset=0.0,
    )

  @staticmethod
  def _color(item: ExternalAIOverlayObject) -> rl.Color:
    alpha = int(45 + 65 * item.confidence)
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
    label = f"{external_ai_display_name(item.class_name, translate=tr)} {item.confidence * 100.0:.0f}%"
    font_size = max(18, min(34, int(height * 0.13)))
    label_font = font_fallback(self._font)
    measured = measure_text_cached(label_font, label, font_size, 0.0)
    label_rect = rl.Rectangle(x + 3.0, y + 3.0, measured.x + 16.0, measured.y + 10.0)
    rl.draw_rectangle_rounded(label_rect, 0.25, 6, rl.Color(5, 8, 12, 205))
    draw_text_ui_style(
      label,
      x + 11.0,
      y + 8.0,
      font_size,
      rl.WHITE,
      font=label_font,
      border_width=1.0,
      shadow_offset=2.0,
      align="left_top",
      y_offset=0.0,
    )

  @staticmethod
  def _draw_car(x, y, width, height, color) -> None:
    # Rear-view passenger-car silhouette: tapered cabin, rear glass, lamps and bumper.
    body = rl.Rectangle(x + width * 0.10, y + height * 0.47, width * 0.80, height * 0.34)
    roof_left = rl.Vector2(x + width * 0.25, y + height * 0.50)
    roof_top_left = rl.Vector2(x + width * 0.36, y + height * 0.24)
    roof_top_right = rl.Vector2(x + width * 0.64, y + height * 0.24)
    roof_right = rl.Vector2(x + width * 0.75, y + height * 0.50)
    rl.draw_rectangle_rounded(body, 0.22, 8, color)
    rl.draw_triangle(roof_left, roof_top_left, roof_top_right, color)
    rl.draw_triangle(roof_left, roof_top_right, roof_right, color)

    glass = rl.Color(16, 30, 42, min(175, color.a + 55))
    rear_glass = rl.Rectangle(x + width * 0.31, y + height * 0.30, width * 0.38, height * 0.20)
    rl.draw_rectangle_rounded(rear_glass, 0.18, 6, glass)
    lamp_alpha = min(215, color.a + 95)
    lamp_width = max(3.0, width * 0.14)
    lamp_height = max(2.0, height * 0.055)
    rl.draw_rectangle_rounded(
      rl.Rectangle(x + width * 0.16, y + height * 0.58, lamp_width, lamp_height),
      0.35, 4, rl.Color(255, 70, 55, lamp_alpha),
    )
    rl.draw_rectangle_rounded(
      rl.Rectangle(x + width * 0.70, y + height * 0.58, lamp_width, lamp_height),
      0.35, 4, rl.Color(255, 70, 55, lamp_alpha),
    )
    bumper_y = y + height * 0.75
    rl.draw_line_ex(
      rl.Vector2(x + width * 0.24, bumper_y),
      rl.Vector2(x + width * 0.76, bumper_y),
      max(2.0, height * 0.018),
      rl.Color(225, 238, 245, min(190, color.a + 70)),
    )
    radius = max(2.0, min(width, height) * 0.075)
    wheel = rl.Color(6, 8, 10, min(225, color.a + 115))
    rl.draw_circle(int(x + width * 0.22), int(y + height * 0.82), radius, wheel)
    rl.draw_circle(int(x + width * 0.78), int(y + height * 0.82), radius, wheel)

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
    glass = rl.Color(20, 48, 70, min(165, color.a + 50))
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

  @staticmethod
  def _draw_signal_indicator(rect: rl.Rectangle, state: str) -> None:
    width = 58.0
    height = 154.0
    x = rect.x + rect.width - width - 18.0
    y = rect.y + 18.0
    panel = rl.Rectangle(x, y, width, height)
    rl.draw_rectangle_rounded(panel, 0.28, 10, rl.Color(7, 10, 13, 225))
    rl.draw_rectangle_rounded_lines_ex(panel, 0.28, 10, 2.0, rl.Color(205, 218, 226, 190))
    lamps = (
      ("red", rl.Color(255, 56, 52, 255)),
      ("yellow", rl.Color(255, 196, 38, 255)),
      ("green", rl.Color(43, 224, 102, 255)),
    )
    for index, (lamp_state, active_color) in enumerate(lamps):
      center_x = int(x + width * 0.5)
      center_y = int(y + 29.0 + index * 48.0)
      if lamp_state == state:
        rl.draw_circle(center_x, center_y, 19.0, rl.Color(active_color.r, active_color.g, active_color.b, 70))
        rl.draw_circle(center_x, center_y, 14.0, active_color)
      else:
        rl.draw_circle(center_x, center_y, 13.0, rl.Color(55, 61, 66, 210))
