from pathlib import Path
import sys

import pytest


CLUSTER_DIR = Path(__file__).resolve().parents[1] / "cluster"
sys.path.insert(0, str(CLUSTER_DIR))

from cluster_vehicle_visuals import (
  VEHICLE_VISUAL_SPECS,
  build_vehicle_mesh_data,
  vehicle_model_key_for_render,
  visual_spec_for_object_class,
)


@pytest.mark.parametrize(
  ("object_class", "expected_key"),
  (
    ("car", "car"),
    ("SUV", "car"),
    ("pickup", "truck"),
    ("bus", "bus"),
    ("motorbike", "motorcycle"),
    ("bicycle", "bicycle"),
    ("pedestrian", "person"),
  ),
)
def test_object_class_selects_expected_visual(object_class, expected_key):
  spec = visual_spec_for_object_class(object_class)

  assert spec is not None
  assert spec.model_key == expected_key


def test_unknown_object_class_keeps_marker_fallback():
  assert visual_spec_for_object_class("traffic light") is None
  assert visual_spec_for_object_class("") is None


@pytest.mark.parametrize("model_key", tuple(VEHICLE_VISUAL_SPECS))
def test_generated_mesh_buffers_are_valid_and_normalized(model_key):
  mesh = build_vehicle_mesh_data(model_key)
  vertex_count = len(mesh.vertices) // 3

  assert vertex_count >= 24
  assert vertex_count % 3 == 0
  assert len(mesh.normals) == len(mesh.vertices)
  assert len(mesh.colors) == vertex_count * 4

  xs = mesh.vertices[0::3]
  ys = mesh.vertices[1::3]
  zs = mesh.vertices[2::3]
  assert min(xs) >= -0.56
  assert max(xs) <= 0.56
  assert min(ys) >= -0.56
  assert max(ys) <= 0.56
  assert min(zs) >= -0.01
  assert max(zs) <= 1.05


def test_each_object_class_has_real_world_dimensions():
  assert VEHICLE_VISUAL_SPECS["truck"].length_m > VEHICLE_VISUAL_SPECS["car"].length_m
  assert VEHICLE_VISUAL_SPECS["bus"].height_m > VEHICLE_VISUAL_SPECS["car"].height_m
  assert VEHICLE_VISUAL_SPECS["motorcycle"].width_m < VEHICLE_VISUAL_SPECS["car"].width_m
  assert VEHICLE_VISUAL_SPECS["person"].length_m < VEHICLE_VISUAL_SPECS["bicycle"].length_m


def test_external_ai_class_model_overrides_marker_fallback():
  available = {"cybertruck", "car", "truck"}

  assert vehicle_model_key_for_render("car", "externalAI", False, False, available) == "car"
  assert vehicle_model_key_for_render("truck", "radarState", False, False, available) == "truck"
  assert vehicle_model_key_for_render("person", "externalAI", False, False, available) is None


def test_legacy_radar_and_model_objects_keep_marker_behavior():
  available = {"cybertruck", "car"}

  assert vehicle_model_key_for_render("", "radarState", True, False, available) is None
  assert vehicle_model_key_for_render("", "modelV2", True, False, available) is None
  assert vehicle_model_key_for_render("", "route", True, False, available) == "cybertruck"
