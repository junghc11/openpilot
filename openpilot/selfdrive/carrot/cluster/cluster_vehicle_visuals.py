from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass


Color = tuple[int, int, int, int]
Point3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class VehicleVisualSpec:
    model_key: str
    width_m: float
    length_m: float
    height_m: float


@dataclass(frozen=True, slots=True)
class VehicleMeshData:
    vertices: tuple[float, ...]
    normals: tuple[float, ...]
    colors: tuple[int, ...]


VEHICLE_VISUAL_SPECS: dict[str, VehicleVisualSpec] = {
    "car": VehicleVisualSpec("car", 1.85, 4.50, 1.50),
    "truck": VehicleVisualSpec("truck", 2.50, 7.50, 3.20),
    "bus": VehicleVisualSpec("bus", 2.55, 10.50, 3.40),
    "motorcycle": VehicleVisualSpec("motorcycle", 0.82, 2.20, 1.35),
    "bicycle": VehicleVisualSpec("bicycle", 0.65, 1.80, 1.25),
    "person": VehicleVisualSpec("person", 0.62, 0.45, 1.75),
}

OBJECT_CLASS_ALIASES = {
    "car": "car",
    "vehicle": "car",
    "sedan": "car",
    "suv": "car",
    "van": "car",
    "pickup": "truck",
    "truck": "truck",
    "lorry": "truck",
    "bus": "bus",
    "coach": "bus",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "cyclist": "bicycle",
    "person": "person",
    "pedestrian": "person",
}

MATERIAL_COLORS: dict[str, Color] = {
    "body": (166, 178, 188, 255),
    "body_dark": (92, 108, 122, 255),
    "glass": (52, 92, 118, 255),
    "wheel": (18, 20, 22, 255),
    "light": (214, 236, 255, 255),
    "stop_light": (230, 38, 32, 255),
    "skin": (214, 166, 132, 255),
    "clothes": (58, 112, 178, 255),
}


def vehicle_model_key_for_render(
    requested_model_key: str,
    source: str,
    primary: bool,
    cut_in: bool,
    available_model_keys: Collection[str],
    legacy_model_key: str = "cybertruck",
) -> str | None:
    model_key = requested_model_key or legacy_model_key
    if model_key not in available_model_keys:
        return None
    if requested_model_key:
        return model_key

    source_marker = source.startswith("modelV2") or source in ("radarState", "radarPoint", "cornerRadar")
    if source_marker or (source and not primary and not cut_in):
        return None
    return model_key


def visual_spec_for_object_class(value: object) -> VehicleVisualSpec | None:
    normalized = str(value or "").strip().lower().replace("-", "_")
    model_key = OBJECT_CLASS_ALIASES.get(normalized)
    return VEHICLE_VISUAL_SPECS.get(model_key) if model_key is not None else None


class _MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[float] = []
        self.normals: list[float] = []
        self.colors: list[int] = []

    @staticmethod
    def _normal(a: Point3, b: Point3, c: Point3) -> Point3:
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length <= 1e-8:
            return 0.0, 0.0, 1.0
        return nx / length, ny / length, nz / length

    def triangle(self, a: Point3, b: Point3, c: Point3, material: str) -> None:
        normal = self._normal(a, b, c)
        color = MATERIAL_COLORS[material]
        for point in (a, b, c):
            self.vertices.extend(point)
            self.normals.extend(normal)
            self.colors.extend(color)

    def quad(self, a: Point3, b: Point3, c: Point3, d: Point3, material: str) -> None:
        self.triangle(a, b, c, material)
        self.triangle(a, c, d, material)

    def hexahedron(self, bottom: tuple[Point3, Point3, Point3, Point3],
                   top: tuple[Point3, Point3, Point3, Point3], material: str) -> None:
        self.quad(bottom[0], bottom[3], bottom[2], bottom[1], material)
        self.quad(top[0], top[1], top[2], top[3], material)
        for index in range(4):
            nxt = (index + 1) % 4
            self.quad(bottom[index], bottom[nxt], top[nxt], top[index], material)

    def box(self, x0: float, x1: float, y0: float, y1: float,
            z0: float, z1: float, material: str) -> None:
        bottom = ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
        top = ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
        self.hexahedron(bottom, top, material)

    def tapered_box(self, bottom_x: float, bottom_y0: float, bottom_y1: float, bottom_z: float,
                    top_x: float, top_y0: float, top_y1: float, top_z: float, material: str) -> None:
        bottom = (
            (-bottom_x, bottom_y0, bottom_z),
            (bottom_x, bottom_y0, bottom_z),
            (bottom_x, bottom_y1, bottom_z),
            (-bottom_x, bottom_y1, bottom_z),
        )
        top = (
            (-top_x, top_y0, top_z),
            (top_x, top_y0, top_z),
            (top_x, top_y1, top_z),
            (-top_x, top_y1, top_z),
        )
        self.hexahedron(bottom, top, material)

    def wheel(self, center_x: float, center_y: float, center_z: float,
              radius: float, half_width: float, segments: int = 8) -> None:
        left_center = (center_x - half_width, center_y, center_z)
        right_center = (center_x + half_width, center_y, center_z)
        left: list[Point3] = []
        right: list[Point3] = []
        for index in range(segments):
            angle = math.tau * index / segments
            y = center_y + math.sin(angle) * radius
            z = center_z + math.cos(angle) * radius
            left.append((center_x - half_width, y, z))
            right.append((center_x + half_width, y, z))
        for index in range(segments):
            nxt = (index + 1) % segments
            self.quad(left[index], right[index], right[nxt], left[nxt], "wheel")
            self.triangle(left_center, left[nxt], left[index], "wheel")
            self.triangle(right_center, right[index], right[nxt], "wheel")

    def octahedron(self, center: Point3, radius: float, material: str) -> None:
        x, y, z = center
        top = (x, y, z + radius)
        bottom = (x, y, z - radius)
        ring = (
            (x + radius, y, z),
            (x, y + radius, z),
            (x - radius, y, z),
            (x, y - radius, z),
        )
        for index in range(4):
            nxt = (index + 1) % 4
            self.triangle(top, ring[index], ring[nxt], material)
            self.triangle(bottom, ring[nxt], ring[index], material)

    def data(self) -> VehicleMeshData:
        return VehicleMeshData(tuple(self.vertices), tuple(self.normals), tuple(self.colors))


def _car_mesh() -> VehicleMeshData:
    mesh = _MeshBuilder()
    mesh.box(-0.50, 0.50, -0.50, 0.50, 0.14, 0.42, "body")
    mesh.tapered_box(0.43, -0.23, 0.27, 0.42, 0.31, -0.11, 0.18, 0.83, "glass")
    mesh.box(-0.46, 0.46, 0.27, 0.50, 0.42, 0.54, "body")
    for x in (-0.49, 0.49):
        for y in (-0.32, 0.33):
            mesh.wheel(x, y, 0.18, 0.17, 0.055)
    mesh.box(-0.34, 0.34, 0.497, 0.505, 0.25, 0.35, "light")
    mesh.box(-0.34, 0.34, -0.505, -0.497, 0.25, 0.35, "stop_light")
    return mesh.data()


def _truck_mesh() -> VehicleMeshData:
    mesh = _MeshBuilder()
    mesh.box(-0.48, 0.48, -0.50, 0.50, 0.10, 0.24, "body_dark")
    mesh.box(-0.49, 0.49, -0.50, 0.13, 0.24, 0.92, "body")
    mesh.tapered_box(0.48, 0.16, 0.50, 0.24, 0.43, 0.20, 0.45, 0.76, "body")
    mesh.box(-0.40, 0.40, 0.43, 0.505, 0.46, 0.68, "glass")
    for x in (-0.49, 0.49):
        for y in (-0.38, -0.08, 0.38):
            mesh.wheel(x, y, 0.16, 0.15, 0.055)
    mesh.box(-0.34, 0.34, 0.497, 0.505, 0.25, 0.34, "light")
    return mesh.data()


def _bus_mesh() -> VehicleMeshData:
    mesh = _MeshBuilder()
    mesh.box(-0.50, 0.50, -0.50, 0.50, 0.10, 0.86, "body")
    mesh.tapered_box(0.50, -0.50, 0.50, 0.86, 0.45, -0.47, 0.47, 0.98, "body_dark")
    mesh.box(-0.505, -0.495, -0.38, 0.38, 0.48, 0.78, "glass")
    mesh.box(0.495, 0.505, -0.38, 0.38, 0.48, 0.78, "glass")
    mesh.box(-0.42, 0.42, 0.495, 0.505, 0.48, 0.78, "glass")
    for x in (-0.49, 0.49):
        for y in (-0.34, 0.34):
            mesh.wheel(x, y, 0.16, 0.15, 0.055)
    return mesh.data()


def _motorcycle_mesh() -> VehicleMeshData:
    mesh = _MeshBuilder()
    for y in (-0.27, 0.27):
        mesh.wheel(0.0, y, 0.23, 0.22, 0.13, 10)
    mesh.tapered_box(0.22, -0.22, 0.24, 0.28, 0.13, -0.10, 0.15, 0.54, "body")
    mesh.box(-0.12, 0.12, -0.10, 0.18, 0.48, 0.60, "body_dark")
    mesh.box(-0.32, 0.32, 0.25, 0.29, 0.56, 0.60, "body_dark")
    return mesh.data()


def _bicycle_mesh() -> VehicleMeshData:
    mesh = _MeshBuilder()
    for y in (-0.25, 0.25):
        mesh.wheel(0.0, y, 0.25, 0.24, 0.055, 12)
    mesh.box(-0.035, 0.035, -0.32, 0.30, 0.30, 0.36, "body")
    mesh.box(-0.045, 0.045, -0.05, 0.02, 0.33, 0.72, "body")
    mesh.box(-0.24, 0.24, 0.29, 0.34, 0.67, 0.72, "body_dark")
    mesh.box(-0.18, 0.18, -0.12, -0.07, 0.61, 0.66, "body_dark")
    return mesh.data()


def _person_mesh() -> VehicleMeshData:
    mesh = _MeshBuilder()
    mesh.box(-0.20, -0.04, -0.18, 0.18, 0.00, 0.47, "clothes")
    mesh.box(0.04, 0.20, -0.18, 0.18, 0.00, 0.47, "clothes")
    mesh.tapered_box(0.29, -0.20, 0.20, 0.45, 0.20, -0.15, 0.15, 0.82, "clothes")
    mesh.box(-0.39, -0.25, -0.13, 0.13, 0.45, 0.76, "skin")
    mesh.box(0.25, 0.39, -0.13, 0.13, 0.45, 0.76, "skin")
    mesh.octahedron((0.0, 0.0, 0.91), 0.13, "skin")
    return mesh.data()


_MESH_BUILDERS = {
    "car": _car_mesh,
    "truck": _truck_mesh,
    "bus": _bus_mesh,
    "motorcycle": _motorcycle_mesh,
    "bicycle": _bicycle_mesh,
    "person": _person_mesh,
}


def build_vehicle_mesh_data(model_key: str) -> VehicleMeshData:
    try:
        builder = _MESH_BUILDERS[model_key]
    except KeyError as exc:
        raise ValueError(f"unknown vehicle visual model: {model_key}") from exc
    return builder()
