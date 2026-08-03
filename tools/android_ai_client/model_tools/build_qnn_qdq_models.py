#!/usr/bin/env python3
"""Build fixed-shape QDQ YOLO models for ONNX Runtime QNN/HTP.

The generated model keeps float32 graph I/O so the Android client can share its
existing preprocessing and YOLO output parser with CPU compatibility models.
Weights are QUInt8 and activations are QUInt16 inside the QDQ graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from onnxruntime.tools.make_dynamic_shape_fixed import fix_output_shapes, make_input_shape_fixed
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def image_tensor(path: Path, size: int) -> np.ndarray:
  with Image.open(path) as source:
    image = source.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

  canvas = np.full((size, size, 3), 114, dtype=np.uint8)
  left = (size - resized_width) // 2
  top = (size - resized_height) // 2
  canvas[top:top + resized_height, left:left + resized_width] = np.asarray(resized)
  return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


class ImageCalibrationReader(CalibrationDataReader):
  def __init__(self, input_name: str, image_paths: list[Path], size: int):
    self.input_name = input_name
    self.image_paths = image_paths
    self.size = size
    self._iterator = None

  def get_next(self) -> dict[str, np.ndarray] | None:
    if self._iterator is None:
      self._iterator = iter(self.image_paths)
    try:
      image_path = next(self._iterator)
    except StopIteration:
      return None
    return {self.input_name: image_tensor(image_path, self.size)}

  def rewind(self) -> None:
    self._iterator = None


def collect_calibration_images(root: Path, sample_count: int) -> list[Path]:
  images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
  if len(images) < sample_count:
    raise ValueError(f"calibration images: requested {sample_count}, found {len(images)} under {root}")
  if sample_count == len(images):
    return images
  indexes = np.linspace(0, len(images) - 1, sample_count, dtype=int)
  return [images[index] for index in indexes]


def make_static_model(source: Path, output: Path, size: int) -> str:
  model = onnx.load(source)
  if len(model.graph.input) != 1:
    raise ValueError(f"expected one model input, found {len(model.graph.input)}")
  # ORT 1.24.3's QNN preprocessor cannot merge ModelProto metadata entries.
  # The Ultralytics metadata is informational and is not used by the Android parser.
  model.ClearField("metadata_props")
  input_name = model.graph.input[0].name
  make_input_shape_fixed(model.graph, input_name, [1, 3, size, size])
  fix_output_shapes(model)
  if len(model.graph.output) != 1:
    raise ValueError(f"expected one model output, found {len(model.graph.output)}")
  output_dims = model.graph.output[0].type.tensor_type.shape.dim
  if len(output_dims) != 3 or output_dims[1].dim_value != 84:
    raise ValueError("expected a YOLO detection output shaped [1, 84, anchors]")
  anchor_count = sum((size // stride) ** 2 for stride in (8, 16, 32))
  for dim, value in zip(output_dims, (1, 84, anchor_count), strict=True):
    dim.ClearField("dim_param")
    dim.dim_value = value
  model = onnxslim.slim(model)
  if model is None:
    raise RuntimeError("onnxslim did not return a model")
  onnx.checker.check_model(model)
  onnx.save_model(model, output)
  return input_name


def validate_model(static_model: Path, quantized_model: Path, image_path: Path, size: int) -> dict[str, object]:
  quantized_graph = onnx.load(quantized_model)
  onnx.checker.check_model(quantized_graph)
  operator_types = sorted({node.op_type for node in quantized_graph.graph.node})
  forbidden_dynamic_ops = sorted({"ConstantOfShape", "Range", "Shape"}.intersection(operator_types))
  if forbidden_dynamic_ops:
    raise ValueError(f"dynamic anchor operators remain after simplification: {forbidden_dynamic_ops}")
  providers = ["CPUExecutionProvider"]
  static_session = ort.InferenceSession(str(static_model), providers=providers)
  quantized_session = ort.InferenceSession(str(quantized_model), providers=providers)
  static_input = static_session.get_inputs()[0]
  quantized_input = quantized_session.get_inputs()[0]
  expected_shape = [1, 3, size, size]
  if static_input.shape != expected_shape or quantized_input.shape != expected_shape:
    raise ValueError(
      f"unexpected input shapes: static={static_input.shape}, quantized={quantized_input.shape}, expected={expected_shape}"
    )

  tensor = image_tensor(image_path, size)
  float_outputs = static_session.run(None, {static_input.name: tensor})
  quantized_outputs = quantized_session.run(None, {quantized_input.name: tensor})
  if len(float_outputs) != len(quantized_outputs):
    raise ValueError("quantized model output count changed")

  comparisons = []
  for float_output, quantized_output in zip(float_outputs, quantized_outputs, strict=True):
    if float_output.shape != quantized_output.shape:
      raise ValueError(f"quantized model output shape changed: {float_output.shape} != {quantized_output.shape}")
    if not np.all(np.isfinite(quantized_output)):
      raise ValueError("quantized model produced non-finite values")
    denominator = max(float(np.sqrt(np.mean(np.square(float_output)))), 1e-8)
    comparisons.append({
      "shape": list(quantized_output.shape),
      "normalized_rmse": float(np.sqrt(np.mean(np.square(float_output - quantized_output))) / denominator),
    })
  return {"input_shape": expected_shape, "operator_types": operator_types, "outputs": comparisons}


def build_model(source: Path, calibration_images: list[Path], output_dir: Path, size: int) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=True)
  static_model = output_dir / f"yolo11n-static-{size}-fp32.onnx"
  preprocessed_model = output_dir / f"yolo11n-static-{size}-qnn-preprocessed.onnx"
  quantized_model = output_dir / f"yolo11n-static-{size}-w8a16-qdq.onnx"

  input_name = make_static_model(source, static_model, size)
  changed = qnn_preprocess_model(static_model, preprocessed_model)
  model_to_quantize = preprocessed_model if changed else static_model
  reader = ImageCalibrationReader(input_name, calibration_images, size)
  config = get_qnn_qdq_config(
    model_to_quantize,
    reader,
    activation_type=QuantType.QUInt16,
    weight_type=QuantType.QUInt8,
    per_channel=False,
    calibration_providers=["CPUExecutionProvider"],
  )
  quantize(model_to_quantize, quantized_model, config)
  validation = validate_model(model_to_quantize, quantized_model, calibration_images[0], size)

  static_model.unlink(missing_ok=True)
  preprocessed_model.unlink(missing_ok=True)
  return {
    "file": quantized_model.name,
    "size": quantized_model.stat().st_size,
    "sha256": sha256(quantized_model),
    "input_size": size,
    "activation_type": "QUInt16",
    "weight_type": "QUInt8",
    "calibration_images": len(calibration_images),
    "validation": validation,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True, help="Ultralytics YOLO11n dynamic FP32 ONNX")
  parser.add_argument("--calibration-dir", type=Path, required=True, help="Directory containing representative images")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--sizes", type=int, nargs="+", default=[320])
  parser.add_argument("--samples", type=int, default=64)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.source.is_file():
    raise FileNotFoundError(args.source)
  if not args.calibration_dir.is_dir():
    raise NotADirectoryError(args.calibration_dir)
  if any(size not in {320, 416, 640} for size in args.sizes):
    raise ValueError("supported sizes are 320, 416, and 640")
  if args.samples < 8:
    raise ValueError("at least 8 calibration images are required")

  calibration_images = collect_calibration_images(args.calibration_dir, args.samples)
  manifest = {
    "format": 1,
    "source": args.source.name,
    "source_sha256": sha256(args.source),
    "models": [build_model(args.source, calibration_images, args.output_dir, size) for size in args.sizes],
  }
  manifest_path = args.output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
  main()
