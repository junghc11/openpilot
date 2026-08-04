#!/usr/bin/env python3
"""Build fixed-shape QDQ YOLO models for ONNX Runtime QNN/HTP.

The generated model keeps float32 graph I/O so the Android client can share its
existing preprocessing and YOLO output parser with CPU compatibility models.
Weights are QUInt8 and activations default to QUInt16. YOLO11's
activation-to-activation attention MatMul receives a targeted QUInt8 conversion
on one input so the pair remains supported by the QNN HTP backend without
reducing the precision of the detection head.
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

from extract_qnn_raw_head_models import extract_model as extract_raw_head_model


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


def decode_raw_head(raw_output: np.ndarray, size: int) -> np.ndarray:
  if raw_output.shape != (1, 144, sum((size // stride) ** 2 for stride in (8, 16, 32))):
    raise ValueError(f"unexpected raw-head shape: {raw_output.shape}")
  distributions = raw_output[:, :64].reshape(1, 4, 16, -1)
  distributions -= distributions.max(axis=2, keepdims=True)
  probabilities = np.exp(distributions)
  probabilities /= probabilities.sum(axis=2, keepdims=True)
  distances = (probabilities * np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1)).sum(axis=2)
  class_scores = 1.0 / (1.0 + np.exp(-np.clip(raw_output[:, 64:], -80.0, 80.0)))

  anchors = []
  strides = []
  for stride in (8, 16, 32):
    grid = size // stride
    grid_y, grid_x = np.meshgrid(
      np.arange(grid, dtype=np.float32) + 0.5,
      np.arange(grid, dtype=np.float32) + 0.5,
      indexing="ij",
    )
    anchors.append(np.stack((grid_x.ravel(), grid_y.ravel())))
    strides.append(np.full((1, grid * grid), stride, dtype=np.float32))
  anchor_points = np.concatenate(anchors, axis=1)[None]
  stride_values = np.concatenate(strides, axis=1)[None]
  top_left = (anchor_points - distances[:, :2]) * stride_values
  bottom_right = (anchor_points + distances[:, 2:]) * stride_values
  boxes = np.concatenate(((top_left + bottom_right) / 2.0, bottom_right - top_left), axis=1)
  return np.concatenate((boxes, class_scores), axis=1)


def validate_raw_head(decoded_model: Path, raw_model: Path, image_path: Path, size: int) -> dict[str, float]:
  providers = ["CPUExecutionProvider"]
  decoded_session = ort.InferenceSession(str(decoded_model), providers=providers)
  raw_session = ort.InferenceSession(str(raw_model), providers=providers)
  tensor = image_tensor(image_path, size)
  decoded = decoded_session.run(None, {decoded_session.get_inputs()[0].name: tensor})[0]
  raw = raw_session.run(None, {raw_session.get_inputs()[0].name: tensor})[0]
  rebuilt = decode_raw_head(raw, size)
  delta = decoded - rebuilt
  max_abs_error = float(np.max(np.abs(delta)))
  rmse = float(np.sqrt(np.mean(np.square(delta))))
  if max_abs_error >= 0.1 or rmse >= 0.02:
    raise ValueError(f"raw-head decode mismatch: max={max_abs_error}, rmse={rmse}")
  return {"max_abs_error": max_abs_error, "rmse": rmse}


def attention_matmul_overrides(model_path: Path, activation_type: QuantType) -> tuple[dict, list[str]]:
  """Use a supported HTP type pair for activation-to-activation MatMul nodes.

  QNN HTP supports UINT8xUINT8, UINT8xUINT16, and UINT16xUINT8 MatMul inputs,
  but not UINT16xUINT16. YOLO11n's C2PSA attention has two MatMul nodes whose
  inputs are both activations, so convert the first input of each node to UINT8.
  """
  if activation_type != QuantType.QUInt16:
    return {}, []
  model = onnx.load(model_path)
  initializer_names = {initializer.name for initializer in model.graph.initializer}
  overrides = {}
  node_names = []
  for node in model.graph.node:
    if node.op_type != "MatMul" or len(node.input) != 2:
      continue
    if any(name in initializer_names for name in node.input):
      continue
    overrides[node.input[0]] = [{"quant_type": QuantType.QUInt8}]
    node_names.append(node.name)
  if not node_names:
    raise ValueError("expected at least one activation-to-activation MatMul for HTP mixed precision")
  return overrides, node_names


def validate_htp_matmul_types(model_path: Path) -> list[dict[str, object]]:
  model = onnx.load(model_path)
  producers = {output: node for node in model.graph.node for output in node.output}
  initializers = {initializer.name: initializer for initializer in model.graph.initializer}
  allowed_pairs = {("UINT8", "UINT8"), ("UINT8", "UINT16"), ("UINT16", "UINT8")}
  results = []
  for node in model.graph.node:
    if node.op_type != "MatMul":
      continue
    input_types = []
    for input_name in node.input:
      dq_node = producers.get(input_name)
      if dq_node is None or dq_node.op_type != "DequantizeLinear" or len(dq_node.input) < 3:
        input_types.append("FLOAT")
        continue
      zero_point = initializers.get(dq_node.input[2])
      input_types.append(onnx.TensorProto.DataType.Name(zero_point.data_type) if zero_point else "UNKNOWN")
    pair = tuple(input_types)
    if pair not in allowed_pairs:
      raise ValueError(f"HTP-unsupported MatMul input types at {node.name}: {pair}")
    results.append({"node": node.name, "input_types": input_types})
  if not results:
    raise ValueError("expected quantized MatMul nodes in YOLO11n HTP model")
  return results


def build_model(
    source: Path,
    calibration_images: list[Path],
    output_dir: Path,
    size: int,
    activation_type: QuantType,
) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=True)
  activation_bits = 8 if activation_type == QuantType.QUInt8 else 16
  profile_suffix = "htp-mixed" if activation_type == QuantType.QUInt16 else ""
  quantized_stem = f"yolo11n-static-{size}-w8a{activation_bits}{f'-{profile_suffix}' if profile_suffix else ''}"
  static_model = output_dir / f"yolo11n-static-{size}-fp32.onnx"
  preprocessed_model = output_dir / f"yolo11n-static-{size}-qnn-preprocessed.onnx"
  decoded_quantized_model = output_dir / f"{quantized_stem}-decoded-qdq.onnx"
  quantized_model = output_dir / f"{quantized_stem}-raw-head-qdq.onnx"

  input_name = make_static_model(source, static_model, size)
  changed = qnn_preprocess_model(static_model, preprocessed_model)
  model_to_quantize = preprocessed_model if changed else static_model
  reader = ImageCalibrationReader(input_name, calibration_images, size)
  init_overrides, mixed_precision_nodes = attention_matmul_overrides(model_to_quantize, activation_type)
  config = get_qnn_qdq_config(
    model_to_quantize,
    reader,
    activation_type=activation_type,
    weight_type=QuantType.QUInt8,
    per_channel=False,
    init_overrides=init_overrides,
    calibration_providers=["CPUExecutionProvider"],
  )
  quantize(model_to_quantize, decoded_quantized_model, config)
  validation = validate_model(model_to_quantize, decoded_quantized_model, calibration_images[0], size)
  raw_head = extract_raw_head_model(decoded_quantized_model, quantized_model)
  raw_head_validation = validate_raw_head(decoded_quantized_model, quantized_model, calibration_images[0], size)
  htp_matmul_validation = validate_htp_matmul_types(quantized_model)

  static_model.unlink(missing_ok=True)
  preprocessed_model.unlink(missing_ok=True)
  decoded_quantized_model.unlink(missing_ok=True)
  return {
    "file": quantized_model.name,
    "size": quantized_model.stat().st_size,
    "sha256": sha256(quantized_model),
    "input_size": size,
    "activation_type": activation_type.name,
    "weight_type": "QUInt8",
    "mixed_precision_matmul_nodes": mixed_precision_nodes,
    "htp_matmul_validation": htp_matmul_validation,
    "calibration_images": len(calibration_images),
    "validation": validation,
    "raw_head": {
      "output_name": raw_head["output_name"],
      "output_shape": raw_head["output_shape"],
      "removed_nodes": raw_head["removed_nodes"],
      **raw_head_validation,
    },
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True, help="Ultralytics YOLO11n dynamic FP32 ONNX")
  parser.add_argument("--calibration-dir", type=Path, required=True, help="Directory containing representative images")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--sizes", type=int, nargs="+", default=[320])
  parser.add_argument("--samples", type=int, default=64)
  parser.add_argument("--activation-bits", type=int, choices=(8, 16), default=16)
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
  activation_type = QuantType.QUInt8 if args.activation_bits == 8 else QuantType.QUInt16
  manifest = {
    "format": 1,
    "source": args.source.name,
    "source_sha256": sha256(args.source),
    "models": [
      build_model(args.source, calibration_images, args.output_dir, size, activation_type)
      for size in args.sizes
    ],
  }
  manifest_path = args.output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
  main()
