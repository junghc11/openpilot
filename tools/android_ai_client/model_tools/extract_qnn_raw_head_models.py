#!/usr/bin/env python3
"""Remove YOLO11 DFL/anchor decoding from a QDQ model.

The resulting model returns the raw 144-channel detection head
(`4 * 16` box-distribution channels plus 80 COCO class logits). Android performs
the small DFL/anchor decode on CPU, leaving the convolutional graph available
for full QNN/HTP placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx.utils import Extractor


RAW_HEAD_CHANNELS = 144


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def tensor_shapes(model: onnx.ModelProto) -> dict[str, tuple[int | str, ...]]:
  shapes: dict[str, tuple[int | str, ...]] = {}
  for value in (*model.graph.value_info, *model.graph.input, *model.graph.output):
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
      dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
    shapes[value.name] = tuple(dims)
  return shapes


def find_raw_head(model: onnx.ModelProto) -> str:
  shapes = tensor_shapes(model)
  consumers: dict[str, list[onnx.NodeProto]] = {}
  for node in model.graph.node:
    for name in node.input:
      consumers.setdefault(name, []).append(node)

  candidates = []
  for node in model.graph.node:
    if node.op_type != "DequantizeLinear" or len(node.output) != 1:
      continue
    output = node.output[0]
    shape = shapes.get(output, ())
    if len(shape) != 3 or shape[0] != 1 or shape[1] != RAW_HEAD_CHANNELS:
      continue
    if any(consumer.op_type == "Split" for consumer in consumers.get(output, ())):
      candidates.append(output)
  if len(candidates) != 1:
    raise ValueError(f"expected one 144-channel DQ tensor consumed by Split, found {candidates}")
  return candidates[0]


def extract_model(source: Path, output: Path) -> dict[str, object]:
  inferred = onnx.shape_inference.infer_shapes(onnx.load(source))
  raw_head = find_raw_head(inferred)
  inputs = [value.name for value in inferred.graph.input]
  extracted = Extractor(inferred).extract_model(inputs, [raw_head])
  extracted.metadata_props.add(key="carrotpilot.output_format", value="yolo11_raw_dfl")
  extracted.metadata_props.add(key="carrotpilot.dfl_bins", value="16")
  onnx.checker.check_model(extracted)
  output.parent.mkdir(parents=True, exist_ok=True)
  onnx.save_model(extracted, output)
  shape = tensor_shapes(onnx.shape_inference.infer_shapes(extracted))[raw_head]
  return {
    "source": source.name,
    "file": output.name,
    "size": output.stat().st_size,
    "sha256": sha256(output),
    "output_name": raw_head,
    "output_shape": list(shape),
    "removed_nodes": len(inferred.graph.node) - len(extracted.graph.node),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("models", type=Path, nargs="+")
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  results = []
  for source in args.models:
    if not source.is_file():
      raise FileNotFoundError(source)
    output = args.output_dir / source.name.replace("-qdq.onnx", "-raw-head-qdq.onnx")
    results.append(extract_model(source, output))
  print(json.dumps({"format": 1, "models": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
