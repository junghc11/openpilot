# Carrot External AI QNN models

The two ONNX files in this directory are fixed-shape raw-head QDQ derivatives
of the official Ultralytics `yolo11n.onnx` v8.4.0 asset. They are intended for
ONNX Runtime QNN with the Qualcomm HTP backend.

- `yolo11n-static-320-w8a16-raw-head-qdq.onnx`: latency-first model, fixed NCHW
  input `[1, 3, 320, 320]`, float32 graph I/O, QUInt16 activations, QUInt8
  weights, and raw output `[1, 144, 2100]`.
- `yolo11n-static-640-w8a16-raw-head-qdq.onnx`: detail-first model, fixed NCHW
  input `[1, 3, 640, 640]`, float32 graph I/O, QUInt16 activations, QUInt8
  weights, and raw output `[1, 144, 8400]`.

The raw output contains 64 DFL box-distribution channels and 80 COCO class
logits. The 57 extracted nodes cover DFL, anchor generation, sigmoid, and box
decoding; the Android client executes them and the already client-side NMS on
CPU. This gives QNN/HTP a smaller convolution-focused graph and avoids forcing
detection decoding through the NPU.

Both models were calibrated with the 128 images in Ultralytics COCO128. The
build script removes dynamic anchor-generation nodes before QDQ quantization,
extracts the raw detection head, and verifies that CPU-side reconstruction stays
close to the decoded QDQ output. `manifest.json` pins the source and output
SHA-256 hashes, file sizes, tensor shapes, calibration count, and CPU validation
results.

## Rebuild

Create an isolated Python environment and install
`model_tools/requirements-qnn.txt`. Download the pinned source model and a
representative calibration image directory, then run:

```powershell
python model_tools/build_qnn_qdq_models.py `
  --source yolo11n.onnx `
  --calibration-dir coco128 `
  --output-dir models `
  --sizes 320 640 `
  --samples 128
```

The source weights and derived model files remain subject to the Ultralytics
AGPL-3.0 or Enterprise license. CPU ONNX validation does not prove physical NPU
placement; the Android client reports `eNPU` only after a full-graph QNN/HTP
session with CPU fallback disabled completes a warm-up inference.
