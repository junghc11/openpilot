# Carrot External AI Android client

This experimental Android app connects to `phoneaid` on a C3X, receives 640×360 JPEG road frames, runs a COCO YOLO ONNX model, and returns only normalized object detections. Results are visualization-only. The app contains no vehicle-control, CAN, Panda, radar, or safety integration.

## Supported first-stage configuration

- 64-bit ARM phone (`arm64-v8a`) with Android 10 (API 29) or newer
- ONNX Runtime Android CPU backend
- Float32 NCHW model input shaped `[1, 3, H, W]` (dynamic H/W defaults to 640)
- Standard Ultralytics YOLOv8/YOLO11 detection output shaped `[1, 84, N]` or `[1, N, 84]`
- COCO classes: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
- Class-aware non-maximum suppression at IoU 0.45 and at most 64 returned objects

End-to-end ONNX exports that already perform NMS and return `[1, N, 6]` are not supported yet. QNN/NPU and GPU execution providers are intentionally deferred until the CPU path has been measured on the actual phone.

No model is bundled. Select a compatible `.onnx` model from the app. A typical Ultralytics export is:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True
```

Review the model and framework licenses before redistribution.

## Build and install

Open this directory in Android Studio, or build from a terminal with Android SDK 35 and JDK 17 or newer:

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug :app:lintDebug
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

The app requests notification permission on Android 13 or newer. Start the client from its visible activity; Android restricts launching foreground services from the background. While running, a persistent notification, partial wake lock, and high-performance Wi-Fi lock keep reception and inference active with the screen off. The wake lock has a six-hour safety timeout, so restart the client for an unusually long continuous session.

## C3X setup

1. Put the phone and C3X on a trusted dedicated Wi-Fi network.
2. Set `ExternalAIEnabled=1` and `ExternalAIShowOverlay=1` on the C3X.
3. Set `ExternalAIPhoneIP` to the phone address. The empty value is useful only for initial diagnosis.
4. Keep TCP frame port `7724` and UDP result port `7725` unless both ends are changed together.
5. Enter the C3X address in the app, select the ONNX model, and press **시작**.

The status panel reports connection state, receive FPS, inference FPS, average inference time, object count, CPU backend, battery temperature, and Android thermal status. End-to-end round-trip age is calculated on the C3X because the phone and C3X monotonic clocks have different origins.

## Protocol

TCP frames use a 12-byte network-order prefix: ASCII magic `CAI1`, JSON-header byte count, and JPEG byte count. The JSON header contains protocol version, frame ID, C3X monotonic source timestamp, width, height, and `jpeg` encoding. The phone echoes the frame ID and source timestamp in the UDP result JSON defined by `openpilot/selfdrive/carrot/external_ai/protocol.py`.

The TCP/UDP path is not encrypted. Source-IP restriction is only a PoC safeguard, not authentication. Do not expose these ports to an untrusted network.
