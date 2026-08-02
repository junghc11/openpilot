# Carrot External AI Android client

This experimental Android app connects to `phoneaid` on a C3, C3X, or C4 running this CarrotPilot branch, receives 640×360 JPEG road frames, runs a COCO YOLO ONNX model, and returns only normalized object detections. Results are visualization-only. The app contains no vehicle-control, CAN, Panda, radar, or safety integration. The process has no hardware-model gate, but the current validation target is C3X; C3 and C4 still require an on-device road-stream and UI test.

## Supported first-stage configuration

- 64-bit ARM phone (`arm64-v8a`) with Android 10 (API 29) or newer
- ONNX Runtime Android NNAPI-first backend with automatic CPU fallback
- Float32 NCHW model input shaped `[1, 3, H, W]` (dynamic H/W defaults to 640)
- Standard Ultralytics YOLOv8/YOLO11 detection output shaped `[1, 84, N]` or `[1, N, 84]`
- COCO classes: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
- Class-aware non-maximum suppression at IoU 0.45 and at most 64 returned objects

End-to-end ONNX exports that already perform NMS and return `[1, N, 6]` are not supported yet. The app registers the Android NNAPI execution provider first with FP16 and NCHW enabled and NNAPI CPU disabled. Supported graph partitions are therefore offered to an available NPU, DSP, or GPU, while unsupported operations can remain on the ONNX Runtime CPU provider. If the accelerated session cannot be created, the app automatically recreates the whole session on CPU and reports that fallback. A direct Qualcomm QNN backend is not bundled yet.

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

## Pairing with C3, C3X, or C4

Being on the same Wi-Fi network is necessary but does not start or discover the connection automatically. The app has no mDNS/discovery, boot start, or automatic foreground-service start.

1. Install this CarrotPilot branch on the device and put it and the phone on the same trusted Wi-Fi or phone hotspot. Disable client/AP isolation.
2. Find both addresses: the app needs the CarrotPilot device address, while `ExternalAIPhoneIP` on the device must contain the phone address. Do not assume the example `192.168.0.10` is correct.
3. In Carrot Web, enable `ExternalAIEnabled=1`. Enable `ExternalAIShowOverlay=1` to show detections; the compute badge remains visible even when object overlays are hidden.
4. Keep TCP frame port `7724` and UDP result port `7725` unless both ends are changed together.
5. Put the CarrotPilot device on-road. `phoneaid` runs only while the manager reports `started`; an off-road device does not open the frame server.
6. Open the Android app, enter the CarrotPilot device address, select a compatible YOLO ONNX model, and press **시작**.
7. Confirm **연결됨** in the app. The device shows green `eNPU` for NNAPI/QNN or blue `eCPU` for the ONNX Runtime CPU backend. CPU fallback still performs YOLO, but usually with lower throughput and higher battery use.

Settings and the selected model URI persist, but the service is `START_NOT_STICKY`: after a phone reboot, an Android process stop, or pressing **중지**, open the app and press **시작** again. The status panel reports connection state, receive FPS, inference FPS, average inference time, object count, selected backend, battery temperature, and Android thermal status. The same backend identifier is returned with every result. End-to-end round-trip age is calculated on the CarrotPilot device because its monotonic clock and the phone clock have different origins.

## Screen-off and power behavior

- Before **시작**, and after the app or notification **중지** action, no inference worker, wake lock, high-performance Wi-Fi lock, or reconnect loop remains. Android can place the app in its normal idle state.
- While TCP video is connected, the foreground service holds a partial wake lock and high-performance Wi-Fi lock so reception and inference continue with the screen off. This is a performance mode, not a battery-saving mode; charging and thermal monitoring are recommended.
- If the CarrotPilot device is off-road, `ExternalAIEnabled` is disabled, Wi-Fi disappears, or TCP disconnects, the app releases both performance locks and retries with a 1, 2, 4, 8, 16, then 30-second capped backoff. The low-priority foreground notification remains. Android or an OEM may defer retries while the phone sleeps because the app does not request a battery-optimization exemption.
- For zero background use, press **중지** instead of leaving the client in reconnect wait. The service is not registered to start at boot.
- The connected partial wake lock has a six-hour safety timeout. Restart the client for an unusually long continuous session.

## Protocol

TCP frames use a 12-byte network-order prefix: ASCII magic `CAI1`, JSON-header byte count, and JPEG byte count. The JSON header contains protocol version, frame ID, C3X monotonic source timestamp, width, height, and `jpeg` encoding. The phone echoes the frame ID and source timestamp in the UDP result JSON defined by `openpilot/selfdrive/carrot/external_ai/protocol.py`.

The TCP/UDP path is not encrypted. Source-IP restriction is only a PoC safeguard, not authentication. Do not expose these ports to an untrusted network.
