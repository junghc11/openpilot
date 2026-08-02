# Carrot External AI Android client

This experimental app receives 640×360 JPEG road frames from `phoneaid` on a C3, C3X, or C4, runs a COCO YOLO ONNX model on the phone, and returns only normalized object detections. Results are visualization-only and are not delivered to vehicle control, CAN, Panda, radar, or the safety model. C3X is the current hardware validation target; C3 and C4 still need device-specific road-stream and UI tests.

## Supported environment and model

- A 64-bit ARM (`arm64-v8a`) phone with Android 10 (API 29) or newer
- ONNX Runtime Android with NNAPI first and automatic CPU fallback
- Float32 NCHW input shaped `[1, 3, H, W]`; dynamic H/W defaults to 640
- Standard Ultralytics YOLOv8/YOLO11 output shaped `[1, 84, N]` or `[1, N, 84]`
- COCO person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign classes

Exports that perform NMS inside the model and return `[1, N, 6]` are not supported yet. The NNAPI session enables FP16 and NCHW and disables NNAPI CPU. Supported graph partitions may run on the phone's NPU, DSP, or GPU. If the accelerated session cannot be created, the app recreates the whole session on CPU. A direct Qualcomm QNN backend is not bundled yet.

No YOLO model is bundled in the APK. **The recommended first-test model is YOLO11n Detection running at 640, FP32 ONNX, with no embedded NMS.** Its Nano size minimizes phone load and its conventional COCO output matches the current parser. Both static `[1,3,640,640]` and dynamic `[-1,3,-1,-1]` inputs are accepted; the app executes at 640×640. YOLO11s/m/l/x, YOLOv8, or a compatible custom model may also work but require separate performance and output validation. YOLO26 end-to-end, segmentation, pose, classification, and OBB models are not currently supported.

Press **권장 모델 다운로드** and review the Ultralytics model-license notice. The app downloads `yolo11n.onnx` directly from the official `ultralytics/assets` v8.4.0 Release. It pins the `10,930,182`-byte size and SHA-256 `634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b`, then verifies an FP32 input compatible with a 640 execution and a conventional COCO YOLO output before moving the model into internal app storage. The official file metadata is dynamic `[-1,3,-1,-1] → [-1,84,-1]`; an app inference resolves it to `[1,3,640,640] → [1,84,N]`. A failed or cancelled download removes its temporary file, and only a successful download becomes active. Uninstalling the app removes the internal model.

Use **다른 ONNX 파일 선택** when offline or when testing another compatible model. To create the recommended format on a PC, retain `nms=False` and `dynamic=False`:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True nms=False dynamic=False batch=1
```

The official model is subject to Ultralytics AGPL-3.0 or Enterprise terms. Review the license link in the confirmation dialog and use the model within the appropriate terms.

## Build and install

Android SDK 35 and JDK 17 or newer are required.

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug :app:lintDebug
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

The app requests notification permission on Android 13 or newer. Android 12 and newer restrict arbitrary foreground-service launches from the background, so open the app once after a reboot. While active, a persistent notification provides status and a **중지** action.

The first recommended-model download requires internet access and about 10.4MB of transfer. Keep the app open until it finishes. The verified model is reused from internal storage on later launches. **권장 모델 다시 다운로드** revalidates the pinned version, while **권장 모델 삭제** removes the internal file.

The current app targets SDK 35, so its `INTERNET` permission provides local TCP/UDP access. If the target SDK is raised to Android 17/API 37 or newer, add the `ACCESS_LOCAL_NETWORK` runtime-permission flow at the same time.

## Automatic C3/C3X/C4 pairing

You do not need to find the device address manually when a phone hotspot assigns a different C3X address.

1. Connect the phone and C3/C3X/C4 to the same trusted Wi-Fi or phone hotspot. Disable AP/client isolation.
2. Set `ExternalAIEnabled=1` in Carrot Web. Object display also requires `ExternalAIShowOverlay=1`.
3. On a phone hotspot, set `ExternalAIPhoneIP` to the phone's hotspot gateway address. It may be left empty while testing if the address is unknown, but only on a trusted dedicated network.
4. Put the C3/C3X/C4 on-road. `phoneaid` opens TCP frame port `7724` only while on-road.
5. Press **권장 모델 다운로드** or use **다른 ONNX 파일 선택** for another compatible model. The recommended path automatically verifies integrity and ONNX input/output shapes.
6. Leave **앱 실행 시 같은 망 자동 검색 및 시작** enabled. The app checks the saved address first, then probes only TCP `7724` in the local private IPv4 `/24` and accepts only a server whose first four bytes are the Carrot frame signature `CAI1`.
7. When found, the app saves the device address and starts YOLO automatically. Confirm **연결됨** in the app and a green `eNPU` or blue `eCPU` badge on the C3X.

Discovery is bounded to at most two local private `/24` networks. It does not scan the internet or a range of ports. If the frame port is changed, enter the same port in the app; discovery then verifies `CAI1` on that port. Discovery may fail on a VPN, guest Wi-Fi, or a network with AP isolation.

For a manual fallback, enter the current C3/C3X/C4 address under **기기 IP** and press **입력 IP로 시작**. When automatic connection is enabled and a model has been saved, opening the app starts discovery without another button press. It does not start at boot; reopen it after a phone reboot or force-stop.

## Screen-off and power behavior

- Before a device is found, the app does not open the YOLO model or NPU session and does not hold a wake lock or high-performance Wi-Fi lock. Its foreground-service discovery wait increases through 5, 10, and 20 seconds to a 30-second cap. Android or OEM power policy may defer discovery further while the screen is off.
- A recommended-model download runs only after a user presses the button and does not acquire inference wake locks. Performance locks begin only after model installation and device connection.
- After TCP video connects, a partial wake lock and high-performance Wi-Fi lock keep reception and inference running with the screen off. This is a performance mode, so charging and thermal monitoring are recommended.
- A disconnect immediately releases both performance locks. Failed connections retry after 1, 2, 4, 8, and 16 seconds, capped at 30 seconds; automatic mode scans again if the address changed.
- Press **중지** in the app for zero background use; this disables automatic connection and ends discovery, inference, networking, and locks. The notification action also ends the current service but preserves the automatic-connection preference, so opening the app later can start it again.
- The connected partial wake lock has a six-hour safety timeout. Restart the client before an unusually long continuous session.

## Protocol and security

TCP frames use a 12-byte network-order prefix containing ASCII `CAI1`, JSON-header length, and JPEG length. The JSON header includes protocol version, frame ID, the C3X monotonic source timestamp, dimensions, and `jpeg` encoding. The phone returns the frame ID and source timestamp in the UDP JSON format defined by `openpilot/selfdrive/carrot/external_ai/protocol.py`.

TCP/UDP traffic is not encrypted, and discovery is not authentication. Do not expose these ports on a public or untrusted network.

The recommended model comes directly from `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx` over HTTPS and is installed only after both the pinned SHA-256 and ONNX structure match. Integrity verification does not replace the model license.
