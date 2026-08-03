# Carrot External AI Android client

This experimental app receives the default 854×480 hardware H.264 stream or compatibility 640×360 JPEG road frames from `phoneaid` on a C3, C3X, or C4, runs a COCO YOLO ONNX model on the phone, and returns only normalized object detections. H.264 is hardware-decoded through Android `MediaCodec`; the connection automatically falls back to JPEG when it is unavailable. Results are visualization-only and are not delivered to vehicle control, CAN, Panda, radar, or the safety model. C3X is the current hardware validation target; C3 and C4 still need device-specific road-stream and UI tests.

## Supported environment and model

- A 64-bit ARM (`arm64-v8a`) phone with Android 10 (API 29) or newer
- Full-graph Qualcomm QNN/HTP first, then automatic NNAPI and CPU fallbacks
- Float32 NCHW input shaped `[1, 3, H, W]`; dynamic H/W selects 320, 416, or 640 and defaults to 320
- Standard Ultralytics YOLOv8/YOLO11 output shaped `[1, 84, N]` or `[1, N, 84]`
- COCO person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign classes

Exports that perform NMS inside the model and return `[1, N, 6]` are not supported yet. The v0.8.0 default APK includes the official ONNX Runtime QNN AAR and Qualcomm QNN Runtime. It first tries the whole graph on HTP. If any operation would need CPU, `session.disable_cpu_ep_fallback=1` rejects that QNN session before the app falls back to NNAPI and finally CPU. The NNAPI session allows FP16, does not force the potentially slower NCHW option, and disables NNAPI CPU. When NNAPI is selected, supported partitions may run on the NPU, DSP, or GPU while other work can still use ORT CPU kernels.

No YOLO model is bundled in the APK. **The recommended first-test model is dynamic-input YOLO11n Detection, FP32 ONNX, with no embedded NMS.** The verified selector contains three official models.

| Model | Profile | COCO mAP50-95 | Download | First-test setting |
|---|---|---:|---:|---|
| YOLO11n | Speed | 39.5 | 10.4MB | 320, 10–15FPS |
| YOLO11s | Balanced; flagship recommendation | 47.0 | 36.3MB | 320/416, 5–10FPS |
| YOLO11m | Accuracy | 51.5 | 76.9MB | 320/416, 3–5FPS |

Use 320 for performance-first testing, 416 for balance, and 640 only to compare small-object quality. A 320 input has one quarter of the pixels of 640, so establish sustained performance and heat at 320 first. All three files use dynamic float32 I/O while QNN and NNAPI may use FP16 internally. A static model always uses its own fixed input regardless of the app selection. YOLO11l/x, YOLOv8, or a compatible custom model may also work through manual selection but require separate performance and output validation. YOLO26 end-to-end, segmentation, pose, classification, and OBB models are not currently supported.

Press **권장 모델 다운로드** for the selected YOLO11n/s/m model and review the Ultralytics model-license notice. Files come directly from the official `ultralytics/assets` v8.4.0 Release. YOLO11n pins `10,930,182` bytes and SHA-256 `634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b`; YOLO11s pins `38,051,729` bytes and `21d6650c5097610c92c76ce5e4b717976059169eaea4962035b90c6a92c07a8f`; YOLO11m pins `80,673,621` bytes and `8a37b5c53ff642831aa454156b548ec2cf2537827445385c3e1c1b276cb666a3`. The app also validates dynamic FP32 input and conventional COCO YOLO output before moving each file into internal storage. A failed or cancelled download removes its temporary file, and only a successful download becomes active.

Use **다른 ONNX 파일 선택** when offline or when testing another compatible model. To create the recommended format on a PC, retain `nms=False` and `dynamic=False`:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True nms=False dynamic=False batch=1
```

The official model is subject to Ultralytics AGPL-3.0 or Enterprise terms. Review the license link in the confirmation dialog and use the model within the appropriate terms. The app binds the downloaded model's dynamic `batch`, `height`, and `width` axes to the selected size when it creates a QNN session. A custom static QDQ W8A16 model can also be selected at 320, 416, or 640 when it preserves float32 I/O and the supported plain YOLO output. Quantization requires representative day and night road calibration data; do not distribute a model calibrated with arbitrary data without an accuracy comparison.

## Build and install

Android SDK 35 and JDK 17 or newer are required.

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug :app:lintDebug
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

The default build pulls [ONNX Runtime QNN from Maven Central](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android-qnn/1.24.3) and its transitive Qualcomm QNN Runtime dependency. It still handles QNN session failure and runs through NNAPI or CPU on non-Qualcomm devices. Native QNN libraries are compressed in the APK and extracted at install time, so the current debug APK is about 79 MB but can require more than 200 MB of installed storage. Backend and model requirements follow the [ONNX Runtime QNN guide](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html). Build a smaller NNAPI/CPU-only test APK with:

```powershell
.\gradlew.bat :app:assembleDebug -PcarrotQnnEnabled=false
```

That lightweight build shows `QNN/HTP runtime not included` at the top of the app and never attempts QNN. Both variants use the same application ID, so installing either replaces the previous app.

The app requests notification permission on Android 13 or newer. Android 12 and newer restrict arbitrary foreground-service launches from the background, so open the app once after a reboot. While active, a persistent notification provides status and a **중지** action.

A recommended-model download requires internet access and about 10.4–76.9MB of transfer. Keep the app open until it finishes. Verified models are reused from internal storage on later launches. The selected model's re-download button revalidates it, while **선택 모델 삭제** removes only that model.

The current app targets SDK 35, so its `INTERNET` permission provides local TCP/UDP access. If the target SDK is raised to Android 17/API 37 or newer, add the `ACCESS_LOCAL_NETWORK` runtime-permission flow at the same time.

## Automatic C3/C3X/C4 pairing

You do not need to find the device address manually when a phone hotspot assigns a different C3X address.

1. Connect the phone and C3/C3X/C4 to the same trusted Wi-Fi or phone hotspot. Disable AP/client isolation.
2. Set `ExternalAIEnabled=1` and the recommended H.264 option `ExternalAITransport=1` in Carrot Web. Object display also requires `ExternalAIShowOverlay=1`. End the current drive and start again after changing the transport.
3. On a phone hotspot, set `ExternalAIPhoneIP` to the phone's hotspot gateway address. It may be left empty while testing if the address is unknown, but only on a trusted dedicated network.
4. Put the C3/C3X/C4 on-road. `phoneaid` opens TCP frame port `7724` only while on-road.
5. Select YOLO11n, YOLO11s, or YOLO11m and press **권장 모델 다운로드**, or use **다른 ONNX 파일 선택** for another compatible model. The recommended path automatically verifies integrity and ONNX input/output shapes.
6. Leave **앱 실행 시 같은 망 자동 검색 및 시작** enabled. The app checks the saved address first, then probes only TCP `7724` in the local private IPv4 `/24` and accepts only a server whose first four bytes are JPEG `CAI1` or H.264 `CAI2`.
7. When found, the app saves the device address and starts YOLO automatically. Confirm **연결됨** in the app and a green `eNPU` or blue `eCPU` badge on the C3X.

Discovery is bounded to at most two local private `/24` networks. It does not scan the internet or a range of ports. If the frame port is changed, enter the same port in the app; discovery then verifies `CAI1` or `CAI2` on that port. Discovery may fail on a VPN, guest Wi-Fi, or a network with AP isolation.

For a manual fallback, enter the current C3/C3X/C4 address under **기기 IP** and press **입력 IP로 시작**. When automatic connection is enabled and a model has been saved, opening the app starts discovery without another button press. It does not start at boot; reopen it after a phone reboot or force-stop.

## Performance measurement

The app's **YOLO input size** defaults to 320, and the inference target is selectable from 1–20 FPS. Run 320 at 5 FPS for at least 15 minutes before raising the FPS or input to 416 or 640. The central HUD derives `VIDEO FPS` from C3X source-frame timestamps, shows measured model `AI FPS`, calculates `follow rate` as AI FPS divided by VIDEO FPS, and reports their difference as `SKIP/s`. This avoids falsely reporting 100% when the phone processing loop itself is slow. The live console keeps the newest 40 analysis entries with wall time, frame ID, object name, confidence, and source-image box and center pixel coordinates. Object names use Korean plus the original COCO label when Android's system language is Korean, and the COCO English label otherwise.

The status view separates the rolling 120-sample average and p95 phone time, current H.264/JPEG decode, preprocessing, ORT runtime, postprocessing, battery temperature, and Android thermal state. The C3X status shows the selected input plus `total latency/AI processing time`. The app lights green `eNPU` only after a full-graph Qualcomm QNN/HTP session with CPU fallback disabled completes its warm-up inference. NNAPI success is shown as amber `eACCEL` because it does not prove which physical accelerator ran the graph; CPU fallback is purple `eCPU`. If the status contains `QNN fallback`, record the following reason and evaluate the selected NNAPI or CPU path instead.

- A large `ORT` value indicates a model or acceleration-backend bottleneck; stay at 320 and check for CPU fallback.
- A large `total latency - phone time` indicates C3X encoding, Wi-Fi, or return-path delay.
- p95 rising far above the average, or a `performance limited` thermal state, indicates likely thermal throttling.
- If sustained performance is insufficient at 320, do not raise the input or select a larger model.

## Screen-off and power behavior

- Before a device is found, the app does not open the YOLO model or NPU session and does not hold a wake lock or high-performance Wi-Fi lock. Its foreground-service discovery wait increases through 5, 10, and 20 seconds to a 30-second cap. Android or OEM power policy may defer discovery further while the screen is off.
- A recommended-model download runs only after a user presses the button and does not acquire inference wake locks. Performance locks begin only after model installation and device connection.
- After TCP video connects, a partial wake lock and high-performance Wi-Fi lock keep reception and inference running with the screen off. This is a performance mode, so charging and thermal monitoring are recommended.
- A disconnect immediately releases both performance locks. Failed connections retry after 1, 2, 4, 8, and 16 seconds, capped at 30 seconds; automatic mode scans again if the address changed.
- Press **중지** in the app for zero background use; this disables automatic connection and ends discovery, inference, networking, and locks. The notification action also ends the current service but preserves the automatic-connection preference, so opening the app later can start it again.
- The connected partial wake lock has a six-hour safety timeout. Restart the client before an unusually long continuous session.

## Protocol and security

TCP frames use a 12-byte network-order prefix. JPEG `CAI1` carries JSON-header and JPEG lengths; H.264 `CAI2` carries JSON-header and access-unit lengths. The H.264 header also carries keyframe state and SPS/PPS size, and recovery after connection or congestion resumes at an IDR. The common JSON includes protocol version, frame ID, the C3X monotonic source timestamp, dimensions, and encoding. The phone returns the frame ID and source timestamp in the UDP JSON format defined by `openpilot/selfdrive/carrot/external_ai/protocol.py`.

TCP/UDP traffic is not encrypted, and discovery is not authentication. Do not expose these ports on a public or untrusted network.

The recommended models come directly from `yolo11n.onnx`, `yolo11s.onnx`, and `yolo11m.onnx` under `https://github.com/ultralytics/assets/releases/download/v8.4.0/`. Installation requires matching pinned file size, SHA-256, and ONNX structure. Integrity verification does not replace the model license.
