# Carrot External AI Android client

This experimental app receives the default 854×480 hardware H.264 stream or compatibility 640×360 JPEG road frames from `phoneaid` on a C3, C3X, or C4, runs a COCO YOLO ONNX model on the phone, and returns only normalized object detections. H.264 is hardware-decoded through Android `MediaCodec`; the connection automatically falls back to JPEG when it is unavailable. Results are visualization-only and are not delivered to vehicle control, CAN, Panda, radar, or the safety model. C3X is the current hardware validation target; C3 and C4 still need device-specific road-stream and UI tests.

## App screen

Starting with v0.10.0, the app uses a bright card-based layout with **Status, Model, and Log** tabs. **Status** shows the connection address, active YOLO model, `eNPU`/`eACCEL`/`eCPU`, VIDEO FPS, AI FPS, frame-follow rate, SKIP, and the five most recent detections. **Model** contains verified-model downloads plus input-size, target-FPS, and confidence settings. Selecting an NPU model automatically locks the input to its static shape. **Log** shows and clears the complete live detection console. Manual IP, ports, and the SSH deployment-command copy action are under **Connection management** on the Status tab.

## Supported environment and model

- A 64-bit ARM (`arm64-v8a`) phone with Android 10 (API 29) or newer
- Full-graph Qualcomm QNN/HTP first, then automatic NNAPI and CPU fallbacks
- Float32 NCHW graph I/O; QNN models use static 320 or 640 while compatibility models allow dynamic 320, 416, or 640
- Standard Ultralytics YOLOv8/YOLO11 output shaped `[1, 84, N]` or `[1, N, 84]`
- COCO person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign classes

Exports that perform NMS inside the model and return `[1, N, 6]` are not supported yet. The v0.10.0 default APK includes the official ONNX Runtime QNN AAR and Qualcomm QNN Runtime. It first tries the whole graph on HTP. If any operation would need CPU, `session.disable_cpu_ep_fallback=1` rejects that QNN session before the app falls back to NNAPI and finally CPU. The NNAPI session allows FP16, does not force the potentially slower NCHW option, and disables NNAPI CPU. When NNAPI is selected, supported partitions may run on the NPU, DSP, or GPU while other work can still use ORT CPU kernels.

No YOLO model is bundled in the APK. **The recommended first-test model is `YOLO11n NPU W8A16 · 320`.** It fixes and simplifies the official YOLO11n graph, uses QDQ calibration from all 128 COCO128 images, and retains float32 I/O with NMS outside the graph. CPU ONNX structure/output validation passed; physical full-graph HTP placement and quantized accuracy still require device testing.

| Model | Profile | Accuracy | Download | First-test setting |
|---|---|---:|---:|---|
| YOLO11n NPU W8A16 · 320 | Default NPU | Device validation pending | 2.9MB | fixed 320, 10–15FPS |
| YOLO11n NPU W8A16 · 640 | Detail NPU | Device validation pending | 2.9MB | fixed 640, 5–10FPS |
| YOLO11n Dynamic FP32 | CPU compatibility; speed | 39.5 | 10.4MB | 320, 5–10FPS |
| YOLO11s Dynamic FP32 | CPU compatibility; balance | 47.0 | 36.3MB | 320/416, 3–5FPS |
| YOLO11m Dynamic FP32 | CPU compatibility; accuracy | 51.5 | 76.9MB | 320, 2–3FPS |

A 320 input has one quarter of the pixels of 640, so establish sustained performance and heat with NPU 320 first. The two NPU models use QUInt16 activations and QUInt8 weights in a QDQ graph, and the app applies their fixed input automatically. The three Dynamic FP32 entries are CPU/NNAPI compatibility paths and intentionally skip QNN. YOLO11l/x, YOLOv8, or a compatible custom model may also work through manual selection but require separate performance and output validation. YOLO26 end-to-end, segmentation, pose, classification, and OBB models are not currently supported.

Press **권장 모델 다운로드** and review the Ultralytics model-license notice. NPU 320 pins `3,047,718` bytes and SHA-256 `42a8170f1ce782cf87b781eb4f249b6e1d04e5034c4c904179dcbbc721110027`; NPU 640 pins `3,085,627` bytes and `b4bdd62de9f07e9b29fd08f3482719d650c853cdfa7230e589770105361259fd`. Dynamic FP32 entries retain the official `ultralytics/assets` v8.4.0 files and hashes. The app validates size, SHA-256, float32 NCHW input, and conventional COCO YOLO output before installation. A failed or cancelled download removes its temporary file, and only a successful download becomes active.

Use **다른 ONNX 파일 선택** when offline or when testing another compatible model. To create the recommended format on a PC, retain `nms=False` and `dynamic=False`:

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True nms=False dynamic=False batch=1
```

The official and derived QDQ models are subject to Ultralytics AGPL-3.0 or Enterprise terms. Review the license link in the confirmation dialog and use the model within the appropriate terms. The reproducible build and manifest are under `model_tools/build_qnn_qdq_models.py` and `models/README.md`. COCO128 is sufficient for the first execution test; the next accuracy phase requires representative day/night road calibration and an mAP comparison against FP32.

## Build and install

Android SDK 35 and JDK 17 or newer are required.

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug :app:lintDebug
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

For UI-only validation on an Android x86_64 emulator, add `-PcarrotQnnEnabled=false -PcarrotTargetAbi=x86_64`. Build the real deployment APK without those properties so the default `arm64-v8a` ABI and QNN Runtime remain enabled.

The default build packages ONNX Runtime Android 1.26.0, Qualcomm [QNN Plugin EP 2.4.0](https://github.com/onnxruntime/onnxruntime-qnn), and QNN Runtime 2.48.0. It first verifies a full HTP graph with `session.disable_cpu_ep_fallback=1`; if that fails, it opens a mixed QNN+CPU session and uses the ORT profile to confirm actual QNN node execution. The acceleration diagnostics retain the SoC and the full-graph/mixed failure text independently of C3X discovery status. Native QNN libraries are compressed in the APK and extracted at install time, so installed storage can be substantially larger than the APK. Build a smaller NNAPI/CPU-only test APK with:

```powershell
.\gradlew.bat :app:assembleDebug -PcarrotQnnEnabled=false
```

That lightweight build shows `QNN/HTP runtime not included` at the top of the app and never attempts QNN. Both variants use the same application ID, so installing either replaces the previous app.

The app requests notification permission on Android 13 or newer. Android 12 and newer restrict arbitrary foreground-service launches from the background, so open the app once after a reboot. While active, a persistent notification provides status and a **중지** action.

A recommended-model download requires internet access and about 2.9–76.9MB of transfer. Keep the app open until it finishes. Verified models are reused from internal storage on later launches. The selected model's re-download button revalidates it, while **선택 모델 삭제** removes only that model.

The current app targets SDK 35, so its `INTERNET` permission provides local TCP/UDP access. If the target SDK is raised to Android 17/API 37 or newer, add the `ACCESS_LOCAL_NETWORK` runtime-permission flow at the same time.

## Copying the C3/C3X/C4 branch deployment command

Press **C3X 브랜치 변경 SSH 명령 복사** to copy a one-line SSH command containing the current **기기 IP**. After automatic discovery, the discovered address is used. Park and end the drive before pasting it into PC PowerShell, a terminal, or an Android terminal app with SSH installed.

The command fetches `external-android-ai` directly from `junghc11/openpilot` into `FETCH_HEAD` under `/data/openpilot`. It switches and fast-forwards an existing local branch, or creates a new local branch from `FETCH_HEAD`. No permanent Git remote or tracking branch is required. It does not run `reset --hard`, delete local files, or reboot automatically. Git stops if local changes make the switch or fast-forward unsafe. Verify the short commit ID printed on success, then reboot the C3/C3X/C4 yourself.

## Automatic C3/C3X/C4 pairing

You do not need to find the device address manually when a phone hotspot assigns a different C3X address.

1. Connect the phone and C3/C3X/C4 to the same trusted Wi-Fi or phone hotspot. Disable AP/client isolation.
2. Set `ExternalAIEnabled=1` and the recommended H.264 option `ExternalAITransport=1` in Carrot Web. Object display also requires `ExternalAIShowOverlay=1`. End the current drive and start again after changing the transport.
3. On a phone hotspot, set `ExternalAIPhoneIP` to the phone's hotspot gateway address. It may be left empty while testing if the address is unknown, but only on a trusted dedicated network.
4. Put the C3/C3X/C4 on-road. `phoneaid` opens TCP frame port `7724` only while on-road.
5. Select the default `YOLO11n NPU W8A16 · 320` and press **권장 모델 다운로드**. Use a Dynamic FP32 compatibility model for non-Qualcomm devices or comparison testing.
6. Leave **앱 실행 시 같은 망 자동 검색 및 시작** enabled. The app checks the saved address first, then probes only TCP `7724` in the local private IPv4 `/24` and accepts only a server whose first four bytes are JPEG `CAI1` or H.264 `CAI2`.
7. When found, the app saves the device address and starts YOLO automatically. Confirm **연결됨** in the app and a green `eNPU` or blue `eCPU` badge on the C3X.

Discovery is bounded to at most two local private `/24` networks. It does not scan the internet or a range of ports. If the frame port is changed, enter the same port in the app; discovery then verifies `CAI1` or `CAI2` on that port. Discovery may fail on a VPN, guest Wi-Fi, or a network with AP isolation.

For a manual fallback, enter the current C3/C3X/C4 address under **기기 IP** and press **입력 IP로 시작**. When automatic connection is enabled and a model has been saved, opening the app starts discovery without another button press. It does not start at boot; reopen it after a phone reboot or force-stop.

## Performance measurement

An NPU model automatically locks **YOLO input size** to 320 or 640, while the inference target remains selectable from 1–20 FPS. Run NPU 320 at 5 FPS for at least 15 minutes before raising it toward 10–15 FPS. The central HUD derives `VIDEO FPS` from C3X source-frame timestamps, shows measured model `AI FPS`, calculates `follow rate` as AI FPS divided by VIDEO FPS, and reports their difference as `SKIP/s`. This avoids falsely reporting 100% when the phone processing loop itself is slow. The live console keeps the newest 40 analysis entries with wall time, frame ID, object name, confidence, and source-image box and center pixel coordinates. Object names use Korean plus the original COCO label when Android's system language is Korean, and the COCO English label otherwise.

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

NPU QDQ models download from `tools/android_ai_client/models/` on this public branch; Dynamic FP32 models download from the official `ultralytics/assets` v8.4.0 Release. Installation requires matching pinned file size, SHA-256, and ONNX structure. Integrity verification does not replace the model license.
