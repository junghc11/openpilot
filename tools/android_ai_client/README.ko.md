# Carrot External AI Android 클라이언트

이 실험용 앱은 C3, C3X 또는 C4의 `phoneaid`에서 640×360 JPEG 도로 영상을 받아 스마트폰에서 COCO YOLO ONNX 모델을 실행하고 정규화된 객체 탐지 결과만 돌려줍니다. 결과는 화면 표시에만 사용하며 차량 제어, CAN, Panda, 레이더 또는 안전 모델에는 전달하지 않습니다. 현재 실기 검증 기준은 C3X이며 C3와 C4는 장치별 도로 영상·UI 시험이 추가로 필요합니다.

## 지원 환경과 모델

- Android 10(API 29) 이상 64비트 ARM(`arm64-v8a`) 스마트폰
- ONNX Runtime Android의 NNAPI 우선 실행과 CPU 자동 폴백
- Float32 NCHW 입력 `[1, 3, H, W]` 형식. 동적 H/W 기본값 640
- 표준 Ultralytics YOLOv8/YOLO11 출력 `[1, 84, N]` 또는 `[1, N, 84]`
- person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign COCO 클래스

NMS를 모델 안에서 끝내고 `[1, N, 6]`을 반환하는 내보내기 형식은 아직 지원하지 않습니다. NNAPI 세션에는 FP16과 NCHW를 허용하고 NNAPI CPU는 제외합니다. 지원되는 그래프는 스마트폰의 NPU·DSP·GPU에 배치될 수 있으며, 가속 세션 생성에 실패하면 전체 세션을 CPU로 다시 엽니다. Qualcomm QNN 직접 백엔드는 아직 포함하지 않습니다.

YOLO 모델은 APK에 포함되지 않습니다. **첫 시험 권장 모델은 `YOLO11n Detection`, 입력 640, FP32 ONNX, 모델 내 NMS 미포함**입니다. Nano 모델이라 스마트폰 실시간 시험에 부담이 가장 작고 현재 앱의 일반 COCO 출력 파서와 맞습니다. `YOLO11s/m/l/x`, YOLOv8 또는 직접 학습한 호환 모델도 선택할 수 있지만 성능·출력 형식은 별도 검증이 필요합니다. YOLO26 end-to-end, segmentation, pose, classification, OBB 모델은 현재 지원하지 않습니다.

모델 선택은 C3X가 아니라 **Android 앱의 `YOLO ONNX 모델 선택 (권장: YOLO11n 640)` 버튼**에서 합니다. 선택한 문서 URI는 다음 실행에도 보존됩니다. PC에서 권장 모델을 만드는 명령은 다음과 같습니다. `nms=False`와 `dynamic=False`를 유지하세요.

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True nms=False dynamic=False batch=1
```

재배포 전 모델과 프레임워크 라이선스를 확인하세요.

## 빌드와 설치

Android SDK 35와 JDK 17 이상이 필요합니다.

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug :app:lintDebug
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

Android 13 이상에서는 알림 권한을 요청합니다. Android 12 이상은 백그라운드 앱의 임의 포그라운드 서비스 시작을 제한하므로 재부팅 후에는 앱을 한 번 직접 열어야 합니다. 실행 중에는 지속 알림에서 상태와 **중지** 동작을 제공합니다.

현재 앱은 target SDK 35이므로 `INTERNET` 권한으로 로컬 TCP/UDP에 접근합니다. 향후 target SDK를 Android 17/API 37 이상으로 올릴 때는 `ACCESS_LOCAL_NETWORK` 런타임 권한 처리를 함께 추가해야 합니다.

## C3/C3X/C4 자동 연결

스마트폰 핫스팟이 C3X에 매번 다른 주소를 할당해도 기기 IP를 직접 찾을 필요가 없습니다.

1. 스마트폰과 C3/C3X/C4를 같은 신뢰할 수 있는 Wi-Fi 또는 스마트폰 핫스팟에 연결합니다. AP·클라이언트 격리는 꺼야 합니다.
2. 웹당근에서 `ExternalAIEnabled=1`을 설정합니다. 객체 표시에는 `ExternalAIShowOverlay=1`도 필요합니다.
3. 스마트폰 핫스팟에서는 `ExternalAIPhoneIP`에 핫스팟 게이트웨이인 스마트폰 주소를 넣습니다. 시험 중 주소를 모르면 비워둘 수 있지만 신뢰할 수 있는 전용망에서만 사용하세요.
4. C3/C3X/C4를 주행 상태로 전환합니다. `phoneaid`는 주행 중에만 TCP 영상 포트 `7724`를 엽니다.
5. Android 앱에서 호환되는 `.onnx` 모델을 한 번 선택합니다.
6. 기본으로 켜진 **앱 실행 시 같은 망 자동 검색 및 시작**을 유지합니다. 앱은 저장된 주소를 먼저 확인한 뒤 같은 사설 IPv4 `/24`에서 TCP `7724`만 병렬 확인하고, 첫 4바이트가 Carrot 프레임 서명 `CAI1`인 기기만 선택합니다.
7. 기기를 찾으면 주소를 자동 저장하고 YOLO를 시작합니다. 앱의 **연결됨**과 C3X의 초록색 `eNPU` 또는 파란색 `eCPU` 배지를 확인합니다.

자동 검색은 최대 두 개의 로컬 사설 `/24`만 확인하며 인터넷이나 임의 포트 범위를 스캔하지 않습니다. 포트를 바꾼 경우 앱의 영상 포트에도 같은 값을 넣으면 그 포트에서 `CAI1`을 확인합니다. VPN·게스트 Wi-Fi·AP 격리 환경에서는 검색되지 않을 수 있습니다.

자동 검색이 실패하면 **기기 IP**에 현재 C3/C3X/C4 주소를 입력하고 **입력 IP로 시작**을 누를 수 있습니다. 자동 연결이 켜져 있고 모델이 저장돼 있으면 이후 앱 실행 시 버튼 없이 검색 서비스를 시작합니다. 다만 Android 재부팅·앱 강제 종료 후에는 앱을 다시 열어야 하며 부팅 자동 시작은 하지 않습니다.

## 화면 꺼짐과 절전 동작

- 기기를 찾기 전에는 YOLO 모델이나 NPU 세션을 열지 않고 Wake Lock과 고성능 Wi-Fi Lock도 사용하지 않습니다. 포그라운드 서비스의 검색 간격은 5·10·20초 뒤 최대 30초로 늘어나며, Android·제조사 절전 정책에 따라 화면이 꺼졌을 때 검색이 더 늦어질 수 있습니다.
- TCP 영상 연결 뒤에는 화면이 꺼져도 추론하도록 부분 Wake Lock과 고성능 Wi-Fi Lock을 사용합니다. 이는 절전 모드가 아니므로 충전과 발열 확인이 필요합니다.
- 연결이 끊기면 두 성능 Lock을 즉시 해제합니다. 실패한 연결은 1·2·4·8·16초 뒤 재시도하고 이후 최대 30초로 제한하며, 주소가 바뀌었으면 같은 망을 다시 검색합니다.
- 앱의 **중지**를 누르면 자동 연결 설정을 끄고 검색·추론·네트워크·Lock을 모두 종료합니다. 알림의 **중지**도 현재 서비스를 완전히 종료하지만 자동 연결 설정은 보존하므로, 다음에 앱을 열면 다시 시작할 수 있습니다.
- 연결 중 부분 Wake Lock에는 6시간 안전 제한이 있습니다. 매우 긴 연속 사용 전에는 클라이언트를 다시 시작하세요.

## 프로토콜과 보안

TCP 프레임은 ASCII `CAI1`, JSON 헤더 길이, JPEG 길이로 구성된 12바이트 네트워크 바이트 순서 접두사를 사용합니다. JSON 헤더에는 프로토콜 버전, 프레임 ID, C3X 단조시계 원본 시각, 크기와 `jpeg` 인코딩이 들어갑니다. 스마트폰은 `openpilot/selfdrive/carrot/external_ai/protocol.py`의 UDP JSON 형식으로 프레임 ID와 원본 시각을 돌려줍니다.

TCP/UDP 통신은 암호화되지 않습니다. 자동 검색도 인증 수단이 아니므로 공용 또는 신뢰할 수 없는 네트워크에 포트를 노출하지 마세요.
