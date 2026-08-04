# Carrot External AI Android 클라이언트

이 실험용 앱은 C3, C3X 또는 C4의 `phoneaid`에서 기본 854×480 H.264 하드웨어 스트림 또는 호환용 640×360 JPEG 도로 영상을 받아 스마트폰에서 COCO YOLO ONNX 모델을 실행하고 정규화된 객체 탐지 결과만 돌려줍니다. H.264는 Android `MediaCodec`으로 하드웨어 디코딩하며 사용할 수 없으면 같은 연결에서 JPEG로 자동 전환합니다. 결과는 화면 표시에만 사용하며 차량 제어, CAN, Panda, 레이더 또는 안전 모델에는 전달하지 않습니다. 현재 실기 검증 기준은 C3X이며 C3와 C4는 장치별 도로 영상·UI 시험이 추가로 필요합니다.

## 앱 화면

v0.10.0부터 밝은 카드형 화면과 **상태·모델·로그** 탭을 사용합니다. **상태**에는 연결 주소, 활성 YOLO 모델, `eNPU`/`eCPU`, VIDEO FPS, AI FPS, 프레임 추종률, SKIP, 최근 5개 탐지가 표시됩니다. **모델**에서 권장 모델 다운로드와 입력 크기·목표 FPS·신뢰도 임계값을 설정합니다. NPU 모델을 선택하면 입력 크기는 모델의 정적 shape로 자동 고정됩니다. **로그**에서는 전체 실시간 객체 분석 기록을 확인하고 지울 수 있습니다. 수동 IP·포트·SSH 명령 복사는 상태 탭의 **연결 관리**에 있습니다.

## 지원 환경과 모델

- Android 10(API 29) 이상 64비트 ARM(`arm64-v8a`) 스마트폰
- Qualcomm QNN/HTP 전체 그래프 우선 실행, NNAPI와 CPU 순서의 자동 폴백
- Float32 NCHW 그래프 입출력. QNN 모델은 정적 320 또는 640, CPU 호환 모델은 동적 320·416·640
- 표준 Ultralytics YOLOv8/YOLO11 출력 `[1, 84, N]` 또는 `[1, N, 84]`
- person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign COCO 클래스

NMS를 모델 안에서 끝내고 `[1, N, 6]`을 반환하는 내보내기 형식은 아직 지원하지 않습니다. v0.10.0 기본 APK는 공식 ONNX Runtime QNN AAR과 Qualcomm QNN Runtime을 포함합니다. 먼저 HTP에서 전체 그래프를 열고, 한 연산이라도 CPU가 필요하면 `session.disable_cpu_ep_fallback=1`에 의해 QNN 세션을 거부한 뒤 NNAPI, 마지막으로 CPU 순서로 폴백합니다. NNAPI 세션은 FP16을 허용하고 느릴 수 있는 NCHW 강제 옵션과 NNAPI CPU는 사용하지 않습니다. NNAPI가 선택되면 지원되는 그래프가 NPU·DSP·GPU와 ORT CPU에 혼합 배치될 수 있습니다.

YOLO 모델은 APK에 포함하지 않습니다. **첫 시험 권장 모델은 `YOLO11n NPU W8A16 · 320`**입니다. 공식 YOLO11n을 고정 shape로 단순화하고 COCO128 128장으로 QDQ 양자화했으며, float32 입출력과 모델 밖 NMS를 유지합니다. CPU ONNX 출력 검증은 완료했지만 실제 HTP 전체 그래프 배치와 양자화 정확도는 실기 검증 중입니다.

| 모델 | 용도 | 정확도 | 다운로드 | 첫 시험값 |
|---|---|---:|---:|---|
| YOLO11n NPU W8A16 · 320 | NPU 기본 | 실기 검증 중 | 2.9MB | 고정 320, 10~15FPS |
| YOLO11n NPU W8A16 · 640 | NPU 고화질 | 실기 검증 중 | 2.9MB | 고정 640, 5~10FPS |
| YOLO11n Dynamic FP32 | CPU 호환·속도 | 39.5 | 10.4MB | 320, 5~10FPS |
| YOLO11s Dynamic FP32 | CPU 호환·균형 | 47.0 | 36.3MB | 320/416, 3~5FPS |
| YOLO11m Dynamic FP32 | CPU 호환·정확도 | 51.5 | 76.9MB | 320, 2~3FPS |

입력 픽셀 수는 320이 640의 1/4이므로 먼저 NPU 320에서 지속 성능과 발열을 확인하세요. 두 NPU 모델은 QUInt16 activation·QUInt8 weight QDQ 그래프이며 고정 입력 크기를 앱이 자동 적용합니다. Dynamic FP32 세 모델은 QNN을 시도하지 않는 CPU/NNAPI 호환 경로입니다. YOLO11l/x, YOLOv8 또는 직접 학습한 호환 모델도 수동 선택할 수 있지만 성능·출력 형식은 별도 검증이 필요합니다. YOLO26 end-to-end, segmentation, pose, classification, OBB 모델은 현재 지원하지 않습니다.

앱의 **권장 모델 다운로드**는 파일 크기·SHA-256·ONNX 입출력을 검증합니다. NPU 320은 `3,047,718`바이트와 SHA-256 `42a8170f1ce782cf87b781eb4f249b6e1d04e5034c4c904179dcbbc721110027`, NPU 640은 `3,085,627`바이트와 `b4bdd62de9f07e9b29fd08f3482719d650c853cdfa7230e589770105361259fd`입니다. Dynamic FP32 모델은 기존 공식 `ultralytics/assets` v8.4.0 파일과 고정 hash를 유지합니다. 임시 파일은 검증 실패·취소 시 삭제하며 다운로드가 성공해야 자동 연결에 사용합니다. 모델은 개별 삭제할 수 있고 앱 삭제 시 모두 제거됩니다.

인터넷이 없거나 다른 호환 모델을 시험할 때는 **다른 ONNX 파일 선택**을 사용합니다. PC에서 권장 형식을 직접 만들려면 `nms=False`와 `dynamic=False`를 유지합니다.

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True nms=False dynamic=False batch=1
```

공식 모델과 파생 QDQ 모델은 Ultralytics AGPL-3.0 또는 Enterprise 조건이 적용됩니다. 다운로드 확인창의 라이선스 링크를 검토하고 사용 범위에 맞게 이용하세요. 생성 절차와 manifest는 `model_tools/build_qnn_qdq_models.py`와 `models/README.md`에 기록돼 있습니다. COCO128은 첫 동작 검증용이며, 다음 정확도 개선 단계에서는 실제 주·야간 도로 영상 calibration과 FP32 대비 mAP 평가가 필요합니다.

## 빌드와 설치

Android SDK 35와 JDK 17 이상이 필요합니다.

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug :app:lintDebug
adb install -r .\app\build\outputs\apk\debug\app-debug.apk
```

UI만 Android x86_64 에뮬레이터에서 확인할 때는 `-PcarrotQnnEnabled=false -PcarrotTargetAbi=x86_64`를 함께 지정합니다. 실제 배포 APK는 옵션 없이 빌드하여 기본 `arm64-v8a`와 QNN Runtime을 유지합니다.

기본 빌드는 ONNX Runtime Android 1.26.0, Qualcomm [QNN Plugin EP 2.4.0](https://github.com/onnxruntime/onnxruntime-qnn), QNN Runtime 2.48.0을 APK에 포함합니다. 먼저 `session.disable_cpu_ep_fallback=1`로 전체 HTP 그래프를 검사하고, 실패하면 QNN+CPU 혼합 세션을 열어 ORT 프로파일에서 실제 QNN 노드 실행을 확인합니다. 앱의 가속 진단에는 SoC와 전체 그래프·혼합 실행 실패 원문이 C3X 검색 상태와 별도로 유지됩니다. QNN 네이티브 라이브러리를 APK 안에서는 압축하고 설치 시 꺼내므로 설치 공간은 APK보다 더 많이 필요할 수 있습니다. APK 크기를 줄인 NNAPI·CPU 전용 시험 빌드는 다음처럼 만듭니다.

```powershell
.\gradlew.bat :app:assembleDebug -PcarrotQnnEnabled=false
```

이 경량 빌드에서는 앱 상단에 `QNN/HTP 런타임 미포함`이 표시되며 QNN을 시도하지 않습니다. 두 빌드는 같은 애플리케이션 ID이므로 하나를 설치하면 기존 앱을 교체합니다.

Android 13 이상에서는 알림 권한을 요청합니다. Android 12 이상은 백그라운드 앱의 임의 포그라운드 서비스 시작을 제한하므로 재부팅 후에는 앱을 한 번 직접 열어야 합니다. 실행 중에는 지속 알림에서 상태와 **중지** 동작을 제공합니다.

권장 모델 다운로드에는 인터넷 연결과 모델에 따라 약 2.9~76.9MB의 전송량이 필요합니다. 다운로드 중에는 앱을 닫지 마세요. 설치가 끝나면 모델은 내부 저장소에서 재사용되므로 다음 실행부터 다시 받을 필요가 없습니다. 선택 모델의 **다시 다운로드**로 고정 버전을 재검증할 수 있고 **선택 모델 삭제**로 해당 내부 파일만 제거할 수 있습니다.

현재 앱은 target SDK 35이므로 `INTERNET` 권한으로 로컬 TCP/UDP에 접근합니다. 향후 target SDK를 Android 17/API 37 이상으로 올릴 때는 `ACCESS_LOCAL_NETWORK` 런타임 권한 처리를 함께 추가해야 합니다.

## C3/C3X/C4 브랜치 변경 명령 복사

앱의 **C3X 브랜치 변경 SSH 명령 복사**를 누르면 현재 **기기 IP**를 넣은 한 줄 명령이 클립보드에 저장됩니다. 자동 검색이 완료됐다면 검색된 주소가 사용됩니다. 주차하고 주행을 끝낸 뒤 SSH가 설치된 PC PowerShell, 터미널 또는 Android 터미널 앱에 붙여넣으세요.

명령은 `/data/openpilot`에서 `junghc11/openpilot`의 `external-android-ai`를 `FETCH_HEAD`로 직접 가져옵니다. 기존 로컬 브랜치가 있으면 전환해 fast-forward하고, 없으면 `FETCH_HEAD`에서 새 로컬 브랜치를 만듭니다. 별도의 Git 원격이나 추적 브랜치를 등록할 필요가 없습니다. `reset --hard`나 로컬 파일 삭제, 자동 재부팅은 수행하지 않습니다. 로컬 변경 때문에 전환 또는 갱신이 안전하지 않으면 Git이 중단합니다. 성공 후 표시되는 짧은 커밋 ID를 확인하고 C3/C3X/C4를 직접 재부팅하세요.

## C3/C3X/C4 자동 연결

스마트폰 핫스팟이 C3X에 매번 다른 주소를 할당해도 기기 IP를 직접 찾을 필요가 없습니다.

1. 스마트폰과 C3/C3X/C4를 같은 신뢰할 수 있는 Wi-Fi 또는 스마트폰 핫스팟에 연결합니다. AP·클라이언트 격리는 꺼야 합니다.
2. 웹당근에서 `ExternalAIEnabled=1`, 권장 H.264에는 `ExternalAITransport=1`을 설정합니다. 객체 표시에는 `ExternalAIShowOverlay=1`도 필요합니다. 전송 방식 변경은 현재 주행을 끝내고 다시 시작한 뒤 적용됩니다.
3. 스마트폰 핫스팟에서는 `ExternalAIPhoneIP`에 핫스팟 게이트웨이인 스마트폰 주소를 넣습니다. 시험 중 주소를 모르면 비워둘 수 있지만 신뢰할 수 있는 전용망에서만 사용하세요.
4. C3/C3X/C4를 주행 상태로 전환합니다. `phoneaid`는 주행 중에만 TCP 영상 포트 `7724`를 엽니다.
5. Android 앱에서 기본 `YOLO11n NPU W8A16 · 320`을 선택해 **권장 모델 다운로드**를 누릅니다. 비 Qualcomm 기기나 비교 시험에는 Dynamic FP32 호환 모델을 선택할 수 있습니다.
6. 기본으로 켜진 **앱 실행 시 같은 망 자동 검색 및 시작**을 유지합니다. 앱은 저장된 주소를 먼저 확인한 뒤 같은 사설 IPv4 `/24`에서 TCP `7724`만 병렬 확인하고, 첫 4바이트가 JPEG `CAI1` 또는 H.264 `CAI2`인 기기만 선택합니다.
7. 기기를 찾으면 주소를 자동 저장하고 YOLO를 시작합니다. 앱의 **연결됨**과 C3X의 초록색 `eNPU` 또는 파란색 `eCPU` 배지를 확인합니다.

자동 검색은 최대 두 개의 로컬 사설 `/24`만 확인하며 인터넷이나 임의 포트 범위를 스캔하지 않습니다. 포트를 바꾼 경우 앱의 영상 포트에도 같은 값을 넣으면 그 포트에서 `CAI1` 또는 `CAI2`를 확인합니다. VPN·게스트 Wi-Fi·AP 격리 환경에서는 검색되지 않을 수 있습니다.

자동 검색이 실패하면 **기기 IP**에 현재 C3/C3X/C4 주소를 입력하고 **입력 IP로 시작**을 누를 수 있습니다. 자동 연결이 켜져 있고 모델이 저장돼 있으면 이후 앱 실행 시 버튼 없이 검색 서비스를 시작합니다. 다만 Android 재부팅·앱 강제 종료 후에는 앱을 다시 열어야 하며 부팅 자동 시작은 하지 않습니다.

## 성능 측정

NPU 모델의 **YOLO 입력 크기**는 각각 320 또는 640으로 자동 고정되며 목표 추론률은 1~20 FPS에서 선택합니다. 먼저 NPU 320·5FPS로 15분 이상 실행한 뒤 10~15FPS로 올리세요. 중앙 HUD의 `VIDEO FPS`는 C3X 원본 프레임 타임스탬프로 계산한 영상률, `AI FPS`는 현재 모델의 실측 처리율, `추종률`은 AI FPS/VIDEO FPS, `SKIP/s`는 두 속도의 차이입니다. 따라서 스마트폰 처리 루프가 느려져도 추종률이 100%로 잘못 보이지 않습니다. 객체 콘솔에는 시각, 프레임 ID, 객체명, 신뢰도, 원본 영상 기준 box와 center 픽셀 좌표가 최근 40개 분석 단위로 표시됩니다. 객체명은 Android 시스템 언어가 한국어이면 한글과 COCO 원문을 함께, 그 외 언어이면 COCO 영문으로 표시합니다.

상태 화면은 최근 120개 처리 표본의 전화 처리 평균과 p95, 현재 H.264/JPEG 디코딩·전처리·ORT 런타임·후처리 평균, 배터리 온도와 Android 열 상태를 구분해 표시합니다. C3X 상태에는 선택 입력 크기와 `총 지연/AI 처리시간`이 표시됩니다. Qualcomm QNN/HTP 전체·혼합 세션 또는 CPU를 제외한 NNAPI 가속 세션이 동작하면 초록색 `eNPU`, CPU 폴백이면 보라색 `eCPU`로 표시합니다. 배지는 NPU 가속 환경을 간단히 나타내며, 전체 QNN·QNN+CPU 혼합·QNN 검증 보류·NNAPI 구분과 프로파일 증거는 가속 진단 상세 문구에서 확인합니다. `QNN 폴백` 문구가 보이면 뒤의 원인과 처리시간을 함께 기록하세요.

- `ORT`만 큰 경우: 모델·가속 백엔드 병목입니다. 320을 유지하고 CPU 폴백 여부를 확인합니다.
- `총 지연 - 전화 처리`가 큰 경우: C3X 인코딩, Wi-Fi 또는 반환 경로 병목입니다.
- p95가 평균보다 크게 상승하거나 열 상태가 `성능 제한`이면 발열 스로틀링 가능성이 큽니다.
- 320에서도 지속 성능이 부족하면 640이나 더 큰 모델로 올리지 않습니다.

## 화면 꺼짐과 절전 동작

- 기기를 찾기 전에는 YOLO 모델이나 NPU 세션을 열지 않고 Wake Lock과 고성능 Wi-Fi Lock도 사용하지 않습니다. 포그라운드 서비스의 검색 간격은 5·10·20초 뒤 최대 30초로 늘어나며, Android·제조사 절전 정책에 따라 화면이 꺼졌을 때 검색이 더 늦어질 수 있습니다.
- 권장 모델 다운로드는 사용자가 버튼을 눌렀을 때만 실행하며 추론용 Wake Lock을 잡지 않습니다. 다운로드가 끝나고 기기 연결이 시작된 뒤에만 추론 성능 Lock을 사용합니다.
- TCP 영상 연결 뒤에는 화면이 꺼져도 추론하도록 부분 Wake Lock과 고성능 Wi-Fi Lock을 사용합니다. 이는 절전 모드가 아니므로 충전과 발열 확인이 필요합니다.
- 연결이 끊기면 두 성능 Lock을 즉시 해제합니다. 실패한 연결은 1·2·4·8·16초 뒤 재시도하고 이후 최대 30초로 제한하며, 주소가 바뀌었으면 같은 망을 다시 검색합니다.
- 앱의 **중지**를 누르면 자동 연결 설정을 끄고 검색·추론·네트워크·Lock을 모두 종료합니다. 알림의 **중지**도 현재 서비스를 완전히 종료하지만 자동 연결 설정은 보존하므로, 다음에 앱을 열면 다시 시작할 수 있습니다.
- 연결 중 부분 Wake Lock에는 6시간 안전 제한이 있습니다. 매우 긴 연속 사용 전에는 클라이언트를 다시 시작하세요.

## 프로토콜과 보안

TCP 프레임은 12바이트 네트워크 바이트 순서 접두사를 사용합니다. JPEG `CAI1`은 JSON 헤더 길이와 JPEG 길이를, H.264 `CAI2`는 JSON 헤더 길이와 access unit 길이를 담습니다. H.264 헤더에는 키프레임 여부와 SPS/PPS 크기도 포함되며 연결·혼잡 복구 뒤에는 IDR부터 다시 시작합니다. 공통 JSON에는 프로토콜 버전, 프레임 ID, C3X 단조시계 원본 시각, 크기와 인코딩이 들어갑니다. 스마트폰은 `openpilot/selfdrive/carrot/external_ai/protocol.py`의 UDP JSON 형식으로 프레임 ID와 원본 시각을 돌려줍니다.

TCP/UDP 통신은 암호화되지 않습니다. 자동 검색도 인증 수단이 아니므로 공용 또는 신뢰할 수 없는 네트워크에 포트를 노출하지 마세요.

NPU QDQ 모델은 이 공개 브랜치의 `tools/android_ai_client/models/`에서, Dynamic FP32 모델은 공식 `ultralytics/assets` v8.4.0 Release에서 HTTPS로 다운로드합니다. 앱에 고정한 파일 크기·SHA-256과 ONNX 형식이 모두 맞아야 설치됩니다. 이 무결성 검사는 모델 라이선스를 대체하지 않습니다.
