from pathlib import Path


OPENPILOT_ROOT = Path(__file__).resolve().parents[3]
ANDROID_ROOT = OPENPILOT_ROOT.parent / "tools" / "android_ai_client"
JAVA_ROOT = ANDROID_ROOT / "app" / "src" / "main" / "java" / "ai" / "carrotpilot" / "external"


def test_android_app_auto_discovers_on_launch_with_manual_fallback() -> None:
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")
  service = (JAVA_ROOT / "ExternalAIService.kt").read_text(encoding="utf-8")
  manifest = (ANDROID_ROOT / "app" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")

  assert "앱 실행 시 같은 망 자동 검색 및 시작" in activity
  assert "autoConnect.isChecked && modelUri != null && !ExternalAIService.serviceActive" in activity
  assert "startClient(autoDiscover = true)" in activity
  assert "입력 IP로 시작" in activity
  assert "startClient(autoDiscover = false)" in activity
  assert "EXTRA_AUTO_DISCOVER" in activity
  assert "DeviceDiscovery.findHost(config.framePort, config.host)" in service
  assert "RECEIVE_BOOT_COMPLETED" not in manifest


def test_android_discovery_is_bounded_to_private_subnets_and_carrot_port() -> None:
  discovery = (JAVA_ROOT / "DeviceDiscovery.kt").read_text(encoding="utf-8")

  assert "ipv4.isSiteLocalAddress" in discovery
  assert "for (lastOctet in 1..254)" in discovery
  assert "MAX_SUBNETS = 2" in discovery
  assert "isCarrotFrameServer(candidate, framePort)" in discovery
  assert "socket.connect(InetSocketAddress(host, framePort), CONNECT_TIMEOUT_MS)" in discovery
  assert "magic.contentEquals(FRAME_MAGIC)" in discovery
  assert "SCAN_WORKERS = 24" in discovery
  assert "CONNECT_TIMEOUT_MS = 300" in discovery
  assert "READ_TIMEOUT_MS = 1_000" in discovery


def test_android_client_holds_performance_locks_only_while_connected() -> None:
  service = (JAVA_ROOT / "ExternalAIService.kt").read_text(encoding="utf-8")
  start_client = service.split("private fun startClient", 1)[1].split("private fun runClient", 1)[0]
  run_client = service.split("private fun runClient", 1)[1].split("private fun processFrames", 1)[0]

  assert "acquirePerformanceLocks()" not in start_client
  assert run_client.index("socket.connect(") < run_client.index("acquirePerformanceLocks()")
  assert "releasePerformanceLocks()" in run_client
  retry_wait = run_client.split("catch (error: Exception)", 1)[1]
  assert retry_wait.index("releasePerformanceLocks()") < retry_wait.index("SystemClock.sleep(retryDelayMs)")
  assert "MIN_RETRY_DELAY_MS = 1_000L" in service
  assert "MAX_RETRY_DELAY_MS = 30_000L" in service
  assert "MIN_DISCOVERY_RETRY_DELAY_MS = 5_000L" in service
  assert "MAX_DISCOVERY_RETRY_DELAY_MS = 30_000L" in service
  assert "START_NOT_STICKY" in service


def test_android_readmes_document_discovery_model_selection_and_power() -> None:
  index = (ANDROID_ROOT / "README.md").read_text(encoding="utf-8")
  readme_ko = (ANDROID_ROOT / "README.ko.md").read_text(encoding="utf-8")
  readme_en = (ANDROID_ROOT / "README.en.md").read_text(encoding="utf-8")

  assert "README.ko.md" in index
  assert "README.en.md" in index
  for text in (
    "Android 앱의 `YOLO ONNX 모델 선택` 버튼",
    "같은 사설 IPv4 `/24`",
    "Carrot 프레임 서명 `CAI1`",
    "ExternalAIPhoneIP",
    "YOLO 모델이나 NPU 세션을 열지 않고",
    "부팅 자동 시작은 하지 않습니다",
  ):
    assert text in readme_ko
  for text in (
    "Select it on the **Android app**",
    "local private IPv4 `/24`",
    "Carrot frame signature `CAI1`",
    "does not open the YOLO model or NPU session",
    "It does not start at boot",
  ):
    assert text in readme_en
