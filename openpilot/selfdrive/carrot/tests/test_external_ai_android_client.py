from pathlib import Path


OPENPILOT_ROOT = Path(__file__).resolve().parents[3]
ANDROID_ROOT = OPENPILOT_ROOT.parent / "tools" / "android_ai_client"
JAVA_ROOT = ANDROID_ROOT / "app" / "src" / "main" / "java" / "ai" / "carrotpilot" / "external"


def test_android_app_requires_explicit_device_pairing() -> None:
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")
  manifest = (ANDROID_ROOT / "app" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")

  assert "같은 Wi-Fi만으로 자동 연결되지 않습니다" in activity
  assert "기기 IP (C3/C3X/C4)" in activity
  assert "setOnClickListener { startClient() }" in activity
  assert "RECEIVE_BOOT_COMPLETED" not in manifest


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
  assert "START_NOT_STICKY" in service


def test_android_readme_documents_network_and_power_requirements() -> None:
  readme = (ANDROID_ROOT / "README.md").read_text(encoding="utf-8")

  for text in (
    "does not start or discover the connection automatically",
    "ExternalAIPhoneIP",
    "phone address",
    "CarrotPilot device address",
    "runs only while the manager reports `started`",
    "blue `eCPU`",
    "1, 2, 4, 8, 16, then 30-second capped backoff",
    "For zero background use, press **중지**",
  ):
    assert text in readme
