from pathlib import Path


OPENPILOT_ROOT = Path(__file__).resolve().parents[3]
ANDROID_ROOT = OPENPILOT_ROOT.parent / "tools" / "android_ai_client"
JAVA_ROOT = ANDROID_ROOT / "app" / "src" / "main" / "java" / "ai" / "carrotpilot" / "external"


def test_android_app_auto_discovers_on_launch_with_manual_fallback() -> None:
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")
  service = (JAVA_ROOT / "ExternalAIService.kt").read_text(encoding="utf-8")
  manifest = (ANDROID_ROOT / "app" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")

  assert "앱 실행 시 같은 망 자동 검색 및 시작" in activity
  assert "권장 모델 선택" in activity
  assert "RecommendedModels.ALL.map" in activity
  assert "선택 모델 삭제" in activity
  assert "YOLO 입력 크기 (320/416/640 · 권장 320)" in activity
  assert "다른 ONNX 파일 선택" in activity
  assert "동의 후 다운로드" in activity
  assert "autoConnect.isChecked && modelUri != null && !ExternalAIService.serviceActive" in activity
  assert "startClient(autoDiscover = true)" in activity
  assert "입력 IP로 시작" in activity
  assert "startClient(autoDiscover = false)" in activity
  assert "EXTRA_AUTO_DISCOVER" in activity
  assert "DeviceDiscovery.findHost(config.framePort, config.host)" in service
  assert "RECEIVE_BOOT_COMPLETED" not in manifest


def test_android_client_uses_dynamic_low_resolution_and_stage_metrics() -> None:
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")
  service = (JAVA_ROOT / "ExternalAIService.kt").read_text(encoding="utf-8")
  detector = (JAVA_ROOT / "YoloDetector.kt").read_text(encoding="utf-8")
  protocol = (JAVA_ROOT / "FrameProtocol.kt").read_text(encoding="utf-8")
  stats = (JAVA_ROOT / "PerformanceStats.kt").read_text(encoding="utf-8")
  decoder = (JAVA_ROOT / "ReusableJpegDecoder.kt").read_text(encoding="utf-8")

  assert "SUPPORTED_INPUT_SIZES = setOf(320, 416, 640)" in detector
  assert "?: dynamicInputSize" in detector
  assert "KEY_INPUT_SIZE, 320" in activity
  assert "EXTRA_INPUT_SIZE" in activity
  assert "NNAPIFlags.USE_FP16" in detector
  assert "NNAPIFlags.USE_NCHW" not in detector
  assert "preprocessMs" in detector
  assert "runtimeMs" in detector
  assert "postprocessMs" in detector
  assert "RollingPerformanceStats()" in service
  assert "p95PhoneTotalMs" in service
  assert "thermalStatusLabel" in service
  for field in ("decode_ms", "preprocess_ms", "runtime_ms", "postprocess_ms", "phone_total_ms", "input_width", "input_height"):
    assert f'put("{field}"' in protocol
  assert "capacity: Int = 120" in stats
  assert "percentile95" in stats
  assert "inBitmap = previous" in decoder
  assert "private val letterboxBitmap" in detector
  assert "private val pixels = IntArray" in detector


def test_android_client_prefers_verified_qnn_htp_before_fallbacks() -> None:
  gradle = (ANDROID_ROOT / "app" / "build.gradle.kts").read_text(encoding="utf-8")
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")
  detector = (JAVA_ROOT / "YoloDetector.kt").read_text(encoding="utf-8")

  assert 'versionName = "0.8.1"' in gradle
  assert 'providers.gradleProperty("carrotQnnEnabled")' in gradle
  assert 'implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3")' in gradle
  assert 'buildConfigField("boolean", "QNN_EP_INCLUDED"' in gradle
  assert "jniLibs.useLegacyPackaging = true" in gradle
  assert "BuildConfig.QNN_EP_INCLUDED" in activity
  assert "Qualcomm QNN/HTP 전체 그래프 → NNAPI → CPU" in activity

  qnn_setup = detector.split("private fun tryCreateQnnSession", 1)[1].split("private fun createBaseOptions", 1)[0]
  assert 'addConfigEntry("session.disable_cpu_ep_fallback", "1")' in qnn_setup
  assert 'setSymbolicDimensionValue("height", dynamicInputSize.toLong())' in qnn_setup
  assert 'setSymbolicDimensionValue("width", dynamicInputSize.toLong())' in qnn_setup
  assert '"backend_path" to "libQnnHtp.so"' in qnn_setup
  assert '"htp_performance_mode" to "sustained_high_performance"' in qnn_setup
  assert '"htp_graph_finalization_optimization_mode" to "3"' in qnn_setup
  assert '"offload_graph_io_quantization" to "0"' in qnn_setup
  assert 'backend = "onnxruntime-qnn"' in qnn_setup
  assert "createAndWarmSession(modelFile, qnnOptions)" in qnn_setup
  assert "candidate.run(mapOf(candidateInputName to input))" in detector
  assert detector.index("tryCreateQnnSession(modelFile)") < detector.index("nnapiOptions.addNnapi")
  assert 'input.shape[2] in longArrayOf(-1, 320, 416, 640)' in detector


def test_android_recommended_model_download_is_pinned_and_validated() -> None:
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")
  downloader = (JAVA_ROOT / "RecommendedModel.kt").read_text(encoding="utf-8")
  detector = (JAVA_ROOT / "YoloDetector.kt").read_text(encoding="utf-8")
  service = (JAVA_ROOT / "ExternalAIService.kt").read_text(encoding="utf-8")

  for name, size, sha256 in (
    ("yolo11n", "10_930_182L", "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b"),
    ("yolo11s", "38_051_729L", "21d6650c5097610c92c76ce5e4b717976059169eaea4962035b90c6a92c07a8f"),
    ("yolo11m", "80_673_621L", "8a37b5c53ff642831aa454156b548ec2cf2537827445385c3e1c1b276cb666a3"),
  ):
    assert f"/v8.4.0/{name}.onnx" in downloader
    assert f"expectedSize = {size}" in downloader
    assert f'expectedSha256 = "{sha256}"' in downloader
  assert "val ALL = listOf(YOLO11N, YOLO11S, YOLO11M)" in downloader
  assert 'require(sourceUrl.protocol == "https")' in downloader
  assert "connection.responseCode == HttpURLConnection.HTTP_OK" in downloader
  assert "received <= model.expectedSize" in downloader
  assert "actualSha256 == model.expectedSha256" in downloader
  assert "YoloDetector.validateModelFile(partial)" in downloader
  assert "StandardCopyOption.ATOMIC_MOVE" in downloader
  assert "Files.deleteIfExists(partial.toPath())" in downloader
  assert "AlertDialog.Builder(this)" in activity
  assert "Uri.fromFile(file)" in activity
  assert "if (activityStarted && autoConnect.isChecked) startClient(autoDiscover = true)" in activity
  assert "input.shape[0] in longArrayOf(-1, 1)" in detector
  assert "input.shape[2] in longArrayOf(-1, 320, 416, 640)" in detector
  assert "output.shape[0] in longArrayOf(-1, 1)" in detector
  assert "output.shape[2] == -1L || output.shape[2] > output.shape[1]" in detector
  assert "output.shape[1] == -1L || output.shape[1] > output.shape[2]" in detector
  assert "channelsFirst || channelsLast" in detector
  assert "uri.scheme == ContentResolver.SCHEME_FILE" in service
  assert "FileInputStream(File(requireNotNull(uri.path)" in service


def test_android_client_shows_live_model_fps_console_and_verified_npu_badge() -> None:
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")
  service = (JAVA_ROOT / "ExternalAIService.kt").read_text(encoding="utf-8")
  style = (ANDROID_ROOT / "app" / "src" / "main" / "res" / "values" / "styles.xml").read_text(encoding="utf-8")

  for text in ("실시간 프레임 추종", "VIDEO -- FPS", "AI -- FPS", "추종률 --%", "SKIP --/s", "실시간 객체 분석 콘솔"):
    assert text in activity
  assert "Color.rgb(247, 249, 252)" in activity
  assert "android:windowLightStatusBar\">true" in style
  assert "ACTION_ANALYSIS" in activity and "ACTION_METRICS" in activity
  assert "ANALYSIS_BROADCAST_INTERVAL_NS = 200_000_000L" in service
  assert '"onnxruntime-qnn" -> "eNPU"' in service
  assert '"onnxruntime-nnapi" -> "eACCEL"' in service
  assert 'else -> "eCPU"' in service
  for field in ("EXTRA_VIDEO_FPS", "EXTRA_AI_FPS", "EXTRA_FOLLOW_RATE", "EXTRA_SKIPPED_FPS"):
    assert field in service and field in activity
  assert "sourceWindowFirstTimestampNs" in service
  assert "sourceWindowLastTimestampNs" in service
  assert "(receivedCount - 1) / sourceSeconds" in service
  assert "Locale.getDefault().language != Locale.KOREAN.language" in service
  for field in ("frame=", "box=", "center=", "confidence * 100f"):
    assert field in service


def test_android_client_copies_safe_c3x_branch_ssh_command() -> None:
  activity = (JAVA_ROOT / "MainActivity.kt").read_text(encoding="utf-8")

  assert "C3X 브랜치 변경 SSH 명령 복사" in activity
  assert "ClipboardManager::class.java" in activity
  assert "ClipData.newPlainText" in activity
  assert 'SSH_USER = "comma"' in activity
  assert 'OPENPILOT_PATH = "/data/openpilot"' in activity
  assert 'DEPLOY_REPOSITORY_URL = "https://github.com/junghc11/openpilot.git"' in activity
  assert 'DEPLOY_BRANCH = "external-android-ai"' in activity
  assert "git switch --track -c" in activity
  assert "git pull --ff-only" in activity
  assert "git rev-parse --short HEAD" in activity
  assert "isValidIpv4Address(targetHost)" in activity
  assert "reset --hard" not in activity


def test_android_discovery_is_bounded_to_private_subnets_and_carrot_port() -> None:
  discovery = (JAVA_ROOT / "DeviceDiscovery.kt").read_text(encoding="utf-8")

  assert "ipv4.isSiteLocalAddress" in discovery
  assert "for (lastOctet in 1..254)" in discovery
  assert "MAX_SUBNETS = 2" in discovery
  assert "isCarrotFrameServer(candidate, framePort)" in discovery
  assert "socket.connect(InetSocketAddress(host, framePort), CONNECT_TIMEOUT_MS)" in discovery
  assert "SUPPORTED_FRAME_MAGICS.any(magic::contentEquals)" in discovery
  assert "'2'.code.toByte()" in discovery
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


def test_android_client_decodes_local_h264_with_mediacodec_and_keeps_jpeg_fallback() -> None:
  service = (JAVA_ROOT / "ExternalAIService.kt").read_text(encoding="utf-8")
  protocol = (JAVA_ROOT / "FrameProtocol.kt").read_text(encoding="utf-8")
  decoder = (JAVA_ROOT / "ReusableH264Decoder.kt").read_text(encoding="utf-8")

  assert 'ENCODING_H264 = "h264"' in protocol
  assert "h264Magic" in protocol
  assert "codec_config_size" in protocol
  assert "ReusableH264Decoder().use" in service
  assert "ReusableJpegDecoder().use" in service
  assert "MediaCodec.createDecoderByType" in decoder
  assert "MediaFormat.KEY_LOW_LATENCY" in decoder
  assert "ImageFormat.YUV_420_888" in decoder
  assert "BUFFER_FLAG_KEY_FRAME" in decoder
  assert "H.264 HW" in service


def test_android_readmes_document_discovery_model_selection_and_power() -> None:
  index = (ANDROID_ROOT / "README.md").read_text(encoding="utf-8")
  readme_ko = (ANDROID_ROOT / "README.ko.md").read_text(encoding="utf-8")
  readme_en = (ANDROID_ROOT / "README.en.md").read_text(encoding="utf-8")

  assert "README.ko.md" in index
  assert "README.en.md" in index
  for text in (
    "앱의 **권장 모델 다운로드**",
    "첫 시험 권장 모델은 동적 입력 `YOLO11n Detection`",
    "기본 320은 성능 우선",
    "전화 처리 평균과 p95",
    "NCHW 강제 옵션",
    "QNN/HTP 전체 그래프",
    "session.disable_cpu_ep_fallback",
    "carrotQnnEnabled=false",
    "공식 `ultralytics/assets` v8.4.0 Release",
    "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b",
    "AGPL-3.0 또는 Enterprise",
    "nms=False dynamic=False batch=1",
    "같은 사설 IPv4 `/24`",
    "JPEG `CAI1` 또는 H.264 `CAI2`",
    "ExternalAIPhoneIP",
    "YOLO 모델이나 NPU 세션을 열지 않고",
    "부팅 자동 시작은 하지 않습니다",
  ):
    assert text in readme_ko
  for text in (
    "Press **권장 모델 다운로드**",
    "recommended first-test model is dynamic-input YOLO11n Detection",
    "Use 320 for performance-first testing",
    "average and p95 phone time",
    "does not force the potentially slower NCHW option",
    "Full-graph Qualcomm QNN/HTP",
    "session.disable_cpu_ep_fallback",
    "carrotQnnEnabled=false",
    "official `ultralytics/assets` v8.4.0 Release",
    "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b",
    "AGPL-3.0 or Enterprise",
    "nms=False dynamic=False batch=1",
    "local private IPv4 `/24`",
    "JPEG `CAI1` or H.264 `CAI2`",
    "does not open the YOLO model or NPU session",
    "It does not start at boot",
  ):
    assert text in readme_en
