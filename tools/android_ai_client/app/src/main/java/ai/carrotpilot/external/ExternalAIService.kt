package ai.carrotpilot.external

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.net.wifi.WifiManager
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.roundToInt

class ExternalAIService : Service() {
  @Volatile private var runToken: AtomicBoolean? = null
  private var worker: Thread? = null
  @Volatile private var frameSocket: Socket? = null
  private var wakeLock: PowerManager.WakeLock? = null
  private var wifiLock: WifiManager.WifiLock? = null
  private var lastStatus = "중지됨"
  private val analysisTimeFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())
  private val captureRequest = AtomicReference<String?>(null)
  private val benchmarkRequest = AtomicBoolean(false)

  override fun onCreate() {
    super.onCreate()
    serviceActive = true
    createNotificationChannel()
  }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    when (intent?.action) {
      ACTION_STOP -> stopClient()
      ACTION_CAPTURE_SAMPLE -> {
        val captureType = intent.getStringExtra(EXTRA_CAPTURE_TYPE)
        if (captureType in setOf(DiagnosticSampleSaver.TYPE_FALSE_POSITIVE, DiagnosticSampleSaver.TYPE_MISSED_DETECTION)) {
          captureRequest.set(captureType)
          publishCaptureStatus("다음 분석 프레임을 저장합니다.")
        }
      }
      ACTION_BENCHMARK_MODELS -> {
        benchmarkRequest.set(true)
        publishBenchmarkResult("연결된 실영상의 동일 프레임으로 설치 모델 비교를 준비합니다.")
      }
      ACTION_START -> {
        val config = ServiceConfig.fromIntent(intent)
        startAsForeground("모델 준비 중")
        startClient(config)
      }
    }
    return START_NOT_STICKY
  }

  override fun onDestroy() {
    stopClient()
    serviceActive = false
    super.onDestroy()
  }

  override fun onBind(intent: Intent?): IBinder? = null

  private fun startClient(config: ServiceConfig) {
    stopWorkerOnly()
    clientConnected = false
    val token = AtomicBoolean(true)
    runToken = token
    worker = Thread({ runClient(config, token) }, "carrot-external-ai").apply { start() }
  }

  private fun runClient(config: ServiceConfig, token: AtomicBoolean) {
    val modelName = config.modelUri.lastPathSegment ?: "yolo.onnx"
    val modelSpec = RecommendedModels.findByUri(this, config.modelUri)
    val modelDisplayName = modelSpec?.displayName ?: modelName
    var retryDelayMs = MIN_RETRY_DELAY_MS
    var discoveryRetryDelayMs = MIN_DISCOVERY_RETRY_DELAY_MS
    var detector: YoloDetector? = null
    try {
      updateStatus("C3X 연결 전 AI 가속 자동 선택 중\n동일 입력으로 QNN·NNAPI·CPU 비교")
      detector = YoloDetector(
        copyModelToCache(config.modelUri),
        config.threshold,
        config.inputSize,
        classThresholds = config.classThresholds,
        tryQnn = modelSpec?.qnnOptimized != false,
        qnnSkipReason = if (modelSpec?.qnnOptimized == false) {
          "QNN 건너뜀: Dynamic FP32 CPU 호환 모델"
        } else {
          ""
        },
      )
      publishMetrics(
        modelDisplayName = modelDisplayName,
        receiveFps = 0.0,
        inferenceFps = 0.0,
        backend = detector.backend,
        backendLabel = detector.backendLabel,
      )
      val preflightSummary = "기기 ${deviceSummary()}\n${detector.backendLabel}"
      updateStatus(
        "가속 자동 선택 완료: ${acceleratorBadge(detector.backend)}\n" +
          "$preflightSummary\nC3X 연결 없이 동일 입력 p50/p95 비교",
      )
      val preflightBadge = acceleratorBadge(detector.backend)
      while (token.get()) {
        val targetHost = if (config.autoDiscover) {
          updateStatus(
            "같은 사설망에서 CarrotPilot 기기 검색 중\n" +
              "가속 사전 점검 $preflightBadge · TCP ${config.framePort} / CAI1·CAI2 확인\n" +
              preflightSummary,
          )
          DeviceDiscovery.findHost(config.framePort, config.host) { token.get() }
        } else {
          config.host
        }
        if (targetHost == null) {
          if (token.get()) {
            updateStatus(
              "기기를 찾지 못함 · 가속 $preflightBadge\n" +
                "$preflightSummary\n${discoveryRetryDelayMs / 1_000}초 후 같은 망 다시 검색",
            )
            SystemClock.sleep(discoveryRetryDelayMs)
            discoveryRetryDelayMs = (discoveryRetryDelayMs * 2).coerceAtMost(MAX_DISCOVERY_RETRY_DELAY_MS)
          }
          continue
        }
        discoveryRetryDelayMs = MIN_DISCOVERY_RETRY_DELAY_MS
        if (config.autoDiscover) {
          getSharedPreferences(PREFERENCES_NAME, MODE_PRIVATE).edit().putString(PREFERENCE_HOST, targetHost).apply()
        }

        val activeDetector = checkNotNull(detector)

        try {
          updateStatus("기기 연결 중: $targetHost:${config.framePort}", targetHost)
          Socket().use { socket ->
            frameSocket = socket
            socket.tcpNoDelay = true
            socket.soTimeout = 5_000
            socket.connect(InetSocketAddress(targetHost, config.framePort), 3_000)
            clientConnected = true
            acquirePerformanceLocks()
            retryDelayMs = MIN_RETRY_DELAY_MS
            DataInputStream(socket.getInputStream().buffered()).use { input ->
              DatagramSocket().use { resultSocket ->
                processFrames(config, targetHost, activeDetector, modelName, modelDisplayName, input, resultSocket, token)
              }
            }
          }
        } catch (error: Exception) {
          if (runToken === token) releasePerformanceLocks()
          if (token.get()) {
            updateStatus("저전력 재연결 대기 ${retryDelayMs / 1_000}초\n${error.javaClass.simpleName}: ${error.message}")
            SystemClock.sleep(retryDelayMs)
            retryDelayMs = (retryDelayMs * 2).coerceAtMost(MAX_RETRY_DELAY_MS)
          }
        } finally {
          clientConnected = false
          if (runToken === token) {
            frameSocket = null
            releasePerformanceLocks()
          }
        }
      }
    } catch (error: Exception) {
      updateStatus("YOLO 초기화 실패\n${error.message}")
    } finally {
      detector?.close()
      clientConnected = false
      token.set(false)
      if (runToken === token) {
        runToken = null
        releasePerformanceLocks()
      }
    }
  }

  private fun processFrames(
    config: ServiceConfig,
    targetHost: String,
    detector: YoloDetector,
    modelName: String,
    modelDisplayName: String,
    input: DataInputStream,
    resultSocket: DatagramSocket,
    token: AtomicBoolean,
  ) {
    val resultAddress = InetAddress.getByName(targetHost)
    var effectiveFps = config.targetFps
    var performanceMode = "normal"
    var targetIntervalNs = 1_000_000_000L / effectiveFps
    var lastInferenceStartNs = 0L
    var receivedCount = 0
    var inferenceCount = 0
    var statusWindowStartNs = SystemClock.elapsedRealtimeNanos()
    var sourceWindowFirstTimestampNs = 0L
    var sourceWindowLastTimestampNs = 0L
    val performanceStats = RollingPerformanceStats()
    var lastObjects = 0
    var lastTransport = "대기"
    var previousEncoding = ""
    var lastAnalysisBroadcastNs = 0L
    val tracker = ObjectTracker()
    val trafficLightStabilizer = TrafficLightStateStabilizer()
    val performanceGovernor = AdaptivePerformanceGovernor(config.targetFps)
    ReusableJpegDecoder().use { jpegDecoder ->
      ReusableH264Decoder().use { h264Decoder ->
        while (token.get()) {
          val receivedFrame = FrameProtocol.readFrame(input)
          receivedCount++
          if (sourceWindowFirstTimestampNs == 0L) sourceWindowFirstTimestampNs = receivedFrame.sourceTimestampNs
          sourceWindowLastTimestampNs = receivedFrame.sourceTimestampNs
          if (receivedFrame.encoding != previousEncoding) {
            if (receivedFrame.encoding == FrameProtocol.ENCODING_JPEG) h264Decoder.reset()
            previousEncoding = receivedFrame.encoding
          }
          val nowNs = SystemClock.elapsedRealtimeNanos()
          val wantInference = nowNs - lastInferenceStartNs >= targetIntervalNs
          val decodeStartNs = SystemClock.elapsedRealtimeNanos()
          val inferenceFrame: C3XFrame
          val bitmap = when (receivedFrame.encoding) {
            FrameProtocol.ENCODING_JPEG -> {
              if (!wantInference) continue
              inferenceFrame = receivedFrame
              jpegDecoder.decode(receivedFrame.data)
            }
            FrameProtocol.ENCODING_H264 -> {
              val decoded = h264Decoder.decode(receivedFrame, wantInference) ?: continue
              inferenceFrame = decoded.frame
              decoded.bitmap
            }
            else -> error("지원하지 않는 영상 인코딩: ${receivedFrame.encoding}")
          }
          val decodeEndNs = SystemClock.elapsedRealtimeNanos()
          val inferenceStartNs = decodeEndNs
          val rawResult = detector.detect(bitmap)
          val trackedDetections = tracker.update(rawResult.detections)
          val stableTrafficLight = trafficLightStabilizer.update(
            rawState = rawResult.trafficLightState,
            rawConfidence = rawResult.trafficLightConfidence,
            trafficLightDetected = trackedDetections.any { it.className == "traffic light" },
          )
          val detectionResult = rawResult.copy(
            detections = trackedDetections,
            trafficLightState = stableTrafficLight.state,
            trafficLightConfidence = stableTrafficLight.confidence,
          )
          val inferenceEndNs = SystemClock.elapsedRealtimeNanos()
          lastInferenceStartNs = inferenceStartNs
          lastTransport = if (inferenceFrame.encoding == FrameProtocol.ENCODING_H264) "H.264 HW" else "JPEG"
          val performance = FramePerformance(
            decodeMs = nanosToMillis(decodeEndNs - decodeStartNs),
            preprocessMs = detectionResult.preprocessMs,
            runtimeMs = detectionResult.runtimeMs,
            postprocessMs = detectionResult.postprocessMs,
            phoneTotalMs = nanosToMillis(inferenceEndNs - inferenceFrame.phoneReceiveTimestampNs),
            effectiveFps = effectiveFps,
            performanceMode = performanceMode,
          )
          performanceStats.add(performance)
          inferenceCount++
          lastObjects = detectionResult.detections.size
          FrameProtocol.sendResult(
            resultSocket,
            resultAddress,
            config.resultPort,
            inferenceFrame,
            inferenceStartNs,
            inferenceEndNs,
            detectionResult,
            performance,
            modelName,
            detector.backend,
          )
          captureRequest.getAndSet(null)?.let { captureType ->
            try {
              val saved = DiagnosticSampleSaver.save(
                context = this,
                bitmap = bitmap,
                captureType = captureType,
                frame = inferenceFrame,
                result = detectionResult,
                performance = performance,
                modelName = modelDisplayName,
                backend = detector.backend,
              )
              publishCaptureStatus("저장 완료: ${saved.baseName}\n사진/Pictures 및 JSON/Download의 CarrotExternalAI 폴더")
            } catch (error: Exception) {
              publishCaptureStatus("저장 실패: ${error.message}")
            }
          }
          if (benchmarkRequest.getAndSet(false)) {
            publishBenchmarkResult(runInstalledModelBenchmark(bitmap, config, detector, modelDisplayName))
          }
          if (inferenceEndNs - lastAnalysisBroadcastNs >= ANALYSIS_BROADCAST_INTERVAL_NS) {
            publishAnalysisLog(buildAnalysisLog(
              modelDisplayName = modelDisplayName,
              frame = inferenceFrame,
              detectionResult = detectionResult,
              performance = performance,
              backend = detector.backend,
              transportLabel = lastTransport,
            ))
            lastAnalysisBroadcastNs = inferenceEndNs
          }

          val windowNs = inferenceEndNs - statusWindowStartNs
          if (windowNs >= 1_000_000_000L) {
            val seconds = windowNs / 1_000_000_000.0
            val sourceSeconds = (sourceWindowLastTimestampNs - sourceWindowFirstTimestampNs) / 1_000_000_000.0
            val receiveFps = if (receivedCount > 1 && sourceSeconds > 0.0) {
              (receivedCount - 1) / sourceSeconds
            } else {
              receivedCount / seconds
            }
            val inferenceFps = inferenceCount / seconds
            val performanceSummary = performanceStats.summary()
            val targetAttainment = if (effectiveFps > 0) (inferenceFps / effectiveFps * 100.0).coerceIn(0.0, 100.0) else 0.0
            val thermalStatus = getSystemService(PowerManager::class.java).currentThermalStatus
            val decision = performanceGovernor.update(
              p95Ms = performanceSummary.p95PhoneTotalMs,
              followRate = targetAttainment,
              thermalStatus = thermalStatus,
            )
            effectiveFps = decision.effectiveFps
            performanceMode = decision.mode
            targetIntervalNs = 1_000_000_000L / effectiveFps
            updateStatus(buildStatus(
              config = config,
              targetHost = targetHost,
              receiveFps = receiveFps,
              inferenceFps = inferenceFps,
              performance = performanceSummary,
              inputWidth = detector.inputWidth,
              inputHeight = detector.inputHeight,
              objectCount = lastObjects,
              backendLabel = detector.backendLabel,
              transportLabel = lastTransport,
              sceneMode = detectionResult.sceneMode,
              sceneBrightness = detectionResult.sceneBrightness,
              effectiveFps = effectiveFps,
              performanceMode = performanceMode,
            ))
            publishMetrics(
              modelDisplayName = modelDisplayName,
              receiveFps = receiveFps,
              inferenceFps = inferenceFps,
              backend = detector.backend,
              backendLabel = detector.backendLabel,
              effectiveFps = effectiveFps,
              performanceMode = performanceMode,
            )
            receivedCount = 0
            inferenceCount = 0
            sourceWindowFirstTimestampNs = 0L
            sourceWindowLastTimestampNs = 0L
            statusWindowStartNs = inferenceEndNs
          }
        }
      }
    }
  }

  private fun buildStatus(
    config: ServiceConfig,
    targetHost: String,
    receiveFps: Double,
    inferenceFps: Double,
    performance: PerformanceSummary,
    inputWidth: Int,
    inputHeight: Int,
    objectCount: Int,
    backendLabel: String,
    transportLabel: String,
    sceneMode: String,
    sceneBrightness: Float,
    effectiveFps: Int,
    performanceMode: String,
  ): String {
    val battery = registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val batteryTemp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)?.div(10.0) ?: 0.0
    val thermalStatus = getSystemService(PowerManager::class.java).currentThermalStatus
    return "연결됨: $targetHost:${config.framePort}\n" +
      "입력 ${inputWidth}×${inputHeight} · 수신 ${"%.1f".format(receiveFps)} · 처리 ${"%.1f".format(inferenceFps)} FPS\n" +
      "장면 ${localizedSceneMode(sceneMode)}(${"%.0f".format(sceneBrightness * 100f)}%) · 자동 성능 ${localizedPerformanceMode(performanceMode)} ${effectiveFps}/${config.targetFps} FPS\n" +
      "폰 처리 평균 ${"%.1f".format(performance.averagePhoneTotalMs)} · p95 ${"%.1f".format(performance.p95PhoneTotalMs)} ms\n" +
      "$transportLabel 디코드 ${"%.1f".format(performance.averageDecodeMs)} · 전처리 ${"%.1f".format(performance.averagePreprocessMs)} · ORT ${"%.1f".format(performance.averageRuntimeMs)} · 후처리 ${"%.1f".format(performance.averagePostprocessMs)} ms\n" +
      "객체 ${objectCount}개 · 표본 ${performance.samples}개\n" +
      "기기 ${deviceSummary()}\n" +
      "백엔드 $backendLabel · 배터리 ${"%.1f".format(batteryTemp)}°C\n" +
      "열 상태 ${thermalStatusLabel(thermalStatus)}($thermalStatus) · 총 지연은 C3X에서 측정"
  }

  private fun deviceSummary(): String {
    val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE
    return "${Build.MANUFACTURER} ${Build.MODEL} · SoC ${soc.ifBlank { "알 수 없음" }}"
  }

  private fun buildAnalysisLog(
    modelDisplayName: String,
    frame: C3XFrame,
    detectionResult: DetectionResult,
    performance: FramePerformance,
    backend: String,
    transportLabel: String,
  ): String {
    val detections = detectionResult.detections.sortedByDescending(Detection::confidence)
    val trafficLightDetected = detections.any { it.className == "traffic light" }
    val trafficLightLine = if (trafficLightDetected) {
      val localizedState = localizedTrafficLightState(detectionResult.trafficLightState)
      val confidence = if (detectionResult.trafficLightState == TrafficLightColorResult.UNKNOWN) {
        ""
      } else {
        " (${"%.0f".format(detectionResult.trafficLightConfidence * 100f)}%)"
      }
      "\n  신호등 색상: $localizedState$confidence [${detectionResult.trafficLightState}]"
    } else {
      ""
    }
    val header = "${analysisTimeFormat.format(Date())} | frame=${frame.frameId} | ${detections.size} objects\n" +
      "$modelDisplayName | ${acceleratorBadge(backend)} | $transportLabel | ${localizedSceneMode(detectionResult.sceneMode)} " +
      "| ${localizedPerformanceMode(performance.performanceMode)} ${performance.effectiveFps} FPS | total ${"%.1f".format(performance.phoneTotalMs)} ms" +
      trafficLightLine
    if (detections.isEmpty()) return "$header\n  객체 없음"
    val objects = detections.take(MAX_CONSOLE_OBJECTS).mapIndexed { index, detection ->
      val x1 = (detection.x1 * frame.width).roundToInt().coerceIn(0, frame.width)
      val y1 = (detection.y1 * frame.height).roundToInt().coerceIn(0, frame.height)
      val x2 = (detection.x2 * frame.width).roundToInt().coerceIn(0, frame.width)
      val y2 = (detection.y2 * frame.height).roundToInt().coerceIn(0, frame.height)
      val centerX = (x1 + x2) / 2
      val centerY = (y1 + y2) / 2
      "  ${index + 1}. #${detection.trackId} ${localizedObjectName(detection.className)}(${detection.className}) " +
        "${"%.1f".format(detection.confidence * 100f)}% | box=($x1,$y1)-($x2,$y2) | center=($centerX,$centerY)"
    }
    val omitted = detections.size - objects.size
    return buildString {
      append(header)
      append('\n')
      append(objects.joinToString("\n"))
      if (omitted > 0) append("\n  ... 외 ${omitted}개")
    }
  }

  private fun localizedObjectName(className: String): String {
    if (Locale.getDefault().language != Locale.KOREAN.language) return className
    return when (className) {
      "person" -> "사람"
      "bicycle" -> "자전거"
      "car" -> "차량"
      "motorcycle" -> "오토바이"
      "bus" -> "버스"
      "truck" -> "트럭"
      "traffic light" -> "신호등"
      "stop sign" -> "정지표지판"
      else -> className
    }
  }

  private fun localizedTrafficLightState(state: String): String {
    if (Locale.getDefault().language != Locale.KOREAN.language) return state
    return when (state) {
      TrafficLightColorResult.RED -> "빨강"
      TrafficLightColorResult.YELLOW -> "노랑"
      TrafficLightColorResult.GREEN -> "초록"
      else -> "미확인"
    }
  }

  private fun localizedSceneMode(mode: String): String = when (mode) {
    LowLightPolicy.NIGHT -> "야간 보정"
    else -> "주간"
  }

  private fun localizedPerformanceMode(mode: String): String = when (mode) {
    "thermal" -> "열 보호"
    "reduced" -> "부하 조절"
    else -> "정상"
  }

  private fun acceleratorBadge(backend: String): String = when (backend) {
    "onnxruntime-qnn",
    "onnxruntime-qnn-mixed",
    "onnxruntime-qnn-mixed-benchmarked",
    "onnxruntime-nnapi" -> "eNPU"
    else -> "eCPU"
  }

  private fun publishAnalysisLog(message: String) {
    sendBroadcast(Intent(ACTION_ANALYSIS).setPackage(packageName).putExtra(EXTRA_ANALYSIS_LOG, message))
  }

  private fun publishMetrics(
    modelDisplayName: String,
    receiveFps: Double,
    inferenceFps: Double,
    backend: String,
    backendLabel: String,
    effectiveFps: Int = 0,
    performanceMode: String = "normal",
  ) {
    val followRate = if (receiveFps > 0.0) (inferenceFps / receiveFps * 100.0).coerceIn(0.0, 100.0) else 0.0
    val skippedFps = (receiveFps - inferenceFps).coerceAtLeast(0.0)
    sendBroadcast(Intent(ACTION_METRICS).setPackage(packageName).apply {
      putExtra(EXTRA_MODEL_NAME, modelDisplayName)
      putExtra(EXTRA_VIDEO_FPS, receiveFps)
      putExtra(EXTRA_AI_FPS, inferenceFps)
      putExtra(EXTRA_FOLLOW_RATE, followRate)
      putExtra(EXTRA_SKIPPED_FPS, skippedFps)
      putExtra(EXTRA_ACCELERATOR_BADGE, acceleratorBadge(backend))
      putExtra(EXTRA_ACCELERATOR_DETAIL, "기기 ${deviceSummary()}\n$backendLabel")
      putExtra(EXTRA_EFFECTIVE_FPS, effectiveFps)
      putExtra(EXTRA_PERFORMANCE_MODE, performanceMode)
    })
  }

  private fun runInstalledModelBenchmark(
    bitmap: android.graphics.Bitmap,
    config: ServiceConfig,
    activeDetector: YoloDetector,
    activeModelName: String,
  ): String {
    val installed = RecommendedModels.ALL.filter { RecommendedModels.isInstalled(this, it) }
    if (installed.isEmpty()) return "비교할 설치 모델이 없습니다."
    val startedTemperature = batteryTemperatureC()
    val results = mutableListOf<ModelBenchmarkResult>()
    installed.forEachIndexed { index, model ->
      publishBenchmarkResult("모델 비교 ${index + 1}/${installed.size}: ${model.displayName}")
      var candidate: YoloDetector? = null
      try {
        val detector = if (model.displayName == activeModelName) activeDetector else {
          YoloDetector(
            modelFile = RecommendedModels.installedFile(this, model),
            confidenceThreshold = config.threshold,
            requestedInputSize = model.fixedInputSize ?: config.inputSize,
            classThresholds = config.classThresholds,
            tryQnn = model.qnnOptimized,
            qnnSkipReason = if (model.qnnOptimized) "" else "Dynamic FP32 CPU 호환 모델",
          ).also { candidate = it }
        }
        detector.detect(bitmap)
        val samples = ArrayList<Double>(MODEL_BENCHMARK_RUNS)
        var objectTotal = 0
        repeat(MODEL_BENCHMARK_RUNS) {
          val startNs = SystemClock.elapsedRealtimeNanos()
          val result = detector.detect(bitmap)
          samples += nanosToMillis(SystemClock.elapsedRealtimeNanos() - startNs)
          objectTotal += result.detections.size
        }
        results += ModelBenchmarkResult(
          modelName = model.displayName,
          backend = acceleratorBadge(detector.backend),
          p95Ms = samples.maxOrNull() ?: 0.0,
          averageMs = samples.average(),
          averageObjects = objectTotal.toDouble() / MODEL_BENCHMARK_RUNS,
        )
      } catch (error: Exception) {
        results += ModelBenchmarkResult(model.displayName, "실패", Double.POSITIVE_INFINITY, 0.0, 0.0, error.message)
      } finally {
        candidate?.close()
      }
    }
    val successful = results.filter { it.p95Ms.isFinite() }
    val maximumObjects = successful.maxOfOrNull { it.averageObjects } ?: 0.0
    val best = successful
      .filter { maximumObjects <= 0.0 || it.averageObjects >= maximumObjects * 0.60 }
      .minByOrNull { it.p95Ms }
    val temperatureDelta = batteryTemperatureC() - startedTemperature
    return buildString {
      append("동일 실영상 프레임 · 각 ${MODEL_BENCHMARK_RUNS}회 · 온도 변화 ${"%+.1f".format(temperatureDelta)}°C")
      results.forEach { result ->
        append("\n${result.modelName}: ")
        if (!result.p95Ms.isFinite()) append("실패 · ${result.error ?: "알 수 없음"}") else {
          append("${result.backend} · ${"%.1f".format(1_000.0 / result.averageMs)} FPS · p95 ${"%.1f".format(result.p95Ms)} ms · 객체 ${"%.1f".format(result.averageObjects)}")
        }
      }
      best?.let { append("\n자동 추천: ${it.modelName} (최대 탐지 수의 60% 이상 중 최저 p95)") }
    }
  }

  private fun batteryTemperatureC(): Double {
    val battery = registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    return battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)?.div(10.0) ?: 0.0
  }

  private fun publishBenchmarkResult(message: String) {
    sendBroadcast(Intent(ACTION_BENCHMARK_RESULT).setPackage(packageName).putExtra(EXTRA_BENCHMARK_RESULT, message))
  }

  private fun publishCaptureStatus(message: String) {
    sendBroadcast(Intent(ACTION_CAPTURE_RESULT).setPackage(packageName).putExtra(EXTRA_CAPTURE_RESULT, message))
  }

  private fun thermalStatusLabel(status: Int): String = when (status) {
    PowerManager.THERMAL_STATUS_NONE -> "정상"
    PowerManager.THERMAL_STATUS_LIGHT -> "약간 뜨거움"
    PowerManager.THERMAL_STATUS_MODERATE -> "성능 저하 가능"
    PowerManager.THERMAL_STATUS_SEVERE -> "성능 제한"
    PowerManager.THERMAL_STATUS_CRITICAL -> "위험"
    PowerManager.THERMAL_STATUS_EMERGENCY -> "긴급"
    PowerManager.THERMAL_STATUS_SHUTDOWN -> "종료 임박"
    else -> "알 수 없음"
  }

  private fun nanosToMillis(nanos: Long): Double = nanos / 1_000_000.0

  private fun copyModelToCache(uri: Uri): File {
    val modelFile = File(cacheDir, "selected-yolo.onnx")
    val source = if (uri.scheme == ContentResolver.SCHEME_FILE) {
      FileInputStream(File(requireNotNull(uri.path) { "내부 모델 경로가 없습니다." }))
    } else {
      contentResolver.openInputStream(uri)
    }
    source.use { input ->
      requireNotNull(input) { "선택한 모델을 읽을 수 없습니다." }
      FileOutputStream(modelFile).use { output -> input.copyTo(output) }
    }
    require(modelFile.length() in 1..MAX_MODEL_BYTES) { "모델 크기는 1바이트~256MB 범위여야 합니다." }
    return modelFile
  }

  private fun startAsForeground(message: String) {
    val notification = buildNotification(message)
    startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
  }

  private fun updateStatus(message: String, discoveredHost: String? = null) {
    lastStatus = message
    (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIFICATION_ID, buildNotification(message))
    sendBroadcast(Intent(ACTION_STATUS).setPackage(packageName).putExtra(EXTRA_STATUS, message).apply {
      discoveredHost?.let { putExtra(EXTRA_DISCOVERED_HOST, it) }
    })
  }

  private fun buildNotification(message: String): Notification {
    val openIntent = PendingIntent.getActivity(
      this,
      0,
      Intent(this, MainActivity::class.java),
      PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
    val stopIntent = PendingIntent.getService(
      this,
      1,
      Intent(this, ExternalAIService::class.java).setAction(ACTION_STOP),
      PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
    return Notification.Builder(this, CHANNEL_ID)
      .setSmallIcon(android.R.drawable.ic_menu_view)
      .setContentTitle("Carrot External AI 실행 중")
      .setContentText(message.lineSequence().firstOrNull() ?: message)
      .setStyle(Notification.BigTextStyle().bigText(message))
      .setContentIntent(openIntent)
      .addAction(Notification.Action.Builder(null, "중지", stopIntent).build())
      .setOngoing(true)
      .build()
  }

  private fun createNotificationChannel() {
    val channel = NotificationChannel(CHANNEL_ID, "External AI", NotificationManager.IMPORTANCE_LOW).apply {
      description = "C3/C3X/C4 영상 수신 및 YOLO 추론 상태"
    }
    (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(channel)
  }

  private fun acquirePerformanceLocks() {
    if (wakeLock?.isHeld != true) {
      wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
        .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:ExternalAI")
        .apply { acquire(6 * 60 * 60 * 1_000L) }
    }
    if (wifiLock?.isHeld != true) {
      @Suppress("DEPRECATION")
      wifiLock = (applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager)
        .createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "$packageName:ExternalAI")
        .apply { acquire() }
    }
  }

  private fun releasePerformanceLocks() {
    wakeLock?.takeIf { it.isHeld }?.release()
    wifiLock?.takeIf { it.isHeld }?.release()
    wakeLock = null
    wifiLock = null
  }

  private fun stopWorkerOnly() {
    runToken?.set(false)
    clientConnected = false
    try {
      frameSocket?.close()
    } catch (_: Exception) {
    }
    worker?.interrupt()
    if (worker !== Thread.currentThread()) {
      worker?.join(2_000)
    }
    worker = null
    runToken = null
    releasePerformanceLocks()
  }

  private fun stopClient() {
    stopWorkerOnly()
    lastStatus = "중지됨"
    sendBroadcast(Intent(ACTION_STATUS).setPackage(packageName).putExtra(EXTRA_STATUS, lastStatus))
    stopForeground(STOP_FOREGROUND_REMOVE)
    stopSelf()
  }

  companion object {
    const val ACTION_START = "ai.carrotpilot.external.START"
    const val ACTION_STOP = "ai.carrotpilot.external.STOP"
    const val ACTION_STATUS = "ai.carrotpilot.external.STATUS"
    const val ACTION_ANALYSIS = "ai.carrotpilot.external.ANALYSIS"
    const val ACTION_METRICS = "ai.carrotpilot.external.METRICS"
    const val ACTION_CAPTURE_SAMPLE = "ai.carrotpilot.external.CAPTURE_SAMPLE"
    const val ACTION_CAPTURE_RESULT = "ai.carrotpilot.external.CAPTURE_RESULT"
    const val ACTION_BENCHMARK_MODELS = "ai.carrotpilot.external.BENCHMARK_MODELS"
    const val ACTION_BENCHMARK_RESULT = "ai.carrotpilot.external.BENCHMARK_RESULT"
    const val EXTRA_HOST = "host"
    const val EXTRA_FRAME_PORT = "frame_port"
    const val EXTRA_RESULT_PORT = "result_port"
    const val EXTRA_THRESHOLD = "threshold"
    const val EXTRA_TARGET_FPS = "target_fps"
    const val EXTRA_INPUT_SIZE = "input_size"
    const val EXTRA_MODEL_URI = "model_uri"
    const val EXTRA_AUTO_DISCOVER = "auto_discover"
    const val EXTRA_CLASS_THRESHOLDS = "class_thresholds"
    const val EXTRA_CAPTURE_TYPE = "capture_type"
    const val EXTRA_STATUS = "status"
    const val EXTRA_DISCOVERED_HOST = "discovered_host"
    const val EXTRA_ANALYSIS_LOG = "analysis_log"
    const val EXTRA_MODEL_NAME = "model_name"
    const val EXTRA_VIDEO_FPS = "video_fps"
    const val EXTRA_AI_FPS = "ai_fps"
    const val EXTRA_FOLLOW_RATE = "follow_rate"
    const val EXTRA_SKIPPED_FPS = "skipped_fps"
    const val EXTRA_ACCELERATOR_BADGE = "accelerator_badge"
    const val EXTRA_ACCELERATOR_DETAIL = "accelerator_detail"
    const val EXTRA_EFFECTIVE_FPS = "effective_fps"
    const val EXTRA_PERFORMANCE_MODE = "performance_mode"
    const val EXTRA_CAPTURE_RESULT = "capture_result"
    const val EXTRA_BENCHMARK_RESULT = "benchmark_result"
    @Volatile var serviceActive = false
      private set
    @Volatile var clientConnected = false
      private set
    private const val CHANNEL_ID = "external_ai"
    private const val NOTIFICATION_ID = 7724
    private const val MAX_MODEL_BYTES = 256L * 1024L * 1024L
    private const val MIN_RETRY_DELAY_MS = 1_000L
    private const val MAX_RETRY_DELAY_MS = 30_000L
    private const val MIN_DISCOVERY_RETRY_DELAY_MS = 5_000L
    private const val MAX_DISCOVERY_RETRY_DELAY_MS = 30_000L
    private const val PREFERENCES_NAME = "external_ai"
    private const val PREFERENCE_HOST = "host"
    private const val ANALYSIS_BROADCAST_INTERVAL_NS = 200_000_000L
    private const val MAX_CONSOLE_OBJECTS = 12
    private const val MODEL_BENCHMARK_RUNS = 3
  }
}

private data class ServiceConfig(
  val host: String,
  val framePort: Int,
  val resultPort: Int,
  val threshold: Float,
  val targetFps: Int,
  val inputSize: Int,
  val modelUri: Uri,
  val autoDiscover: Boolean,
  val classThresholds: ClassThresholds,
) {
  companion object {
    fun fromIntent(intent: Intent): ServiceConfig = ServiceConfig(
      host = requireNotNull(intent.getStringExtra(ExternalAIService.EXTRA_HOST)),
      framePort = intent.getIntExtra(ExternalAIService.EXTRA_FRAME_PORT, 7724),
      resultPort = intent.getIntExtra(ExternalAIService.EXTRA_RESULT_PORT, 7725),
      threshold = intent.getFloatExtra(ExternalAIService.EXTRA_THRESHOLD, 0.35f),
      targetFps = intent.getIntExtra(ExternalAIService.EXTRA_TARGET_FPS, 5).coerceIn(1, 20),
      inputSize = intent.getIntExtra(ExternalAIService.EXTRA_INPUT_SIZE, 320).also {
        require(it in YoloDetector.SUPPORTED_INPUT_SIZES) { "YOLO 입력 크기는 320, 416, 640 중 하나여야 합니다." }
      },
      modelUri = Uri.parse(requireNotNull(intent.getStringExtra(ExternalAIService.EXTRA_MODEL_URI))),
      autoDiscover = intent.getBooleanExtra(ExternalAIService.EXTRA_AUTO_DISCOVER, true),
      classThresholds = ClassThresholds.decode(
        intent.getStringExtra(ExternalAIService.EXTRA_CLASS_THRESHOLDS),
        intent.getFloatExtra(ExternalAIService.EXTRA_THRESHOLD, 0.35f),
      ),
    )
  }
}

private data class ModelBenchmarkResult(
  val modelName: String,
  val backend: String,
  val p95Ms: Double,
  val averageMs: Double,
  val averageObjects: Double,
  val error: String? = null,
)
