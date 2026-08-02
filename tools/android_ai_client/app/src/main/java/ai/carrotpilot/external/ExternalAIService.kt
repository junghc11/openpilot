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
import java.util.concurrent.atomic.AtomicBoolean

class ExternalAIService : Service() {
  @Volatile private var runToken: AtomicBoolean? = null
  private var worker: Thread? = null
  @Volatile private var frameSocket: Socket? = null
  private var wakeLock: PowerManager.WakeLock? = null
  private var wifiLock: WifiManager.WifiLock? = null
  private var lastStatus = "중지됨"

  override fun onCreate() {
    super.onCreate()
    serviceActive = true
    createNotificationChannel()
  }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    when (intent?.action) {
      ACTION_STOP -> stopClient()
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
    var retryDelayMs = MIN_RETRY_DELAY_MS
    var discoveryRetryDelayMs = MIN_DISCOVERY_RETRY_DELAY_MS
    var detector: YoloDetector? = null
    try {
      while (token.get()) {
        val targetHost = if (config.autoDiscover) {
          updateStatus("같은 사설망에서 CarrotPilot 기기 검색 중\nTCP ${config.framePort} / CAI1·CAI2 확인")
          DeviceDiscovery.findHost(config.framePort, config.host) { token.get() }
        } else {
          config.host
        }
        if (targetHost == null) {
          if (token.get()) {
            updateStatus("기기를 찾지 못함\n${discoveryRetryDelayMs / 1_000}초 후 같은 망 다시 검색")
            SystemClock.sleep(discoveryRetryDelayMs)
            discoveryRetryDelayMs = (discoveryRetryDelayMs * 2).coerceAtMost(MAX_DISCOVERY_RETRY_DELAY_MS)
          }
          continue
        }
        discoveryRetryDelayMs = MIN_DISCOVERY_RETRY_DELAY_MS
        if (config.autoDiscover) {
          getSharedPreferences(PREFERENCES_NAME, MODE_PRIVATE).edit().putString(PREFERENCE_HOST, targetHost).apply()
        }

        val activeDetector = detector ?: try {
          updateStatus("기기 발견: $targetHost\nYOLO 모델 준비 중", targetHost)
          YoloDetector(copyModelToCache(config.modelUri), config.threshold, config.inputSize).also { detector = it }
        } catch (error: Exception) {
          updateStatus("YOLO 모델 열기 실패\n${error.message}")
          token.set(false)
          break
        }

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
                processFrames(config, targetHost, activeDetector, modelName, input, resultSocket, token)
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
    input: DataInputStream,
    resultSocket: DatagramSocket,
    token: AtomicBoolean,
  ) {
    val resultAddress = InetAddress.getByName(targetHost)
    val targetIntervalNs = 1_000_000_000L / config.targetFps
    var lastInferenceStartNs = 0L
    var receivedCount = 0
    var inferenceCount = 0
    var statusWindowStartNs = SystemClock.elapsedRealtimeNanos()
    val performanceStats = RollingPerformanceStats()
    var lastObjects = 0
    var lastTransport = "대기"
    var previousEncoding = ""
    ReusableJpegDecoder().use { jpegDecoder ->
      ReusableH264Decoder().use { h264Decoder ->
        while (token.get()) {
          val receivedFrame = FrameProtocol.readFrame(input)
          receivedCount++
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
          val detectionResult = detector.detect(bitmap)
          val inferenceEndNs = SystemClock.elapsedRealtimeNanos()
          lastInferenceStartNs = inferenceStartNs
          lastTransport = if (inferenceFrame.encoding == FrameProtocol.ENCODING_H264) "H.264 HW" else "JPEG"
          val performance = FramePerformance(
            decodeMs = nanosToMillis(decodeEndNs - decodeStartNs),
            preprocessMs = detectionResult.preprocessMs,
            runtimeMs = detectionResult.runtimeMs,
            postprocessMs = detectionResult.postprocessMs,
            phoneTotalMs = nanosToMillis(inferenceEndNs - inferenceFrame.phoneReceiveTimestampNs),
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

          val windowNs = inferenceEndNs - statusWindowStartNs
          if (windowNs >= 1_000_000_000L) {
            val seconds = windowNs / 1_000_000_000.0
            updateStatus(buildStatus(
              config = config,
              targetHost = targetHost,
              receiveFps = receivedCount / seconds,
              inferenceFps = inferenceCount / seconds,
              performance = performanceStats.summary(),
              inputWidth = detector.inputWidth,
              inputHeight = detector.inputHeight,
              objectCount = lastObjects,
              backendLabel = detector.backendLabel,
              transportLabel = lastTransport,
            ))
            receivedCount = 0
            inferenceCount = 0
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
  ): String {
    val battery = registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val batteryTemp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)?.div(10.0) ?: 0.0
    val thermalStatus = getSystemService(PowerManager::class.java).currentThermalStatus
    return "연결됨: $targetHost:${config.framePort}\n" +
      "입력 ${inputWidth}×${inputHeight} · 수신 ${"%.1f".format(receiveFps)} · 처리 ${"%.1f".format(inferenceFps)} FPS\n" +
      "폰 처리 평균 ${"%.1f".format(performance.averagePhoneTotalMs)} · p95 ${"%.1f".format(performance.p95PhoneTotalMs)} ms\n" +
      "$transportLabel 디코드 ${"%.1f".format(performance.averageDecodeMs)} · 전처리 ${"%.1f".format(performance.averagePreprocessMs)} · ORT ${"%.1f".format(performance.averageRuntimeMs)} · 후처리 ${"%.1f".format(performance.averagePostprocessMs)} ms\n" +
      "객체 ${objectCount}개 · 표본 ${performance.samples}개\n" +
      "백엔드 $backendLabel · 배터리 ${"%.1f".format(batteryTemp)}°C\n" +
      "열 상태 ${thermalStatusLabel(thermalStatus)}($thermalStatus) · 총 지연은 C3X에서 측정"
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
    const val EXTRA_HOST = "host"
    const val EXTRA_FRAME_PORT = "frame_port"
    const val EXTRA_RESULT_PORT = "result_port"
    const val EXTRA_THRESHOLD = "threshold"
    const val EXTRA_TARGET_FPS = "target_fps"
    const val EXTRA_INPUT_SIZE = "input_size"
    const val EXTRA_MODEL_URI = "model_uri"
    const val EXTRA_AUTO_DISCOVER = "auto_discover"
    const val EXTRA_STATUS = "status"
    const val EXTRA_DISCOVERED_HOST = "discovered_host"
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
    )
  }
}
