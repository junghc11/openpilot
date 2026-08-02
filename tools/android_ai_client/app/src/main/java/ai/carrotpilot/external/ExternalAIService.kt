package ai.carrotpilot.external

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.BitmapFactory
import android.net.Uri
import android.net.wifi.WifiManager
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import java.io.DataInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

class ExternalAIService : Service() {
  @Volatile private var runToken: AtomicBoolean? = null
  private var worker: Thread? = null
  @Volatile private var frameSocket: Socket? = null
  private var wakeLock: PowerManager.WakeLock? = null
  private var wifiLock: WifiManager.WifiLock? = null
  private var lastStatus = "중지됨"

  override fun onCreate() {
    super.onCreate()
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
    super.onDestroy()
  }

  override fun onBind(intent: Intent?): IBinder? = null

  private fun startClient(config: ServiceConfig) {
    stopWorkerOnly()
    acquirePerformanceLocks()
    val token = AtomicBoolean(true)
    runToken = token
    worker = Thread({ runClient(config, token) }, "carrot-external-ai").apply { start() }
  }

  private fun runClient(config: ServiceConfig, token: AtomicBoolean) {
    val modelFile = try {
      copyModelToCache(config.modelUri)
    } catch (error: Exception) {
      updateStatus("모델 열기 실패\n${error.message}")
      token.set(false)
      if (runToken === token) {
        runToken = null
        releasePerformanceLocks()
      }
      return
    }
    val modelName = config.modelUri.lastPathSegment ?: "yolo.onnx"
    try {
      YoloDetector(modelFile, config.threshold).use { detector ->
        while (token.get()) {
          try {
            updateStatus("C3X 연결 중: ${config.host}:${config.framePort}")
            Socket().use { socket ->
              frameSocket = socket
              socket.tcpNoDelay = true
              socket.soTimeout = 5_000
              socket.connect(InetSocketAddress(config.host, config.framePort), 3_000)
              DataInputStream(socket.getInputStream().buffered()).use { input ->
                DatagramSocket().use { resultSocket ->
                  processFrames(config, detector, modelName, input, resultSocket, token)
                }
              }
            }
          } catch (error: Exception) {
            if (token.get()) {
              updateStatus("연결 재시도 중\n${error.javaClass.simpleName}: ${error.message}")
              SystemClock.sleep(1_000)
            }
          } finally {
            if (runToken === token) frameSocket = null
          }
        }
      }
    } catch (error: Exception) {
      updateStatus("YOLO 초기화 실패\n${error.message}")
    } finally {
      token.set(false)
      if (runToken === token) {
        runToken = null
        releasePerformanceLocks()
      }
    }
  }

  private fun processFrames(
    config: ServiceConfig,
    detector: YoloDetector,
    modelName: String,
    input: DataInputStream,
    resultSocket: DatagramSocket,
    token: AtomicBoolean,
  ) {
    val resultAddress = InetAddress.getByName(config.host)
    val targetIntervalNs = 1_000_000_000L / config.targetFps
    var lastInferenceStartNs = 0L
    var receivedCount = 0
    var inferenceCount = 0
    var statusWindowStartNs = SystemClock.elapsedRealtimeNanos()
    var averageInferenceMs = 0.0
    var lastObjects = 0
    while (token.get()) {
      val frame = FrameProtocol.readFrame(input)
      receivedCount++
      val nowNs = SystemClock.elapsedRealtimeNanos()
      if (nowNs - lastInferenceStartNs < targetIntervalNs) continue
      val bitmap = BitmapFactory.decodeByteArray(frame.jpeg, 0, frame.jpeg.size)
        ?: error("C3X JPEG 디코딩 실패")
      val inferenceStartNs = SystemClock.elapsedRealtimeNanos()
      val detections = try {
        detector.detect(bitmap)
      } finally {
        bitmap.recycle()
      }
      val inferenceEndNs = SystemClock.elapsedRealtimeNanos()
      lastInferenceStartNs = inferenceStartNs
      val inferenceMs = (inferenceEndNs - inferenceStartNs) / 1_000_000.0
      averageInferenceMs = if (inferenceCount == 0) inferenceMs else averageInferenceMs * 0.9 + inferenceMs * 0.1
      inferenceCount++
      lastObjects = detections.size
      FrameProtocol.sendResult(
        resultSocket,
        resultAddress,
        config.resultPort,
        frame,
        inferenceStartNs,
        inferenceEndNs,
        detections,
        modelName,
      )

      val windowNs = inferenceEndNs - statusWindowStartNs
      if (windowNs >= 1_000_000_000L) {
        val seconds = windowNs / 1_000_000_000.0
        updateStatus(buildStatus(
          config = config,
          receiveFps = receivedCount / seconds,
          inferenceFps = inferenceCount / seconds,
          averageInferenceMs = averageInferenceMs,
          objectCount = lastObjects,
        ))
        receivedCount = 0
        inferenceCount = 0
        statusWindowStartNs = inferenceEndNs
      }
    }
  }

  private fun buildStatus(
    config: ServiceConfig,
    receiveFps: Double,
    inferenceFps: Double,
    averageInferenceMs: Double,
    objectCount: Int,
  ): String {
    val battery = registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val batteryTemp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)?.div(10.0) ?: 0.0
    val thermalStatus = getSystemService(PowerManager::class.java).currentThermalStatus
    return "연결됨: ${config.host}:${config.framePort}\n" +
      "수신 ${"%.1f".format(receiveFps)} FPS · 추론 ${"%.1f".format(inferenceFps)} FPS\n" +
      "평균 추론 ${"%.1f".format(averageInferenceMs)} ms · 객체 ${objectCount}개\n" +
      "백엔드 ONNX Runtime CPU · 배터리 ${"%.1f".format(batteryTemp)}°C\n" +
      "열 상태 $thermalStatus · 왕복 지연은 C3X에서 측정"
  }

  private fun copyModelToCache(uri: Uri): File {
    val modelFile = File(cacheDir, "selected-yolo.onnx")
    contentResolver.openInputStream(uri).use { input ->
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

  private fun updateStatus(message: String) {
    lastStatus = message
    (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIFICATION_ID, buildNotification(message))
    sendBroadcast(Intent(ACTION_STATUS).setPackage(packageName).putExtra(EXTRA_STATUS, message))
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
      description = "C3X 영상 수신 및 YOLO 추론 상태"
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
    const val EXTRA_MODEL_URI = "model_uri"
    const val EXTRA_STATUS = "status"
    private const val CHANNEL_ID = "external_ai"
    private const val NOTIFICATION_ID = 7724
    private const val MAX_MODEL_BYTES = 256L * 1024L * 1024L
  }
}

private data class ServiceConfig(
  val host: String,
  val framePort: Int,
  val resultPort: Int,
  val threshold: Float,
  val targetFps: Int,
  val modelUri: Uri,
) {
  companion object {
    fun fromIntent(intent: Intent): ServiceConfig = ServiceConfig(
      host = requireNotNull(intent.getStringExtra(ExternalAIService.EXTRA_HOST)),
      framePort = intent.getIntExtra(ExternalAIService.EXTRA_FRAME_PORT, 7724),
      resultPort = intent.getIntExtra(ExternalAIService.EXTRA_RESULT_PORT, 7725),
      threshold = intent.getFloatExtra(ExternalAIService.EXTRA_THRESHOLD, 0.35f),
      targetFps = max(1, intent.getIntExtra(ExternalAIService.EXTRA_TARGET_FPS, 5)),
      modelUri = Uri.parse(requireNotNull(intent.getStringExtra(ExternalAIService.EXTRA_MODEL_URI))),
    )
  }
}
