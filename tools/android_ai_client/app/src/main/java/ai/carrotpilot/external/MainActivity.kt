package ai.carrotpilot.external

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import java.io.File

@SuppressLint("SetTextI18n")
class MainActivity : Activity() {
  private val preferences by lazy { getSharedPreferences(PREFS, MODE_PRIVATE) }
  private lateinit var host: EditText
  private lateinit var framePort: EditText
  private lateinit var resultPort: EditText
  private lateinit var threshold: EditText
  private lateinit var inferenceFps: EditText
  private lateinit var inputSize: EditText
  private lateinit var autoConnect: CheckBox
  private lateinit var downloadModelButton: Button
  private lateinit var deleteModelButton: Button
  private lateinit var modelLabel: TextView
  private lateinit var status: TextView
  private var modelUri: Uri? = null
  private var activityStarted = false
  @Volatile private var downloadingModel = false
  private var downloadThread: Thread? = null

  private val statusReceiver = object : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
      status.text = intent?.getStringExtra(ExternalAIService.EXTRA_STATUS) ?: "상태 정보 없음"
      intent?.getStringExtra(ExternalAIService.EXTRA_DISCOVERED_HOST)?.let { discoveredHost ->
        if (discoveredHost.isNotBlank() && host.text.toString() != discoveredHost) {
          host.setText(discoveredHost)
          preferences.edit().putString(KEY_HOST, discoveredHost).apply()
        }
      }
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(buildContent())
    loadSettings()
    if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
    }
  }

  @SuppressLint("UnspecifiedRegisterReceiverFlag")
  override fun onStart() {
    super.onStart()
    activityStarted = true
    val filter = IntentFilter(ExternalAIService.ACTION_STATUS)
    if (Build.VERSION.SDK_INT >= 33) {
      registerReceiver(statusReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
    } else {
      @Suppress("DEPRECATION")
      registerReceiver(statusReceiver, filter)
    }
    if (autoConnect.isChecked && modelUri != null && !ExternalAIService.serviceActive) {
      startClient(autoDiscover = true)
    } else if (autoConnect.isChecked && modelUri == null) {
      status.text = "권장 모델을 다운로드하거나 ONNX 파일을 선택하세요."
    }
  }

  override fun onStop() {
    activityStarted = false
    unregisterReceiver(statusReceiver)
    super.onStop()
  }

  override fun onDestroy() {
    downloadThread?.interrupt()
    super.onDestroy()
  }

  @Deprecated("Uses the platform document picker for broad Android compatibility")
  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    if (requestCode != REQUEST_MODEL || resultCode != RESULT_OK) return
    val uri = data?.data ?: return
    try {
      contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
    } catch (_: SecurityException) {
      // Some document providers grant access without offering persistable permissions.
    }
    modelUri = uri
    preferences.edit().putString(KEY_MODEL_URI, uri.toString()).apply()
    updateModelLabel()
    if (activityStarted && autoConnect.isChecked) startClient(autoDiscover = true)
  }

  private fun buildContent(): ScrollView {
    val density = resources.displayMetrics.density
    val padding = (20 * density).toInt()
    val root = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(padding, padding, padding, padding)
      setBackgroundColor(Color.rgb(16, 20, 24))
    }
    root.addView(TextView(this).apply {
      text = "Carrot External AI"
      textSize = 26f
      setTextColor(Color.WHITE)
    })
    root.addView(TextView(this).apply {
      text = "YOLO 탐지 결과는 화면 표시 전용이며 차량 제어에는 사용되지 않습니다."
      textSize = 15f
      setTextColor(Color.rgb(255, 183, 77))
      setPadding(0, (8 * density).toInt(), 0, (18 * density).toInt())
    })
    root.addView(TextView(this).apply {
      text = "스마트폰 핫스팟을 포함한 같은 사설망에서 C3/C3X/C4를 자동 검색합니다."
      textSize = 14f
      setTextColor(Color.LTGRAY)
      setPadding(0, 0, 0, (10 * density).toInt())
    })
    root.addView(TextView(this).apply {
      text = if (BuildConfig.QNN_EP_INCLUDED) {
        "추론 우선순위: Qualcomm QNN/HTP 전체 그래프 → NNAPI → CPU"
      } else {
        "기본 빌드: NNAPI → CPU · QNN/HTP 런타임 미포함"
      }
      textSize = 13f
      setTextColor(if (BuildConfig.QNN_EP_INCLUDED) Color.rgb(76, 175, 80) else Color.LTGRAY)
      setPadding(0, 0, 0, (10 * density).toInt())
    })

    autoConnect = CheckBox(this).apply {
      text = "앱 실행 시 같은 망 자동 검색 및 시작"
      setTextColor(Color.WHITE)
      setOnCheckedChangeListener { _, checked ->
        preferences.edit().putBoolean(KEY_AUTO_CONNECT, checked).apply()
        if (checked && activityStarted && modelUri != null && !ExternalAIService.serviceActive) {
          startClient(autoDiscover = true)
        }
      }
    }
    root.addView(autoConnect, matchWidth())

    host = addField(root, "기기 IP (수동 연결 또는 최근 검색값)", "192.168.0.10")
    framePort = addField(root, "영상 TCP 포트", "7724")
    resultPort = addField(root, "결과 UDP 포트", "7725")
    threshold = addField(root, "신뢰도 임계값 (0.1~0.95)", "0.35")
    inferenceFps = addField(root, "목표 추론 FPS (1~20)", "5")
    inputSize = addField(root, "YOLO 입력 크기 (320/416/640 · 권장 320)", "320")

    modelLabel = TextView(this).apply {
      setTextColor(Color.LTGRAY)
      textSize = 14f
      setPadding(0, (12 * density).toInt(), 0, (8 * density).toInt())
    }
    root.addView(modelLabel)
    downloadModelButton = Button(this).apply {
      setOnClickListener { confirmRecommendedModelDownload() }
    }
    root.addView(downloadModelButton, matchWidth())

    val modelControls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    modelControls.addView(Button(this).apply {
      text = "다른 ONNX 파일 선택"
      setOnClickListener { selectModel() }
    }, weighted())
    deleteModelButton = Button(this).apply {
      text = "권장 모델 삭제"
      setOnClickListener { deleteRecommendedModel() }
    }
    modelControls.addView(deleteModelButton, weighted())
    root.addView(modelControls, matchWidth())

    root.addView(Button(this).apply {
      text = "같은 망 자동 검색 후 시작"
      setOnClickListener { startClient(autoDiscover = true) }
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
      topMargin = (16 * density).toInt()
    })

    val controls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    controls.addView(Button(this).apply {
      text = "입력 IP로 시작"
      setOnClickListener { startClient(autoDiscover = false) }
    }, weighted())
    controls.addView(Button(this).apply {
      text = "중지"
      setOnClickListener {
        autoConnect.isChecked = false
        startService(Intent(this@MainActivity, ExternalAIService::class.java).setAction(ExternalAIService.ACTION_STOP))
        status.text = "중지됨"
      }
    }, weighted())
    root.addView(controls, matchWidth())

    status = TextView(this).apply {
      text = "중지됨"
      textSize = 16f
      setTextColor(Color.WHITE)
      setBackgroundColor(Color.rgb(36, 43, 49))
      setPadding(padding, padding, padding, padding)
    }
    root.addView(status, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
      topMargin = (16 * density).toInt()
    })
    return ScrollView(this).apply { addView(root) }
  }

  private fun addField(parent: LinearLayout, label: String, hintValue: String): EditText {
    parent.addView(TextView(this).apply {
      text = label
      textSize = 13f
      setTextColor(Color.LTGRAY)
    })
    return EditText(this).also { field ->
      field.hint = hintValue
      field.setTextColor(Color.WHITE)
      field.setHintTextColor(Color.GRAY)
      field.setSingleLine(true)
      parent.addView(field, matchWidth())
    }
  }

  private fun selectModel() {
    val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      addCategory(Intent.CATEGORY_OPENABLE)
      type = "application/octet-stream"
      putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("application/octet-stream", "application/onnx", "*/*"))
      addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
    }
    @Suppress("DEPRECATION")
    startActivityForResult(intent, REQUEST_MODEL)
  }

  private fun confirmRecommendedModelDownload() {
    if (downloadingModel) return
    AlertDialog.Builder(this)
      .setTitle("권장 모델 다운로드")
      .setMessage(
        "공식 Ultralytics YOLO11n ONNX ${RecommendedModel.VERSION} 모델을 다운로드합니다.\n\n" +
          "크기: 10.4 MB\n입력: 동적 FP32 · 앱 기본 실행 320×320\n라이선스: AGPL-3.0 또는 Enterprise\n\n" +
          "라이선스 조건을 확인하고 개인·오픈소스 실험 범위에 맞게 사용하세요.\n${RecommendedModel.LICENSE_URL}",
      )
      .setNegativeButton("취소", null)
      .setPositiveButton("동의 후 다운로드") { _, _ -> downloadRecommendedModel() }
      .show()
  }

  private fun downloadRecommendedModel() {
    if (downloadingModel) return
    downloadingModel = true
    status.text = "권장 YOLO11n 모델 다운로드 준비 중"
    updateModelLabel()
    downloadThread = Thread({
      try {
        val file = RecommendedModel.download(this) { received, total ->
          val percent = (received * 100L / total).toInt().coerceIn(0, 100)
          val receivedMegabytes = received / 1_048_576.0
          runOnUiThread {
            if (!isDestroyed) {
              downloadModelButton.text = "권장 모델 다운로드 중 $percent%"
              status.text = "YOLO11n 다운로드 $percent% · ${"%.1f".format(receivedMegabytes)} MB"
            }
          }
        }
        runOnUiThread {
          if (isDestroyed) return@runOnUiThread
          modelUri = Uri.fromFile(file)
          preferences.edit().putString(KEY_MODEL_URI, modelUri.toString()).apply()
          downloadingModel = false
          updateModelLabel()
          status.text = "권장 모델 설치 및 SHA-256·ONNX 검증 완료"
          Toast.makeText(this, "YOLO11n 권장 모델 설치 완료", Toast.LENGTH_LONG).show()
          if (activityStarted && autoConnect.isChecked) startClient(autoDiscover = true)
        }
      } catch (error: Exception) {
        runOnUiThread {
          if (isDestroyed) return@runOnUiThread
          downloadingModel = false
          updateModelLabel()
          status.text = "권장 모델 다운로드 실패\n${error.message ?: error.javaClass.simpleName}"
          Toast.makeText(this, "모델 다운로드 실패: ${error.message}", Toast.LENGTH_LONG).show()
        }
      } finally {
        downloadThread = null
      }
    }, "recommended-model-download").apply { start() }
  }

  private fun deleteRecommendedModel() {
    downloadThread?.interrupt()
    val selectedRecommended = isRecommendedModelUri(modelUri)
    if (selectedRecommended && ExternalAIService.serviceActive) {
      startService(Intent(this, ExternalAIService::class.java).setAction(ExternalAIService.ACTION_STOP))
    }
    RecommendedModel.delete(this)
    if (selectedRecommended) {
      modelUri = null
      preferences.edit().remove(KEY_MODEL_URI).apply()
    }
    downloadingModel = false
    updateModelLabel()
    status.text = "권장 모델 삭제 완료"
  }

  private fun startClient(autoDiscover: Boolean) {
    val uri = modelUri
    val config = try {
      ClientConfig(
        host = host.text.toString().trim(),
        framePort = framePort.text.toString().toInt(),
        resultPort = resultPort.text.toString().toInt(),
        threshold = threshold.text.toString().toFloat(),
        targetFps = inferenceFps.text.toString().toInt(),
        inputSize = inputSize.text.toString().toInt(),
        modelUri = uri ?: error("YOLO ONNX 모델을 선택하세요."),
        autoDiscover = autoDiscover,
      ).also { it.validate() }
    } catch (error: Exception) {
      Toast.makeText(this, error.message ?: "설정값을 확인하세요.", Toast.LENGTH_LONG).show()
      return
    }
    saveSettings(config)
    status.text = if (autoDiscover) "같은 망에서 CarrotPilot 기기 검색 준비 중" else "입력 IP 연결 준비 중"
    val intent = Intent(this, ExternalAIService::class.java).apply {
      action = ExternalAIService.ACTION_START
      putExtra(ExternalAIService.EXTRA_HOST, config.host)
      putExtra(ExternalAIService.EXTRA_FRAME_PORT, config.framePort)
      putExtra(ExternalAIService.EXTRA_RESULT_PORT, config.resultPort)
      putExtra(ExternalAIService.EXTRA_THRESHOLD, config.threshold)
      putExtra(ExternalAIService.EXTRA_TARGET_FPS, config.targetFps)
      putExtra(ExternalAIService.EXTRA_INPUT_SIZE, config.inputSize)
      putExtra(ExternalAIService.EXTRA_MODEL_URI, config.modelUri.toString())
      putExtra(ExternalAIService.EXTRA_AUTO_DISCOVER, config.autoDiscover)
    }
    startForegroundService(intent)
  }

  private fun loadSettings() {
    host.setText(preferences.getString(KEY_HOST, "192.168.0.10"))
    framePort.setText(preferences.getInt(KEY_FRAME_PORT, 7724).toString())
    resultPort.setText(preferences.getInt(KEY_RESULT_PORT, 7725).toString())
    threshold.setText(preferences.getFloat(KEY_THRESHOLD, 0.35f).toString())
    inferenceFps.setText(preferences.getInt(KEY_TARGET_FPS, 5).toString())
    inputSize.setText(preferences.getInt(KEY_INPUT_SIZE, 320).toString())
    autoConnect.isChecked = preferences.getBoolean(KEY_AUTO_CONNECT, true)
    val storedUri = preferences.getString(KEY_MODEL_URI, null)?.let(Uri::parse)
    modelUri = when {
      storedUri?.scheme == "file" && storedUri.path?.let(::File)?.isFile == true -> storedUri
      storedUri != null && storedUri.scheme != "file" -> storedUri
      RecommendedModel.isInstalled(this) -> Uri.fromFile(RecommendedModel.installedFile(this))
      else -> null
    }
    updateModelLabel()
  }

  private fun saveSettings(config: ClientConfig) {
    preferences.edit()
      .putString(KEY_HOST, config.host)
      .putInt(KEY_FRAME_PORT, config.framePort)
      .putInt(KEY_RESULT_PORT, config.resultPort)
      .putFloat(KEY_THRESHOLD, config.threshold)
      .putInt(KEY_TARGET_FPS, config.targetFps)
      .putInt(KEY_INPUT_SIZE, config.inputSize)
      .putString(KEY_MODEL_URI, config.modelUri.toString())
      .apply()
  }

  private fun updateModelLabel() {
    val installed = RecommendedModel.isInstalled(this)
    modelLabel.text = when {
      isRecommendedModelUri(modelUri) && installed ->
        "권장 모델 준비됨: ${RecommendedModel.DISPLAY_NAME}\nSHA-256 검증 버전: ${RecommendedModel.VERSION}"
      modelUri != null -> "사용자 선택 모델: ${modelUri?.lastPathSegment ?: modelUri}"
      installed -> "권장 모델 설치됨 · 사용하려면 권장 모델 버튼을 누르세요."
      else -> "선택된 모델 없음 · 권장: YOLO11n 동적 FP32 · 기본 실행 320"
    }
    downloadModelButton.text = when {
      downloadingModel -> "권장 모델 다운로드 중"
      installed -> "권장 모델 다시 다운로드"
      else -> "권장 모델 다운로드 (YOLO11n 동적 · 10.4 MB)"
    }
    downloadModelButton.isEnabled = !downloadingModel
    deleteModelButton.isEnabled = installed && !downloadingModel
  }

  private fun isRecommendedModelUri(uri: Uri?): Boolean =
    uri?.scheme == "file" && uri.path == RecommendedModel.installedFile(this).absolutePath

  private fun matchWidth() = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
  private fun weighted() = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

  companion object {
    private const val PREFS = "external_ai"
    private const val KEY_HOST = "host"
    private const val KEY_FRAME_PORT = "frame_port"
    private const val KEY_RESULT_PORT = "result_port"
    private const val KEY_THRESHOLD = "threshold"
    private const val KEY_TARGET_FPS = "target_fps"
    private const val KEY_INPUT_SIZE = "input_size"
    private const val KEY_MODEL_URI = "model_uri"
    private const val KEY_AUTO_CONNECT = "auto_connect"
    private const val REQUEST_MODEL = 10
    private const val REQUEST_NOTIFICATIONS = 11
  }
}

data class ClientConfig(
  val host: String,
  val framePort: Int,
  val resultPort: Int,
  val threshold: Float,
  val targetFps: Int,
  val inputSize: Int,
  val modelUri: Uri,
  val autoDiscover: Boolean,
) {
  fun validate() {
    require(autoDiscover || host.isNotBlank()) { "수동 연결에는 C3/C3X/C4 IP가 필요합니다." }
    require(framePort in 1..65535) { "영상 포트는 1~65535 범위여야 합니다." }
    require(resultPort in 1..65535) { "결과 포트는 1~65535 범위여야 합니다." }
    require(threshold in 0.1f..0.95f) { "신뢰도는 0.1~0.95 범위여야 합니다." }
    require(targetFps in 1..20) { "추론 FPS는 1~20 범위여야 합니다." }
    require(inputSize in YoloDetector.SUPPORTED_INPUT_SIZES) { "YOLO 입력 크기는 320, 416, 640 중 하나여야 합니다." }
  }
}
