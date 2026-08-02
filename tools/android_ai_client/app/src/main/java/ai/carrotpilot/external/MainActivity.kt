package ai.carrotpilot.external

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
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

@SuppressLint("SetTextI18n")
class MainActivity : Activity() {
  private val preferences by lazy { getSharedPreferences(PREFS, MODE_PRIVATE) }
  private lateinit var host: EditText
  private lateinit var framePort: EditText
  private lateinit var resultPort: EditText
  private lateinit var threshold: EditText
  private lateinit var inferenceFps: EditText
  private lateinit var autoConnect: CheckBox
  private lateinit var modelLabel: TextView
  private lateinit var status: TextView
  private var modelUri: Uri? = null
  private var activityStarted = false

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
      status.text = "YOLO ONNX 모델을 선택하면 자동 검색을 시작합니다."
    }
  }

  override fun onStop() {
    activityStarted = false
    unregisterReceiver(statusReceiver)
    super.onStop()
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
    inferenceFps = addField(root, "목표 추론 FPS (1~15)", "5")

    modelLabel = TextView(this).apply {
      setTextColor(Color.LTGRAY)
      textSize = 14f
      setPadding(0, (12 * density).toInt(), 0, (8 * density).toInt())
    }
    root.addView(modelLabel)
    root.addView(Button(this).apply {
      text = "YOLO ONNX 모델 선택"
      setOnClickListener { selectModel() }
    }, matchWidth())
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

  private fun startClient(autoDiscover: Boolean) {
    val uri = modelUri
    val config = try {
      ClientConfig(
        host = host.text.toString().trim(),
        framePort = framePort.text.toString().toInt(),
        resultPort = resultPort.text.toString().toInt(),
        threshold = threshold.text.toString().toFloat(),
        targetFps = inferenceFps.text.toString().toInt(),
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
    autoConnect.isChecked = preferences.getBoolean(KEY_AUTO_CONNECT, true)
    modelUri = preferences.getString(KEY_MODEL_URI, null)?.let(Uri::parse)
    updateModelLabel()
  }

  private fun saveSettings(config: ClientConfig) {
    preferences.edit()
      .putString(KEY_HOST, config.host)
      .putInt(KEY_FRAME_PORT, config.framePort)
      .putInt(KEY_RESULT_PORT, config.resultPort)
      .putFloat(KEY_THRESHOLD, config.threshold)
      .putInt(KEY_TARGET_FPS, config.targetFps)
      .putString(KEY_MODEL_URI, config.modelUri.toString())
      .apply()
  }

  private fun updateModelLabel() {
    modelLabel.text = modelUri?.let { "선택 모델: ${it.lastPathSegment ?: it}" } ?: "선택된 모델 없음"
  }

  private fun matchWidth() = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
  private fun weighted() = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

  companion object {
    private const val PREFS = "external_ai"
    private const val KEY_HOST = "host"
    private const val KEY_FRAME_PORT = "frame_port"
    private const val KEY_RESULT_PORT = "result_port"
    private const val KEY_THRESHOLD = "threshold"
    private const val KEY_TARGET_FPS = "target_fps"
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
  val modelUri: Uri,
  val autoDiscover: Boolean,
) {
  fun validate() {
    require(autoDiscover || host.isNotBlank()) { "수동 연결에는 C3/C3X/C4 IP가 필요합니다." }
    require(framePort in 1..65535) { "영상 포트는 1~65535 범위여야 합니다." }
    require(resultPort in 1..65535) { "결과 포트는 1~65535 범위여야 합니다." }
    require(threshold in 0.1f..0.95f) { "신뢰도는 0.1~0.95 범위여야 합니다." }
    require(targetFps in 1..15) { "추론 FPS는 1~15 범위여야 합니다." }
  }
}
