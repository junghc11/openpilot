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
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import java.io.File
import java.util.ArrayDeque

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
  private lateinit var modelSelector: Spinner
  private lateinit var downloadModelButton: Button
  private lateinit var deleteModelButton: Button
  private lateinit var modelLabel: TextView
  private lateinit var performanceHud: TextView
  private lateinit var backendBadge: TextView
  private lateinit var analysisConsole: TextView
  private lateinit var analysisConsoleScroll: ScrollView
  private lateinit var status: TextView
  private var modelUri: Uri? = null
  private var selectedCatalogModel = RecommendedModels.DEFAULT
  private var suppressModelSelection = false
  private val analysisEntries = ArrayDeque<String>()
  private var activityStarted = false
  @Volatile private var downloadingModel = false
  private var downloadThread: Thread? = null

  private val statusReceiver = object : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
      intent ?: return
      intent.getStringExtra(ExternalAIService.EXTRA_STATUS)?.let { status.text = it }
      intent.getStringExtra(ExternalAIService.EXTRA_ANALYSIS_LOG)?.let(::appendAnalysisLog)
      if (intent.action == ExternalAIService.ACTION_METRICS) updatePerformanceHud(intent)
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
    val filter = IntentFilter(ExternalAIService.ACTION_STATUS).apply {
      addAction(ExternalAIService.ACTION_ANALYSIS)
      addAction(ExternalAIService.ACTION_METRICS)
    }
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
      setBackgroundColor(Color.rgb(247, 249, 252))
    }
    root.addView(TextView(this).apply {
      text = "Carrot External AI"
      textSize = 26f
      setTextColor(Color.rgb(25, 35, 48))
    })
    root.addView(TextView(this).apply {
      text = "YOLO 탐지 결과는 화면 표시 전용이며 차량 제어에는 사용되지 않습니다."
      textSize = 15f
      setTextColor(Color.rgb(196, 82, 0))
      setPadding(0, (8 * density).toInt(), 0, (18 * density).toInt())
    })
    root.addView(TextView(this).apply {
      text = "스마트폰 핫스팟을 포함한 같은 사설망에서 C3/C3X/C4를 자동 검색합니다."
      textSize = 14f
      setTextColor(Color.rgb(74, 85, 104))
      setPadding(0, 0, 0, (10 * density).toInt())
    })
    root.addView(TextView(this).apply {
      text = if (BuildConfig.QNN_EP_INCLUDED) {
        "추론 우선순위: Qualcomm QNN/HTP 전체 그래프 → NNAPI → CPU"
      } else {
        "기본 빌드: NNAPI → CPU · QNN/HTP 런타임 미포함"
      }
      textSize = 13f
      setTextColor(if (BuildConfig.QNN_EP_INCLUDED) Color.rgb(20, 120, 70) else Color.rgb(74, 85, 104))
      setPadding(0, 0, 0, (10 * density).toInt())
    })

    autoConnect = CheckBox(this).apply {
      text = "앱 실행 시 같은 망 자동 검색 및 시작"
      setTextColor(Color.rgb(31, 41, 55))
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

    root.addView(TextView(this).apply {
      text = "권장 모델 선택"
      textSize = 16f
      setTypeface(typeface, Typeface.BOLD)
      setTextColor(Color.rgb(31, 41, 55))
      setPadding(0, (14 * density).toInt(), 0, (4 * density).toInt())
    })
    modelSelector = Spinner(this).apply {
      adapter = ArrayAdapter(
        this@MainActivity,
        android.R.layout.simple_spinner_item,
        RecommendedModels.ALL.map(VerifiedModelSpec::selectorLabel),
      ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
      onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
          if (!suppressModelSelection) selectCatalogModel(RecommendedModels.ALL[position])
        }

        override fun onNothingSelected(parent: AdapterView<*>?) = Unit
      }
    }
    root.addView(modelSelector, matchWidth())

    modelLabel = TextView(this).apply {
      setTextColor(Color.rgb(74, 85, 104))
      textSize = 14f
      setPadding(0, (12 * density).toInt(), 0, (8 * density).toInt())
    }
    root.addView(modelLabel)
    downloadModelButton = Button(this).apply {
      setOnClickListener { confirmSelectedModelDownload() }
    }
    root.addView(downloadModelButton, matchWidth())

    val modelControls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    modelControls.addView(Button(this).apply {
      text = "다른 ONNX 파일 선택"
      setOnClickListener { selectModel() }
    }, weighted())
    deleteModelButton = Button(this).apply {
      text = "선택 모델 삭제"
      setOnClickListener { deleteSelectedModel() }
    }
    modelControls.addView(deleteModelButton, weighted())
    root.addView(modelControls, matchWidth())

    val liveHeader = LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
      setPadding(0, (18 * density).toInt(), 0, (6 * density).toInt())
    }
    liveHeader.addView(TextView(this).apply {
      text = "실시간 프레임 추종"
      textSize = 18f
      setTypeface(typeface, Typeface.BOLD)
      setTextColor(Color.rgb(31, 41, 55))
    }, weighted())
    backendBadge = TextView(this).apply {
      text = "대기"
      textSize = 15f
      gravity = Gravity.CENTER
      setTypeface(Typeface.DEFAULT_BOLD)
      setTextColor(Color.WHITE)
      setPadding((14 * density).toInt(), (7 * density).toInt(), (14 * density).toInt(), (7 * density).toInt())
      background = roundedBackground(Color.rgb(100, 116, 139), 18f)
    }
    liveHeader.addView(backendBadge)
    root.addView(liveHeader, matchWidth())

    performanceHud = TextView(this).apply {
      text = "MODEL 대기\nVIDEO -- FPS   AI -- FPS\n추종률 --%   SKIP --/s"
      textSize = 20f
      setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
      setTextColor(Color.rgb(15, 23, 42))
      setPadding(padding, (14 * density).toInt(), padding, (14 * density).toInt())
      background = roundedBackground(Color.rgb(232, 241, 250), 14f, Color.rgb(174, 199, 224))
    }
    root.addView(performanceHud, matchWidth())

    val consoleHeader = LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
      setPadding(0, (14 * density).toInt(), 0, (5 * density).toInt())
    }
    consoleHeader.addView(TextView(this).apply {
      text = "실시간 객체 분석 콘솔"
      textSize = 17f
      setTypeface(typeface, Typeface.BOLD)
      setTextColor(Color.rgb(31, 41, 55))
    }, weighted())
    consoleHeader.addView(Button(this).apply {
      text = "로그 지우기"
      setOnClickListener { clearAnalysisLog() }
    })
    root.addView(consoleHeader, matchWidth())

    analysisConsole = TextView(this).apply {
      text = CONSOLE_PLACEHOLDER
      textSize = 12f
      typeface = Typeface.MONOSPACE
      setTextColor(Color.rgb(15, 23, 42))
      setTextIsSelectable(true)
      setPadding((12 * density).toInt(), (10 * density).toInt(), (12 * density).toInt(), (10 * density).toInt())
      background = roundedBackground(Color.rgb(241, 245, 249), 12f, Color.rgb(203, 213, 225))
    }
    analysisConsoleScroll = ScrollView(this).apply {
      isFillViewport = true
      addView(analysisConsole, matchWidth())
    }
    root.addView(analysisConsoleScroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (230 * density).toInt()))

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
      setTextColor(Color.rgb(31, 41, 55))
      background = roundedBackground(Color.rgb(255, 247, 230), 12f, Color.rgb(242, 193, 108))
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
      setTextColor(Color.rgb(74, 85, 104))
    })
    return EditText(this).also { field ->
      field.hint = hintValue
      field.setTextColor(Color.rgb(31, 41, 55))
      field.setHintTextColor(Color.rgb(148, 163, 184))
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

  private fun confirmSelectedModelDownload() {
    if (downloadingModel) return
    val model = selectedCatalogModel
    AlertDialog.Builder(this)
      .setTitle("${model.displayName} 다운로드")
      .setMessage(
        "공식 Ultralytics ${RecommendedModels.VERSION} 모델을 다운로드합니다.\n\n" +
          "용도: ${model.profileLabel}\n" +
          "COCO mAP50-95: ${model.map5095}\n" +
          "크기: ${"%.1f".format(model.sizeMegabytes)} MB\n" +
          "형식: 동적 FP32 ONNX · QNN/NNAPI에서 FP16 가속 허용\n" +
          "시험 권장: ${model.suggestedSettings}\n" +
          "라이선스: AGPL-3.0 또는 Enterprise\n\n" +
          "라이선스 조건을 확인하고 개인·오픈소스 실험 범위에 맞게 사용하세요.\n${RecommendedModels.LICENSE_URL}",
      )
      .setNegativeButton("취소", null)
      .setPositiveButton("동의 후 다운로드") { _, _ -> downloadSelectedModel(model) }
      .show()
  }

  private fun downloadSelectedModel(model: VerifiedModelSpec) {
    if (downloadingModel) return
    downloadingModel = true
    status.text = "${model.displayName} 다운로드 준비 중"
    updateModelLabel()
    downloadThread = Thread({
      try {
        val file = RecommendedModels.download(this, model) { received, total ->
          val percent = (received * 100L / total).toInt().coerceIn(0, 100)
          val receivedMegabytes = received / 1_048_576.0
          runOnUiThread {
            if (!isDestroyed) {
              downloadModelButton.text = "${model.id} 다운로드 중 $percent%"
              status.text = "${model.displayName} 다운로드 $percent% · ${"%.1f".format(receivedMegabytes)} MB"
            }
          }
        }
        runOnUiThread {
          if (isDestroyed) return@runOnUiThread
          modelUri = Uri.fromFile(file)
          preferences.edit().putString(KEY_MODEL_URI, modelUri.toString()).apply()
          downloadingModel = false
          updateModelLabel()
          status.text = "${model.displayName} 설치 및 SHA-256·ONNX 검증 완료"
          Toast.makeText(this, "${model.displayName} 설치 완료", Toast.LENGTH_LONG).show()
          if (activityStarted && autoConnect.isChecked) startClient(autoDiscover = true)
        }
      } catch (error: Exception) {
        runOnUiThread {
          if (isDestroyed) return@runOnUiThread
          downloadingModel = false
          updateModelLabel()
          status.text = "${model.displayName} 다운로드 실패\n${error.message ?: error.javaClass.simpleName}"
          Toast.makeText(this, "모델 다운로드 실패: ${error.message}", Toast.LENGTH_LONG).show()
        }
      } finally {
        downloadThread = null
      }
    }, "verified-model-${model.id}-download").apply { start() }
  }

  private fun deleteSelectedModel() {
    downloadThread?.interrupt()
    val model = selectedCatalogModel
    val deletingActiveModel = RecommendedModels.findByUri(this, modelUri) == model
    if (deletingActiveModel && ExternalAIService.serviceActive) {
      startService(Intent(this, ExternalAIService::class.java).setAction(ExternalAIService.ACTION_STOP))
    }
    RecommendedModels.delete(this, model)
    if (deletingActiveModel) {
      val fallback = RecommendedModels.firstInstalled(this)
      modelUri = fallback?.let { Uri.fromFile(RecommendedModels.installedFile(this, it)) }
      preferences.edit().apply {
        if (modelUri == null) remove(KEY_MODEL_URI) else putString(KEY_MODEL_URI, modelUri.toString())
      }.apply()
    }
    downloadingModel = false
    updateModelLabel()
    status.text = "${model.displayName} 삭제 완료"
  }

  private fun selectCatalogModel(model: VerifiedModelSpec) {
    selectedCatalogModel = model
    if (RecommendedModels.isInstalled(this, model)) {
      modelUri = Uri.fromFile(RecommendedModels.installedFile(this, model))
      preferences.edit().putString(KEY_MODEL_URI, modelUri.toString()).apply()
      status.text = "${model.displayName} 선택됨 · 다음 시작부터 적용"
      if (activityStarted && autoConnect.isChecked && !ExternalAIService.serviceActive) startClient(autoDiscover = true)
    } else {
      status.text = "${model.displayName}을 사용하려면 먼저 다운로드하세요."
    }
    updateModelLabel()
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
      RecommendedModels.firstInstalled(this) != null -> RecommendedModels.firstInstalled(this)?.let {
        Uri.fromFile(RecommendedModels.installedFile(this, it))
      }
      else -> null
    }
    RecommendedModels.findByUri(this, modelUri)?.let { activeModel ->
      selectedCatalogModel = activeModel
      suppressModelSelection = true
      modelSelector.setSelection(RecommendedModels.ALL.indexOf(activeModel))
      suppressModelSelection = false
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
    val selected = selectedCatalogModel
    val active = RecommendedModels.findByUri(this, modelUri)
    val selectedInstalled = RecommendedModels.isInstalled(this, selected)
    val activeLabel = when {
      active != null -> active.displayName
      modelUri != null -> "사용자 ONNX: ${modelUri?.lastPathSegment ?: modelUri}"
      else -> "없음"
    }
    modelLabel.text = "현재 사용 모델: $activeLabel\n" +
      "선택 모델: ${selected.displayName} · ${selected.profileLabel}\n" +
      "COCO mAP50-95 ${selected.map5095} · ${selected.suggestedSettings}\n" +
      "공식 동적 FP32 ONNX · SHA-256 고정 검증"
    downloadModelButton.text = when {
      downloadingModel -> "모델 다운로드 중"
      selectedInstalled -> "${selected.displayName} 다시 다운로드"
      else -> "${selected.displayName} 다운로드 (${"%.1f".format(selected.sizeMegabytes)} MB)"
    }
    downloadModelButton.isEnabled = !downloadingModel
    modelSelector.isEnabled = !downloadingModel
    deleteModelButton.isEnabled = selectedInstalled && !downloadingModel
  }

  private fun appendAnalysisLog(entry: String) {
    analysisEntries.addLast(entry)
    while (analysisEntries.size > MAX_CONSOLE_ENTRIES) analysisEntries.removeFirst()
    analysisConsole.text = analysisEntries.joinToString("\n\n")
    analysisConsoleScroll.post { analysisConsoleScroll.fullScroll(View.FOCUS_DOWN) }
  }

  private fun clearAnalysisLog() {
    analysisEntries.clear()
    analysisConsole.text = CONSOLE_PLACEHOLDER
  }

  private fun updatePerformanceHud(intent: Intent) {
    val model = intent.getStringExtra(ExternalAIService.EXTRA_MODEL_NAME) ?: "사용자 ONNX"
    val videoFps = intent.getDoubleExtra(ExternalAIService.EXTRA_VIDEO_FPS, 0.0)
    val aiFps = intent.getDoubleExtra(ExternalAIService.EXTRA_AI_FPS, 0.0)
    val followRate = intent.getDoubleExtra(ExternalAIService.EXTRA_FOLLOW_RATE, 0.0)
    val skippedFps = intent.getDoubleExtra(ExternalAIService.EXTRA_SKIPPED_FPS, 0.0)
    val badge = intent.getStringExtra(ExternalAIService.EXTRA_ACCELERATOR_BADGE) ?: "대기"
    performanceHud.text = "$model\n" +
      "VIDEO ${"%.1f".format(videoFps)} FPS   AI ${"%.1f".format(aiFps)} FPS\n" +
      "추종률 ${"%.0f".format(followRate)}%   SKIP ${"%.1f".format(skippedFps)}/s"
    backendBadge.text = badge
    backendBadge.background = roundedBackground(when (badge) {
      "eNPU" -> Color.rgb(0, 145, 92)
      "eACCEL" -> Color.rgb(214, 118, 0)
      "eCPU" -> Color.rgb(79, 70, 229)
      else -> Color.rgb(100, 116, 139)
    }, 18f)
  }

  private fun roundedBackground(fillColor: Int, radiusDp: Float, strokeColor: Int? = null): GradientDrawable =
    GradientDrawable().apply {
      shape = GradientDrawable.RECTANGLE
      setColor(fillColor)
      cornerRadius = radiusDp * resources.displayMetrics.density
      strokeColor?.let { setStroke((1 * resources.displayMetrics.density).toInt().coerceAtLeast(1), it) }
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
    private const val KEY_INPUT_SIZE = "input_size"
    private const val KEY_MODEL_URI = "model_uri"
    private const val KEY_AUTO_CONNECT = "auto_connect"
    private const val REQUEST_MODEL = 10
    private const val REQUEST_NOTIFICATIONS = 11
    private const val MAX_CONSOLE_ENTRIES = 40
    private const val CONSOLE_PLACEHOLDER = "[대기] 연결 후 객체 분석 로그가 표시됩니다.\n시간 · 프레임 · 객체 · 신뢰도 · 픽셀 좌표"
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
