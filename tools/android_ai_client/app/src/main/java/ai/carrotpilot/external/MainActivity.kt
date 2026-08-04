package ai.carrotpilot.external

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.ClipData
import android.content.ClipboardManager
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
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
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
  private lateinit var backendBadge: TextView
  private lateinit var acceleratorDetail: TextView
  private lateinit var analysisConsole: TextView
  private lateinit var analysisConsoleScroll: ScrollView
  private lateinit var status: TextView
  private lateinit var connectionChip: TextView
  private lateinit var deviceHeadline: TextView
  private lateinit var deviceAddress: TextView
  private lateinit var currentModelValue: TextView
  private lateinit var videoFpsValue: TextView
  private lateinit var aiFpsValue: TextView
  private lateinit var followRateValue: TextView
  private lateinit var skippedFpsValue: TextView
  private lateinit var recentDetectionPreview: TextView
  private lateinit var connectionSettingsPanel: LinearLayout
  private lateinit var manageConnectionButton: Button
  private lateinit var contentScroll: ScrollView
  private val tabButtons = mutableListOf<TextView>()
  private val tabPages = mutableListOf<View>()
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
      intent.getStringExtra(ExternalAIService.EXTRA_STATUS)?.let(::updateConnectionStatus)
      intent.getStringExtra(ExternalAIService.EXTRA_ANALYSIS_LOG)?.let(::appendAnalysisLog)
      if (intent.action == ExternalAIService.ACTION_METRICS) updatePerformanceHud(intent)
      intent?.getStringExtra(ExternalAIService.EXTRA_DISCOVERED_HOST)?.let { discoveredHost ->
        if (discoveredHost.isNotBlank() && host.text.toString() != discoveredHost) {
          host.setText(discoveredHost)
          deviceAddress.text = discoveredHost
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
      updateConnectionStatus("권장 모델을 다운로드하거나 ONNX 파일을 선택하세요.")
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
    applyModelInputPolicy(null)
    updateModelLabel()
    if (activityStarted && autoConnect.isChecked) startClient(autoDiscover = true)
  }

  private fun buildContent(): ScrollView {
    val root = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(dp(20), dp(18), dp(20), dp(28))
      setBackgroundColor(COLOR_BACKGROUND)
    }
    root.addView(buildHeader(), matchWidth())
    root.addView(buildTabs(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(52)).apply {
      topMargin = dp(22)
      bottomMargin = dp(18)
    })

    val statusPage = buildStatusPage()
    val modelPage = buildModelPage().apply { visibility = View.GONE }
    val logPage = buildLogPage().apply { visibility = View.GONE }
    tabPages += listOf(statusPage, modelPage, logPage)
    root.addView(statusPage, matchWidth())
    root.addView(modelPage, matchWidth())
    root.addView(logPage, matchWidth())

    contentScroll = ScrollView(this).apply {
      isFillViewport = true
      overScrollMode = View.OVER_SCROLL_IF_CONTENT_SCROLLS
      addView(root)
      setOnApplyWindowInsetsListener { view, insets ->
        val topInset: Int
        val bottomInset: Int
        if (Build.VERSION.SDK_INT >= 30) {
          val systemBars = insets.getInsets(WindowInsets.Type.systemBars())
          topInset = systemBars.top
          bottomInset = systemBars.bottom
        } else {
          @Suppress("DEPRECATION")
          topInset = insets.systemWindowInsetTop
          @Suppress("DEPRECATION")
          bottomInset = insets.systemWindowInsetBottom
        }
        view.setPadding(0, topInset, 0, bottomInset)
        insets
      }
    }
    selectTab(0)
    return contentScroll
  }

  private fun buildHeader(): View {
    val header = LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
    }
    header.addView(ImageView(this).apply {
      setImageResource(ai.carrotpilot.external.R.drawable.ic_carrot_ai)
      contentDescription = "Carrot External AI"
      scaleType = ImageView.ScaleType.FIT_CENTER
    }, LinearLayout.LayoutParams(dp(44), dp(44)).apply { marginEnd = dp(12) })
    header.addView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      addView(label("Carrot External AI", 25f, COLOR_TEXT, Typeface.BOLD))
      addView(label("CarrotPilot C3/C3X/C4 전용", 14f, COLOR_MUTED).apply { setPadding(0, dp(2), 0, 0) })
    }, weighted())
    connectionChip = label("연결 대기", 13f, COLOR_MUTED, Typeface.BOLD).apply {
      gravity = Gravity.CENTER
      setPadding(dp(12), dp(8), dp(12), dp(8))
      background = roundedBackground(COLOR_SURFACE_ALT, 16f, COLOR_BORDER)
    }
    header.addView(connectionChip)
    return header
  }

  private fun buildTabs(): View {
    val row = LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER
      background = roundedBackground(COLOR_SURFACE_ALT, 16f, COLOR_BORDER)
      setPadding(dp(4), dp(4), dp(4), dp(4))
    }
    listOf("상태", "모델", "로그").forEachIndexed { index, title ->
      val tab = label(title, 15f, COLOR_MUTED, Typeface.BOLD).apply {
        gravity = Gravity.CENTER
        isClickable = true
        isFocusable = true
        contentDescription = "$title 탭"
        setOnClickListener { selectTab(index) }
      }
      tabButtons += tab
      row.addView(tab, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f).apply {
        if (index > 0) marginStart = dp(3)
      })
    }
    return row
  }

  private fun buildStatusPage(): LinearLayout = LinearLayout(this).apply {
    orientation = LinearLayout.VERTICAL

    addView(card().apply {
      val topRow = LinearLayout(this@MainActivity).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
      }
      topRow.addView(LinearLayout(this@MainActivity).apply {
        orientation = LinearLayout.VERTICAL
        deviceHeadline = label("연결 준비", 27f, COLOR_TEXT, Typeface.BOLD)
        addView(deviceHeadline)
        deviceAddress = label("기기 검색 전", 14f, COLOR_MUTED).apply { setPadding(0, dp(3), 0, 0) }
        addView(deviceAddress)
      }, weighted())
      topRow.addView(styledButton("재연결", primary = false, compact = true).apply {
        setOnClickListener { startClient(autoDiscover = true) }
      })
      addView(topRow, matchWidth())

      addView(divider(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1)).apply {
        topMargin = dp(18)
        bottomMargin = dp(18)
      })

      val runtimeRow = LinearLayout(this@MainActivity).apply { orientation = LinearLayout.HORIZONTAL }
      runtimeRow.addView(LinearLayout(this@MainActivity).apply {
        orientation = LinearLayout.VERTICAL
        addView(label("YOLO 모델", 12f, COLOR_MUTED))
        currentModelValue = label("선택되지 않음", 18f, COLOR_TEXT, Typeface.BOLD).apply { setPadding(0, dp(4), 0, 0) }
        addView(currentModelValue)
      }, weighted())
      runtimeRow.addView(LinearLayout(this@MainActivity).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(16), 0, 0, 0)
        addView(label("가속 상태", 12f, COLOR_MUTED))
        backendBadge = label("대기", 14f, Color.WHITE, Typeface.BOLD).apply {
          gravity = Gravity.CENTER
          setPadding(dp(13), dp(6), dp(13), dp(6))
          background = roundedBackground(COLOR_BADGE_IDLE, 16f)
        }
        addView(backendBadge, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
          topMargin = dp(5)
        })
      }, weighted())
      addView(runtimeRow, matchWidth())

      acceleratorDetail = label("가속 진단 대기", 12f, COLOR_MUTED).apply {
        setPadding(dp(12), dp(10), dp(12), dp(10))
        setTextIsSelectable(true)
        background = roundedBackground(COLOR_SURFACE_ALT, 10f, COLOR_BORDER)
      }
      addView(acceleratorDetail, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
        topMargin = dp(14)
      })

      addView(divider(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1)).apply {
        topMargin = dp(18)
        bottomMargin = dp(14)
      })

      val metrics = LinearLayout(this@MainActivity).apply { orientation = LinearLayout.HORIZONTAL }
      videoFpsValue = addMetric(metrics, "VIDEO FPS", "--", "FPS")
      aiFpsValue = addMetric(metrics, "AI FPS", "--", "FPS")
      followRateValue = addMetric(metrics, "FRAME FOLLOW", "--", "%")
      skippedFpsValue = addMetric(metrics, "SKIP", "--", "/s")
      addView(metrics, matchWidth())

      status = label("중지됨", 14f, COLOR_TEXT, Typeface.NORMAL).apply {
        setPadding(dp(14), dp(12), dp(14), dp(12))
        background = roundedBackground(COLOR_ORANGE_TINT, 12f, COLOR_ORANGE_BORDER)
        setTextIsSelectable(true)
      }
      addView(status, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
        topMargin = dp(18)
      })
    }, matchWidth())

    val recentHeader = LinearLayout(this@MainActivity).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
      setPadding(0, dp(24), 0, dp(10))
      addView(label("최근 감지", 20f, COLOR_TEXT, Typeface.BOLD), weighted())
      addView(label("최근 5개 분석", 13f, COLOR_MUTED))
    }
    addView(recentHeader, matchWidth())
    recentDetectionPreview = label(RECENT_PLACEHOLDER, 13f, COLOR_MUTED).apply {
      typeface = Typeface.MONOSPACE
      setTextIsSelectable(true)
      setLineSpacing(0f, 1.12f)
      setPadding(dp(16), dp(14), dp(16), dp(14))
      background = roundedBackground(Color.WHITE, 16f, COLOR_BORDER)
    }
    addView(recentDetectionPreview, matchWidth())

    addView(styledButton("세션 중지", primary = true).apply {
      setOnClickListener { stopClient() }
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(54)).apply { topMargin = dp(18) })
    manageConnectionButton = styledButton("연결 관리", primary = false).apply {
      setOnClickListener {
        val opening = connectionSettingsPanel.visibility != View.VISIBLE
        connectionSettingsPanel.visibility = if (opening) View.VISIBLE else View.GONE
        text = if (opening) "연결 설정 닫기" else "연결 관리"
      }
    }
    addView(manageConnectionButton, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(54)).apply { topMargin = dp(10) })

    connectionSettingsPanel = buildConnectionSettings().apply { visibility = View.GONE }
    addView(connectionSettingsPanel, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
      topMargin = dp(12)
    })
  }

  private fun buildConnectionSettings(): LinearLayout = card().apply {
    addView(sectionTitle("연결 설정"))
    addView(label("같은 사설망의 C3/C3X/C4를 자동 검색합니다. 수동 IP와 배포 명령은 고급 설정에서 사용할 수 있습니다.", 13f, COLOR_MUTED).apply {
      setPadding(0, dp(6), 0, dp(12))
    })
    autoConnect = CheckBox(this@MainActivity).apply {
      text = "앱 실행 시 같은 망 자동 검색 및 시작"
      textSize = 14f
      setTextColor(COLOR_TEXT)
      buttonTintList = android.content.res.ColorStateList.valueOf(COLOR_ORANGE)
      setOnCheckedChangeListener { _, checked ->
        preferences.edit().putBoolean(KEY_AUTO_CONNECT, checked).apply()
        if (checked && activityStarted && modelUri != null && !ExternalAIService.serviceActive) {
          startClient(autoDiscover = true)
        }
      }
    }
    addView(autoConnect, matchWidth())
    host = addField(this, "기기 IP (수동 연결 또는 최근 검색값)", "192.168.0.10")
    framePort = addField(this, "영상 TCP 포트", "7724")
    resultPort = addField(this, "결과 UDP 포트", "7725")
    addView(styledButton("C3X 브랜치 변경 SSH 명령 복사", primary = false).apply {
      setOnClickListener { copyC3xBranchSshCommand() }
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(50)).apply { topMargin = dp(8) })
    addView(label("주차·주행 종료 상태에서 SSH 가능한 PC 또는 Android 터미널에 붙여넣으세요.", 12f, COLOR_MUTED).apply {
      setPadding(0, dp(6), 0, dp(8))
    })
    addView(styledButton("같은 망 자동 검색 후 시작", primary = true).apply {
      setOnClickListener { startClient(autoDiscover = true) }
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(52)).apply { topMargin = dp(6) })
    addView(styledButton("입력 IP로 시작", primary = false).apply {
      setOnClickListener { startClient(autoDiscover = false) }
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(52)).apply { topMargin = dp(8) })
  }

  private fun buildModelPage(): LinearLayout = LinearLayout(this).apply {
    orientation = LinearLayout.VERTICAL
    addView(card().apply {
      addView(sectionTitle("권장 모델 선택"))
      addView(label("NPU 모델은 QNN용 정적 QDQ 형식이며 입력 크기가 자동 고정됩니다.", 13f, COLOR_MUTED).apply { setPadding(0, dp(5), 0, dp(12)) })
      modelSelector = Spinner(this@MainActivity).apply {
        minimumHeight = dp(50)
        background = roundedBackground(COLOR_SURFACE_ALT, 12f, COLOR_BORDER)
        setPadding(dp(12), 0, dp(8), 0)
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
      addView(modelSelector, matchWidth())
      modelLabel = label("", 13f, COLOR_MUTED).apply {
        setPadding(0, dp(12), 0, dp(8))
        setLineSpacing(0f, 1.12f)
      }
      addView(modelLabel, matchWidth())
      downloadModelButton = styledButton("권장 모델 다운로드", primary = true).apply {
        setOnClickListener { confirmSelectedModelDownload() }
      }
      addView(downloadModelButton, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(52)))
      addView(styledButton("다른 ONNX 파일 선택", primary = false).apply {
        setOnClickListener { selectModel() }
      }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(50)).apply { topMargin = dp(8) })
      deleteModelButton = styledButton("선택 모델 삭제", primary = false).apply {
        setOnClickListener { deleteSelectedModel() }
      }
      addView(deleteModelButton, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(50)).apply { topMargin = dp(8) })
    }, matchWidth())

    addView(card().apply {
      addView(sectionTitle("추론 성능"))
      addView(label("먼저 320·5~10 FPS로 발열과 프레임 추종률을 확인하세요.", 13f, COLOR_MUTED).apply {
        setPadding(0, dp(5), 0, dp(8))
      })
      threshold = addField(this, "신뢰도 임계값 (0.1~0.95)", "0.35")
      inferenceFps = addField(this, "목표 추론 FPS (1~20)", "5")
      inputSize = addField(this, "YOLO 입력 크기 (NPU 모델은 자동 고정)", "320")
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(12) })

    addView(card().apply {
      addView(sectionTitle("가속 우선순위"))
      addView(label(if (BuildConfig.QNN_EP_INCLUDED) {
        "Qualcomm QNN/HTP 전체 그래프 → QNN+CPU 혼합 → NNAPI → CPU"
      } else {
        "NNAPI → CPU · QNN/HTP 런타임 미포함"
      }, 14f, if (BuildConfig.QNN_EP_INCLUDED) COLOR_GREEN else COLOR_MUTED).apply { setPadding(0, dp(7), 0, 0) })
      addView(label("eNPU는 CPU 폴백을 금지한 QNN/HTP 전체 그래프 예열까지 성공했을 때만 표시됩니다.", 12f, COLOR_MUTED).apply {
        setPadding(0, dp(8), 0, 0)
      })
      addView(label("eNPU+CPU는 ORT 프로파일에서 QNN 노드 실행이 확인된 혼합 경로입니다. eQNN?은 QNN 세션만 열렸고 실제 노드 배치는 확인하지 못한 상태입니다.", 12f, COLOR_MUTED).apply {
        setPadding(0, dp(6), 0, 0)
      })
      addView(label("C3X가 없어도 세션 시작 즉시 더미 입력으로 사전 점검하며, SoC와 실패 원문을 가속 진단에 유지합니다.", 12f, COLOR_MUTED).apply {
        setPadding(0, dp(6), 0, 0)
      })
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
      topMargin = dp(12)
      bottomMargin = dp(12)
    })
  }

  private fun buildLogPage(): LinearLayout = LinearLayout(this).apply {
    orientation = LinearLayout.VERTICAL
    val header = LinearLayout(this@MainActivity).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
      addView(LinearLayout(this@MainActivity).apply {
        orientation = LinearLayout.VERTICAL
        addView(sectionTitle("실시간 객체 분석"))
        addView(label("시간 · 프레임 · 객체 · 신뢰도 · 픽셀 좌표", 12f, COLOR_MUTED).apply { setPadding(0, dp(4), 0, 0) })
      }, weighted())
      addView(styledButton("로그 지우기", primary = false, compact = true).apply {
        setOnClickListener { clearAnalysisLog() }
      })
    }
    addView(header, matchWidth())
    analysisConsole = label(CONSOLE_PLACEHOLDER, 12f, COLOR_TEXT).apply {
      typeface = Typeface.MONOSPACE
      setTextIsSelectable(true)
      setLineSpacing(0f, 1.12f)
      setPadding(dp(16), dp(16), dp(16), dp(16))
      background = roundedBackground(Color.WHITE, 16f, COLOR_BORDER)
    }
    analysisConsoleScroll = ScrollView(this@MainActivity).apply {
      isFillViewport = true
      addView(analysisConsole, matchWidth())
    }
    addView(analysisConsoleScroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(520)).apply { topMargin = dp(14) })
    addView(label("탐지 결과는 화면 표시와 진단 전용이며 차량 제어에는 사용되지 않습니다.", 12f, COLOR_ORANGE_DARK).apply {
      setPadding(dp(14), dp(12), dp(14), dp(12))
      background = roundedBackground(COLOR_ORANGE_TINT, 12f, COLOR_ORANGE_BORDER)
    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(12) })
  }

  private fun selectTab(index: Int) {
    tabPages.forEachIndexed { pageIndex, page -> page.visibility = if (pageIndex == index) View.VISIBLE else View.GONE }
    tabButtons.forEachIndexed { tabIndex, tab ->
      val selected = tabIndex == index
      tab.setTextColor(if (selected) COLOR_ORANGE_DARK else COLOR_MUTED)
      tab.background = roundedBackground(if (selected) Color.WHITE else Color.TRANSPARENT, 12f, if (selected) COLOR_BORDER else null)
      tab.isSelected = selected
    }
    if (::contentScroll.isInitialized) contentScroll.post { contentScroll.smoothScrollTo(0, 0) }
  }

  private fun addMetric(parent: LinearLayout, title: String, initialValue: String, unit: String): TextView {
    val value = label(initialValue, 28f, COLOR_TEXT, Typeface.BOLD).apply { gravity = Gravity.CENTER }
    parent.addView(LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      gravity = Gravity.CENTER
      setPadding(dp(2), dp(2), dp(2), dp(2))
      addView(label(title, 10f, COLOR_MUTED, Typeface.BOLD).apply { gravity = Gravity.CENTER })
      addView(value, matchWidth())
      addView(label(unit, 11f, COLOR_MUTED).apply { gravity = Gravity.CENTER })
    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
    return value
  }

  private fun addField(parent: LinearLayout, fieldLabel: String, hintValue: String): EditText {
    parent.addView(label(fieldLabel, 12f, COLOR_MUTED, Typeface.BOLD), LinearLayout.LayoutParams(
      ViewGroup.LayoutParams.MATCH_PARENT,
      ViewGroup.LayoutParams.WRAP_CONTENT,
    ).apply { topMargin = dp(10) })
    return EditText(this).also { field ->
      field.hint = hintValue
      field.textSize = 15f
      field.setTextColor(COLOR_TEXT)
      field.setHintTextColor(COLOR_MUTED_LIGHT)
      field.setSingleLine(true)
      field.inputType = when {
        fieldLabel.contains("IP") -> InputType.TYPE_CLASS_PHONE
        fieldLabel.contains("임계값") -> InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
        else -> InputType.TYPE_CLASS_NUMBER
      }
      field.setPadding(dp(14), dp(11), dp(14), dp(11))
      field.background = roundedBackground(COLOR_SURFACE_ALT, 12f, COLOR_BORDER)
      parent.addView(field, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(48)).apply { topMargin = dp(5) })
    }
  }

  private fun card(): LinearLayout = LinearLayout(this).apply {
    orientation = LinearLayout.VERTICAL
    setPadding(dp(18), dp(18), dp(18), dp(18))
    background = roundedBackground(Color.WHITE, 18f, COLOR_BORDER)
    elevation = dp(1).toFloat()
  }

  private fun label(value: String, size: Float, color: Int, style: Int = Typeface.NORMAL): TextView = TextView(this).apply {
    text = value
    textSize = size
    setTextColor(color)
    typeface = Typeface.create("sans-serif", style)
    includeFontPadding = false
  }

  private fun sectionTitle(value: String): TextView = label(value, 20f, COLOR_TEXT, Typeface.BOLD)

  private fun styledButton(value: String, primary: Boolean, compact: Boolean = false): Button = Button(this).apply {
    text = value
    textSize = if (compact) 13f else 15f
    isAllCaps = false
    setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL))
    setTextColor(if (primary) Color.WHITE else COLOR_TEXT)
    setPadding(dp(if (compact) 13 else 16), 0, dp(if (compact) 13 else 16), 0)
    minWidth = 0
    minHeight = dp(if (compact) 42 else 50)
    stateListAnimator = null
    elevation = 0f
    background = roundedBackground(if (primary) COLOR_ORANGE else Color.WHITE, 14f, if (primary) COLOR_ORANGE else COLOR_BORDER_STRONG)
  }

  private fun divider(): View = View(this).apply { setBackgroundColor(COLOR_BORDER) }

  private fun stopClient() {
    autoConnect.isChecked = false
    startService(Intent(this, ExternalAIService::class.java).setAction(ExternalAIService.ACTION_STOP))
    updateConnectionStatus("중지됨")
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

  private fun copyC3xBranchSshCommand() {
    val targetHost = host.text.toString().trim()
    if (!isValidIpv4Address(targetHost)) {
      Toast.makeText(this, "먼저 올바른 C3/C3X/C4 IPv4 주소를 입력하거나 자동 검색하세요.", Toast.LENGTH_LONG).show()
      return
    }
    val command = buildC3xBranchSshCommand(targetHost)
    getSystemService(ClipboardManager::class.java).setPrimaryClip(
      ClipData.newPlainText("CarrotPilot external-android-ai SSH command", command),
    )
    status.text = "SSH 브랜치 변경 명령을 복사했습니다. 완료 후 C3/C3X/C4를 재부팅하세요."
    Toast.makeText(this, "SSH 명령을 클립보드에 복사했습니다.", Toast.LENGTH_LONG).show()
  }

  private fun buildC3xBranchSshCommand(targetHost: String): String =
    "ssh $SSH_USER@$targetHost \"cd $OPENPILOT_PATH && " +
      "git fetch $DEPLOY_REPOSITORY_URL $DEPLOY_BRANCH && " +
      "if git show-ref --verify --quiet refs/heads/$DEPLOY_BRANCH; then " +
      "git switch $DEPLOY_BRANCH && git merge --ff-only FETCH_HEAD; " +
      "else git switch -c $DEPLOY_BRANCH FETCH_HEAD; fi && " +
      "git rev-parse --short HEAD\""

  private fun isValidIpv4Address(value: String): Boolean {
    val octets = value.split('.')
    return octets.size == 4 && octets.all { octet ->
      octet.isNotEmpty() && octet.length <= 3 && octet.all(Char::isDigit) && octet.toInt() in 0..255
    }
  }

  private fun confirmSelectedModelDownload() {
    if (downloadingModel) return
    val model = selectedCatalogModel
    AlertDialog.Builder(this)
      .setTitle("${model.displayName} 다운로드")
      .setMessage(
        "검증된 ${RecommendedModels.VERSION} 모델을 다운로드합니다.\n\n" +
          "용도: ${model.profileLabel}\n" +
          "정확도: ${model.accuracyLabel}\n" +
          "크기: ${"%.1f".format(model.sizeMegabytes)} MB\n" +
          "형식: ${model.formatLabel}\n" +
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
          applyModelInputPolicy(model)
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
    applyModelInputPolicy(RecommendedModels.findByUri(this, modelUri))
    updateModelLabel()
    status.text = "${model.displayName} 삭제 완료"
  }

  private fun selectCatalogModel(model: VerifiedModelSpec) {
    selectedCatalogModel = model
    applyModelInputPolicy(model)
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
    val activeModel = RecommendedModels.findByUri(this, uri)
    val config = try {
      ClientConfig(
        host = host.text.toString().trim(),
        framePort = framePort.text.toString().toInt(),
        resultPort = resultPort.text.toString().toInt(),
        threshold = threshold.text.toString().toFloat(),
        targetFps = inferenceFps.text.toString().toInt(),
        inputSize = activeModel?.fixedInputSize ?: inputSize.text.toString().toInt(),
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
    applyModelInputPolicy(RecommendedModels.findByUri(this, modelUri))
    deviceAddress.text = host.text.toString().ifBlank { "기기 검색 전" }
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
    currentModelValue.text = activeLabel
    modelLabel.text = "현재 사용 모델: $activeLabel\n" +
      "선택 모델: ${selected.displayName} · ${selected.profileLabel}\n" +
      "${selected.accuracyLabel} · ${selected.suggestedSettings}\n" +
      "${selected.formatLabel} · SHA-256 고정 검증"
    downloadModelButton.text = when {
      downloadingModel -> "모델 다운로드 중"
      selectedInstalled -> "${selected.displayName} 다시 다운로드"
      else -> "${selected.displayName} 다운로드 (${"%.1f".format(selected.sizeMegabytes)} MB)"
    }
    downloadModelButton.isEnabled = !downloadingModel
    downloadModelButton.alpha = if (downloadingModel) 0.55f else 1f
    modelSelector.isEnabled = !downloadingModel
    deleteModelButton.isEnabled = selectedInstalled && !downloadingModel
    deleteModelButton.alpha = if (deleteModelButton.isEnabled) 1f else 0.45f
  }

  private fun applyModelInputPolicy(model: VerifiedModelSpec?) {
    if (!::inputSize.isInitialized) return
    val fixedSize = model?.fixedInputSize
    if (fixedSize != null) inputSize.setText(fixedSize.toString())
    inputSize.isEnabled = fixedSize == null
    inputSize.alpha = if (fixedSize == null) 1f else 0.62f
  }

  private fun appendAnalysisLog(entry: String) {
    analysisEntries.addLast(entry)
    while (analysisEntries.size > MAX_CONSOLE_ENTRIES) analysisEntries.removeFirst()
    analysisConsole.text = analysisEntries.joinToString("\n\n")
    recentDetectionPreview.text = analysisEntries.toList().takeLast(5).joinToString("\n\n")
    recentDetectionPreview.setTextColor(COLOR_TEXT)
    analysisConsoleScroll.post { analysisConsoleScroll.fullScroll(View.FOCUS_DOWN) }
  }

  private fun clearAnalysisLog() {
    analysisEntries.clear()
    analysisConsole.text = CONSOLE_PLACEHOLDER
    recentDetectionPreview.text = RECENT_PLACEHOLDER
    recentDetectionPreview.setTextColor(COLOR_MUTED)
  }

  private fun updatePerformanceHud(intent: Intent) {
    val model = intent.getStringExtra(ExternalAIService.EXTRA_MODEL_NAME) ?: "사용자 ONNX"
    val videoFps = intent.getDoubleExtra(ExternalAIService.EXTRA_VIDEO_FPS, 0.0)
    val aiFps = intent.getDoubleExtra(ExternalAIService.EXTRA_AI_FPS, 0.0)
    val followRate = intent.getDoubleExtra(ExternalAIService.EXTRA_FOLLOW_RATE, 0.0)
    val skippedFps = intent.getDoubleExtra(ExternalAIService.EXTRA_SKIPPED_FPS, 0.0)
    val badge = intent.getStringExtra(ExternalAIService.EXTRA_ACCELERATOR_BADGE) ?: "대기"
    val detail = intent.getStringExtra(ExternalAIService.EXTRA_ACCELERATOR_DETAIL)
    currentModelValue.text = model
    videoFpsValue.text = "%.1f".format(videoFps)
    aiFpsValue.text = "%.1f".format(aiFps)
    followRateValue.text = "%.0f".format(followRate)
    skippedFpsValue.text = "%.1f".format(skippedFps)
    backendBadge.text = badge
    backendBadge.background = roundedBackground(when (badge) {
      "eNPU" -> COLOR_GREEN
      "eNPU+CPU" -> COLOR_GREEN
      "eQNN?" -> COLOR_ORANGE_DARK
      "eACCEL" -> COLOR_ORANGE_DARK
      "eCPU" -> COLOR_PURPLE
      else -> COLOR_BADGE_IDLE
    }, 18f)
    if (!detail.isNullOrBlank()) acceleratorDetail.text = detail
  }

  private fun updateConnectionStatus(message: String) {
    status.text = message
    val (chipText, headline, chipFill, chipTextColor, statusFill, statusBorder) = when {
      message.startsWith("연결됨:") -> StatusVisual("연결됨", "CarrotPilot 연결됨", COLOR_GREEN_TINT, COLOR_GREEN, COLOR_GREEN_TINT, COLOR_GREEN_BORDER)
      message.contains("검색") || message.contains("연결 중") || message.contains("준비 중") ->
        StatusVisual("연결 중", "기기 연결 중", COLOR_ORANGE_TINT, COLOR_ORANGE_DARK, COLOR_ORANGE_TINT, COLOR_ORANGE_BORDER)
      message.contains("실패") || message.contains("찾지 못함") ->
        StatusVisual("확인 필요", "연결 확인 필요", COLOR_RED_TINT, COLOR_RED, COLOR_RED_TINT, COLOR_RED_BORDER)
      message == "중지됨" -> StatusVisual("중지됨", "세션 중지됨", COLOR_SURFACE_ALT, COLOR_MUTED, COLOR_SURFACE_ALT, COLOR_BORDER)
      else -> StatusVisual("대기", "연결 준비", COLOR_SURFACE_ALT, COLOR_MUTED, COLOR_ORANGE_TINT, COLOR_ORANGE_BORDER)
    }
    connectionChip.text = chipText
    connectionChip.setTextColor(chipTextColor)
    connectionChip.background = roundedBackground(chipFill, 16f, statusBorder)
    deviceHeadline.text = headline
    deviceAddress.text = host.text.toString().ifBlank { "기기 검색 전" }
    status.background = roundedBackground(statusFill, 12f, statusBorder)
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
  private fun dp(value: Int): Int = (value * resources.displayMetrics.density + 0.5f).toInt()

  private data class StatusVisual(
    val chipText: String,
    val headline: String,
    val chipFill: Int,
    val chipTextColor: Int,
    val statusFill: Int,
    val statusBorder: Int,
  )

  companion object {
    private val COLOR_BACKGROUND = Color.rgb(250, 249, 247)
    private val COLOR_SURFACE_ALT = Color.rgb(247, 246, 243)
    private val COLOR_TEXT = Color.rgb(34, 34, 34)
    private val COLOR_MUTED = Color.rgb(110, 110, 110)
    private val COLOR_MUTED_LIGHT = Color.rgb(158, 158, 158)
    private val COLOR_BORDER = Color.rgb(231, 228, 223)
    private val COLOR_BORDER_STRONG = Color.rgb(198, 194, 188)
    private val COLOR_ORANGE = Color.rgb(255, 91, 15)
    private val COLOR_ORANGE_DARK = Color.rgb(220, 73, 0)
    private val COLOR_ORANGE_TINT = Color.rgb(255, 244, 237)
    private val COLOR_ORANGE_BORDER = Color.rgb(255, 201, 170)
    private val COLOR_GREEN = Color.rgb(28, 158, 75)
    private val COLOR_GREEN_TINT = Color.rgb(236, 250, 240)
    private val COLOR_GREEN_BORDER = Color.rgb(173, 224, 187)
    private val COLOR_RED = Color.rgb(194, 51, 51)
    private val COLOR_RED_TINT = Color.rgb(255, 240, 240)
    private val COLOR_RED_BORDER = Color.rgb(241, 182, 182)
    private val COLOR_PURPLE = Color.rgb(98, 82, 194)
    private val COLOR_BADGE_IDLE = Color.rgb(112, 112, 112)
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
    private const val RECENT_PLACEHOLDER = "아직 감지 결과가 없습니다.\n연결 후 최근 분석 5개가 이곳에 표시됩니다."
    private const val SSH_USER = "comma"
    private const val OPENPILOT_PATH = "/data/openpilot"
    private const val DEPLOY_REPOSITORY_URL = "https://github.com/junghc11/openpilot.git"
    private const val DEPLOY_BRANCH = "external-android-ai"
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
