package ai.carrotpilot.external

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEpDevice
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import org.json.JSONArray

data class DetectionResult(
  val detections: List<Detection>,
  val trafficLightState: String,
  val trafficLightConfidence: Float,
  val preprocessMs: Double,
  val runtimeMs: Double,
  val postprocessMs: Double,
  val inputWidth: Int,
  val inputHeight: Int,
  val sceneMode: String,
  val sceneBrightness: Float,
) {
  val aiPipelineMs: Double
    get() = preprocessMs + runtimeMs + postprocessMs
}

class YoloDetector(
  modelFile: File,
  private val confidenceThreshold: Float,
  requestedInputSize: Int,
  private val classThresholds: ClassThresholds = ClassThresholds.uniform(confidenceThreshold),
  private val tryQnn: Boolean = true,
  private val qnnSkipReason: String = "",
) : Closeable {
  private val dynamicInputSize = requestedInputSize.also {
    require(it in SUPPORTED_INPUT_SIZES) {
      "YOLO 입력 크기는 ${SUPPORTED_INPUT_SIZES.joinToString()} 중 하나여야 합니다."
    }
  }
  private val environment = OrtEnvironment.getEnvironment()
  private val sessionSetup = createPreferredSession(modelFile)
  private val options = sessionSetup.options
  private val session = sessionSetup.session
  val backend = sessionSetup.backend
  val backendLabel = sessionSetup.backendLabel
  private val inputName = session.inputNames.first()
  private val inputShape = (session.inputInfo[inputName]?.info as? TensorInfo)?.shape
    ?: error("YOLO 입력 텐서 정보를 읽을 수 없습니다.")
  val inputHeight = inputShape.getOrNull(2)?.takeIf { it > 0 }?.toInt() ?: dynamicInputSize
  val inputWidth = inputShape.getOrNull(3)?.takeIf { it > 0 }?.toInt() ?: dynamicInputSize
  private val inputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * Float.SIZE_BYTES)
    .order(ByteOrder.nativeOrder())
    .asFloatBuffer()
  private val letterboxBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
  private val letterboxCanvas = Canvas(letterboxBitmap)
  private val pixels = IntArray(inputWidth * inputHeight)
  private val lowLightPolicy = LowLightPolicy()

  init {
    require(inputShape.size == 4 && (inputShape[1] == 3L || inputShape[1] == -1L)) {
      "NCHW 형식의 YOLO 입력 [1,3,H,W]만 지원합니다: ${inputShape.contentToString()}"
    }
  }

  fun detect(source: Bitmap): DetectionResult {
    val preprocessStartNs = SystemClock.elapsedRealtimeNanos()
    val prepared = preprocess(source)
    val preprocessEndNs = SystemClock.elapsedRealtimeNanos()
    val runtimeStartNs = preprocessEndNs
    OnnxTensor.createTensor(
      environment,
      prepared.tensor,
      longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()),
    ).use { input ->
      session.run(mapOf(inputName to input)).use { result ->
        val runtimeEndNs = SystemClock.elapsedRealtimeNanos()
        val output = result[0] as? OnnxTensor ?: error("첫 YOLO 출력이 텐서가 아닙니다.")
        val detections = parseOutput(output, prepared)
        val trafficLight = TrafficLightColorClassifier.classify(source, detections)
        val postprocessEndNs = SystemClock.elapsedRealtimeNanos()
        return DetectionResult(
          detections = detections,
          trafficLightState = trafficLight.state,
          trafficLightConfidence = trafficLight.confidence,
          preprocessMs = nanosToMillis(preprocessEndNs - preprocessStartNs),
          runtimeMs = nanosToMillis(runtimeEndNs - runtimeStartNs),
          postprocessMs = nanosToMillis(postprocessEndNs - runtimeEndNs),
          inputWidth = inputWidth,
          inputHeight = inputHeight,
          sceneMode = prepared.sceneMode,
          sceneBrightness = prepared.sceneBrightness,
        )
      }
    }
  }

  private fun preprocess(source: Bitmap): PreparedInput {
    val scale = min(inputWidth.toFloat() / source.width, inputHeight.toFloat() / source.height)
    val scaledWidth = max(1, (source.width * scale).toInt())
    val scaledHeight = max(1, (source.height * scale).toInt())
    val padX = (inputWidth - scaledWidth) / 2f
    val padY = (inputHeight - scaledHeight) / 2f
    letterboxCanvas.apply {
      drawColor(Color.rgb(114, 114, 114))
      drawBitmap(
        source,
        null,
        Rect(padX.toInt(), padY.toInt(), padX.toInt() + scaledWidth, padY.toInt() + scaledHeight),
        null,
      )
    }
    letterboxBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
    val left = padX.toInt().coerceIn(0, inputWidth - 1)
    val top = padY.toInt().coerceIn(0, inputHeight - 1)
    val right = (left + scaledWidth).coerceAtMost(inputWidth)
    val bottom = (top + scaledHeight).coerceAtMost(inputHeight)
    var luminanceSum = 0.0
    var luminanceSamples = 0
    val sampleStep = max(1, min(scaledWidth, scaledHeight) / 80)
    var sampleY = top
    while (sampleY < bottom) {
      var sampleX = left
      while (sampleX < right) {
        val pixel = pixels[sampleY * inputWidth + sampleX]
        luminanceSum += (0.2126 * Color.red(pixel) + 0.7152 * Color.green(pixel) + 0.0722 * Color.blue(pixel)) / 255.0
        luminanceSamples++
        sampleX += sampleStep
      }
      sampleY += sampleStep
    }
    val scene = lowLightPolicy.assess(if (luminanceSamples > 0) (luminanceSum / luminanceSamples).toFloat() else 1f)
    val planeSize = inputWidth * inputHeight
    val tensor = inputBuffer.apply { clear() }
    pixels.forEachIndexed { index, pixel ->
      tensor.put(index, lowLightPolicy.normalizeChannel(Color.red(pixel), scene.mode))
      tensor.put(planeSize + index, lowLightPolicy.normalizeChannel(Color.green(pixel), scene.mode))
      tensor.put(planeSize * 2 + index, lowLightPolicy.normalizeChannel(Color.blue(pixel), scene.mode))
    }
    tensor.rewind()
    return PreparedInput(tensor, scale, padX, padY, source.width, source.height, scene.mode, scene.brightness)
  }

  private fun parseOutput(output: OnnxTensor, prepared: PreparedInput): List<Detection> {
    val shape = (output.info as TensorInfo).shape
    require(shape.size == 3 && shape[0] == 1L) { "지원하지 않는 YOLO 출력 형상: ${shape.contentToString()}" }
    val channelsFirst = shape[1] in 84..256 && shape[2] > shape[1]
    val channels = (if (channelsFirst) shape[1] else shape[2]).toInt()
    val boxes = (if (channelsFirst) shape[2] else shape[1]).toInt()
    require(channels >= 84) { "COCO YOLO 출력에는 최소 84개 채널이 필요합니다: ${shape.contentToString()}" }
    val values = output.floatBuffer ?: error("YOLO 출력이 float 텐서가 아닙니다.")

    fun value(box: Int, channel: Int): Float = if (channelsFirst) {
      values.get(channel * boxes + box)
    } else {
      values.get(box * channels + channel)
    }

    if (channels == RAW_HEAD_CHANNELS) {
      return parseRawDflOutput(values, boxes, channelsFirst, prepared)
    }

    val candidates = ArrayList<Detection>()
    for (box in 0 until boxes) {
      var bestClass = -1
      var bestScore = 0f
      for (classId in 0 until channels - 4) {
        val score = value(box, 4 + classId)
        if (score > bestScore) {
          bestScore = score
          bestClass = classId
        }
      }
      val className = supportedClasses[bestClass] ?: continue
      if (bestScore < classThresholds.forClass(bestClass)) continue
      var centerX = value(box, 0)
      var centerY = value(box, 1)
      var width = value(box, 2)
      var height = value(box, 3)
      if (!centerX.isFinite() || !centerY.isFinite() || !width.isFinite() || !height.isFinite()) continue
      if (max(max(centerX, centerY), max(width, height)) <= 2f) {
        centerX *= inputWidth
        width *= inputWidth
        centerY *= inputHeight
        height *= inputHeight
      }
      val x1 = ((centerX - width / 2f - prepared.padX) / prepared.scale / prepared.sourceWidth).coerceIn(0f, 1f)
      val y1 = ((centerY - height / 2f - prepared.padY) / prepared.scale / prepared.sourceHeight).coerceIn(0f, 1f)
      val x2 = ((centerX + width / 2f - prepared.padX) / prepared.scale / prepared.sourceWidth).coerceIn(0f, 1f)
      val y2 = ((centerY + height / 2f - prepared.padY) / prepared.scale / prepared.sourceHeight).coerceIn(0f, 1f)
      if (x2 - x1 < 0.002f || y2 - y1 < 0.002f) continue
      candidates += Detection(bestClass, className, bestScore, x1, y1, x2, y2)
    }
    return nonMaximumSuppression(candidates).take(64)
  }

  private fun parseRawDflOutput(
    values: FloatBuffer,
    boxes: Int,
    channelsFirst: Boolean,
    prepared: PreparedInput,
  ): List<Detection> {
    fun value(box: Int, channel: Int): Float = if (channelsFirst) {
      values.get(channel * boxes + box)
    } else {
      values.get(box * RAW_HEAD_CHANNELS + channel)
    }

    val expectedBoxes = DETECTION_STRIDES.sumOf { stride ->
      (inputWidth / stride) * (inputHeight / stride)
    }
    require(boxes == expectedBoxes) {
      "YOLO raw DFL anchor 수가 입력 크기와 맞지 않습니다: $boxes != $expectedBoxes"
    }
    val candidates = ArrayList<Detection>()
    val distances = FloatArray(4)
    for (box in 0 until boxes) {
      var bestClass = -1
      var bestScore = 0f
      for (classId in 0 until COCO_CLASS_COUNT) {
        val score = sigmoid(value(box, DFL_BOX_CHANNELS + classId))
        if (score > bestScore) {
          bestScore = score
          bestClass = classId
        }
      }
      val className = supportedClasses[bestClass] ?: continue
      if (bestScore < classThresholds.forClass(bestClass)) continue

      for (side in 0 until 4) {
        val channelOffset = side * DFL_BINS
        var maxLogit = Float.NEGATIVE_INFINITY
        for (bin in 0 until DFL_BINS) {
          maxLogit = max(maxLogit, value(box, channelOffset + bin))
        }
        var weighted = 0.0
        var total = 0.0
        for (bin in 0 until DFL_BINS) {
          val probability = exp((value(box, channelOffset + bin) - maxLogit).toDouble())
          weighted += probability * bin
          total += probability
        }
        distances[side] = (weighted / total).toFloat()
      }

      var localIndex = box
      var stride = 0
      var gridWidth = 0
      for (candidateStride in DETECTION_STRIDES) {
        val candidateWidth = inputWidth / candidateStride
        val candidateHeight = inputHeight / candidateStride
        val count = candidateWidth * candidateHeight
        if (localIndex < count) {
          stride = candidateStride
          gridWidth = candidateWidth
          break
        }
        localIndex -= count
      }
      check(stride > 0 && gridWidth > 0) { "YOLO raw DFL anchor index를 해석할 수 없습니다: $box" }
      val anchorX = localIndex % gridWidth + 0.5f
      val anchorY = localIndex / gridWidth + 0.5f
      val inputX1 = (anchorX - distances[0]) * stride
      val inputY1 = (anchorY - distances[1]) * stride
      val inputX2 = (anchorX + distances[2]) * stride
      val inputY2 = (anchorY + distances[3]) * stride
      val x1 = ((inputX1 - prepared.padX) / prepared.scale / prepared.sourceWidth).coerceIn(0f, 1f)
      val y1 = ((inputY1 - prepared.padY) / prepared.scale / prepared.sourceHeight).coerceIn(0f, 1f)
      val x2 = ((inputX2 - prepared.padX) / prepared.scale / prepared.sourceWidth).coerceIn(0f, 1f)
      val y2 = ((inputY2 - prepared.padY) / prepared.scale / prepared.sourceHeight).coerceIn(0f, 1f)
      if (x2 - x1 < 0.002f || y2 - y1 < 0.002f) continue
      candidates += Detection(bestClass, className, bestScore, x1, y1, x2, y2)
    }
    return nonMaximumSuppression(candidates).take(64)
  }

  private fun sigmoid(value: Float): Float = (1.0 / (1.0 + exp(-value.coerceIn(-80f, 80f).toDouble()))).toFloat()

  private fun nonMaximumSuppression(candidates: List<Detection>): List<Detection> {
    val selected = ArrayList<Detection>()
    candidates.sortedByDescending { it.confidence }.take(512).forEach { candidate ->
      if (selected.none { it.classId == candidate.classId && intersectionOverUnion(it, candidate) > 0.45f }) {
        selected += candidate
      }
    }
    return selected
  }

  private fun intersectionOverUnion(a: Detection, b: Detection): Float {
    val intersectionWidth = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
    val intersectionHeight = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
    val intersection = intersectionWidth * intersectionHeight
    val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return if (union > 0f) intersection / union else 0f
  }

  override fun close() {
    letterboxBitmap.recycle()
    session.close()
    options.close()
  }

  private fun createPreferredSession(modelFile: File): SessionSetup {
    val candidates = mutableListOf<SessionSetup>()
    val diagnostics = mutableListOf<String>()
    var lastError: Throwable? = null
    val strictQnnAttempt = if (BuildConfig.QNN_EP_INCLUDED && tryQnn) {
      tryCreateQnnSession(modelFile, requireFullGraph = true)
    } else {
      null
    }
    strictQnnAttempt?.setup?.let(candidates::add)
    strictQnnAttempt?.error?.let { error ->
      lastError = error
      diagnostics += qnnErrorLabel("전체 그래프", error)
    }
    val mixedQnnAttempt = if (strictQnnAttempt?.setup == null && BuildConfig.QNN_EP_INCLUDED && tryQnn) {
      tryCreateQnnSession(
        modelFile,
        requireFullGraph = false,
        strictFailure = strictQnnAttempt?.error,
      )
    } else {
      null
    }
    mixedQnnAttempt?.setup?.let { setup ->
      // Some Qualcomm stacks create and run the HTP-backed mixed session but do not
      // expose provider nodes through the ORT/QNN profiling files. Keep that session
      // as a benchmark candidate: it is promoted to eNPU only when it measurably
      // outperforms the same model on the plain ORT CPU session.
      if (setup.backend == "onnxruntime-qnn-mixed-unverified") {
        diagnostics += "QNN 혼합 프로파일 증거 없음: CPU 대비 실측 벤치마크로 확인"
      }
      candidates += setup
    }
    mixedQnnAttempt?.error?.let { error ->
      lastError = error
      diagnostics += qnnErrorLabel("혼합 실행", error)
    }
    when {
      !tryQnn -> diagnostics += qnnSkipReason.ifBlank { "QNN 건너뜀: CPU 호환 모델" }
      !BuildConfig.QNN_EP_INCLUDED -> diagnostics += "QNN EP 미포함 빌드"
    }

    val nnapiOptions = createBaseOptions()
    try {
      nnapiOptions.addNnapi(EnumSet.of(
        NNAPIFlags.CPU_DISABLED,
        NNAPIFlags.USE_FP16,
      ))
      candidates += SessionSetup(
        options = nnapiOptions,
        session = createAndWarmSession(modelFile, nnapiOptions),
        backend = "onnxruntime-nnapi",
        backendLabel = "NNAPI 가속 요청(NPU/DSP/GPU · ORT CPU 혼합 가능)",
      )
    } catch (nnapiError: Exception) {
      nnapiOptions.close()
      lastError = nnapiError
      diagnostics += "NNAPI 실패: ${shortError(nnapiError)}"
    }

    val cpuOptions = createBaseOptions()
    try {
      candidates += SessionSetup(
        options = cpuOptions,
        session = createAndWarmSession(modelFile, cpuOptions),
        backend = "onnxruntime-cpu-fallback",
        backendLabel = "ONNX Runtime CPU",
      )
    } catch (cpuError: Exception) {
      cpuOptions.close()
      lastError?.let(cpuError::addSuppressed)
      lastError = cpuError
      diagnostics += "CPU 세션 실패: ${shortError(cpuError)}"
    }
    if (candidates.isEmpty()) throw IllegalStateException("사용 가능한 YOLO 실행 백엔드가 없습니다.", lastError)
    return selectBestSession(candidates, diagnostics)
  }

  private fun tryCreateQnnSession(
    modelFile: File,
    requireFullGraph: Boolean,
    strictFailure: Throwable? = null,
  ): QnnAttempt {
    return try {
      val qnnDevices = qnnEpDevices()
      if (requireFullGraph) {
        val graphId = SystemClock.elapsedRealtimeNanos()
        val qnnGraphDir = File(modelFile.parentFile, "qnn-strict-graph-$graphId").apply {
          check(mkdirs() || isDirectory) { "QNN 진단 폴더를 만들지 못했습니다: $absolutePath" }
        }
        val qnnOptions = try {
          createQnnOptions(
            qnnDevices,
            requireFullGraph = true,
            qnnGraphDir = qnnGraphDir.absolutePath,
          )
        } catch (error: Throwable) {
          inspectQnnGraphDump(qnnGraphDir)
          throw error
        }
        val session = try {
          createAndWarmSession(modelFile, qnnOptions)
        } catch (error: Throwable) {
          qnnOptions.close()
          val graphEvidence = inspectQnnGraphDump(qnnGraphDir)
          throw IllegalStateException(
            "${graphEvidence.compactLabel()} · ${diagnosticError(error)}",
            error,
          )
        }
        inspectQnnGraphDump(qnnGraphDir)
        return QnnAttempt(
          setup = SessionSetup(
            options = qnnOptions,
            session = session,
            backend = "onnxruntime-qnn",
            backendLabel = "$QNN_STACK_LABEL · Qualcomm QNN/HTP NPU(전체 그래프 · 예열 완료)",
          ),
        )
      }

      // A short-lived probe uses both the ORT timeline and QNN's HTP profiler. The real
      // inference session is recreated without profiling so continuous driving does not
      // pay the profiling overhead or keep writing trace files.
      val profileId = SystemClock.elapsedRealtimeNanos()
      val ortProfilePrefix = File(modelFile.parentFile, "ort-qnn-mixed-$profileId").absolutePath
      val qnnProfileFile = File(modelFile.parentFile, "qnn-htp-mixed-$profileId.csv")
      val qnnGraphDir = File(modelFile.parentFile, "qnn-mixed-graph-$profileId").apply {
        check(mkdirs() || isDirectory) { "QNN 진단 폴더를 만들지 못했습니다: $absolutePath" }
      }
      val ortEvidence = try {
        createQnnOptions(
          qnnDevices,
          ortProfilePrefix = ortProfilePrefix,
          qnnProfilePath = qnnProfileFile.absolutePath,
          qnnGraphDir = qnnGraphDir.absolutePath,
        ).use { probeOptions ->
          createAndWarmSession(modelFile, probeOptions).use(::finishProfilingAndInspect)
        }
      } catch (error: Throwable) {
        inspectQnnProfile(qnnProfileFile)
        inspectQnnGraphDump(qnnGraphDir)
        throw error
      }
      val qnnEvidence = inspectQnnProfile(qnnProfileFile)
      val graphEvidence = inspectQnnGraphDump(qnnGraphDir)
      val qnnObserved = ortEvidence.qnnNodeCount > 0 ||
        qnnEvidence.executeEventCount > 0 ||
        graphEvidence.graphCount > 0

      val runtimeOptions = createQnnOptions(qnnDevices)
      val runtimeSession = try {
        createAndWarmSession(modelFile, runtimeOptions)
      } catch (error: Throwable) {
        runtimeOptions.close()
        throw error
      }
      val evidenceLabel = buildList {
        if (qnnEvidence.executeEventCount > 0) {
          add("HTP 실행 ${qnnEvidence.executeEventCount}건")
        } else {
          add("HTP 실행 이벤트 미확인")
        }
        add("ORT QNN 노드 ${ortEvidence.qnnNodeCount} · CPU 노드 ${ortEvidence.cpuNodeCount}")
        add(graphEvidence.compactLabel())
        qnnEvidence.error?.let { add("QNN 프로파일 $it") }
        ortEvidence.error?.let { add("ORT 프로파일 $it") }
      }.joinToString(" · ")
      val backend = if (qnnObserved) {
        "onnxruntime-qnn-mixed"
      } else {
        "onnxruntime-qnn-mixed-unverified"
      }
      val backendLabel = if (qnnObserved) {
        "Qualcomm QNN/HTP + CPU 혼합 · $evidenceLabel · 전체 그래프 실패: ${shortError(strictFailure)}"
      } else {
        "QNN 혼합 세션 예열 완료 · $evidenceLabel · 전체 그래프 실패: ${shortError(strictFailure)}"
      }
      QnnAttempt(
        setup = SessionSetup(
          options = runtimeOptions,
          session = runtimeSession,
          backend = backend,
          backendLabel = "$QNN_STACK_LABEL · $backendLabel",
        ),
      )
    } catch (error: Exception) {
      Log.w(TAG, "QNN ${if (requireFullGraph) "full" else "mixed"} failed: ${diagnosticError(error)}")
      QnnAttempt(error = error)
    } catch (error: LinkageError) {
      Log.w(TAG, "QNN native load failed: ${diagnosticError(error)}")
      QnnAttempt(error = error)
    }
  }

  private fun createQnnOptions(
    qnnDevices: List<OrtEpDevice>,
    requireFullGraph: Boolean = false,
    ortProfilePrefix: String? = null,
    qnnProfilePath: String? = null,
    qnnGraphDir: String? = null,
  ): OrtSession.SessionOptions {
    val options = createBaseOptions()
    return try {
      if (requireFullGraph) {
        // Keep eNPU strict: every operator must stay on HTP and the warm-up must complete.
        options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
      }
      ortProfilePrefix?.let(options::enableProfiling)
      // Ultralytics names its dynamic axes batch/height/width. QNN requires concrete shapes,
      // so bind the downloaded dynamic model to the input size selected in the app.
      options.setSymbolicDimensionValue("batch", 1L)
      options.setSymbolicDimensionValue("height", dynamicInputSize.toLong())
      options.setSymbolicDimensionValue("width", dynamicInputSize.toLong())
      val providerOptions = mutableMapOf(
        "backend_path" to "libQnnHtp.so",
        "htp_performance_mode" to "sustained_high_performance",
        "htp_graph_finalization_optimization_mode" to "3",
        "enable_htp_fp16_precision" to "1",
        "offload_graph_io_quantization" to "0",
      )
      if (qnnProfilePath != null) {
        providerOptions["profiling_level"] = "detailed"
        providerOptions["profiling_file_path"] = qnnProfilePath
      }
      if (qnnGraphDir != null) {
        providerOptions["dump_json_qnn_graph"] = "1"
        providerOptions["json_qnn_graph_dir"] = qnnGraphDir
      }
      options.addExecutionProvider(qnnDevices, providerOptions)
      options
    } catch (error: Throwable) {
      options.close()
      throw error
    }
  }

  private fun qnnEpDevices(): List<OrtEpDevice> {
    synchronized(QNN_REGISTRATION_LOCK) {
      if (!qnnPluginRegistered) {
        environment.registerExecutionProviderLibrary(QNN_EP_NAME, QNN_PLUGIN_LIBRARY)
        qnnPluginRegistered = true
      }
    }
    return environment.epDevices.filter { it.epName == QNN_EP_NAME }.also {
      require(it.isNotEmpty()) { "$QNN_EP_NAME 기기를 찾지 못했습니다." }
    }
  }

  private fun finishProfilingAndInspect(session: OrtSession): OrtProfileEvidence {
    val profilePath = try {
      session.endProfiling()
    } catch (error: Exception) {
      Log.w(TAG, "ORT QNN profile finish failed: ${diagnosticError(error)}")
      return OrtProfileEvidence(error = "종료 실패: ${shortError(error)}")
    }
    val profileFile = File(profilePath)
    return try {
      val events = JSONArray(profileFile.readText())
      val providers = (0 until events.length()).mapNotNull { index ->
        events.optJSONObject(index)
          ?.optJSONObject("args")
          ?.optString("provider")
          ?.takeIf(String::isNotBlank)
      }
      OrtProfileEvidence(
        qnnNodeCount = providers.count { it.equals(QNN_EP_NAME, ignoreCase = true) },
        cpuNodeCount = providers.count { it.equals(CPU_EP_NAME, ignoreCase = true) },
      )
    } catch (error: Exception) {
      Log.w(TAG, "ORT QNN profile parse failed: ${diagnosticError(error)}")
      OrtProfileEvidence(error = "해석 실패: ${shortError(error)}")
    } finally {
      profileFile.delete()
    }
  }

  private fun inspectQnnProfile(profileFile: File): QnnProfileEvidence {
    val qnnLog = File(profileFile.parentFile, "${profileFile.nameWithoutExtension}_qnn.log")
    return try {
      if (!profileFile.isFile) {
        QnnProfileEvidence(error = "파일 없음")
      } else {
        val executeEvents = profileFile.useLines { lines ->
          lines.count { it.contains("execute", ignoreCase = true) }
        }
        QnnProfileEvidence(
          executeEventCount = executeEvents,
          error = if (executeEvents == 0) "실행 이벤트 없음" else null,
        )
      }
    } catch (error: Exception) {
      Log.w(TAG, "QNN HTP profile parse failed: ${diagnosticError(error)}")
      QnnProfileEvidence(error = "해석 실패: ${shortError(error)}")
    } finally {
      profileFile.delete()
      qnnLog.delete()
    }
  }

  private fun inspectQnnGraphDump(graphDir: File): QnnGraphEvidence {
    return try {
      val files = if (graphDir.isDirectory) {
        graphDir.walkTopDown().filter { it.isFile }.toList()
      } else {
        emptyList()
      }
      val jsonFiles = files.filter { it.extension.equals("json", ignoreCase = true) }
      QnnGraphEvidence(
        graphCount = jsonFiles.size,
        totalBytes = jsonFiles.sumOf(File::length),
        error = if (jsonFiles.isEmpty()) "JSON 없음" else null,
      )
    } catch (error: Exception) {
      Log.w(TAG, "QNN graph dump inspect failed: ${diagnosticError(error)}")
      QnnGraphEvidence(error = "확인 실패: ${shortError(error)}")
    } finally {
      if (graphDir.isDirectory) {
        graphDir.walkBottomUp().forEach(File::delete)
      }
    }
  }

  private fun createAndWarmSession(modelFile: File, sessionOptions: OrtSession.SessionOptions): OrtSession {
    val candidate = environment.createSession(modelFile.absolutePath, sessionOptions)
    try {
      val candidateInputName = candidate.inputNames.first()
      val candidateShape = (candidate.inputInfo[candidateInputName]?.info as? TensorInfo)?.shape
        ?: error("YOLO 입력 텐서 정보를 읽을 수 없습니다.")
      val warmHeight = candidateShape.getOrNull(2)?.takeIf { it > 0 }?.toInt() ?: dynamicInputSize
      val warmWidth = candidateShape.getOrNull(3)?.takeIf { it > 0 }?.toInt() ?: dynamicInputSize
      val warmBuffer = ByteBuffer.allocateDirect(warmWidth * warmHeight * 3 * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
      OnnxTensor.createTensor(
        environment,
        warmBuffer,
        longArrayOf(1, 3, warmHeight.toLong(), warmWidth.toLong()),
      ).use { input ->
        candidate.run(mapOf(candidateInputName to input)).use { }
      }
      return candidate
    } catch (error: Throwable) {
      candidate.close()
      throw error
    }
  }

  private fun selectBestSession(
    candidates: List<SessionSetup>,
    diagnostics: List<String>,
  ): SessionSetup {
    val benchmarked = candidates.mapNotNull { setup ->
      try {
        SessionBenchmark(setup, benchmarkSession(setup.session))
      } catch (error: Throwable) {
        Log.w(TAG, "${setup.backend} benchmark failed: ${diagnosticError(error)}")
        setup.close()
        null
      }
    }
    check(benchmarked.isNotEmpty()) { "모든 YOLO 백엔드 벤치마크가 실패했습니다." }

    val cpu = benchmarked.firstOrNull { it.setup.backend == "onnxruntime-cpu-fallback" }
    val fastestAccelerator = benchmarked
      .filter { it.setup.backend != "onnxruntime-cpu-fallback" }
      .minWithOrNull(compareBy<SessionBenchmark> { it.timing.p95Ms }.thenBy { it.timing.p50Ms })
    val selected = when {
      cpu == null -> fastestAccelerator ?: benchmarked.minBy { it.timing.p95Ms }
      fastestAccelerator != null && BackendAutoSelector.shouldUseAccelerator(fastestAccelerator.timing, cpu.timing) -> {
        fastestAccelerator
      }
      else -> cpu
    }

    benchmarked.filter { it !== selected }.forEach { it.setup.close() }
    val timingSummary = benchmarked.joinToString(" · ") { result ->
      "${backendShortName(result.setup.backend)} ${result.timing.compactLabel()}"
    }
    val decision = if (selected.setup.backend == "onnxruntime-cpu-fallback" && fastestAccelerator != null) {
      "CPU 자동 선택(가속기 p95 개선 10% 미만)"
    } else {
      "${backendShortName(selected.setup.backend)} 자동 선택"
    }
    val selectedSetup = if (
      selected.setup.backend == "onnxruntime-qnn-mixed-unverified" &&
      cpu != null &&
      BackendAutoSelector.shouldUseAccelerator(selected.timing, cpu.timing)
    ) {
      selected.setup.copy(
        backend = "onnxruntime-qnn-mixed-benchmarked",
        backendLabel = "${selected.setup.backendLabel} · CPU 대비 실측 가속 확인",
      )
    } else {
      selected.setup
    }
    val diagnosticSummary = diagnostics.take(3).joinToString(" · ")
    return selectedSetup.copy(
      backendLabel = listOf(
        selectedSetup.backendLabel,
        "사전 벤치마크 p50/p95 $timingSummary",
        decision,
        diagnosticSummary,
      ).filter(String::isNotBlank).joinToString(" · "),
    )
  }

  private fun benchmarkSession(candidate: OrtSession): BackendTiming {
    val candidateInputName = candidate.inputNames.first()
    val candidateShape = (candidate.inputInfo[candidateInputName]?.info as? TensorInfo)?.shape
      ?: error("YOLO 입력 텐서 정보를 읽을 수 없습니다.")
    val height = candidateShape.getOrNull(2)?.takeIf { it > 0 }?.toInt() ?: dynamicInputSize
    val width = candidateShape.getOrNull(3)?.takeIf { it > 0 }?.toInt() ?: dynamicInputSize
    val elementCount = width * height * 3
    val buffer = ByteBuffer.allocateDirect(elementCount * Float.SIZE_BYTES)
      .order(ByteOrder.nativeOrder())
      .asFloatBuffer()
    for (index in 0 until elementCount) {
      buffer.put(index, ((index * 37) % 255) / 255f)
    }
    buffer.rewind()
    val samples = ArrayList<Double>(BENCHMARK_RUNS)
    OnnxTensor.createTensor(
      environment,
      buffer,
      longArrayOf(1, 3, height.toLong(), width.toLong()),
    ).use { input ->
      repeat(BENCHMARK_WARMUP_RUNS) {
        candidate.run(mapOf(candidateInputName to input)).use { }
      }
      repeat(BENCHMARK_RUNS) {
        val startedNs = SystemClock.elapsedRealtimeNanos()
        candidate.run(mapOf(candidateInputName to input)).use { }
        samples += nanosToMillis(SystemClock.elapsedRealtimeNanos() - startedNs)
      }
    }
    return BackendAutoSelector.summarize(samples)
  }

  private fun backendShortName(backend: String): String = when (backend) {
    "onnxruntime-qnn" -> "QNN"
    "onnxruntime-qnn-mixed" -> "QNN+CPU"
    "onnxruntime-qnn-mixed-benchmarked" -> "QNN+CPU(실측)"
    "onnxruntime-qnn-mixed-unverified" -> "QNN+CPU(검증 중)"
    "onnxruntime-nnapi" -> "NNAPI"
    "onnxruntime-cpu-fallback" -> "CPU"
    else -> backend
  }

  private fun createBaseOptions() = OrtSession.SessionOptions().apply {
    setIntraOpNumThreads(max(1, Runtime.getRuntime().availableProcessors() / 2))
  }

  private fun shortError(error: Throwable?): String = error?.let(::diagnosticError)?.take(240) ?: "알 수 없음"

  private fun diagnosticError(error: Throwable): String =
    generateSequence(error) { it.cause }
      .take(4)
      .joinToString(" <- ") { cause ->
        val message = (cause.message ?: "메시지 없음").replace(Regex("\\s+"), " ").trim()
        "${cause.javaClass.simpleName}: $message"
      }
      .take(800)

  private fun qnnErrorLabel(stage: String, error: Throwable): String {
    val message = diagnosticError(error)
    val reason = when {
      message.contains("default CPU EP", ignoreCase = true) -> "모델 그래프 일부 HTP 미지원"
      message.contains("dynamic", ignoreCase = true) -> "동적 shape 미지원"
      message.contains("device", ignoreCase = true) -> "QNN HTP 기기 검색 실패"
      message.contains("library", ignoreCase = true) -> "QNN 라이브러리 로딩 실패"
      message.contains("backend", ignoreCase = true) -> "QNN HTP 백엔드 초기화 실패"
      else -> "QNN 세션 생성 실패"
    }
    return "QNN $stage 실패: $reason (${shortError(error)})"
  }

  private data class PreparedInput(
    val tensor: FloatBuffer,
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val sceneMode: String,
    val sceneBrightness: Float,
  )

  private data class SessionSetup(
    val options: OrtSession.SessionOptions,
    val session: OrtSession,
    val backend: String,
    val backendLabel: String,
  ) {
    fun close() {
      session.close()
      options.close()
    }
  }

  private data class SessionBenchmark(
    val setup: SessionSetup,
    val timing: BackendTiming,
  )

  private data class QnnAttempt(
    val setup: SessionSetup? = null,
    val error: Throwable? = null,
  )

  private data class OrtProfileEvidence(
    val qnnNodeCount: Int = 0,
    val cpuNodeCount: Int = 0,
    val error: String? = null,
  )

  private data class QnnProfileEvidence(
    val executeEventCount: Int = 0,
    val error: String? = null,
  )

  private data class QnnGraphEvidence(
    val graphCount: Int = 0,
    val totalBytes: Long = 0L,
    val error: String? = null,
  ) {
    fun compactLabel(): String = if (graphCount > 0) {
      "QNN 부분 그래프 ${graphCount}개 · JSON ${totalBytes / 1_024} KB"
    } else {
      "QNN 부분 그래프 없음${error?.let { "($it)" }.orEmpty()}"
    }
  }

  companion object {
    private const val TAG = "CarrotExternalAI"
    private const val QNN_EP_NAME = "QNNExecutionProvider"
    private const val CPU_EP_NAME = "CPUExecutionProvider"
    private const val QNN_PLUGIN_LIBRARY = "libonnxruntime_providers_qnn.so"
    private const val QNN_STACK_LABEL = "ORT 1.26.0 · QNN EP 2.4.0 · QAIRT 2.48.0"
    private const val BENCHMARK_WARMUP_RUNS = 2
    private const val BENCHMARK_RUNS = 7
    private const val DFL_BINS = 16
    private const val DFL_BOX_CHANNELS = 4 * DFL_BINS
    private const val COCO_CLASS_COUNT = 80
    private const val RAW_HEAD_CHANNELS = DFL_BOX_CHANNELS + COCO_CLASS_COUNT
    private val DETECTION_STRIDES = intArrayOf(8, 16, 32)
    private val QNN_REGISTRATION_LOCK = Any()
    @Volatile private var qnnPluginRegistered = false
    val SUPPORTED_INPUT_SIZES = setOf(320, 416, 640)

    private fun nanosToMillis(nanos: Long): Double = nanos / 1_000_000.0

    fun validateModelFile(modelFile: File) {
      require(modelFile.isFile && modelFile.length() > 0L) { "ONNX 모델 파일이 비어 있습니다." }
      val environment = OrtEnvironment.getEnvironment()
      OrtSession.SessionOptions().use { options ->
        environment.createSession(modelFile.absolutePath, options).use { session ->
          val inputName = session.inputNames.firstOrNull() ?: error("ONNX 입력이 없습니다.")
          val input = session.inputInfo[inputName]?.info as? TensorInfo ?: error("ONNX 입력이 텐서가 아닙니다.")
          require(input.type == OnnxJavaType.FLOAT) { "권장 모델 입력은 FP32여야 합니다: ${input.type}" }
          require(
            input.shape.size == 4 &&
              input.shape[0] in longArrayOf(-1, 1) &&
              input.shape[1] == 3L &&
              input.shape[2] in longArrayOf(-1, 320, 416, 640) &&
              input.shape[3] in longArrayOf(-1, 320, 416, 640) &&
              input.shape[2] == input.shape[3]
          ) {
            "권장 모델은 동적 또는 320/416/640 정사각 NCHW 입력이어야 합니다: ${input.shape.contentToString()}"
          }

          val output = session.outputInfo.values.firstOrNull()?.info as? TensorInfo
            ?: error("ONNX 출력이 텐서가 아닙니다.")
          require(output.type == OnnxJavaType.FLOAT && output.shape.size == 3 && output.shape[0] in longArrayOf(-1, 1)) {
            "권장 모델 출력 형식이 올바르지 않습니다: ${output.shape.contentToString()}"
          }
          val channelsFirst = output.shape[1] in 84..256 &&
            (output.shape[2] == -1L || output.shape[2] > output.shape[1])
          val channelsLast = output.shape[2] in 84..256 &&
            (output.shape[1] == -1L || output.shape[1] > output.shape[2])
          require(channelsFirst || channelsLast) {
            "권장 모델은 일반 COCO YOLO 출력을 사용해야 합니다: ${output.shape.contentToString()}"
          }
        }
      }
    }

    private val supportedClasses = mapOf(
      0 to "person",
      1 to "bicycle",
      2 to "car",
      3 to "motorcycle",
      5 to "bus",
      7 to "truck",
      9 to "traffic light",
      11 to "stop sign",
    )
  }
}
