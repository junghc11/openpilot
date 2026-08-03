package ai.carrotpilot.external

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.os.SystemClock
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
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
import kotlin.math.max
import kotlin.math.min

data class DetectionResult(
  val detections: List<Detection>,
  val trafficLightState: String,
  val trafficLightConfidence: Float,
  val preprocessMs: Double,
  val runtimeMs: Double,
  val postprocessMs: Double,
  val inputWidth: Int,
  val inputHeight: Int,
) {
  val aiPipelineMs: Double
    get() = preprocessMs + runtimeMs + postprocessMs
}

class YoloDetector(
  modelFile: File,
  private val confidenceThreshold: Float,
  requestedInputSize: Int,
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
    val planeSize = inputWidth * inputHeight
    val tensor = inputBuffer.apply { clear() }
    pixels.forEachIndexed { index, pixel ->
      tensor.put(index, Color.red(pixel) / 255f)
      tensor.put(planeSize + index, Color.green(pixel) / 255f)
      tensor.put(planeSize * 2 + index, Color.blue(pixel) / 255f)
    }
    tensor.rewind()
    return PreparedInput(tensor, scale, padX, padY, source.width, source.height)
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

    val candidates = ArrayList<Detection>()
    for (box in 0 until boxes) {
      var bestClass = -1
      var bestScore = confidenceThreshold
      for (classId in 0 until channels - 4) {
        val score = value(box, 4 + classId)
        if (score > bestScore) {
          bestScore = score
          bestClass = classId
        }
      }
      val className = supportedClasses[bestClass] ?: continue
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
    val qnnAttempt = if (BuildConfig.QNN_EP_INCLUDED && tryQnn) tryCreateQnnSession(modelFile) else null
    qnnAttempt?.setup?.let { return it }
    val qnnFallback = when {
      !tryQnn -> qnnSkipReason.ifBlank { "QNN 건너뜀: CPU 호환 모델" }
      !BuildConfig.QNN_EP_INCLUDED -> "QNN EP 미포함 빌드"
      qnnAttempt?.error != null -> qnnErrorLabel(qnnAttempt.error)
      else -> ""
    }

    val nnapiOptions = createBaseOptions()
    try {
      nnapiOptions.addNnapi(EnumSet.of(
        NNAPIFlags.CPU_DISABLED,
        NNAPIFlags.USE_FP16,
      ))
      return SessionSetup(
        options = nnapiOptions,
        session = createAndWarmSession(modelFile, nnapiOptions),
        backend = "onnxruntime-nnapi",
        backendLabel = listOf("NNAPI 가속 요청(NPU/DSP/GPU · 혼합 실행 가능)", qnnFallback)
          .filter(String::isNotBlank)
          .joinToString(" · "),
      )
    } catch (nnapiError: Exception) {
      nnapiOptions.close()
      val cpuOptions = createBaseOptions()
      try {
        return SessionSetup(
          options = cpuOptions,
          session = createAndWarmSession(modelFile, cpuOptions),
          backend = "onnxruntime-cpu-fallback",
          backendLabel = listOf(
            "ONNX Runtime CPU",
            "NNAPI 폴백: ${shortError(nnapiError)}",
            qnnFallback,
          )
            .filter(String::isNotBlank)
            .joinToString(" · "),
        )
      } catch (cpuError: Exception) {
        cpuOptions.close()
        cpuError.addSuppressed(nnapiError)
        qnnAttempt?.error?.let(cpuError::addSuppressed)
        throw cpuError
      }
    }
  }

  private fun tryCreateQnnSession(modelFile: File): QnnAttempt {
    val qnnOptions = createBaseOptions()
    return try {
      // A QNN session is accepted only when every operator can stay on HTP. This makes the
      // eNPU badge an actual full-graph QNN result rather than an unnoticed CPU partition.
      qnnOptions.addConfigEntry("session.disable_cpu_ep_fallback", "1")
      // Ultralytics names its dynamic axes batch/height/width. QNN requires concrete shapes,
      // so bind the downloaded dynamic model to the input size selected in the app.
      qnnOptions.setSymbolicDimensionValue("batch", 1L)
      qnnOptions.setSymbolicDimensionValue("height", dynamicInputSize.toLong())
      qnnOptions.setSymbolicDimensionValue("width", dynamicInputSize.toLong())
      qnnOptions.addQnn(mapOf(
        "backend_path" to "libQnnHtp.so",
        "htp_performance_mode" to "sustained_high_performance",
        "htp_graph_finalization_optimization_mode" to "3",
        "enable_htp_fp16_precision" to "1",
        "offload_graph_io_quantization" to "0",
      ))
      QnnAttempt(
        setup = SessionSetup(
          options = qnnOptions,
          session = createAndWarmSession(modelFile, qnnOptions),
          backend = "onnxruntime-qnn",
          backendLabel = "Qualcomm QNN/HTP NPU(전체 그래프 · 예열 완료)",
        ),
      )
    } catch (error: Exception) {
      qnnOptions.close()
      QnnAttempt(error = error)
    } catch (error: LinkageError) {
      qnnOptions.close()
      QnnAttempt(error = error)
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

  private fun createBaseOptions() = OrtSession.SessionOptions().apply {
    setIntraOpNumThreads(max(1, Runtime.getRuntime().availableProcessors() / 2))
  }

  private fun shortError(error: Throwable): String =
    (error.message ?: error.javaClass.simpleName).lineSequence().first().take(120)

  private fun qnnErrorLabel(error: Throwable): String {
    val message = error.message.orEmpty()
    val reason = when {
      message.contains("default CPU EP", ignoreCase = true) -> "모델 그래프 일부 HTP 미지원"
      message.contains("dynamic", ignoreCase = true) -> "동적 shape 미지원"
      message.contains("backend", ignoreCase = true) -> "QNN HTP 백엔드 초기화 실패"
      else -> "QNN 세션 생성 실패"
    }
    return "QNN 폴백: $reason (${shortError(error)})"
  }

  private data class PreparedInput(
    val tensor: FloatBuffer,
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val sourceWidth: Int,
    val sourceHeight: Int,
  )

  private data class SessionSetup(
    val options: OrtSession.SessionOptions,
    val session: OrtSession,
    val backend: String,
    val backendLabel: String,
  )

  private data class QnnAttempt(
    val setup: SessionSetup? = null,
    val error: Throwable? = null,
  )

  companion object {
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
