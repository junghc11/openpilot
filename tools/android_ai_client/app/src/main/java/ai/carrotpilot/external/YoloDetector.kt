package ai.carrotpilot.external

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class YoloDetector(modelFile: File, private val confidenceThreshold: Float) : Closeable {
  private val environment = OrtEnvironment.getEnvironment()
  private val options = OrtSession.SessionOptions().apply {
    setIntraOpNumThreads(max(1, Runtime.getRuntime().availableProcessors() / 2))
  }
  private val session = environment.createSession(modelFile.absolutePath, options)
  private val inputName = session.inputNames.first()
  private val inputShape = (session.inputInfo[inputName]?.info as? TensorInfo)?.shape
    ?: error("YOLO 입력 텐서 정보를 읽을 수 없습니다.")
  private val inputHeight = inputShape.getOrNull(2)?.takeIf { it > 0 }?.toInt() ?: 640
  private val inputWidth = inputShape.getOrNull(3)?.takeIf { it > 0 }?.toInt() ?: 640
  private val inputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * Float.SIZE_BYTES)
    .order(ByteOrder.nativeOrder())
    .asFloatBuffer()

  init {
    require(inputShape.size == 4 && (inputShape[1] == 3L || inputShape[1] == -1L)) {
      "NCHW 형식의 YOLO 입력 [1,3,H,W]만 지원합니다: ${inputShape.contentToString()}"
    }
  }

  fun detect(source: Bitmap): List<Detection> {
    val prepared = preprocess(source)
    OnnxTensor.createTensor(
      environment,
      prepared.tensor,
      longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()),
    ).use { input ->
      session.run(mapOf(inputName to input)).use { result ->
        val output = result[0] as? OnnxTensor ?: error("첫 YOLO 출력이 텐서가 아닙니다.")
        return parseOutput(output, prepared)
      }
    }
  }

  private fun preprocess(source: Bitmap): PreparedInput {
    val scale = min(inputWidth.toFloat() / source.width, inputHeight.toFloat() / source.height)
    val scaledWidth = max(1, (source.width * scale).toInt())
    val scaledHeight = max(1, (source.height * scale).toInt())
    val padX = (inputWidth - scaledWidth) / 2f
    val padY = (inputHeight - scaledHeight) / 2f
    val letterboxed = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
    Canvas(letterboxed).apply {
      drawColor(Color.rgb(114, 114, 114))
      drawBitmap(
        source,
        null,
        Rect(padX.toInt(), padY.toInt(), padX.toInt() + scaledWidth, padY.toInt() + scaledHeight),
        null,
      )
    }
    val pixels = IntArray(inputWidth * inputHeight)
    letterboxed.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
    letterboxed.recycle()
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
    session.close()
    options.close()
  }

  private data class PreparedInput(
    val tensor: FloatBuffer,
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val sourceWidth: Int,
    val sourceHeight: Int,
  )

  companion object {
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
