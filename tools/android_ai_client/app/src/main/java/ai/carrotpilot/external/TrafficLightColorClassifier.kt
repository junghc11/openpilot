package ai.carrotpilot.external

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

data class TrafficLightColorResult(
  val state: String = UNKNOWN,
  val confidence: Float = 0f,
) {
  companion object {
    const val UNKNOWN = "unknown"
    const val RED = "red"
    const val YELLOW = "yellow"
    const val GREEN = "green"
  }
}

/** Lightweight crop color analysis for COCO's color-agnostic traffic-light detections. */
object TrafficLightColorClassifier {
  fun classify(source: Bitmap, detections: List<Detection>): TrafficLightColorResult {
    var best = TrafficLightColorResult()
    var bestWeightedConfidence = 0f
    detections.asSequence()
      .filter { it.className == "traffic light" }
      .sortedByDescending(Detection::confidence)
      .take(MAX_TRAFFIC_LIGHTS)
      .forEach { detection ->
        val candidate = classifyCrop(source, detection)
        val weightedConfidence = candidate.confidence * detection.confidence
        if (weightedConfidence > bestWeightedConfidence) {
          best = candidate.copy(confidence = weightedConfidence.coerceIn(0f, 1f))
          bestWeightedConfidence = weightedConfidence
        }
      }
    return if (bestWeightedConfidence >= MIN_RESULT_CONFIDENCE) best else TrafficLightColorResult()
  }

  private fun classifyCrop(source: Bitmap, detection: Detection): TrafficLightColorResult {
    val left = (detection.x1 * source.width).toInt().coerceIn(0, source.width - 1)
    val top = (detection.y1 * source.height).toInt().coerceIn(0, source.height - 1)
    val right = ceil(detection.x2 * source.width).toInt().coerceIn(left + 1, source.width)
    val bottom = ceil(detection.y2 * source.height).toInt().coerceIn(top + 1, source.height)
    val cropWidth = right - left
    val cropHeight = bottom - top
    val stepX = max(1, cropWidth / SAMPLE_COLUMNS)
    val stepY = max(1, cropHeight / SAMPLE_ROWS)
    val hsv = FloatArray(3)
    var sampled = 0
    var chromaticScore = 0f
    var redScore = 0f
    var yellowScore = 0f
    var greenScore = 0f

    var y = top
    while (y < bottom) {
      var x = left
      while (x < right) {
        Color.colorToHSV(source.getPixel(x, y), hsv)
        val saturation = hsv[1]
        val value = hsv[2]
        sampled++
        if (saturation >= MIN_SATURATION && value >= MIN_VALUE) {
          val weight = saturation * value * value
          chromaticScore += weight
          when {
            hsv[0] <= 24f || hsv[0] >= 336f -> redScore += weight
            hsv[0] in 25f..72f -> yellowScore += weight
            hsv[0] in 73f..175f -> greenScore += weight
          }
        }
        x += stepX
      }
      y += stepY
    }
    if (sampled == 0 || chromaticScore <= 0f) return TrafficLightColorResult()

    val scores = listOf(
      TrafficLightColorResult.RED to redScore,
      TrafficLightColorResult.YELLOW to yellowScore,
      TrafficLightColorResult.GREEN to greenScore,
    ).sortedByDescending { it.second }
    val best = scores[0]
    val second = scores[1].second
    val density = best.second / sampled
    val dominance = best.second / chromaticScore
    if (density < MIN_COLOR_DENSITY || best.second < second * MIN_DOMINANCE_RATIO) {
      return TrafficLightColorResult()
    }
    val confidence = min(1f, dominance * min(1f, density / TARGET_COLOR_DENSITY))
    return TrafficLightColorResult(best.first, confidence)
  }

  private const val MAX_TRAFFIC_LIGHTS = 6
  private const val SAMPLE_COLUMNS = 18
  private const val SAMPLE_ROWS = 36
  private const val MIN_SATURATION = 0.38f
  private const val MIN_VALUE = 0.36f
  private const val MIN_COLOR_DENSITY = 0.006f
  private const val TARGET_COLOR_DENSITY = 0.055f
  private const val MIN_DOMINANCE_RATIO = 1.18f
  private const val MIN_RESULT_CONFIDENCE = 0.22f
}
