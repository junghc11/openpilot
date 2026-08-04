package ai.carrotpilot.external

import java.util.ArrayDeque
import kotlin.math.ceil

data class FramePerformance(
  val decodeMs: Double,
  val preprocessMs: Double,
  val runtimeMs: Double,
  val postprocessMs: Double,
  val phoneTotalMs: Double,
  val effectiveFps: Int = 0,
  val performanceMode: String = "normal",
) {
  val aiPipelineMs: Double
    get() = preprocessMs + runtimeMs + postprocessMs
}

data class PerformanceSummary(
  val samples: Int,
  val averageDecodeMs: Double,
  val averagePreprocessMs: Double,
  val averageRuntimeMs: Double,
  val averagePostprocessMs: Double,
  val averagePhoneTotalMs: Double,
  val p95PhoneTotalMs: Double,
)

class RollingPerformanceStats(private val capacity: Int = 120) {
  private val samples = ArrayDeque<FramePerformance>(capacity)

  init {
    require(capacity > 0) { "성능 표본 용량은 양수여야 합니다." }
  }

  fun add(sample: FramePerformance) {
    require(
      listOf(
        sample.decodeMs,
        sample.preprocessMs,
        sample.runtimeMs,
        sample.postprocessMs,
        sample.phoneTotalMs,
      ).all { it.isFinite() && it >= 0.0 }
    ) { "성능 시간은 유한한 0 이상의 값이어야 합니다." }
    if (samples.size == capacity) samples.removeFirst()
    samples.addLast(sample)
  }

  fun summary(): PerformanceSummary {
    if (samples.isEmpty()) return PerformanceSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return PerformanceSummary(
      samples = samples.size,
      averageDecodeMs = samples.averageOf { it.decodeMs },
      averagePreprocessMs = samples.averageOf { it.preprocessMs },
      averageRuntimeMs = samples.averageOf { it.runtimeMs },
      averagePostprocessMs = samples.averageOf { it.postprocessMs },
      averagePhoneTotalMs = samples.averageOf { it.phoneTotalMs },
      p95PhoneTotalMs = percentile95(samples.map { it.phoneTotalMs }),
    )
  }

  private fun percentile95(values: List<Double>): Double {
    val sorted = values.sorted()
    val index = (ceil(sorted.size * 0.95).toInt() - 1).coerceIn(sorted.indices)
    return sorted[index]
  }

  private inline fun ArrayDeque<FramePerformance>.averageOf(selector: (FramePerformance) -> Double): Double =
    sumOf(selector) / size
}
