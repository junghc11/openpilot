package ai.carrotpilot.external

import java.util.Locale
import kotlin.math.ceil

data class BackendTiming(
  val p50Ms: Double,
  val p95Ms: Double,
  val samples: Int,
) {
  fun compactLabel(): String = String.format(Locale.US, "%.1f/%.1fms", p50Ms, p95Ms)
}

object BackendAutoSelector {
  const val MINIMUM_SPEEDUP_RATIO = 0.10

  fun summarize(samplesMs: List<Double>): BackendTiming {
    require(samplesMs.isNotEmpty()) { "benchmark samples must not be empty" }
    require(samplesMs.all { it.isFinite() && it >= 0.0 }) { "benchmark samples must be finite and non-negative" }
    val sorted = samplesMs.sorted()
    return BackendTiming(
      p50Ms = percentile(sorted, 0.50),
      p95Ms = percentile(sorted, 0.95),
      samples = sorted.size,
    )
  }

  fun shouldUseAccelerator(accelerator: BackendTiming, cpu: BackendTiming): Boolean {
    val requiredP95 = cpu.p95Ms * (1.0 - MINIMUM_SPEEDUP_RATIO)
    return accelerator.p95Ms <= requiredP95 && accelerator.p50Ms <= cpu.p50Ms
  }

  private fun percentile(sorted: List<Double>, fraction: Double): Double {
    val index = (ceil(sorted.size * fraction).toInt() - 1).coerceIn(0, sorted.lastIndex)
    return sorted[index]
  }
}
