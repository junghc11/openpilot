package ai.carrotpilot.external

import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

data class ClassThresholds(
  private val values: Map<Int, Float>,
  val fallback: Float,
) {
  init {
    require(fallback in 0.1f..0.95f)
    require(values.values.all { it in 0.1f..0.95f })
  }

  fun forClass(classId: Int): Float = values[classId] ?: fallback

  fun encode(): String = SUPPORTED_CLASS_IDS.joinToString(",") { classId ->
    "$classId:${forClass(classId)}"
  }

  companion object {
    val SUPPORTED_CLASS_IDS = listOf(0, 1, 2, 3, 5, 7, 9, 11)

    fun uniform(value: Float) = ClassThresholds(emptyMap(), value)

    fun decode(encoded: String?, fallback: Float): ClassThresholds {
      val parsed = encoded.orEmpty().split(',').mapNotNull { item ->
        val parts = item.split(':', limit = 2)
        val classId = parts.getOrNull(0)?.toIntOrNull() ?: return@mapNotNull null
        val value = parts.getOrNull(1)?.toFloatOrNull() ?: return@mapNotNull null
        if (classId !in SUPPORTED_CLASS_IDS || value !in 0.1f..0.95f) null else classId to value
      }.toMap()
      return ClassThresholds(parsed, fallback)
    }
  }
}

class ObjectTracker(
  private val maximumMissedFrames: Int = 5,
  private val minimumIou: Float = 0.18f,
  private val maximumCenterDistance: Float = 0.16f,
) {
  private data class Track(var detection: Detection, var missedFrames: Int = 0)

  private val tracks = linkedMapOf<Int, Track>()
  private var nextId = 1

  fun update(detections: List<Detection>): List<Detection> {
    tracks.values.forEach { it.missedFrames++ }
    val unmatchedTracks = tracks.keys.toMutableSet()
    val assigned = ArrayList<Detection>(detections.size)
    detections.sortedByDescending(Detection::confidence).forEach { detection ->
      val matchId = unmatchedTracks.asSequence()
        .filter { tracks.getValue(it).detection.classId == detection.classId }
        .map { trackId -> trackId to matchScore(tracks.getValue(trackId).detection, detection) }
        .filter { (_, score) -> score > 0f }
        .maxByOrNull { it.second }
        ?.first
      val trackId = matchId ?: allocateTrackId()
      tracks[trackId] = Track(detection.copy(trackId = trackId))
      unmatchedTracks.remove(trackId)
      assigned += detection.copy(trackId = trackId)
    }
    tracks.entries.removeAll { (_, track) -> track.missedFrames > maximumMissedFrames }
    return assigned.sortedByDescending(Detection::confidence)
  }

  fun reset() {
    tracks.clear()
    nextId = 1
  }

  private fun allocateTrackId(): Int {
    while (tracks.containsKey(nextId)) nextId = if (nextId == Int.MAX_VALUE) 1 else nextId + 1
    return nextId.also { nextId = if (it == Int.MAX_VALUE) 1 else it + 1 }
  }

  private fun matchScore(previous: Detection, current: Detection): Float {
    val iou = intersectionOverUnion(previous, current)
    val previousX = (previous.x1 + previous.x2) * 0.5f
    val previousY = (previous.y1 + previous.y2) * 0.5f
    val currentX = (current.x1 + current.x2) * 0.5f
    val currentY = (current.y1 + current.y2) * 0.5f
    val distance = sqrt((previousX - currentX).pow(2) + (previousY - currentY).pow(2))
    if (iou < minimumIou && distance > maximumCenterDistance) return 0f
    return iou * 0.72f + (1f - distance.coerceIn(0f, 1f)) * 0.28f
  }

  private fun intersectionOverUnion(a: Detection, b: Detection): Float {
    val intersectionWidth = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
    val intersectionHeight = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
    val intersection = intersectionWidth * intersectionHeight
    val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return if (union > 0f) intersection / union else 0f
  }
}

data class StableTrafficLight(val state: String, val confidence: Float, val rawState: String)

class TrafficLightStateStabilizer(
  private val requiredSamples: Int = 3,
  private val unknownHoldSamples: Int = 5,
) {
  private var stableState = TrafficLightColorResult.UNKNOWN
  private var stableConfidence = 0f
  private var candidateState = TrafficLightColorResult.UNKNOWN
  private var candidateCount = 0
  private var unknownCount = 0

  fun update(rawState: String, rawConfidence: Float, trafficLightDetected: Boolean): StableTrafficLight {
    val normalized = rawState.takeIf { trafficLightDetected && rawConfidence >= 0.12f }
      ?: TrafficLightColorResult.UNKNOWN
    if (normalized == TrafficLightColorResult.UNKNOWN) {
      unknownCount++
      candidateCount = 0
      candidateState = TrafficLightColorResult.UNKNOWN
      if (unknownCount >= unknownHoldSamples) {
        stableState = TrafficLightColorResult.UNKNOWN
        stableConfidence = 0f
      }
    } else {
      unknownCount = 0
      if (normalized == stableState) {
        candidateCount = 0
        stableConfidence = stableConfidence * 0.65f + rawConfidence * 0.35f
      } else {
        if (candidateState == normalized) candidateCount++ else {
          candidateState = normalized
          candidateCount = 1
        }
        if (candidateCount >= requiredSamples) {
          stableState = candidateState
          stableConfidence = rawConfidence
          candidateCount = 0
        }
      }
    }
    return StableTrafficLight(stableState, stableConfidence.coerceIn(0f, 1f), normalized)
  }
}

data class SceneAssessment(val mode: String, val brightness: Float)

class LowLightPolicy(
  private val enterNightBrightness: Float = 0.25f,
  private val exitNightBrightness: Float = 0.33f,
) {
  private var night = false

  fun assess(brightness: Float): SceneAssessment {
    val safeBrightness = brightness.coerceIn(0f, 1f)
    night = if (night) safeBrightness < exitNightBrightness else safeBrightness < enterNightBrightness
    return SceneAssessment(if (night) NIGHT else DAY, safeBrightness)
  }

  fun normalizeChannel(channel: Int, sceneMode: String): Float {
    val unit = channel.coerceIn(0, 255) / 255f
    if (sceneMode != NIGHT) return unit
    return (unit.toDouble().pow(0.72) * 1.08).toFloat().coerceIn(0f, 1f)
  }

  companion object {
    const val DAY = "day"
    const val NIGHT = "night"
  }
}

data class PerformanceDecision(val effectiveFps: Int, val mode: String)

class AdaptivePerformanceGovernor(private val configuredFps: Int) {
  private var effectiveFps = configuredFps.coerceIn(1, 20)
  private var overloadWindows = 0
  private var headroomWindows = 0

  fun update(p95Ms: Double, followRate: Double, thermalStatus: Int): PerformanceDecision {
    val frameBudgetMs = 1_000.0 / effectiveFps
    val severeThermal = thermalStatus >= 3
    val overloaded = thermalStatus >= 2 || p95Ms > frameBudgetMs * 0.92 || followRate < 70.0
    if (severeThermal) {
      effectiveFps = max(1, effectiveFps / 2)
      overloadWindows = 0
      headroomWindows = 0
    } else if (overloaded) {
      overloadWindows++
      headroomWindows = 0
      if (overloadWindows >= 3) {
        effectiveFps = max(1, effectiveFps - 1)
        overloadWindows = 0
      }
    } else {
      overloadWindows = 0
      val hasHeadroom = thermalStatus <= 1 && p95Ms < frameBudgetMs * 0.58 && followRate >= 88.0
      headroomWindows = if (hasHeadroom) headroomWindows + 1 else 0
      if (headroomWindows >= 10 && effectiveFps < configuredFps) {
        effectiveFps++
        headroomWindows = 0
      }
    }
    return PerformanceDecision(
      effectiveFps = effectiveFps,
      mode = when {
        severeThermal -> "thermal"
        effectiveFps < configuredFps -> "reduced"
        else -> "normal"
      },
    )
  }
}
