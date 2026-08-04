package ai.carrotpilot.external

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InferencePoliciesTest {
  private fun car(x1: Float, x2: Float, confidence: Float = 0.8f) = Detection(
    classId = 2,
    className = "car",
    confidence = confidence,
    x1 = x1,
    y1 = 0.2f,
    x2 = x2,
    y2 = 0.6f,
  )

  @Test
  fun trackerKeepsIdentityAcrossSmallMotionAndExpiresOldTracks() {
    val tracker = ObjectTracker(maximumMissedFrames = 1)
    val firstId = tracker.update(listOf(car(0.10f, 0.30f))).single().trackId
    val movedId = tracker.update(listOf(car(0.12f, 0.32f))).single().trackId
    assertEquals(firstId, movedId)

    tracker.update(emptyList())
    tracker.update(emptyList())
    val replacementId = tracker.update(listOf(car(0.12f, 0.32f))).single().trackId
    assertNotEquals(firstId, replacementId)
  }

  @Test
  fun trafficLightRequiresRepeatedEvidenceAndHoldsShortDropout() {
    val stabilizer = TrafficLightStateStabilizer(requiredSamples = 3, unknownHoldSamples = 3)
    repeat(2) {
      assertEquals("unknown", stabilizer.update("red", 0.8f, true).state)
    }
    assertEquals("red", stabilizer.update("red", 0.8f, true).state)
    assertEquals("red", stabilizer.update("unknown", 0f, false).state)
    assertEquals("red", stabilizer.update("unknown", 0f, false).state)
    assertEquals("unknown", stabilizer.update("unknown", 0f, false).state)
  }

  @Test
  fun lowLightModeUsesHysteresisAndBrightensNightInput() {
    val policy = LowLightPolicy()
    assertEquals("day", policy.assess(0.40f).mode)
    assertEquals("night", policy.assess(0.20f).mode)
    assertEquals("night", policy.assess(0.30f).mode)
    assertEquals("day", policy.assess(0.36f).mode)
    assertTrue(policy.normalizeChannel(50, "night") > policy.normalizeChannel(50, "day"))
  }

  @Test
  fun governorReducesSustainedOverloadAndRecoversWithHeadroom() {
    val governor = AdaptivePerformanceGovernor(configuredFps = 10)
    repeat(3) { governor.update(p95Ms = 150.0, followRate = 50.0, thermalStatus = 0) }
    assertEquals(9, governor.update(p95Ms = 30.0, followRate = 95.0, thermalStatus = 0).effectiveFps)
    var decision = PerformanceDecision(0, "")
    repeat(10) { decision = governor.update(p95Ms = 20.0, followRate = 95.0, thermalStatus = 0) }
    assertEquals(10, decision.effectiveFps)
  }

  @Test
  fun classThresholdsRoundTripAndFallBack() {
    val source = ClassThresholds(mapOf(0 to 0.42f, 9 to 0.61f), 0.35f)
    val restored = ClassThresholds.decode(source.encode(), 0.35f)
    assertEquals(0.42f, restored.forClass(0), 0.0001f)
    assertEquals(0.61f, restored.forClass(9), 0.0001f)
    assertEquals(0.35f, restored.forClass(99), 0.0001f)
  }
}
