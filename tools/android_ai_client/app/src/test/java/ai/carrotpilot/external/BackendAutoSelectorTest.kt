package ai.carrotpilot.external

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BackendAutoSelectorTest {
  @Test
  fun summarizesMedianAndTailLatency() {
    val timing = BackendAutoSelector.summarize(listOf(7.0, 2.0, 5.0, 4.0, 3.0, 6.0, 1.0))

    assertEquals(4.0, timing.p50Ms, 0.0)
    assertEquals(7.0, timing.p95Ms, 0.0)
    assertEquals(7, timing.samples)
  }

  @Test
  fun selectsAcceleratorOnlyWithMeaningfulP95Gain() {
    val cpu = BackendTiming(p50Ms = 100.0, p95Ms = 120.0, samples = 7)

    assertTrue(BackendAutoSelector.shouldUseAccelerator(
      BackendTiming(p50Ms = 70.0, p95Ms = 90.0, samples = 7),
      cpu,
    ))
    assertFalse(BackendAutoSelector.shouldUseAccelerator(
      BackendTiming(p50Ms = 90.0, p95Ms = 112.0, samples = 7),
      cpu,
    ))
    assertFalse(BackendAutoSelector.shouldUseAccelerator(
      BackendTiming(p50Ms = 101.0, p95Ms = 90.0, samples = 7),
      cpu,
    ))
  }
}
