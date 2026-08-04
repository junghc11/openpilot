package ai.carrotpilot.external

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.provider.MediaStore
import org.json.JSONArray
import org.json.JSONObject
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class SavedDiagnosticSample(val imageUri: String, val metadataUri: String, val baseName: String)

object DiagnosticSampleSaver {
  fun save(
    context: Context,
    bitmap: Bitmap,
    captureType: String,
    frame: C3XFrame,
    result: DetectionResult,
    performance: FramePerformance,
    modelName: String,
    backend: String,
  ): SavedDiagnosticSample {
    require(captureType in setOf(TYPE_FALSE_POSITIVE, TYPE_MISSED_DETECTION))
    val timestamp = SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())
    val baseName = "carrot-ai-$captureType-$timestamp-frame${frame.frameId}"
    val resolver = context.contentResolver
    val imageUri = requireNotNull(resolver.insert(
      MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
      ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "$baseName.jpg")
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CarrotExternalAI")
        put(MediaStore.Images.Media.IS_PENDING, 1)
      },
    )) { "진단 이미지 저장 공간을 열 수 없습니다." }
    try {
      resolver.openOutputStream(imageUri, "w").use { output ->
        requireNotNull(output)
        check(bitmap.compress(Bitmap.CompressFormat.JPEG, 92, output)) { "진단 이미지 압축 실패" }
      }
      resolver.update(imageUri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
    } catch (error: Throwable) {
      resolver.delete(imageUri, null, null)
      throw error
    }

    val metadata = buildMetadata(captureType, frame, result, performance, modelName, backend)
    val metadataUri = requireNotNull(resolver.insert(
      MediaStore.Downloads.EXTERNAL_CONTENT_URI,
      ContentValues().apply {
        put(MediaStore.Downloads.DISPLAY_NAME, "$baseName.json")
        put(MediaStore.Downloads.MIME_TYPE, "application/json")
        put(MediaStore.Downloads.RELATIVE_PATH, "Download/CarrotExternalAI")
        put(MediaStore.Downloads.IS_PENDING, 1)
      },
    )) { "진단 메타데이터 저장 공간을 열 수 없습니다." }
    try {
      resolver.openOutputStream(metadataUri, "w").use { output ->
        requireNotNull(output).write(metadata.toString(2).toByteArray(StandardCharsets.UTF_8))
      }
      resolver.update(metadataUri, ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }, null, null)
    } catch (error: Throwable) {
      resolver.delete(metadataUri, null, null)
      resolver.delete(imageUri, null, null)
      throw error
    }
    return SavedDiagnosticSample(imageUri.toString(), metadataUri.toString(), baseName)
  }

  internal fun buildMetadata(
    captureType: String,
    frame: C3XFrame,
    result: DetectionResult,
    performance: FramePerformance,
    modelName: String,
    backend: String,
  ) = JSONObject().apply {
    put("schema_version", 1)
    put("capture_type", captureType)
    put("frame_id", frame.frameId)
    put("source_timestamp_monotonic_ns", frame.sourceTimestampNs)
    put("source_width", frame.width)
    put("source_height", frame.height)
    put("transport", frame.encoding)
    put("model", modelName)
    put("backend", backend)
    put("input_width", result.inputWidth)
    put("input_height", result.inputHeight)
    put("scene_mode", result.sceneMode)
    put("scene_brightness", result.sceneBrightness.toDouble())
    put("traffic_light_state", result.trafficLightState)
    put("traffic_light_confidence", result.trafficLightConfidence.toDouble())
    put("phone_total_ms", performance.phoneTotalMs)
    put("effective_fps", performance.effectiveFps)
    put("performance_mode", performance.performanceMode)
    put("objects", JSONArray().apply {
      result.detections.forEach { detection ->
        put(JSONObject().apply {
          put("track_id", detection.trackId)
          put("class_id", detection.classId)
          put("class_name", detection.className)
          put("confidence", detection.confidence.toDouble())
          put("x1", detection.x1.toDouble())
          put("y1", detection.y1.toDouble())
          put("x2", detection.x2.toDouble())
          put("y2", detection.y2.toDouble())
        })
      }
    })
  }

  const val TYPE_FALSE_POSITIVE = "false_positive"
  const val TYPE_MISSED_DETECTION = "missed_detection"
}
