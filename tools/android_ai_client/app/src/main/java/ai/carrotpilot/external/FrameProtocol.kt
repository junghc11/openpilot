package ai.carrotpilot.external

import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.charset.StandardCharsets

data class C3XFrame(
  val frameId: Long,
  val sourceTimestampNs: Long,
  val width: Int,
  val height: Int,
  val jpeg: ByteArray,
  val phoneReceiveTimestampNs: Long,
)

data class Detection(
  val classId: Int,
  val className: String,
  val confidence: Float,
  val x1: Float,
  val y1: Float,
  val x2: Float,
  val y2: Float,
)

object FrameProtocol {
  private val magic = byteArrayOf('C'.code.toByte(), 'A'.code.toByte(), 'I'.code.toByte(), '1'.code.toByte())
  private const val maxHeaderBytes = 4_096
  private const val maxJpegBytes = 2 * 1024 * 1024

  fun readFrame(input: DataInputStream): C3XFrame {
    val receivedMagic = ByteArray(4).also(input::readFully)
    require(receivedMagic.contentEquals(magic)) { "잘못된 기기 영상 매직 값" }
    val headerSize = input.readInt()
    val jpegSize = input.readInt()
    require(headerSize in 1..maxHeaderBytes) { "잘못된 영상 헤더 크기: $headerSize" }
    require(jpegSize in 1..maxJpegBytes) { "잘못된 JPEG 크기: $jpegSize" }
    val headerBytes = ByteArray(headerSize).also(input::readFully)
    val header = JSONObject(String(headerBytes, StandardCharsets.UTF_8))
    require(header.getInt("protocol_version") == 1) { "지원하지 않는 영상 규약" }
    require(header.getString("encoding") == "jpeg") { "지원하지 않는 영상 인코딩" }
    val width = header.getInt("width")
    val height = header.getInt("height")
    require(width in 1..8192 && height in 1..8192) { "잘못된 영상 해상도" }
    val jpeg = ByteArray(jpegSize).also(input::readFully)
    return C3XFrame(
      frameId = header.getLong("frame_id").also { require(it > 0) },
      sourceTimestampNs = header.getLong("source_timestamp_monotonic_ns").also { require(it > 0) },
      width = width,
      height = height,
      jpeg = jpeg,
      phoneReceiveTimestampNs = SystemClock.elapsedRealtimeNanos(),
    )
  }

  fun sendResult(
    socket: DatagramSocket,
    address: InetAddress,
    port: Int,
    frame: C3XFrame,
    inferenceStartNs: Long,
    inferenceEndNs: Long,
    detectionResult: DetectionResult,
    performance: FramePerformance,
    modelName: String,
    backend: String,
  ) {
    val objects = JSONArray()
    detectionResult.detections.forEach { detection ->
      objects.put(JSONObject().apply {
        put("class_id", detection.classId)
        put("class_name", detection.className)
        put("confidence", detection.confidence.toDouble())
        put("x1", detection.x1.toDouble())
        put("y1", detection.y1.toDouble())
        put("x2", detection.x2.toDouble())
        put("y2", detection.y2.toDouble())
      })
    }
    val json = JSONObject().apply {
      put("protocol_version", 1)
      put("frame_id", frame.frameId)
      put("source_timestamp_monotonic_ns", frame.sourceTimestampNs)
      put("phone_receive_timestamp_ns", frame.phoneReceiveTimestampNs)
      put("inference_start_timestamp_ns", inferenceStartNs)
      put("inference_end_timestamp_ns", inferenceEndNs)
      put("model", modelName.take(64))
      put("backend", backend.take(64))
      put("decode_ms", performance.decodeMs)
      put("preprocess_ms", performance.preprocessMs)
      put("runtime_ms", performance.runtimeMs)
      put("postprocess_ms", performance.postprocessMs)
      put("phone_total_ms", performance.phoneTotalMs)
      put("input_width", detectionResult.inputWidth)
      put("input_height", detectionResult.inputHeight)
      put("objects", objects)
    }.toString().toByteArray(StandardCharsets.UTF_8)
    require(json.size <= 65_507) { "탐지 결과 UDP 패킷이 너무 큽니다." }
    socket.send(DatagramPacket(json, json.size, address, port))
  }
}
