package ai.carrotpilot.external

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaFormat
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import java.nio.ByteBuffer
import java.util.ArrayDeque
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min

data class DecodedH264Frame(
  val frame: C3XFrame,
  val bitmap: Bitmap,
)

class ReusableH264Decoder : AutoCloseable {
  private data class PendingFrame(val frame: C3XFrame, val wantBitmap: Boolean)

  private var codec: MediaCodec? = null
  private var imageReader: ImageReader? = null
  private var callbackThread: HandlerThread? = null
  private var width = 0
  private var height = 0
  private var pixels = IntArray(0)
  private var bitmap: Bitmap? = null
  private val pendingByPts = LinkedHashMap<Long, PendingFrame>()
  private val releasedFrames = ArrayDeque<PendingFrame>()
  private val decodedImages = LinkedBlockingQueue<Image>()

  fun decode(frame: C3XFrame, wantBitmap: Boolean): DecodedH264Frame? {
    require(frame.encoding == FrameProtocol.ENCODING_H264)
    if (codec == null || frame.width != width || frame.height != height) {
      releaseCodec()
      if (frame.codecConfig.isEmpty()) return null
      configure(frame.width, frame.height, frame.codecConfig)
    }

    val activeCodec = requireNotNull(codec)
    val presentationTimeUs = frame.sourceTimestampNs / 1_000L
    var inputIndex = MediaCodec.INFO_TRY_AGAIN_LATER
    repeat(5) {
      if (inputIndex < 0) inputIndex = activeCodec.dequeueInputBuffer(INPUT_TIMEOUT_US)
    }
    check(inputIndex >= 0) { "H.264 decoder input stalled" }
    val inputBuffer = requireNotNull(activeCodec.getInputBuffer(inputIndex))
    inputBuffer.clear()
    require(frame.data.size <= inputBuffer.remaining()) { "H.264 access unit exceeds decoder input buffer" }
    inputBuffer.put(frame.data)
    val flags = if (frame.keyFrame) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
    pendingByPts[presentationTimeUs] = PendingFrame(frame, wantBitmap)
    activeCodec.queueInputBuffer(inputIndex, 0, frame.data.size, presentationTimeUs, flags)

    val info = MediaCodec.BufferInfo()
    var waitUs = OUTPUT_TIMEOUT_US
    while (true) {
      when (val outputIndex = activeCodec.dequeueOutputBuffer(info, waitUs)) {
        MediaCodec.INFO_TRY_AGAIN_LATER -> break
        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
        else -> if (outputIndex >= 0) {
          pendingByPts.remove(info.presentationTimeUs)?.let(releasedFrames::addLast)
          activeCodec.releaseOutputBuffer(outputIndex, true)
        }
      }
      waitUs = 0L
    }
    return consumeDecodedImages(if (releasedFrames.isEmpty()) 0L else IMAGE_TIMEOUT_MS)
  }

  private fun configure(frameWidth: Int, frameHeight: Int, codecConfig: ByteArray) {
    width = frameWidth
    height = frameHeight
    pixels = IntArray(width * height)
    bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    callbackThread = HandlerThread("external-ai-h264-images").apply { start() }
    imageReader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, MAX_IMAGES).also { reader ->
      reader.setOnImageAvailableListener({ source ->
        val image = try {
          source.acquireNextImage()
        } catch (_: IllegalStateException) {
          null
        }
        if (image != null) decodedImages.offer(image)
      }, Handler(requireNotNull(callbackThread).looper))
    }

    val format = MediaFormat.createVideoFormat(MIME_AVC, width, height).apply {
      setByteBuffer("csd-0", ByteBuffer.wrap(codecConfig))
      setInteger(MediaFormat.KEY_PRIORITY, 0)
      if (Build.VERSION.SDK_INT >= 30) setInteger(MediaFormat.KEY_LOW_LATENCY, 1)
    }
    codec = MediaCodec.createDecoderByType(MIME_AVC).also { decoder ->
      decoder.configure(format, requireNotNull(imageReader).surface, null, 0)
      decoder.start()
    }
  }

  private fun consumeDecodedImages(firstWaitMs: Long): DecodedH264Frame? {
    var result: DecodedH264Frame? = null
    var image = if (firstWaitMs > 0) decodedImages.poll(firstWaitMs, TimeUnit.MILLISECONDS) else decodedImages.poll()
    while (image != null) {
      val pending = if (releasedFrames.isEmpty()) null else releasedFrames.removeFirst()
      try {
        if (pending?.wantBitmap == true) {
          val output = requireNotNull(bitmap)
          imageToBitmap(image, output)
          result = DecodedH264Frame(pending.frame, output)
        }
      } finally {
        image.close()
      }
      image = decodedImages.poll()
    }
    return result
  }

  private fun imageToBitmap(image: Image, output: Bitmap) {
    require(image.format == ImageFormat.YUV_420_888)
    val planes = image.planes
    require(planes.size >= 3)
    val yPlane = planes[0]
    val uPlane = planes[1]
    val vPlane = planes[2]
    val yBuffer = yPlane.buffer
    val uBuffer = uPlane.buffer
    val vBuffer = vPlane.buffer
    val yBase = yBuffer.position()
    val uBase = uBuffer.position()
    val vBase = vBuffer.position()
    val frameWidth = min(width, image.width)
    val frameHeight = min(height, image.height)

    for (row in 0 until frameHeight) {
      val chromaRow = row / 2
      for (column in 0 until frameWidth) {
        val chromaColumn = column / 2
        val yValue = yBuffer.get(yBase + row * yPlane.rowStride + column * yPlane.pixelStride).toInt() and 0xff
        val uValue = uBuffer.get(uBase + chromaRow * uPlane.rowStride + chromaColumn * uPlane.pixelStride).toInt() and 0xff
        val vValue = vBuffer.get(vBase + chromaRow * vPlane.rowStride + chromaColumn * vPlane.pixelStride).toInt() and 0xff
        val c = max(0, yValue - 16)
        val d = uValue - 128
        val e = vValue - 128
        val red = ((298 * c + 409 * e + 128) shr 8).coerceIn(0, 255)
        val green = ((298 * c - 100 * d - 208 * e + 128) shr 8).coerceIn(0, 255)
        val blue = ((298 * c + 516 * d + 128) shr 8).coerceIn(0, 255)
        pixels[row * width + column] = (0xff shl 24) or (red shl 16) or (green shl 8) or blue
      }
    }
    output.setPixels(pixels, 0, width, 0, 0, width, height)
  }

  private fun releaseCodec() {
    while (true) decodedImages.poll()?.close() ?: break
    releasedFrames.clear()
    pendingByPts.clear()
    codec?.let { decoder ->
      try {
        decoder.stop()
      } catch (_: Exception) {
      }
      decoder.release()
    }
    codec = null
    imageReader?.close()
    imageReader = null
    callbackThread?.quitSafely()
    callbackThread?.join(1_000)
    callbackThread = null
    bitmap?.recycle()
    bitmap = null
    pixels = IntArray(0)
    width = 0
    height = 0
  }

  fun reset() = releaseCodec()

  override fun close() = releaseCodec()

  companion object {
    private const val MIME_AVC = "video/avc"
    private const val MAX_IMAGES = 4
    private const val INPUT_TIMEOUT_US = 20_000L
    private const val OUTPUT_TIMEOUT_US = 10_000L
    private const val IMAGE_TIMEOUT_MS = 20L
  }
}
