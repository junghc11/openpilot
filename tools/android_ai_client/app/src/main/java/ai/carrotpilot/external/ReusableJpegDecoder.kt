package ai.carrotpilot.external

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.Closeable

class ReusableJpegDecoder : Closeable {
  private var reusableBitmap: Bitmap? = null

  fun decode(jpeg: ByteArray): Bitmap {
    val previous = reusableBitmap?.takeUnless { it.isRecycled }
    val options = BitmapFactory.Options().apply {
      inMutable = true
      inPreferredConfig = Bitmap.Config.ARGB_8888
      inBitmap = previous
    }
    val decoded = try {
      BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, options)
    } catch (_: IllegalArgumentException) {
      previous?.recycle()
      BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, BitmapFactory.Options().apply {
        inMutable = true
        inPreferredConfig = Bitmap.Config.ARGB_8888
      })
    } ?: error("수신 JPEG 디코딩 실패")
    if (decoded !== previous && previous?.isRecycled == false) previous.recycle()
    reusableBitmap = decoded
    return decoded
  }

  override fun close() {
    reusableBitmap?.takeUnless { it.isRecycled }?.recycle()
    reusableBitmap = null
  }
}
