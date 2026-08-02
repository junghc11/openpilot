package ai.carrotpilot.external

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.InterruptedIOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

object RecommendedModel {
  const val DISPLAY_NAME = "YOLO11n Detection 640 FP32"
  const val VERSION = "Ultralytics v8.4.0"
  const val LICENSE_URL = "https://www.ultralytics.com/license"
  const val DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx"
  const val EXPECTED_SIZE = 10_930_182L
  const val EXPECTED_SHA256 = "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b"
  private const val FILE_NAME = "yolo11n-v8.4.0-640-fp32.onnx"

  fun installedFile(context: Context): File = File(File(context.filesDir, "models"), FILE_NAME)

  fun isInstalled(context: Context): Boolean {
    val file = installedFile(context)
    return file.isFile && file.length() == EXPECTED_SIZE
  }

  fun delete(context: Context): Boolean {
    val file = installedFile(context)
    val partial = File(file.parentFile, "$FILE_NAME.part")
    val deletedFile = Files.deleteIfExists(file.toPath())
    Files.deleteIfExists(partial.toPath())
    return deletedFile
  }

  fun download(context: Context, onProgress: (receivedBytes: Long, totalBytes: Long) -> Unit): File {
    val target = installedFile(context)
    val directory = requireNotNull(target.parentFile)
    require(directory.exists() || directory.mkdirs()) { "모델 저장 폴더를 만들 수 없습니다." }
    val partial = File(directory, "$FILE_NAME.part")
    Files.deleteIfExists(partial.toPath())

    val sourceUrl = URL(DOWNLOAD_URL)
    require(sourceUrl.protocol == "https") { "권장 모델은 HTTPS로만 다운로드할 수 있습니다." }
    val connection = sourceUrl.openConnection() as HttpURLConnection
    try {
      connection.instanceFollowRedirects = true
      connection.connectTimeout = CONNECT_TIMEOUT_MS
      connection.readTimeout = READ_TIMEOUT_MS
      connection.setRequestProperty("Accept-Encoding", "identity")
      connection.setRequestProperty("User-Agent", "Carrot-External-AI/${BuildConfig.VERSION_NAME}")
      connection.connect()
      require(connection.responseCode == HttpURLConnection.HTTP_OK) {
        "모델 다운로드 HTTP 오류: ${connection.responseCode}"
      }
      val declaredSize = connection.contentLengthLong
      require(declaredSize < 0L || declaredSize == EXPECTED_SIZE) {
        "모델 파일 크기가 예상과 다릅니다: $declaredSize"
      }

      val digest = MessageDigest.getInstance("SHA-256")
      var received = 0L
      connection.inputStream.buffered().use { input ->
        FileOutputStream(partial).use { output ->
          val buffer = ByteArray(DOWNLOAD_BUFFER_SIZE)
          while (true) {
            if (Thread.currentThread().isInterrupted) throw InterruptedIOException("모델 다운로드가 취소되었습니다.")
            val count = input.read(buffer)
            if (count < 0) break
            received += count
            require(received <= EXPECTED_SIZE) { "권장 모델 최대 크기를 초과했습니다." }
            digest.update(buffer, 0, count)
            output.write(buffer, 0, count)
            onProgress(received, EXPECTED_SIZE)
          }
          output.fd.sync()
        }
      }
      require(received == EXPECTED_SIZE) { "모델 다운로드가 완료되지 않았습니다: $received/$EXPECTED_SIZE" }
      val actualSha256 = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
      require(actualSha256 == EXPECTED_SHA256) { "모델 SHA-256 검증에 실패했습니다." }
      YoloDetector.validateModelFile(partial)

      try {
        Files.move(
          partial.toPath(),
          target.toPath(),
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING,
        )
      } catch (_: AtomicMoveNotSupportedException) {
        Files.move(partial.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
      }
      return target
    } catch (error: Exception) {
      Files.deleteIfExists(partial.toPath())
      throw error
    } finally {
      connection.disconnect()
    }
  }

  private const val CONNECT_TIMEOUT_MS = 15_000
  private const val READ_TIMEOUT_MS = 30_000
  private const val DOWNLOAD_BUFFER_SIZE = 64 * 1024
}
