package ai.carrotpilot.external

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.FileOutputStream
import java.io.InterruptedIOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

data class VerifiedModelSpec(
  val id: String,
  val displayName: String,
  val profileLabel: String,
  val map5095: Double?,
  val suggestedSettings: String,
  val formatLabel: String,
  val downloadUrl: String,
  val expectedSize: Long,
  val expectedSha256: String,
  val fileName: String,
  val fixedInputSize: Int? = null,
  val qnnOptimized: Boolean = false,
) {
  val sizeMegabytes: Double
    get() = expectedSize / 1_048_576.0

  val selectorLabel: String
    get() = "$displayName · $profileLabel · ${"%.1f".format(sizeMegabytes)} MB"

  val accuracyLabel: String
    get() = map5095?.let { "COCO mAP50-95 $it" } ?: "양자화 정확도 실기 검증 중"
}

object RecommendedModels {
  const val VERSION = "CarrotPilot raw-head QDQ v2 / Ultralytics assets v8.4.0"
  const val LICENSE_URL = "https://www.ultralytics.com/license"

  val YOLO11N_QDQ_320 = VerifiedModelSpec(
    id = "yolo11n_qdq_320",
    displayName = "YOLO11n NPU W8A16 · 320",
    profileLabel = "NPU 속도 권장",
    map5095 = null,
    suggestedSettings = "고정 입력 320 · 목표 10~15 FPS",
    formatLabel = "Static raw-head QDQ · W8A16 · QNN/HTP 우선",
    downloadUrl = "https://media.githubusercontent.com/media/junghc11/openpilot/external-android-ai/tools/android_ai_client/models/yolo11n-static-320-w8a16-raw-head-qdq.onnx",
    expectedSize = 3_064_576L,
    expectedSha256 = "6982255a239c7d66577378eb6c910c1a333b1151fa1b80f1a114e55d6eefceb8",
    fileName = "yolo11n-static-320-w8a16-raw-head-qdq.onnx",
    fixedInputSize = 320,
    qnnOptimized = true,
  )

  val YOLO11N_QDQ_640 = VerifiedModelSpec(
    id = "yolo11n_qdq_640",
    displayName = "YOLO11n NPU W8A16 · 640",
    profileLabel = "NPU 고화질",
    map5095 = null,
    suggestedSettings = "고정 입력 640 · 목표 5~10 FPS",
    formatLabel = "Static raw-head QDQ · W8A16 · QNN/HTP 우선",
    downloadUrl = "https://media.githubusercontent.com/media/junghc11/openpilot/external-android-ai/tools/android_ai_client/models/yolo11n-static-640-w8a16-raw-head-qdq.onnx",
    expectedSize = 3_064_734L,
    expectedSha256 = "156184ea20f1ae78753b0ee841e0d4177d3b6b5f61380e993bfe699dbff56b74",
    fileName = "yolo11n-static-640-w8a16-raw-head-qdq.onnx",
    fixedInputSize = 640,
    qnnOptimized = true,
  )

  val YOLO11N = VerifiedModelSpec(
    id = "yolo11n",
    displayName = "YOLO11n Dynamic FP32",
    profileLabel = "CPU 호환 · 속도 우선",
    map5095 = 39.5,
    suggestedSettings = "입력 320 · 목표 5~10 FPS",
    formatLabel = "Dynamic FP32 · CPU/NNAPI 호환",
    downloadUrl = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx",
    expectedSize = 10_930_182L,
    expectedSha256 = "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b",
    fileName = "yolo11n-v8.4.0-640-fp32.onnx",
  )

  val YOLO11S = VerifiedModelSpec(
    id = "yolo11s",
    displayName = "YOLO11s Dynamic FP32",
    profileLabel = "CPU 호환 · 균형형",
    map5095 = 47.0,
    suggestedSettings = "입력 320/416 · 목표 3~5 FPS",
    formatLabel = "Dynamic FP32 · CPU/NNAPI 호환",
    downloadUrl = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.onnx",
    expectedSize = 38_051_729L,
    expectedSha256 = "21d6650c5097610c92c76ce5e4b717976059169eaea4962035b90c6a92c07a8f",
    fileName = "yolo11s-v8.4.0-640-fp32.onnx",
  )

  val YOLO11M = VerifiedModelSpec(
    id = "yolo11m",
    displayName = "YOLO11m Dynamic FP32",
    profileLabel = "CPU 호환 · 정확도 우선",
    map5095 = 51.5,
    suggestedSettings = "입력 320 · 목표 2~3 FPS",
    formatLabel = "Dynamic FP32 · CPU/NNAPI 호환",
    downloadUrl = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.onnx",
    expectedSize = 80_673_621L,
    expectedSha256 = "8a37b5c53ff642831aa454156b548ec2cf2537827445385c3e1c1b276cb666a3",
    fileName = "yolo11m-v8.4.0-640-fp32.onnx",
  )

  val ALL = listOf(YOLO11N_QDQ_320, YOLO11N_QDQ_640, YOLO11N, YOLO11S, YOLO11M)
  val DEFAULT = YOLO11N_QDQ_320

  private val LEGACY_QDQ_REPLACEMENTS = mapOf(
    "yolo11n-static-320-w8a16-qdq.onnx" to YOLO11N_QDQ_320,
    "yolo11n-static-640-w8a16-qdq.onnx" to YOLO11N_QDQ_640,
  )

  fun installedFile(context: Context, model: VerifiedModelSpec): File =
    File(File(context.filesDir, "models"), model.fileName)

  fun isInstalled(context: Context, model: VerifiedModelSpec): Boolean {
    val file = installedFile(context, model)
    return file.isFile && file.length() == model.expectedSize
  }

  fun firstInstalled(context: Context): VerifiedModelSpec? = ALL.firstOrNull { isInstalled(context, it) }

  fun findByUri(context: Context, uri: Uri?): VerifiedModelSpec? {
    if (uri?.scheme != "file") return null
    return ALL.firstOrNull { uri.path == installedFile(context, it).absolutePath }
  }

  fun replacementForLegacyUri(uri: Uri?): VerifiedModelSpec? =
    LEGACY_QDQ_REPLACEMENTS[uri?.lastPathSegment]

  fun delete(context: Context, model: VerifiedModelSpec): Boolean {
    val file = installedFile(context, model)
    val partial = File(file.parentFile, "${model.fileName}.part")
    val deletedFile = Files.deleteIfExists(file.toPath())
    Files.deleteIfExists(partial.toPath())
    return deletedFile
  }

  fun download(
    context: Context,
    model: VerifiedModelSpec,
    onProgress: (receivedBytes: Long, totalBytes: Long) -> Unit,
  ): File {
    val target = installedFile(context, model)
    val directory = requireNotNull(target.parentFile)
    require(directory.exists() || directory.mkdirs()) { "모델 저장 폴더를 만들 수 없습니다." }
    val partial = File(directory, "${model.fileName}.part")
    Files.deleteIfExists(partial.toPath())

    val sourceUrl = URL(model.downloadUrl)
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
      require(declaredSize < 0L || declaredSize == model.expectedSize) {
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
            require(received <= model.expectedSize) { "권장 모델 최대 크기를 초과했습니다." }
            digest.update(buffer, 0, count)
            output.write(buffer, 0, count)
            onProgress(received, model.expectedSize)
          }
          output.fd.sync()
        }
      }
      require(received == model.expectedSize) { "모델 다운로드가 완료되지 않았습니다: $received/${model.expectedSize}" }
      val actualSha256 = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
      require(actualSha256 == model.expectedSha256) { "모델 SHA-256 검증에 실패했습니다." }
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
