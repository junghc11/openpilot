package ai.carrotpilot.external

import android.content.Context
import android.net.Uri
import android.os.Build
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

data class DiagnosticBundleSnapshot(
  val createdAtMillis: Long,
  val device: String,
  val appVersion: String,
  val c3xHost: String,
  val connectionStatus: String,
  val model: String,
  val accelerator: String,
  val acceleratorDetail: String,
  val videoFps: String,
  val aiFps: String,
  val frameFollowPercent: String,
  val skippedFps: String,
  val analysisLogs: List<String>,
)

data class DiagnosticUploadResult(val fileName: String, val bytesSent: Long)

object DiagnosticBundleUploader {
  private const val CONNECT_TIMEOUT_MS = 12_000
  private const val READ_TIMEOUT_MS = 30_000
  private const val MAX_RESPONSE_CHARS = 4_096

  fun createBundle(context: Context, snapshot: DiagnosticBundleSnapshot): File {
    val timestamp = fileTimeFormat().format(Date(snapshot.createdAtMillis))
    val output = File(context.cacheDir, "carrot-external-ai-$timestamp.zip")
    val metadata = JSONObject().apply {
      put("schemaVersion", 1)
      put("createdAt", isoTime(snapshot.createdAtMillis))
      put("appVersion", snapshot.appVersion)
      put("device", snapshot.device)
      put("manufacturer", Build.MANUFACTURER)
      put("model", Build.MODEL)
      put("androidVersion", Build.VERSION.RELEASE)
      put("c3xHost", snapshot.c3xHost)
      put("connectionStatus", snapshot.connectionStatus)
      put("yoloModel", snapshot.model)
      put("accelerator", snapshot.accelerator)
      put("acceleratorDetail", snapshot.acceleratorDetail)
      put("videoFps", snapshot.videoFps)
      put("aiFps", snapshot.aiFps)
      put("frameFollowPercent", snapshot.frameFollowPercent)
      put("skippedFps", snapshot.skippedFps)
      put("analysisEntryCount", snapshot.analysisLogs.size)
      put("analysisLogs", JSONArray(snapshot.analysisLogs))
      put("privacy", "diagnostics-only; no camera frames, passwords, or upload keys")
    }
    ZipOutputStream(FileOutputStream(output)).use { zip ->
      zip.putNextEntry(ZipEntry("diagnostics.json"))
      zip.write(metadata.toString(2).toByteArray(StandardCharsets.UTF_8))
      zip.closeEntry()
      zip.putNextEntry(ZipEntry("analysis.log"))
      zip.write(snapshot.analysisLogs.joinToString("\n\n").toByteArray(StandardCharsets.UTF_8))
      zip.closeEntry()
    }
    return output
  }

  fun upload(
    rawBaseUrl: String,
    sharedKey: String,
    bundle: File,
    snapshot: DiagnosticBundleSnapshot,
  ): DiagnosticUploadResult {
    require(bundle.isFile && bundle.length() > 0) { "로그 묶음을 만들지 못했습니다." }
    val baseUrl = validateBaseUrl(rawBaseUrl)
    val sessionPayload = JSONObject().apply {
      put("deviceId", deviceId())
      put("purpose", "external-ai")
      put("device", snapshot.device.take(160))
      put("appVersion", snapshot.appVersion.take(160))
      put("c3xHost", snapshot.c3xHost.take(160))
    }
    val sessionHeaders = if (sharedKey.isBlank()) emptyMap() else mapOf("X-Carrot-Upload-Key" to sharedKey)
    val session = requestJson("POST", apiUrl(baseUrl, "session"), sessionPayload, sessionHeaders)
    val token = session.optString("token").trim()
    require(token.isNotEmpty()) { "서버가 업로드 세션을 발급하지 않았습니다." }

    val remotePath = fileTimeFormat("yyyy-MM-dd").format(Date(snapshot.createdAtMillis))
    val uploadUrl = apiUrl(baseUrl, "upload", "external-ai", remotePath, bundle.name)
    val connection = open(uploadUrl, "PUT", mapOf(
      "Authorization" to "Bearer $token",
      "Content-Type" to "application/octet-stream",
      "X-File-Size" to bundle.length().toString(),
    )).apply {
      doOutput = true
      setFixedLengthStreamingMode(bundle.length())
    }
    connection.outputStream.use { output -> bundle.inputStream().use { input -> input.copyTo(output, 256 * 1024) } }
    val uploadBody = responseJson(connection)
    val remoteSize = uploadBody.optLong("size", -1L)
    require(remoteSize == bundle.length()) { "업로드 크기가 일치하지 않습니다: $remoteSize / ${bundle.length()}" }

    requestJson(
      "POST",
      apiUrl(baseUrl, "complete"),
      JSONObject().apply {
        put("ok", true)
        put("target", "external-ai")
        put("deviceId", deviceId())
        put("uploaded", 1)
        put("total", 1)
        put("fileName", bundle.name)
        put("size", bundle.length())
        put("createdAt", isoTime(snapshot.createdAtMillis))
      },
      mapOf("Authorization" to "Bearer $token"),
    )
    return DiagnosticUploadResult(bundle.name, bundle.length())
  }

  internal fun validateBaseUrl(raw: String): String {
    val value = raw.trim().trimEnd('/')
    require(value.isNotEmpty()) { "업로드 서버 URL을 입력하세요." }
    val uri = runCatching { URI(value) }.getOrElse { throw IllegalArgumentException("업로드 서버 URL 형식이 잘못되었습니다.") }
    require(uri.scheme == "https" || uri.scheme == "http") { "서버 URL은 http:// 또는 https://로 시작해야 합니다." }
    require(!uri.host.isNullOrBlank() && uri.rawUserInfo == null && uri.rawQuery == null && uri.rawFragment == null) {
      "서버 URL에는 호스트와 포트만 입력하세요."
    }
    if (uri.scheme == "http") {
      require(uri.host == "192.168.1.35") { "HTTP는 LAN 서버 192.168.1.35만 허용합니다. 다른 서버는 HTTPS를 사용하세요." }
    }
    return value
  }

  private fun requestJson(method: String, url: String, payload: JSONObject, headers: Map<String, String>): JSONObject {
    val bytes = payload.toString().toByteArray(StandardCharsets.UTF_8)
    val connection = open(url, method, headers + mapOf("Content-Type" to "application/json")).apply {
      doOutput = true
      setFixedLengthStreamingMode(bytes.size)
    }
    connection.outputStream.use { it.write(bytes) }
    return responseJson(connection)
  }

  private fun responseJson(connection: HttpURLConnection): JSONObject {
    val status = connection.responseCode
    val stream = if (status in 200..299) connection.inputStream else connection.errorStream
    val text = stream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText().take(MAX_RESPONSE_CHARS) }.orEmpty()
    connection.disconnect()
    val body = runCatching { JSONObject(text) }.getOrElse { JSONObject() }
    if (status !in 200..299 || !body.optBoolean("ok", false)) {
      val error = body.optString("error").ifBlank { text.ifBlank { "응답 없음" } }
      throw IllegalStateException("로그 서버 HTTP $status: ${error.take(300)}")
    }
    return body
  }

  private fun open(url: String, method: String, headers: Map<String, String>): HttpURLConnection =
    (URL(url).openConnection() as HttpURLConnection).apply {
      requestMethod = method
      connectTimeout = CONNECT_TIMEOUT_MS
      readTimeout = READ_TIMEOUT_MS
      useCaches = false
      setRequestProperty("Accept", "application/json")
      setRequestProperty("User-Agent", "Carrot-External-AI/${BuildConfig.VERSION_NAME}")
      headers.forEach { (name, value) -> setRequestProperty(name, value) }
    }

  private fun apiUrl(baseUrl: String, vararg parts: String): String =
    "$baseUrl/api/v1/${parts.joinToString("/") { Uri.encode(it) }}"

  private fun deviceId(): String = "${Build.MANUFACTURER}-${Build.MODEL}".replace(Regex("[^A-Za-z0-9._-]"), "-").take(80)

  private fun fileTimeFormat(pattern: String = "yyyyMMdd-HHmmss"): SimpleDateFormat =
    SimpleDateFormat(pattern, Locale.US)

  private fun isoTime(timeMillis: Long): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
    timeZone = TimeZone.getTimeZone("UTC")
  }.format(Date(timeMillis))
}
