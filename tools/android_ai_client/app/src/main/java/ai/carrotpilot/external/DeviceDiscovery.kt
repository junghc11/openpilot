package ai.carrotpilot.external

import java.io.DataInputStream
import java.net.Inet4Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.Socket
import java.util.Collections
import java.util.LinkedHashSet
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorCompletionService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

object DeviceDiscovery {
  fun findHost(framePort: Int, preferredHost: String?, shouldContinue: () -> Boolean): String? {
    require(framePort in 1..65535)
    val preferredAddress = parsePrivateIpv4(preferredHost)
    if (preferredAddress != null && shouldContinue() && isCarrotFrameServer(preferredAddress, framePort)) {
      return preferredAddress
    }

    val candidates = LinkedHashSet<String>()
    localPrivateIpv4Addresses().forEach { localAddress ->
      val octets = localAddress.address.map { it.toInt() and 0xff }
      for (lastOctet in 1..254) {
        val candidate = "${octets[0]}.${octets[1]}.${octets[2]}.$lastOctet"
        if (candidate != localAddress.hostAddress && candidate != preferredAddress) candidates.add(candidate)
      }
    }
    if (candidates.isEmpty() || !shouldContinue()) return null

    val executor = Executors.newFixedThreadPool(SCAN_WORKERS)
    val completion = ExecutorCompletionService<String?>(executor)
    return try {
      candidates.forEach { candidate ->
        completion.submit(Callable {
          if (shouldContinue() && isCarrotFrameServer(candidate, framePort)) candidate else null
        })
      }
      val deadlineNs = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(SCAN_DEADLINE_MS)
      var completed = 0
      var found: String? = null
      while (completed < candidates.size && found == null && shouldContinue() && System.nanoTime() < deadlineNs) {
        val result = completion.poll(POLL_INTERVAL_MS, TimeUnit.MILLISECONDS) ?: continue
        completed++
        found = result.get()
      }
      found
    } finally {
      executor.shutdownNow()
    }
  }

  private fun localPrivateIpv4Addresses(): List<Inet4Address> {
    val availableInterfaces = try {
      NetworkInterface.getNetworkInterfaces()
    } catch (_: Exception) {
      null
    } ?: return emptyList()
    val interfaces = Collections.list(availableInterfaces)
      .filter { network ->
        try {
          network.isUp && !network.isLoopback && !network.isVirtual
        } catch (_: Exception) {
          false
        }
      }
      .sortedByDescending { network ->
        val name = network.name.lowercase()
        name.startsWith("wlan") || name.startsWith("wifi") || name.startsWith("ap") || name.startsWith("swlan")
      }

    val subnets = LinkedHashSet<String>()
    val addresses = mutableListOf<Inet4Address>()
    for (network in interfaces) {
      for (address in Collections.list(network.inetAddresses)) {
        val ipv4 = address as? Inet4Address ?: continue
        if (!ipv4.isSiteLocalAddress) continue
        val bytes = ipv4.address
        val subnet = "${bytes[0].toInt() and 0xff}.${bytes[1].toInt() and 0xff}.${bytes[2].toInt() and 0xff}"
        if (subnets.add(subnet)) addresses.add(ipv4)
        if (addresses.size >= MAX_SUBNETS) return addresses
      }
    }
    return addresses
  }

  private fun parsePrivateIpv4(host: String?): String? = try {
    (InetAddress.getByName(host?.trim().orEmpty()) as? Inet4Address)
      ?.takeIf { it.isSiteLocalAddress }
      ?.hostAddress
  } catch (_: Exception) {
    null
  }

  private fun isCarrotFrameServer(host: String, framePort: Int): Boolean = try {
    Socket().use { socket ->
      socket.soTimeout = READ_TIMEOUT_MS
      socket.connect(InetSocketAddress(host, framePort), CONNECT_TIMEOUT_MS)
      val magic = ByteArray(4)
      DataInputStream(socket.getInputStream()).readFully(magic)
      SUPPORTED_FRAME_MAGICS.any(magic::contentEquals)
    }
  } catch (_: Exception) {
    false
  }

  private val SUPPORTED_FRAME_MAGICS = listOf(
    byteArrayOf('C'.code.toByte(), 'A'.code.toByte(), 'I'.code.toByte(), '1'.code.toByte()),
    byteArrayOf('C'.code.toByte(), 'A'.code.toByte(), 'I'.code.toByte(), '2'.code.toByte()),
  )
  private const val MAX_SUBNETS = 2
  private const val SCAN_WORKERS = 24
  private const val CONNECT_TIMEOUT_MS = 300
  private const val READ_TIMEOUT_MS = 1_000
  private const val POLL_INTERVAL_MS = 200L
  private const val SCAN_DEADLINE_MS = 8_000L
}
