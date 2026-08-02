from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from openpilot.selfdrive.carrot.external_ai.protocol import (
  DEFAULT_CONNECTION_TIMEOUT_MS,
  DEFAULT_MAX_LATENCY_MS,
  DEFAULT_MAX_OBJECTS,
  MAX_RESULT_BYTES,
  ExternalAIProtocolError,
  ExternalAIResult,
  ExternalAIResultTracker,
)


DEFAULT_RESULT_PORT = 7725
DEFAULT_BIND_HOST = "0.0.0.0"
MAX_DATAGRAM_BYTES = MAX_RESULT_BYTES + 1


@dataclass(frozen=True, slots=True)
class ExternalAIReceiverStats:
  accepted_packets: int
  rejected_packets: int
  rejected_senders: int
  last_sender_ip: str | None
  last_error: str | None
  connected: bool


class ExternalAIUdpReceiver:
  def __init__(
      self,
      *,
      host: str = DEFAULT_BIND_HOST,
      port: int = DEFAULT_RESULT_PORT,
      allowed_phone_ip: str = "",
      max_latency_ms: float = DEFAULT_MAX_LATENCY_MS,
      connection_timeout_ms: float = DEFAULT_CONNECTION_TIMEOUT_MS,
      max_objects: int = DEFAULT_MAX_OBJECTS,
  ) -> None:
    if not 0 <= port <= 65_535:
      raise ValueError("port must be between 0 and 65535")
    self.host = host
    self.port = port
    self.allowed_phone_ip = allowed_phone_ip.strip()
    self.tracker = ExternalAIResultTracker(
      max_latency_ms=max_latency_ms,
      connection_timeout_ms=connection_timeout_ms,
      max_objects=max_objects,
    )
    self.socket: socket.socket | None = None
    self.accepted_packets = 0
    self.rejected_packets = 0
    self.rejected_senders = 0
    self.last_sender_ip: str | None = None
    self.last_error: str | None = None

  @property
  def bound_port(self) -> int | None:
    if self.socket is None:
      return None
    return int(self.socket.getsockname()[1])

  def open(self) -> None:
    if self.socket is not None:
      return
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      receiver.bind((self.host, self.port))
    except Exception:
      receiver.close()
      raise
    self.socket = receiver

  def close(self) -> None:
    receiver, self.socket = self.socket, None
    if receiver is not None:
      receiver.close()

  def __enter__(self) -> ExternalAIUdpReceiver:
    self.open()
    return self

  def __exit__(self, _exc_type, _exc, _traceback) -> None:
    self.close()

  def poll(self, *, timeout_s: float = 0.2, now_monotonic_ns: int | None = None) -> ExternalAIResult | None:
    if timeout_s < 0.0:
      raise ValueError("timeout_s must be non-negative")
    if self.socket is None:
      self.open()
    assert self.socket is not None
    self.socket.settimeout(timeout_s)
    try:
      payload, sender = self.socket.recvfrom(MAX_DATAGRAM_BYTES)
    except (BlockingIOError, TimeoutError, socket.timeout):
      return None

    sender_ip = sender[0]
    self.last_sender_ip = sender_ip
    if self.allowed_phone_ip and sender_ip != self.allowed_phone_ip:
      self.rejected_senders += 1
      self.last_error = f"unexpected sender {sender_ip}"
      return None
    if len(payload) > MAX_RESULT_BYTES:
      self.rejected_packets += 1
      self.last_error = f"result exceeds {MAX_RESULT_BYTES} bytes"
      return None

    now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    try:
      result = self.tracker.accept(payload, now_monotonic_ns=now_ns)
    except ExternalAIProtocolError as exc:
      self.rejected_packets += 1
      self.last_error = str(exc)
      return None
    self.accepted_packets += 1
    self.last_error = None
    return result

  def stats(self, *, now_monotonic_ns: int | None = None) -> ExternalAIReceiverStats:
    return ExternalAIReceiverStats(
      accepted_packets=self.accepted_packets,
      rejected_packets=self.rejected_packets,
      rejected_senders=self.rejected_senders,
      last_sender_ip=self.last_sender_ip,
      last_error=self.last_error,
      connected=self.tracker.connected(now_monotonic_ns=now_monotonic_ns),
    )
