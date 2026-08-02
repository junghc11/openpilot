import json
import socket
import time

from openpilot.selfdrive.carrot.external_ai.receiver import ExternalAIUdpReceiver


def result_payload(frame_id: int, source_timestamp_ns: int) -> bytes:
  return json.dumps({
    "protocol_version": 1,
    "frame_id": frame_id,
    "source_timestamp_monotonic_ns": source_timestamp_ns,
    "phone_receive_timestamp_ns": 1_000_000_000,
    "inference_start_timestamp_ns": 1_010_000_000,
    "inference_end_timestamp_ns": 1_040_000_000,
    "model": "dummy",
    "backend": "mock",
    "objects": [{
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.9,
      "x1": 0.3,
      "y1": 0.3,
      "x2": 0.7,
      "y2": 0.8,
    }],
  }).encode()


def send(port: int, payload: bytes) -> None:
  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
    sender.sendto(payload, ("127.0.0.1", port))


def test_udp_receiver_accepts_valid_local_result() -> None:
  now_ns = time.monotonic_ns()
  with ExternalAIUdpReceiver(host="127.0.0.1", port=0) as receiver:
    assert receiver.bound_port is not None
    send(receiver.bound_port, result_payload(1, now_ns - 20_000_000))

    result = receiver.poll(timeout_s=0.5, now_monotonic_ns=now_ns)
    stats = receiver.stats(now_monotonic_ns=now_ns)

  assert result is not None
  assert result.objects[0].class_name == "car"
  assert stats.accepted_packets == 1
  assert stats.rejected_packets == 0
  assert stats.last_sender_ip == "127.0.0.1"
  assert stats.connected


def test_udp_receiver_rejects_malformed_packet_without_crashing() -> None:
  with ExternalAIUdpReceiver(host="127.0.0.1", port=0) as receiver:
    assert receiver.bound_port is not None
    send(receiver.bound_port, b"not-json")

    assert receiver.poll(timeout_s=0.5) is None
    stats = receiver.stats()

  assert stats.accepted_packets == 0
  assert stats.rejected_packets == 1
  assert stats.last_error == "result is not valid JSON"
  assert not stats.connected


def test_udp_receiver_rejects_replayed_frame_and_keeps_last_good_result() -> None:
  first_now_ns = time.monotonic_ns()
  with ExternalAIUdpReceiver(host="127.0.0.1", port=0) as receiver:
    assert receiver.bound_port is not None
    send(receiver.bound_port, result_payload(7, first_now_ns - 10_000_000))
    first = receiver.poll(timeout_s=0.5, now_monotonic_ns=first_now_ns)

    second_now_ns = first_now_ns + 5_000_000
    send(receiver.bound_port, result_payload(7, second_now_ns - 10_000_000))
    replay = receiver.poll(timeout_s=0.5, now_monotonic_ns=second_now_ns)
    stats = receiver.stats(now_monotonic_ns=second_now_ns)

  assert first is not None
  assert replay is None
  assert receiver.tracker.last_result == first
  assert stats.accepted_packets == 1
  assert stats.rejected_packets == 1
  assert stats.last_error == "frame_id did not advance"


def test_udp_receiver_timeout_is_non_blocking_and_marks_connection_stale() -> None:
  now_ns = time.monotonic_ns()
  with ExternalAIUdpReceiver(host="127.0.0.1", port=0, connection_timeout_ms=10.0) as receiver:
    assert receiver.poll(timeout_s=0.0, now_monotonic_ns=now_ns) is None
    assert not receiver.stats(now_monotonic_ns=now_ns).connected

    assert receiver.bound_port is not None
    send(receiver.bound_port, result_payload(1, now_ns - 1_000_000))
    assert receiver.poll(timeout_s=0.5, now_monotonic_ns=now_ns) is not None
    assert not receiver.stats(now_monotonic_ns=now_ns + 11_000_000).connected


def test_udp_receiver_can_restrict_sender_ip() -> None:
  with ExternalAIUdpReceiver(host="127.0.0.1", port=0, allowed_phone_ip="192.0.2.1") as receiver:
    assert receiver.bound_port is not None
    send(receiver.bound_port, result_payload(1, time.monotonic_ns() - 1_000_000))

    assert receiver.poll(timeout_s=0.5) is None
    stats = receiver.stats()

  assert stats.rejected_senders == 1
  assert stats.rejected_packets == 0
  assert stats.last_error == "unexpected sender 127.0.0.1"
