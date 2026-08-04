#!/usr/bin/env python3
"""Small Carrot External AI diagnostic upload receiver.

Run behind an HTTPS reverse proxy for any port-forwarded deployment. A CIFS
share may be mounted on the server and supplied as --output-dir; SMB itself is
never exposed to the phone or the internet.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


MAX_JSON_BYTES = 64 * 1024
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SESSION_TTL_SECONDS = 15 * 60


class ReceiverState:
  def __init__(self, output_dir: Path, shared_key: str) -> None:
    self.output_dir = output_dir.resolve()
    self.shared_key = shared_key
    self.tokens: dict[str, float] = {}
    self.lock = threading.Lock()

  def issue_token(self) -> str:
    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    with self.lock:
      self.tokens = {key: expiry for key, expiry in self.tokens.items() if expiry > now}
      self.tokens[token] = now + SESSION_TTL_SECONDS
    return token

  def accepts(self, token: str) -> bool:
    now = time.monotonic()
    with self.lock:
      expiry = self.tokens.get(token, 0.0)
      if expiry <= now:
        self.tokens.pop(token, None)
        return False
      return True


class DiagnosticUploadHandler(BaseHTTPRequestHandler):
  server_version = "CarrotExternalAILogReceiver/1.0"

  @property
  def state(self) -> ReceiverState:
    return self.server.state  # type: ignore[attr-defined]

  def log_message(self, format: str, *args: object) -> None:
    print(f"[{self.log_date_time_string()}] {self.client_address[0]} {format % args}")

  def do_GET(self) -> None:
    if urlsplit(self.path).path == "/api/v1/health":
      self._json(HTTPStatus.OK, {"ok": True, "service": "external-ai-log-receiver"})
      return
    self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

  def do_POST(self) -> None:
    path = urlsplit(self.path).path
    if path == "/api/v1/session":
      if not self._shared_key_ok():
        self._json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "invalid upload key"})
        return
      body = self._read_json()
      if body is None:
        return
      if str(body.get("purpose") or "") != "external-ai":
        self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "unsupported purpose"})
        return
      self._json(HTTPStatus.OK, {"ok": True, "token": self.state.issue_token(), "expiresIn": SESSION_TTL_SECONDS})
      return

    if path == "/api/v1/complete":
      if not self._bearer_ok():
        return
      body = self._read_json()
      if body is None:
        return
      completion_dir = self.state.output_dir / "_complete"
      completion_dir.mkdir(parents=True, exist_ok=True)
      completion_file = completion_dir / f"{int(time.time() * 1000)}-{secrets.token_hex(4)}.json"
      completion_file.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
      self._json(HTTPStatus.OK, {"ok": True})
      return

    self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

  def do_PUT(self) -> None:
    if not self._bearer_ok():
      return
    path = urlsplit(self.path).path
    prefix = "/api/v1/upload/"
    if not path.startswith(prefix):
      self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
      return
    parts = [unquote(part) for part in path[len(prefix):].split("/")]
    if len(parts) != 3 or any(not self._safe_segment(part) for part in parts):
      self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid upload path"})
      return
    raw_length = self.headers.get("Content-Length", "")
    try:
      length = int(raw_length)
    except ValueError:
      length = -1
    if length <= 0 or length > MAX_UPLOAD_BYTES:
      self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "invalid upload size"})
      return
    expected_size = self.headers.get("X-File-Size", "")
    if expected_size and expected_size != str(length):
      self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "file size header mismatch"})
      return

    target = (self.state.output_dir / parts[0] / parts[1] / parts[2]).resolve()
    if self.state.output_dir not in target.parents:
      self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid upload target"})
      return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(4)}.part")
    remaining = length
    try:
      with temporary.open("wb") as output:
        while remaining:
          chunk = self.rfile.read(min(remaining, 1024 * 1024))
          if not chunk:
            raise OSError("upload ended before Content-Length")
          output.write(chunk)
          remaining -= len(chunk)
      temporary.replace(target)
    except Exception as error:
      temporary.unlink(missing_ok=True)
      self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)[:200]})
      return
    self._json(HTTPStatus.OK, {"ok": True, "size": length, "path": str(target.relative_to(self.state.output_dir))})

  def _read_json(self) -> dict[str, object] | None:
    try:
      length = int(self.headers.get("Content-Length", "0"))
    except ValueError:
      length = 0
    if length <= 0 or length > MAX_JSON_BYTES:
      self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid JSON size"})
      return None
    try:
      value = json.loads(self.rfile.read(length).decode("utf-8"))
    except Exception:
      self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid JSON"})
      return None
    if not isinstance(value, dict):
      self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "JSON object required"})
      return None
    return value

  def _shared_key_ok(self) -> bool:
    expected = self.state.shared_key
    if not expected:
      return True
    return hmac.compare_digest(self.headers.get("X-Carrot-Upload-Key", ""), expected)

  def _bearer_ok(self) -> bool:
    authorization = self.headers.get("Authorization", "")
    token = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    if token and self.state.accepts(token):
      return True
    self._json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "invalid or expired session"})
    return False

  @staticmethod
  def _safe_segment(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value and "\x00" not in value

  def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    self.send_response(status.value)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)


def main() -> None:
  parser = argparse.ArgumentParser(description="Receive Carrot External AI diagnostic log bundles")
  parser.add_argument("--bind", default="127.0.0.1", help="listen address; use 0.0.0.0 only on a trusted LAN")
  parser.add_argument("--port", type=int, default=8088)
  parser.add_argument("--output-dir", type=Path, default=Path("external-ai-log-bundles"))
  parser.add_argument("--shared-key", default=os.environ.get("CARROT_LOG_UPLOAD_KEY", ""))
  args = parser.parse_args()
  if args.port < 1 or args.port > 65535:
    parser.error("--port must be between 1 and 65535")

  args.output_dir.mkdir(parents=True, exist_ok=True)
  server = ThreadingHTTPServer((args.bind, args.port), DiagnosticUploadHandler)
  server.state = ReceiverState(args.output_dir, args.shared_key)  # type: ignore[attr-defined]
  print(f"Carrot External AI log receiver: http://{args.bind}:{args.port}")
  print(f"Output directory: {args.output_dir.resolve()}")
  print("Shared key: configured" if args.shared_key else "Shared key: NOT configured (trusted LAN only)")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()


if __name__ == "__main__":
  main()
