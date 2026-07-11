from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import PIPE, Popen


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "sq8-openwebui-http-client.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "sq8_openwebui_http_client", TOOL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


ROLE_EVENT = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
CONTENT_EVENT = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
USAGE_EVENT = b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
DONE_EVENT = b"data: [DONE]\n\n"


class EvidenceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    credential = ""
    requests: list[dict[str, object]] = []
    requests_lock = threading.Lock()
    role_sent = threading.Event()
    allow_content = threading.Event()
    hold_sent = threading.Event()
    release_handlers = threading.Event()

    def log_message(self, _format, *args):
        del args

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with self.requests_lock:
            self.requests.append(
                {
                    "path": self.path,
                    "body": body,
                    "authorization": self.headers.get("Authorization"),
                }
            )
        if self.path == "/full":
            payload = ROLE_EVENT + CONTENT_EVENT + USAGE_EVENT + DONE_EVENT
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Duplicate", "first")
            self.send_header("X-Duplicate", "second")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            return
        if self.path == "/split":
            self._stream_headers()
            pieces = (
                b'data: {"choices":[{"delta":{"con',
                b'tent":"hel',
                b'lo"}}]}\n',
                b"\n" + DONE_EVENT,
            )
            for piece in pieces:
                try:
                    self.wfile.write(piece)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(0.04)
            return
        if self.path == "/role":
            self._stream_headers()
            self.wfile.write(ROLE_EVENT)
            self.wfile.flush()
            type(self).role_sent.set()
            type(self).allow_content.wait(3.0)
            try:
                self.wfile.write(CONTENT_EVENT)
                self.wfile.flush()
                time.sleep(0.05)
                self.wfile.write(DONE_EVENT)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path == "/hold":
            self._stream_headers()
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
            type(self).hold_sent.set()
            type(self).release_handlers.wait(3.0)
            return
        self.send_error(404)

    def _stream_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


class ClientProcess:
    def __init__(self, base_url: str, key_file: Path, *, chunk_bytes: int = 65536):
        self.process = Popen(
            [
                sys.executable,
                str(TOOL_PATH),
                "--base-url",
                base_url,
                "--api-key-file",
                str(key_file),
                "--socket-timeout-seconds",
                "5",
                "--read-chunk-bytes",
                str(chunk_bytes),
            ],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            bufsize=0,
        )
        assert self.process.stdout is not None
        self._events: queue.Queue[tuple[bytes, dict[str, object] | None]] = (
            queue.Queue()
        )
        self.raw_stdout: list[bytes] = []
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        ready = self.read_event()
        if ready["event"] != "ready":
            raise AssertionError(f"first event was not ready: {ready}")

    def _read_stdout(self):
        assert self.process.stdout is not None
        for raw in self.process.stdout:
            self.raw_stdout.append(raw)
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = None
            self._events.put((raw, value))

    def send(self, value: dict[str, object]):
        assert self.process.stdin is not None
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.process.stdin.write(raw + b"\n")
        self.process.stdin.flush()

    def send_raw(self, raw: bytes):
        assert self.process.stdin is not None
        self.process.stdin.write(raw + b"\n")
        self.process.stdin.flush()

    def read_event(self, timeout: float = 3.0) -> dict[str, object]:
        try:
            raw, value = self._events.get(timeout=timeout)
        except queue.Empty as error:
            raise AssertionError("timed out waiting for client event") from error
        if value is None or not isinstance(value, dict):
            raise AssertionError(f"invalid client output: {raw!r}")
        return value

    def assert_no_event(self, timeout: float = 0.15):
        try:
            raw, _value = self._events.get(timeout=timeout)
        except queue.Empty:
            return
        raise AssertionError(f"unexpected client event: {raw!r}")

    def events_through_end(self, timeout: float = 5.0):
        events = []
        deadline = time.monotonic() + timeout
        while True:
            event = self.read_event(max(0.01, deadline - time.monotonic()))
            events.append(event)
            if event["event"] == "http_response_end":
                return events

    def shutdown(self):
        self.send({"schema_version": TOOL.COMMAND_SCHEMA, "command": "shutdown"})
        event = self.read_event()
        if event["event"] != "shutdown_complete":
            raise AssertionError(f"unexpected shutdown event: {event}")
        self.process.wait(timeout=3.0)

    def stderr(self) -> bytes:
        if self.process.poll() is None:
            return b""
        assert self.process.stderr is not None
        return self.process.stderr.read()

    def terminate(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=3.0)
        if self.process.stdin is not None:
            self.process.stdin.close()
        self._reader.join(timeout=1.0)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()


def request_command(
    request_key: str,
    target: str,
    *,
    close_on_content: bool,
    body: bytes = b'{"model":"test"}',
):
    return {
        "schema_version": TOOL.COMMAND_SCHEMA,
        "command": "request",
        "request_key": request_key,
        "method": "POST",
        "target": target,
        "body_base64": base64.b64encode(body).decode("ascii"),
        "authorization_mode": "valid_bearer",
        "close_on_first_nonempty_sse_content": close_on_content,
    }


class StrictInputTests(unittest.TestCase):
    def test_duplicate_nonfinite_unknown_and_oversize_fail_closed(self):
        for raw, message in (
            (b'{"a":1,"a":2}', "duplicate"),
            (b'{"a":NaN}', "non-finite"),
            (b'{"a":1e999}', "non-finite"),
        ):
            with self.assertRaisesRegex(TOOL.ClientError, message):
                TOOL.strict_json_bytes(raw, "test", maximum=1024)
        with self.assertRaisesRegex(TOOL.ClientError, "size limit"):
            TOOL.strict_json_bytes(b" " * 17, "test", maximum=16)

        command = request_command("req-unknown", "/full", close_on_content=False)
        command["unknown"] = True
        with self.assertRaisesRegex(TOOL.ClientError, "fields differ"):
            TOOL.parse_command_line(json.dumps(command).encode("utf-8"))

        with self.assertRaisesRegex(TOOL.ClientError, "LF termination"):
            TOOL.read_bounded_line(io.BytesIO(b"{}\r\n"))

    def test_api_key_is_regular_bounded_single_line_and_nofollow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good"
            good.write_bytes(b"valid-key\n")
            self.assertEqual(TOOL.read_api_key(good), b"valid-key")

            multiline = root / "multiline"
            multiline.write_bytes(b"line-one\nline-two\n")
            with self.assertRaisesRegex(TOOL.ClientError, "exactly one"):
                TOOL.read_api_key(multiline)

            oversized = root / "oversized"
            oversized.write_bytes(b"x" * (TOOL.MAX_API_KEY_FILE_BYTES + 1))
            with self.assertRaisesRegex(TOOL.ClientError, "size"):
                TOOL.read_api_key(oversized)

            link = root / "link"
            os.symlink(good, link)
            with self.assertRaisesRegex(TOOL.ClientError, "without following"):
                TOOL.read_api_key(link)

    def test_incremental_sse_trigger_ignores_role_and_handles_split_content(self):
        parser = TOOL.StrictSseContentTrigger()
        self.assertFalse(parser.feed(ROLE_EVENT))
        split_at = len(CONTENT_EVENT) // 2
        self.assertFalse(parser.feed(CONTENT_EVENT[:split_at]))
        self.assertTrue(parser.feed(CONTENT_EVENT[split_at:]))


class HttpClientIntegrationTests(unittest.TestCase):
    def setUp(self):
        EvidenceHandler.requests = []
        EvidenceHandler.role_sent = threading.Event()
        EvidenceHandler.allow_content = threading.Event()
        EvidenceHandler.hold_sent = threading.Event()
        EvidenceHandler.release_handlers = threading.Event()
        self.temporary = tempfile.TemporaryDirectory()
        self.key_file = Path(self.temporary.name) / "api-key"
        self.secret = "test-api-key-never-emit-92f641dd"
        self.key_file.write_text(self.secret + "\n", encoding="ascii")
        EvidenceHandler.credential = self.secret
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EvidenceHandler)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.clients: list[ClientProcess] = []

    def tearDown(self):
        EvidenceHandler.allow_content.set()
        EvidenceHandler.release_handlers.set()
        for client in self.clients:
            client.terminate()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=3.0)
        self.temporary.cleanup()

    def client(self, *, chunk_bytes: int = 65536) -> ClientProcess:
        client = ClientProcess(self.base_url, self.key_file, chunk_bytes=chunk_bytes)
        self.clients.append(client)
        return client

    def test_full_sse_raw_events_order_timestamps_hashes_and_secret_absence(self):
        client = self.client()
        body = b'{"model":"test","stream":true}'
        client.send(
            request_command("req-full", "/full", close_on_content=False, body=body)
        )
        events = client.events_through_end()
        event_names = [event["event"] for event in events]
        self.assertEqual(event_names[0:2], ["http_request", "http_response_start"])
        self.assertEqual(event_names[-1], "http_response_end")
        self.assertTrue(all(name == "http_body_chunk" for name in event_names[2:-1]))

        request = events[0]
        response_start = events[1]
        chunks = events[2:-1]
        response_end = events[-1]
        self.assertLessEqual(
            request["connect_completed_monotonic_ns"],
            request["write_started_monotonic_ns"],
        )
        self.assertLessEqual(
            request["write_started_monotonic_ns"],
            request["last_body_byte_sent_monotonic_ns"],
        )
        self.assertLessEqual(
            request["last_body_byte_sent_monotonic_ns"],
            response_start["observed_monotonic_ns"],
        )
        self.assertEqual(request["body_bytes"], len(body))
        self.assertEqual(request["body_sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(request["headers"]["authorization_mode"], "valid_bearer")
        self.assertEqual(response_start["status"], 200)
        duplicate_values = [
            pair[1]
            for pair in response_start["headers"]
            if pair[0].lower() == "x-duplicate"
        ]
        self.assertEqual(duplicate_values, ["first", "second"])

        decoded_chunks = []
        previous_time = response_start["observed_monotonic_ns"]
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["chunk_index"], index)
            decoded = base64.b64decode(chunk["body_base64"], validate=True)
            decoded_chunks.append(decoded)
            self.assertEqual(chunk["body_bytes"], len(decoded))
            self.assertEqual(chunk["body_sha256"], hashlib.sha256(decoded).hexdigest())
            self.assertGreaterEqual(chunk["observed_monotonic_ns"], previous_time)
            previous_time = chunk["observed_monotonic_ns"]
        complete_body = b"".join(decoded_chunks)
        self.assertEqual(
            complete_body, ROLE_EVENT + CONTENT_EVENT + USAGE_EVENT + DONE_EVENT
        )
        self.assertEqual(response_end["outcome"], "eof")
        self.assertIsNone(response_end["error"])
        self.assertEqual(response_end["body_bytes"], len(complete_body))
        self.assertEqual(
            response_end["body_sha256"], hashlib.sha256(complete_body).hexdigest()
        )
        self.assertGreaterEqual(response_end["observed_monotonic_ns"], previous_time)

        client.shutdown()
        with EvidenceHandler.requests_lock:
            observed_request = EvidenceHandler.requests[0]
        self.assertEqual(observed_request["authorization"], f"Bearer {self.secret}")
        all_output = b"".join(client.raw_stdout) + client.stderr()
        self.assertNotIn(self.secret.encode("ascii"), all_output)

    def test_split_sse_closes_only_after_complete_nonempty_content_object(self):
        client = self.client(chunk_bytes=8)
        client.send(request_command("req-split", "/split", close_on_content=True))
        events = client.events_through_end()
        chunks = [event for event in events if event["event"] == "http_body_chunk"]
        self.assertGreaterEqual(len(chunks), 2)
        complete_body = b"".join(
            base64.b64decode(chunk["body_base64"], validate=True) for chunk in chunks
        )
        self.assertIn(CONTENT_EVENT, complete_body)
        self.assertEqual(events[-1]["outcome"], "client_closed")
        self.assertIsNone(events[-1]["error"])
        client.shutdown()

    def test_role_delta_is_not_first_content_trigger(self):
        client = self.client()
        client.send(request_command("req-role", "/role", close_on_content=True))
        self.assertTrue(EvidenceHandler.role_sent.wait(2.0))
        observed = []
        while True:
            event = client.read_event()
            observed.append(event)
            if event["event"] == "http_body_chunk":
                role_bytes = base64.b64decode(event["body_base64"], validate=True)
                self.assertIn(b'"role":"assistant"', role_bytes)
                break
        client.assert_no_event(0.15)
        EvidenceHandler.allow_content.set()
        tail = client.events_through_end()
        self.assertEqual(tail[-1]["outcome"], "client_closed")
        client.shutdown()

    def test_control_close_interrupts_active_socket_and_next_request_runs(self):
        client = self.client()
        client.send(request_command("req-hold", "/hold", close_on_content=False))
        self.assertTrue(EvidenceHandler.hold_sent.wait(2.0))
        first_events = []
        while True:
            event = client.read_event()
            first_events.append(event)
            if event["event"] == "http_body_chunk":
                break
        client.send(
            {
                "schema_version": TOOL.COMMAND_SCHEMA,
                "command": "close",
                "request_key": "req-hold",
            }
        )
        first_events.extend(client.events_through_end())
        self.assertEqual(first_events[-1]["outcome"], "client_closed")
        self.assertIsNone(first_events[-1]["error"])

        client.send(request_command("req-after-close", "/full", close_on_content=False))
        second_events = client.events_through_end()
        self.assertEqual(second_events[-1]["outcome"], "eof")
        client.shutdown()

    def test_second_active_request_emits_command_error_and_exits_nonzero(self):
        client = self.client()
        client.send(request_command("req-active", "/hold", close_on_content=False))
        self.assertTrue(EvidenceHandler.hold_sent.wait(2.0))
        while client.read_event()["event"] != "http_body_chunk":
            pass
        client.send(request_command("req-forbidden", "/full", close_on_content=False))
        events = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            event = client.read_event(timeout=max(0.01, deadline - time.monotonic()))
            events.append(event)
            if event["event"] == "command_error":
                break
        self.assertIn("waiting requests are forbidden", events[-1]["error"])
        self.assertNotIn(
            "req-forbidden",
            "\n".join(
                json.dumps(event)
                for event in events
                if event["event"] != "command_error"
            ),
        )
        self.assertEqual(client.process.wait(timeout=3.0), 2)
        output = b"".join(client.raw_stdout) + client.stderr()
        self.assertNotIn(self.secret.encode("ascii"), output)


if __name__ == "__main__":
    unittest.main()
