from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "openwebui" / "browser-failure-smoke.cjs"


def run_node(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "-e", program, str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
    )


class BrowserFailureSmokeStaticTest(unittest.TestCase):
    def test_node_syntax_is_valid(self):
        completed = subprocess.run(
            ["node", "--check", str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_socket_parser_hashes_content_and_tracks_failure(self):
        completed = run_node(
            r"""
const assert = require("node:assert/strict");
const subject = require(process.argv[1]);
const content = subject.socketEvent(
  '42["events",{"chat_id":"chat-secret","message_id":"message-secret","data":{"type":"chat:completion","data":{"content":"body-secret"}}}]',
  () => 123n,
);
assert.equal(content.observedMonotonicNs, 123n);
assert.equal(content.chatId, "chat-secret");
assert.equal(content.messageId, "message-secret");
assert.equal(content.contentUtf8Bytes, 11);
assert.match(content.contentSha256, /^[0-9a-f]{64}$/);
assert.equal(Object.hasOwn(content, "content"), false);
const failed = subject.socketEvent(
  '42["events",{"chat_id":"c","message_id":"m","data":{"type":"chat:completion","data":{"error":{"detail":"secret"}}}}]',
  () => 456n,
);
assert.equal(failed.hasError, true);
assert.equal(failed.done, false);
assert.equal(failed.contentUtf8Bytes, 0);
const empty = subject.socketEvent(
  '42["events",{"chat_id":"c","message_id":"m","data":{"type":"chat:active","data":{"content":"","error":""}}}]',
);
assert.equal(empty.contentUtf8Bytes, 0);
assert.equal(empty.contentSha256, null);
assert.equal(empty.hasError, false);
const cancelled = subject.socketEvent(
  '42["events",{"chat_id":"c","message_id":"m","data":{"type":"chat:tasks:cancel"}}]',
);
assert.equal(cancelled.type, "chat:tasks:cancel");
assert.throws(
  () => subject.redactSocketEvents([content, {...content, chatId: "foreign"}], content),
  /foreign Socket.IO event/,
);
console.log(JSON.stringify({schema: subject.SUMMARY_SCHEMA}));
"""
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout)["schema"],
            "ullm.openwebui.failure_smoke.v1",
        )

    def test_summary_guard_rejects_cleartext_keys_and_values(self):
        completed = run_node(
            r"""
const assert = require("node:assert/strict");
const subject = require(process.argv[1]);
assert.throws(
  () => subject.assertNoSensitiveSummary({url: "https://secret.invalid"}, []),
  /forbidden cleartext field/,
);
assert.throws(
  () => subject.assertNoSensitiveSummary({digest: "sensitive-value"}, ["sensitive-value"]),
  /sensitive cleartext value/,
);
const raw = subject.assertNoSensitiveSummary(
  {text_utf8_bytes: 4, text_sha256: "a".repeat(64)},
  ["sensitive-value"],
);
assert.equal(JSON.parse(raw).text_utf8_bytes, 4);
"""
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_token_and_two_control_files_are_strict(self):
        completed = run_node(
            r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const subject = require(process.argv[1]);
assert.equal(subject.strictSingleLineToken(Buffer.from("token-value\n")), "token-value");
assert.throws(() => subject.strictSingleLineToken(Buffer.from("one\ntwo\n")), /one line/);
assert.throws(() => subject.strictSingleLineToken(Buffer.from(" token-value\n")), /whitespace/);
assert.throws(() => subject.strictSingleLineToken(Buffer.from([0xff])), /strict UTF-8/);
const nonce = "a".repeat(64);
const killed = `${subject.CONTROL_SCHEMA}:worker_killed:${nonce}\n`;
const recovered = `${subject.CONTROL_SCHEMA}:gateway_recovered:${nonce}\n`;
assert.equal(subject.controlContent("worker_killed", nonce), killed);
assert.equal(subject.controlContent("gateway_recovered", nonce), recovered);
assert.throws(() => subject.controlContent("other", nonce), /stage/);
assert.throws(() => subject.controlContent("worker_killed", "bad"), /nonce/);
const directory = fs.mkdtempSync(path.join(os.tmpdir(), "ullm-failure-control-"));
const token = path.join(directory, "token");
fs.writeFileSync(token, "token-value\n", {mode: 0o600});
assert.equal(subject.readStrictToken(token), "token-value");
const control = path.join(directory, "worker-killed");
setTimeout(() => fs.writeFileSync(control, killed, {flag: "wx", mode: 0o600}), 10);
(async () => {
  const observed = await subject.waitForControl(control, killed, 1000);
  assert.equal(typeof observed, "bigint");
  fs.unlinkSync(control);
  fs.symlinkSync(path.join(directory, "missing"), control);
  await assert.rejects(subject.waitForControl(control, killed, 1000), /regular file/);
  fs.rmSync(directory, {recursive: true, force: true});
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
"""
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_failure_and_recovery_action_order_is_frozen(self):
        completed = run_node(
            r"""
const subject = require(process.argv[1]);
console.log(JSON.stringify({
  actions: subject.EXPECTED_ACTION_SEQUENCE,
  failurePrompt: subject.FAILURE_PROMPT,
  recoveryPrompt: subject.RECOVERY_PROMPT,
  screenshot: subject.SCREENSHOT_FILE,
}));
"""
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        exported = json.loads(completed.stdout)
        self.assertEqual(
            exported["actions"],
            [
                "navigate",
                "select_model",
                "submit_chat",
                "wait_visible",
                "wait_failed",
                "wait_ready",
                "submit_chat",
                "wait_visible",
                "wait_ready",
            ],
        )
        self.assertIn("FAIL_STREAM_MARKER", exported["failurePrompt"])
        self.assertIn("FAILURE_RECOVERY_OK", exported["recoveryPrompt"])
        self.assertEqual(exported["screenshot"], "browser/post-header-failure.png")

        source = SCRIPT.read_text(encoding="utf-8")
        kill_interim = source.index('record_type: "openwebui_failure_worker_kill_wait"')
        kill_flush = source.index("await writeStdoutLine(killSerialized)", kill_interim)
        kill_control = source.index("await waitForControl(", kill_flush)
        wait_failure = source.index("page.getByText(FAILURE_TEXT", kill_control)
        screenshot = source.index("await page.screenshot", wait_failure)
        recovery_interim = source.index(
            'record_type: "openwebui_failure_gateway_recovery_wait"', screenshot
        )
        recovery_flush = source.index(
            "await writeStdoutLine(recoverySerialized)", recovery_interim
        )
        recovery_control = source.index("await waitForControl(", recovery_flush)
        recovery_submit = source.index(
            "await submitChat(page, actions, RECOVERY_PROMPT)", recovery_control
        )
        self.assertLess(kill_interim, kill_flush)
        self.assertLess(kill_flush, kill_control)
        self.assertLess(kill_control, wait_failure)
        self.assertLess(wait_failure, screenshot)
        self.assertLess(screenshot, recovery_interim)
        self.assertLess(recovery_interim, recovery_flush)
        self.assertLess(recovery_flush, recovery_control)
        self.assertLess(recovery_control, recovery_submit)

    def test_evidence_is_exclusive_redacted_and_nonpersistent(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'navigationUrl.searchParams.set("temporary-chat", "true")', source
        )
        self.assertIn("process.hrtime.bigint()", source)
        self.assertIn('fs.openSync(config.screenshotPath, "wx"', source)
        self.assertIn('flag: "wx"', source)
        self.assertIn("errorEvents.length !== 1", source)
        self.assertIn("cancelEvents.length !== 1", source)
        self.assertIn("doneEvents.length !== 0", source)
        self.assertIn("contentAfterError.length !== 0", source)
        self.assertIn("recoveryDoneEvents.length !== 1", source)
        self.assertIn("recoveryCancelEvents.length !== 0", source)
        self.assertIn("pageErrors.length !== 0", source)
        self.assertIn('window.location.pathname === "/"', source)
        self.assertIn('query.get("temporary-chat") === "true"', source)
        context_close = source.index("await context.close()")
        final_summary = source.index("const summary = {", context_close)
        self.assertLess(context_close, final_summary)
        self.assertNotIn("assistantText:", source)
        self.assertNotIn("errorText:", source)


if __name__ == "__main__":
    unittest.main()
