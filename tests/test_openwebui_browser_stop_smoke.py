import json
import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "openwebui" / "browser-stop-smoke.cjs"


class BrowserStopSmokeStaticTest(unittest.TestCase):
    def test_node_syntax_is_valid(self):
        completed = subprocess.run(
            ["node", "--check", str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_exported_socket_parser_redacts_content_and_tracks_terminal_state(self):
        program = r"""
const assert = require("node:assert/strict");
const moduleUnderTest = require(process.argv[1]);
const frame = '42["events",{"chat_id":"chat-secret","message_id":"message-secret","data":{"type":"chat:completion","data":{"content":"body-secret","done":true}}}]';
const event = moduleUnderTest.socketEvent(frame, () => 123n);
assert.equal(event.observedMonotonicNs, 123n);
assert.equal(event.chatId, "chat-secret");
assert.equal(event.messageId, "message-secret");
assert.equal(event.type, "chat:completion");
assert.equal(event.done, true);
assert.equal(event.contentUtf8Bytes, 11);
assert.match(event.contentSha256, /^[0-9a-f]{64}$/);
assert.equal(Object.hasOwn(event, "content"), false);
const cancel = moduleUnderTest.socketEvent(
  '42["events",{"chat_id":"c","message_id":"m","data":{"type":"chat:tasks:cancel"}}]',
  () => 456n,
);
assert.equal(cancel.type, "chat:tasks:cancel");
assert.equal(cancel.done, false);
assert.equal(cancel.contentUtf8Bytes, 0);
console.log(JSON.stringify({ schema: moduleUnderTest.SUMMARY_SCHEMA, ok: true }));
"""
        completed = subprocess.run(
            ["node", "-e", program, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["schema"], "ullm.openwebui.stop_smoke.v1")
        self.assertTrue(result["ok"])

    def test_summary_guard_rejects_cleartext_keys_and_values(self):
        program = r"""
const assert = require("node:assert/strict");
const moduleUnderTest = require(process.argv[1]);
assert.throws(
  () => moduleUnderTest.assertNoSensitiveSummary({ url: "https://secret.invalid" }, []),
  /forbidden cleartext field/,
);
assert.throws(
  () => moduleUnderTest.assertNoSensitiveSummary({ digest: "sensitive-value" }, ["sensitive-value"]),
  /sensitive cleartext value/,
);
const encoded = moduleUnderTest.assertNoSensitiveSummary(
  { text_utf8_bytes: 4, text_sha256: "a".repeat(64) },
  ["sensitive-value"],
);
assert.equal(JSON.parse(encoded).text_utf8_bytes, 4);
"""
        completed = subprocess.run(
            ["node", "-e", program, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_token_and_gateway_release_control_are_strict(self):
        program = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const moduleUnderTest = require(process.argv[1]);
assert.equal(moduleUnderTest.strictSingleLineToken(Buffer.from("token-value\n")), "token-value");
assert.throws(() => moduleUnderTest.strictSingleLineToken(Buffer.from("one\ntwo\n")), /one line/);
assert.throws(() => moduleUnderTest.strictSingleLineToken(Buffer.from(" token-value\n")), /whitespace/);
assert.throws(() => moduleUnderTest.strictSingleLineToken(Buffer.from([0xff])), /strict UTF-8/);
const nonce = "a".repeat(64);
const expected = `${moduleUnderTest.CONTROL_SCHEMA}:${nonce}\n`;
assert.equal(moduleUnderTest.gatewayReleaseControlContent(nonce), expected);
const directory = fs.mkdtempSync(path.join(os.tmpdir(), "ullm-stop-control-"));
const control = path.join(directory, "gateway-released");
setTimeout(() => fs.writeFileSync(control, expected, { flag: "wx", mode: 0o600 }), 10);
(async () => {
  const observed = await moduleUnderTest.waitForGatewayReleaseControl(control, expected, 1000);
  assert.equal(typeof observed, "bigint");
  fs.unlinkSync(control);
  fs.symlinkSync(path.join(directory, "missing"), control);
  await assert.rejects(
    moduleUnderTest.waitForGatewayReleaseControl(control, expected, 1000),
    /regular file/,
  );
  fs.rmSync(directory, { recursive: true, force: true });
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
"""
        completed = subprocess.run(
            ["node", "-e", program, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_stop_and_recovery_action_order_is_frozen(self):
        source = SCRIPT.read_text(encoding="utf-8")
        completed = subprocess.run(
            [
                "node",
                "-e",
                (
                    'const m=require(process.argv[1]);const c=require("node:crypto");'
                    "console.log(JSON.stringify({actions:m.EXPECTED_ACTION_SEQUENCE,"
                    'selectorSha256:c.createHash("sha256").update(m.STOP_SELECTOR).digest("hex")}))'
                ),
                str(SCRIPT),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        exported = json.loads(completed.stdout)
        actions = exported["actions"]
        self.assertEqual(
            actions,
            [
                "navigate",
                "select_model",
                "submit_chat",
                "wait_visible",
                "click_stop",
                "wait_ready",
                "submit_chat",
                "wait_visible",
                "wait_ready",
            ],
        )
        self.assertEqual(
            exported["selectorSha256"],
            "54645d692a5f666ab16dad72cb68e5dcddd02e3adc4d7d200f39efb624aab2f1",
        )
        screenshot = source.index("await page.screenshot")
        click = source.index("await stopButton.click")
        click_completed = source.index("const clickCompleted", click)
        cancel = source.index(
            "event.observedMonotonicNs >= clickCompleted", click_completed
        )
        interim = source.index(
            'record_type: "openwebui_stop_gateway_release_wait"', cancel
        )
        flush = source.index("await writeStdoutLine(interimSerialized)", interim)
        control = source.index("await waitForGatewayReleaseControl", flush)
        recovery = source.index("await submitChat(page, actions, RECOVERY_PROMPT)")
        self.assertLess(screenshot, click)
        self.assertLess(click, click_completed)
        self.assertLess(click_completed, cancel)
        self.assertLess(cancel, interim)
        self.assertLess(interim, flush)
        self.assertLess(flush, control)
        self.assertLess(control, recovery)

    def test_evidence_is_bounded_exclusive_and_nonpersistent(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'navigationUrl.searchParams.set("temporary-chat", "true")', source
        )
        self.assertIn("process.hrtime.bigint()", source)
        self.assertIn("CANCEL_TIMEOUT_MS = 5_000", source)
        self.assertIn('flag: "wx"', source)
        self.assertIn("pageErrors.length !== 0", source)
        self.assertIn("contentAfterCancel.length !== 0", source)
        self.assertIn("recoveryDoneEvents.length !== 1", source)
        self.assertIn("socket_events: redactedSocketEvents", source)
        self.assertIn("content_sha256: event.contentSha256", source)
        self.assertIn("event.observedMonotonicNs >= initialSubmit.started", source)
        self.assertIn("event.chatId === target.chatId", source)
        self.assertIn("event.messageId !== target.messageId", source)
        self.assertIn("fs.lstatSync(tokenFile", source)
        self.assertIn("requireAbsent(gatewayReleaseControlFile", source)
        self.assertNotIn("page.url()", source)
        self.assertNotIn("error.stack", source)
        self.assertTrue(os.access(SCRIPT, os.X_OK))


if __name__ == "__main__":
    unittest.main()
