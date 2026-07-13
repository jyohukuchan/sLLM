import json
import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "openwebui" / "browser-soak.cjs"


class BrowserSoakStaticTest(unittest.TestCase):
    def run_node(self, program: str):
        return subprocess.run(
            ["node", "-e", program, str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_node_syntax_is_valid(self):
        completed = subprocess.run(
            ["node", "--check", str(SCRIPT)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_frozen_twenty_case_schedule_has_unique_redacted_inputs(self):
        program = r"""
const assert = require("node:assert/strict");
const m = require(process.argv[1]);
assert.equal(m.SOAK_CHAT_COUNT, 20);
assert.deepEqual(m.EXPECTED_ACTION_SEQUENCE, [
  "navigate", "select_model", "submit_chat", "wait_visible", "wait_ready",
]);
const cases = Array.from({ length: 20 }, (_, index) => ({
  browserCase: m.browserCase(index + 1),
  marker: m.caseMarker(index + 1),
  prompt: m.casePrompt(index + 1),
}));
assert.equal(new Set(cases.map((item) => item.browserCase)).size, 20);
assert.equal(new Set(cases.map((item) => item.marker)).size, 20);
assert.equal(new Set(cases.map((item) => item.prompt)).size, 20);
assert.equal(cases[0].browserCase, "openwebui_soak_chat_01");
assert.equal(cases[19].browserCase, "openwebui_soak_chat_20");
console.log(JSON.stringify({ schema: m.SUMMARY_SCHEMA, count: cases.length }));
"""
        completed = self.run_node(program)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(
            result, {"schema": "ullm.openwebui.browser_soak.v1", "count": 20}
        )

    def test_socket_parser_hashes_content_and_normalizes_empty_content(self):
        program = r"""
const assert = require("node:assert/strict");
const m = require(process.argv[1]);
const content = m.socketEvent(
  '42["events",{"chat_id":"chat-secret","message_id":"message-secret","data":{"type":"chat:completion","data":{"content":"response-secret","done":true}}}]',
  () => 123n,
);
assert.equal(content.chatId, "chat-secret");
assert.equal(content.messageId, "message-secret");
assert.equal(content.contentUtf8Bytes, 15);
assert.match(content.contentSha256, /^[0-9a-f]{64}$/);
assert.equal(Object.hasOwn(content, "content"), false);
const state = m.socketEvent(
  '42["events",{"chat_id":"c","message_id":"m","data":{"type":"chat:active","data":{"content":"","done":false}}}]',
  () => 124n,
);
assert.equal(state.contentUtf8Bytes, 0);
assert.equal(state.contentSha256, null);
assert.doesNotThrow(() => m.validateTargetEvents([
  state,
  { ...content, observedMonotonicNs: 125n, done: false },
  { ...content, observedMonotonicNs: 126n, contentUtf8Bytes: 0, contentSha256: null },
]));
assert.throws(
  () => m.validateTargetEvents([{ ...state, contentUtf8Bytes: 1 }]),
  /state event/,
);
assert.throws(
  () => m.validateTargetEvents([{ ...content, hasError: true }]),
  /provider error/,
);
assert.throws(
  () => m.validateTargetEvents([
    { ...content, observedMonotonicNs: 125n, done: false },
    { ...content, observedMonotonicNs: 126n, contentUtf8Bytes: 0, contentSha256: null },
    { ...content, observedMonotonicNs: 127n, done: false },
  ]),
  /followed terminal done/,
);
"""
        completed = self.run_node(program)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_summary_guard_rejects_cleartext_fields_and_values(self):
        program = r"""
const assert = require("node:assert/strict");
const m = require(process.argv[1]);
for (const key of ["url", "token", "prompt", "response", "chat_id", "message_id"]) {
  assert.throws(
    () => m.assertNoSensitiveSummary({ [key]: "cleartext-secret" }, []),
    /forbidden cleartext field/,
  );
}
for (const key of ["openwebui_url", "api_token", "raw_prompt", "model_response"]) {
  assert.throws(
    () => m.assertNoSensitiveSummary({ [key]: "cleartext-secret" }, []),
    /forbidden cleartext field/,
  );
}
assert.throws(
  () => m.assertNoSensitiveSummary({ digest: "cleartext-secret" }, ["cleartext-secret"]),
  /sensitive cleartext value/,
);
const encoded = m.assertNoSensitiveSummary(
  { text_utf8_bytes: 4, text_sha256: "a".repeat(64) },
  ["cleartext-secret"],
);
assert.equal(JSON.parse(encoded).text_utf8_bytes, 4);
"""
        completed = self.run_node(program)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_token_parser_is_strict(self):
        program = r"""
const assert = require("node:assert/strict");
const m = require(process.argv[1]);
assert.equal(m.strictSingleLineToken(Buffer.from("token-value\n")), "token-value");
assert.throws(() => m.strictSingleLineToken(Buffer.from("short\n")), /guard minimum/);
assert.throws(() => m.strictSingleLineToken(Buffer.from("one\ntwo\n")), /one line/);
assert.throws(() => m.strictSingleLineToken(Buffer.from(" token-value\n")), /whitespace/);
assert.throws(() => m.strictSingleLineToken(Buffer.from([0xff])), /strict UTF-8/);
"""
        completed = self.run_node(program)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_real_ui_uses_one_context_and_twenty_closed_temporary_pages(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(source.count("chromium.launch({ headless: true })"), 1)
        self.assertEqual(source.count("browser.newContext("), 1)
        self.assertIn("caseIndex <= SOAK_CHAT_COUNT", source)
        self.assertIn("const page = await context.newPage()", source)
        self.assertIn("if (!page.isClosed()) await page.close()", source)
        self.assertIn(
            'navigationUrl.searchParams.set("temporary-chat", "true")', source
        )
        self.assertIn('navigationUrl.searchParams.set("models", MODEL_ID)', source)
        self.assertIn("page.locator(INPUT_SELECTOR)", source)
        self.assertIn("input.fill(prompt", source)
        self.assertIn('input.press("Enter"', source)
        self.assertIn('window.location.pathname === "/"', source)
        self.assertIn('query.get("temporary-chat") === "true"', source)
        self.assertIn('event.type === "chat:completion" &&', source)
        self.assertIn("event.done", source)
        self.assertIn("const visibleText = await visibleAnswerText(assistant)", source)
        self.assertIn("if (visibleText !== marker)", source)
        self.assertIn("const finalText = await visibleAnswerText(assistant)", source)
        self.assertIn("if (finalText !== marker)", source)
        self.assertNotIn("visibleText.includes(marker)", source)
        self.assertNotIn("finalText.includes(marker)", source)
        self.assertIn('flag: "wx"', source)
        self.assertNotIn("page.url()", source)
        self.assertNotIn("error.stack", source)
        self.assertNotIn("fetch(", source)
        self.assertTrue(os.access(SCRIPT, os.X_OK))


if __name__ == "__main__":
    unittest.main()
