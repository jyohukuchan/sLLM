from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy/openwebui/browser-reasoning-smoke.cjs"


def node(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "-e", program, str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_browser_reasoning_script_has_valid_node_syntax() -> None:
    completed = subprocess.run(
        ["node", "--check", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_request_summary_redacts_body_and_detects_history_reasoning() -> None:
    program = r'''
const assert = require("node:assert/strict");
const tool = require(process.argv[1]);
const raw = JSON.stringify({
  model: "model",
  messages: [
    {role: "user", content: "prompt-secret"},
    {role: "assistant", reasoning_content: "reasoning-secret", content: "answer"},
  ],
});
const summary = tool.summarizeRequestBody(raw);
assert.match(summary.sha256, /^[0-9a-f]{64}$/);
assert.equal(summary.assistant_has_reasoning_content, true);
assert.equal(summary.has_reasoning_content_key, true);
assert.equal(Object.hasOwn(summary, "raw"), false);
const noReasoning = tool.summarizeRequestBody(JSON.stringify({
  messages: [{role: "user", content: "prompt-secret"}, {role: "assistant", content: "answer"}],
}));
assert.equal(noReasoning.assistant_has_reasoning_content, false);
assert.equal(noReasoning.has_reasoning_content_key, false);
console.log(JSON.stringify({schema: tool.SUMMARY_SCHEMA, ok: true}));
'''
    completed = node(program)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "schema": "ullm.openwebui.reasoning_browser_smoke.v1",
        "ok": True,
    }


def test_browser_reasoning_script_rejects_unsafe_inputs() -> None:
    program = r'''
const assert = require("node:assert/strict");
const tool = require(process.argv[1]);
assert.throws(() => tool.normalizedBaseUrl("http://user:password@example.test/"));
assert.throws(() => tool.strictToken("line-one\nline-two"));
assert.equal(tool.requestIsChatCompletion({method: () => "POST", url: () => "http://x/api/chat/completions"}), true);
assert.equal(tool.requestIsChatCompletion({method: () => "GET", url: () => "http://x/api/chat/completions"}), false);
console.log("ok");
'''
    completed = node(program)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
