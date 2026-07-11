#!/usr/bin/env node

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.env.OPENWEBUI_URL || "http://192.168.0.66:3000";
const tokenFile = process.env.OPENWEBUI_TOKEN_FILE || "/run/secrets/openwebui-token";
const controlFile =
  process.env.OPENWEBUI_WORKER_KILLED_FILE || "/run/control/worker-killed";
const beforeScreenshot =
  process.env.OPENWEBUI_BEFORE_SCREENSHOT || "/output/openwebui-failure-before.png";
const afterScreenshot =
  process.env.OPENWEBUI_AFTER_SCREENSHOT || "/output/openwebui-failure-after.png";
const summaryFile = process.env.OPENWEBUI_FAILURE_SUMMARY || null;
const minimumVisibleCharacters = Number.parseInt(
  process.env.OPENWEBUI_MINIMUM_VISIBLE_CHARACTERS || "40",
  10,
);
const token = fs.readFileSync(tokenFile, "utf8").trim();

if (!token) {
  throw new Error("OpenWebUI test token is empty");
}
if (!Number.isSafeInteger(minimumVisibleCharacters) || minimumVisibleCharacters < 1) {
  throw new Error("minimum visible character count is invalid");
}

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const monotonicNs = () => process.hrtime.bigint().toString();
const sha256 = (value) => crypto.createHash("sha256").update(value).digest("hex");

function socketEvent(payload) {
  const text = Buffer.isBuffer(payload) ? payload.toString("utf8") : String(payload);
  const start = text.indexOf("[");
  if (start < 0) return null;
  try {
    const decoded = JSON.parse(text.slice(start));
    if (!Array.isArray(decoded) || decoded[0] !== "events") return null;
    const envelope = decoded[1];
    const event = envelope?.data;
    if (!envelope || !event || typeof event.type !== "string") return null;
    return {
      observedMonotonicNs: monotonicNs(),
      chatId: envelope.chat_id ?? null,
      messageId: envelope.message_id ?? null,
      type: event.type,
      hasError: Boolean(event.data?.error),
      done: event.data?.done === true,
    };
  } catch {
    return null;
  }
}

async function waitForFile(file, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (fs.existsSync(file)) return;
    await sleep(100);
  }
  throw new Error(`timed out waiting for control file: ${file}`);
}

async function run(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript((value) => {
    window.localStorage.setItem("token", value);
  }, token);

  const page = await context.newPage();
  const pageErrors = [];
  const events = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("websocket", (websocket) => {
    websocket.on("framereceived", ({ payload }) => {
      const event = socketEvent(payload);
      if (event) events.push(event);
    });
  });

  const query = new URLSearchParams({
    models: "ullm-qwen3-14b-sq8",
    q: [
      "Begin with FAIL_STREAM_MARKER.",
      "Then write the integers from 1 through 200, one per line.",
      "Do not summarize and do not stop early.",
    ].join(" "),
    submit: "true",
  });
  await page.goto(`${baseUrl}/?${query}`, {
    waitUntil: "domcontentloaded",
    timeout: 60_000,
  });

  const assistant = page.locator(".chat-assistant").last();
  await assistant.waitFor({ state: "visible", timeout: 60_000 });
  await page.waitForFunction(
    (minimum) => {
      const values = [...document.querySelectorAll(".chat-assistant")];
      return values.length > 0 && (values.at(-1)?.innerText?.length ?? 0) >= minimum;
    },
    minimumVisibleCharacters,
    { timeout: 90_000 },
  );

  const stopSelector =
    '#message-input-container button:has(svg[viewBox="0 0 24 24"] path[d^="M2.25 12c0-5.385"])';
  const stopButton = page.locator(stopSelector);
  await stopButton.waitFor({ state: "visible", timeout: 10_000 });
  if (!(await stopButton.isEnabled())) {
    throw new Error("OpenWebUI Stop button is not enabled before fault injection");
  }

  fs.mkdirSync(path.dirname(beforeScreenshot), { recursive: true });
  await page.screenshot({ path: beforeScreenshot, fullPage: true });
  const visibleText = (await assistant.innerText()).trim();
  const ready = {
    type: "ready_for_worker_kill",
    observedMonotonicNs: monotonicNs(),
    assistantTextUtf8Bytes: Buffer.byteLength(visibleText),
    assistantTextSha256: sha256(visibleText),
    stopSelector,
    url: page.url(),
  };
  console.log(JSON.stringify(ready));

  await waitForFile(controlFile, 60_000);
  const errorText = "The generation failed.";
  await page.getByText(errorText, { exact: false }).last().waitFor({
    state: "visible",
    timeout: 30_000,
  });
  await sleep(2_000);

  fs.mkdirSync(path.dirname(afterScreenshot), { recursive: true });
  await page.screenshot({ path: afterScreenshot, fullPage: true });

  const errorIndex = events.findIndex(
    (event) => event.type === "chat:completion" && event.hasError,
  );
  if (errorIndex < 0) {
    throw new Error("no provider error event was observed on Socket.IO");
  }
  const target = events[errorIndex];
  const relatedAfterError = events.slice(errorIndex + 1).filter(
    (event) => event.chatId === target.chatId && event.messageId === target.messageId,
  );
  const doneAfterError = relatedAfterError.filter(
    (event) => event.type === "chat:completion" && event.done,
  );
  const cancelAfterError = relatedAfterError.filter(
    (event) => event.type === "chat:tasks:cancel",
  );

  const summary = {
    type: "post_header_failure_result",
    errorObserved: true,
    errorEventCount: events.filter(
      (event) =>
        event.chatId === target.chatId &&
        event.messageId === target.messageId &&
        event.type === "chat:completion" &&
        event.hasError,
    ).length,
    doneAfterErrorCount: doneAfterError.length,
    cancelAfterErrorCount: cancelAfterError.length,
    pageErrors,
    beforeScreenshot,
    afterScreenshot,
    chatId: target.chatId,
    messageId: target.messageId,
    observedMonotonicNs: monotonicNs(),
  };
  if (summary.doneAfterErrorCount !== 0) {
    throw new Error("OpenWebUI emitted normal done completion after provider error");
  }
  if (summary.cancelAfterErrorCount !== 1) {
    throw new Error(
      `expected one task cancellation after provider error, got ${summary.cancelAfterErrorCount}`,
    );
  }
  if (pageErrors.length !== 0) {
    throw new Error(`browser page errors: ${JSON.stringify(pageErrors)}`);
  }

  if (summaryFile) {
    fs.mkdirSync(path.dirname(summaryFile), { recursive: true });
    fs.writeFileSync(summaryFile, `${JSON.stringify(summary, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600,
      flag: "wx",
    });
  }
  console.log(JSON.stringify(summary));
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  try {
    await run(browser);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
