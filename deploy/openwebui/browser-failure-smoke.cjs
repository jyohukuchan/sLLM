#!/usr/bin/env node

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const SUMMARY_SCHEMA = "ullm.openwebui.failure_smoke.v1";
const CONTROL_SCHEMA = "ullm.openwebui.failure_control.v1";
const BROWSER_CASE = "post_header_worker_failure";
const MODEL_ID = "ullm-qwen3-14b-sq8";
const MODEL_LABEL = "uLLM Qwen3 14B SQ8";
const SCREENSHOT_FILE = "browser/post-header-failure.png";
const ASSISTANT_SELECTOR = ".chat-assistant";
const INPUT_SELECTOR = "#chat-input";
const STOP_SELECTOR =
  '#message-input-container button:has(svg[viewBox="0 0 24 24"] path[d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm6-2.438c0-.724.588-1.312 1.313-1.312h4.874c.725 0 1.313.588 1.313 1.313v4.874c0 .725-.588 1.313-1.313 1.313H9.564a1.312 1.312 0 01-1.313-1.313V9.564z"])';
const EXPECTED_ACTION_SEQUENCE = [
  "navigate",
  "select_model",
  "submit_chat",
  "wait_visible",
  "wait_failed",
  "wait_ready",
  "submit_chat",
  "wait_visible",
  "wait_ready",
];

const FAILURE_PROMPT = [
  "Begin with FAIL_STREAM_MARKER.",
  "Then write the integers from 1 through 1000, one per line.",
  "Do not summarize and do not stop early.",
].join(" ");
const RECOVERY_MARKER = "FAILURE_RECOVERY_OK";
const RECOVERY_PROMPT =
  "For this new turn, reply with exactly FAILURE_RECOVERY_OK and nothing else.";
const FAILURE_TEXT = "The generation failed.";

const NAVIGATION_TIMEOUT_MS = 60_000;
const VISIBLE_TIMEOUT_MS = 90_000;
const FAILURE_TIMEOUT_MS = 30_000;
const RECOVERY_TIMEOUT_MS = 90_000;
const UI_TIMEOUT_MS = 10_000;
const POST_FAILURE_STABLE_MS = 1_000;
const SOCKET_POLL_MS = 20;
const MAX_SOCKET_EVENTS = 2048;
const DEFAULT_KILL_CONTROL_FILE = "/run/control/worker-killed";
const DEFAULT_RECOVERY_CONTROL_FILE = "/run/control/gateway-recovered";

const monotonicNs = () => process.hrtime.bigint();
const nsString = (value = monotonicNs()) => value.toString();
const sha256 = (value) => crypto.createHash("sha256").update(value).digest("hex");
const textEvidence = (value) => ({
  text_utf8_bytes: Buffer.byteLength(value, "utf8"),
  text_sha256: sha256(value),
});
const identityEvidence = (value, prefix) => ({
  [`${prefix}_utf8_bytes`]: Buffer.byteLength(value, "utf8"),
  [`${prefix}_sha256`]: sha256(value),
});

function boundedInteger(name, fallback, minimum, maximum) {
  const raw = process.env[name] ?? String(fallback);
  if (!/^[0-9]+$/.test(raw)) {
    throw new Error(`${name} is not a decimal integer`);
  }
  const value = Number.parseInt(raw, 10);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} is outside its bounded range`);
  }
  return value;
}

function lstatOrNull(file) {
  try {
    return fs.lstatSync(file, { bigint: true });
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

function requireAbsent(file, label) {
  if (lstatOrNull(file) !== null) {
    throw new Error(`${label} must be absent at run start`);
  }
}

function strictSingleLineToken(raw) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  } catch {
    throw new Error("OpenWebUI test token is not strict UTF-8");
  }
  if (text.endsWith("\n")) text = text.slice(0, -1);
  if (!text || text.includes("\n") || text.includes("\r") || text.includes("\0")) {
    throw new Error("OpenWebUI test token is not one line");
  }
  if (text.trim() !== text) {
    throw new Error("OpenWebUI test token has surrounding whitespace");
  }
  return text;
}

function readStrictToken(file) {
  const metadata = fs.lstatSync(file, { bigint: true });
  if (!metadata.isFile() || metadata.size < 1n || metadata.size > 65_536n) {
    throw new Error("OpenWebUI test token file has an invalid type or size");
  }
  const flags = fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0);
  const descriptor = fs.openSync(file, flags);
  try {
    const opened = fs.fstatSync(descriptor, { bigint: true });
    if (
      !opened.isFile() ||
      opened.dev !== metadata.dev ||
      opened.ino !== metadata.ino ||
      opened.size !== metadata.size
    ) {
      throw new Error("OpenWebUI test token identity changed while opening");
    }
    const raw = fs.readFileSync(descriptor);
    const after = fs.fstatSync(descriptor, { bigint: true });
    if (
      after.dev !== opened.dev ||
      after.ino !== opened.ino ||
      after.size !== opened.size ||
      after.mtimeNs !== opened.mtimeNs ||
      after.ctimeNs !== opened.ctimeNs
    ) {
      throw new Error("OpenWebUI test token changed while reading");
    }
    return strictSingleLineToken(raw);
  } finally {
    fs.closeSync(descriptor);
  }
}

function controlContent(stage, nonce) {
  if (!new Set(["worker_killed", "gateway_recovered"]).has(stage)) {
    throw new Error("failure control stage is invalid");
  }
  if (!/^[0-9a-f]{64}$/.test(nonce)) {
    throw new Error("failure control nonce is invalid");
  }
  return `${CONTROL_SCHEMA}:${stage}:${nonce}\n`;
}

async function waitForControl(file, expectedContent, timeoutMs) {
  const expectedBytes = Buffer.from(expectedContent, "utf8");
  const deadline = monotonicNs() + BigInt(timeoutMs) * 1_000_000n;
  while (monotonicNs() < deadline) {
    const metadata = lstatOrNull(file);
    if (metadata === null) {
      await sleep(SOCKET_POLL_MS);
      continue;
    }
    if (!metadata.isFile()) {
      throw new Error("failure control is not a regular file");
    }
    if (metadata.size === 0n) {
      await sleep(SOCKET_POLL_MS);
      continue;
    }
    if (metadata.size !== BigInt(expectedBytes.length)) {
      throw new Error("failure control size differs");
    }
    const flags = fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0);
    const descriptor = fs.openSync(file, flags);
    let actual;
    try {
      const opened = fs.fstatSync(descriptor, { bigint: true });
      if (!opened.isFile() || opened.dev !== metadata.dev || opened.ino !== metadata.ino) {
        throw new Error("failure control identity changed while opening");
      }
      actual = fs.readFileSync(descriptor);
      const after = fs.fstatSync(descriptor, { bigint: true });
      if (
        after.dev !== opened.dev ||
        after.ino !== opened.ino ||
        after.size !== opened.size ||
        after.mtimeNs !== opened.mtimeNs ||
        after.ctimeNs !== opened.ctimeNs
      ) {
        throw new Error("failure control changed while reading");
      }
    } finally {
      fs.closeSync(descriptor);
    }
    if (!actual.equals(expectedBytes)) {
      throw new Error("failure control nonce or content differs");
    }
    return monotonicNs();
  }
  throw new Error("timed out waiting for failure control");
}

function normalizedBaseUrl(raw) {
  const value = new URL(raw);
  if (!["http:", "https:"].includes(value.protocol)) {
    throw new Error("OPENWEBUI_URL must use HTTP or HTTPS");
  }
  if (value.username || value.password || value.search || value.hash) {
    throw new Error("OPENWEBUI_URL must not contain credentials, query, or fragment");
  }
  if (value.pathname !== "/") {
    throw new Error("OPENWEBUI_URL must have the root path");
  }
  return value.origin;
}

function loadConfig() {
  const baseUrl = normalizedBaseUrl(
    process.env.OPENWEBUI_URL || "http://192.168.0.66:3000/",
  );
  const tokenFile =
    process.env.OPENWEBUI_TOKEN_FILE || "/run/secrets/openwebui-token";
  const screenshotPath =
    process.env.OPENWEBUI_FAILURE_SCREENSHOT || "/output/post-header-failure.png";
  const summaryFile =
    process.env.OPENWEBUI_FAILURE_SUMMARY || "/output/openwebui-failure-summary.json";
  const killControlFile =
    process.env.OPENWEBUI_WORKER_KILLED_FILE || DEFAULT_KILL_CONTROL_FILE;
  const recoveryControlFile =
    process.env.OPENWEBUI_GATEWAY_RECOVERED_FILE || DEFAULT_RECOVERY_CONTROL_FILE;
  for (const [name, value] of [
    ["worker-killed control", killControlFile],
    ["gateway-recovered control", recoveryControlFile],
  ]) {
    if (!path.isAbsolute(value) || value.includes("\0")) {
      throw new Error(`${name} path must be absolute`);
    }
  }
  if (killControlFile === recoveryControlFile) {
    throw new Error("failure control paths must differ");
  }
  const controlTimeoutMs = boundedInteger(
    "OPENWEBUI_FAILURE_CONTROL_TIMEOUT_MS",
    180_000,
    10_000,
    600_000,
  );
  const minimumVisibleCharacters = boundedInteger(
    "OPENWEBUI_MINIMUM_VISIBLE_CHARACTERS",
    40,
    1,
    4096,
  );
  requireAbsent(screenshotPath, "failure screenshot output");
  requireAbsent(summaryFile, "failure summary output");
  requireAbsent(killControlFile, "worker-killed control");
  requireAbsent(recoveryControlFile, "gateway-recovered control");
  return {
    baseUrl,
    token: readStrictToken(tokenFile),
    screenshotPath,
    summaryFile,
    killControlFile,
    recoveryControlFile,
    controlTimeoutMs,
    minimumVisibleCharacters,
    killNonce: crypto.randomBytes(32).toString("hex"),
    recoveryNonce: crypto.randomBytes(32).toString("hex"),
  };
}

function socketEvent(payload, observed = monotonicNs) {
  const text = Buffer.isBuffer(payload) ? payload.toString("utf8") : String(payload);
  const start = text.indexOf("[");
  if (start < 0) return null;
  try {
    const decoded = JSON.parse(text.slice(start));
    if (!Array.isArray(decoded) || decoded[0] !== "events") return null;
    const envelope = decoded[1];
    const event = envelope?.data;
    if (
      !envelope ||
      !event ||
      typeof event.type !== "string" ||
      typeof envelope.chat_id !== "string" ||
      typeof envelope.message_id !== "string"
    ) {
      return null;
    }
    const content =
      typeof event.data?.content === "string" ? event.data.content : null;
    const contentUtf8Bytes =
      content === null ? 0 : Buffer.byteLength(content, "utf8");
    const error = event.data?.error;
    return {
      observedMonotonicNs: observed(),
      chatId: envelope.chat_id,
      messageId: envelope.message_id,
      type: event.type,
      done: event.data?.done === true,
      hasError:
        error !== undefined && error !== null && error !== false && error !== "",
      contentUtf8Bytes,
      contentSha256: contentUtf8Bytes === 0 ? null : sha256(content),
    };
  } catch {
    return null;
  }
}

const sleep = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitUntil(predicate, timeoutMs, label) {
  const deadline = monotonicNs() + BigInt(timeoutMs) * 1_000_000n;
  while (monotonicNs() < deadline) {
    const value = predicate();
    if (value) return value;
    await sleep(SOCKET_POLL_MS);
  }
  throw new Error(`timed out waiting for ${label}`);
}

function exactBrowserResult({ visible = null, enabled = null, text = null } = {}) {
  return {
    visible,
    enabled,
    ...(text === null
      ? { text_utf8_bytes: null, text_sha256: null }
      : textEvidence(text)),
  };
}

function addBrowserAction(actions, fields) {
  if (fields.completedMonotonicNs < fields.startedMonotonicNs) {
    throw new Error("browser action monotonic timestamps regress");
  }
  const action = {
    browser_case: BROWSER_CASE,
    action_index: actions.length,
    action: fields.action,
    selector: fields.selector ?? null,
    input_sha256: fields.inputSha256 ?? null,
    started_monotonic_ns: fields.startedMonotonicNs.toString(),
    completed_monotonic_ns: fields.completedMonotonicNs.toString(),
    result: fields.result,
    screenshot_file: fields.screenshotFile ?? null,
    screenshot_sha256: fields.screenshotSha256 ?? null,
  };
  actions.push(action);
  return action;
}

function validateActionSequence(actions) {
  const observed = actions.map((action) => action.action);
  if (
    observed.length !== EXPECTED_ACTION_SEQUENCE.length ||
    observed.some((action, index) => action !== EXPECTED_ACTION_SEQUENCE[index])
  ) {
    throw new Error("browser action sequence differs from the frozen failure schedule");
  }
}

function sameSocketTarget(event, target) {
  return event.chatId === target.chatId && event.messageId === target.messageId;
}

function redactSocketEvents(events, target, recoveryTarget = null) {
  const correlated = events.filter(
    (event) =>
      sameSocketTarget(event, target) ||
      (recoveryTarget !== null && sameSocketTarget(event, recoveryTarget)),
  );
  if (correlated.length !== events.length) {
    throw new Error("foreign Socket.IO event was observed during failure gate");
  }
  return correlated.map((event, index) => ({
      sequence: index,
      observed_monotonic_ns: event.observedMonotonicNs.toString(),
      correlation_target: sameSocketTarget(event, target)
        ? "failure_target"
        : "recovery_target",
      type: event.type,
      done: event.done,
      has_error: event.hasError,
      content_utf8_bytes: event.contentUtf8Bytes,
      content_sha256: event.contentSha256,
    }));
}

function assertNoSensitiveSummary(summary, sensitiveValues) {
  const forbiddenKeys = new Set([
    "url",
    "token",
    "prompt",
    "body",
    "content",
    "assistant_text",
    "page_error_message",
  ]);
  const visit = (value) => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        if (forbiddenKeys.has(key.toLowerCase())) {
          throw new Error("summary contains a forbidden cleartext field");
        }
        visit(item);
      }
    }
  };
  visit(summary);
  const serialized = JSON.stringify(summary);
  for (const value of sensitiveValues) {
    if (typeof value === "string" && value.length >= 8 && serialized.includes(value)) {
      throw new Error("summary contains a sensitive cleartext value");
    }
  }
  return serialized;
}

function controlEvidence(stage, file, nonce, timeoutMs) {
  const expected = controlContent(stage, nonce);
  return {
    control_schema: CONTROL_SCHEMA,
    control_stage: stage,
    control_file: file,
    nonce,
    content_utf8_bytes: Buffer.byteLength(expected, "utf8"),
    content_sha256: sha256(expected),
    timeout_ms: timeoutMs,
  };
}

function redactedControlEvidence(stage, file, nonce, requested, observed) {
  const expected = controlContent(stage, nonce);
  return {
    control_schema: CONTROL_SCHEMA,
    control_stage: stage,
    ...identityEvidence(file, "control_file"),
    nonce_sha256: sha256(nonce),
    content_utf8_bytes: Buffer.byteLength(expected, "utf8"),
    content_sha256: sha256(expected),
    requested_monotonic_ns: requested.toString(),
    observed_monotonic_ns: observed.toString(),
  };
}

function writeStdoutLine(serialized) {
  return new Promise((resolve, reject) => {
    process.stdout.write(`${serialized}\n`, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

async function submitChat(page, actions, prompt) {
  const input = page.locator(INPUT_SELECTOR);
  await input.waitFor({ state: "visible", timeout: UI_TIMEOUT_MS });
  if (!(await input.isEnabled())) {
    throw new Error("OpenWebUI chat input is not enabled");
  }
  const started = monotonicNs();
  await input.fill(prompt, { timeout: UI_TIMEOUT_MS });
  await input.press("Enter", { timeout: UI_TIMEOUT_MS });
  const completed = monotonicNs();
  addBrowserAction(actions, {
    action: "submit_chat",
    selector: INPUT_SELECTOR,
    inputSha256: sha256(prompt),
    startedMonotonicNs: started,
    completedMonotonicNs: completed,
    result: exactBrowserResult({ visible: true, enabled: true }),
  });
  return { started, completed };
}

async function run(browser, config) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript((value) => {
    window.localStorage.setItem("token", value);
  }, config.token);

  const page = await context.newPage();
  page.setDefaultTimeout(UI_TIMEOUT_MS);
  const pageErrors = [];
  const events = [];
  page.on("pageerror", (error) => {
    const message = String(error?.message ?? error);
    pageErrors.push({
      observed_monotonic_ns: nsString(),
      message_utf8_bytes: Buffer.byteLength(message, "utf8"),
      message_sha256: sha256(message),
      raw: message,
    });
  });
  page.on("websocket", (websocket) => {
    websocket.on("framereceived", ({ payload }) => {
      const event = socketEvent(payload);
      if (!event) return;
      if (events.length >= MAX_SOCKET_EVENTS) {
        if (!pageErrors.some((item) => item.raw === "socket event bound exceeded")) {
          pageErrors.push({
            observed_monotonic_ns: nsString(),
            message_utf8_bytes: 0,
            message_sha256: sha256("socket event bound exceeded"),
            raw: "socket event bound exceeded",
          });
        }
        return;
      }
      events.push(event);
    });
  });

  const actions = [];
  const navigationUrl = new URL("/", config.baseUrl);
  navigationUrl.searchParams.set("temporary-chat", "true");
  navigationUrl.searchParams.set("models", MODEL_ID);
  const navigateStarted = monotonicNs();
  await page.goto(navigationUrl.toString(), {
    waitUntil: "domcontentloaded",
    timeout: NAVIGATION_TIMEOUT_MS,
  });
  const navigateCompleted = monotonicNs();
  addBrowserAction(actions, {
    action: "navigate",
    inputSha256: sha256(navigationUrl.toString()),
    startedMonotonicNs: navigateStarted,
    completedMonotonicNs: navigateCompleted,
    result: exactBrowserResult({ visible: true }),
  });

  const selectStarted = monotonicNs();
  await page.waitForFunction(
    (modelLabel) => document.body?.innerText?.includes(modelLabel) === true,
    MODEL_LABEL,
    { timeout: NAVIGATION_TIMEOUT_MS },
  );
  await page.locator(INPUT_SELECTOR).waitFor({
    state: "visible",
    timeout: NAVIGATION_TIMEOUT_MS,
  });
  const selectCompleted = monotonicNs();
  addBrowserAction(actions, {
    action: "select_model",
    selector: "body",
    inputSha256: sha256(MODEL_ID),
    startedMonotonicNs: selectStarted,
    completedMonotonicNs: selectCompleted,
    result: exactBrowserResult({ visible: true }),
  });

  const initialSubmit = await submitChat(page, actions, FAILURE_PROMPT);
  const visibleStarted = monotonicNs();
  const assistant = page.locator(ASSISTANT_SELECTOR).last();
  await assistant.waitFor({ state: "visible", timeout: VISIBLE_TIMEOUT_MS });
  await page.waitForFunction(
    ({ selector, minimum }) => {
      const values = [...document.querySelectorAll(selector)];
      return (
        values.length > 0 &&
        (values.at(-1)?.innerText?.trim().length ?? 0) >= minimum
      );
    },
    { selector: ASSISTANT_SELECTOR, minimum: config.minimumVisibleCharacters },
    { timeout: VISIBLE_TIMEOUT_MS },
  );
  const target = await waitUntil(
    () =>
      events.find(
        (event) =>
          event.type === "chat:completion" &&
          event.contentUtf8Bytes > 0 &&
          event.observedMonotonicNs >= initialSubmit.started,
      ),
    VISIBLE_TIMEOUT_MS,
    "the first correlated Socket.IO content event",
  );
  const visibleText = await assistant.innerText();
  if (visibleText.trim().length < config.minimumVisibleCharacters) {
    throw new Error("assistant content is not visibly non-empty");
  }
  const stopButton = page.locator(STOP_SELECTOR);
  await stopButton.waitFor({ state: "visible", timeout: UI_TIMEOUT_MS });
  if (!(await stopButton.isEnabled())) {
    throw new Error("OpenWebUI Stop button is not enabled before fault injection");
  }
  const targetBeforeKill = events.filter((event) => sameSocketTarget(event, target));
  if (
    targetBeforeKill.some(
      (event) => event.done || event.hasError || event.type === "chat:tasks:cancel",
    )
  ) {
    throw new Error("target request reached a terminal event before fault injection");
  }
  const visibleCompleted = monotonicNs();
  addBrowserAction(actions, {
    action: "wait_visible",
    selector: ASSISTANT_SELECTOR,
    startedMonotonicNs: visibleStarted,
    completedMonotonicNs: visibleCompleted,
    result: exactBrowserResult({ visible: true, text: visibleText }),
  });

  const killExpected = controlContent("worker_killed", config.killNonce);
  const killRequested = monotonicNs();
  if (pageErrors.length !== 0) {
    throw new Error("browser page errors were observed before fault injection");
  }
  const killInterim = {
    schema_version: SUMMARY_SCHEMA,
    record_type: "openwebui_failure_worker_kill_wait",
    browser_case: BROWSER_CASE,
    observed_monotonic_ns: killRequested.toString(),
    browser_actions: actions,
    socket_correlation: {
      target: {
        ...identityEvidence(target.chatId, "chat_id"),
        ...identityEvidence(target.messageId, "message_id"),
      },
      submit_completed_monotonic_ns: initialSubmit.completed.toString(),
      visible_completed_monotonic_ns: visibleCompleted.toString(),
      pre_fault_done_count: 0,
      pre_fault_error_count: 0,
      pre_fault_cancel_count: 0,
    },
    socket_events: redactSocketEvents(events, target),
    page_error_count: 0,
    worker_killed_control: controlEvidence(
      "worker_killed",
      config.killControlFile,
      config.killNonce,
      config.controlTimeoutMs,
    ),
  };
  const sensitiveValues = [
    config.baseUrl,
    config.token,
    FAILURE_PROMPT,
    RECOVERY_PROMPT,
    RECOVERY_MARKER,
    FAILURE_TEXT,
    visibleText,
    target.chatId,
    target.messageId,
    ...pageErrors.map((error) => error.raw),
  ];
  const killSerialized = assertNoSensitiveSummary(killInterim, sensitiveValues);
  await writeStdoutLine(killSerialized);
  const killObserved = await waitForControl(
    config.killControlFile,
    killExpected,
    config.controlTimeoutMs,
  );

  const failedStarted = monotonicNs();
  await page.getByText(FAILURE_TEXT, { exact: false }).last().waitFor({
    state: "visible",
    timeout: FAILURE_TIMEOUT_MS,
  });
  const errorEvent = await waitUntil(
    () =>
      events.find(
        (event) =>
          sameSocketTarget(event, target) &&
          event.type === "chat:completion" &&
          event.hasError &&
          event.observedMonotonicNs >= killRequested,
      ),
    FAILURE_TIMEOUT_MS,
    "the target Socket.IO provider error",
  );
  const cancelEvent = await waitUntil(
    () =>
      events.find(
        (event) =>
          sameSocketTarget(event, target) &&
          event.type === "chat:tasks:cancel" &&
          event.observedMonotonicNs >= errorEvent.observedMonotonicNs,
      ),
    FAILURE_TIMEOUT_MS,
    "the target Socket.IO cancellation",
  );
  await sleep(POST_FAILURE_STABLE_MS);
  const failedText = await assistant.innerText();
  await sleep(POST_FAILURE_STABLE_MS);
  if ((await assistant.innerText()) !== failedText) {
    throw new Error("assistant content changed after provider failure");
  }
  const targetAfterKill = events.filter(
    (event) =>
      sameSocketTarget(event, target) &&
      event.observedMonotonicNs >= killRequested,
  );
  const errorEvents = targetAfterKill.filter(
    (event) => event.type === "chat:completion" && event.hasError,
  );
  const cancelEvents = targetAfterKill.filter(
    (event) => event.type === "chat:tasks:cancel",
  );
  const doneEvents = targetAfterKill.filter(
    (event) => event.type === "chat:completion" && event.done,
  );
  const contentAfterError = targetAfterKill.filter(
    (event) =>
      event.type === "chat:completion" &&
      event.contentUtf8Bytes > 0 &&
      event.observedMonotonicNs > errorEvent.observedMonotonicNs,
  );
  if (errorEvents.length !== 1) {
    throw new Error("provider error event count differs");
  }
  if (cancelEvents.length !== 1 || cancelEvent !== cancelEvents[0]) {
    throw new Error("task cancellation event count differs");
  }
  if (doneEvents.length !== 0) {
    throw new Error("OpenWebUI emitted normal done after provider error");
  }
  if (contentAfterError.length !== 0) {
    throw new Error("OpenWebUI emitted content after provider error");
  }
  fs.mkdirSync(path.dirname(config.screenshotPath), { recursive: true, mode: 0o700 });
  const reservation = fs.openSync(config.screenshotPath, "wx", 0o600);
  fs.closeSync(reservation);
  await page.screenshot({ path: config.screenshotPath, fullPage: true });
  const screenshotSha256 = sha256(fs.readFileSync(config.screenshotPath));
  const failedCompleted = monotonicNs();
  addBrowserAction(actions, {
    action: "wait_failed",
    selector: ASSISTANT_SELECTOR,
    startedMonotonicNs: failedStarted,
    completedMonotonicNs: failedCompleted,
    result: exactBrowserResult({ visible: true, text: failedText }),
    screenshotFile: SCREENSHOT_FILE,
    screenshotSha256,
  });

  if (pageErrors.length !== 0) {
    throw new Error("browser page errors were observed before recovery wait");
  }
  const recoveryExpected = controlContent("gateway_recovered", config.recoveryNonce);
  const recoveryRequested = monotonicNs();
  const recoveryInterim = {
    schema_version: SUMMARY_SCHEMA,
    record_type: "openwebui_failure_gateway_recovery_wait",
    browser_case: BROWSER_CASE,
    observed_monotonic_ns: recoveryRequested.toString(),
    browser_actions: actions,
    socket_correlation: {
      target: {
        ...identityEvidence(target.chatId, "chat_id"),
        ...identityEvidence(target.messageId, "message_id"),
      },
      error_first_observed_monotonic_ns: errorEvent.observedMonotonicNs.toString(),
      cancel_first_observed_monotonic_ns: cancelEvent.observedMonotonicNs.toString(),
      error_event_count: errorEvents.length,
      cancel_event_count: cancelEvents.length,
      done_after_fault_count: doneEvents.length,
      content_after_error_count: contentAfterError.length,
    },
    socket_events: redactSocketEvents(events, target),
    page_error_count: 0,
    worker_killed_control: redactedControlEvidence(
      "worker_killed",
      config.killControlFile,
      config.killNonce,
      killRequested,
      killObserved,
    ),
    gateway_recovered_control: controlEvidence(
      "gateway_recovered",
      config.recoveryControlFile,
      config.recoveryNonce,
      config.controlTimeoutMs,
    ),
  };
  const recoverySerialized = assertNoSensitiveSummary(recoveryInterim, [
    ...sensitiveValues,
    failedText,
  ]);
  await writeStdoutLine(recoverySerialized);
  const recoveryObserved = await waitForControl(
    config.recoveryControlFile,
    recoveryExpected,
    config.controlTimeoutMs,
  );

  const readyStarted = monotonicNs();
  const input = page.locator(INPUT_SELECTOR);
  await input.waitFor({ state: "visible", timeout: UI_TIMEOUT_MS });
  if (!(await input.isEnabled())) {
    throw new Error("OpenWebUI input is disabled after gateway recovery");
  }
  const readyCompleted = monotonicNs();
  addBrowserAction(actions, {
    action: "wait_ready",
    selector: INPUT_SELECTOR,
    startedMonotonicNs: readyStarted,
    completedMonotonicNs: readyCompleted,
    result: exactBrowserResult({ visible: true, enabled: true }),
  });

  const assistantsBeforeRecovery = await page.locator(ASSISTANT_SELECTOR).count();
  const recoverySubmit = await submitChat(page, actions, RECOVERY_PROMPT);
  const recoveryVisibleStarted = monotonicNs();
  await page.waitForFunction(
    ({ selector, priorCount }) => document.querySelectorAll(selector).length > priorCount,
    { selector: ASSISTANT_SELECTOR, priorCount: assistantsBeforeRecovery },
    { timeout: RECOVERY_TIMEOUT_MS },
  );
  const recoveryAssistant = page.locator(ASSISTANT_SELECTOR).last();
  await recoveryAssistant.waitFor({ state: "visible", timeout: RECOVERY_TIMEOUT_MS });
  await page
    .locator(ASSISTANT_SELECTOR, { hasText: RECOVERY_MARKER })
    .last()
    .waitFor({ state: "visible", timeout: RECOVERY_TIMEOUT_MS });
  const recoveryTarget = await waitUntil(
    () =>
      events.find(
        (event) =>
          event.type === "chat:completion" &&
          event.contentUtf8Bytes > 0 &&
          event.observedMonotonicNs >= recoverySubmit.started &&
          event.chatId === target.chatId &&
          event.messageId !== target.messageId,
      ),
    RECOVERY_TIMEOUT_MS,
    "the recovery Socket.IO content event",
  );
  const recoveryText = (await recoveryAssistant.innerText()).trim();
  if (recoveryText !== RECOVERY_MARKER) {
    throw new Error("OpenWebUI recovery response differs from its marker");
  }
  const recoveryVisibleCompleted = monotonicNs();
  addBrowserAction(actions, {
    action: "wait_visible",
    selector: ASSISTANT_SELECTOR,
    startedMonotonicNs: recoveryVisibleStarted,
    completedMonotonicNs: recoveryVisibleCompleted,
    result: exactBrowserResult({ visible: true, text: recoveryText }),
  });

  const recoveryReadyStarted = monotonicNs();
  const recoveryDone = await waitUntil(
    () =>
      events.find(
        (event) =>
          sameSocketTarget(event, recoveryTarget) &&
          event.type === "chat:completion" &&
          event.done,
      ),
    RECOVERY_TIMEOUT_MS,
    "the recovery Socket.IO done event",
  );
  await sleep(100);
  await stopButton.waitFor({ state: "hidden", timeout: UI_TIMEOUT_MS });
  if (!(await input.isEnabled())) {
    throw new Error("OpenWebUI input is disabled after recovery completion");
  }
  const recoveryEvents = events.filter((event) => sameSocketTarget(event, recoveryTarget));
  const recoveryDoneEvents = recoveryEvents.filter(
    (event) => event.type === "chat:completion" && event.done,
  );
  const recoveryCancelEvents = recoveryEvents.filter(
    (event) => event.type === "chat:tasks:cancel",
  );
  const recoveryErrorEvents = recoveryEvents.filter((event) => event.hasError);
  if (recoveryDoneEvents.length !== 1 || recoveryDone !== recoveryDoneEvents[0]) {
    throw new Error("recovery Socket.IO done correlation differs");
  }
  if (recoveryCancelEvents.length !== 0 || recoveryErrorEvents.length !== 0) {
    throw new Error("recovery chat was cancelled or failed");
  }
  if ((await recoveryAssistant.innerText()).trim() !== recoveryText) {
    throw new Error("recovery assistant content changed after done");
  }
  const recoveryReadyCompleted = monotonicNs();
  addBrowserAction(actions, {
    action: "wait_ready",
    selector: INPUT_SELECTOR,
    startedMonotonicNs: recoveryReadyStarted,
    completedMonotonicNs: recoveryReadyCompleted,
    result: exactBrowserResult({ visible: true, enabled: true, text: recoveryText }),
  });

  validateActionSequence(actions);
  const temporaryChatObserved = await page.evaluate(() => {
    const query = new URLSearchParams(window.location.search);
    return (
      window.location.pathname === "/" && query.get("temporary-chat") === "true"
    );
  });
  if (!temporaryChatObserved) {
    throw new Error("OpenWebUI failure chat did not remain temporary");
  }
  await context.close();
  if (context.pages().length !== 0) {
    throw new Error("browser context retained a page after final close");
  }
  if (pageErrors.length !== 0) {
    throw new Error("browser page errors were observed");
  }
  const sealedTargetAfterKill = events.filter(
    (event) =>
      sameSocketTarget(event, target) && event.observedMonotonicNs >= killRequested,
  );
  const sealedRecoveryEvents = events.filter((event) =>
    sameSocketTarget(event, recoveryTarget),
  );
  if (
    sealedTargetAfterKill.length !== targetAfterKill.length ||
    sealedRecoveryEvents.length !== recoveryEvents.length
  ) {
    throw new Error("Socket.IO evidence changed while closing the browser context");
  }
  const screenshotStat = fs.statSync(config.screenshotPath, { bigint: true });
  if (!screenshotStat.isFile() || screenshotStat.size < 1n) {
    throw new Error("failure screenshot is absent or empty");
  }
  if (screenshotStat.size > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("failure screenshot size exceeds exact JSON integer range");
  }

  const summary = {
    schema_version: SUMMARY_SCHEMA,
    record_type: "openwebui_failure_smoke",
    browser_case: BROWSER_CASE,
    observed_monotonic_ns: nsString(),
    browser_actions: actions,
    socket_correlation: {
      target: {
        ...identityEvidence(target.chatId, "chat_id"),
        ...identityEvidence(target.messageId, "message_id"),
      },
      error_first_observed_monotonic_ns: errorEvent.observedMonotonicNs.toString(),
      cancel_first_observed_monotonic_ns: cancelEvent.observedMonotonicNs.toString(),
      error_event_count: errorEvents.length,
      cancel_event_count: cancelEvents.length,
      done_after_fault_count: doneEvents.length,
      content_after_error_count: contentAfterError.length,
      recovery: {
        ...identityEvidence(recoveryTarget.chatId, "chat_id"),
        ...identityEvidence(recoveryTarget.messageId, "message_id"),
        submit_completed_monotonic_ns: recoverySubmit.completed.toString(),
        done_observed_monotonic_ns: recoveryDone.observedMonotonicNs.toString(),
        done_event_count: recoveryDoneEvents.length,
        cancel_event_count: recoveryCancelEvents.length,
        error_event_count: recoveryErrorEvents.length,
      },
    },
    page_error_count: 0,
    page_errors: [],
    socket_events: redactSocketEvents(events, target, recoveryTarget),
    controls: {
      worker_killed: redactedControlEvidence(
        "worker_killed",
        config.killControlFile,
        config.killNonce,
        killRequested,
        killObserved,
      ),
      gateway_recovered: redactedControlEvidence(
        "gateway_recovered",
        config.recoveryControlFile,
        config.recoveryNonce,
        recoveryRequested,
        recoveryObserved,
      ),
    },
    screenshot: {
      screenshot_file: SCREENSHOT_FILE,
      screenshot_bytes: Number(screenshotStat.size),
      screenshot_sha256: screenshotSha256,
    },
  };
  const serialized = assertNoSensitiveSummary(summary, [
    ...sensitiveValues,
    failedText,
    recoveryText,
    recoveryTarget.chatId,
    recoveryTarget.messageId,
  ]);
  fs.mkdirSync(path.dirname(config.summaryFile), { recursive: true, mode: 0o700 });
  fs.writeFileSync(config.summaryFile, `${serialized}\n`, {
    encoding: "utf8",
    mode: 0o600,
    flag: "wx",
  });
  await writeStdoutLine(serialized);
}

async function main() {
  const config = loadConfig();
  const { chromium } = require("playwright");
  const browser = await chromium.launch({ headless: true });
  try {
    await run(browser, config);
  } finally {
    await browser.close();
  }
}

function safeFailure(error) {
  const name = String(error?.name ?? "Error");
  const message = String(error?.message ?? error);
  return {
    schema_version: SUMMARY_SCHEMA,
    record_type: "openwebui_failure_smoke_failure",
    observed_monotonic_ns: nsString(),
    error_name_utf8_bytes: Buffer.byteLength(name, "utf8"),
    error_name_sha256: sha256(name),
    error_message_utf8_bytes: Buffer.byteLength(message, "utf8"),
    error_message_sha256: sha256(message),
  };
}

module.exports = {
  BROWSER_CASE,
  CONTROL_SCHEMA,
  EXPECTED_ACTION_SEQUENCE,
  FAILURE_PROMPT,
  RECOVERY_MARKER,
  RECOVERY_PROMPT,
  SCREENSHOT_FILE,
  SUMMARY_SCHEMA,
  assertNoSensitiveSummary,
  controlContent,
  readStrictToken,
  redactSocketEvents,
  socketEvent,
  strictSingleLineToken,
  waitForControl,
};

if (require.main === module) {
  main().catch((error) => {
    console.error(JSON.stringify(safeFailure(error)));
    process.exitCode = 1;
  });
}
