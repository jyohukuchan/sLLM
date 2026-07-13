#!/usr/bin/env node

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const SUMMARY_SCHEMA = "ullm.openwebui.browser_soak.v1";
const COMBINED_SUMMARY_SCHEMA = "ullm.openwebui.browser_smoke_soak.v1";
const RUN_CASE = "openwebui_20_chat_soak";
const COMBINED_RUN_CASE = "openwebui_smoke_and_20_chat_soak";
const COMBINED_MODE = "smoke_then_soak20";
const CASE_PREFIX = "openwebui_soak_chat_";
const SMOKE_CASE = "openwebui_smoke";
const SMOKE_MARKER = "OPENWEBUI_SMOKE_OK";
const SOAK_COUNT_TEXT = process.env.ULLM_OPENWEBUI_SOAK_COUNT || "20";
if (SOAK_COUNT_TEXT !== "20" && SOAK_COUNT_TEXT !== "100") {
  throw new Error("ULLM_OPENWEBUI_SOAK_COUNT must be 20 or 100");
}
const SOAK_CHAT_COUNT = Number.parseInt(SOAK_COUNT_TEXT, 10);
const RUN_CASE_FOR_COUNT = `openwebui_${SOAK_CHAT_COUNT}_chat_soak`;
const COMBINED_RUN_CASE_FOR_COUNT =
  `openwebui_smoke_and_${SOAK_CHAT_COUNT}_chat_soak`;
const COMBINED_MODE_FOR_COUNT = `smoke_then_soak${SOAK_CHAT_COUNT}`;
const MODEL_ID = process.env.ULLM_MODEL_ID || "ullm-qwen3-14b-sq8";
const MODEL_LABEL = process.env.ULLM_MODEL_NAME || "uLLM Qwen3 14B SQ8";
const ASSISTANT_SELECTOR = ".chat-assistant";
const INPUT_SELECTOR = "#chat-input";
const STOP_SELECTOR =
  '#message-input-container button:has(svg[viewBox="0 0 24 24"] path[d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm6-2.438c0-.724.588-1.312 1.313-1.312h4.874c.725 0 1.313.588 1.313 1.313v4.874c0 .725-.588 1.313-1.313 1.313H9.564a1.312 1.312 0 01-1.313-1.313V9.564z"])';
const EXPECTED_ACTION_SEQUENCE = [
  "navigate",
  "select_model",
  "submit_chat",
  "wait_visible",
  "wait_ready",
];

const NAVIGATION_TIMEOUT_MS = 60_000;
const RESPONSE_TIMEOUT_MS = 90_000;
const READY_TIMEOUT_MS = 15_000;
const SOCKET_POLL_MS = 20;
const POST_DONE_STABLE_MS = 100;
const MAX_SOCKET_EVENTS_PER_CASE = 128;

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
const caseSuffix = (caseIndex) => String(caseIndex).padStart(2, "0");
const browserCase = (caseIndex) => `${CASE_PREFIX}${caseSuffix(caseIndex)}`;
const caseMarker = (caseIndex) => `OPENWEBUI_SOAK_OK_${caseSuffix(caseIndex)}`;
const casePrompt = (caseIndex) =>
  `Reply with exactly ${caseMarker(caseIndex)} and nothing else.`;

function caseSchedule(mode) {
  const soak = [];
  for (let caseIndex = 1; caseIndex <= SOAK_CHAT_COUNT; caseIndex += 1) {
    soak.push({
      caseIndex,
      caseKind: "soak",
      browserCase: browserCase(caseIndex),
      marker: caseMarker(caseIndex),
      prompt: casePrompt(caseIndex),
      recordType: "openwebui_soak_chat",
    });
  }
  if (mode === `soak${SOAK_CHAT_COUNT}`) return soak;
  if (mode !== COMBINED_MODE_FOR_COUNT) {
    throw new Error("browser soak mode is unsupported");
  }
  return [
    {
      caseIndex: 0,
      caseKind: "smoke",
      browserCase: SMOKE_CASE,
      marker: SMOKE_MARKER,
      prompt: `Reply with exactly ${SMOKE_MARKER} and nothing else.`,
      recordType: "openwebui_smoke_chat",
    },
    ...soak,
  ];
}

const scheduleEvidence = (schedule) =>
  schedule.map((item, position) => ({
    position,
    case_index: item.caseIndex,
    case_kind: item.caseKind,
    browser_case: item.browserCase,
  }));

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
  if (Buffer.byteLength(text, "utf8") < 8) {
    throw new Error("OpenWebUI test token is shorter than the evidence guard minimum");
  }
  if (text.trim() !== text) {
    throw new Error("OpenWebUI test token has surrounding whitespace");
  }
  return text;
}

function normalizedBaseUrl(raw) {
  const value = new URL(raw);
  if (!['http:', 'https:'].includes(value.protocol)) {
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
  const summaryFile =
    process.env.OPENWEBUI_SOAK_SUMMARY || "/output/openwebui-soak-summary.json";
  const mode =
    process.env.OPENWEBUI_SOAK_MODE || `soak${SOAK_CHAT_COUNT}`;
  if (mode !== `soak${SOAK_CHAT_COUNT}` && mode !== COMBINED_MODE_FOR_COUNT) {
    throw new Error("OPENWEBUI_SOAK_MODE is unsupported");
  }
  const tokenStat = fs.lstatSync(tokenFile, { bigint: true });
  if (!tokenStat.isFile() || tokenStat.size < 1n || tokenStat.size > 65_536n) {
    throw new Error("OpenWebUI test token file has an invalid type or size");
  }
  const token = strictSingleLineToken(fs.readFileSync(tokenFile));
  requireAbsent(summaryFile, "browser soak summary output");
  return { baseUrl, mode, token, summaryFile };
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

function addBrowserAction(actions, browserCaseId, fields) {
  const action = {
    browser_case: browserCaseId,
    action_index: actions.length,
    action: fields.action,
    selector: fields.selector ?? null,
    input_sha256: fields.inputSha256 ?? null,
    started_monotonic_ns: fields.startedMonotonicNs.toString(),
    completed_monotonic_ns: fields.completedMonotonicNs.toString(),
    result: fields.result,
    screenshot_file: null,
    screenshot_sha256: null,
  };
  if (fields.completedMonotonicNs < fields.startedMonotonicNs) {
    throw new Error("browser action monotonic timestamps regress");
  }
  actions.push(action);
  return action;
}

function validateActionSequence(actions) {
  const observed = actions.map((action) => action.action);
  if (
    observed.length !== EXPECTED_ACTION_SEQUENCE.length ||
    observed.some((action, index) => action !== EXPECTED_ACTION_SEQUENCE[index])
  ) {
    throw new Error("browser action sequence differs from the frozen soak schedule");
  }
}

function sameSocketTarget(event, target) {
  return event.chatId === target.chatId && event.messageId === target.messageId;
}

function validateTargetEvents(events) {
  if (events.length === 0 || events.length > MAX_SOCKET_EVENTS_PER_CASE) {
    throw new Error("correlated Socket.IO event count is outside its bound");
  }
  const allowed = new Set([
    "chat:active",
    "chat:completion",
    "chat:outlet",
    "chat:tasks:cancel",
  ]);
  for (const event of events) {
    if (!allowed.has(event.type)) {
      throw new Error("correlated Socket.IO event type differs");
    }
    if (
      ["chat:active", "chat:outlet"].includes(event.type) &&
      (event.done || event.contentUtf8Bytes !== 0 || event.contentSha256 !== null)
    ) {
      throw new Error("Socket.IO state event carries content or terminal state");
    }
    if (event.done && event.type !== "chat:completion") {
      throw new Error("Socket.IO non-completion event is terminal");
    }
    if (event.hasError) {
      throw new Error("provider error was observed in Socket.IO evidence");
    }
  }
  const contentEvents = events.filter(
    (event) => event.type === "chat:completion" && event.contentUtf8Bytes > 0,
  );
  const doneEvents = events.filter(
    (event) => event.type === "chat:completion" && event.done,
  );
  const cancelEvents = events.filter(
    (event) => event.type === "chat:tasks:cancel",
  );
  if (contentEvents.length < 1 || doneEvents.length !== 1 || cancelEvents.length !== 0) {
    throw new Error("Socket.IO content, done, or cancellation count differs");
  }
  const doneIndex = events.indexOf(doneEvents[0]);
  if (
    events.some(
      (event, index) => index > doneIndex && event.type === "chat:completion",
    )
  ) {
    throw new Error("Socket.IO completion event followed terminal done");
  }
  return { contentEvents, doneEvents, cancelEvents };
}

function redactSocketEvents(events) {
  return events.map((event, index) => ({
    sequence: index,
    observed_monotonic_ns: event.observedMonotonicNs.toString(),
    correlation_target: "chat_target",
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
    "response",
    "assistant_text",
    "page_error_message",
    "chat_id",
    "message_id",
  ]);
  const visit = (value) => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        const normalized = key.toLowerCase();
        if (
          forbiddenKeys.has(normalized) ||
          [...forbiddenKeys].some((name) => normalized.endsWith(`_${name}`))
        ) {
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

function writeStdoutLine(serialized) {
  return new Promise((resolve, reject) => {
    process.stdout.write(`${serialized}\n`, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

async function runCase(context, config, spec, tracker) {
  const { browserCase: caseId, caseIndex, marker, prompt, recordType } = spec;
  const actions = [];
  const events = [];
  const pageErrors = [];
  const page = await context.newPage();
  tracker.pagesCreated += 1;
  tracker.openPages += 1;
  tracker.maximumOpenPages = Math.max(tracker.maximumOpenPages, tracker.openPages);
  if (tracker.openPages !== 1 || context.pages().length !== 1) {
    throw new Error("browser page concurrency differs from one");
  }
  page.setDefaultTimeout(READY_TIMEOUT_MS);
  page.on("pageerror", (error) => {
    const message = String(error?.message ?? error);
    pageErrors.push({
      observedMonotonicNs: monotonicNs(),
      messageUtf8Bytes: Buffer.byteLength(message, "utf8"),
      messageSha256: sha256(message),
      raw: message,
    });
  });
  page.on("websocket", (websocket) => {
    websocket.on("framereceived", ({ payload }) => {
      const event = socketEvent(payload);
      if (!event) return;
      if (events.length >= MAX_SOCKET_EVENTS_PER_CASE * 4) {
        if (pageErrors.some((item) => item.raw === "socket event bound exceeded")) {
          return;
        }
        pageErrors.push({
          observedMonotonicNs: monotonicNs(),
          messageUtf8Bytes: 0,
          messageSha256: sha256("socket event bound exceeded"),
          raw: "socket event bound exceeded",
        });
        return;
      }
      events.push(event);
    });
  });

  let evidence;
  let targetForFinalCheck = null;
  try {
    const navigationUrl = new URL("/", config.baseUrl);
    navigationUrl.searchParams.set("temporary-chat", "true");
    navigationUrl.searchParams.set("models", MODEL_ID);
    const navigateStarted = monotonicNs();
    await page.goto(navigationUrl.toString(), {
      waitUntil: "domcontentloaded",
      timeout: NAVIGATION_TIMEOUT_MS,
    });
    const navigateCompleted = monotonicNs();
    addBrowserAction(actions, caseId, {
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
    const input = page.locator(INPUT_SELECTOR);
    await input.waitFor({ state: "visible", timeout: NAVIGATION_TIMEOUT_MS });
    if (!(await input.isEnabled())) {
      throw new Error("OpenWebUI chat input is not enabled before submit");
    }
    const selectCompleted = monotonicNs();
    addBrowserAction(actions, caseId, {
      action: "select_model",
      selector: "body",
      inputSha256: sha256(MODEL_ID),
      startedMonotonicNs: selectStarted,
      completedMonotonicNs: selectCompleted,
      result: exactBrowserResult({ visible: true }),
    });

    const submitStarted = monotonicNs();
    await input.fill(prompt, { timeout: READY_TIMEOUT_MS });
    await input.press("Enter", { timeout: READY_TIMEOUT_MS });
    const submitCompleted = monotonicNs();
    addBrowserAction(actions, caseId, {
      action: "submit_chat",
      selector: INPUT_SELECTOR,
      inputSha256: sha256(prompt),
      startedMonotonicNs: submitStarted,
      completedMonotonicNs: submitCompleted,
      result: exactBrowserResult({ visible: true, enabled: true }),
    });

    const visibleStarted = monotonicNs();
    const assistant = page.locator(ASSISTANT_SELECTOR).last();
    await assistant.waitFor({ state: "visible", timeout: RESPONSE_TIMEOUT_MS });
    await page.waitForFunction(
      ({ selector, expected }) => {
        const values = [...document.querySelectorAll(selector)];
        return values.at(-1)?.innerText?.includes(expected) === true;
      },
      { selector: ASSISTANT_SELECTOR, expected: marker },
      { timeout: RESPONSE_TIMEOUT_MS },
    );
    const target = await waitUntil(
      () =>
        events.find(
          (event) =>
            event.type === "chat:completion" &&
            event.contentUtf8Bytes > 0 &&
            event.observedMonotonicNs >= submitStarted &&
            event.chatId &&
            event.messageId,
        ),
      RESPONSE_TIMEOUT_MS,
      "the correlated Socket.IO content event",
    );
    targetForFinalCheck = target;
    const visibleText = (await assistant.innerText()).trim();
    if (visibleText !== marker) {
      throw new Error("OpenWebUI visible response differs from its marker");
    }
    const visibleCompleted = monotonicNs();
    addBrowserAction(actions, caseId, {
      action: "wait_visible",
      selector: ASSISTANT_SELECTOR,
      startedMonotonicNs: visibleStarted,
      completedMonotonicNs: visibleCompleted,
      result: exactBrowserResult({ visible: true, text: visibleText }),
    });

    const readyStarted = monotonicNs();
    const done = await waitUntil(
      () =>
        events.find(
          (event) =>
            sameSocketTarget(event, target) &&
            event.type === "chat:completion" &&
            event.done,
        ),
      RESPONSE_TIMEOUT_MS,
      "the correlated Socket.IO done event",
    );
    await sleep(POST_DONE_STABLE_MS);
    await page.locator(STOP_SELECTOR).waitFor({
      state: "hidden",
      timeout: READY_TIMEOUT_MS,
    });
    await input.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    if (!(await input.isEnabled())) {
      throw new Error("OpenWebUI input is disabled after completion");
    }
    const finalText = (await assistant.innerText()).trim();
    if (finalText !== marker) {
      throw new Error("OpenWebUI completed response differs from its marker");
    }
    const temporaryChatObserved = await page.evaluate(() => {
      const query = new URLSearchParams(window.location.search);
      return (
        window.location.pathname === "/" && query.get("temporary-chat") === "true"
      );
    });
    if (!temporaryChatObserved) {
      throw new Error("OpenWebUI chat did not remain temporary");
    }
    const targetEvents = events.filter((event) => sameSocketTarget(event, target));
    const counts = validateTargetEvents(targetEvents);
    if (done !== counts.doneEvents[0]) {
      throw new Error("Socket.IO done correlation differs");
    }
    if (pageErrors.length !== 0) {
      throw new Error("browser page errors were observed");
    }
    const readyCompleted = monotonicNs();
    addBrowserAction(actions, caseId, {
      action: "wait_ready",
      selector: INPUT_SELECTOR,
      startedMonotonicNs: readyStarted,
      completedMonotonicNs: readyCompleted,
      result: exactBrowserResult({ visible: true, enabled: true, text: finalText }),
    });
    validateActionSequence(actions);

    evidence = {
      schema_version:
        config.mode === COMBINED_MODE ? COMBINED_SUMMARY_SCHEMA : SUMMARY_SCHEMA,
      record_type: recordType,
      browser_case: caseId,
      case_index: caseIndex,
      observed_monotonic_ns: nsString(),
      browser_actions: actions,
      socket_correlation: {
        target: {
          ...identityEvidence(target.chatId, "chat_id"),
          ...identityEvidence(target.messageId, "message_id"),
        },
        submit_started_monotonic_ns: submitStarted.toString(),
        submit_completed_monotonic_ns: submitCompleted.toString(),
        first_content_observed_monotonic_ns:
          counts.contentEvents[0].observedMonotonicNs.toString(),
        done_observed_monotonic_ns: done.observedMonotonicNs.toString(),
        done_event_count: counts.doneEvents.length,
        cancellation_event_count: counts.cancelEvents.length,
        provider_error_count: targetEvents.filter((event) => event.hasError).length,
      },
      socket_events: redactSocketEvents(targetEvents),
      visible_marker: {
        expected_marker_utf8_bytes: Buffer.byteLength(marker, "utf8"),
        expected_marker_sha256: sha256(marker),
        observed: true,
      },
      page_error_count: 0,
      page_errors: [],
      page_state: {
        page_index: caseIndex,
        temporary_chat: temporaryChatObserved,
        created: true,
        closed: true,
        open_pages_after_close: 0,
      },
    };
    evidence.sensitiveValues = [
      config.baseUrl,
      config.token,
      MODEL_ID,
      MODEL_LABEL,
      marker,
      prompt,
      visibleText,
      finalText,
      target.chatId,
      target.messageId,
      ...pageErrors.map((error) => error.raw),
    ];
  } finally {
    if (!page.isClosed()) await page.close();
    tracker.pagesClosed += 1;
    tracker.openPages -= 1;
  }

  if (tracker.openPages !== 0 || context.pages().length !== 0) {
    throw new Error("temporary browser page did not close cleanly");
  }
  if (pageErrors.length !== 0) {
    throw new Error("browser page errors were observed during page close");
  }
  const finalTargetEvents = events.filter((event) =>
    sameSocketTarget(event, targetForFinalCheck),
  );
  const finalCounts = validateTargetEvents(finalTargetEvents);
  if (
    finalCounts.contentEvents[0].observedMonotonicNs.toString() !==
      evidence.socket_correlation.first_content_observed_monotonic_ns ||
    finalCounts.doneEvents[0].observedMonotonicNs.toString() !==
      evidence.socket_correlation.done_observed_monotonic_ns
  ) {
    throw new Error("Socket.IO correlation changed while closing the page");
  }
  evidence.socket_events = redactSocketEvents(finalTargetEvents);
  evidence.socket_correlation.done_event_count = finalCounts.doneEvents.length;
  evidence.socket_correlation.cancellation_event_count =
    finalCounts.cancelEvents.length;
  evidence.socket_correlation.provider_error_count = finalTargetEvents.filter(
    (event) => event.hasError,
  ).length;
  const sensitiveValues = evidence.sensitiveValues;
  delete evidence.sensitiveValues;
  evidence.observed_monotonic_ns = nsString();
  return {
    value: evidence,
    serialized: assertNoSensitiveSummary(evidence, sensitiveValues),
  };
}

async function run(browser, config) {
  const schedule = caseSchedule(config.mode);
  const tracker = {
    browserProcesses: 1,
    contextsCreated: 0,
    contextsClosed: 0,
    pagesCreated: 0,
    pagesClosed: 0,
    openPages: 0,
    maximumOpenPages: 0,
  };
  const caseRecordSha256 = [];
  let actionCount = 0;
  let socketEventCount = 0;
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  tracker.contextsCreated += 1;
  try {
    await context.addInitScript((value) => {
      window.localStorage.setItem("token", value);
    }, config.token);
    for (const spec of schedule) {
      const record = await runCase(context, config, spec, tracker);
      actionCount += record.value.browser_actions.length;
      socketEventCount += record.value.socket_events.length;
      caseRecordSha256.push(sha256(record.serialized));
      await writeStdoutLine(record.serialized);
    }
  } finally {
    await context.close();
    tracker.contextsClosed += 1;
  }

  if (
    tracker.browserProcesses !== 1 ||
    tracker.contextsCreated !== 1 ||
    tracker.contextsClosed !== 1 ||
    tracker.pagesCreated !== schedule.length ||
    tracker.pagesClosed !== schedule.length ||
    tracker.openPages !== 0 ||
    tracker.maximumOpenPages !== 1
  ) {
    throw new Error("bounded browser process, context, or page count differs");
  }
  const summary = {
    schema_version:
      config.mode === COMBINED_MODE_FOR_COUNT
        ? COMBINED_SUMMARY_SCHEMA
        : SUMMARY_SCHEMA,
    record_type:
      config.mode === COMBINED_MODE_FOR_COUNT
        ? "openwebui_smoke_soak_summary"
        : "openwebui_soak_summary",
    browser_case:
      config.mode === COMBINED_MODE_FOR_COUNT
        ? COMBINED_RUN_CASE_FOR_COUNT
        : RUN_CASE_FOR_COUNT,
    observed_monotonic_ns: nsString(),
    chat_count: schedule.length,
    action_count: actionCount,
    socket_event_count: socketEventCount,
    browser_process_count: tracker.browserProcesses,
    browser_context_count: tracker.contextsCreated,
    browser_context_closed_count: tracker.contextsClosed,
    page_count_created: tracker.pagesCreated,
    page_count_closed: tracker.pagesClosed,
    maximum_open_pages: tracker.maximumOpenPages,
    page_error_count: 0,
    cancellation_event_count: 0,
    provider_error_count: 0,
    case_record_sha256: caseRecordSha256,
  };
  if (config.mode === COMBINED_MODE_FOR_COUNT) {
    summary.mode = COMBINED_MODE_FOR_COUNT;
    summary.schedule = scheduleEvidence(schedule);
  }
  const serialized = assertNoSensitiveSummary(summary, [
    config.baseUrl,
    config.token,
    MODEL_ID,
    MODEL_LABEL,
    ...schedule.map((item) => item.marker),
    ...schedule.map((item) => item.prompt),
  ]);
  fs.mkdirSync(path.dirname(config.summaryFile), {
    recursive: true,
    mode: 0o700,
  });
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
    schema_version:
      process.env.OPENWEBUI_SOAK_MODE === COMBINED_MODE_FOR_COUNT
        ? COMBINED_SUMMARY_SCHEMA
        : SUMMARY_SCHEMA,
    record_type: "openwebui_soak_failure",
    observed_monotonic_ns: nsString(),
    error_name_utf8_bytes: Buffer.byteLength(name, "utf8"),
    error_name_sha256: sha256(name),
    error_message_utf8_bytes: Buffer.byteLength(message, "utf8"),
    error_message_sha256: sha256(message),
  };
}

module.exports = {
  CASE_PREFIX,
  COMBINED_MODE,
  COMBINED_MODE_FOR_COUNT,
  COMBINED_RUN_CASE,
  COMBINED_RUN_CASE_FOR_COUNT,
  COMBINED_SUMMARY_SCHEMA,
  EXPECTED_ACTION_SEQUENCE,
  MODEL_ID,
  MODEL_LABEL,
  RUN_CASE,
  RUN_CASE_FOR_COUNT,
  SMOKE_CASE,
  SMOKE_MARKER,
  SOAK_CHAT_COUNT,
  SUMMARY_SCHEMA,
  assertNoSensitiveSummary,
  browserCase,
  caseMarker,
  casePrompt,
  caseSchedule,
  scheduleEvidence,
  socketEvent,
  strictSingleLineToken,
  validateTargetEvents,
};

if (require.main === module) {
  main().catch((error) => {
    console.error(JSON.stringify(safeFailure(error)));
    process.exitCode = 1;
  });
}
