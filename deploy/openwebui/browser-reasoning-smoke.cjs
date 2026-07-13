#!/usr/bin/env node

const crypto = require("node:crypto");
const fs = require("node:fs");
const net = require("node:net");

const SUMMARY_SCHEMA = "ullm.openwebui.reasoning_browser_smoke.v2";
const BASE_URL = process.env.OPENWEBUI_URL || "http://192.168.0.66:3000";
const MODEL_ID = process.env.ULLM_MODEL_ID || "ullm-qwen3.5-9b-aq4";
const MODEL_LABEL = process.env.ULLM_MODEL_NAME || "uLLM Qwen3.5 9B AQ4";
const SWITCH_MODEL_ID =
  process.env.OPENWEBUI_SWITCH_MODEL_ID || "llama-qwen3.5-9b-ud-q4";
const SWITCH_MODEL_LABEL =
  process.env.OPENWEBUI_SWITCH_MODEL_NAME ||
  "llama.cpp Qwen3.5 9B UD-Q4_K_XL";
const TOKEN_FILE = process.env.OPENWEBUI_TOKEN_FILE || "/run/secrets/openwebui-token";
const FIRST_MARKER = process.env.OPENWEBUI_REASONING_ANSWER || "REASONING_BROWSER_OK";
const SECOND_MARKER =
  process.env.OPENWEBUI_REASONING_SECOND_ANSWER || "REASONING_HISTORY_OK";
const FIRST_PROMPT =
  process.env.OPENWEBUI_REASONING_PROMPT ||
  `Reply exactly ${FIRST_MARKER} and nothing else.`;
const SECOND_PROMPT =
  process.env.OPENWEBUI_REASONING_SECOND_PROMPT ||
  `Reply exactly ${SECOND_MARKER} and nothing else.`;
const SWITCH_MARKER =
  process.env.OPENWEBUI_REASONING_SWITCH_ANSWER || "PROVIDER_SWITCH_OK";
const SWITCH_PROMPT =
  process.env.OPENWEBUI_REASONING_SWITCH_PROMPT ||
  `Reply exactly ${SWITCH_MARKER} and nothing else.`;
const SWITCH_BACK_MARKER =
  process.env.OPENWEBUI_REASONING_SWITCH_BACK_ANSWER || "PROVIDER_RETURN_OK";
const SWITCH_BACK_PROMPT =
  process.env.OPENWEBUI_REASONING_SWITCH_BACK_PROMPT ||
  `Reply exactly ${SWITCH_BACK_MARKER} and nothing else.`;
const INPUT_SELECTOR = "#chat-input";
const ASSISTANT_SELECTOR = ".chat-assistant";
const TOGGLE_SELECTOR = "div.w-fit.text-gray-500";
const RESPONSE_TIMEOUT_MS = 120_000;
const NAVIGATION_TIMEOUT_MS = 60_000;
const MAX_POST_DATA_BYTES = 2 * 1024 * 1024;
const TRANSITION_SOCKET = process.env.OPENWEBUI_TRANSITION_SOCKET || "";

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function textEvidence(value) {
  return {
    utf8_bytes: Buffer.byteLength(value, "utf8"),
    sha256: sha256(value),
  };
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

function strictToken(raw) {
  let token;
  try {
    token = new TextDecoder("utf-8", { fatal: true }).decode(raw).trim();
  } catch {
    throw new Error("OpenWebUI token is not strict UTF-8");
  }
  if (!token || token.length > 65_536 || /[\r\n\0]/u.test(token)) {
    throw new Error("OpenWebUI token is invalid");
  }
  return token;
}

function hasKey(value, wanted) {
  if (Array.isArray(value)) return value.some((child) => hasKey(child, wanted));
  if (!value || typeof value !== "object") return false;
  return Object.entries(value).some(
    ([key, child]) => key === wanted || hasKey(child, wanted),
  );
}

function assistantHasReasoningContent(value) {
  if (!value || typeof value !== "object") {
    throw new Error("OpenWebUI request is not an object");
  }
  const messages = Array.isArray(value.messages)
    ? value.messages
    : value.user_message && typeof value.user_message === "object"
      ? [value.user_message]
      : null;
  if (messages === null) throw new Error("OpenWebUI request has no message payload");
  return messages.some(
    (message) =>
      message &&
      typeof message === "object" &&
      message.role === "assistant" &&
      Object.prototype.hasOwnProperty.call(message, "reasoning_content"),
  );
}

function summarizeRequestBody(raw) {
  if (typeof raw !== "string" || Buffer.byteLength(raw, "utf8") > MAX_POST_DATA_BYTES) {
    throw new Error("OpenWebUI provider request body is outside its bound");
  }
  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error("OpenWebUI provider request body is not JSON");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("OpenWebUI provider request body is not an object");
  }
  if (typeof value.model !== "string" || !value.model) {
    throw new Error("OpenWebUI provider request has no model ID");
  }
  return {
    sha256: sha256(raw),
    utf8_bytes: Buffer.byteLength(raw, "utf8"),
    model_id_sha256: sha256(value.model),
    has_reasoning_content_key: hasKey(value, "reasoning_content"),
    assistant_has_reasoning_content: assistantHasReasoningContent(value),
  };
}

function requestIsChatCompletion(request) {
  try {
    const url = new URL(request.url());
    return request.method() === "POST" && url.pathname.endsWith("/api/chat/completions");
  } catch {
    return false;
  }
}

async function waitForAnswer(page, marker) {
  const assistant = page.locator(ASSISTANT_SELECTOR).last();
  await assistant.waitFor({ state: "visible", timeout: RESPONSE_TIMEOUT_MS });
  await page.waitForFunction(
    ({ selector, expected }) =>
      [...document.querySelectorAll(selector)].at(-1)?.innerText?.includes(expected) === true,
    { selector: ASSISTANT_SELECTOR, expected: marker },
    { timeout: RESPONSE_TIMEOUT_MS },
  );
  return assistant;
}

async function visibleAnswerText(assistant) {
  const result = await assistant.evaluate((element) => {
    const fullText = element.innerText || "";
    const toggle = element.querySelector("div.w-fit.text-gray-500");
    const reasoningBlock = toggle?.closest(".w-full.space-y-1");
    const reasoningText = reasoningBlock?.innerText || "";
    return fullText.replace(reasoningText, "").trim();
  });
  return result;
}

function rememberChatId(page, chatIds) {
  const match = new URL(page.url()).pathname.match(/^\/c\/([0-9a-f-]{36})$/u);
  if (match) chatIds.add(match[1]);
}

async function deleteChat(page, token, chatId) {
  const status = await page.evaluate(async ({ token: authToken, chatId: id }) => {
    const response = await fetch(`/api/v1/chats/${encodeURIComponent(id)}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${authToken}` },
    });
    return response.status;
  }, { token, chatId });
  if (status !== 200) throw new Error(`OpenWebUI chat cleanup failed: HTTP ${status}`);
}

async function submit(page, prompt) {
  const input = page.locator(INPUT_SELECTOR);
  await input.waitFor({ state: "visible", timeout: NAVIGATION_TIMEOUT_MS });
  if (!(await input.isEnabled())) throw new Error("OpenWebUI chat input is disabled");
  await input.fill(prompt, { timeout: NAVIGATION_TIMEOUT_MS });
  await input.press("Enter", { timeout: NAVIGATION_TIMEOUT_MS });
}

async function waitForHostTransition(phase) {
  if (!TRANSITION_SOCKET) return;
  await new Promise((resolve, reject) => {
    const client = net.createConnection(TRANSITION_SOCKET);
    let buffer = "";
    client.setEncoding("utf8");
    client.on("connect", () => client.write(`${phase}\n`));
    client.on("data", (chunk) => {
      buffer += chunk;
      if (buffer.includes("continue\n")) {
        client.end();
        resolve();
      } else if (buffer.includes("abort\n")) {
        client.destroy();
        reject(new Error(`host rejected browser phase: ${phase}`));
      }
    });
    client.on("error", (error) => reject(error));
    client.on("end", () => {
      if (!buffer.includes("continue\n")) reject(new Error(`host closed browser phase: ${phase}`));
    });
  });
}

async function run(browser) {
  const baseUrl = normalizedBaseUrl(BASE_URL);
  const token = strictToken(fs.readFileSync(TOKEN_FILE));
  if (SWITCH_MODEL_ID === MODEL_ID) {
    throw new Error("provider switch model must differ from the uLLM model");
  }
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript((value) => {
    window.localStorage.setItem("token", value);
  }, token);
  const page = await context.newPage();
  const chatIds = new Set();
  const pageErrors = [];
  const requestBodies = [];
  const requestBodyErrors = [];
  let requestBodyError = false;
  page.on("pageerror", (error) => {
    const message = String(error?.message ?? error);
    pageErrors.push(textEvidence(message));
  });
  page.on("request", (request) => {
    if (!requestIsChatCompletion(request)) return;
    if (requestBodies.length >= 4) {
      requestBodyError = true;
      return;
    }
    const raw = request.postData();
    if (raw === null) {
      requestBodyError = true;
      return;
    }
    try {
      requestBodies.push(summarizeRequestBody(raw));
    } catch (error) {
      requestBodyError = true;
      requestBodyErrors.push(String(error?.message ?? error));
    }
  });

  const navigationUrl = new URL("/", baseUrl);
  navigationUrl.searchParams.set("models", MODEL_ID);
  await page.goto(navigationUrl.toString(), {
    waitUntil: "domcontentloaded",
    timeout: NAVIGATION_TIMEOUT_MS,
  });
  await page.waitForFunction(
    (modelLabel) => document.body?.innerText?.includes(modelLabel) === true,
    MODEL_LABEL,
    { timeout: NAVIGATION_TIMEOUT_MS },
  );

  await submit(page, FIRST_PROMPT);
  const firstAssistant = await waitForAnswer(page, FIRST_MARKER);
  rememberChatId(page, chatIds);
  const toggle = firstAssistant.locator(TOGGLE_SELECTOR).last();
  await toggle.waitFor({ state: "visible", timeout: RESPONSE_TIMEOUT_MS });
  const firstText = await visibleAnswerText(firstAssistant);
  if (firstText !== FIRST_MARKER) {
    throw new Error("first answer is not separated from reasoning details");
  }
  await toggle.click();
  const reasoningDetails = firstAssistant.locator("div.mb-1\\.5").last();
  await reasoningDetails.waitFor({ state: "visible", timeout: NAVIGATION_TIMEOUT_MS });
  const expandedText = (await firstAssistant.innerText()).trim();
  const toggleText = (await toggle.innerText()).trim();
  if (expandedText.length <= firstText.length + toggleText.length) {
    throw new Error("reasoning details are empty after expansion");
  }

  await page.reload({ waitUntil: "domcontentloaded", timeout: NAVIGATION_TIMEOUT_MS });
  await waitForAnswer(page, FIRST_MARKER);
  await submit(page, SECOND_PROMPT);
  const secondAssistant = await waitForAnswer(page, SECOND_MARKER);
  const secondText = await visibleAnswerText(secondAssistant);
  if (secondText !== SECOND_MARKER) {
    throw new Error("second answer is not separated from reasoning details");
  }
  if (requestBodies.length < 2) {
    throw new Error(
      `two provider requests were not observed (count=${requestBodies.length}, errors=${requestBodyErrors.join("|")})`,
    );
  }
  const secondRequest = requestBodies[requestBodies.length - 1];
  if (secondRequest.assistant_has_reasoning_content) {
    throw new Error("hidden reasoning was reinserted into the next turn");
  }

  await waitForHostTransition("before-switch");
  const switchUrl = new URL("/", baseUrl);
  switchUrl.searchParams.set("models", SWITCH_MODEL_ID);
  await page.goto(switchUrl.toString(), {
    waitUntil: "domcontentloaded",
    timeout: NAVIGATION_TIMEOUT_MS,
  });
  await page.waitForFunction(
    (modelLabel) => document.body?.innerText?.includes(modelLabel) === true,
    SWITCH_MODEL_LABEL,
    { timeout: NAVIGATION_TIMEOUT_MS },
  );
  await submit(page, SWITCH_PROMPT);
  const switchedAssistant = await waitForAnswer(page, SWITCH_MARKER);
  rememberChatId(page, chatIds);
  const switchedText = await visibleAnswerText(switchedAssistant);
  if (switchedText !== SWITCH_MARKER) {
    throw new Error("provider switch answer is not separated from reasoning details");
  }
  if (requestBodies.length < 3) throw new Error("provider switch request was not observed");
  const switchRequest = requestBodies[requestBodies.length - 1];
  if (switchRequest.model_id_sha256 !== sha256(SWITCH_MODEL_ID)) {
    throw new Error("provider switch request used the wrong model");
  }

  await waitForHostTransition("before-return");
  const switchBackUrl = new URL("/", baseUrl);
  switchBackUrl.searchParams.set("models", MODEL_ID);
  await page.goto(switchBackUrl.toString(), {
    waitUntil: "domcontentloaded",
    timeout: NAVIGATION_TIMEOUT_MS,
  });
  await page.waitForFunction(
    (modelLabel) => document.body?.innerText?.includes(modelLabel) === true,
    MODEL_LABEL,
    { timeout: NAVIGATION_TIMEOUT_MS },
  );
  await submit(page, SWITCH_BACK_PROMPT);
  const switchedBackAssistant = await waitForAnswer(page, SWITCH_BACK_MARKER);
  rememberChatId(page, chatIds);
  const switchedBackText = await visibleAnswerText(switchedBackAssistant);
  if (switchedBackText !== SWITCH_BACK_MARKER) {
    throw new Error("provider return answer is not separated from reasoning details");
  }
  if (requestBodies.length < 4) throw new Error("provider return request was not observed");
  const switchBackRequest = requestBodies[requestBodies.length - 1];
  if (switchBackRequest.model_id_sha256 !== sha256(MODEL_ID)) {
    throw new Error("provider return request used the wrong model");
  }

  if (requestBodyError) throw new Error("provider request body validation failed");
  if (requestBodies.length !== 4) {
    throw new Error("unexpected provider request count");
  }
  if (pageErrors.length > 0) throw new Error("OpenWebUI page errors were observed");
  for (const chatId of chatIds) await deleteChat(page, token, chatId);
  await context.close();
  return {
    schema_version: SUMMARY_SCHEMA,
    model_id_sha256: sha256(MODEL_ID),
    first_answer: textEvidence(firstText),
    expanded_view: textEvidence(expandedText),
    second_answer: textEvidence(secondText),
    provider_switch_performed: true,
    provider_switch_model_id_sha256: sha256(SWITCH_MODEL_ID),
    provider_switch_answer: textEvidence(switchedText),
    provider_return_performed: true,
    provider_return_model_id_sha256: sha256(MODEL_ID),
    provider_return_answer: textEvidence(switchedBackText),
    reasoning_details_expanded: true,
    provider_request_count: requestBodies.length,
    provider_requests: requestBodies,
    hidden_reasoning_reinserted: false,
    page_error_count: pageErrors.length,
    page_error_digests: pageErrors,
  };
}

async function main() {
  const { chromium } = require("playwright");
  const browser = await chromium.launch({ headless: true });
  try {
    console.log(JSON.stringify(await run(browser), null, 2));
  } finally {
    await browser.close();
  }
}

if (require.main === module) {
  main().catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`OpenWebUI reasoning browser smoke failed: ${message}`);
    process.exitCode = 1;
  });
}

module.exports = {
  SUMMARY_SCHEMA,
  assistantHasReasoningContent,
  hasKey,
  normalizedBaseUrl,
  requestIsChatCompletion,
  strictToken,
  summarizeRequestBody,
};
