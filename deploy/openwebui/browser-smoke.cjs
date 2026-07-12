#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const baseUrl = process.env.OPENWEBUI_URL || "http://192.168.0.66:3000";
const tokenFile = process.env.OPENWEBUI_TOKEN_FILE || "/run/secrets/openwebui-token";
const screenshotPath =
  process.env.OPENWEBUI_SCREENSHOT || "/output/openwebui-browser-smoke.png";
const token = fs.readFileSync(tokenFile, "utf8").trim();

if (!token) {
  throw new Error("OpenWebUI test token is empty");
}

async function runSmoke(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript((value) => {
    window.localStorage.setItem("token", value);
  }, token);

  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const query = new URLSearchParams({
    "temporary-chat": "true",
    models: "ullm-qwen3-14b-sq8",
    q: "Reply exactly BROWSER_OK.",
    submit: "true",
  });
  await page.goto(`${baseUrl}/?${query}`, {
    waitUntil: "domcontentloaded",
    timeout: 60_000,
  });
  try {
    await page
      .locator(".chat-assistant", { hasText: "BROWSER_OK" })
      .last()
      .waitFor({ state: "visible", timeout: 90_000 });
  } catch (error) {
    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.error(
      JSON.stringify({
        assistantCount: await page.locator(".chat-assistant").count(),
        loginVisible: await page.locator('input[type="password"]').isVisible(),
        title: await page.title(),
        url: page.url(),
      }),
    );
    throw error;
  }

  const assistants = page.locator(".chat-assistant");
  const assistantText = (await assistants.last().innerText()).trim();
  const infoButton = page.locator('button[id^="info-"]').last();
  await infoButton.waitFor({ state: "visible", timeout: 10_000 });
  await infoButton.hover();
  const metricFields = [
    "predicted_per_second",
    "finish_reason",
    "termination_reason",
  ];
  await page.waitForFunction(
    (fields) => fields.every((field) => document.body.innerText.includes(field)),
    metricFields,
    { timeout: 10_000 },
  );
  const bodyText = await page.locator("body").innerText();
  const modelVisible = bodyText.includes("uLLM Qwen3 14B SQ8");
  const browserOk = assistantText.includes("BROWSER_OK");
  const metricsVisible = metricFields.every((field) => bodyText.includes(field));
  if (!browserOk || !modelVisible || !metricsVisible) {
    throw new Error(
      `browser smoke mismatch: browserOk=${browserOk} modelVisible=${modelVisible} metricsVisible=${metricsVisible}`,
    );
  }

  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(
    JSON.stringify({
      assistantText,
      browserOk,
      metricFields,
      metricsVisible,
      modelVisible,
      pageErrors,
      screenshotPath,
      title: await page.title(),
      url: page.url(),
    }),
  );
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  try {
    await runSmoke(browser);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
