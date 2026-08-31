import { SseDecoder, type SseEvent } from './sse.ts';

export interface ApiConfig {
  baseUrl: string;
  userKey: string;
  adminKey: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  reasoning?: string;
}

export interface ServerModel {
  id: string;
  lifecycle?: string;
  residentBytes?: number;
}

export interface ServerProps {
  schema_version?: string;
  state?: string;
  models?: Array<Record<string, unknown>>;
  scheduler?: Record<string, unknown>;
  hardware?: {
    vendor?: string;
    device_index?: number;
    name?: string;
    target?: string;
    memory_bytes?: number;
  } | null;
  features?: Record<string, unknown>;
}

export interface ServerSnapshot {
  health: string;
  ready: string;
  models: ServerModel[];
  props: ServerProps | null;
}

export interface ModelLibraryModel {
  alias: string;
  file_name: string;
  size_bytes: number;
  architecture: string;
  supported_architecture: boolean;
  compatible: boolean;
  reason?: string | null;
  mtp_companion_file_name?: string | null;
  mtp_companion_for?: string | null;
}

export interface ModelLibrarySnapshot {
  schema_version: string;
  selected_path?: string | null;
  models: ModelLibraryModel[];
  error?: string | null;
}

export interface ModelLibraryDirectory {
  name: string;
  path: string;
}

export interface ModelLibraryBrowse {
  schema_version: string;
  current_path: string;
  parent_path?: string | null;
  directories: ModelLibraryDirectory[];
}

export interface HuggingFaceStatus {
  schema_version: string;
  cli_available: boolean;
  cli_version?: string | null;
  auth_state: 'authenticated' | 'unauthenticated' | 'unknown';
  authenticated: boolean;
  username?: string | null;
  active_downloads: number;
}

export interface HuggingFaceModel {
  repo_id: string;
  revision: string;
  downloads: number;
  likes: number;
  gated: boolean;
  private: boolean;
  last_modified?: string | null;
}

export interface HuggingFaceSearch {
  schema_version: string;
  query: string;
  models: HuggingFaceModel[];
}

export interface HuggingFaceGgufFile {
  path: string;
  size_bytes: number;
  sha256?: string | null;
  derived_lock_path?: string | null;
  download_command: string;
}

export interface HuggingFaceFiles {
  schema_version: string;
  repo_id: string;
  revision: string;
  selected_path: string;
  files: HuggingFaceGgufFile[];
}

export type HuggingFaceDownloadState =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed';

export interface HuggingFaceDownloadJob {
  schema_version: string;
  id: string;
  repo_id: string;
  revision: string;
  file_path: string;
  destination: string;
  command: string;
  state: HuggingFaceDownloadState;
  message?: string | null;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  temperature: number;
  topP: number;
  maxTokens: number;
  responseFormat: 'text' | 'json_object';
  reasoning: boolean;
  reasoningBudget: number;
}

function endpoint(config: ApiConfig, path: string): string {
  const url = new URL(config.baseUrl);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('Endpoint must use HTTP or HTTPS.');
  }
  return new URL(path, `${url.toString().replace(/\/$/, '')}/`).toString();
}

function auth(key: string): Record<string, string> {
  return key ? { Authorization: `Bearer ${key}` } : {};
}

async function errorMessage(response: Response): Promise<string> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  try {
    const payload = (await response.json()) as { error?: { message?: string } };
    return payload.error?.message?.slice(0, 400) || fallback;
  } catch {
    return fallback;
  }
}

async function request(config: ApiConfig, path: string, init?: RequestInit) {
  const headers = new Headers(init?.headers);
  if (config.userKey) headers.set('Authorization', `Bearer ${config.userKey}`);
  const response = await fetch(endpoint(config, path), {
    ...init,
    headers,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response;
}

async function readiness(config: ApiConfig): Promise<string> {
  const response = await fetch(endpoint(config, '/readyz'));
  if (response.status !== 200 && response.status !== 503) {
    throw new Error(await errorMessage(response));
  }
  try {
    const payload = (await response.json()) as { status?: string };
    return payload.status ?? (response.ok ? 'ready' : 'not_ready');
  } catch {
    return response.ok ? 'ready' : 'not_ready';
  }
}

export async function fetchServerSnapshot(
  config: ApiConfig,
): Promise<ServerSnapshot> {
  const [health, ready, modelResponse, propsResponse] = await Promise.all([
    request(config, '/healthz').then((value) => value.text()),
    readiness(config),
    request(config, '/v1/models').then((value) => value.json()),
    request(config, '/props').then((value) => value.json()),
  ]);
  const listed = modelResponse as { data?: Array<{ id?: string }> };
  const props = propsResponse as ServerProps;
  const stringValue = (value: unknown) =>
    typeof value === 'string' ? value : '';
  const lifecycle = new Map(
    (props.models ?? []).map((model) => [
      stringValue(model.alias ?? model.id),
      model,
    ]),
  );
  const models = (listed.data ?? [])
    .filter((model): model is { id: string } => Boolean(model.id))
    .map((model) => {
      const detail = lifecycle.get(model.id);
      return {
        id: model.id,
        lifecycle: detail
          ? stringValue(detail.lifecycle ?? detail.state)
          : undefined,
        residentBytes:
          typeof detail?.resident_bytes === 'number'
            ? detail.resident_bytes
            : undefined,
      };
    });
  return { health: health.trim(), ready: ready.trim(), models, props };
}

export async function fetchServerMetrics(config: ApiConfig): Promise<string> {
  return request(config, '/metrics').then((response) => response.text());
}

export async function modelAction(
  config: ApiConfig,
  alias: string,
  action: 'load' | 'unload',
): Promise<void> {
  const response = await fetch(
    endpoint(config, `/admin/models/${encodeURIComponent(alias)}/${action}`),
    { method: 'POST', headers: auth(config.adminKey) },
  );
  if (!response.ok) throw new Error(await errorMessage(response));
}

async function adminRequest(
  config: ApiConfig,
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const headers = new Headers(init?.headers);
  for (const [name, value] of Object.entries(auth(config.adminKey)))
    headers.set(name, value);
  const response = await fetch(endpoint(config, path), { ...init, headers });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response;
}

export async function fetchModelLibrary(
  config: ApiConfig,
): Promise<ModelLibrarySnapshot> {
  return adminRequest(config, '/admin/model-library').then((response) =>
    response.json(),
  );
}

export async function browseModelLibrary(
  config: ApiConfig,
  path?: string,
): Promise<ModelLibraryBrowse> {
  return adminRequest(config, '/admin/model-library/browse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(path ? { path } : {}),
  }).then((response) => response.json());
}

export async function selectModelLibraryFolder(
  config: ApiConfig,
  path: string,
): Promise<ModelLibrarySnapshot> {
  return adminRequest(config, '/admin/model-library/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  }).then((response) => response.json());
}

export async function rescanModelLibrary(
  config: ApiConfig,
): Promise<ModelLibrarySnapshot> {
  return adminRequest(config, '/admin/model-library/rescan', {
    method: 'POST',
  }).then((response) => response.json());
}

export async function fetchHuggingFaceStatus(
  config: ApiConfig,
): Promise<HuggingFaceStatus> {
  return adminRequest(config, '/admin/hugging-face/status').then((response) =>
    response.json(),
  );
}

export async function searchHuggingFaceModels(
  config: ApiConfig,
  query: string,
): Promise<HuggingFaceSearch> {
  return adminRequest(config, '/admin/hugging-face/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  }).then((response) => response.json());
}

export async function fetchHuggingFaceFiles(
  config: ApiConfig,
  repoId: string,
  revision: string,
): Promise<HuggingFaceFiles> {
  return adminRequest(config, '/admin/hugging-face/files', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_id: repoId, revision }),
  }).then((response) => response.json());
}

export async function startHuggingFaceDownload(
  config: ApiConfig,
  repoId: string,
  revision: string,
  file: HuggingFaceGgufFile,
): Promise<HuggingFaceDownloadJob> {
  return adminRequest(config, '/admin/hugging-face/downloads', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      repo_id: repoId,
      revision,
      file_path: file.path,
      ...(file.derived_lock_path
        ? { derived_lock_path: file.derived_lock_path }
        : {}),
    }),
  }).then((response) => response.json());
}

export async function fetchHuggingFaceDownloadJob(
  config: ApiConfig,
  id: string,
): Promise<HuggingFaceDownloadJob> {
  return adminRequest(
    config,
    `/admin/hugging-face/downloads/${encodeURIComponent(id)}`,
  ).then((response) => response.json());
}

function consumeEvent(
  event: SseEvent,
  callbacks: {
    onContent(value: string): void;
    onReasoning(value: string): void;
  },
): boolean {
  if (event.data === '[DONE]') return true;
  let payload: {
    error?: { message?: string };
    choices?: Array<{
      delta?: { content?: string; reasoning_content?: string };
    }>;
  };
  try {
    payload = JSON.parse(event.data) as typeof payload;
  } catch {
    throw new Error('The server returned malformed SSE JSON.');
  }
  if (payload.error)
    throw new Error(payload.error.message || 'The stream failed.');
  const delta = payload.choices?.[0]?.delta;
  if (delta?.content) callbacks.onContent(delta.content);
  if (delta?.reasoning_content) callbacks.onReasoning(delta.reasoning_content);
  return false;
}

export async function streamChatCompletion(
  config: ApiConfig,
  requestBody: ChatRequest,
  signal: AbortSignal,
  callbacks: {
    onContent(value: string): void;
    onReasoning(value: string): void;
  },
): Promise<void> {
  const body = {
    model: requestBody.model,
    messages: requestBody.messages.map(({ role, content, reasoning }) => ({
      role,
      content,
      ...(reasoning ? { reasoning_content: reasoning } : {}),
    })),
    stream: true,
    temperature: requestBody.temperature,
    top_p: requestBody.topP,
    max_completion_tokens: requestBody.maxTokens,
    ...(requestBody.responseFormat === 'json_object'
      ? { response_format: { type: 'json_object' } }
      : {}),
    ...(requestBody.reasoning
      ? {
          sllm: {
            thinking: 'enabled',
            separate_reasoning: true,
            max_reasoning_tokens: requestBody.reasoningBudget,
          },
        }
      : {}),
  };
  const response = await request(config, '/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.body)
    throw new Error('The server returned no response stream.');

  const reader = response.body.getReader();
  const text = new TextDecoder();
  const decoder = new SseDecoder();
  let done = false;
  while (!done) {
    const chunk = await reader.read();
    for (const event of decoder.push(
      text.decode(chunk.value, { stream: !chunk.done }),
    )) {
      done = consumeEvent(event, callbacks) || done;
    }
    if (chunk.done) break;
  }
  for (const event of decoder.finish())
    done = consumeEvent(event, callbacks) || done;
  if (!done) throw new Error('The stream ended before [DONE].');
}
