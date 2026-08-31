export interface IntegratedRuntimeConfig {
  schemaVersion: 'sllm-webui-runtime-v1';
  apiBaseUrl: string;
}

export async function fetchIntegratedRuntimeConfig(
  signal?: AbortSignal,
): Promise<IntegratedRuntimeConfig | null> {
  const response = await fetch('/api/runtime-config', {
    cache: 'no-store',
    signal,
  });
  if (response.status === 404) return null;
  if (!response.ok)
    throw new Error(`Runtime configuration failed: ${response.status}.`);
  const payload = (await response.json()) as {
    schema_version?: unknown;
    integrated?: unknown;
    api_base_url?: unknown;
  };
  if (payload.integrated !== true) return null;
  if (
    payload.schema_version !== 'sllm-webui-runtime-v1' ||
    typeof payload.api_base_url !== 'string'
  )
    throw new Error('Runtime configuration is malformed.');
  const url = new URL(payload.api_base_url);
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.pathname !== '/' ||
    url.search ||
    url.hash
  )
    throw new Error('Runtime API endpoint is invalid.');
  return {
    schemaVersion: 'sllm-webui-runtime-v1',
    apiBaseUrl: url.toString().replace(/\/$/, ''),
  };
}
