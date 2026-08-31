const RUNTIME_SCHEMA = 'sllm-webui-runtime-v1';

function integratedApiBaseUrl(): string | null {
  const value = process.env.SLLM_API_BASE_URL?.trim();
  if (!value) return null;
  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) return null;
    if (
      url.username ||
      url.password ||
      url.pathname !== '/' ||
      url.search ||
      url.hash
    )
      return null;
    return url.toString().replace(/\/$/, '');
  } catch {
    return null;
  }
}

export function GET(): Response {
  const apiBaseUrl = integratedApiBaseUrl();
  return Response.json(
    {
      schema_version: RUNTIME_SCHEMA,
      integrated: apiBaseUrl !== null,
      api_base_url: apiBaseUrl,
    },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}
