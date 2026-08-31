import assert from 'node:assert/strict';
import test from 'node:test';

import { fetchIntegratedRuntimeConfig } from '../lib/runtime-config.ts';

function mockRuntimeConfig(payload: unknown, status = 200) {
  const original = globalThis.fetch;
  globalThis.fetch = (input) => {
    assert.equal(input, '/api/runtime-config');
    return Promise.resolve(Response.json(payload, { status }));
  };
  return () => {
    globalThis.fetch = original;
  };
}

void test('accepts the credential-free API endpoint from integrated startup', async () => {
  const restore = mockRuntimeConfig({
    schema_version: 'sllm-webui-runtime-v1',
    integrated: true,
    api_base_url: 'http://127.0.0.1:8080',
  });
  try {
    assert.deepEqual(await fetchIntegratedRuntimeConfig(), {
      schemaVersion: 'sllm-webui-runtime-v1',
      apiBaseUrl: 'http://127.0.0.1:8080',
    });
  } finally {
    restore();
  }
});

void test('keeps standalone and hosted builds in safe demo mode', async () => {
  const restore = mockRuntimeConfig({
    schema_version: 'sllm-webui-runtime-v1',
    integrated: false,
    api_base_url: null,
  });
  try {
    assert.equal(await fetchIntegratedRuntimeConfig(), null);
  } finally {
    restore();
  }
});

void test('rejects credentials and non-HTTP endpoints in runtime configuration', async () => {
  for (const apiBaseUrl of [
    'file:///tmp/socket',
    'http://token@127.0.0.1:8080',
    'http://127.0.0.1:8080/path',
  ]) {
    const restore = mockRuntimeConfig({
      schema_version: 'sllm-webui-runtime-v1',
      integrated: true,
      api_base_url: apiBaseUrl,
    });
    try {
      await assert.rejects(fetchIntegratedRuntimeConfig(), /invalid/);
    } finally {
      restore();
    }
  }
});
