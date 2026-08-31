import assert from 'node:assert/strict';
import test from 'node:test';

import {
  browseModelLibrary,
  fetchHuggingFaceDownloadJob,
  fetchHuggingFaceFiles,
  fetchHuggingFaceStatus,
  fetchServerSnapshot,
  searchHuggingFaceModels,
  selectModelLibraryFolder,
  startHuggingFaceDownload,
  streamChatCompletion,
  type ApiConfig,
} from '../lib/sllm-api.ts';

const config: ApiConfig = {
  baseUrl: 'https://sllm.test',
  userKey: 'user-key',
  adminKey: '',
};

function mockFetch(
  handler: (request: Request) => Response | Promise<Response>,
) {
  const original = globalThis.fetch;
  globalThis.fetch = (input, init) =>
    Promise.resolve(handler(new Request(input, init)));
  return () => {
    globalThis.fetch = original;
  };
}

void test('connects to a healthy dynamic server that is not ready yet', async () => {
  const restore = mockFetch((request) => {
    const path = new URL(request.url).pathname;
    if (path === '/healthz') return new Response('{"status":"ok"}');
    if (path === '/readyz') {
      return Response.json({ status: 'not_ready' }, { status: 503 });
    }
    if (path === '/v1/models') {
      return Response.json({ data: [{ id: 'qwen' }] });
    }
    if (path === '/props') {
      return Response.json({
        state: 'unloaded',
        models: [{ alias: 'qwen', lifecycle: 'configured' }],
      });
    }
    return new Response(null, { status: 404 });
  });
  try {
    const snapshot = await fetchServerSnapshot(config);
    assert.equal(snapshot.ready, 'not_ready');
    assert.deepEqual(snapshot.models, [
      { id: 'qwen', lifecycle: 'configured', residentBytes: undefined },
    ]);
  } finally {
    restore();
  }
});

void test('uses the server-side model folder API without exposing browser files', async () => {
  const requests: Request[] = [];
  const restore = mockFetch(async (request) => {
    requests.push(request);
    if (request.url.endsWith('/admin/model-library/browse')) {
      return Response.json({
        schema_version: 'sllm-model-library-browse-v1',
        current_path: '/models',
        parent_path: '/',
        directories: [{ name: 'qwen', path: '/models/qwen' }],
      });
    }
    return Response.json({
      schema_version: 'sllm-model-library-v1',
      selected_path: '/models/qwen',
      models: [],
    });
  });
  try {
    const listing = await browseModelLibrary(config, '/models');
    assert.equal(listing.directories[0]?.name, 'qwen');
    const snapshot = await selectModelLibraryFolder(config, '/models/qwen');
    assert.equal(snapshot.selected_path, '/models/qwen');
    assert.equal(requests[0]?.method, 'POST');
    assert.deepEqual(await requests[0]?.json(), { path: '/models' });
    assert.deepEqual(await requests[1]?.json(), { path: '/models/qwen' });
  } finally {
    restore();
  }
});

void test('uses structured Hugging Face admin requests without sending commands or destinations', async () => {
  const requests: Request[] = [];
  const restore = mockFetch(async (request) => {
    requests.push(request);
    const path = new URL(request.url).pathname;
    if (path.endsWith('/status')) {
      return Response.json({
        schema_version: 'sllm-hugging-face-status-v1',
        cli_available: true,
        auth_state: 'unauthenticated',
        authenticated: false,
        active_downloads: 0,
      });
    }
    if (path.endsWith('/search')) {
      return Response.json({
        schema_version: 'sllm-hugging-face-search-v1',
        query: 'Qwen GGUF',
        models: [
          {
            repo_id: 'owner/Qwen-GGUF',
            revision: '0123456789abcdef0123456789abcdef01234567',
            downloads: 12,
            likes: 3,
            gated: false,
            private: false,
          },
        ],
      });
    }
    if (path.endsWith('/files')) {
      return Response.json({
        schema_version: 'sllm-hugging-face-files-v1',
        repo_id: 'owner/Qwen-GGUF',
        revision: '0123456789abcdef0123456789abcdef01234567',
        selected_path: '/srv/Model Store',
        files: [
          {
            path: 'Qwen BF16.gguf',
            size_bytes: 100,
            derived_lock_path: 'Qwen BF16.derived-lock.json',
            download_command: 'hf download ...',
          },
        ],
      });
    }
    if (path.endsWith('/downloads')) {
      return Response.json(
        {
          schema_version: 'sllm-hugging-face-download-v1',
          id: 'hf-download-1',
          repo_id: 'owner/Qwen-GGUF',
          revision: '0123456789abcdef0123456789abcdef01234567',
          file_path: 'Qwen BF16.gguf',
          destination: '/srv/Model Store',
          command: 'hf download ...',
          state: 'queued',
        },
        { status: 202 },
      );
    }
    return Response.json({
      schema_version: 'sllm-hugging-face-download-v1',
      id: 'hf-download-1',
      repo_id: 'owner/Qwen-GGUF',
      revision: '0123456789abcdef0123456789abcdef01234567',
      file_path: 'Qwen BF16.gguf',
      destination: '/srv/Model Store',
      command: 'hf download ...',
      state: 'completed',
    });
  });
  const adminConfig = { ...config, adminKey: 'admin-secret' };
  try {
    const status = await fetchHuggingFaceStatus(adminConfig);
    assert.equal(status.auth_state, 'unauthenticated');
    const search = await searchHuggingFaceModels(adminConfig, 'Qwen GGUF');
    const model = search.models[0]!;
    const files = await fetchHuggingFaceFiles(
      adminConfig,
      model.repo_id,
      model.revision,
    );
    const job = await startHuggingFaceDownload(
      adminConfig,
      model.repo_id,
      model.revision,
      files.files[0]!,
    );
    const completed = await fetchHuggingFaceDownloadJob(adminConfig, job.id);
    assert.equal(completed.state, 'completed');
    for (const request of requests)
      assert.equal(request.headers.get('Authorization'), 'Bearer admin-secret');
    assert.deepEqual(await requests[1]!.json(), { query: 'Qwen GGUF' });
    assert.deepEqual(await requests[2]!.json(), {
      repo_id: 'owner/Qwen-GGUF',
      revision: '0123456789abcdef0123456789abcdef01234567',
    });
    const downloadBody = (await requests[3]!.json()) as Record<string, unknown>;
    assert.deepEqual(downloadBody, {
      repo_id: 'owner/Qwen-GGUF',
      revision: '0123456789abcdef0123456789abcdef01234567',
      file_path: 'Qwen BF16.gguf',
      derived_lock_path: 'Qwen BF16.derived-lock.json',
    });
    assert.equal('command' in downloadBody, false);
    assert.equal('destination' in downloadBody, false);
  } finally {
    restore();
  }
});

void test('requires the terminal DONE marker for chat streams', async () => {
  let requestBody: Record<string, unknown> | undefined;
  const restore = mockFetch(async (request) => {
    requestBody = (await request.json()) as Record<string, unknown>;
    return new Response(
      'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
      {
        headers: { 'Content-Type': 'text/event-stream' },
      },
    );
  });
  let content = '';
  try {
    await assert.rejects(
      streamChatCompletion(
        config,
        {
          model: 'qwen',
          messages: [{ role: 'user', content: 'hello' }],
          temperature: 0.2,
          topP: 0.9,
          maxTokens: 8,
          responseFormat: 'text',
          reasoning: false,
          reasoningBudget: 1,
        },
        new AbortController().signal,
        {
          onContent: (value) => {
            content += value;
          },
          onReasoning: () => undefined,
        },
      ),
      /before \[DONE\]/,
    );
    assert.equal(content, 'hello');
    assert.equal(requestBody?.max_completion_tokens, 8);
    assert.equal('max_tokens' in (requestBody ?? {}), false);
  } finally {
    restore();
  }
});
