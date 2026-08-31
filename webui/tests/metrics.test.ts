import assert from 'node:assert/strict';
import test from 'node:test';

import { modelMetricSnapshot, subtractMetrics } from '../lib/metrics.ts';

const before = `
sllm_tokens_total{model="qwen",stream="true",direction="prompt"} 100
sllm_tokens_total{model="qwen",stream="true",direction="completion"} 20
sllm_request_ttft_seconds_sum{model="qwen",stream="true"} 1.5
sllm_request_e2e_seconds_sum{model="qwen",stream="true"} 3.5
sllm_requests_total{model="qwen",stream="true",outcome="success"} 2
`;

void test('extracts and subtracts one model stream metric window', () => {
  const first = modelMetricSnapshot(before, 'qwen');
  const second = modelMetricSnapshot(
    before
      .replace(' 100', ' 164')
      .replace(' 20', ' 52')
      .replace(' 1.5', ' 1.7')
      .replace(' 3.5', ' 3.95')
      .replace(' 2\n', ' 3\n'),
    'qwen',
  );
  const delta = subtractMetrics(second, first);
  assert.equal(delta.promptTokens, 64);
  assert.equal(delta.completionTokens, 32);
  assert.ok(Math.abs(delta.ttftSeconds - 0.2) < Number.EPSILON * 2);
  assert.ok(Math.abs(delta.e2eSeconds - 0.45) < Number.EPSILON * 2);
  assert.equal(delta.successes, 1);
});
