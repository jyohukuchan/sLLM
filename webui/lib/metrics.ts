export interface ModelMetricSnapshot {
  promptTokens: number;
  completionTokens: number;
  ttftSeconds: number;
  e2eSeconds: number;
  successes: number;
}

interface MetricSample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

function unescapeLabel(value: string): string {
  return value
    .replaceAll('\\n', '\n')
    .replaceAll('\\"', '"')
    .replaceAll('\\\\', '\\');
}

function parseLabels(source: string): Record<string, string> {
  const labels: Record<string, string> = {};
  const pattern = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"/g;
  for (const match of source.matchAll(pattern))
    labels[match[1]] = unescapeLabel(match[2]);
  return labels;
}

export function parsePrometheus(text: string): MetricSample[] {
  const samples: MetricSample[] = [];
  for (const line of text.split('\n')) {
    if (!line || line.startsWith('#')) continue;
    const match = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)$/.exec(
      line.trim(),
    );
    if (!match) continue;
    const value = Number(match[3]);
    if (!Number.isFinite(value)) continue;
    samples.push({
      name: match[1],
      labels: parseLabels(match[2] ?? ''),
      value,
    });
  }
  return samples;
}

function select(
  samples: MetricSample[],
  name: string,
  labels: Record<string, string>,
): number {
  return samples
    .filter(
      (sample) =>
        sample.name === name &&
        Object.entries(labels).every(
          ([key, value]) => sample.labels[key] === value,
        ),
    )
    .reduce((total, sample) => total + sample.value, 0);
}

export function modelMetricSnapshot(
  text: string,
  model: string,
): ModelMetricSnapshot {
  const samples = parsePrometheus(text);
  const common = { model, stream: 'true' };
  return {
    promptTokens: select(samples, 'sllm_tokens_total', {
      ...common,
      direction: 'prompt',
    }),
    completionTokens: select(samples, 'sllm_tokens_total', {
      ...common,
      direction: 'completion',
    }),
    ttftSeconds: select(samples, 'sllm_request_ttft_seconds_sum', common),
    e2eSeconds: select(samples, 'sllm_request_e2e_seconds_sum', common),
    successes: select(samples, 'sllm_requests_total', {
      ...common,
      outcome: 'success',
    }),
  };
}

export function subtractMetrics(
  after: ModelMetricSnapshot,
  before: ModelMetricSnapshot,
): ModelMetricSnapshot {
  return {
    promptTokens: after.promptTokens - before.promptTokens,
    completionTokens: after.completionTokens - before.completionTokens,
    ttftSeconds: after.ttftSeconds - before.ttftSeconds,
    e2eSeconds: after.e2eSeconds - before.e2eSeconds,
    successes: after.successes - before.successes,
  };
}
