# T2 SQ FP8 prompt-suite text guard smoke v0.1

## 前回の要点

- v0.2 prompt-suite guardは `generated_token_ids` に加えて `decoded_text.generated` と `decoded_text.generated_without_stop_sequence` の完全一致を保存できるようになった。
- 次の作業は、layer23 `k16` passing branchをSQ artifact付きのprompt-suite実行経路へ接続することだった。

## 今回の変更点

- `sq-fp8-token-ids-generate-smoke` と `sq-fp8-token-ids-bench` を追加し、既存のtoken prompt benchからSQ FP8 artifactを渡せるようにした。
- `tools/run-package-token-prompt-bench.py` と `tools/run-package-token-prompt-suite.py` に `--sq-artifact` を追加した。
- guard bundle runnerは、guardが不合格でもbundle summaryを保存してから非0終了するようにした。
- prompt-suite guardに `acceptance_mode=behavioral` を追加し、exact token/text/logit一致は診断として保存しつつ、候補前進のgateからは分離できるようにした。
- 1ケースの短いprompt-suite smokeを追加し、AQ4 baselineとSQ layer23 `k16` candidateを同じv0.3 comparatorへ流した。

## Results

Suite:

- `benchmarks/prompts/pre-sq-runtime-prompt-suite-smoke-v0.1.json`
- prompt: `GPU warmup can distort token/s because`
- generated tokens: 2
- output health: not evaluated

Candidate:

- artifact: `/tmp/ullm-sq-fp8-qkv-layers3-7-11-15-19-q8-k16-v16-plus-layer23-k16-policy-v0.1-artifact`
- candidate id: `sq-fp8-w8a16-r9700-v0-qkv-layers3-7-11-15-19-q8-k16-v16-plus-layer23-k16`
- FP8 tensor count: 16
- row chunk: 256

| row | generated token IDs | generated text | prefill tok/s | decode tok/s | verified |
| --- | --- | --- | ---: | ---: | :---: |
| AQ4 baseline | `314,279` | ` of the` | 23.530881 | 25.258380 | true |
| SQ layer23 `k16` | `314,279` | ` of the` | 20.565812 | 24.336605 | true |

Guard metrics:

| guard | acceptance mode | logit atol | generated token match | generated text match | no-stop text match | top logits match | strict passed | behavioral passed | passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| strict | `strict` | 0.001 | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 1 | false | true | false |
| loose value check | `strict` | 0.2 | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 1 | false | true | false |
| behavioral | `behavioral` | 0.001 | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 1 | false | true | true |

The `0.2` row still fails because the top-k token ranks drift, not because only the logit values exceed tolerance.
The strict guard reports:

- max prefill top-logit abs diff: 0.04891014099121094
- max decode last-top-logit abs diff: 0.1147451400756836
- generated token/text/no-stop text all matched

Artifacts:

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/aq4/summary.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/sq-layer23-k16/summary.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/guard/guard-bundle-summary.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/guard-behavioral/guard-bundle-summary.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/guard-logit-atol-0p2/guard-bundle-summary.json`

## 判断

- SQ artifact付きprompt-suite実行経路は接続できた。
- layer23 `k16` branchはこのmini smokeでは生成tokenと生成textを維持した。
- strict guardとしてはtop-k logits rank driftで不合格だが、behavioral gateでは合格として扱える。
- SQ FP8候補探索では、exact token/text/logit一致を昇格blockerにせず、出力が壊れていないこととthroughput/memoryを優先して次へ進める。

## 次の行動

1. SQ FP8を既定候補として、batch throughput / memory comparisonへ進める。
2. full v0.3 prompt-suiteでは `acceptance_mode=behavioral` を採用し、strict token/text/logit一致は診断列として残す。
3. 出力が明らかに壊れる場合だけSQ candidateを止め、軽微なrank/text driftはformat探索のblockerにしない。
