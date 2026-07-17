# Pre-SQ Runtime TPS T0

## 前回の要点

- `docs/plans/pre-sq-runtime-tps-plan-v0.1.md` が追加済み。
- Goalはsq format策定の入力になる実推論TPS測定基盤を作ること。
- まずT0でaccepted package、device IDs、benchmark schema、出力規約を固定する。

## 今回の変更点

- T0 artifact indexを追加した。
  - `uLLM-project/benchmarks/results/2026-07-06/engine/pre-sq-runtime-artifact-index.md`
- pre-sq runtime benchmark exampleを追加した。
  - `uLLM-project/benchmarks/results/2026-07-06/engine/pre-sq-runtime-benchmark-example.json`
- `docs/specs/inference-benchmark-result-v0.1.md` にpre-sq向けoptional fieldsを追記した。
  - prefill/decode/total wall time
  - TTFT/TPOT
  - KV cache bytes/blocks
  - correctness sanity object

## Evidence

- `target/debug/ullm-engine inspect-devices`
  - CPU fallback: device `0`
  - V620/RDNA2: device `1`
  - R9700/RDNA4: device `2`
  - V620/RDNA2: device `3`
- accepted package exists:
  - `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d`
- baseline package exists:
  - `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- JSON example validation:
  - `jq -e '.schema_version == "inference-benchmark-result-v0.1" and .engine.name == "uLLM" and .workload.prompt_tokens == 512 and .workload.generated_tokens == 256' ...`
- `git diff --check` passed.

## 次の行動

T1へ進む。最初はtoken IDsからembedding/final norm/lm_headを読める境界を作り、`package-token-ids-logits-smoke` の最小形を実装する。
