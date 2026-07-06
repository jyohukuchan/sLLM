# AQ4 RDNA Prototype Status 2026-07-06

## 前回の要点

- R9700/RDNA4では、AQ4 prototype pathがsynthetic `prompt=16/generated=512` で `18.957 tok/s` まで到達した。
- 実text promptでも、controlled v0.3 suiteでR9700/RDNA4はmean decode `19.796 tok/s`、V620/RDNA2はmean decode `15.434 tok/s` を記録した。
- v0.3 suiteでは、品質評価対象6件がR9700/V620の両方でwarningなし、timing probe 1件は品質評価から分離した。

## 今回の変更点

- 現時点で外に出せるAQ4 prototype claimと、まだ出せないclaimを分けた。
- SQ format設計へ進む前に残すべき測定・正しさ・制約を整理した。
- 発表時に必要なartifactと、次のgateを1ページにまとめた。

## 次の行動

1. SQ format v0.1では、compact resident storageとbounded materialized working setを最優先にする。
2. 発表用には、このstatus briefとv0.3 suite summaryを根拠にして、single-request prototypeの範囲に限定して説明する。
3. product-quality claimへ進む前に、final logitsまたはgenerated-token reference guardを追加する。

## Current Claim

The current uLLM AQ4 prototype can be described as:

> A single-request Qwen3.5-9B AQ4 runtime prototype that runs on local RDNA4 and RDNA2 GPUs,
> reaches roughly `20 tok/s` on R9700/RDNA4 and `15 tok/s` on V620/RDNA2 under a controlled
> text-prompt suite, and does not show obvious output collapse in that suite.

This claim is intentionally narrow. It does not include tensor parallelism, batching, server API,
sampling policy, product-quality tokenizer integration, or SQ format support.

## Primary Evidence

Package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d
```

Prompt suite:

```text
benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.3.json
```

Summary artifacts:

```text
benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v0.3/summary.json
benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-v620-v0.3/summary.json
```

| target | device | mean decode tok/s | min decode tok/s | max decode tok/s | mean prefill tok/s | verified all | output ok | output warn | output not evaluated |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| R9700/RDNA4 | `2` | 19.796 | 18.500 | 20.166 | 19.850 | true | 6 | 0 | 1 |
| V620/RDNA2 | `1` | 15.434 | 14.553 | 15.668 | 16.503 | true | 6 | 0 | 1 |

Quality-scored cases only:

| target | decode tok/s range | output-health result |
| --- | ---: | --- |
| R9700/RDNA4 | 19.938-20.166 | 6 ok, 0 warn |
| V620/RDNA2 | 15.432-15.668 | 6 ok, 0 warn |

The seventh v0.3 case is a repeated 256-token prefill timing probe. It is retained for throughput
pressure but marked `not_evaluated` for output health because prompt echo is expected there.

## What Improved

- The original long `512/256` path was only about `0.14 tok/s`; that result was a runtime-path
  artifact rather than an AQ format wall.
- GPU-resident lm_head, direct AQ4 matvec, fused decode steps, recurrent linear-attention fast path,
  self-attention output-only step, and paged-attention score reuse moved R9700 into the expected
  `15-20 tok/s` range.
- Task-aware stop sequences and controlled v0.3 prompts made output observation less noisy without
  hiding the timing probe.
- V620/RDNA2 now has the same controlled prompt-suite evidence as R9700/RDNA4, at lower but still
  usable single-request decode speed.

## Known Limits

| limit | current status |
| --- | --- |
| SQ format | not implemented yet |
| tensor parallel | out of scope for current prototype |
| batching / continuous batching | out of scope for current prototype |
| server API | not implemented |
| sampling | greedy only in the measured suite |
| tokenizer integration | handled by Python wrapper, not native runtime API |
| correctness guard | hidden-state golden prefix guard exists; final logits/generated-token reference is still missing |
| BF16 baseline | deferred because current package/runtime cannot express a full decoder BF16 baseline cleanly |
| memory residency | current path still uses large resident/runtime buffers; SQ must address compact residency |
| model scope | Qwen3.5-9B package path, not a broad model zoo claim |

## SQ Design Consequence

The next format work should not start by chasing another generic quantization comparison. The useful
SQ baseline is:

1. current AQ4 prototype with v0.3 suite on R9700 and V620;
2. SQ compact-resident candidate with the same v0.3 suite;
3. the same short reference guard plus an added logits or generated-token guard.

The SQ run record should include at least:

- compact resident bytes;
- materialized working-set bytes;
- materialization time;
- prefill tok/s;
- decode tok/s;
- stop policy;
- output-health summary;
- correctness guard result.

## Publication Wording

Acceptable wording:

- "uLLM currently has a local AQ4 prototype path running on RDNA4 and RDNA2."
- "On a controlled Qwen3.5-9B prompt suite, R9700 reaches about `20 tok/s` and V620 reaches about
  `15 tok/s` for single-request greedy decode."
- "This is a prototype measurement, not a server benchmark."

Avoid wording:

- "uLLM supports production inference."
- "SQ format is implemented."
- "The engine supports batching or tensor parallelism."
- "The output quality is fully validated."
- "This is a model-general performance number."
