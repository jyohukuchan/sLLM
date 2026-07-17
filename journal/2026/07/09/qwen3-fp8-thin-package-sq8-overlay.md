# Qwen3 FP8 thin package SQ8 overlay

## 前回の要点

- SQ8_0とvLLM FP8の同一モデル比較には、Qwen3-14B-FP8のuLLM側実行行が必要。
- Qwen3 tensor namespace差はruntime lookup側で吸収済み。
- Qwen3-14B-FP8の `F8_E4M3` weightと `*.weight_scale_inv` はAQ4再量子化対象ではなく、SQ8_0として扱う必要がある。

## 今回の変更点

- `ullm-quant` に direct/merge package用の passthrough filter を追加した。
  - `--convert-passthrough-dtype`
  - `--convert-passthrough-exclude-suffix`
  - `--merge-passthrough-dtype`
  - `--merge-passthrough-exclude-suffix`
- Qwen3-14B-FP8 sourceから、BF16-onlyの薄いpackage shellを生成した。
  - `/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d`
  - `quantized_tensors=0`, `passthrough_tensors=163`, `codebooks=0`
  - package sizeは約 `2.9G`
- Qwen3-14B-FP8 layer0用SQ8_0 sidecar artifactを生成した。
  - `/tmp/ullm-qwen3-14b-fp8-layer0-sq8-artifact`
  - `fp8_tensor_count=7`, `passthrough_tensor_count=716`
  - artifact sizeは約 `316M`
- `ullm-engine` の layer kind検出を、thin packageの passthrough `q_norm/k_norm` からも自己注意層を推定できるようにした。
- 実packageで `package-layer-kind-inventory-smoke ... manifest-all` を実行し、40層すべて `self_attention` として検出できることを確認した。
- 薄いpackage + layer0 SQ8_0 artifactで `sq-fp8-token-ids-logits-smoke` が `verified=true` まで到達した。

## Follow-up: 40-layer same-model M10 rows

## 前回の要点

- layer0で、Qwen3-14B-FP8 thin package + SQ8_0 sidecar overlayの接続は確認済み。
- M10のvLLM + FP8比較には、Qwen3-14B-FP8のuLLM側40層行が必要。
- Qwen3-14B-FP8 configは `head_dim=128`、`rope_theta=1000000`。

## 今回の変更点

- full 40-layer SQ8_0 artifactを生成した。
  - `/tmp/ullm-qwen3-14b-fp8-full-sq8-artifact`
  - `fp8_tensor_count=281`, `passthrough_tensor_count=442`
  - sizeは約 `14G`
- 40-layer `manifest-all` の短い接続確認をR9700で通した。
  - `device_index=2`
  - `sq_execution_mode=direct_fp8_dequant_matvec`
  - `sq_projection_boundary=single+triple`
  - `verified=true`
- `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl` に、Qwen3-14B-FP8同一モデルのuLLM SQ8_0行を追加した。
  - preliminary smoke: `ullm-r9700-qwen3-14b-fp8-sq8-smoke-pp16-tg8-b1`
  - preliminary representative: `ullm-r9700-qwen3-14b-fp8-sq8-rep-pp512-tg128-b1`
  - config-aligned smoke: `ullm-r9700-qwen3-14b-fp8-sq8-smoke-pp16-tg8-b1-rope128-theta1e6`
  - config-aligned representative: `ullm-r9700-qwen3-14b-fp8-sq8-rep-pp512-tg128-b1-rope128-theta1e6`
- config-aligned smokeは `prefill=3.013316 tok/s`, `decode=3.057004 tok/s`, consumed VRAM `13763940352` bytes。
- config-aligned representativeは `prefill=2.909147 tok/s`, `decode=2.774043 tok/s`, consumed VRAM `14242410496` bytes。

## 次の行動

- Qwen3-14B-FP8同一モデル行へprompt guard bundleまたは同等のbehavioral guardを添付する。
- 現在のuLLM行はtoken-id model-loop / final logits込み / `prefill_real_batch=false` / `decode_real_batch=false` なので、vLLM serving benchmarkとの最終性能比較にはreal-batchまたはserver-style uLLM pathが必要。
- preliminary rope32/theta1e7行は接続履歴として残し、比較本文ではconfig-aligned行を使う。

## 次の行動

- layer0で通った経路を40層SQ8_0 artifactへ拡張する。
- `manifest-all` のQwen3-14B-FP8 uLLM rowを取得し、M10のvLLM + FP8比較表へ同一モデル行として追加する。
- その後、`pp16/tg8/b1` と `pp512/tg128/b1` をvLLM側と同条件で再測定し、同一モデルthroughput比較として扱えるか判定する。

## Follow-up: prompt-suite smoke guard

## 前回の要点

- Qwen3-14B-FP8のconfig-aligned uLLM SQ8_0行は、smokeとrepresentativeの両方が `status=ok` / `verified=true`。
- ただし、vLLM + FP8との最終比較には、同一モデル行にprompt guard bundleまたは同等のbehavioral guardを添付する必要があった。

## 今回の変更点

- Qwen3-14B-FP8 thin package + full SQ8_0 sidecar artifactで、既存の短いprompt-suite smokeを実行した。
  - output:
    `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/summary.json`
  - `verified_all=true`
  - `output_not_evaluated_count=1`
  - generated previewは `准准`
- 同じsummaryをreference/candidateにしたself-behavioral guard bundleを作成した。
  - output:
    `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/guard-self-behavioral/guard-bundle-summary.json`
  - `passed=true`, `acceptance_mode=behavioral`, `strict_passed=true`, `behavioral_passed=true`
- `results.jsonl` のconfig-aligned smoke / representative行へ、このguard bundleを添付した。
- 計画文書では、M10のvLLM + FP8比較をSQ8_0実装計画の後半フェーズとして明記した。

## 次の行動

- 非自己比較のbehavioral guard、またはoutput-healthを評価するQwen3-14B-FP8用prompt suiteを追加する。
- real-batchまたはserver-style uLLM pathを作って、vLLM serving benchmarkと同じ比較クラスの行を取る。

## Follow-up: backend dispatch flexible selectors

## 前回の要点

- SQ8_0計画では、C++ kernel familyをGPU architecture、GPU name、model architectureに応じて選べるようにする必要がある。
- 既存の `backend_dispatch` は `format_id` / `model_arch` / `gpu_arch` / `gpu_name` の任意条件を持っていたが、任意条件は厳密文字列一致だった。

## 今回の変更点

- 実装用subagent `gpt-5.3-codex-spark` に `backend_dispatch.rs` の小さな実装を任せた。
- `backend_dispatch` の任意selectorを柔軟化した。
  - ASCII大文字小文字差を吸収する。
  - `_`、スペース、`-` などの区切り差を吸収する。
  - implementation側の末尾 `*` をprefix matchとして扱う。
  - 完全一致はprefix matchより高く、prefix matchは広い `*` より高く選択される。
- 追加テスト:
  - GPU名の表記ゆれ吸収
  - `Qwen3*` が `Qwen3.5-9B` に一致すること
  - 完全一致のmodel_archがprefix selectorより優先されること

## 次の行動

- SQ8_0 projection dispatch requestへ実runtime側のmodel architectureを渡す。
- 実際に非direct kernel familyを追加するときは、今回のselectorで `model_arch=Qwen3* && gpu_arch=RDNA4 && gpu_name=...` のような実装候補を追加する。

## Follow-up: SQ8_0 projection model_arch request

## 前回の要点

- `backend_dispatch` はmodel/GPU selectorの表記ゆれ吸収とprefix matchに対応した。
- ただし、SQ8_0 projection runtime側の `BackendRequest.model_arch` はまだ `None` だった。

## 今回の変更点

- 実装用subagent `gpt-5.3-codex-spark` に、runtime側のmodel_arch接続を任せた。
- `SqFp8ProjectionDispatches::from_info` が `model_arch` を受け取り、single/batch/pair/tripleの各SQ8_0 projection dispatchへ伝播するようにした。
- Qwen3/Qwen3.5 model-loop系のSQ8_0 telemetry集計では、`Some("Qwen3")` をdispatch requestへ渡す。
- unknownまたは非model-loop側の既存経路では `None` を渡し、既定選択を維持する。
- stdout/JSON schemaは変更していない。

## 次の行動

- 実際の非direct kernel familyを追加する段階で、`model_arch=Qwen3*` やGPU名条件を持つimplementation entryを追加する。
- 必要になったら、dispatch requestのmodel_archをresult schemaへ保存するかを別途判断する。

## Follow-up: R9700-specific SQ8_0 matvec dispatch

## 前回の要点

- SQ8_0 projection dispatchは `model_arch=Qwen3` を受け取るようになった。
- GPU architectureではRDNA4を選べるが、GPU名ごとの実装ID選択はまだactive registryには存在しなかった。

## 今回の変更点

- 実装用subagent `gpt-5.3-codex-spark` に、R9700固有のmatvec dispatch descriptorを追加させた。
- SQ8_0 matvec projectionに `sq8_0_matvec*_r9700_direct` 系のactive descriptorを追加した。
  - single
  - batch
  - pair
  - triple
- これらは現時点では既存の `Direct` familyを使う。非direct kernelを偽装していない。
- ROCmがR9700を `AMD Radeon Graphics` と返す場合に備えて、dispatch専用のcanonical GPU名を追加した。
  - `compute_major == 12`
  - `gcn_arch_name == gfx1201`
  - メモリ量がローカルR9700相当の範囲
  - この条件のときだけ `Radeon_AI_PRO_R9700` としてdispatchへ渡す。
- 人間向けのstdout/JSON上のGPU名は変更していない。
- fused projection descriptor ID関数はR9700名を返せるが、active fused catalogは従来どおりGeneric/RDNA4に留めた。
- 実機layer0 smokeで、`sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct` を確認した。
  - command:
    `target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d /tmp/ullm-qwen3-14b-fp8-layer0-sq8-artifact 2 1048576 0 len:4x1 1 1 1024 128 1000000 0`
  - 出力上のdevice nameは従来どおり `AMD Radeon Graphics` のまま。

## 次の行動

- 実際の非direct kernel familyを追加するときに、このR9700 descriptorへ接続する。
- 必要なら、full 40-layer Qwen3-14B-FP8 SQ8_0行を取り直して、比較JSONLにもR9700固有IDを反映する。
