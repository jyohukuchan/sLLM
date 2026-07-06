# Pre-SQ Runtime TPS Plan v0.1

## 前回の要点

- Qwen3.5-9B `.ullm.d` packageは、RDNA4 R9700とRDNA2 V620でprefix correctness smokeを通せる段階まで来た。
- `qwen35_9b_p4p46_hidden3994_v1` は、CPU/R9700/V620の5 fixture gateでacceptedになった。
- ただし現状は、主にlayer/prefix smokeであり、sq formatを決めるために必要な現実的なprefill/decode token/sはまだ取れていない。

## 今回の変更点

この計画は、sq formatを策定する直前までの実装と検証を対象にする。

sq formatは、実際の推論速度を見ながら決める必要がある。そのため、数tokenのsmokeではなく、少なくとも数百token以上のprefillとdecodeを単一GPUで実行し、prefill TPS、decode TPS、VRAM、正しさ指標を同じrun recordへ保存できる状態を作る。

今回の範囲では、tensor parallel、batch処理、複数同時request、server APIは省く。

## 次の行動

1. token IDs入力のend-to-end Qwen3.5 package runtimeを作る。
2. 512 token以上のprefillをR9700/V620で測る。decodeは256 tokenを目安にするが、pilot runで明らかに低速かつ安定している場合は短縮する。
3. BF16/materialize AQを同じbenchmark harnessで比較できる状態にしてから、sq format候補の策定へ進む。

## Goal

sq format策定の入力として使える、実推論に近いtoken/s測定基盤を作る。

ここでの「実推論に近い」は次を満たすことを指す。

- Qwen3.5-9Bの全decoder layerを通す。
- 入力はtoken IDsでよい。tokenizer統合は必須ではない。
- prefillは最低 `512` tokens、可能なら `2048` tokensまで測る。
- decodeは最低 `256` generated tokens、可能なら `512` tokensまで測る。
- single GPU、single request、batch size `1`、tensor parallel `1`で測る。
- R9700/RDNA4とV620/RDNA2の両方で、同じrun schemaの結果を保存する。

## Non-Goals

- tensor parallel
- batch処理またはcontinuous batching
- 複数同時request
- OpenAI互換HTTP server
- tokenizer統合を必須にすること
- prompt品質や会話品質の評価
- sq formatの最終仕様決定
- sq向けfused kernelの最適化完了

## Current Baseline

できていること:

- `.ullm.d` package生成と検証
- AQ packageのdirect conversion
- Qwen3.5-9B向けaccepted policy `qwen35_9b_p4p46_hidden3994_v1`
- package loader、runtime weight registry、materialize path
- decoder layer、paged KV、scheduler、model-loop smoke
- CPU/R9700/V620でのlayer/prefix correctness smoke

足りないこと:

- token IDsからembeddingを引いて全layerを通すend-to-end runtime
- final RMSNorm、lm_head、logits/top-k/sampling
- 長いprefill/decodeを測るbenchmark CLI
- prefill TPS、decode TPS、VRAM、KV cache bytesを保存するrun schema
- BF16 baselineとAQ runtime pathを同じ条件で比較する仕組み

## Acceptance Criteria

sq策定へ進める条件:

1. `ullm-engine` に、token IDs入力でQwen3.5-9B packageをend-to-end実行するCLIがある。
2. CLIは少なくとも次を保存する。
   - model/package path
   - git commit
   - backend and device
   - prompt tokens
   - generated tokens
   - prefill wall time
   - decode wall time
   - prefill TPS
   - decode TPS
   - total TPS
   - VRAM baseline/peak/consumed
   - KV cache bytes
   - correctness summary
3. R9700で `prompt_tokens=512`, `generated_tokens=256` が完走する。
4. V620で `prompt_tokens=512`, `generated_tokens=256` が完走する。
5. 可能ならR9700で `prompt_tokens=2048`, `generated_tokens=512` を完走する。
6. 1つ以上の短いfixtureで、HF/PyTorchまたは既存golden fixtureに対するlogits/top-k sanity checkが通る。
7. BF16/materialize AQのどちらか片方だけではなく、sq候補と比較可能な基準線が最低1本ある。
8. 結果が `benchmarks/results/YYYY-MM-DD/engine/` に機械可読JSONLまたはJSON summaryとして残る。

## Measurement Grid

初期はsingle GPU、single requestだけに固定する。

Primary devices:

| label | device | target |
| --- | --- | --- |
| `r9700` | RDNA4 R9700 | primary speed target |
| `v620` | RDNA2 V620 | compatibility target |

Required grid:

| prompt tokens | generated tokens | purpose |
| ---: | ---: | --- |
| `128` | `32` | correctness and warmup |
| `512` | `256` | minimum sq-planning speed input |
| `2048` | `256` | realistic prefill pressure |

Stretch grid:

| prompt tokens | generated tokens | purpose |
| ---: | ---: | --- |
| `2048` | `512` | sustained decode |
| `4096` | `512` | context pressure if memory allows |

Run each condition with:

- batch size: `1`
- concurrent requests: `1`
- tensor parallel: `1`
- sampling: greedy first
- repeat count: `3` after one warmup, if runtime is stable

## Measurement Stop Policy

`512/256` は、持続decode速度を測る価値がある場合の基準gridである。pilot runでdecodeが明らかに実用域から遠く、かつ32 token以上でper-token latencyが安定している場合、256 token完走の追加価値は低い。

その場合は次の扱いにする。

- 代表deviceで1本だけ長いrunを完走し、下限性能のanchorとして残す。
- 以後の比較は、prefill圧力を見る長いpromptと、decode律速を見る短いgenerated token数を分ける。
- 同じ遅い経路でV620/R9700の長いdecodeを繰り返さない。
- `256` token以上のdecodeは、runtime実装またはsq候補が変わった後、または発表用に持続値が必要な場合だけ再実行する。
- 打ち切ったrunは失敗ではなく、`intentionally stopped` として理由と途中のVRAM到達値だけを記録する。

## Milestones

### T0: State Freeze and Benchmark Contract, 0.5 day

目的:

- 今のaccepted package、device IDs、測定schema、除外範囲を固定する。

手順:

1. accepted package pathを固定する。
2. R9700/V620のdevice indexを記録する。
3. benchmark JSON schemaを既存 `inference-benchmark-result-v0.1.md` と照合する。
4. 足りない項目だけ小さく追加する。
5. smoke用とthroughput用の出力path規約を決める。

成果物:

- `benchmarks/results/YYYY-MM-DD/engine/pre-sq-runtime-artifact-index.md`
- benchmark JSON example

Exit criteria:

- 以後の測定結果を同じ列で比較できる。

### T1: End-to-End Token-ID Runtime Skeleton, 2-3 days

目的:

- token IDs入力から、embedding、全decoder layer、final RMSNorm、lm_headまでを通す。

手順:

1. packageからembedding、final norm、lm_headを読む。
2. token ID列からinitial hidden stateを作る。
3. 既存 `Qwen3PackageModelRuntime` を全layer実行へ広げる。
4. final RMSNormを実行する。
5. lm_headで最終logitsを出す。
6. top-kを出力する。
7. まずは短い固定token列でCPU pathを通す。

成果物:

- `ullm-engine package-token-ids-logits-smoke`
- logits/top-k JSON report

Exit criteria:

- token IDsから最終logits/top-kが得られる。
- 短いfixtureでNaNなし、shape一致、top-k sanity checkが通る。

### T2: Single-Request Prefill and Decode Loop, 2-3 days

目的:

- 1 requestで、長いprefillと複数token decodeを動かす。

手順:

1. prompt token列を受け取るCLI引数またはJSON inputを定義する。
2. prefillを全prompt token分実行する。
3. paged KV cacheにprefill結果を残す。
4. greedy decodeを `N` tokens分回す。
5. generated token ID、top-k、各token latencyを保存する。
6. decode中のKV cache進行とscheduler進行を検証する。

成果物:

- `ullm-engine package-token-ids-generate-smoke`
- generated token IDs JSON
- per-token latency JSONL

Exit criteria:

- `prompt_tokens=128`, `generated_tokens=32` がCPU/R9700で完走する。
- R9700とV620で同じCLIが動く。

### T3: Throughput Benchmark Harness, 1-2 days

目的:

- 数百token以上のprefill/decodeで現実的なTPSを保存する。

手順:

1. `package-token-ids-bench` CLIを追加する。
2. warmup runとmeasured runを分ける。
3. prefill wall time、decode wall time、total wall timeを測る。
4. VRAM baseline/peak/consumedを取得する。
5. KV cache bytesとallocated blocksを保存する。
6. JSONL schemaを固定する。
7. summary toolでmedian/mean/stdevを出す。

成果物:

- `ullm-engine package-token-ids-bench`
- `tools/summarize-runtime-tps.py`
- benchmark JSONL and summary markdown

Exit criteria:

- R9700/V620で `prompt_tokens=512`, `generated_tokens=256` のTPSが保存される。
- prefill TPSとdecode TPSが別々に出る。

### T4: Correctness Guard for Long Runs, 1-2 days

目的:

- 長いTPS runが壊れた値を測っていないことを確認する。

手順:

1. 短いreference promptをHF/PyTorchまたは既存golden fixtureで固定する。
2. final logitsのtop-k agreementを比較する。
3. generated tokenの最初の数stepだけreferenceと比較する。
4. 長いrunでは全token reference比較をしない代わりに、NaN/Inf、logit range、top-k stabilityを記録する。

成果物:

- correctness guard JSON
- long-run sanity summary

Exit criteria:

- 短いrunはreferenceと一致または許容範囲内。
- 長いrunはNaN/Infなしで、logit rangeが異常値にならない。

### T5: BF16 and Materialized-AQ Baselines, 2-4 days

目的:

- sq候補と比較するための速度基準線を作る。

手順:

1. BF16/passthrough baselineを測る。
2. accepted AQ packageのmaterialize pathを測る。
3. materialize costとdecode loop costを分離する。
4. package load time、first-run materialization time、steady-state decode timeを分ける。
5. R9700/V620で同じgridを走らせる。ただし、decodeが明らかに低速なbaselineではStop Policyに従い、短いdecode測定とVRAM到達確認を優先する。

成果物:

- BF16 runtime benchmark summary
- materialized AQ runtime benchmark summary
- device comparison table

Exit criteria:

- sq候補が勝つべき基準線が明確になる。
- 「保存形式の良し悪し」と「runtime実装の未成熟」が分けて見える。
- 真のBF16 baselineを現行package/runtimeで作れない場合は、その理由をartifactとして残し、pre-sq範囲では明示的にdeferできる。

### T6: Pre-SQ Decision Pack, 0.5-1 day

目的:

- sq format策定に入るための判断材料を1つにまとめる。

手順:

1. R9700/V620のTPSとVRAMをまとめる。
2. BF16 vs materialized AQの差をまとめる。
3. prefill律速かdecode律速かを分類する。
4. sqで最初に試すべき保存粒度とscale候補を列挙する。
5. 未解決リスクを明記する。

成果物:

- `docs/research/pre-sq-runtime-tps-results-YYYY-MM-DD.md`
- `docs/plans/sq-format-design-plan-v0.1.md` の入力メモ

Exit criteria:

- sq format会議または実装判断で、実測TPSを根拠に候補を削れる。

## Schedule Estimate

集中して進めた場合の目安:

| milestone | estimate |
| --- | ---: |
| T0 | `0.5 day` |
| T1 | `2-3 days` |
| T2 | `2-3 days` |
| T3 | `1-2 days` |
| T4 | `1-2 days` |
| T5 | `2-4 days` |
| T6 | `0.5-1 day` |

Total:

- optimistic: `9 days`
- realistic: `2-3 weeks`
- if lm_head/full logits or VRAM handling needs major rework: `4 weeks`

## User Involvement

sq策定までは、君の判断が必要な箇所は少ない。

必要になりそうな確認は次だけでよい。

1. 最初の発表対象をQwen3.5-9Bに固定してよいか。
2. benchmark gridで `512/256` を最低条件、`2048/512` をstretch条件にしてよいか。
3. sq策定に入る時点で、速度、VRAM、正しさのどれを最優先にするか。

それ以外は、実装と測定を進めながら結果をartifactとして残せばよい。

## Risks

| risk | impact | mitigation |
| --- | --- | --- |
| lm_headが重く、初期decodeが遅すぎる | TPSが実用値から遠くなる | まずtop-k/argmax用の単純実装で測り、後でkernel最適化する |
| V620のVRAMが足りない | V620 gridが縮む | `512/256` を最低条件にし、`2048/512` はR9700優先にする |
| materialize AQがVRAMを食いすぎる | sq比較が歪む | materialize costとsteady-stateを分離して記録する |
| correctness referenceが重い | 長いrunの検証が遅くなる | 短いreferenceだけ厳密比較し、長いrunはsanity guardにする |
| tokenizer統合で時間を使いすぎる | sq前のTPS取得が遅れる | token IDs入力を正式なpre-sq仕様にする |

## Out-of-Scope Backlog

sq策定後または発表プロトタイプ後に戻す項目:

- tokenizer統合
- HTTP server
- tensor parallel
- batch/continuous batching
- prefix cache reuse
- speculative decode/MTP
- multi-GPU prefill/decode分離
- sq fused kernel最適化
- MI300X benchmark expansion
