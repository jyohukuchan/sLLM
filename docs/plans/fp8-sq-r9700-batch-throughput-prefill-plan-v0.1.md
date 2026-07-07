# FP8 SQ R9700 batch throughput and prefill plan v0.1

## 前回の要点

- AQ4 decodeはR9700で `66-68 tok/s`、V620で約 `41 tok/s` まで改善し、AQ4 decode速度改善はいったん完了扱いにした。
- SQ候補評価では、single request decode tok/sだけでは不十分で、batch時のtotal throughput、prefill throughput、decode throughputを分けて測る必要がある。
- 現行prefillは十分に最適化されていないため、今のprefill tok/sをSQ候補のformat性能として読むと判断を誤る。
- FP8はSQ候補1として扱う。ただし採用決定ではなく、SQ候補評価の基準線として使う。

## 今回の変更点

- この計画では、初期実装と計測対象をR9700/RDNA4に限定する。
- V620/RDNA2は、FP8 native実行ではなくdequant経路が必要になる可能性が高く、今回の主旨から外す。
- batch処理、total token throughput計測、prefill最適化をSQ候補評価の前提作業として同時に進める。
- ある程度prefill/decodeが最適化できた段階で、vLLMで同等条件を動かした場合との比較を行う。
- vLLMのFP8対応は環境依存が強いため、R9700でunsupportedの場合も比較結果として明示的に記録する。
- vLLM比較では、R9700でFP8が動くことを前提にせず、backend、dtype、quantization、failure reasonを結果schemaへ残す。

## 次の行動

1. R9700向けのbatch throughput result schemaを固定する。
2. logical batch runnerを先に作り、その後real batch kernelへ広げる。
3. FP8 SQ候補1のpackage/runtime prototypeを作る。
4. prefillをtoken-by-token実行からbatched/tiled実行へ移す。
5. uLLM側でR9700のprefill/decodeが比較可能な速度になった後、vLLM baselineを同じworkload gridで測る。

## Goal

SQ候補を評価するために、R9700上で次を同じ測定基盤から取得できる状態を作る。

- `prefill total input tok/s`
- `decode total generated tok/s`
- `end-to-end total tok/s`
- request latency p50/p95
- time to first token
- time per output token
- VRAM baseline/peak/consumed
- compact resident bytes
- materialized working-set bytes
- output quality guard

ここでの最初のSQ候補はFP8である。

## Non-Goals

- V620/RDNA2でのFP8 runtime対応
- tensor parallel
- multi-GPU execution
- OpenAI互換server APIの完成
- continuous batchingの完全実装
- SQ formatの最終仕様決定
- vLLMをR9700上で必ずFP8成功させること

## R9700-only execution boundary

この計画の実装・計測対象は、当面R9700/RDNA4の単一GPUに固定する。

- pass/fail判定はR9700だけで行う。
- V620/RDNA2は、FP8 nativeではなくdequant経路が必要になる可能性が高いため、この計画の合否から外す。
- tensor parallel、multi-GPU、V620向けdequant kernelは後続計画へ回す。
- 全resultにGPU名、gfx arch、device index、ROCm version、runtime commit、package/artifact id、warmup回数、measured repeat数を保存する。
- 低速runは、warmup後のper-token latencyが安定していて追加計測で判断が変わらない場合、長時間継続しない。

## Definitions

### Token throughput

この計画では、token throughputを次の3種類に分ける。

```text
prefill total input tok/s =
  sum(prompt tokens processed across all requests) / prefill wall time

decode total generated tok/s =
  sum(generated tokens across all requests) / decode wall time

end-to-end total tok/s =
  (sum(prompt tokens) + sum(generated tokens)) / end-to-end wall time
```

SQ候補のdecode性能を見る主指標は `decode total generated tok/s` とする。
prefill性能を見る主指標は `prefill total input tok/s` とする。
`end-to-end total tok/s` は補助指標であり、prefill/decodeの内訳なしでは採用判断に使わない。

vLLM/ROCm系のbenchmarkで使われる `throughput_gen` は、requests x output length / elapsed timeであり、この計画の `decode total generated tok/s` と近い。
ただしvLLM側のelapsed timeがprefill込みのserver benchmark全体を指す場合は、uLLMのdecode-only値と直接比較しない。
同じ行には、可能な限り `prefill`, `decode`, `end_to_end` のどの時間窓で割った値なのかを保存する。

### Logical batch and real batch

- logical batch:
  - 複数requestを同じbenchmark runで扱うが、内部kernelはまだrequestごとに順次実行してよい段階。
  - scheduler、result schema、latency計測、VRAM計測を先に固定するための段階。
- real batch:
  - 複数tokenまたは複数requestを同じkernelまたはGEMM/GEMV群で実行し、GPU利用率を上げる段階。
  - SQ候補のformat性能評価にはreal batchが必要。

logical batchの結果はcontrol planeや計測基盤の検証には使えるが、SQ候補の最終性能判断には使わない。

## FP8 SQ candidate 1

### Candidate intent

FP8 SQ候補1は、AQ4より低bppを狙うものではない。
目的は、次を満たす基準線を作ることである。

- 8bit級の単純で高速なcompact resident format
- R9700でnativeまたは低overheadに読めるpayload
- prefillのbatched GEMM/GEMV化と相性がいいlayout
- vLLM/ROCm系FP8 baselineと比較しやすい形式

### Candidate variants

最初から1案に固定しない。次の順に試す。

| candidate | weight payload | activation | scale | purpose |
| --- | --- | --- | --- | --- |
| `sq-fp8-w8a16-r9700-v0` | FP8 weight | BF16/F32 activation | tensor or row scale | correctness and simple runtime baseline |
| `sq-fp8-w8a8-r9700-v0` | FP8 weight | FP8 activation | row/channel + token scale | throughput candidate |
| `sq-fp8-kv-r9700-v0` | same as selected weight variant | same | KV FP8 optional | concurrency and context memory experiment |

まず `sq-fp8-w8a16-r9700-v0` を通し、output guardとbatch result schemaを安定させる。
その後、R9700で意味のある速度差が見える場合だけ `w8a8` とKV FP8へ進む。

### Package/runtime metadata

FP8 candidate packageまたはruntime artifactには、少なくとも次を記録する。

- FP8 format: `e4m3`, `e5m2`, or documented variant
- scale granularity: tensor, row, channel, block, token
- scale dtype and layout
- tensor family
- resident bytes
- materialized working-set bytes
- whether full dequant/materialize is used
- kernel path: native FP8, dequant-to-BF16, dequant-to-F32, or mixed

## Measurement schema changes

既存の `docs/specs/inference-benchmark-result-v0.1.md` と
`docs/specs/sq-candidate-runtime-result-v0.1.md` を拡張する。

追加したい主な項目:

```json
{
  "workload": {
    "batch_size": 8,
    "concurrent_requests": 8,
    "prompt_tokens_per_request": [512, 512, 512, 512, 512, 512, 512, 512],
    "generated_tokens_per_request": [128, 128, 128, 128, 128, 128, 128, 128],
    "fixed_decode_steps": true
  },
  "metrics": {
    "prefill_total_input_tokens": 4096,
    "decode_total_generated_tokens": 1024,
    "end_to_end_total_tokens": 5120,
    "prefill_total_input_tps": 0.0,
    "decode_total_generated_tps": 0.0,
    "end_to_end_total_tps": 0.0,
    "per_request_decode_tps_mean": 0.0,
    "time_to_first_token_ms_p50": null,
    "time_to_first_token_ms_p95": null,
    "request_latency_ms_p50": null,
    "request_latency_ms_p95": null,
    "time_per_output_token_ms_p50": null,
    "time_per_output_token_ms_p95": null
  },
  "batching": {
    "mode": "logical|real|continuous",
    "prefill_executor": "token_loop|chunked|batched_gemm",
    "decode_executor": "single_request|batched_decode_step",
    "scheduler_policy": "fixed_batch|continuous"
  }
}
```

## Workload grid

初期はR9700のみで実行する。

### Phase A: correctness and warmup

| concurrent requests | prompt tokens | generated tokens | purpose |
| ---: | ---: | ---: | --- |
| 1 | 128 | 32 | single request sanity |
| 2 | 128 | 32 | multi-request control plane sanity |
| 4 | 128 | 32 | logical batch sanity |

### Phase B: SQ candidate evaluation minimum

| concurrent requests | prompt tokens/request | generated tokens/request | purpose |
| ---: | ---: | ---: | --- |
| 1 | 512 | 128 | single request baseline |
| 4 | 512 | 128 | low concurrency throughput |
| 8 | 512 | 128 | main total throughput check |
| 16 | 512 | 128 | occupancy and VRAM pressure |

### Phase C: prefill pressure

| concurrent requests | prompt tokens/request | generated tokens/request | purpose |
| ---: | ---: | ---: | --- |
| 1 | 2048 | 64 | long prompt single request |
| 4 | 2048 | 64 | batched prefill pressure |
| 8 | 2048 | 64 | high prefill total throughput |

### Phase D: sustained decode

| concurrent requests | prompt tokens/request | generated tokens/request | purpose |
| ---: | ---: | ---: | --- |
| 4 | 512 | 256 | sustained decode |
| 8 | 512 | 256 | main decode total throughput |
| 16 | 512 | 256 | decode concurrency limit |

各caseは原則としてwarmup 1回、measured 3回。
ただし長時間runは、per-token latencyが安定した時点で代表1回に短縮してよい。

## Milestones

### T0: State freeze and result contract, 0.5-1 day

目的:

- R9700限定のSQ候補評価条件を固定する。

手順:

1. 使用するR9700 device indexを固定する。
2. AQ4 latest baseline commitとpackage pathを記録する。
3. FP8 candidate artifact path規約を決める。
4. total throughput schemaの追加項目をdocs/specsへ反映する。
5. result path規約を決める。

成果物:

- updated benchmark schema
- result directory convention
- baseline artifact index

Exit criteria:

- 以後のAQ4/FP8/vLLM結果を同じ列で比較できる。

### T1: Batch throughput benchmark runner, 2-3 days

目的:

- 複数requestのtotal throughputを測れるrunnerを作る。

手順:

1. workload JSONを定義する。
2. requestごとにprompt token列とgeneration lengthを持てるようにする。
3. fixed decode stepsでstop condition差を排除する。
4. prefill/decode/end-to-endのwall timeを分ける。
5. per-request latencyを記録する。
6. VRAMとKV cache使用量を記録する。
7. logical batch modeでまず動かす。

成果物:

- `ullm-engine package-batch-throughput-bench`
- batch throughput JSON/JSONL
- summary markdown

Exit criteria:

- R9700でconcurrent requests `1,2,4` のlogical batch結果が出る。
- `prefill_total_input_tps`, `decode_total_generated_tps`, `end_to_end_total_tps` が別々に保存される。

### T2: FP8 SQ candidate package/runtime prototype, 3-5 days

目的:

- FP8をSQ候補1として、R9700で読み込めるruntime pathを作る。

手順:

1. FP8 payload writerを追加する。
2. scale granularityをまずtensorまたはrowに固定する。
3. MLP、attention projection、linear attention projection、lm_head、embeddingをFP8化する。
4. normや小さいbias/conv/state系はpassthroughのまま残す。
5. R9700 runtimeでFP8 payloadを読む。
6. まずはdequant-to-BF16/F32またはnative FP8 readのどちらが最短か確認する。
7. short prompt guardを通す。

成果物:

- FP8 candidate package or runtime artifact
- FP8 candidate load path
- short guard result

Exit criteria:

- R9700でshort promptが完走する。
- NaN/Infが出ない。
- AQ4 baselineまたはBF16 referenceに対するoutput guardが通る、または失敗原因が記録される。

### T3: Prefill optimization v0.1, 4-7 days

目的:

- token-by-token prefillから、SQ候補評価に使えるbatched/tiled prefillへ移す。

手順:

1. hidden stateを `[tokens, hidden]` layoutで扱うprefill bufferを作る。
2. RMSNormを複数token同時に実行する。
3. MLP gate/up/downをtoken batchに対して実行する。
4. self-attention projectionをtoken batchに対して実行する。
5. self-attention prefillは、まずchunked causal attentionで実装する。
6. linear attentionは、recurrent state更新を壊さない範囲でprojection/MLPをbatched化し、state scanは段階的に最適化する。
7. KV writeをprompt token列に対してまとめる。
8. prefill resultをdecode stateへ接続する。

成果物:

- chunked or batched prefill executor
- prefill component timing
- before/after prefill throughput comparison

Exit criteria:

- R9700で `prompt_tokens=512` のprefillが現行token-loopより明確に速い。
- `prompt_tokens=2048` のprefillがOOMせず完走する。
- output guardが維持される。

### T4: Real batch decode v0.1, 3-5 days

目的:

- decode total generated tok/sを、concurrent requestsで伸ばせる実行経路にする。

手順:

1. scheduler decode batchをfull model decode stepへ接続する。
2. requestごとのblock tableとcache positionをbatched inputとして渡す。
3. embedding/top1/lm_headを複数request分まとめる。
4. AQ4/FP8 projection matvecをbatch方向でまとめる。
5. paged decode attentionをbatch内requestごとに同時実行する。
6. fixed decode stepsで全requestを同じ回数進める。
7. per-request generated tokensとlatencyを検証する。

成果物:

- batched decode step executor
- batch decode component timing
- total generated tok/s summary

Exit criteria:

- R9700でconcurrent requests `4,8` のdecode total generated tok/sがsingle requestの単純逐次実行より改善する。
- generated token countが全requestで一致する。
- guard bundleが通る。

### T5: FP8 candidate evaluation pack, 1-2 days

目的:

- FP8 SQ候補1をAQ4 baselineと比較可能にする。

手順:

1. AQ4 latest baselineを同じbatch runnerで測り直す。
2. FP8 candidateを同じworkload gridで測る。
3. storage/memory fieldsを埋める。
4. output healthを比較する。
5. `sq-candidate-runtime-result-v0.1` をbatch対応へ拡張して記録する。

成果物:

- AQ4 batch baseline rows
- FP8 SQ candidate rows
- comparison markdown

Exit criteria:

- FP8 candidateについて、R9700で少なくとも `batch=1,4,8` のprefill/decode total throughputがある。
- AQ4との差が、速度、VRAM、resident bytes、working-set bytes、qualityの観点で説明できる。

### T6: vLLM comparison preparation, 1-2 days

目的:

- uLLMのR9700結果と比較できるvLLM測定条件を固定する。

手順:

1. vLLM ROCm環境のversion、commit、ROCm versionを固定する。
2. R9700でvLLMが使えるかをsmokeする。
3. Qwen3.5-9Bまたは比較可能なQwen系FP8 modelを選ぶ。
4. FP8 W8A8、FP8 KV-cache、BF16/FP16の対応可否を記録する。
5. R9700でFP8がunsupportedなら、unsupported rowを必ず保存し、可能なdtypeで参考baselineを取る。
6. uLLMと同じprompt/generated/concurrency gridをvLLM側benchmarkへ落とす。
7. startup logからattention backend、quantization backend、fallback有無を抽出する。
8. Radeon fallback backendが必要な場合は、FP8比較とは別行として記録する。

成果物:

- vLLM environment report
- supported/unsupported matrix
- vLLM benchmark command list
- backend/fallback log excerpt

Exit criteria:

- vLLM比較を成功/失敗どちらでも機械可読に記録できる。
- unsupportedの場合も、理由が `unsupported_hardware`, `unsupported_quantization`, `missing_kernel`, `runtime_failure` のどれかに分類される。

### T7: vLLM comparison run, 1-3 days

目的:

- ある程度最適化済みのuLLM FP8/AQ4結果と、vLLM R9700結果を比較する。

開始条件:

- uLLM側でR9700 `batch=4` と `batch=8` のprefill/decode total throughputが安定している。
- output guardが通っている。
- batch runnerのschemaが固定されている。
- prefillは少なくとも主要projection/MLP/self-attention入力処理がtoken-loop主体ではなく、batch/tiled pathへ移っている。
- decodeはscheduler decode batchがfull model stepに接続され、single request逐次実行との差を説明できる。

手順:

1. vLLM smokeを実行する。
2. vLLM throughput benchmarkを実行する。
3. vLLM server benchmarkが必要ならOpenAI-compatible endpoint経由でも測る。
4. VRAMを同じ方法で測る。
5. uLLMとの比較表を作る。
6. FP8がunsupportedの場合、BF16/FP16 baselineとunsupported FP8 rowを併記する。

成果物:

- vLLM benchmark JSONL
- uLLM vs vLLM comparison markdown
- unsupported reason table if needed

Exit criteria:

- R9700でvLLM比較が成功または明示的unsupportedとして記録される。
- uLLMがvLLMに対してどこで負けているか、prefill、decode、batch scaling、VRAMのどれかに分けて説明できる。

## Acceptance criteria

この計画を完了とみなす条件:

1. R9700でbatch throughput benchmark runnerが動く。
2. `prefill total input tok/s`, `decode total generated tok/s`, `end-to-end total tok/s` が別々に保存される。
3. logical batchとreal batchの区別がresultに残る。
4. FP8 SQ候補1がR9700で少なくともshort guardを通る。
5. R9700でFP8 candidateのbatch `1,4,8` 結果がある。
6. prefillが現行token-loopより明確に改善している。
7. decode total throughputがsingle request逐次実行より改善している。
8. AQ4 baselineとFP8 candidateの比較表がある。
9. vLLM比較が成功またはunsupported reason付きで記録されている。

## Progress 2026-07-07

T3の前提作業として、Qwen3.5-9B packageのlinear attention layer 0に対して、R9700上のtoken batch prefill component smokeを追加した。

追加済み:

- `package-linear-attn-recurrent-batch-smoke`
- `package-linear-attn-post-batch-smoke`
- `package-linear-attn-attention-batch-smoke`
- `package-linear-attn-mlp-batch-smoke`
- `package-linear-attn-layer-batch-smoke`

R9700 release results:

| component | prompt tokens | wall ms mean | token/s mean | note |
| --- | ---: | ---: | ---: | --- |
| linear attention recurrent-side | 512 | 27.790506 | 18423.558252 | qkv/a/b projection、qkv prepare、gate/beta、recurrent |
| linear attention post-side | 512 | 22.089981 | 23177.928492 | z projection、post RMSNorm/SiLU、out projection、residual |
| linear attention attention integrated | 512 | 49.605634 | 10321.408180 | attention側を同一stream・同一bufferで接続 |
| linear attention MLP-side | 512 | 83.995376 | 6095.573666 | post RMSNorm、gate/up/down projection、SiLU積、residual |
| linear attention layer partial | 512 | 137.143145 | 3733.325506 | attention側からMLP側まで同一stream・同一bufferで接続 |

解釈:

- linear attention attention側は、分割smoke単純合算と統合smokeの差が小さく、host境界やbuffer接続による大きな追加損失は見えていない。
- MLP側はgate/up/downの3本の大きいAQ4 batch projectionが支配的で、attention側より重い。
- `attention batch + MLP batch` の512 token単純合算は `133.601010 ms`、同一buffer接続後は `137.143145 ms` で、接続追加分は約 `3.54 ms`、約 `2.7%`。
- 次はself-attention layerのprefill batch化、layer stack接続、decode state接続へ進む。

2026-07-07 later:

- self-attention prefill batch化の前提として、`qwen35_qk_norm_rope_batch_f32` runtime primitiveを追加した。
- これはtoken-majorのQwen3.5 gated q projectionとk projectionを受け取り、Q gate分離、Q/K headwise RMSNorm、RoPEを複数token分まとめて処理する。
- CPU testとHIP testを追加し、`cargo test -p ullm-runtime-sys qwen35_qk_norm_rope_batch -- --test-threads=1` で検証した。
- 次はpackage内self-attn layerで `input RMSNorm -> q/k/v AQ4 batch projection -> q/k norm+RoPE batch` を一続きで測る。

2026-07-07 self-attention prefill front-half batch:

- `package-self-attn-qkv-rope-batch-smoke` を追加した。
- Qwen3.5-9B packageのself-attention layer 3に対して、`input RMSNorm -> q/k/v AQ4 batch projection -> qwen35_qk_norm_rope_batch_f32` を同一token batchで接続した。
- このsmokeはcausal attentionやo projectionまでは含まない。self-attention prefill前半のdevice-resident component timingとQ gate/Q RoPE/K RoPEのguardを目的にする。

R9700 release results:

| component | prompt tokens | wall ms mean | token/s mean | note |
| --- | ---: | ---: | ---: | --- |
| self-attention qkv+QK RoPE front-half | 4 | 0.315858 | 12663.905232 | warmup 1、measured 3 |
| self-attention qkv+QK RoPE front-half | 128 | 7.385021 | 17332.381786 | warmup 1、measured 5 |
| self-attention qkv+QK RoPE front-half | 512 | 24.703001 | 20726.227024 | warmup 1、measured 3 |

Guard:

- `input_norm_max_abs_diff <= 0.000072479`
- `q_gate_max_abs_diff = 0`
- `q_rope_max_abs_diff <= 0.000059426`
- `k_rope_max_abs_diff <= 0.000045419`

解釈:

- self-attention prefillのprojection+QK norm/RoPE前半は、512 tokenで約 `20.7k tok/s` までbatch化できた。
- linear attentionのattention側front-halfと同程度の粒度では、Qwen3.5 self-attention側もhost境界なしでdevice-residentに接続できることを確認した。
- 次はこの出力をcausal attention prefillへ接続し、その後o projection/residual、MLP、layer stackへ広げる。

## Decision gates

### FP8 candidate can continue if

- output guardが通る。
- AQ4より品質が大きく崩れない。
- R9700でprefillまたはdecode total throughputがAQ4と同等以上、またはVRAM/working setに明確な利点がある。
- real batch時のscalingがAQ4より悪くない。

### FP8 candidate should be paused if

- native FP8 pathが使えず、dequant overheadでAQ4より明確に遅い。
- output guardが不安定になる。
- prefill最適化の主要ボトルネックがformatではなくattention/linear-attention executor側にある。
- vLLMや外部FP8 baselineと比べて、SQ formatではなくruntime未成熟が支配的だと分かる。

## vLLM comparison notes

2026-07-07時点で確認した公式情報では、vLLM/ROCmのFP8 W8A8やFP8 KV-cacheはAMD Instinct MI300系やCDNA GPUを中心に説明されている。
R9700/RDNA4で同じFP8 pathが動くとは限らない。
一方でROCmのvLLM最適化docsには、Radeon/fallback backendとして `ROCM_ATTN` やTriton系fallbackを使う記述があるため、R9700ではまず実機smokeでbackend選択を確認する。

そのため、この計画ではvLLM比較を次のように扱う。

- R9700でFP8 vLLMが動く場合:
  - uLLM FP8 candidateと同じworkload gridで比較する。
- R9700でFP8 vLLMがunsupportedの場合:
  - unsupported rowを保存する。
  - BF16/FP16またはvLLMが対応するdtypeで参考baselineを取る。
  - FP8同士の速度比較は未成立と明記する。
- vLLMがR9700自体で不安定な場合:
  - failure reasonを記録し、uLLMのSQ策定を止めない。

比較表の最小列:

- engine
- engine commit/version
- model/artifact id
- quantization
- dtype
- GPU and gfx arch
- backend
- concurrent requests
- prompt tokens/request
- generated tokens/request
- prefill total input tok/s
- decode total generated tok/s
- end-to-end total tok/s
- TTFT p50/p95
- TPOT p50/p95
- VRAM baseline/peak/consumed
- status
- unsupported/failure reason

参考:

- AMD ROCm docs: FP8 quantization with AMD Quark for vLLM
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm.html
- AMD ROCm docs: vLLM optimization and FP8 KV-cache
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- vLLM docs: FP8 W8A8
  - https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/

## Risks

| risk | impact | handling |
| --- | --- | --- |
| FP8 native kernels on R9700 are unavailable or incomplete | FP8 candidateがdequant経路になり速度が出ない | unsupported/native-unavailableとして記録し、W8A16やBF16 fallbackを分ける |
| prefill bottleneck is executor-side rather than format-side | SQ候補差が見えない | prefill component timingを先に取り、format評価前にexecutorを直す |
| logical batch result is mistaken for real batch performance | SQ候補を過大評価または過小評価する | result schemaにbatching.modeを必須化する |
| vLLM FP8 is unsupported on R9700 | 直接比較できない | unsupported rowとBF16/FP16参考baselineを併記する |
| vLLM fallback backend result is mistaken for FP8 native result | 外部比較を誤解する | backend、dtype、quantization、fallback有無を比較表の必須列にする |
| continuous batchingまで広げすぎる | 計画が肥大化する | v0.1ではfixed batchに限定し、continuousは後続へ回す |
| quality guardがbatch pathで壊れる | speed結果が無効になる | short guardとprompt guard bundleをbatch pathにも必須にする |

## Deferred items

- V620 FP8 dequant path
- tensor parallel
- continuous batching
- API/server integration
- SQ final format freeze
- multi-model architecture support
- MI300X or NVIDIA hardware comparison
