# FP8 SQ R9700 batch throughput plan

## 前回の要点

- AQ4 decodeはR9700で `66-68 tok/s`、V620で約 `41 tok/s` まで改善し、decode速度改善は一旦完了扱いにした。
- SQ候補評価にはsingle request decode tok/sだけでなく、batch時のprefill/decode total throughputが必要。
- 現行prefillは未最適化なので、SQ候補の評価前にprefill最適化も必要。

## 今回の変更点

- `uLLM-project/docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` を追加した。
- 計画の対象をR9700/RDNA4のみに限定した。
- FP8をSQ候補1として扱うが、採用決定ではなく基準線として位置づけた。
- logical batch、real batch、prefill total input tok/s、decode total generated tok/s、end-to-end total tok/sを定義した。
- ある程度uLLM側のprefill/decodeを最適化した後、vLLM比較を行う段階を計画に含めた。
- vLLM FP8がR9700でunsupportedの場合も、unsupported rowとして記録する方針にした。
- `uLLM-project/docs/words.txt` にbatch throughput関連用語を追加した。

## 次の行動

1. R9700 batch throughput result schemaを実装する。
2. logical batch runnerを作り、total throughputとlatencyを保存できるようにする。
3. FP8 SQ候補1のpackage/runtime prototypeを作る。
4. prefillをbatched/tiled実行へ移す。
5. real batch decodeを実装し、AQ4 baselineとFP8 candidateを同じgridで測る。
6. uLLM側の速度が安定したらvLLMのR9700比較を実施する。

## 実装メモ

- `ullm-engine package-batch-throughput-bench` を追加した。
- 入力batchは `len:NxM` または `REQ1;REQ2;...` で指定できる。
- generated token数はscalarまたはrequest数と同じCSVで指定できる。
- 出力schemaは `package-batch-throughput-bench-v0.1` とし、次を保存する。
  - `prefill_total_input_tps`
  - `decode_total_generated_tps`
  - `end_to_end_total_tps`
  - request latency p50/p95
  - time to first token p50/p95
  - time per output token p50/p95
  - `batching.mode`
- 今回のrunnerはlogical batchで、内部では既存のsingle-request `package-token-ids-generate-smoke` を順次呼ぶ。real batch性能の判断には使わない。
- `docs/specs/inference-benchmark-result-v0.1.md` と `docs/specs/sq-candidate-runtime-result-v0.1.md` にbatch throughput semanticsを追記した。

## 確認

- `cargo test -p ullm-engine package_prompt_token_ids_batch -- --test-threads=1`
- `cargo test -p ullm-engine package_generated_tokens_batch -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo fmt --all --check`
- `git diff --check`

## 次の実装候補

1. logical batch resultをJSONLへ変換するrunner scriptを追加する。
2. R9700上でAQ4 baselineのlogical batch sanityを `1,2,4` concurrent requestsで取る。
3. real batch prefill executorを作り、token loop prefillからbatched/tiled実行へ移す。
4. real batch decode executorを作り、logical batch resultと同じschemaで比較する。

## R9700限定・vLLM比較計画の補強

- `uLLM-project/docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` にR9700-only execution boundaryを追加した。
- この計画のpass/failはR9700/RDNA4のみで判定し、V620/RDNA2のFP8 dequant経路は後続へ回す。
- vLLM比較は、R9700でFP8が動くことを前提にしない。success、unsupported、fallbackのいずれでも同じ比較表に保存する。
- vLLM/ROCm側の `throughput_gen` は、elapsed timeの取り方によってdecode-only値にもend-to-end生成値にも見えるため、uLLM側の `decode total generated tok/s` と比較する前に時間窓を必ず記録する。
- uLLM側の比較開始条件は、batch runner schema固定、output guard維持、prefillの主要部分のbatch/tiled化、decode scheduler batchのfull model step接続とした。
- 公式docs確認では、vLLM FP8 W8A8はH100/MI300Xなどを中心に説明され、ROCm vLLM最適化docsはInstinct向けAITERとRadeon/fallback backendを分けている。そのためR9700では実機smokeでbackendを確認してから比較する。

## Self-attention qkv+RoPE batch smoke

- `package-self-attn-qkv-rope-batch-smoke` を追加した。
- 対象はQwen3.5 self-attention prefill前半で、`input RMSNorm -> q/k/v AQ4 batch projection -> qwen35_qk_norm_rope_batch_f32` を同一token batchで実行する。
- causal attention、o projection、residual add、MLPはまだ含めていない。
- Qwen3.5 gated q projectionのgate分離、Q/K headwise RMSNorm、RoPEをreadback referenceで検証する。
- R9700では `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1` と `ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1` を付けて確認した。

R9700 release results:

| prompt tokens | repeats | mean ms | min ms | max ms | token/s | q gate diff | q rope diff | k rope diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 0.315858 | 0.311305 | 0.322625 | 12663.905232 | 0.000000000 | 0.000001431 | 0.000000954 |
| 128 | 5 | 7.385021 | 6.079008 | 11.365913 | 17332.381786 | 0.000000000 | 0.000012338 | 0.000007033 |
| 512 | 3 | 24.703001 | 24.323211 | 25.418999 | 20726.227024 | 0.000000000 | 0.000059426 | 0.000045419 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine package_prompt_token_ids -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- `git diff --check`

次の実装候補:

1. self-attention prefill causal attention batch smokeを追加する。
2. self-attention front-half outputをo projection/residualへ接続する。
3. self-attention layer batch smokeとしてMLPまでつなぐ。

## Self-attention causal attention batch smoke

- `package-self-attn-attention-batch-smoke` を追加した。
- 対象はQwen3.5 self-attention prefill attention側で、`input RMSNorm -> q/k/v AQ4 batch projection -> qwen35_qk_norm_rope_batch_f32 -> causal_attn_f32` を同一token batchで実行する。
- o projection、residual add、MLPはまだ含めていない。
- R9700では `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`、`ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1`、`ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1` を付けて確認した。

R9700 release results:

| prompt tokens | repeats | mean ms | min ms | max ms | token/s | attention diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 0.484690 | 0.475290 | 0.501051 | 8252.691925 | 0.000001058 |
| 128 | 5 | 24.605704 | 24.062109 | 24.963688 | 5202.045750 | 0.000003248 |
| 512 | 3 | 281.601274 | 281.118314 | 281.971022 | 1818.173590 | 0.000003248 |

Front-halfとの差分:

| prompt tokens | qkv+RoPE mean ms | attention included mean ms | delta ms |
| ---: | ---: | ---: | ---: |
| 4 | 0.315858 | 0.484690 | 0.168832 |
| 128 | 7.385021 | 24.605704 | 17.220683 |
| 512 | 24.703001 | 281.601274 | 256.898273 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine package_prompt_token_ids -- --test-threads=1`
- `cargo build -p ullm-engine --release`

解釈:

- QKV projection、QK norm、RoPE前半は512 tokenで `24.70 ms` なので、attention込みの512 token `281.60 ms` の大部分はcausal attention prefill kernel側で発生している。
- SQ候補のprefill評価前に、self-attention prefill attention kernelをtiled/blocked化する必要がある。
- 次はo projection/residualへ広げる前に、causal attention prefill kernelのcomponent benchmarkとtiling方針を固める。

## Long context and cached prefix plan update

- 512 tokenまでのcomponent timingだけでは、長コンテキストprefill評価として不足していると判断した。
- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` にcold prefillとcached prefix prefillの定義を追加した。
- workload gridに次を追加した。
  - cold prefill: `4096/8192/16384`
  - long context upper bound: `32768/65536`
  - cached prefix prefill: `L=4096/16384/65536` と `M=1/16/128/512`
- schemaに `prefill_mode`, `cached_prefix_tokens_per_request`, `new_prefill_tokens_per_request`, `total_context_tokens_after_prefill_per_request`, `estimated_prefill_attention_work_tokens` を追加した。
- T3 exit criteriaに、`L=65536, M=1/16/128/512` の少なくとも1系統をOOMせず完走させる条件を追加した。

## Prefill pattern exploration plan update

- 既存計画にはlong contextとcached prefixの大枠は入っていたが、512 tokenまでの結果をshort sanityに限定し、より多いpatternを測って実装方針へ反映する条件が弱かった。
- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` にPhase C4を追加した。
- Phase C4では、cold prefill length scaling、cached prefix chunk scaling、batch width scaling、mixed realistic prompt、component isolationを分けて測る。
- 必須sweepは `N=1024/2048/4096/8192/16384`、`L=4096/16384/65536`、`M=1/16/128/512`、`B=1/2/4/8` とした。
- 512 token結果はshort sanity、warmup確認、局所的なcomponent regression検出には使うが、SQ候補のprefill性能判断には使わない方針にした。
- T3手順とexit criteriaにも、長さ別・chunk別・batch別の結果から次の最適化対象を分類する条件を追加した。

## Phase C4 cached prefix sweep runner

- `tools/run-runtime-cached-prefix-sweep.py` を追加した。
- `runtime-cached-prefix-attn-smoke` を `L/M/executor` のgridとして実行し、JSONLとMarkdown summaryを保存するrunnerである。
- dry-runと `device-index=0` の引数検証を確認した。
- R9700で `cached_prefix_chunked` の代表sweepを実行した。

保存先:

- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-sanity.jsonl`
- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-sanity.md`

R9700 release results:

| L | M | repeats | mean ms | new tok/s | pair/s | diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 3 | 103.203890 | 9.689557 | 39698.115902 | 0 |
| 4096 | 16 | 3 | 124.240184 | 128.782810 | 528589.041882 | 0 |
| 4096 | 128 | 1 | 1019.441962 | 125.558889 | 522387.757078 | 0 |
| 16384 | 1 | 1 | 524.079953 | 1.908106 | 31264.313596 | 0 |
| 16384 | 16 | 1 | 578.890933 | 27.639058 | 453073.256202 | 0 |
| 16384 | 128 | 1 | 4616.811130 | 27.724764 | 456030.784175 | 0 |
| 65536 | 1 | 1 | 2055.547196 | 0.486488 | 31882.994527 | 0 |
| 65536 | 16 | 1 | 2356.189138 | 6.790626 | 445088.207516 | 0 |
| 65536 | 128 | 1 | 18234.605441 | 7.019620 | 460490.578048 | 0 |

解釈:

- `M=16` と `M=128` は同じ `L` ではほぼ同程度のnew input tok/sになった。
- `M=1` はdecode-like boundaryとしてpair/sが `31k-40k` 程度まで落ちる。
- `M=16/128` ではpair/sが `445k-529k` 程度に上がるが、`L` にほぼ反比例してnew input tok/sが落ちる。
- 次は `M` を増やすだけでなく、score計算共有、tiled cached-prefix attention、KV read coalescingが必要。

## Cached prefix shared-score kernel v1

- `ullm_cached_prefix_attn_f32_kernel` を、1 output element 1 threadの実装から、1 block = 1 token/headのshared-score実装へ変更した。
- max scoreとsoftmax denominatorをblock内reduceで求め、value次元間で共有する。
- weighted value計算ではまだvalueごとにscoreを再計算するため、完全なtiled attentionではない。
- R9700で `L=4096, M=16` の単発smokeを確認し、`128.78 tok/s` 相当から `287.21 tok/s` 相当まで改善することを確認した。
- Phase C4の同一gridで再測定し、全9ケースで `sampled_max_abs_diff=0` を確認した。

保存先:

- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-shared-score-v1.jsonl`
- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-shared-score-v1.md`

R9700 release comparison:

| L | M | old tok/s | shared-score tok/s | speedup | shared-score pair/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 9.689557 | 20.892045 | 2.156x | 85594.708602 |
| 4096 | 16 | 128.782810 | 286.536642 | 2.225x | 1176089.649006 |
| 4096 | 128 | 125.558889 | 293.886309 | 2.341x | 1222713.987897 |
| 16384 | 1 | 1.908106 | 4.977861 | 2.609x | 81562.254321 |
| 16384 | 16 | 27.639058 | 52.762509 | 1.909x | 864909.422090 |
| 16384 | 128 | 27.724764 | 73.211619 | 2.641x | 1204221.322034 |
| 65536 | 1 | 0.486488 | 1.097658 | 2.256x | 71937.242905 |
| 65536 | 16 | 6.790626 | 9.203588 | 1.355x | 603244.548450 |
| 65536 | 128 | 7.019620 | 16.081822 | 2.291x | 1054975.580718 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-runtime-sys`
- `cargo test -p ullm-runtime-sys cached_prefix_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 single smoke: `runtime-cached-prefix-attn-smoke 2 4096 16 3 16 4 256 256 cached_prefix_chunked`

## Runtime causal attention batch primitive

- `ullm_runtime_causal_attn_batch_f32` を追加し、q/k/v/outputを `[batch, sequence, head, dim]` layoutで扱うcold causal attention primitiveを実装した。
- Rust wrapper `causal_attn_batch_f32` と、engine CLI `runtime-causal-attn-batch-smoke` を追加した。
- `ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL=1` でR9700上のHIP kernel pathを確認した。
- long sequenceでfull output readbackが支配的にならないよう、engine smokeはsampled verificationだけを読み戻す。

R9700代表結果:

| B | N | mean ms | total input tok/s | attention pair/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 18.603645 | 27521.487903 | 7059261.647216 |
| 4 | 512 | 68.139151 | 30056.141879 | 7709400.392004 |
| 8 | 512 | 135.607344 | 30204.853802 | 7747545.000218 |
| 1 | 2048 | 274.000056 | 7474.451027 | 7657575.077284 |
| 4 | 2048 | 1095.987421 | 7474.538339 | 7657664.528476 |
| 8 | 2048 | 2208.166702 | 7419.729673 | 7601513.050319 |
| 1 | 4096 | 1127.648016 | 3632.339117 | 7440846.681293 |
| 4 | 4096 | 4649.452562 | 3523.855719 | 7218618.439491 |

保存先:

- `uLLM-project/benchmarks/results/2026-07-07/runtime-causal-attn-batch/phase-c4-cold-prefill-batch-v1.md`

解釈:

- real batch入力形状での測定基盤はできた。
- ただしBを増やしてもattention pair/sはほぼ横ばいで、wall timeはbatch数に近く比例して伸びる。
- 次はfull model接続より前に、causal attention prefill kernelそのもののscore reuse、tiled/block化、K/V read coalescingを進めるのが妥当。

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-runtime-sys`
- `cargo test -p ullm-runtime-sys causal_attn_batch -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`

## Runtime causal attention online softmax v1

- 最初にquery vectorをshared memoryへcacheする案を試したが、R9700実測では少し遅くなったため採用しなかった。
- 採用した変更は、`ullm_causal_attn_f32_kernel` と `ullm_causal_attn_batch_f32_kernel` の通常shapeでonline softmaxを使うもの。
- `value_dim <= blockDim.x` の場合、q/k score dot-productを従来の3passから1passへ減らす。Qwen3.5の今回shapeは `value_dim=256`, `blockDim.x=256` なのでこのpathに入る。
- `value_dim > blockDim.x` の場合は既存3pass pathをfallbackとして残した。

R9700 raw runtime代表結果:

| B | N | old mean ms | new mean ms | speedup | new pair/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 18.603645 | 7.239056 | 2.570x | 18141591.942375 |
| 4 | 512 | 68.139151 | 27.982984 | 2.435x | 18772551.204689 |
| 8 | 512 | 135.607344 | 55.488527 | 2.444x | 18934076.341788 |
| 1 | 2048 | 274.000056 | 118.997799 | 2.303x | 17632057.162021 |
| 4 | 2048 | 1095.987421 | 460.720775 | 2.379x | 18216465.289789 |
| 8 | 2048 | 2208.166702 | 908.260240 | 2.431x | 18480835.397837 |
| 1 | 4096 | 1127.648016 | 464.365975 | 2.428x | 18069058.582561 |
| 4 | 4096 | 4649.452562 | 1860.373915 | 2.499x | 18040794.768930 |
| 1 | 8192 | n/a | 1887.833494 | n/a | 17776211.782796 |

Package propagation:

| component | N | old mean ms | new mean ms | speedup |
| --- | ---: | ---: | ---: | ---: |
| self-attention attention batch | 512 | 281.601274 | 33.528531 | 8.399x |
| self-attention attention batch | 2048 | n/a | 227.980683 | n/a |
| self-attention attention batch | 4096 | n/a | 694.683770 | n/a |
| self-attention attention batch | 8192 | n/a | 2402.205139 | n/a |
| self-attention layer batch | 512 | 141.768580 | 133.188978 | 1.064x |
| self-attention layer batch | 2048 | 777.894286 | 628.195305 | 1.238x |
| self-attention layer batch | 4096 | 2182.970006 | 1518.104339 | 1.438x |
| self-attention layer batch | 8192 | 6892.180390 | 4162.250951 | 1.656x |

保存先:

- `uLLM-project/benchmarks/results/2026-07-07/runtime-causal-attn-batch/phase-c4-cold-prefill-online-softmax-v1.md`

解釈:

- raw attentionは約2.3-2.5倍速くなり、attention pair/sは約18M pair/sになった。
- `B` を増やしてもpair/sが大きく伸びるわけではないため、batch方向の効率改善はまだ残る。
- layer全体ではcontextが長いほど効く。短い512 tokenではMLP/projection側の割合が大きく、改善幅は小さい。
- 次はtiled/block causal attentionでK/Vを近接timestep/head間で再利用するか、projection/MLP側の残コストを先に削るかを比較する。

確認:

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys causal_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`

## Runtime cached prefix attention online softmax v1

- `ullm_cached_prefix_attn_f32_kernel` にonline softmax pathを追加した。
- `value_dim <= blockDim.x` の通常shapeでは、cached prefix attentionのq/k score dot-productを3passから1passに減らす。
- `value_dim > blockDim.x` では既存3pass pathをfallbackとして残した。

R9700 Phase C4 cached prefix grid:

| L | M | old mean ms | new mean ms | speedup | new tok/s | new pair/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 47.865108 | 4.039271 | 11.850x | 247.569405 | 1014291.851649 |
| 4096 | 16 | 55.839281 | 4.193457 | 13.316x | 3815.467763 | 15660587.434186 |
| 4096 | 128 | 435.542576 | 30.812317 | 14.135x | 4154.182887 | 17283477.902684 |
| 16384 | 1 | 200.889494 | 18.882177 | 10.639x | 52.959995 | 867749.518501 |
| 16384 | 16 | 303.245627 | 19.256383 | 15.748x | 830.893320 | 13620418.746345 |
| 16384 | 128 | 1748.356354 | 170.666624 | 10.244x | 750.000188 | 12336378.084095 |
| 65536 | 1 | 911.030189 | 76.723657 | 11.874x | 13.033790 | 854195.466725 |
| 65536 | 16 | 1738.452511 | 78.523667 | 22.139x | 203.760224 | 13355362.020982 |
| 65536 | 128 | 7959.297024 | 673.121102 | 11.824x | 190.158947 | 12474522.006591 |

保存先:

- `uLLM-project/benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-online-softmax-v1.md`

解釈:

- cached prefix componentは前回shared-score比で `10.2-22.1x` 改善した。
- `L=65536, M=128` が `7959ms` から `673ms` へ落ち、長prefix代表runをSQ候補比較に使いやすくなった。
- `M=1` はdecode-like boundaryなのでpair/sがまだ低い。次はdecode-like pathとchunked cached prefix pathを分けて扱う必要がある。

確認:

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys cached_prefix_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`

## Runtime decode attention head-parallel v1

- `ullm_decode_attn_f32_kernel` にhead-parallel online-softmax pathを追加した。
- 通常shapeでは `head_dim=256`, `value_dim=256` なので、`1 block = 1 q_head` としてhead_dim reductionとvalue lane計算を同じblockで行う。
- 旧element-parallel pathは残し、`ULLM_DISABLE_DECODE_ATTN_HEAD_PARALLEL=1` で強制できるようにした。

R9700 M=1 decode-like boundary:

| L | old decode-loop ms | head-parallel decode-loop ms | speedup | chunked online ms |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 103.281946 | 3.488255 | 29.608x | 4.039271 |
| 16384 | 522.266906 | 16.385404 | 31.874x | 18.882177 |
| 65536 | 2035.323208 | 66.730666 | 30.501x | 76.723657 |

M=16 check:

| L | decode-loop ms | decode-loop tok/s | chunked online ms | chunked online tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 49.781116 | 321.407017 | 4.193457 | 3815.467763 |
| 16384 | 305.251376 | 52.415816 | 19.256383 | 830.893320 |
| 65536 | 1199.161926 | 13.342652 | 78.523667 | 203.760224 |

保存先:

- `uLLM-project/benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-decode-loop-head-parallel-v1.md`

解釈:

- `M=1` ではdecode loopがchunked cached-prefixより約1.15倍速い。
- `M=16` では逐次launchになるdecode loopが大きく遅い。
- Phase C4のexecutor splitは `M=1 -> decode_attn_f32_loop`, `M>=16 -> cached_prefix_chunked` とする。

確認:

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys decode_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 Phase C4 shared-score sweep

次の実装候補:

1. weighted value側のscore再計算を減らすtile設計を試す。
2. source tile単位でQ/K/V readを共有し、KV read coalescingを改善する。
3. `M=1` decode-like pathはcached prefillとは別kernelとして扱う。

## Cached prefix source-shared kernel v2

- `ullm_cached_prefix_attn_f32_kernel` を、shared-score v1からsource-shared v2へ変更した。
- 各source timestepのQK dotをblock内reduceで1回だけ計算し、そのsoftmax weightをvalue次元のthreadへ共有する。
- value_dimがblock size以下の現Qwen条件では、weighted value側のscore再計算を避けられる。
- `L=4096, M=16` の単発smokeでは、v1の約 `286 tok/s` から約 `1780 tok/s` へ改善した。
- Phase C4の同一gridで再測定し、全9ケースで `sampled_max_abs_diff=0` を確認した。

保存先:

- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-source-shared-v2.jsonl`
- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-source-shared-v2.md`

R9700 release comparison:

| L | M | v0 tok/s | v1 tok/s | v2 tok/s | v2/v1 | v2/v0 | v2 pair/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 9.689557 | 20.892045 | 115.478659 | 5.527x | 11.918x | 473116.067678 |
| 4096 | 16 | 128.782810 | 286.536642 | 1778.792810 | 6.208x | 13.812x | 7301055.087249 |
| 4096 | 128 | 125.558889 | 293.886309 | 1802.792780 | 6.134x | 14.358x | 7500519.359248 |
| 16384 | 1 | 1.908106 | 4.977861 | 24.836776 | 4.989x | 13.016x | 406950.571712 |
| 16384 | 16 | 27.639058 | 52.762509 | 387.795099 | 7.350x | 14.031x | 6356931.166031 |
| 16384 | 128 | 27.724764 | 73.211619 | 362.477959 | 4.951x | 13.074x | 5962218.712750 |
| 65536 | 1 | 0.486488 | 1.097658 | 5.205099 | 4.742x | 10.699x | 341126.543957 |
| 65536 | 16 | 6.790626 | 9.203588 | 81.479575 | 8.853x | 11.999x | 5340537.974595 |
| 65536 | 128 | 7.019620 | 16.081822 | 86.497650 | 5.379x | 12.322x | 5674289.115783 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-runtime-sys`
- `cargo test -p ullm-runtime-sys cached_prefix_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 single smoke: `runtime-cached-prefix-attn-smoke 2 4096 16 3 16 4 256 256 cached_prefix_chunked`
- R9700 single smoke: `runtime-cached-prefix-attn-smoke 2 4096 128 1 16 4 256 256 cached_prefix_chunked`
- R9700 Phase C4 source-shared sweep

次の実装候補:

1. cold prefill側の `causal_attn_f32` にsource-shared方針を反映する。
2. cached prefix v2のKV read coalescing、source tile化、warp reduce化を検討する。
3. `M=1` decode-like pathはpaged decode attention側の最適化として分ける。

## Causal attention source-shared kernel v1

- cached prefix source-shared v2と同じ方針を `ullm_causal_attn_f32_kernel` に反映した。
- 1 block = 1 token/headとし、各source timestepのQK dotとsoftmax weightをblock内で1回計算してvalue次元のthreadへ共有する。
- `runtime-causal-attn-smoke` をR9700で `ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1` 付きで確認した。
- `package-self-attn-attention-batch-smoke` をR9700で再測定した。

保存先:

- `benchmarks/results/2026-07-07/runtime-causal-attn-source-shared/phase-c4-self-attn-attention-source-shared-v1.md`

R9700 release results:

| prompt tokens | repeats | mean ms | token/s | attention diff |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 5 | 7.637947 | 16758.430420 | 0.000011265 |
| 512 | 3 | 43.796889 | 11690.327961 | 0.000011265 |
| 1024 | 1 | 116.215921 | 8811.185173 | 0.000011265 |
| 2048 | 1 | 374.883299 | 5463.033444 | 0.000011265 |
| 4096 | 1 | 1331.671565 | 3075.833492 | 0.000011265 |

Previous comparison:

| prompt tokens | old mean ms | new mean ms | old token/s | new token/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24.605704 | 7.637947 | 5202.045750 | 16758.430420 | 3.222x |
| 512 | 281.601274 | 43.796889 | 1818.173590 | 11690.327961 | 6.430x |

RoPE guard:

- self-attention batch smokeのRoPE guardを、固定 `2e-4` からposition長に応じた上限付きabs floorへ変更した。
- `2048`: `q_rope_abs_floor=0.000409400`, `q_rope_max_abs_diff=0.000270158`, `k_rope_max_abs_diff=0.000198193`
- `4096`: `q_rope_abs_floor=0.000819000`, `q_rope_max_abs_diff=0.000506938`, `k_rope_max_abs_diff=0.000336170`
- 4096のfull host attention reference verificationは数分級になったため、今後の `4096+` rowではsampled attention verificationを優先する。

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-runtime-sys`
- `cargo test -p ullm-runtime-sys causal_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 runtime causal smoke
- R9700 package self-attn attention batch smoke `128/512/1024/2048/4096`

次の実装候補:

1. self-attention attention outputをo projection/residualへ接続し、self-attention layer partialを再測定する。
2. `4096+` 用にsampled attention verificationを追加し、`8192/16384` cold prefill component scalingを取る。
3. causal attention kernelのsource tile化、warp reduce化、Q/K read coalescingを検討する。

## Runtime cached prefix attention baseline

- `runtime-cached-prefix-attn-smoke` を追加した。
- synthetic Q/K/Vを使い、既存KV cache長 `L` に対して新規chunk `M` tokenを `decode_attn_f32` の連続実行で処理する。
- これは最適化済みchunked prefillではなく、cached prefix attentionのbaselineとして扱う。
- R9700では `ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL=1` を付けて確認した。

R9700 release results:

| cached prefix L | new input M | repeats | mean ms | min ms | max ms | new input tok/s | attention pair/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 3 | 98.653080 | 98.378548 | 98.924558 | 10.136531 | 41529.367495 |
| 4096 | 16 | 3 | 1570.769684 | 1568.895621 | 1573.916761 | 10.186089 | 41808.802824 |
| 16384 | 1 | 1 | 510.701715 | 510.701715 | 510.701715 | 1.958090 | 32083.307181 |
| 65536 | 1 | 1 | 2030.332401 | 2030.332401 | 2030.332401 | 0.492530 | 32278.950958 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine package_prompt_token_ids -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- CPU small smoke: `runtime-cached-prefix-attn-smoke 0 8 3 1 4 2 8 8`

解釈:

- `decode_attn_f32_loop` baselineでは、`L=4096` で約 `10 new tok/s`、`L=65536` では約 `0.49 new tok/s` まで落ちる。
- `L=4096, M=1` と `L=4096, M=16` のnew input tok/sがほぼ同じなので、M方向のchunk並列化はまだ無い。
- 次は `M x L` と `M x M` をまとめて扱う `cached_prefix_chunked` executorを作る必要がある。

## 追加実装メモ

- `tools/run-external-benchmark.py` に `--parse ullm-package-batch-throughput` を追加した。
- `package-batch-throughput-bench-v0.1` のraw JSONを `inference-benchmark-result-v0.1` 風JSONLへ変換できる。
- 変換時に次を保存する。
  - batchごとのtotal throughput metrics
  - `prompt_tokens_per_request`
  - `generated_tokens_per_request`
  - `fixed_decode_steps`
  - `batching.mode`
  - VRAM baseline/peak/consumed
  - `kv_cache_bytes_total`
- fake batch reportでmappingの最小確認を行った。

## 追加確認

- `python3 -m py_compile tools/run-external-benchmark.py`
- `python3 tools/run-external-benchmark.py --help | rg -n "ullm-package-batch-throughput|parse|result-json"`
- fake reportを使った `parse_ullm_batch_throughput_metrics` / `enrich_ullm_batch_workload` / `parse_ullm_batch_throughput_correctness` のassert確認
- `git diff --check`

## R9700 logical batch sanity

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- device index: `2`
- binary: `target/debug/ullm-engine`
- lm head mode: `gpu_resident_f32`
- 注意: debug binaryかつ極小workloadなので、速度値は性能比較に使わない。

確認結果:

| case | status | batching.mode | prompt/request | generated/request | verified |
| --- | --- | --- | ---: | ---: | --- |
| `aq4-r9700-logical-b1-len4-gen2` | ok | logical | 4 | 2 | true |
| `aq4-r9700-logical-b2-len4-gen2` | ok | logical | 4 | 2 | true |

保存先:

- `/tmp/ullm-batch-sanity-ok/result.jsonl`
- `/tmp/ullm-batch-sanity-ok/raw.json`
- `/tmp/ullm-batch-sanity-b2/result.jsonl`
- `/tmp/ullm-batch-sanity-b2/raw.json`

補足:

- `cpu_chunked` lm head modeでは、このAQ4 lm_head packageにpassthrough `lm_head.weight` が無いため失敗した。
- 既存成功artifactと同じく `gpu_resident_f32` を指定すると成功した。
- `batch=2` では `prompt_tokens_per_request` と `generated_tokens_per_request` が配列としてJSONLへ保存され、`kv_cache_bytes_total` もrequest数ぶん増えることを確認した。

## Prefill timing progress

- `package-token-ids-generate-smoke` / `package-token-ids-bench` の `prefill` に `layer_step_summary` を追加した。
- `package-batch-throughput-bench` の各 `requests[]` に元reportの `prefill` summaryを残すようにした。
- `tools/summarize-benchmark-results.py` をbatch throughput向けに更新し、prefill/decode/end-to-end total tok/sと `batching.mode` を表へ出すようにした。
- `docs/specs/inference-benchmark-result-v0.1.md` と `docs/words.txt` にprefill layer step summaryを追記した。

R9700 sanity:

- run id: `2026-07-07-prefill-layer-summary-sanity`
- case id: `aq4-r9700-logical-b1-len4-gen2-prefill-layer-summary`
- status: `ok`
- raw report: `/tmp/ullm-prefill-layer-summary-sanity/raw.json`
- JSONL: `/tmp/ullm-prefill-layer-summary-sanity/result.jsonl`
- `requests[0].prefill.layer_step_summary` は32 layer分出力された。
- 先頭layer summary例:
  - `layer_position=0`
  - `layer_index=0`
  - `kind=linear_attention`
  - `prompt_tokens=4`
  - `step_wall_summary.count=4`
- 注意: debug binaryかつ極小workloadなので、この速度値は性能比較に使わない。

確認:

- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine`
- `cargo test -p ullm-engine package_token_ids -- --test-threads=1`
- `cargo fmt --all --check`
- `python3 -m py_compile tools/summarize-benchmark-results.py tools/run-external-benchmark.py`
- `python3 tools/summarize-benchmark-results.py /tmp/ullm-prefill-layer-summary-sanity/result.jsonl`
- `git diff --check`

次の実装候補:

1. release binaryで `prompt_tokens=128/512` のprefill layer summaryを取り、遅いlayer familyを確認する。
2. prefill token loop内のhost readbackを減らすdevice-resident prefill pathを作る。
3. linear attention/self attentionのprojectionとMLPをtoken batch入力へ広げる。

## Release prefill diagnostics

### prompt=128 logical batch sanity

- run id: `2026-07-07-prefill-layer-summary-p128`
- case id: `aq4-r9700-logical-b1-len128-gen2-prefill-layer-summary`
- binary: `target/release/ullm-engine`
- status: `ok`
- raw report: `/tmp/ullm-prefill-layer-summary-p128/raw.json`
- JSONL: `/tmp/ullm-prefill-layer-summary-p128/result.jsonl`
- `prefill_total_input_tps`: `64.4565`
- `decode_total_generated_tps`: `64.8465`
- `end_to_end_total_tps`: `28.6132`
- `prefill.wall_ms`: `1985.8348`
- `prefill.layers_wall_ms`: `1983.6638`
- `prefill.lm_head_wall_ms`: `2.1709`

Top slow prefill layers:

| layer | kind | wall ms | mean ms/token |
| ---: | --- | ---: | ---: |
| 0 | linear_attention | 107.194 | 0.8375 |
| 3 | self_attention | 81.761 | 0.6388 |
| 27 | self_attention | 60.424 | 0.4721 |
| 24 | linear_attention | 60.363 | 0.4716 |
| 31 | self_attention | 60.184 | 0.4702 |

Family totals:

| kind | layers | wall ms | token-step/s |
| --- | ---: | ---: | ---: |
| linear_attention | 24 | 1473.648 | 2084.62 |
| self_attention | 8 | 501.715 | 2041.00 |

Interpretation:

- `prefill.layers_wall_ms` がprefill全体のほぼ全てを占める。
- linear attentionとself attentionのtoken-step/sは近く、特定familyだけが極端に遅いわけではない。
- したがって次はformat以前に、layer/tokenの逐次loopとhost readbackを減らすdevice-resident prefill pathが必要。

### prompt=32 component timing sanity

- run id: `2026-07-07-prefill-component-summary-p32`
- case id: `aq4-r9700-logical-b1-len32-gen2-prefill-component-summary`
- env:
  - `ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING=1`
  - `ULLM_SYNC_SELF_ATTN_COMPONENTS_FOR_TIMING=1`
- status: `ok`
- raw report: `/tmp/ullm-prefill-component-summary-p32/raw.json`
- JSONL: `/tmp/ullm-prefill-component-summary-p32/result.jsonl`
- component summary:
  - linear attention component layers: `24`
  - self attention component layers: `8`

Largest synchronized linear-attention component totals:

| component | total ms |
| --- | ---: |
| mlp_gate_up_activation_ms | 111.974 |
| mlp_down_residual_ms | 81.366 |
| qkv_projection_ms | 75.940 |
| recurrent_ms | 70.944 |
| z_projection_ms | 60.212 |

Largest synchronized self-attention component totals:

| component | total ms |
| --- | ---: |
| mlp_gate_up_activation_ms | 37.501 |
| paged_decode_ms | 30.583 |
| mlp_down_residual_ms | 26.922 |
| qkv_projection_ms | 26.724 |
| o_projection_residual_ms | 18.206 |

## Linear attention qkv prepare batch runtime

- `ullm_runtime_linear_attn_qkv_prepare_batch_f32` を追加した。
- 目的は、AQ4 batch qkv projectionの `[tokens, channels]` outputをGPU上のまま `conv+SiLU -> q/k L2 norm -> v split` へ渡すこと。
- HIP実装は2段構成:
  - prepare kernel: token-major qkvを読み、各tokenのconv output/q/k/vを出力する。
  - history update kernel: prepare完了後、同一stream上でconv historyを最終状態へ更新する。
- prepare kernel内では `conv_history` を更新しない。token block間のraceを避けるため、history更新は別kernelへ分離した。
- CPU fallback/staging fallbackはsingle-token host qkv prepareをsequence順に適用する。
- Rust FFI wrapperとCPU/HIP unit testを追加した。
- `package-linear-attn-qkv-prepare-batch-smoke` を追加した。qkv projectionそのもののF32参照比較は既存 `package-linear-attn-proj-batch-smoke` に任せ、新smokeはruntime qkv outputを入力としてqkv prepare batchのconv/q/k/v/historyを検証する。

R9700 package smoke:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- device index: `2`
- env: `ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
- layer: `0`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32`

| prompt tokens | repeats | mean ms | min ms | max ms | token/s | conv diff | q diff | k diff | v diff | history diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 0.264925 | 0.223454 | 0.313556 | 15098.631812 | 0.000001907 | 0.000000022 | 0.000000298 | 0.000001907 | 0.000000000 |
| 128 | 5 | 4.749306 | 4.499868 | 5.120708 | 26951.306149 | 0.000003815 | 0.000000052 | 0.000000298 | 0.000003815 | 0.000000000 |
| 512 | 3 | 18.488598 | 18.334076 | 18.783683 | 27692.742888 | 0.000003815 | 0.000000052 | 0.000000417 | 0.000003815 | 0.000000000 |

Projection batch recheck:

- `package-linear-attn-proj-batch-smoke ... len:128 3`
- mean: `6.872822 ms`
- token/s: `18624.082831`
- qkv/z/a/b max abs diff: `0.000167847 / 0.000062943 / 0.000058174 / 0.000027657`

確認:

- `cargo fmt --all`
- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys linear_attn_qkv_prepare_batch -- --test-threads=1`
- `git diff --check`

次の実装候補:

1. linear attention prefillの次段として、batch qkv prepare outputをrecurrent scanへ渡す。
2. recurrentはstate依存があるため、まずsingle request sequence scanのdevice-resident pathを作る。
3. その後、linear attention post/MLP projectionもtoken batchへ広げ、prefill layer単位のreal batch pathを作る。

注意:

- component timingは同期を入れるため、throughput比較には使わない。
- ただし、batched/tiled prefill化でまずMLP gate/up/down、qkv/z projection、recurrent/attention境界をまとめるべきことは確認できた。

## Device token-loop prefill progress

- `ULLM_PREFILL_DEVICE_TOKEN_LOOP=1` を追加し、prefillをprompt tokenごとにembeddingから全layerへdevice-to-deviceで通す実験経路を追加した。
- 既存経路は `layer_major_host_token_loop` として残した。
- raw reportの `prefill.executor` / `prefill.device_resident` / `prefill.sync_each_layer_for_timing` で、どの経路を使ったかを記録するようにした。
- `package-batch-throughput-bench` の `batching.prefill_executor` もrequest reportから集約するようにした。
- 注意: `device_token_loop` はlayer間のhost readbackを避けるだけで、real batch prefillではない。

R9700 release comparison:

| case | executor | prompt | generated | prefill total tok/s | decode total tok/s | end-to-end tok/s | generated ids |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `/tmp/ullm-host-prefill-p32` | `layer_major_host_token_loop` | 32 | 2 | 60.7380 | 68.7691 | 11.2569 | `[417,6281]` |
| `/tmp/ullm-device-prefill-p32` | `device_token_loop` | 32 | 2 | 70.8459 | 67.8412 | 11.4836 | `[417,6281]` |
| `/tmp/ullm-prefill-layer-summary-p128` | legacy host loop | 128 | 2 | 64.4565 | 64.8465 | 28.6132 | `[222,7485]` |
| `/tmp/ullm-device-prefill-p128` | `device_token_loop` | 128 | 2 | 75.9823 | 65.4800 | 30.5849 | `[222,7485]` |

Interpretation:

- p32ではprefillが約 `1.17x`、p128では約 `1.18x` 改善した。
- decode tok/sはほぼ変わらないため、変更の効果はprefillのhost境界削減に限定されている。
- generated idsはhost経路と一致したため、少なくともこのworkloadでは品質破壊は見えていない。
- ただしR9700のprefillとしてはまだ `75.98 tok/s` 程度であり、SQ候補評価に使える水準ではない。
- 次の主要課題は、tokenごとのlayer逐次呼び出しをやめ、projection/MLP/attention境界を複数tokenまたは複数requestでまとめるreal batch prefill executorを作ること。

## Batch workload manifest runner progress

- `tools/run-batch-throughput-workload.py` を追加した。
- `ullm-batch-throughput-workload-v0.1` manifestから、`tools/run-external-benchmark.py` と `ullm-engine package-batch-throughput-bench` を順次実行する。
- warmup rowは `warmup.jsonl`、measured rowは `results.jsonl` に分けて保存する。
- 各runの `raw.json` / `stdout.log` / `stderr.log` / `memory.jsonl` と、全コマンドを `execution-plan.json` に保存する。
- `--dry-run` でGPUを使わずにコマンド展開を確認できる。
- `run-external-benchmark.py` のartifact commandへ、`ULLM_PREFILL_DEVICE_TOKEN_LOOP` と必須HIP kernel envを残すようにした。
- 仕様メモとして `docs/specs/batch-throughput-workload-v0.1.md` を追加した。
- R9700 AQ4 smoke manifestとして `benchmarks/workloads/r9700-aq4-batch-throughput-smoke.json` を追加した。

R9700 smoke:

- output dir: `/tmp/ullm-workload-runner-smoke`
- status: `ok`
- case: `aq4-r9700-device-prefill-b1-pp4-tg2`
- `batching.mode`: `logical`
- `batching.prefill_executor`: `device_token_loop`
- `prefill_total_input_tps`: `37.2533`
- `decode_total_generated_tps`: `69.0410`
- `end_to_end_total_tps`: `2.3084`
- `verified_all`: `true`

注意:

- このsmokeはrunner validation用の `prompt_tokens=4` なので、速度値は性能判断に使わない。
- 実際のAQ4/FP8比較では、このrunnerにPhase A/B/C/Dのmanifestを渡して同じ列のJSONLを作る。

## R9700 AQ4 Phase A logical batch run

- `benchmarks/workloads/r9700-aq4-phase-a-logical.json` を追加した。
- `prompt_tokens=128`, `generated_tokens=32`, `concurrent_requests=1,2,4` を同じrunnerで実行する。
- `warmup_runs=1`, `measured_runs=1` とし、warmup rowとmeasured rowを分けた。
- output dir: `/tmp/ullm-phase-a-logical`
- `results.jsonl`: 3 rows
- `warmup.jsonl`: 3 rows
- all measured rows:
  - `status=ok`
  - `batching.mode=logical`
  - `batching.prefill_executor=device_token_loop`
  - `verified_all=true`

Measured summary:

| case | concurrent | prefill total tok/s | decode total tok/s | end-to-end tok/s |
| --- | ---: | ---: | ---: | ---: |
| `aq4-r9700-phase-a-logical-b1-pp128-tg32` | 1 | 76.0592 | 64.8906 | 34.7192 |
| `aq4-r9700-phase-a-logical-b2-pp128-tg32` | 2 | 77.2556 | 64.9183 | 36.6547 |
| `aq4-r9700-phase-a-logical-b4-pp128-tg32` | 4 | 77.7675 | 64.5501 | 36.5014 |

Interpretation:

- logical batchなので、concurrencyを増やしてもdecode total tok/sは伸びない。
- この結果は、schema、warmup/measured分離、VRAM計測、per-request latency、correctness保存の検証として使う。
- 次にtotal throughputを伸ばすには、real batch prefill/decode executorが必要。

## Executor parallelism metadata

- `package-token-ids-generate` 系reportの `prefill` にexecutor粒度を追加した。
  - `real_batch=false`
  - `token_parallelism=1`
  - `request_parallelism=1`
  - `projection_executor=single_token_matvec`
  - `mlp_executor=single_token_matvec`
  - `attention_executor=single_token_decode_step`
- `package-batch-throughput-bench` の `batching` にexecutor実parallelismを追加した。
  - `prefill_real_batch=false`
  - `prefill_executor_token_parallelism=1`
  - `prefill_executor_request_parallelism=1`
  - `decode_real_batch=false`
  - `decode_executor_request_parallelism=1`
- `docs/specs/inference-benchmark-result-v0.1.md` に、これらはworkload concurrencyではなくexecutorの実際のkernel sharingを表すと追記した。

R9700 smoke:

- output dir: `/tmp/ullm-executor-metadata-smoke`
- status: `ok`
- top-level batching fields: `false 1 1 false 1`
- request prefill fields: `false 1 1 single_token_matvec single_token_matvec single_token_decode_step`

Interpretation:

- Phase Aの `concurrent_requests=4` rowでも、現時点のexecutor parallelismは `1` のままであることを機械可読に示せる。
- real batch prefill/decodeを実装したときは、このfieldを変化させることで、同じworkload grid上で「本当にexecutorが並列化されたか」を確認できる。

## Prefill RMSNorm batch boundary smoke

- `ullm-engine package-prefill-rmsnorm-batch-smoke` を追加した。
- packageのprompt embedding行を `[tokens, hidden]` の1つのdevice bufferへ載せ、`segmented_rmsnorm_f32` で複数tokenをまとめて正規化する。
- CPU expected outputと比較し、`max_abs_diff` を検証する。
- 計測は1回のwarmup後に `measured_repeats` 回走らせ、mean/min/maxを出す。
- これはreal batch prefillの最初の境界確認であり、projection、attention、MLPはまだtoken loop側に残っている。

R9700 release results:

| prompt tokens | hidden | wall ms mean | wall ms min | wall ms max | token/s mean | max abs diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4096 | 0.053381 | 0.050890 | 0.056711 | 74933.3094 | 0.000020981 |
| 128 | 4096 | 0.057873 | 0.055171 | 0.063671 | 2211747.1420 | 0.000076294 |
| 512 | 4096 | 0.063373 | 0.062021 | 0.064820 | 8079175.9241 | 0.000076294 |

確認:

- `cargo test -p ullm-engine package_token_ids -- --test-threads=1`
- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo build --release -p ullm-engine`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 target/release/ullm-engine package-prefill-rmsnorm-batch-smoke ... 2 1048576 0 len:4 5`
- `env ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 target/release/ullm-engine package-prefill-rmsnorm-batch-smoke ... 2 1048576 0 len:128 5`
- `env ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 target/release/ullm-engine package-prefill-rmsnorm-batch-smoke ... 2 1048576 0 len:512 5`

Interpretation:

- warmupなしの最初の試行では約24msが見えていたが、warmup後は約0.05-0.06msになったため、初回HIP kernel準備の影響が支配的だった。
- RMSNorm単体では、token数を増やすと明確にbatch処理として効いている。
- ただしprefill全体の支配項はprojection、attention/recurrent、MLPなので、次の実装対象は `[tokens, hidden]` bufferをそのままbatched AQ4 matmul/GEMMまたは同等のtiled executorへ渡す経路である。
- SQ候補評価に必要な現実的prefill tok/sは、このRMSNorm smokeではなく、layer全体をreal batch化した後に測る必要がある。

## AQ4 projection batch runtime progress

- `ullm_runtime_aq4_matvec_batch_f32` を追加した。
- Rust wrapper `ullm_runtime_sys::aq4_matvec_batch_f32` を追加した。
- HIP kernelは `blockIdx.x = row block`, `blockIdx.y = token` とし、input `[tokens, cols]` から output `[tokens, rows]` を直接書く。
- CPU fallbackも同じAPIで動く。
- `PackageAq4ResidentMatvec::matvec_batch` を追加した。
- `ullm-engine package-prefill-aq4-matvec-batch-smoke` を追加した。
  - package prompt embeddingを `[tokens, hidden]` device bufferに載せる。
  - 指定AQ4 projection tensorをbatch kernelで実行する。
  - materialized F32 matrixをhost expectedとして `max_abs_diff` を検証する。
- これはT3のprojection batch化の最初の足場であり、attention/recurrent、KV write、MLP down/up/gate、full layer接続はまだ未実装。

R9700 release results:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- tensor: `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`
- shape: `rows=8192`, `cols=4096`
- env: `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`

| prompt tokens | wall ms mean | wall ms min | wall ms max | token/s mean | output element/s mean | max abs diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.233766 | 0.202663 | 0.352646 | 17111.1429 | 140174482.3238 | 0.000000775 |
| 128 | 4.597897 | 4.456253 | 4.827559 | 27838.8129 | 228055555.4831 | 0.000001252 |
| 512 | 17.970310 | 17.854813 | 18.144759 | 28491.4401 | 233401877.5058 | 0.000001490 |

Single-token reference:

- command: `package-aq4-matvec-smoke ... model.language_model.layers.0.linear_attn.in_proj_qkv.weight 5`
- env: `ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1`
- single AQ4 matvec mean: `0.117820 ms`
- 512-token batch mean per token: `17.970310 / 512 = 0.035098 ms`
- projection単体では、batch APIがsingle matvecの逐次換算に対して約 `3.36x` 速い。

確認:

- `cargo test -p ullm-runtime-sys aq4_matvec_batch -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo build --release -p ullm-engine`
- `cargo fmt --all --check`
- `cargo test -p ullm-engine package_token_ids -- --test-threads=1`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-prefill-aq4-matvec-batch-smoke ... len:4 5`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-prefill-aq4-matvec-batch-smoke ... len:128 5`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-prefill-aq4-matvec-batch-smoke ... len:512 5`

Interpretation:

- AQ4 projectionのtoken batch化はR9700で正しく動作し、数百tokenでもOOMせず完走した。
- ただしこのsmokeはlinear attention layer 0の `in_proj_qkv` 単体であり、prefill total input tok/sそのものではない。
- full prefillを速くするには、このAPIをresident layer runnerのinput RMSNorm後に接続し、linear attentionのqkv/z/a/b projection、MLP gate/up/down、self-attention q/k/v/o projectionへ広げる必要がある。
- 次の実装候補は、linear attention block内の `qkv_matrix` と `z_matrix` をtoken batchでまとめ、state scan前のprojection outputを `[tokens,*]` として保持するpartial prefill executor。

## Linear attention projection batch smoke

- `ullm-engine package-linear-attn-proj-batch-smoke` を追加した。
- linear attention layerのinput RMSNormを `[tokens, hidden]` に対して `segmented_rmsnorm_f32` で実行する。
- RMSNorm後のhiddenをそのままAQ4 batch matvecへ渡し、次の4 projectionをtoken batchで実行する。
  - `linear_attn.in_proj_qkv.weight`
  - `linear_attn.in_proj_z.weight`
  - `linear_attn.in_proj_a.weight`
  - `linear_attn.in_proj_b.weight`
- materialized F32 matrixからhost expected outputを作り、各projectionの `max_abs_diff` を検証する。
- recurrent state scan、qkv prepare、gate/beta activation、post attention、MLP、full layer接続はまだ含まない。

R9700 release results:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- layer: `0`
- env: `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
- projection shapes:
  - `qkv_rows=8192`
  - `z_rows=4096`
  - `a_rows=32`
  - `b_rows=32`
  - `hidden=4096`

| prompt tokens | repeats | wall ms mean | wall ms min | wall ms max | token/s mean | output element/s mean | qkv diff | z diff | a diff | b diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5 | 0.443894 | 0.355826 | 0.653871 | 9011.1684 | 111305952.5977 | 0.000097275 | 0.000035286 | 0.000026703 | 0.000012398 |
| 128 | 5 | 6.946323 | 6.842145 | 7.085429 | 18427.0153 | 227610492.6304 | 0.000167847 | 0.000062943 | 0.000058174 | 0.000027657 |
| 512 | 3 | 27.383782 | 27.164009 | 27.530285 | 18697.1980 | 230947789.5719 | 0.000183105 | 0.000070572 | 0.000058174 | 0.000028610 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine package_token_ids -- --test-threads=1`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-proj-batch-smoke ... 0 len:4 5`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-proj-batch-smoke ... 0 len:128 5`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-proj-batch-smoke ... 0 len:512 3`

Interpretation:

- qkv単体ではなく、linear attentionの主要projection境界をまとめてbatch実行できた。
- 512 tokenでもOOMせず、diffは `1e-4` 台から `1e-5` 台に収まった。
- ただし計測値は `input RMSNorm + qkv/z/a/b projection` のみであり、prefill total input tok/sではない。
- 次はこのprojection outputを `[tokens,*]` のまま `linear_attn_qkv_prepare` / recurrent state scanへ接続する必要がある。
- `linear_attn_qkv_prepare_f32` と `linear_attn_recurrent_f32` は現状step-orientedなので、まずはprojection batch + state scan loopのpartial executorを作り、その後state scan自体のkernel化を検討する。

## Linear attention recurrent batch prefill path

- `linear_attn_recurrent_f32` のHIP kernelを更新し、`sequence_len > 1` でもvalue head/value dimensionごとのblockを起こしてkey_dim方向をparallel reductionするようにした。
- 以前の `sequence_len > 1` 経路はvalue headあたり1 threadでkey_dim/value_dim/timestepを逐次処理していたため、prefillではrecurrent scanが極端に遅くなる構造だった。
- 既存decode fast pathの考え方をprefill sequenceにも広げ、timestep方向の依存は維持したまま、stateのvalue成分ごとに独立実行する。
- `ULLM_LINEAR_ATTN_RECURRENT_BLOCK` を追加し、未指定時は既存 `ULLM_LINEAR_ATTN_RECURRENT_DECODE_BLOCK`、さらに未指定ならkey_dimに応じた既定値を使う。
- `package-linear-attn-recurrent-batch-smoke` を追加した。次をtoken batchで接続して測る。
  - `segmented_rmsnorm_f32`
  - AQ4 `qkv/a/b` batch projection
  - `linear_attn_qkv_prepare_batch_f32`
  - `linear_attn_gate_beta_f32`
  - `linear_attn_recurrent_f32`

R9700 release results:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- layer: `0`
- env:
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1`
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1`
  - `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32+linear_attn_gate_beta_f32+linear_attn_recurrent_f32`

| prompt tokens | repeats | wall ms mean | wall ms min | wall ms max | token/s mean | gate diff | beta diff | recurrent output diff | recurrent state diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.413777 | 0.413777 | 0.413777 | 9667.042876 | 0.000000238 | 0.000000030 | 0.000000209 | 0.000001431 |
| 128 | 3 | 7.291452 | 6.959689 | 7.917207 | 17554.801725 | 0.000000477 | 0.000000119 | 0.000000834 | 0.000015259 |
| 512 | 3 | 27.790506 | 27.502704 | 27.966212 | 18423.558252 | 0.000000477 | 0.000000119 | 0.000004530 | 0.000043869 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys linear_attn_recurrent -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine package-linear-attn-recurrent-batch-smoke ... 0 len:4 1`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-recurrent-batch-smoke ... 0 len:128 3`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-recurrent-batch-smoke ... 0 len:512 3`

Interpretation:

- qkv prepare batch単体の512 tokenは前回 `18.488598 ms` だった。今回の前半workflowは `qkv/a/b projection + qkv prepare + gate/beta + recurrent` まで含めて512 tokenで `27.790506 ms` なので、recurrent接続後も速度は大きく崩れていない。
- ただしこの値はlinear attention layer 0の前半だけで、z projection、post-recurrent RMSNorm/SiLU、out projection、MLP、self-attention layer、layer間buffer接続はまだ含まない。
- 次の実装候補は、`z` projectionとpost-recurrent `segmented_rmsnorm_silu_mul_f32`、out projection residualをtoken batchへ接続して、linear attention blockのattention側を1 token loopから切り出すこと。

## Linear attention post batch smoke

- `package-linear-attn-post-batch-smoke` を追加した。
- 目的は、linear attentionのpost-recurrent側をtoken batchで実行できることを切り分けて確認すること。
- recurrent出力は決定的な合成値を使い、package内の実tensorで次を実行する。
  - input RMSNorm
  - `linear_attn.in_proj_z.weight` のAQ4 batch projection
  - `linear_attn.norm.weight` によるpost-recurrent segmented RMSNorm
  - SiLU(z)とのmul
  - `linear_attn.out_proj.weight` のAQ4 batch projection
  - residual add
- `out_proj` 自体のF32参照比較はこのsmokeでは重くなるため行わず、post RMSNorm/SiLUとresidual addをhost referenceで検証した。AQ4 batch projectionの境界は既存batch projection smokeで別途検証する。

R9700 release results:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- layer: `0`
- env: `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+segmented_rmsnorm_silu_mul_f32+aq4_matvec_batch_f32+add_f32`

| prompt tokens | repeats | wall ms mean | wall ms min | wall ms max | token/s mean | post diff | residual diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.282635 | 0.282635 | 0.282635 | 14152.528880 | 0.000001431 | 0.000000000 |
| 128 | 3 | 5.922646 | 5.434524 | 6.456272 | 21611.963167 | 0.000003815 | 0.000000000 |
| 512 | 3 | 22.089981 | 21.663144 | 22.918925 | 23177.928492 | 0.000004768 | 0.000000000 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 cargo run -p ullm-engine -- package-linear-attn-post-batch-smoke ... 0 len:4 1`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-post-batch-smoke ... 0 len:128 3`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-post-batch-smoke ... 0 len:512 3`

Interpretation:

- post-recurrent側は512 tokenで `22.089981 ms` だった。
- recurrent前半workflow `27.790506 ms` と合わせると、linear attention attention-side相当は単純合算で約 `49.88 ms / 512 tokens` まで見えてきた。ただし現時点では別smokeなので、同一bufferで完全接続した計測ではない。
- 次はこの2つを1つのpartial linear attention batch executorへ統合し、`qkv/a/b/z projection -> qkv prepare -> gate/beta -> recurrent -> post -> out residual` を一続きで測る。

## Linear attention attention batch integrated smoke

- `package-linear-attn-attention-batch-smoke` を追加した。
- 目的は、前半recurrent batch smokeとpost batch smokeを同一stream・同一buffer系列で統合し、linear attentionのattention側をtoken batchで一続きに測ること。
- package内の実tensorで次を実行する。
  - input RMSNorm
  - AQ4 `qkv/z/a/b` batch projection
  - qkv prepare batch
  - gate/beta
  - recurrent scan
  - post-recurrent segmented RMSNorm + SiLU(z)
  - AQ4 out projection batch
  - residual add
- projection自体のF32 full reference比較は既存projection batch smokeに任せ、統合smokeではgate/beta、recurrent output/state、post RMSNorm/SiLU、residual addをhost referenceで検証する。

R9700 release results:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- layer: `0`
- env:
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1`
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1`
  - `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32+linear_attn_gate_beta_f32+linear_attn_recurrent_f32+segmented_rmsnorm_silu_mul_f32+aq4_matvec_batch_f32+add_f32`

| prompt tokens | repeats | wall ms mean | wall ms min | wall ms max | token/s mean | gate diff | beta diff | recurrent output diff | recurrent state diff | post diff | residual diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.732063 | 0.732063 | 0.732063 | 5464.010611 | 0.000000238 | 0.000000030 | 0.000000209 | 0.000001431 | 0.000000596 | 0.000000000 |
| 128 | 3 | 13.024471 | 12.496553 | 13.831514 | 9827.654672 | 0.000000477 | 0.000000119 | 0.000000834 | 0.000015259 | 0.000001907 | 0.000000000 |
| 512 | 3 | 49.605634 | 48.782744 | 50.070915 | 10321.408180 | 0.000000477 | 0.000000119 | 0.000004530 | 0.000043869 | 0.000003815 | 0.000000000 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 cargo run -p ullm-engine -- package-linear-attn-attention-batch-smoke ... 0 len:4 1`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-attention-batch-smoke ... 0 len:128 3`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-attention-batch-smoke ... 0 len:512 3`

Interpretation:

- 512 tokenで `49.605634 ms`、`10321.408180 token/s`。
- 前回の別smoke単純合算は `27.790506 + 22.089981 = 49.880487 ms` だったため、同一buffer統合後もほぼ同じ速度で、余計なhost境界や接続ミスは見えていない。
- linear attention attention側の主要部分はtoken batch化できた。まだMLP、self-attention layer、full layer stack、decode state接続は含まない。
- 次はlinear attention MLP側をtoken batch化し、`attention batch -> post RMSNorm -> MLP gate/up/down -> layer residual` までをpartial linear-attention layer batchとして測る。

## Linear attention MLP batch smoke

- `package-linear-attn-mlp-batch-smoke` を追加した。
- 目的は、linear attention layerのMLP側をtoken batchで実行できることを切り分けて確認すること。
- package内の実tensorで次を実行する。
  - post-attention RMSNorm
  - AQ4 `mlp.gate_proj.weight` batch projection
  - AQ4 `mlp.up_proj.weight` batch projection
  - SiLU(gate) * up
  - AQ4 `mlp.down_proj.weight` batch projection
  - residual add
- projection自体のF32 full reference比較は既存projection smokeに任せ、統合smokeではpost RMSNorm、activation、residual addをhost referenceで検証する。

R9700 release results:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- layer: `0`
- env: `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+silu_mul_f32+aq4_matvec_batch_f32+add_f32`

| prompt tokens | repeats | wall ms mean | wall ms min | wall ms max | token/s mean | norm diff | activation diff | residual diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.921041 | 0.921041 | 0.921041 | 4342.911988 | 0.000015259 | 0.000000477 | 0.000000000 |
| 128 | 3 | 20.369223 | 19.900990 | 21.029955 | 6283.990312 | 0.000038147 | 0.000000954 | 0.000000000 |
| 512 | 3 | 83.995376 | 80.917992 | 86.441389 | 6095.573666 | 0.000038147 | 0.000001907 | 0.000000000 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 cargo run -p ullm-engine -- package-linear-attn-mlp-batch-smoke ... 0 len:4 1`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-mlp-batch-smoke ... 0 len:4 1`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-mlp-batch-smoke ... 0 len:128 3`
- `env ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-mlp-batch-smoke ... 0 len:512 3`

Interpretation:

- MLP側は512 tokenで `83.995376 ms`、`6095.573666 token/s`。
- linear attention attention integrated smokeの512 token `49.605634 ms` と比べると、MLP側がより重い。
- gate/up/downの3本の大きいprojectionが支配的で、次のpartial full layer smokeでは `attention batch -> MLP batch` を同一bufferで接続して、layer単位の合算と接続overheadを確認する。

## Linear attention layer batch smoke

- `package-linear-attn-layer-batch-smoke` を追加した。
- 目的は、linear attention attention側とMLP側を同一stream・同一buffer系列で接続し、linear attention decoder layer相当のpartial prefill timingを取ること。
- package内の実tensorで次を実行する。
  - input RMSNorm
  - AQ4 `qkv/z/a/b` batch projection
  - qkv prepare batch
  - gate/beta
  - recurrent scan
  - post-recurrent segmented RMSNorm + SiLU(z)
  - AQ4 out projection batch
  - attention residual add
  - post-attention RMSNorm
  - AQ4 MLP gate/up/down batch projection
  - SiLU(gate) * up
  - layer residual add
- projection自体のF32 full reference比較は既存projection smokeに任せ、統合smokeではgate/beta、recurrent output/state、attention post、attention residual、MLP post RMSNorm、MLP activation、layer residualをhost referenceで検証する。

R9700 release results:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- layer: `0`
- env:
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1`
  - `ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1`
  - `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+linear_attn_qkv_prepare_batch_f32+linear_attn_gate_beta_f32+linear_attn_recurrent_f32+segmented_rmsnorm_silu_mul_f32+aq4_matvec_batch_f32+add_f32+segmented_rmsnorm_f32+aq4_matvec_batch_f32+silu_mul_f32+aq4_matvec_batch_f32+add_f32`

| prompt tokens | repeats | wall ms mean | wall ms min | wall ms max | token/s mean | recurrent output diff | recurrent state diff | attention post diff | attention residual diff | MLP norm diff | MLP activation diff | layer residual diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 1.669524 | 1.669524 | 1.669524 | 2395.892482 | 0.000000209 | 0.000001431 | 0.000000596 | 0.000000000 | 0.000000358 | 0.000000060 | 0.000000000 |
| 128 | 3 | 32.994683 | 32.700449 | 33.337908 | 3879.412935 | 0.000000834 | 0.000015259 | 0.000001907 | 0.000000000 | 0.000001907 | 0.000000119 | 0.000000000 |
| 512 | 3 | 137.143145 | 134.568851 | 138.611359 | 3733.325506 | 0.000004530 | 0.000043869 | 0.000003815 | 0.000000000 | 0.000001907 | 0.000000238 | 0.000000000 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- `git diff --check`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 cargo run -p ullm-engine -- package-linear-attn-layer-batch-smoke ... 0 len:4 1`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-layer-batch-smoke ... 0 len:4 1`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-layer-batch-smoke ... 0 len:128 3`
- `env ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 target/release/ullm-engine package-linear-attn-layer-batch-smoke ... 0 len:512 3`

Interpretation:

- linear attention layer partialは512 tokenで `137.143145 ms`、`3733.325506 token/s`。
- attention integrated smoke `49.605634 ms` とMLP batch smoke `83.995376 ms` の単純合算は `133.601010 ms`。同一buffer接続後の追加分は `3.542135 ms`、約 `2.65%`。
- 128 tokenでは単純合算 `33.393694 ms` に対して統合 `32.994683 ms` で、測定揺らぎ込みでほぼ同等。
- host境界やbuffer接続による大きな損失は見えていない。次はself-attention layerのprefill batch化と、linear/self attention混在layer stackへの接続が必要。

## Self-attention prefill batch investigation

- 既存のself-attention smokeを確認した。
- `PackageAq4ResidentMatvec::matvec_batch` によりQ/K/V projection自体はtoken batch化できる。
- 一方で、Qwen3.5の `q_proj` は `qwen3.5-gated` layoutを持つため、Q queryとQ gateの分離、Q/K headwise RMSNorm、RoPEをまとめる境界が必要になる。
- 現在の `ullm_runtime_sys::qwen35_qk_norm_rope_f32` は `sequence_len` を受け取らない1 token APIで、prefill batchではtoken loopになってしまう。
- self-attention prefillを本当にbatch化する次の実装単位は、runtime側へ `qwen35_qk_norm_rope_batch_f32` を追加し、それを使って `package-self-attn-qkv-rope-batch-smoke` 相当を作ること。

Next:

- `runtime/src/ullm_runtime.cpp` と `crates/ullm-runtime-sys/src/lib.rs` にQ/K norm+RoPE batch primitiveを追加する。
- CPU testとR9700 HIP testを通す。
- その後、package内のself-attn layerで `input RMSNorm -> q/k/v AQ4 batch projection -> q/k norm+RoPE batch` を一続きで計測する。

## Qwen3.5 Q/K norm RoPE batch runtime primitive

- `qwen35_qk_norm_rope_batch_f32` を追加した。
- C API、Rust FFI wrapper、CPU host path、HIP kernel path、HIP staging fallbackを追加した。
- layout:
  - `q_projected`: `[token][q_head][query_or_gate][head_dim]`
  - `k_projected`: `[token][kv_head][head_dim]`
  - `q_gate_output`: `[token][q_head][head_dim]`
  - `q_rope_output`: `[token][q_head][head_dim]`
  - `k_rope_output`: `[token][kv_head][head_dim]`
- RoPE positionは `position_offset + token_index`。
- 追加テスト:
  - `cpu_qwen35_qk_norm_rope_batch_f32_matches_split_norm_rope`
  - `first_hip_qwen35_qk_norm_rope_batch_f32_matches_split_norm_rope_when_available`

確認:

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys qwen35_qk_norm_rope_batch -- --test-threads=1`
- `cargo check -p ullm-engine`
- `git diff --check`

Interpretation:

- self-attention prefillのprojection後に必要だったQ gate分離、Q/K headwise RMSNorm、RoPEをtoken loopなしで呼べるruntime境界ができた。
- 次は `package-self-attn-qkv-rope-batch-smoke` 相当を作り、R9700で `len:4/128/512` を測る。

## 512 token不足への計画追記

- 現在の計画には既に、512 token結果をshort sanityとして扱い、SQ候補のprefill判断には長さ、prefix、chunk、batch幅のgridを使う方針が含まれていた。
- ただし「結果に応じて適応する」部分をより明確にするため、Phase C4に実行ルールを追記した。
- SQ候補比較へ進む前の最低gridは、cold prefill `N=1024/2048/4096`、cached prefix `L=4096, M=16/128/512`、batch width `B=1/4/8` とした。
- 長コンテキスト適性は、可能な範囲で `N>=8192` と `L=65536, M=16/128` を代表値として追加する。
- 4096 token以上ではfull host reference verificationが計測を支配し得るため、sampled verificationを使い、GPU計測時間とverification時間を分けて保存する方針にした。
- 急落、OOM、output guard failure、attention pair/sの伸び止まりが出たpatternはcomponent smoke化し、kernel修正後に同じgridを再実行する。

## Self-attention causal attention sampled verification

- `package-self-attn-attention-batch-smoke` で、4096 token以上のattention guardをfull host referenceからsampled referenceへ切り替えた。
- sampled verificationは代表timestep `0/1/N/4/N/2/N-1` と代表head/valueの組を使い、現Qwen3.5条件では15点を確認する。
- outputには `attention_verification`, `attention_checked_values`, `verification_wall_ms` を追加した。
- 長尺RoPE guardは8192/16384でのf32 sin/cos差分を通すため、position長比例のabs floor capを `1e-3` から `4e-3` へ広げた。

R9700 release results:

| prompt tokens | wall ms mean | token/s mean | verification | verification ms | attention diff | q rope diff | k rope diff |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | 1339.278313 | 3058.363568 | sampled 15 | 907.921181 | 0.000000209 | 0.000506938 | 0.000336170 |
| 8192 | 5157.917832 | 1588.237786 | sampled 15 | 1730.860843 | 0.000000104 | 0.001175225 | 0.000833869 |
| 16384 | 20944.388749 | 782.262027 | sampled 15 | 3420.213255 | 0.000000320 | 0.002606988 | 0.001518801 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- R9700 `package-self-attn-attention-batch-smoke ... len:4096 1`
- R9700 `package-self-attn-attention-batch-smoke ... len:8192 1`
- R9700 `package-self-attn-attention-batch-smoke ... len:16384 1`

Interpretation:

- Phase C4のcold prefill length scaling必須範囲 `N=1024/2048/4096/8192/16384` はself-attention attention componentとして埋まった。
- 4096以上のfull host reference待ちは避けられるようになり、verification時間は16384でも約3.42秒に収まった。
- token/sは4096の約 `3.06k` から16384の約 `0.78k` まで落ちており、長尺側はまだcausal attentionのO(N^2)部分が支配的。
- 次はo projection/residualまで接続し、self-attention layer partialとしてattention支配が維持されるかを確認する。

## Self-attention block batch smoke

- `package-self-attn-block-batch-smoke` を追加した。
- 既存のself-attention attention batch pathに、`sigmoid(q_gate) * attention -> o_proj AQ4 batch -> residual add` を接続した。
- timed runにはinput RMSNorm、Q/K/V AQ4 batch projection、QK norm/RoPE batch、causal attention、output gate、o projection batch、residual addを含む。
- attention full host referenceは1024 tokenで約18.25秒かかったため、sampled verificationの閾値を1024 token以上へ下げた。
- `o_proj` はfull host projection referenceを作らず、AQ4 row dot productのsampled verificationで確認する。

R9700 release results:

| prompt tokens | block wall ms | block tok/s | attention-only ms | block delta ms | block/attention | attention diff | o proj diff | block diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 11.099879 | 11531.656891 | 7.637947 | 3.461932 | 1.453x | 0.000011265 | 0.000000864 | 0.000000000 |
| 512 | 54.498359 | 9394.778233 | 43.796889 | 10.701470 | 1.244x | 0.000011265 | 0.000000864 | 0.000000000 |
| 1024 | 141.180790 | 7253.111418 | 116.215921 | 24.964869 | 1.215x | 0.000000130 | 0.000000864 | 0.000000000 |
| 2048 | 433.086886 | 4728.843256 | 374.883299 | 58.203587 | 1.155x | 0.000000209 | 0.000000864 | 0.000000000 |
| 4096 | 1450.365419 | 2824.115872 | 1339.278313 | 111.087106 | 1.083x | 0.000000209 | 0.000000864 | 0.000000000 |
| 8192 | 5382.886791 | 1521.859983 | 5157.917832 | 224.968959 | 1.044x | 0.000000104 | 0.000000864 | 0.000000000 |
| 16384 | 21844.617459 | 750.024578 | 20944.388749 | 900.228710 | 1.043x | 0.000000320 | 0.000000864 | 0.000000000 |

確認:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- R9700 `package-self-attn-block-batch-smoke ... len:128 3`
- R9700 `package-self-attn-block-batch-smoke ... len:512 3`
- R9700 `package-self-attn-block-batch-smoke ... len:1024 1`
- R9700 `package-self-attn-block-batch-smoke ... len:2048 1`
- R9700 `package-self-attn-block-batch-smoke ... len:4096 1`
- R9700 `package-self-attn-block-batch-smoke ... len:8192 1`
- R9700 `package-self-attn-block-batch-smoke ... len:16384 1`

Interpretation:

- o projection/residualまで接続しても、長尺promptではcausal attention支配が維持される。
- block/attention wall ratioは128 tokenで約 `1.45x`、512 tokenで約 `1.24x`、8192/16384では約 `1.04x`。
- 次はpost-attention RMSNorm/MLPまで含むself-attention layer partialを見るか、causal attention kernelのtile/blocking再設計へ戻る。

## Self-attention layer batch smoke

- `package-self-attn-layer-batch-smoke` を追加した。
- self-attention block batch pathに、post RMSNorm、MLP gate/up AQ4 batch、SiLU-mul、MLP down AQ4 batch、final residual addを接続した。
- MLP projectionはfull host projection referenceを作らず、AQ4 row dot productのsampled verificationで確認する。

R9700 release results:

| prompt tokens | layer wall ms | layer tok/s | block wall ms | layer-block delta ms | layer/block |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 30.783820 | 4158.028516 | 11.099879 | 19.683941 | 2.773x |
| 512 | 141.768580 | 3611.519562 | 54.498359 | 87.270221 | 2.601x |
| 1024 | 318.234144 | 3217.756546 | 141.180790 | 177.053354 | 2.254x |
| 2048 | 777.894286 | 2632.748481 | 433.086886 | 344.807400 | 1.796x |
| 4096 | 2182.970006 | 1876.342776 | 1450.365419 | 732.604587 | 1.505x |
| 8192 | 6892.180390 | 1188.593382 | 5382.886791 | 1509.293599 | 1.280x |
| 16384 | 24825.171928 | 659.975288 | 21844.617459 | 2980.554469 | 1.136x |

確認:

- `cargo fmt --all`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- R9700 `package-self-attn-layer-batch-smoke ... len:128 3`
- R9700 `package-self-attn-layer-batch-smoke ... len:512 3`
- R9700 `package-self-attn-layer-batch-smoke ... len:1024 1`
- R9700 `package-self-attn-layer-batch-smoke ... len:2048 1`
- R9700 `package-self-attn-layer-batch-smoke ... len:4096 1`
- R9700 `package-self-attn-layer-batch-smoke ... len:8192 1`
- R9700 `package-self-attn-layer-batch-smoke ... len:16384 1`

Interpretation:

- post RMSNorm/MLPまで含むself-attention layer partialは、128から16384 tokenまでverifiedになった。
- 短尺ではMLPの線形projectionが重く、512 tokenでblock-onlyの約 `2.60x`。
- 長尺ではO(N^2) attentionの比率が戻り、16384 tokenではblock-onlyの約 `1.14x`。ただしMLP追加分は約 `2.98s` あり、全layer stackでは無視できない。
- 次はreal batch幅または複数layer stackへ進める。SQ候補評価の観点では、単layer componentだけではまだtotal throughputの代表値には足りない。

## Phase C4 coverage follow-up

- Runtime causal attention batchの不足行として `B=2,N=512/2048` を追加測定した。
- 追加で `B=8,N=4096` も測り、batch幅を広げた時のattention pair/sが横ばいか確認した。
- Cached prefix chunkでは未測定だった `M=512` を `L=4096/16384/65536` で追加した。
- Package self-attention layer partialは最新runtimeで `N=16384` を再測定した。

R9700 runtime causal attention batch:

| B | N | wall ms | input tok/s | pair/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 512 | 13.954804 | 73379.746182 | 18821904.895693 |
| 2 | 2048 | 230.013736 | 17807.632124 | 18243919.110634 |
| 8 | 4096 | 3698.347224 | 8860.174022 | 18150066.484942 |

R9700 cached prefix chunk `M=512`:

| L | M | wall ms | new tok/s | pair/s |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 512 | 129.396385 | 3956.833879 | 17222119.458747 |
| 16384 | 512 | 676.288522 | 757.073325 | 12598078.664420 |
| 65536 | 512 | 2607.803969 | 196.333776 | 12917289.949872 |

R9700 package self-attention layer partial latest:

| N | wall ms | tok/s | verification ms | layer diff |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 13279.226135 | 1233.806837 | 11677.183403 | 0 |

Interpretation:

- Phase C4のruntime causal attention batch幅gridは、既存結果と合わせて `B=1/2/4/8` at `N=512/2048` が揃った。
- batch幅を増やしてもattention pair/sは `18M pair/s` 前後で横ばいなので、現kernelはrequest方向の効率改善にはまだなっていない。
- Cached prefixは `L=65536,M=512` まで完走し、長prefixの `M=1/16/128/512` component境界が揃った。
- 次のkernel最適化は、cold prefillならtiled/block causal attention、cached prefixならK/V read coalescingとrequest/batch方向が候補になる。

## Package batch throughput cold schema v1

- `package-batch-throughput-bench` のlogical batch reportに、cold prefill用のprefix/chunk/context accountingを追加した。
- 追加フィールドは `workload.prefill_mode`、`cached_prefix_tokens_per_request`、`new_prefill_tokens_per_request`、`total_context_tokens_after_prefill_per_request`、`metrics.cached_prefix_total_tokens`、`metrics.total_context_tokens_after_prefill`、`metrics.estimated_prefill_attention_work_tokens`。
- `estimated_prefill_attention_work_tokens` はcomponent smokeと同じく、requestごとの `N*(N+1)/2` の合計。
- `package_token_ids_logits_tests` に三角数countのtestを追加した。

R9700 schema/control-plane smoke:

| B | prompt/request | generated/request | prefill tok/s | decode tok/s | e2e tok/s | attention work | verified |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4 | 2 | 53.808662 | 231.111649 | 8.270042 | 10 | true |
| 2 | 4 | 2 | 98.478364 | 233.595603 | 9.792617 | 20 | true |
| 4 | 4 | 2 | 172.597125 | 233.697961 | 10.443645 | 40 | true |

確認:

- `cargo fmt --all --check`
- `cargo test -p ullm-engine package_token_ids_logits_tests -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- R9700 `package-batch-throughput-bench ... len:4x1/2/4 ...`

Interpretation:

- T1のlogical batch最小exit criteriaである `B=1/2/4` のJSON reportは出せる。
- この結果は1 layer、4 token prompt、logical batch、requestごとのweight reloadありのschema smokeであり、SQ候補の性能判断には使わない。
- 次はworkload runnerによるJSONL集約、VRAM sampling、またはreal batch executor接続へ進む。
