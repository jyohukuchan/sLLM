# FP8 SQ R9700 batch throughput and prefill plan v0.1

## 前回の要点

- AQ4 decodeはR9700で `66-68 tok/s`、V620で約 `41 tok/s` まで改善し、AQ4 decode速度改善はいったん完了扱いにした。
- SQ候補評価では、single request decode tok/sだけでは不十分で、batch時のtotal throughput、prefill throughput、decode throughputを分けて測る必要がある。
- prefillは当初十分に最適化されていなかったため、素のprefill tok/sをSQ候補のformat性能として読むと判断を誤る状態だった。
- FP8はSQ候補1として扱う。ただし採用決定ではなく、SQ候補評価の基準線として使う。
- R9700/RDNA4向けcached-prefix attentionは、`cached_prefix_rdna4_fp8_auto` まで進み、SQ候補評価に使う暫定速度としては十分と判断する。

## 今回の変更点

- この計画では、初期実装と計測対象をR9700/RDNA4に限定する。
- V620/RDNA2は、FP8 native実行ではなくdequant経路が必要になる可能性が高く、今回の主旨から外す。
- batch処理、total token throughput計測、prefill最適化をSQ候補評価の前提作業として同時に進める。
- ある程度prefill/decodeが最適化できた段階で、vLLMで同等条件を動かした場合との比較を行う。
- vLLMのFP8対応は環境依存が強いため、R9700でunsupportedの場合も比較結果として明示的に記録する。
- vLLM比較では、R9700でFP8が動くことを前提にせず、backend、dtype、quantization、failure reasonを結果schemaへ残す。
- 512 tokenまでのcomponent timingだけでは長コンテキストprefillの評価として不足するため、cold prefillとcached prefix付きprefillのworkload gridを追加する。
- 512 tokenの結果はshort sanityとlocal bottleneck検出には使えるが、SQ候補のprefill性能判断には使わない。prompt長、cached prefix長、新規chunk長、batch幅を変えた探索gridを追加し、結果に合わせてprefill kernelのtiling/blocking方針を更新する。
- FlashAttention2-style cached-prefix最適化は、SQ候補評価へ進むための前提作業として一旦完了扱いにする。
- 次の主タスクは、`sq-fp8-w8a16-r9700-v0` のpackage/runtime prototype、result schema固定、AQ4 baselineとの比較準備である。
- `sq-fp8-w8a16-r9700-v0` のT2 guardでは、row scaleのsafe subsetが4層で崩れたため、row-block scaleとfallback policyを追加した。
- 現在の最有力な部分候補は、`v` fallback + `q/k/o/gate/up/down` row-block32 FP8である。
- この混合候補は layers `3,7,11,15` の短い3ケースと layers `3,7,11,15,19` のlen4 caseではstrict top1を維持した。
- 同じ混合候補は layers `3,7,11,15,19,23` と all self-attention probe layers `3,7,11,15,19,23,27,31` ではstrict top1を維持できなかった。
- layer `23` は境界層であり、`q/v` fallbackでlayer `23` 単体は回復するが、6層bundleではまだ累積driftが残る。
- 6層bundleのfamily splitでは、row-block32の `k` と `up` は単独でstrict top1を維持したが、`o/gate/down` は単独でtop1を動かした。
- row-block16でも `o/gate/down` はstrict top1に戻らなかった。
- `k/up` row-block32の6層部分候補は短い3 promptでstrict top1一致だったが、case_aのtop8 overlapは `2 / 8` と低い。これは回帰guardであり、full SQ policyではない。
- 追加の組み合わせ探索では、`k/up` 全6層に `o/gate/down` のうち1 familyまたは2 familyを layers `3,7,11,15,19` で足してもlen4 strict top1を維持した。
- `k/up` 全6層に `o/gate/down` 全3 familyを layers `3,7,11,15,19` で足すとlen4 strict top1が崩れた。
- len4上の次のprompt-bundle候補は、top8 overlapが最も高い `kup6_gate5_down5` である。
- `kup6_gate5_down5` はcase_a/case_bへ広げてもstrict top1を維持した。ただしcase_aのtop8 overlapは `2 / 8` と低く、full SQ policyではなく6層regression subsetとして扱う。
- `kup6_gate5_down5` の選択FP8 family/layerとfallback family/layerを `sq-fp8-policy-v0.1` として保存した。
- `tools/build-sq-fp8-w8a16-artifact.py` は `--policy-json` で `sq-fp8-policy-v0.1` を読み、policyのinclude regex、candidate ID、row-block scaleをartifact manifestへ反映できる。
- `kup6_gate5_down5` policyから実FP8 payload artifactを生成し、R9700で `sq-fp8-materialize-smoke` を通した。
- T1では、`.ullm.d` package pathからpackage-backed prefill component real-batch smokeを実行し、`inference-benchmark-result-v0.1` JSONLへ保存できるrunnerを追加した。
- したがって現時点では、T2は「品質境界をかなり狭めたがfull-target SQ guard未完了」と扱う。
- 一方で、prefill/cached-prefix attentionはSQ候補評価の前提速度としては一旦十分と判断する。追加のFlashAttention2-like最適化は有益だが、SQ策定フェーズを止めるblockerにはしない。
- 以後の主軸は、FP8 SQ候補の品質境界を固定し、real batch total throughputを測り、AQ4 baselineとvLLM参考値へ同じschemaで接続することに移す。

## 次の行動

1. T2 short guardの昇格条件は、text-level guardが実装・採用されるまではstrict top1一致にする。
2. `kup6_gate5_down5` を現在の6層strict-top1 regression subsetとして固定する。これはfull SQ policyではなく、次の候補探索の基準点として扱う。
3. 次の主タスクをSQ format evaluationに移す。追加のFlashAttention2-like最適化は、SQ候補比較で明確な阻害要因になったcaseだけに限定する。
4. SQ候補探索では、row-block幅、scale dtype/layout、FP8 scale有無、fallback family/layer、W8A16/W8A8を候補軸として保存し、quality guardとthroughput/memoryを同じresult schemaに流す。
5. `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json` を現在のpolicy representationとして使う。実payload artifactは `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` で確認済みだが、`/tmp` 配下なので必要時に `--policy-json` から再生成する。
6. `kup6_ogatedown5` はstrict-top1 failureかつnear-miss guardとして保存する。
7. top-k overlap、AQ4 top1 rank、logit gapは診断指標として保存するが、strict top1失敗を上書きしない。
8. cached-prefix測定では `cached_prefix_rdna4_fp8_auto` を暫定default executorにし、`resolved_executor` を必ず保存する。
9. full-package real batch runnerは最終比較には必要だが、SQ候補探索の開始blockerにはしない。候補探索中はcomponent、selected-layer stack、materialization-aware runtime pathの結果も、用途を明示して使う。
10. AQ4 baselineとFP8 SQ候補を同じworkload gridで比較し、uLLM側のR9700結果が揃った後にvLLM baselineを同じgridで測る。

## 2026-07-08 cached prefix FP8 KV cache note

前回の要点:

- cached prefix prefillは、既存KV cache長 `L` と新規chunk長 `M` を分けて測る必要がある。
- 現行のcached-prefix componentはF32 K/V cacheで速度を測っていた。
- SQ候補評価ではKV cache byte量とprefill tok/sを同時に見る必要がある。

今回の変更点:

- `runtime-cached-prefix-attn-smoke` に `kv_cache_dtype=fp8_e4m3|f32` を追加し、既定値を `fp8_e4m3` にした。
- runtime C APIとして `ullm_runtime_cached_prefix_attn_fp8_e4m3` を追加した。
- FP8 K/V cacheはper-tensor scale付きE4M3 byte列として保持し、QとoutputはF32のままにした。
- R9700で `L={4096,16384,65536}`、`M={16,128,512}` のcached prefix prefillをFP8/F32同一buildで比較した。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c4-fp8-e4m3-kv-cache-v1.md` に保存した。
- FP8はK/V cache byteをF32比 `25%` にできたが、現kernelでは速度が一様には改善しない。`L=16384` では `1.10-1.29x`、`L=4096` では `0.90-0.93x`、`L=65536` では `0.49-0.58x` だった。

次の行動:

1. package decode state / paged decodeのK/V cacheはまだF32なので、必要なら別タスクでFP8化する。
2. 長いprefixでFP8が遅い原因は、FP8復号がscore/value accumulation内で繰り返されることだと考える。
3. cached prefix attentionは、decoded K/V tileの再利用、score再計算削減、FlashAttention系のtilingを次の最適化候補にする。
4. SQ候補評価では、このv1 FP8 cached-prefix結果を「byte削減は確認済み、速度はkernel構造依存」という基準線として扱う。

## 2026-07-08 FP8 builtin conversion update

前回の要点:

- v1のFP8 K/V cacheはbyte量をF32比 `25%` にできたが、device側のFP8復号を手書きbit復号で行っていた。
- R9700/gfx1200にはFP8からF32へ変換する専用命令がある可能性があった。

今回の変更点:

- gfx1200向けに `__builtin_amdgcn_cvt_f32_fp8` を使い、HIPRTC生成ISAで `v_cvt_f32_fp8_e32` へ落ちることを確認した。
- scale込みbuiltinはROCm 7.2のgfx1200 targetでは `fp8-cvt-scale-insts` feature不足で使えないため、scaleは従来通り別のF32 multiplyにした。
- packed builtin `__builtin_amdgcn_cvt_pk_f32_fp8` も `v_cvt_pk_f32_fp8_e32` へ落ちるが、CKにgfx12 compiler issueの注意があるため今回のruntime kernelでは単体変換だけを使った。
- R9700再測結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c4-fp8-e4m3-kv-cache-builtin-cvt-v2.md` に保存した。
- 専用命令化によりFP8 pathはv1比で全条件改善した。`L=4096` はF32比 `0.98x` 付近、`L=16384` は `1.25-1.37x`、`L=65536` はまだ `0.51-0.74x` だった。

次の行動:

1. 長いprefixで残る遅さは、変換命令よりkernel構造の問題として扱う。
2. K score計算でdecoded Kを再利用するtile設計を検討する。
3. packed変換はCKのgfx12注意点を踏まえ、単体変換版の正しさと速度を基準にしてから別途検証する。

## 2026-07-08 prerequisite: FlashAttention2-style cached-prefix implementation

前回の要点:

- FP8 K/V cacheは専用変換命令により、R9700/RDNA4では変換そのもののoverheadをかなり削減できた。
- `L=4096` ではF32に近く、`L=16384` ではF32より速いが、`L=65536` ではまだF32より遅い。
- 残っている主因は、現cached-prefix kernelが長いprefixでK/Vをtile化せず、decoded K/Vやscoreを十分に再利用できていないことだと考える。

今回の変更点:

- SQ候補評価へ進む前提作業として、R9700/RDNA4向けFlashAttention2-style tiled attention実装を行う。
- 対象はまずcached prefix prefillとcold prefillのattention componentに限定する。
- R9700/RDNA4では、FP8変換命令、BF16/FP16系演算、十分なLDS/波面制御を前提に、そのまま実装できるものとして進める。
- RDNA2/V620などBF16が使えない、またはFP8変換・scale変換の条件が異なる環境は今回の合否から外す。
- RDNA2向けには、必要ならFP32へのdequant機構、別accumulator path、またはfallback attention kernelを後続で設計する。今回は保留する。

当初の行動:

1. `runtime-cached-prefix-attn-smoke` にFlashAttention2 executorを追加する。
   - 例: `cached_prefix_flash2` または `flash2_cached_prefix`
   - 既存の `cached_prefix_chunked` は比較baselineとして残す。
2. RDNA4向けに、Q block x K/V tileのonline softmax kernelを実装する。
   - K/VはFP8 E4M3 byte cacheから専用命令でF32またはBF16/FP16計算値へ変換する。
   - score max、denominator、weighted valueをtile単位で更新し、K score再計算を避ける。
   - `M`方向とhead方向の並列性を確保し、`L=65536`でdecode-likeに落ち込む問題を優先して見る。
3. correctness guardを既存のsampled referenceに接続する。
   - `L={4096,16384,65536}`、`M={16,128,512}` を最低限の保存gridにする。
   - `M=1` または `M=16` のdecode境界に近い条件はrepeat数を増やして外れ値を記録する。
4. FP8/F32の両方でFlashAttention2 executorを測る。
   - FP8専用最適化だけでなく、kernel構造改善そのものの効果をF32でも分離して見る。
   - 指標は `prefill_total_input_tps`、`attention_pair_tps_mean`、`cache_kv_bytes_total`、sampled diff。
5. 結果が安定したら、package self-attention prefillのattention componentへ接続する。
   - 最初はattention-only smokeでよい。
   - Q/K/V projection、QK norm/RoPE、o projection/MLPとの統合は後続に回す。

完了条件:

- R9700でFlashAttention2 executorがHIP kernel必須モードで動く。
- cached-prefix FP8 K/V cacheの保存gridでsampled guardが通る。
- v2 builtin conversion結果より `L=65536` のFP8 tok/sまたはpair/sが明確に改善する。
- F32 baselineに対して、kernel構造による改善とFP8 cache byte削減による改善を分けて説明できる。

現状:

- `cached_prefix_flash2`、`cached_prefix_flash2_fp8q`、`cached_prefix_rocwmma_fp8`、`cached_prefix_rdna4_fp8_auto` まで実装済み。
- `cached_prefix_rdna4_fp8_auto` は `M<64` を `cached_prefix_flash2_fp8q`、`M>=64` を `cached_prefix_rocwmma_fp8` に解決する。
- R9700 cached-prefix SQ評価では、このauto executorを暫定defaultとして使う。
- 追加のmulti-query-token tilingは残るが、SQ候補プロトタイプ着手を止める blocker ではない。

### 2026-07-08 progress: cached-prefix flash2 FP8/F32 v1

前回の要点:

- FlashAttention2-style tiled attentionを、まずR9700/RDNA4のcached-prefix FP8 K/V cache向けに実装する方針だった。
- 既存の `cached_prefix_chunked` は比較baselineとして残す方針だった。
- F32でも同じexecutor構造を測り、FP8 byte削減効果とkernel構造改善を分けて見る方針だった。

今回の変更点:

- `ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2` を追加した。
- `ullm_runtime_cached_prefix_attn_f32_flash2` も追加し、F32 KV cacheで同じFlashAttention2-style executorを測れるようにした。
- `runtime-cached-prefix-attn-smoke` に `cached_prefix_flash2` executorを追加した。
- `runtime-cached-prefix-attn-smoke` と `tools/run-runtime-cached-prefix-sweep.py` は、`cached_prefix_flash2` を `fp8_e4m3` と `f32` の両方で実行できる。
- HIPRTC kernel `ullm_cached_prefix_attn_fp8_e4m3_flash2_kernel` と `ullm_cached_prefix_attn_f32_flash2_kernel` は、64 token tileのscoreをshared memoryに置き、online softmaxでmax、denominator、weighted valueを更新する。
- R9700で `M=16` と `M=128` の代表gridを測定し、旧FP8 `cached_prefix_chunked` 比で `1.15x-1.50x` のtok/sを確認した。
- `M=512` でも旧FP8比 `1.23x-1.36x` の改善を確認した。
- F32 isolation sweepでは、旧F32 `cached_prefix_chunked` 比で `1.20x-1.49x` の改善を確認した。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c5-flash2-tiled-online-softmax-v1.md` に保存した。

次の行動:

1. 現v1はWMMA/MFMA未使用なので、QK/Vのmatmul構造をRDNA4向けに詰める。
2. cold prefill causal attention側にも同じtile online-softmax方針を展開する。
3. package self-attention prefillのattention componentに接続し、SQ候補評価用のprefill pathに近づける。

### 2026-07-08 progress: cold-prefill causal flash2 F32 v1

前回の要点:

- cached-prefix flash2はFP8/F32の両方で改善し、tile online-softmax構造自体の効果を確認した。
- 次はcold prefill causal attention側にも同じ方針を展開する段階だった。
- 既存のcold prefill kernelは `value_dim <= 256` ではすでにonline softmax 1-passなので、単純置換より別API/別executorで比較する方が安全だった。

今回の変更点:

- `ullm_runtime_causal_attn_f32_flash2` と `ullm_runtime_causal_attn_batch_f32_flash2` を追加した。
- Rust FFIに `causal_attn_f32_flash2` と `causal_attn_batch_f32_flash2` を追加した。
- `runtime-causal-attn-batch-smoke` に `EXECUTOR=causal_attn_batch_f32|default|flash2|causal_attn_batch_f32_flash2` を追加した。
- HIPRTC kernelは64 token tileのscoreをshared memoryに置き、online softmaxでmax、denominator、weighted valueを更新する。
- R9700の初期safe gridでは、旧 `causal_attn_batch_f32` 比で `1.19x-1.24x` のinput tok/s改善を確認した。
- 結果は `benchmarks/results/2026-07-08/runtime-causal-attn/phase-c6-causal-flash2-tiled-online-softmax-v1.md` に保存した。

次の行動:

1. `runtime-causal-attn-batch-smoke` の8MiB上限とは別に、長context prefill用の安全なbenchmark harnessを作る。
2. 複数query row/blockでK/V tileを再利用し、cold prefillでFlashAttention2に近い構造へ進める。
3. RDNA4向けMFMA/WMMA layoutを小さいprototypeで検証し、QK/V accumulationをscalar loopから置き換える。
4. package self-attention attention componentのexecutor選択にflash2を接続する。

### 2026-07-08 progress: RDNA4 FP8 WMMA probe v1

前回の要点:

- cached-prefix/cold-prefillのFlashAttention2-style v1は、tile online-softmaxで改善したが、QK/V accumulationはまだscalar loopだった。
- 次はRDNA4向けMFMA/WMMA layoutを小さいprototypeで検証し、attention本体へ組み込めるかを確認する段階だった。

今回の変更点:

- C ABI `ullm_runtime_wmma_fp8_probe`、Rust FFI `wmma_fp8_probe`、CLI `runtime-wmma-fp8-probe-smoke [DEVICE_INDEX]` を追加した。
- HIPRTC kernelから `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` を直接呼ぶ最小プローブを追加した。
- R9700/RDNA4 runtime device index `2` で、非0 markerが返ることを確認した。
- V620/RDNA2 runtime device index `1` では0 markerで失敗扱いになり、RDNA4 FP8 WMMA検証が誤って成功扱いにならないことを確認した。
- 結果は `benchmarks/results/2026-07-08/runtime-wmma/phase-c7-rdna4-fp8-wmma-probe-v1.md` に保存した。

次の行動:

1. このbuiltin呼び出しをattention score kernelへ直接入れるのではなく、まずQ/K tile layout、lane mapping、accumulator配置を小さいQK microkernelで固める。
2. cached-prefix flash2とcausal flash2のscalar dot部分を、RDNA4 FP8 WMMAまたはF32/BF16 MFMA相当のmicrokernelで置き換える候補を作る。
3. 置き換え後はtok/sだけではなくsampled diffを保存し、出力品質が壊滅的に崩れていないことを確認する。

### 2026-07-08 progress: RDNA4 FP8 WMMA QK probe v1

前回の要点:

- RDNA4ではHIPRTC kernelからFP8 WMMA builtinを直接呼べることを確認した。
- ただし前回のprobeはmarker確認であり、Q/K tileを入れてaccumulator値を見る段階には達していなかった。

今回の変更点:

- C ABI `ullm_runtime_wmma_fp8_qk_probe`、Rust FFI `wmma_fp8_qk_probe`、CLI `runtime-wmma-fp8-qk-probe-smoke [DEVICE_INDEX]` を追加した。
- 入力は16x16 FP8 E4M3 byte tileを2枚、出力は16x16 F32 accumulator tileに固定した。
- HIPRTC kernelはRDNA4/gfx12で `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` を使い、32 lane x 8 accumulatorを出力する。
- 初期sanityとしてQ/KをFP8 1.0相当の `0x38` で埋め、R9700 runtime device index `2` で `max_abs=16.0` を確認した。
- `layout` patternを追加し、非一様Q/K入力でraw accumulator previewを見られるようにした。`layout 256` では最大値が `374` になり、CPU row-major Q*K^Tの最大 `255` を超えるため、output順だけでなくA/B input register packingもrow-majorではないと判断した。
- V620/RDNA2とCPU CLIはこのRDNA4 QK probeでは失敗扱いにした。
- 結果は `benchmarks/results/2026-07-08/runtime-wmma/phase-c8-rdna4-fp8-wmma-qk-probe-v1.md` に保存した。

次の行動:

1. 任意Q/K tileでCPU row-major Q*K^Tと比較できるように、WMMA accumulatorのlane/register layoutを特定する。
2. accumulator layoutが確定したら、cached-prefix flash2のQK dot部分を小さい条件で置き換え、sampled diffを保存する。
3. その後、複数query row/blockでK/V tileを再利用する本命のFlashAttention2-like構造へ進める。

### 2026-07-08 progress: RDNA4 FP8 rocWMMA QK probe v1

前回の要点:

- direct builtinのFP8 WMMA QK probeでは、`ones` は期待どおり `16.0` になった。
- ただし `layout` patternではraw accumulator previewがCPU row-major Q*K^Tと一致せず、A/B input register packingとaccumulator orderの両方を自前で扱う必要がある状態だった。

今回の変更点:

- C ABI `ullm_runtime_rocwmma_fp8_qk_probe`、Rust FFI `rocwmma_fp8_qk_probe`、CLI `runtime-rocwmma-fp8-qk-probe-smoke [DEVICE_INDEX] [PATTERN=ones|layout] [PREVIEW_COUNT]` を追加した。
- HIPRTC kernelから `rocwmma::fragment`、`load_matrix_sync`、`mma_sync`、`store_matrix_sync` を使う16x16 FP8 QK probeを追加した。
- HIPRTC compile helperにrocWMMA include pathを追加できる経路を入れ、既存kernelのcompile optionは従来どおりに保った。
- R9700 runtime device index `2` で、`ones` は `max_abs=16.0`、`layout` はrow-major `0..255`、preview `0..63` を確認した。
- V620/RDNA2 runtime device index `1` ではRDNA4必須として拒否されることを確認した。
- 結果は `benchmarks/results/2026-07-08/runtime-wmma/phase-c9-rdna4-fp8-rocwmma-qk-probe-v1.md` に保存した。

次の行動:

1. direct builtinのraw layout解析を主経路から外し、RDNA4向けFlashAttention2-like実装ではrocWMMA fragment APIを第一候補にする。
2. まず16x16 QK tileを既存cached-prefix/cold-prefill flash2のQK dot部分へ小さい条件で組み込み、sampled diffとtok/sを測る。
3. その後、online softmaxとV accumulationを同じtile loopへ寄せ、FlashAttention2-likeの実装としてattention matrixを展開しない経路に育てる。

### 2026-07-08 progress: RDNA4 FP8 rocWMMA attention probe v1

前回の要点:

- rocWMMA QK probeで、16x16 FP8 Q*K^Tをrow-major出力として扱えることを確認した。
- 次はQKだけでなく、online softmaxとV accumulationまで接続できるかをstandalone smokeで確認する段階だった。

今回の変更点:

- C ABI `ullm_runtime_rocwmma_fp8_attn_probe`、Rust FFI `rocwmma_fp8_attn_probe`、CLI `runtime-rocwmma-fp8-attn-probe-smoke [DEVICE_INDEX] [PATTERN=ones|layout]` を追加した。
- 固定shapeはQ `16x16` FP8、K `32x16` FP8、V `32x16` F32、output `16x16` F32にした。
- HIPRTC kernelではrocWMMA QK tileを2回実行し、per-row online softmaxでVを畳み込む。
- R9700 runtime device index `2` で、`ones` はCPU参照diff `0`、`layout` はCPU参照diff `0.000000119` だった。
- V620/RDNA2 runtime device index `1` ではRDNA4必須として拒否されることを確認した。
- 結果は `benchmarks/results/2026-07-08/runtime-wmma/phase-c10-rdna4-fp8-rocwmma-attn-probe-v1.md` に保存した。

次の行動:

1. このstandalone attention probeをcached-prefix flash2のQK dot/softmax/V accumulationへ小さい条件で移植する。
2. 続いてcold-prefill causal flash2へ、causal maskと複数query row/blockの扱いを入れて移植する。
3. どちらもsampled diffを先に固定し、その後にtok/sを測ってtile sizeやblock割り当てを調整する。

### 2026-07-08 progress: RDNA4 FP8 rocWMMA cached-prefix v1

前回の要点:

- rocWMMA attention probeで、FP8 QK tile、online softmax、V accumulationをstandalone固定shapeで動かせることを確認した。
- 次はprobeではなく、`runtime-cached-prefix-attn-smoke` の計測経路へ接続する段階だった。

今回の変更点:

- C ABI `ullm_runtime_cached_prefix_attn_fp8_e4m3_rocwmma`、Rust FFI `cached_prefix_attn_fp8_e4m3_rocwmma`、CLI executor `cached_prefix_rocwmma_fp8` を追加した。
- HIPRTC kernel `ullm_cached_prefix_attn_fp8_e4m3_rocwmma_kernel` は、FP8 Q/K/V byte列、rocWMMA QK tile、online softmax、V accumulationを1 kernel内で処理する。
- 現時点の制約はR9700/RDNA4、`head_dim=16`、`value_dim=16`、`q_heads/kv_heads` が16の倍数であること。
- このexecutorだけQもFP8に量子化し、sampled referenceもdecoded FP8 Qを使うようにした。
- R9700 device index `2` で `L=65536,M=512,q_heads=16,kv_heads=1,dim=16` まで通し、`14.627084ms`、`input tok/s=35003.559151`、`sampled_max_abs_diff=0.000000415` を確認した。
- 追加で `q_heads=32,kv_heads=2` の複数KV head smokeも通し、strided K/V cache読み出しのsampled diffが `0.000000002` に収まることを確認した。
- 同じ `L=4096,M=16,dim=16` では、`cached_prefix_rocwmma_fp8` が `0.911433ms`、既存 `cached_prefix_flash2 fp8_e4m3` が `3.252488ms`、旧 `cached_prefix_chunked fp8_e4m3` が `3.778853ms` だった。
- V620/RDNA2 device index `1` ではRDNA4必須として拒否されることを確認した。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c11-rdna4-fp8-rocwmma-cached-prefix-v1.md` に保存した。

次の行動:

1. `head_dim=16` 固定を外す前に、Q row groupingとK/V tile再利用の設計を整理する。
2. 実モデルのhead_dimに近いshapeへ拡張し、QをFP8にする影響とkernel構造改善を分離して測る。
3. cold-prefill causal attention側へ、causal mask込みのrocWMMA tile方針を展開する。

### 2026-07-08 progress: RDNA4 FP8 rocWMMA cached-prefix 16n dimensions v1

前回の要点:

- `cached_prefix_rocwmma_fp8` はRDNA4上で動いたが、`head_dim=16,value_dim=16` 固定だった。
- 実モデルのattention headに近づけるには、少なくとも16の倍数dimensionを受けられる必要があった。

今回の変更点:

- `cached_prefix_rocwmma_fp8` の制約を `head_dim=16,value_dim=16` から `head_dim` と `value_dim` が16の倍数へ広げた。
- QKは `head_dim` 方向を16ずつrocWMMAで累積する。
- `value_dim` は16列ずつ別blockに分け、`[new token, KV head, Q-head group, value tile]` のgridで出力する。
- Rust FFI、CLI、sweep toolにも同じ16倍数制約を追加した。
- R9700で `q_heads=32,kv_heads=2,head_dim=32,value_dim=32` と、`q_heads=16,kv_heads=1,head_dim=256,value_dim=256` のsmokeを通した。
- 256次元では `cached_prefix_rocwmma_fp8` が `17.222257ms`、既存 `cached_prefix_flash2 fp8_e4m3` が `3.952818ms` だった。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c12-rdna4-fp8-rocwmma-dim16n-v1.md` に保存した。

観察:

- 16倍数dimensionの正しさは確認できた。
- ただし、現在のvalue tile分割はtileごとにQKとonline softmaxを再計算するため、`value_dim=256` では既存flash2より遅い。
- 次の最適化は、value tile並列を残しつつQK/softmax再計算を避けるblock設計、または複数laneでfull value accumulationを分担する設計に進む必要がある。

次の行動:

1. QK/softmaxをvalue tile間で再利用できるblock設計を検討する。
2. `head_dim=256,value_dim=256` で既存flash2を超えることを次の性能gateにする。
3. その後、cold-prefill causal attention側へ同じrocWMMA tile方針を展開する。

### 2026-07-08 progress: RDNA4 FP8 rocWMMA cached-prefix value group 64 v1

前回の要点:

- `cached_prefix_rocwmma_fp8` は16倍数dimensionに対応した。
- ただし `value_dim=256` では16列value tileごとにQK/online softmaxを再計算するため、既存 `cached_prefix_flash2 fp8_e4m3` より遅かった。

今回の変更点:

- 1 blockが64列のvalue groupを担当するように変更した。
- `value_dim=256` では、QK/online softmaxの再計算回数が16回から4回に減る。
- full-value dynamic shared accumulator案も試したが、block並列性を失ってさらに遅くなったため採用しなかった。
- R9700で `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256` を測定し、`17.222257ms` から `15.438269ms` へ改善した。
- ただし既存 `cached_prefix_flash2 fp8_e4m3` の `3.952818ms` にはまだ届いていない。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c13-rdna4-fp8-rocwmma-value-group64-v1.md` に保存した。

観察:

- 64列groupは、32列groupより長prefixで速く、128列groupやfull-value groupよりも速かった。
- 現時点の律速は、QK/softmax再計算がまだ4回残ることと、V accumulationが十分に効率化されていないこと。
- rocWMMAでQKだけ高速化しても、`value_dim=256` のV側をうまく並列化できないと既存scalar flash2に勝てない。

次の行動:

1. QK/softmaxを1回だけ計算しつつ、V accumulationのblock並列性を保つ2-stageまたはcooperative-group構造を検討する。
2. その設計が重い場合は、先にcold-prefill causal attentionへrocWMMA QKを限定導入し、prefill側で効果が出るshapeを探す。
3. `value_dim=256` のcached-prefixでは、既存flash2超えを引き続き性能gateにする。

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

### Cold prefill and cached prefix prefill

長コンテキストでは、単純なprompt長だけではworkloadを説明できない。
この計画ではprefill系workloadを次の3種類に分けて保存する。

- cold prefill:
  - KV cacheが空の状態から、`N` tokenのpromptを一度に処理する。
  - attention workはおおむね `N^2 / 2` に比例する。
- cached prefix prefill:
  - 既に `L` token分のKV cacheが存在する状態で、新規input chunk `M` tokenを追加する。
  - attention workはおおむね `M * L + M^2 / 2` に比例する。
  - 速度指標の分母は新規input token数 `M` とする。ただし、比較解釈のために `cached_prefix_tokens=L` も必ず保存する。
- decode:
  - `L` token分のKV cacheに対して、新規token `M=1` を生成する。
  - これはdecode throughputとして扱い、cached prefix prefillとは別に保存する。

実ワークロードでは、長コンテキストでも毎回 `N=2^16` tokenをcold prefillするケースは少ない。
しかし、cached prefixが長い場合でも `M * L` のattention costは残るため、SQ候補評価では `N` だけでなく `L` と `M` を分けて測る。

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
    "prefill_mode": "cold|cached_prefix|decode",
    "prompt_tokens_per_request": [512, 512, 512, 512, 512, 512, 512, 512],
    "cached_prefix_tokens_per_request": [0, 0, 0, 0, 0, 0, 0, 0],
    "new_prefill_tokens_per_request": [512, 512, 512, 512, 512, 512, 512, 512],
    "total_context_tokens_after_prefill_per_request": [512, 512, 512, 512, 512, 512, 512, 512],
    "generated_tokens_per_request": [128, 128, 128, 128, 128, 128, 128, 128],
    "fixed_decode_steps": true
  },
  "metrics": {
    "prefill_total_input_tokens": 4096,
    "cached_prefix_total_tokens": 0,
    "total_context_tokens_after_prefill": 4096,
    "estimated_prefill_attention_work_tokens": 1048576,
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
    "prefill_executor": "token_loop|chunked|batched_gemm|cached_prefix_chunked",
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

### Phase C: cold prefill pressure

| concurrent requests | cold prompt tokens/request | generated tokens/request | purpose |
| ---: | ---: | ---: | --- |
| 1 | 2048 | 64 | long prompt single request |
| 4 | 2048 | 64 | batched prefill pressure |
| 8 | 2048 | 64 | high prefill total throughput |
| 1 | 4096 | 32 | medium-long cold prefill scaling |
| 1 | 8192 | 16 | cold prefill attention pressure |
| 1 | 16384 | 8 | long context cold prefill scaling |

### Phase C2: long context upper bound

| concurrent requests | cold prompt tokens/request | generated tokens/request | purpose |
| ---: | ---: | ---: | --- |
| 1 | 32768 | 1 | high context cold prefill upper bound |
| 1 | 65536 | 1 | `2^16` cold prefill upper bound |

Phase C2は、full modelの常用benchmarkではなく、長コンテキスト上限の実行可能性とscaling確認である。
OOMや極端に低速なrunを避けるため、component smoke、single layer smoke、またはchunked full prefillの順に段階化し、per-token latencyが安定したら代表1回で止めてよい。

### Phase C3: cached prefix prefill

| concurrent requests | cached prefix tokens/request | new input tokens/request | generated tokens/request | purpose |
| ---: | ---: | ---: | ---: | --- |
| 1 | 4096 | 16 | 16 | small chunk over medium prefix |
| 1 | 4096 | 128 | 16 | common extension over medium prefix |
| 1 | 4096 | 512 | 16 | large chunk over medium prefix |
| 1 | 16384 | 16 | 8 | small chunk over long prefix |
| 1 | 16384 | 128 | 8 | common extension over long prefix |
| 1 | 16384 | 512 | 8 | large chunk over long prefix |
| 1 | 65536 | 1 | 8 | decode-like boundary over `2^16` prefix |
| 1 | 65536 | 16 | 4 | small cached prefill over `2^16` prefix |
| 1 | 65536 | 128 | 4 | main cached prefill over `2^16` prefix |
| 1 | 65536 | 512 | 4 | large cached prefill over `2^16` prefix |

Phase C3では、速度表に必ず `cached_prefix_tokens`, `new_prefill_tokens`, `total_context_tokens_after_prefill`, `estimated_prefill_attention_work_tokens` を併記する。
`prefill_total_input_tps` は新規input token `M` だけで割るため、`L` が異なるcaseを単純比較しない。

### Phase C4: prefill pattern exploration and adaptation

Phase Bの512 token結果は、short sanity、warmup確認、局所的なcomponent regression検出に限定する。
SQ候補のprefill性能判断では、少なくとも次のpattern familyを測り、prompt長、prefix長、chunk長、batch幅に対するscalingを見る。

| family | required sweep | optional extension | optimization decision |
| --- | --- | --- | --- |
| cold prefill length scaling | `N=1024/2048/4096/8192/16384` | `N=32768/65536` | causal attentionのtiling/blocking、streamed prefillの必要性を決める |
| cached prefix chunk scaling | `L=4096/16384/65536`, `M=1/16/128/512` | `M=4/64/1024` | `M` 方向の並列化、score共有、KV read patternを決める |
| batch width scaling | `B=1/2/4/8` at `N=512/2048` | `B=16` if VRAM allows | real batch化の効果とoccupancy不足を分ける |
| mixed realistic prompt | prompt mix `128,512,2048,8192` | cached prefix mixを含める | scheduler/result schemaが平均値で問題を隠していないかを見る |
| component isolation | projection+RoPE, attention-only, MLP-only, full-layer partial | full layer stack | format差ではなくexecutor差で詰まっている箇所を分ける |

512 token不足への対応ルール:

- 512 tokenだけの結果では、prefill kernelやSQ候補の採用判断を行わない。
- SQ候補比較へ進む前に、最低でも `N=1024/2048/4096` のcold prefill component scaling、`L=4096, M=16/128/512` のcached prefix chunk scaling、`B=1/4/8` のbatch width scalingを保存する。
- 長コンテキスト適性を見るため、OOMまたは極端な低速がない範囲で `N>=8192` のcold prefill代表値と `L=65536, M=16/128` のcached prefix代表値を追加する。
- 4096 token以上でfull host reference verificationが支配的になる場合は、sampled verificationを使い、GPU計測時間とverification時間を分けて保存する。
- どれかのpattern familyで急落、OOM、output guard failure、またはattention pair/sの伸び止まりが出た場合は、そのpatternを再現するcomponent smokeを追加し、kernel修正後に同じgridを再実行する。
- 新しいkernel方針は、単一の512 token改善ではなく、少なくとも2つ以上の長さまたはprefix/chunk条件で改善が確認できた場合に次段階へ進める。

各familyで保存する最小項目:

- `prefill_total_input_tps`
- `wall_ms_mean`, `wall_ms_min`, `wall_ms_max`
- `estimated_prefill_attention_work_tokens`
- `attention_pair_tps` when attention work is isolated
- `resident_bytes`, `materialized_working_set_bytes`, `vram_peak_bytes`
- warmup回数、measured repeat数、長時間runを短縮した理由
- output guardまたはreference diff

適応ルール:

- `N` を増やした時にtoken/sが急落し、projection/MLP単体が維持される場合は、causal attention prefill kernelを先にtiled/blocked化する。
- `M=1` は遅いが `M=16/128/512` が伸びる場合は、decode-like pathとcached prefill pathを分けて扱う。
- `L=65536` で `M>1` でもattention pair/sが低い場合は、cached prefix attentionのscore計算共有とKV read coalescingを優先する。
- `B` を増やしてもtotal throughputが伸びない場合は、schedulerではなくruntime kernel側のbatch入力形状を疑う。
- あるpatternだけoutput guardが崩れる場合は、速度比較に進まず、そのpatternを再現する最小component smokeを追加する。

### Phase C4 current status: cold causal attention real batch primitive

`runtime-causal-attn-batch-smoke` を追加し、cold causal self-attention componentでreal batch入力を測れるようにした。

- q/k/v/output layoutは `[batch, sequence, head, dim]` とする。
- `ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL=1` でstaging fallbackを禁止し、R9700上のHIP kernel経路を確認する。
- 指標は `prefill_total_input_tps` と `attention_pair_tps_mean` を同時に保存する。
- verificationは長いsequenceでもfull output readbackを避けるためsampled guardを使う。

R9700 release代表値:

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

- `benchmarks/results/2026-07-07/runtime-causal-attn-batch/phase-c4-cold-prefill-batch-v1.md`

解釈:

- `N=512/2048` の `B=1/4/8` は保存済みで、Phase C4のbatch width component gapは一部埋まった。
- ただしbatch幅を増やしてもattention pair/sはほぼ横ばいで、wall timeはbatch数に近く比例して伸びる。
- これはscheduler/control planeではなく、runtime causal attention kernel自体がまだ十分にbatched efficiencyを出せていないことを示す。
- 次のprefill最適化は、full modelへ広げる前に、causal attention prefill kernelのscore reuse、tiled/block化、K/V read pattern改善を優先する。

### Phase C4 current status: causal attention online softmax v1

`ullm_causal_attn_f32_kernel` と `ullm_causal_attn_batch_f32_kernel` にonline softmax pathを追加した。
通常のQwen3.5 shapeでは `value_dim=256` かつ `blockDim.x=256` なので、このpathに入り、q/k score dot-productを従来の3passから1passへ減らす。
`value_dim > blockDim.x` の場合は既存の3pass pathをfallbackとして残す。

保存先:

- `benchmarks/results/2026-07-07/runtime-causal-attn-batch/phase-c4-cold-prefill-online-softmax-v1.md`

R9700 release代表値:

| component | condition | old mean ms | new mean ms | speedup |
| --- | --- | ---: | ---: | ---: |
| runtime causal attention batch | `B=1, N=512` | 18.603645 | 7.239056 | 2.570x |
| runtime causal attention batch | `B=8, N=2048` | 2208.166702 | 908.260240 | 2.431x |
| runtime causal attention batch | `B=4, N=4096` | 4649.452562 | 1860.373915 | 2.499x |
| package self-attention attention batch | `N=512` | 281.601274 | 33.528531 | 8.399x |
| package self-attention layer batch | `N=4096` | 2182.970006 | 1518.104339 | 1.438x |
| package self-attention layer batch | `N=8192` | 6892.180390 | 4162.250951 | 1.656x |

解釈:

- runtime attention pair/sはおおむね `7.2-7.7M pair/s` から `17.8-18.9M pair/s` へ改善した。
- self-attention attention componentでは大きく効くが、layer全体ではprojection/MLPの比率が残るため改善幅はcontext長に依存する。
- `B` を増やしてもsuperlinearなthroughput増加は出ていない。今回の改善はbatch幅の効率化ではなく、query/headごとの重複score計算削減である。
- 次のkernel最適化は、neighboring timestep/head間でK/Vを再利用するtiled/block causal attention、またはprojection/MLP側の残コスト削減を比較して選ぶ。

### Phase C4 current status: cached prefix online softmax v1

`ullm_cached_prefix_attn_f32_kernel` にonline softmax pathを追加した。
通常shapeでは `value_dim=256` かつ `blockDim.x=256` なので、このpathに入り、cached prefix attentionでもq/k score dot-productを3passから1passへ減らす。
`value_dim > blockDim.x` の場合は既存の3pass pathをfallbackとして残す。

保存先:

- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-online-softmax-v1.md`

R9700 release代表値:

| L | M | old mean ms | new mean ms | speedup | new input tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 55.839281 | 4.193457 | 13.316x | 3815.467763 |
| 4096 | 128 | 435.542576 | 30.812317 | 14.135x | 4154.182887 |
| 16384 | 16 | 303.245627 | 19.256383 | 15.748x | 830.893320 |
| 16384 | 128 | 1748.356354 | 170.666624 | 10.244x | 750.000188 |
| 65536 | 16 | 1738.452511 | 78.523667 | 22.139x | 203.760224 |
| 65536 | 128 | 7959.297024 | 673.121102 | 11.824x | 190.158947 |

解釈:

- cached prefix componentは前回のshared-score kernel比で `10.2-22.1x` 改善した。
- `M=16/128` では `12.3-17.3M pair/s` 程度になり、cold causal attention online-softmax componentに近い範囲へ入った。
- `M=1` はdecode-like boundaryで並列性が少なく、pair/sはまだ低い。
- `L=65536, M=128` が `673.121102 ms` まで短縮されたため、SQ候補比較用の長prefix代表runとして繰り返し測りやすくなった。
- 次のcached prefix最適化は、score再計算ではなく、request/batch方向、`M=1`境界、K/V read coalescingを優先する。

### Phase C4 current status: decode-like M=1 executor split

`ullm_decode_attn_f32_kernel` にhead-parallel online-softmax pathを追加した。
通常shapeでは `head_dim=256` かつ `value_dim=256` なので、`1 block = 1 q_head` としてhead_dim reductionとvalue lane計算を同じblockで実行する。
旧element-parallel pathは診断用に残し、`ULLM_DISABLE_DECODE_ATTN_HEAD_PARALLEL=1` で強制できる。

保存先:

- `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-decode-loop-head-parallel-v1.md`

R9700 release代表値:

| L | old decode-loop ms | head-parallel decode-loop ms | speedup | chunked online ms |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 103.281946 | 3.488255 | 29.608x | 4.039271 |
| 16384 | 522.266906 | 16.385404 | 31.874x | 18.882177 |
| 65536 | 2035.323208 | 66.730666 | 30.501x | 76.723657 |

実行方針:

- `M=1` のdecode-like boundaryでは、`decode_attn_f32_loop` を使う。
- `M>=16` のcached prefix chunkでは、tokenごとにdecode kernelをlaunchするdecode loopではなく、`cached_prefix_chunked` を使う。
- このsplitにより、Phase C4の `L=65536, M=1/16/128` を全て現実的な時間で繰り返し測れる。

### Phase C4 current status: batch width and long-context coverage v1

C4の不足coverageとして、runtime causal attention batchの `B=2`、cached prefix chunkの `M=512`、package self-attention layer partialの最新 `N=16384` を追加測定した。

保存先:

- `benchmarks/results/2026-07-07/phase-c4-coverage/batch-width-and-long-context-v1.md`

Runtime causal attention batch追加行:

| B | N | mean ms | input tok/s | attention pair/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 512 | 13.954804 | 73379.746182 | 18821904.895693 |
| 2 | 2048 | 230.013736 | 17807.632124 | 18243919.110634 |
| 8 | 4096 | 3698.347224 | 8860.174022 | 18150066.484942 |

Cached prefix chunk `M=512` 追加行:

| L | M | mean ms | new input tok/s | attention pair/s |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 512 | 129.396385 | 3956.833879 | 17222119.458747 |
| 16384 | 512 | 676.288522 | 757.073325 | 12598078.664420 |
| 65536 | 512 | 2607.803969 | 196.333776 | 12917289.949872 |

Package self-attention layer partial latest:

| N | mean ms | token/s | verification ms | layer diff |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 13279.226135 | 1233.806837 | 11677.183403 | 0 |

解釈:

- Phase C4のruntime causal attention batch幅componentは、既存行と合わせて `B=1/2/4/8` at `N=512/2048` が揃った。
- `B=2` と `B=8,N=4096` は既存行と同じ `18M pair/s` 前後で、batch幅を増やしてもrequest方向の追加効率はまだ出ていない。
- Cached prefixは `L=65536, M=512` までOOMなしで完走し、`M=1/16/128/512` の長prefix代表境界がcomponentとして揃った。
- 長prefixのcached prefix pair/sは `L=4096` より低く、次のcached-prefix最適化はK/V read coalescingとrequest/batch方向を優先する。
- package self-attention layer partialの最新 `N=16384` は `13279.226135 ms`、`1233.806837 tok/s` で、単layer componentとして長尺cold prefill圧力を引き続き測れる。

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
- FlashAttention2-style cached-prefix prerequisite完了後の、AQ4/FP8/vLLM比較schemaを固定する。

手順:

1. 使用するR9700 device indexを固定する。
2. AQ4 latest baseline commitとpackage pathを記録する。
3. FP8 candidate artifact path規約を決める。
4. total throughput schemaの追加項目をdocs/specsへ反映する。
5. result path規約を決める。
6. cached-prefix resultには `executor` と `resolved_executor` を必須列として残す。
7. R9700 FP8 cached-prefixの暫定default executorを `cached_prefix_rdna4_fp8_auto` と明記する。

成果物:

- updated benchmark schema
- result directory convention
- baseline artifact index
- R9700 SQ evaluation state freeze note

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

### T1 current status: logical batch cold schema v1

`package-batch-throughput-bench` のlogical batch reportに、cold prefill用のprefix/chunk/context accountingを追加した。
具体的には `workload.prefill_mode="cold"`、`cached_prefix_tokens_per_request=0`、`new_prefill_tokens_per_request`、`total_context_tokens_after_prefill_per_request`、`metrics.cached_prefix_total_tokens=0`、`metrics.total_context_tokens_after_prefill`、`metrics.estimated_prefill_attention_work_tokens` を保存する。
`estimated_prefill_attention_work_tokens` は既存component smokeと同じく、requestごとの `N * (N + 1) / 2` の合計とする。

保存先:

- `benchmarks/results/2026-07-07/package-batch-throughput/phase-t1-logical-batch-cold-schema-v1.md`

R9700 schema/control-plane smoke:

| B | prompt/request | generated/request | prefill tok/s | decode tok/s | end-to-end tok/s | estimated attention work | verified |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4 | 2 | 53.808662 | 231.111649 | 8.270042 | 10 | true |
| 2 | 4 | 2 | 98.478364 | 233.595603 | 9.792617 | 20 | true |
| 4 | 4 | 2 | 172.597125 | 233.697961 | 10.443645 | 40 | true |

解釈:

- T1のlogical batch最小exit criteriaである `B=1/2/4` のJSON reportは出せる状態になった。
- prefill、decode、end-to-endのtotal throughputは別々に保存される。
- ただしこのsmokeは1 layer、4 token prompt、logical batch、requestごとのweight reloadありのschema確認であり、SQ候補の性能判断には使わない。
- 次のT1実装課題は、workload runnerからこのreportをJSONLへ集約すること、VRAM samplingを付けること、そしてT4/T5へ向けてreal batch executorへ置き換えることである。

### T1 current status: package batch JSONL preservation v1

前回の要点:

- `package-batch-throughput-bench` のraw JSONは、T1に必要なtotal throughputとcold prefill accountingを出していた。
- ただし、`tools/run-external-benchmark.py --parse ullm-package-batch-throughput` を通した
  `inference-benchmark-result-v0.1` JSONLで、SQ比較に必要なfieldが落ちないことはテストで固定できていなかった。

今回の変更点:

- `tools/run-external-benchmark.py` のpackage-batch変換で、raw `batching.prefill_executor` と
  `batching.resolved_prefill_executor` をJSONL `workload.*` executor fieldへfallback保存するようにした。
- package-batch専用のmemory enrichment helperを追加し、`memory.kv_cache_bytes_total` の保持を明示した。
- `tests/test_external_benchmark_batch_parser.py` を追加し、次のfield preservationを単体テストで固定した。
  `prefill_total_input_tokens_per_second`、`decode_total_generated_tokens_per_second`、
  `end_to_end_total_tokens_per_second`、prefix/chunk/context accounting、executor accounting、
  `memory.kv_cache_bytes_total`。
- 合成 `package-batch-throughput-bench-v0.1` reportを `run-external-benchmark.py` のmain pathへ通し、
  `resolved_prefill_executor` と `kv_cache_bytes_total` がJSONL rowに残ることを確認した。
- 結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-jsonl-preservation-v1.md` に保存した。

次の行動:

1. T1 JSONL/schema preservationはv0.1として完了扱いにする。
2. T1 real batch runnerは未完了のまま残す。
3. 次のT1実装は、logical batchではなく、prefillまたはdecodeのreal batch executorをfull package pathへ接続することに集中する。

### T1 current status: component prefill real-batch parser v1

前回の要点:

- `package-batch-throughput-bench` のlogical batch JSONL preservationは完了した。
- ただし、既存のreal-batch component smokeはkey-value stdoutであり、`inference-benchmark-result-v0.1` JSONLへ流せなかった。

今回の変更点:

- `tools/run-external-benchmark.py` に `--parse ullm-component-prefill` を追加した。
- `runtime-causal-attn-batch-smoke` のようなcomponent prefill real-batch outputをkey-value parseし、`inference-benchmark-result-v0.1` rowへ変換できるようにした。
- `batching.mode=real`、`batching.prefill_real_batch=true`、request/token parallelism、`prefill_total_input_tokens_per_second`、`attention_pair_tps_mean`、sampled correctnessを保存する。
- R9700で `runtime-causal-attn-batch-smoke` のB=2/N=32 smokeをJSONLへ変換し、`prefill_total_input_tokens_per_second=850713.136872`、`prefill_real_batch=true` を確認した。
- 結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-component-prefill-real-batch-parser-v1.md` に保存した。

次の行動:

1. component real-batch rowsはkernel/schema検証に使う。
2. full package throughput判断にはまだ使わない。
3. 次はpackage prefillまたはdecode runnerをこのreal-batch executor pathへ接続する。

### T1 current status: package prefill component runner v1

前回の要点:

- component prefill real-batch outputはJSONLへ変換できるようになった。
- ただしsynthetic runtime componentであり、`.ullm.d` package pathとはまだ接続されていなかった。

今回の変更点:

- `tools/run-package-prefill-component-workload.py` を追加した。
- `ullm-package-prefill-component-workload-v0.1` manifestからpackage-backed component smokeを実行し、`run-external-benchmark.py --parse ullm-component-prefill` で `inference-benchmark-result-v0.1` JSONLへ保存する。
- parserは `package-prefill-aq4-matvec-batch-smoke` のようなpackage component stdoutも読めるようになり、`token_tps_mean` から `prefill_total_input_tokens_per_second` を補完し、`real_batch=true` を `batching.mode=real` として保存する。
- R9700で `package-prefill-aq4-matvec-batch-smoke` を `.ullm.d` packageに対して実行し、`batching.mode=real`、`prefill_real_batch=true`、`prefill_executor=aq4_matvec_batch_f32`、`prefill_total_input_tokens_per_second=19063.596157` を確認した。
- 結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-runner-v1.md` に保存した。

次の行動:

1. package-backed component runnerはT1の中間段階として保持する。
2. full package throughput判断にはまだ使わない。
3. 次はrequest batch `batch=1/4/8` とdecode/end-to-end total throughputへ広げる。

### T1 current status: package prefill component batch grid v1

前回の要点:

- `.ullm.d` package由来のprefill component rowはJSONLへ保存できるようになった。
- ただし、最初のsmokeは`batch_size=1`の単一caseだった。

今回の変更点:

- `tools/run-package-prefill-component-workload.py` に `component_args_template` を追加した。
- `prompt_tokens * concurrent_requests` から `component_total_prompt_tokens` を計算し、projection componentでは `len:{component_total_prompt_tokens}` としてflattened token-parallel実行できるようにした。
- `run-external-benchmark.py --parse ullm-component-prefill` は、reportに明示的な`batch_count`がないpackage component rowでは、CLI/workload側の`batch_size`と`concurrent_requests`を保持するようにした。
- R9700 AQ4 package `k_proj` componentで、`batch=1, prompt=2` と `batch=4, prompt=2` を測った。
- B=4 caseでは `workload.batch_size=4`、`prompt_tokens_per_request=[2,2,2,2]`、`component_total_input_tokens=8`、`prefill_executor_token_parallelism=8`、`prefill_executor_request_parallelism=1` として保存された。
- 結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-batch-grid-v1.md` に保存した。

次の行動:

1. この結果はpackage-backed component batch gridとして保持する。
2. request boundaryが効くself-attention layer componentへ広げる。
3. full package throughput判断には、decode/end-to-endを含むpackage total-throughput runnerが必要。

### T2: FP8 SQ candidate package/runtime prototype, 3-5 days

目的:

- FP8をSQ候補1として、R9700で読み込めるruntime pathを作る。
- 最初の対象を `sq-fp8-w8a16-r9700-v0` に限定し、SQ format策定前の速度・品質基準線を作る。

手順:

1. `sq-fp8-w8a16-r9700-v0` のartifact manifestとmetadataを定義する。
2. FP8 payload writerを追加する。
3. scale granularityをまずtensorまたはrowに固定する。
4. scale dtypeとlayoutをmetadataへ保存する。
5. resident bytes、materialized working-set bytes、passthrough tensor一覧を保存する。
6. MLP、attention projection、linear attention projection、lm_head、embeddingをFP8化する。
7. normや小さいbias/conv/state系はpassthroughのまま残す。
8. R9700 runtimeでFP8 payloadを読む。
9. まずはdequant-to-BF16/F32またはnative FP8 readのどちらが最短か確認する。
10. short prompt guardを通す。
11. guard失敗時は、format不適合、scale granularity不足、runtime変換問題、passthrough不足のどれかに分類して記録する。

成果物:

- FP8 candidate package or runtime artifact
- FP8 candidate load path
- short guard result
- FP8 candidate artifact metadata

Exit criteria:

- R9700でshort promptが完走する。
- NaN/Infが出ない。
- AQ4 baselineまたはBF16 referenceに対するoutput guardが通る、または失敗原因が記録される。

### T2 current status: mixed row-block candidate boundary v1

前回の要点:

- `sq-fp8-w8a16-r9700-v0` は、artifact manifest、FP8 payload writer、runtime materialize smoke、package overlay guardまで進んだ。
- row-scale FP8では、layer 3のprojection setは通るが、layers `3,7` 以上でtop1 ranking driftが出た。
- family splitでは `q/v/down` がstrict top1上のrisk family、`k/o/gate/up` が相対的に安全なfamilyだった。
- row-block scaleは `q` と `down` を回復できたが、`v` はblock16/32/64/128でもstrict top1を維持できなかった。

今回の変更点:

- `v` fallback + `q/k/o/gate/up/down` row-block32 FP8を混合候補としてlayer scaling guardへ広げた。
- layers `3,7,11,15` は、len4、case_a、case_bの `3 / 3` short promptでstrict top1一致だった。
- layers `3,7,11,15,19` はlen4でstrict top1一致だった。
- layers `3,7,11,15,19,23` はlen4でstrict top1不一致になった。
- all self-attention probe layers `3,7,11,15,19,23,27,31` もstrict top1不一致だった。
- layer `23` 単体では `q` row-block32がriskで、`q/v` fallbackなら回復した。
- ただし6層bundleでは、layer `23` の `q` だけ、または全6層の `q/v` をfallbackしてもstrict top1は回復しなかった。
- 結果は `benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-layer-scaling-guard-v0.1.md` に保存した。

次の行動:

1. 混合row-block32候補は「4-5層まで通った部分候補」として保持し、full SQ policyには昇格しない。
2. T2 short guardの合格条件を先に固定する。strict top1だけで進めるか、top-k overlapやtext-level guardを使うかを決める。
3. strict top1を維持する方針なら、6層bundleの累積driftを、追加fallback family、per-layer sensitivity policy、または強いscale/layoutで潰す。
4. strict top1を緩める方針なら、短文生成品質、logit drift、top-k overlapを同じresult schemaで記録して、速度評価へ進める条件を明文化する。
5. overlay load timingはSQ runtime速度ではないため、T5のthroughput比較には使わない。速度評価はT1 real batch runnerとnative/materialization-aware runtime pathで行う。

### T2 current status: overlay acceptance rule v0.1

前回の要点:

- `v` fallback + `q/k/o/gate/up/down` row-block32 FP8は、4-5層までは有望だった。
- 6層bundleとall self-attention probeではstrict top1が崩れた。
- top-k overlapやAQ4 top1 rankは有用な診断指標だが、正式なT2昇格条件としては未確定だった。

今回の変更点:

- `tools/evaluate-sq-fp8-overlay-acceptance.py` を追加した。
- `tests/test_sq_fp8_overlay_acceptance.py` を追加した。
- T2 promotion rule v0.1を `strict_top1` として固定した。
- 診断専用ruleとして `topk_common >= 5`、`baseline_top1_rank_in_sq_topk <= 2`、
  `abs(sq_top1_minus_baseline_top1_logit) <= 0.15` を保存する。
- 診断ruleはstrict top1 failureを上書きしない。
- mixed candidate guard bundleを評価し、10ケース中strict top1 passは `5 / 10`、
  accepted for T2 promotionは `false` だった。
- 結果は `benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-acceptance-v0.1.md` と
  `benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-acceptance-v0.1.json` に保存した。

次の行動:

1. text-level guardを正式採用するまでは、T2 promotionにはstrict top1一致を要求する。
2. top-k/rank/gapは、near missと強い失敗を分ける診断列として使う。
3. 6層bundleのstrict top1 failureを、per-layer/family fallbackまたはstronger scale/layoutで潰す。
4. mixed candidateはまだT5 throughput比較用のpromoted SQ policyとして扱わない。

### T2 current status: six-layer family boundary v1

前回の要点:

- `v` fallback + `q/k/o/gate/up/down` row-block32 FP8は6層でstrict top1を維持できなかった。
- `q/v` fallbackでlayer `23` 単体は回復したが、6層bundleは回復しなかった。
- 6層の累積driftがどのfamilyから来ているかをさらに切る必要があった。

今回の変更点:

- layers `3,7,11,15,19,23` のfamily splitをrow-block32で実行した。
- `k` と `up` は単独でstrict top1一致だった。
- `o`、`gate`、`down` は単独でstrict top1不一致だった。
- `o/gate/down` をrow-block16にしてもstrict top1は回復しなかった。
- `k/up` row-block32を同時にFP8化した6層部分候補は、len4、case_a、case_bの `3 / 3` でstrict top1一致だった。
- ただしcase_aのtop8 overlapは `2 / 8` と低く、診断ruleは失敗した。
- 結果は `benchmarks/results/2026-07-08/sq-fp8-six-layer-family-boundary-v0.1.md` に保存した。

次の行動:

1. `k/up` row-block32は6層strict-top1 regression subsetとして保持する。
2. これはcoverageが低すぎるため、promoted SQ policyとは扱わない。
3. 次のT2は `q/v/o/gate/down` に対するper-layer fallback、別scale/layout、またはstronger formatを試す。
4. 診断gapだけで順序を付けるなら、`o/down` を `gate` より先に見る。

### T2 current status: six-layer per-layer combination boundary v1

前回の要点:

- `k/up` row-block32は6層3 promptでstrict top1一致だった。
- `o/gate/down` は6層単独ではstrict top1不一致だった。
- ただし5層までなら安全なのか、組み合わせで崩れるのかは未確認だった。

今回の変更点:

- layers `3,7,11,15,19` では、`o/gate/down` row-block32がそれぞれstrict top1一致だった。
- `k/up` 全6層に `o5`、`gate5`、`down5` のどれか1 familyを足した3ケースはすべてlen4でstrict top1一致だった。
- `k/up` 全6層に `o5/gate5`、`o5/down5`、`gate5/down5` の2 familyを足した3ケースもすべてlen4でstrict top1一致だった。
- `k/up` 全6層に `o5/gate5/down5` の3 familyをすべて足すとlen4でstrict top1不一致だった。
- 結果は `benchmarks/results/2026-07-08/sq-fp8-six-layer-per-layer-combination-boundary-v0.1.md` に保存した。

次の行動:

1. len4上の最有力candidateは `kup6_gate5_down5` とする。
2. `kup6_gate5_down5` をcase_a/case_bへ広げ、6層prompt bundleでstrict top1が維持されるか確認する。
3. `kup6_ogatedown5` はnear-miss failureとして保持する。
4. full SQ policyにはまだ昇格しない。

### T2 current status: six-layer `kup6_gate5_down5` prompt bundle v1

前回の要点:

- len4上の最有力candidateは `kup6_gate5_down5` だった。
- `kup6_gate5_down5` は、`k/up` をlayers `3,7,11,15,19,23`、`gate/down` をlayers `3,7,11,15,19` でFP8 row-block32にする。
- `q/v/o` とlayer `23` の `gate/down` はfallbackに残す。

今回の変更点:

- `kup6_gate5_down5` をcase_a/case_bへ広げた。
- len4、case_a、case_bの `3 / 3` でstrict top1一致だった。
- case_aのtop8 overlapは `2 / 8` と低く、diagnostic qualityはまだ弱い。
- 結果は `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup6-gate5-down5-prompt-bundle-v0.1.md` に保存した。
- 選択FP8 family/layerとfallback family/layerは `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json` に保存した。

次の行動:

1. `kup6_gate5_down5` は6層strict-top1 regression subsetとして固定する。
2. full SQ policyにはまだ昇格しない。
3. `sq-fp8-kup6-gate5-down5-policy-v0.1.json` を、次のartifact生成とfallback記録の基準にする。
4. T1 real batch runnerを進め、SQ候補評価で使えるthroughput行を作る。

### T2 current status: policy JSON builder input v1

前回の要点:

- `kup6_gate5_down5` の選択FP8/fallback方針を `sq-fp8-policy-v0.1` として保存した。
- ただし保存直後は、artifact builderへ手動でinclude regexを渡す必要があった。

今回の変更点:

- `tools/build-sq-fp8-w8a16-artifact.py` に `--policy-json` を追加した。
- policyの `candidate_id`、`fp8_selection.include_regex`、scale granularity、row-block widthをbuilder defaultとして使えるようにした。
- 生成manifestへ `policy` blockを追加し、policy ID、source policy path、FP8 selection、fallback policy、prompt bundle resultを保存する。
- dry-runで `kup6_gate5_down5` が `22` FP8 tensors、`753` passthrough tensors、row-block32として解決されることを確認した。
- 結果は `benchmarks/results/2026-07-08/sq-fp8-policy-json-builder-v0.1.md` に保存した。

次の行動:

1. 次のSQ FP8 artifact生成では `--policy-json` を使う。
2. T1 real batch runnerを進め、SQ候補評価で使えるthroughput行を作る。
3. throughput比較ではoverlay load timingを使わず、native FP8またはmaterialization-aware runtime pathを使う。

### T2 current status: policy artifact materialize v1

前回の要点:

- `sq-fp8-policy-v0.1` をbuilderが読めるようになった。
- dry-runでは `kup6_gate5_down5` が `22` FP8 tensors、`753` passthrough tensors、row-block32として解決された。

今回の変更点:

- `--policy-json benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json` から実FP8 payload artifactを生成した。
- artifactは `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` に保存した。
- manifestにはpolicy blockが入り、`policy_id=kup6_gate5_down5`、`fp8_tensor_count=22`、`passthrough_tensor_count=753`、row-block32が確認できた。
- R9700 device index `2` で `model.language_model.layers.3.self_attn.k_proj.weight` の2 rowsをmaterializeし、`roundtrip_max_abs_diff=0`、`verified=true` を確認した。
- 結果は `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-artifact-v0.1.md` に保存した。

次の行動:

1. このartifactはruntime boundary check用に使える。
2. throughput比較ではhost-side materialize/load timingを使わない。
3. T1 package-backed component runnerをfull package total throughput runnerへ広げる。

### T3: Prefill optimization v0.1, 4-7 days

目的:

- token-by-token prefillから、SQ候補評価に使えるbatched/tiled prefillへ移す。

手順:

1. hidden stateを `[tokens, hidden]` layoutで扱うprefill bufferを作る。
2. RMSNormを複数token同時に実行する。
3. MLP gate/up/downをtoken batchに対して実行する。
4. self-attention projectionをtoken batchに対して実行する。
5. self-attention prefillは、まずchunked causal attentionで実装する。
6. cached prefix `L` と新規input chunk `M` を分け、`M x L` と `M x M` のattentionを同じrunnerから測れるようにする。
7. `N=2^16` cold prefillはstreamed/chunked pathで扱い、full `[N,N]` attention matrixや過大な中間bufferを確保しない。
8. `cached_prefix_tokens`, `new_prefill_tokens`, `total_context_tokens_after_prefill`, `estimated_prefill_attention_work_tokens` をresultへ保存する。
9. Phase C4の探索gridを使い、512 tokenだけではなく `N=1024/2048/4096/8192/16384`、`L=4096/16384/65536`、`M=1/16/128/512`、`B=1/2/4/8` のscalingから次のkernel修正を決める。
10. 512 token結果はshort sanityとして扱い、SQ候補のprefill採用判断には長さ別・chunk別・batch別の結果を必須にする。
11. linear attentionは、recurrent state更新を壊さない範囲でprojection/MLPをbatched化し、state scanは段階的に最適化する。
12. KV writeをprompt token列に対してまとめる。
13. prefill resultをdecode stateへ接続する。

成果物:

- chunked or batched prefill executor
- prefill component timing
- before/after prefill throughput comparison

Exit criteria:

- R9700で `prompt_tokens=512` のprefillが現行token-loopより明確に速い。
- `prompt_tokens=2048` のprefillがOOMせず完走する。
- `prompt_tokens=8192` 以上のcold prefill scalingが代表測定として取れる。
- cached prefix `L=65536, M=1/16/128/512` の少なくとも1系統がOOMせず完走し、prefix長と新規input長を分けた結果として保存される。
- 512 tokenより長い複数patternを測り、次に最適化する対象がcausal attention、cached prefix attention、projection/MLP batch、scheduler/batch入力のどれかに分類されている。
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
8. cold prefill `8192/16384/65536` とcached prefix `L=65536, M=1/16/128/512` の結果が、少なくともcomponentまたはchunked prefill runnerで保存されている。
9. AQ4 baselineとFP8 candidateの比較表がある。
10. vLLM比較が成功またはunsupported reason付きで記録されている。

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

2026-07-07 self-attention prefill causal attention batch:

- `package-self-attn-attention-batch-smoke` を追加した。
- Qwen3.5-9B packageのself-attention layer 3に対して、`input RMSNorm -> q/k/v AQ4 batch projection -> qwen35_qk_norm_rope_batch_f32 -> causal_attn_f32` を同一token batchで接続した。
- R9700では `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`、`ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1`、`ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1` を付けて、staging fallbackなしで確認した。
- このsmokeはo projection、residual add、MLPまでは含まない。self-attention prefill attention側のdevice-resident component timingとattention output guardを目的にする。

R9700 release results:

| component | prompt tokens | wall ms mean | token/s mean | note |
| --- | ---: | ---: | ---: | --- |
| self-attention qkv+QK RoPE+causal attention | 4 | 0.484690 | 8252.691925 | warmup 1、measured 3 |
| self-attention qkv+QK RoPE+causal attention | 128 | 24.605704 | 5202.045750 | warmup 1、measured 5 |
| self-attention qkv+QK RoPE+causal attention | 512 | 281.601274 | 1818.173590 | warmup 1、measured 3 |

Front-halfとの差分:

| prompt tokens | front-half ms | attention-included ms | delta ms |
| ---: | ---: | ---: | ---: |
| 4 | 0.315858 | 0.484690 | 0.168832 |
| 128 | 7.385021 | 24.605704 | 17.220683 |
| 512 | 24.703001 | 281.601274 | 256.898273 |

Guard:

- `input_norm_max_abs_diff <= 0.000072479`
- `q_gate_max_abs_diff = 0`
- `q_rope_max_abs_diff <= 0.000059426`
- `k_rope_max_abs_diff <= 0.000045419`
- `attention_max_abs_diff <= 0.000003248`

解釈:

- causal attention込みのprefill速度は、512 tokenで約 `1.82k tok/s` まで低下した。
- QKV projection、QK norm、RoPE前半は512 tokenで `24.70 ms` なので、512 token時の追加 `256.90 ms` はほぼcausal attention prefill kernel側で発生している。
- SQ候補のprefill評価へ進む前に、self-attention prefill attention kernelをtiled/blocked化して、長いpromptでのO(N^2)部分を現実的な速度に近づける必要がある。
- 次はo projection/residualへ広げる前に、causal attention prefill kernel自体のcomponent benchmarkと最小限のtiling方針を固める。

2026-07-07 causal attention source-shared v1:

- cached prefix source-shared v2と同じ方針を `ullm_causal_attn_f32_kernel` に反映した。
- 1 block = 1 token/headとし、各source timestepのQK dotとsoftmax weightをblock内で1回計算してvalue次元のthreadへ共有する。
- `package-self-attn-attention-batch-smoke` で、Qwen3.5-9B package layer 3の `input RMSNorm -> q/k/v AQ4 batch projection -> qwen35_qk_norm_rope_batch_f32 -> causal_attn_f32` を再測定した。
- 保存先:
  - `benchmarks/results/2026-07-07/runtime-causal-attn-source-shared/phase-c4-self-attn-attention-source-shared-v1.md`

R9700 release results:

| component | prompt tokens | wall ms mean | token/s mean | verification | attention diff | note |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| self-attention qkv+QK RoPE+causal attention | 128 | 7.637947 | 16758.430420 | full | 0.000011265 | measured 5 |
| self-attention qkv+QK RoPE+causal attention | 512 | 43.796889 | 11690.327961 | full | 0.000011265 | measured 3 |
| self-attention qkv+QK RoPE+causal attention | 1024 | 116.215921 | 8811.185173 | full | 0.000011265 | measured 1 |
| self-attention qkv+QK RoPE+causal attention | 2048 | 374.883299 | 5463.033444 | full | 0.000011265 | measured 1 |
| self-attention qkv+QK RoPE+causal attention | 4096 | 1339.278313 | 3058.363568 | sampled 15 | 0.000000209 | measured 1 |
| self-attention qkv+QK RoPE+causal attention | 8192 | 5157.917832 | 1588.237786 | sampled 15 | 0.000000104 | measured 1 |
| self-attention qkv+QK RoPE+causal attention | 16384 | 20944.388749 | 782.262027 | sampled 15 | 0.000000320 | measured 1 |

Previous comparison:

| prompt tokens | old wall ms | new wall ms | old token/s | new token/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24.605704 | 7.637947 | 5202.045750 | 16758.430420 | 3.222x |
| 512 | 281.601274 | 43.796889 | 1818.173590 | 11690.327961 | 6.430x |

RoPE guard:

- self-attention batch smokeのRoPE guardを、固定 `2e-4` からposition長に応じた上限付きabs floorへ変更した。現capは `4e-3`。
- `2048`: `q_rope_abs_floor=0.000409400`, `q_rope_max_abs_diff=0.000270158`, `k_rope_max_abs_diff=0.000198193`
- `4096`: `q_rope_abs_floor=0.000819000`, `q_rope_max_abs_diff=0.000506938`, `k_rope_max_abs_diff=0.000336170`
- `8192`: `q_rope_abs_floor=0.001638200`, `q_rope_max_abs_diff=0.001175225`, `k_rope_max_abs_diff=0.000833869`
- `16384`: `q_rope_abs_floor=0.003276600`, `q_rope_max_abs_diff=0.002606988`, `k_rope_max_abs_diff=0.001518801`
- 1024以上ではfull host attention reference verificationを避け、15点のsampled attention verificationを使う。確認時間は4096で約 `0.91s`、8192で約 `1.73s`、16384で約 `3.42s`。

解釈:

- 512 tokenのself-attention attention込みcomponentは約 `1.82k tok/s` から約 `11.69k tok/s` へ改善した。
- Phase C4 cold prefill length scalingの必須範囲 `N=1024/2048/4096/8192/16384` は、self-attention attention componentとして取得できた。
- 4096 tokenで約 `3.06k tok/s`、8192 tokenで約 `1.59k tok/s`、16384 tokenで約 `0.78k tok/s` まで落ちており、長尺側は引き続きcausal attentionのO(N^2)部分が支配的である。
- 次はo projection/residualまで接続してself-attention layer partialを再測定し、layer単位でattention支配が維持されるかを確認する。その後、必要ならcausal attention kernelのtile/blocking再設計へ戻る。

2026-07-07 self-attention block batch v1:

- `package-self-attn-block-batch-smoke` を追加した。
- 既存のself-attention attention batch pathに、`sigmoid(q_gate) * attention -> o_proj AQ4 batch -> residual add` を接続した。
- attention verificationは、full host referenceが1024 token時点で約18秒かかるため、1024以上ではsampled verificationへ切り替えた。
- `o_proj` は長尺promptでfull host projection referenceを作らず、AQ4 row dot productのsampled verificationで確認する。
- 保存先:
  - `benchmarks/results/2026-07-07/runtime-causal-attn-source-shared/phase-c4-self-attn-block-batch-v1.md`

R9700 release results:

| component | prompt tokens | wall ms mean | token/s mean | attention diff | o proj diff | block diff | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| self-attention block batch | 128 | 11.099879 | 11531.656891 | 0.000011265 | 0.000000864 | 0.000000000 | measured 3 |
| self-attention block batch | 512 | 54.498359 | 9394.778233 | 0.000011265 | 0.000000864 | 0.000000000 | measured 3 |
| self-attention block batch | 1024 | 141.180790 | 7253.111418 | 0.000000130 | 0.000000864 | 0.000000000 | sampled |
| self-attention block batch | 2048 | 433.086886 | 4728.843256 | 0.000000209 | 0.000000864 | 0.000000000 | sampled |
| self-attention block batch | 4096 | 1450.365419 | 2824.115872 | 0.000000209 | 0.000000864 | 0.000000000 | sampled |
| self-attention block batch | 8192 | 5382.886791 | 1521.859983 | 0.000000104 | 0.000000864 | 0.000000000 | sampled |
| self-attention block batch | 16384 | 21844.617459 | 750.024578 | 0.000000320 | 0.000000864 | 0.000000000 | sampled |

Attention-only comparison:

| prompt tokens | attention-only ms | block ms | block delta ms | block/attention wall |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 7.637947 | 11.099879 | 3.461932 | 1.453x |
| 512 | 43.796889 | 54.498359 | 10.701470 | 1.244x |
| 1024 | 116.215921 | 141.180790 | 24.964869 | 1.215x |
| 2048 | 374.883299 | 433.086886 | 58.203587 | 1.155x |
| 4096 | 1339.278313 | 1450.365419 | 111.087106 | 1.083x |
| 8192 | 5157.917832 | 5382.886791 | 224.968959 | 1.044x |
| 16384 | 20944.388749 | 21844.617459 | 900.228710 | 1.043x |

解釈:

- o projection/residualまで接続しても、長尺promptではcausal attention支配が維持される。
- block/attention wall ratioは128 tokenで約 `1.45x`、512 tokenで約 `1.24x` だが、8192/16384では約 `1.04x` まで下がる。
- self-attention block単位でもPhase C4のcold prefill length scalingは取れた。次はpost-attention RMSNorm/MLPまで含むself-attention layer partialか、causal attention kernelのtile/blocking再設計へ進む。

2026-07-07 self-attention layer batch v1:

- `package-self-attn-layer-batch-smoke` を追加した。
- 既存のself-attention block batch pathに、`post RMSNorm -> gate/up AQ4 batch -> SiLU-mul -> down AQ4 batch -> residual add` を接続した。
- `mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj` はfull host projection referenceを作らず、AQ4 row dot productのsampled verificationで確認する。
- 保存先:
  - `benchmarks/results/2026-07-07/runtime-causal-attn-source-shared/phase-c4-self-attn-layer-batch-v1.md`

R9700 release results:

| component | prompt tokens | wall ms mean | token/s mean | block-only ms | layer-block delta ms | layer/block wall | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| self-attention layer batch | 128 | 30.783820 | 4158.028516 | 11.099879 | 19.683941 | 2.773x | measured 3 |
| self-attention layer batch | 512 | 141.768580 | 3611.519562 | 54.498359 | 87.270221 | 2.601x | measured 3 |
| self-attention layer batch | 1024 | 318.234144 | 3217.756546 | 141.180790 | 177.053354 | 2.254x | sampled |
| self-attention layer batch | 2048 | 777.894286 | 2632.748481 | 433.086886 | 344.807400 | 1.796x | sampled |
| self-attention layer batch | 4096 | 2182.970006 | 1876.342776 | 1450.365419 | 732.604587 | 1.505x | sampled |
| self-attention layer batch | 8192 | 6892.180390 | 1188.593382 | 5382.886791 | 1509.293599 | 1.280x | sampled |
| self-attention layer batch | 16384 | 24825.171928 | 659.975288 | 21844.617459 | 2980.554469 | 1.136x | sampled |

Guard:

- `mlp_norm_max_abs_diff <= 0.000008106`
- `mlp_gate_max_abs_diff <= 0.000003099`
- `mlp_up_max_abs_diff <= 0.000004172`
- `mlp_activation_max_abs_diff <= 0.000001907`
- `mlp_down_max_abs_diff <= 0.000001788`
- `layer_residual_max_abs_diff = 0`

解釈:

- post RMSNorm/MLPまで含めても、全長でverifiedになり、self-attention layer partialとしてdevice-resident prefill batchを測れる状態になった。
- 128/512 tokenではMLPの線形コストが支配的に見え、layer/block wall ratioは約 `2.77x` / `2.60x`。
- 8192/16384 tokenではO(N^2)のcausal attention比率が再び大きくなり、layer/block wall ratioは約 `1.28x` / `1.14x` まで下がる。ただしMLP追加分は16384でも約 `2.98s` あり、全layer stackでは無視できない。
- SQ候補評価の前に、self-attention layer partialの長尺gridは最低限揃った。次のprefill側の主な不足は、複数layer stack、real batch幅、cached-prefix chunk pathでの同等のcomponent計測である。

2026-07-07 cached prefix attention baseline:

- `runtime-cached-prefix-attn-smoke` を追加した。
- synthetic Q/K/Vを使い、既存KV cache長 `L` に対して新規input chunk `M` tokenを `decode_attn_f32` の連続実行で処理する。
- これは最適化済みchunked cached-prefix prefillではない。cached prefix attentionのbaseline、OOM境界、prefix長scalingを測るためのruntime component smokeである。
- R9700では `ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL=1` を付けて、staging fallbackなしで確認した。
- 巨大prefixではfull host verificationが支配的になるため、output guardは代表座標のsampled verificationとして記録する。

R9700 release results:

| cached prefix L | new input M | wall ms mean | new input tok/s | estimated attention work | attention pair/s | note |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 1 | 98.653080 | 10.136531 | 4097 | 41529.367495 | measured 3 |
| 4096 | 16 | 1570.769684 | 10.186089 | 65672 | 41808.802824 | measured 3 |
| 16384 | 1 | 510.701715 | 1.958090 | 16385 | 32083.307181 | measured 1 |
| 65536 | 1 | 2030.332401 | 0.492530 | 65537 | 32278.950958 | measured 1 |

Guard:

- `verification=sampled`
- `sampled_max_abs_diff = 0` in all measured rows

解釈:

- `decode_attn_f32_loop` baselineでは、cached prefix `L=4096` 時点で約 `10 tok/s`、`L=65536` では約 `0.49 tok/s` まで落ちる。
- `L=4096, M=1` と `L=4096, M=16` のnew input tok/sはほぼ同じなので、現baselineは `M` 方向のchunk並列化ができていない。
- `attention_pair_tps` は `32k-42k pair/s` 程度で、prefix長方向にはおおむね線形に悪化している。
- SQ/FP8候補のprefill評価へ進む前に、`M x L` と `M x M` をまとめて扱うchunked cached-prefix attention kernelが必要である。
- 次の実装対象は、`decode_attn_f32` loopではなく、`cached_prefix_chunked` executorである。

2026-07-07 cached prefix chunked attention v0:

- `ullm_runtime_cached_prefix_attn_f32` を追加した。
- 入力shapeは `q=[M,q_heads,head_dim]`、`k/v=[L+M,kv_heads,head_dim|value_dim]`、`output=[M,q_heads,value_dim]`。
- `runtime-cached-prefix-attn-smoke` の既定executorを `cached_prefix_chunked` にし、末尾引数で `decode_loop` baselineも選べるようにした。
- kernelはまだ1 output element 1 threadの素朴な実装で、score計算をvalue次元ごとに繰り返す。FlashAttention系のtiling実装ではない。

R9700 release results:

| executor | cached prefix L | new input M | wall ms mean | new input tok/s | attention pair/s | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| decode loop | 4096 | 16 | 1575.284095 | 10.156898 | 41688.988169 | measured 1、現行binaryで再測 |
| cached prefix chunked | 4096 | 1 | 103.192291 | 9.690646 | 39702.578299 | measured 3 |
| cached prefix chunked | 4096 | 16 | 124.760442 | 128.245778 | 526384.796108 | measured 3 |
| cached prefix chunked | 65536 | 1 | 2027.916899 | 0.493117 | 32317.399215 | measured 1 |
| cached prefix chunked | 65536 | 16 | 2326.898321 | 6.876106 | 450690.943620 | measured 1 |

Guard:

- `verification=sampled`
- `sampled_max_abs_diff = 0` in all measured rows

解釈:

- `L=4096, M=16` では、`decode_loop` の約 `10.16 tok/s` から `cached_prefix_chunked` の約 `128.25 tok/s` へ改善した。M方向のchunk並列化は効いている。
- `L=4096, M=1` と `L=65536, M=1` は単発decode相当なので、chunked化だけでは改善しない。
- `L=65536, M=16` でも約 `6.88 tok/s` に留まり、長prefixではL方向のscore/value計算が支配している。
- 次は1 output element 1 threadの素朴実装から、head/value内でscore計算を共有するtiled cached-prefix attentionへ進む必要がある。
- ただし、SQ/FP8候補評価用のworkload gridでは、`M>1` cached prefillの現実的なbaselineとして `cached_prefix_chunked` を使える。

2026-07-07 Phase C4 cached prefix sweep runner:

- `tools/run-runtime-cached-prefix-sweep.py` を追加した。
- `runtime-cached-prefix-attn-smoke` を `L/M/executor` のgridとして実行し、JSONLとMarkdown summaryを保存する。
- まずR9700で `cached_prefix_chunked` の代表sweepを実行した。
- 保存先:
  - `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-sanity.jsonl`
  - `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-sanity.md`

R9700 release results:

| executor | cached prefix L | new input M | repeats | wall ms mean | new input tok/s | attention pair/s | sampled diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cached prefix chunked | 4096 | 1 | 3 | 103.203890 | 9.689557 | 39698.115902 | 0 |
| cached prefix chunked | 4096 | 16 | 3 | 124.240184 | 128.782810 | 528589.041882 | 0 |
| cached prefix chunked | 4096 | 128 | 1 | 1019.441962 | 125.558889 | 522387.757078 | 0 |
| cached prefix chunked | 16384 | 1 | 1 | 524.079953 | 1.908106 | 31264.313596 | 0 |
| cached prefix chunked | 16384 | 16 | 1 | 578.890933 | 27.639058 | 453073.256202 | 0 |
| cached prefix chunked | 16384 | 128 | 1 | 4616.811130 | 27.724764 | 456030.784175 | 0 |
| cached prefix chunked | 65536 | 1 | 1 | 2055.547196 | 0.486488 | 31882.994527 | 0 |
| cached prefix chunked | 65536 | 16 | 1 | 2356.189138 | 6.790626 | 445088.207516 | 0 |
| cached prefix chunked | 65536 | 128 | 1 | 18234.605441 | 7.019620 | 460490.578048 | 0 |

解釈:

- `M=16` と `M=128` のnew input tok/sは、同じ `L` ではほぼ同程度になった。chunk sizeを16以上にしても、現v0 kernelでは大きな追加改善は見えていない。
- `M=1` はdecode-like boundaryであり、attention pair/sが `31k-40k pair/s` 程度まで落ちる。一方、`M=16/128` では `445k-529k pair/s` まで上がるため、decode-like pathとcached prefill pathは分けて評価する。
- `L` を4倍にすると `M=16/128` のnew input tok/sはおおむね4分の1になる。長prefixではKV readとscore/value計算が支配的で、SQ/FP8 format差を見る前にattention executor側の効率が上限を決めている。
- 次の最適化対象は、`M` を増やすだけではなく、head/value内でscore計算を共有するtiled cached-prefix attentionと、KV read coalescingである。

2026-07-07 cached prefix shared-score kernel v1:

- `ullm_cached_prefix_attn_f32_kernel` を、1 output element 1 threadの実装から、1 block = 1 token/headのshared-score実装へ変更した。
- max scoreとsoftmax denominatorはblock内reduceで求め、value次元間で共有する。
- weighted value計算ではまだvalueごとにscoreを再計算するため、完全なtiled/blocked attentionではない。
- 保存先:
  - `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-shared-score-v1.jsonl`
  - `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-shared-score-v1.md`

R9700 release results:

| cached prefix L | new input M | old tok/s | shared-score tok/s | speedup | shared-score pair/s | sampled diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 9.689557 | 20.892045 | 2.156x | 85594.708602 | 0 |
| 4096 | 16 | 128.782810 | 286.536642 | 2.225x | 1176089.649006 | 0 |
| 4096 | 128 | 125.558889 | 293.886309 | 2.341x | 1222713.987897 | 0 |
| 16384 | 1 | 1.908106 | 4.977861 | 2.609x | 81562.254321 | 0 |
| 16384 | 16 | 27.639058 | 52.762509 | 1.909x | 864909.422090 | 0 |
| 16384 | 128 | 27.724764 | 73.211619 | 2.641x | 1204221.322034 | 0 |
| 65536 | 1 | 0.486488 | 1.097658 | 2.256x | 71937.242905 | 0 |
| 65536 | 16 | 6.790626 | 9.203588 | 1.355x | 603244.548450 | 0 |
| 65536 | 128 | 7.019620 | 16.081822 | 2.291x | 1054975.580718 | 0 |

解釈:

- shared-score化だけでも、代表gridで `1.35x-2.64x` の改善が出た。
- `L=4096, M=16` は約 `128.78 tok/s` から約 `286.54 tok/s` へ改善し、Phase C4のcached prefill baselineとしては前進した。
- `L=65536, M=128` は約 `7.02 tok/s` から約 `16.08 tok/s` へ改善したが、長prefixではまだ低い。
- `M=16/128` のpair/sは最大で約 `1.22M pair/s` まで上がったが、R9700のメモリ帯域や演算性能から見ればまだ低効率である。
- 次はweighted value側のscore再計算削減、source tile単位のQ/K/V read共有、KV read coalescingを検討する。

2026-07-07 cached prefix source-shared kernel v2:

- `ullm_cached_prefix_attn_f32_kernel` を、shared-score v1からsource-shared v2へ変更した。
- 各source timestepのQK dotをblock内reduceで1回だけ計算し、そのsoftmax weightをvalue次元のthreadへ共有する。
- value_dimがblock size以下の現Qwen条件では、weighted value側のscore再計算を避けられる。
- 保存先:
  - `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-source-shared-v2.jsonl`
  - `benchmarks/results/2026-07-07/runtime-cached-prefix-sweep/phase-c4-cached-prefix-source-shared-v2.md`

R9700 release comparison:

| cached prefix L | new input M | v0 tok/s | shared-score v1 tok/s | source-shared v2 tok/s | v2/v1 | v2/v0 | v2 pair/s |
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

解釈:

- v2は代表gridでv1比 `4.74x-8.85x`、v0比 `10.70x-14.36x` の改善になった。
- `L=4096, M=16/128` は約 `1.78k-1.80k new tok/s` まで上がり、短中prefixのcached prefill componentとしては次の段階へ進める速度になった。
- `L=65536, M=16/128` は約 `81-86 new tok/s` まで上がった。まだR9700の理論帯域から見ると低いが、以前の約 `7 tok/s` とは別物になった。
- `M=1` はv2でもdecode-like boundaryとして残り、`L=65536` では約 `5.2 tok/s` である。decode pathは別kernelまたはpaged decode attention側の最適化として扱う。
- 次はcold prefill側のcausal attentionにも同じsource-shared方針を反映できるかを検討する。SQ候補評価では、cached prefillについてはv2を現baselineとして使う。

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

### 2026-07-08 progress: cached-prefix flash2 FP8 Q baseline v1

前回の要点:

- `cached_prefix_rocwmma_fp8` はRDNA4 FP8 WMMA/rocWMMAを使う方向で進めたが、`head_dim=value_dim=256` では既存scalar `cached_prefix_flash2` よりかなり遅かった。
- rocWMMA版はQ/K/VをすべてFP8として読む一方、既存 `cached_prefix_flash2` はQだけF32だった。
- そのため、rocWMMA版の遅さが「FP8 Q復号そのもの」なのか「rocWMMA kernel構造」なのかを切り分ける必要があった。

今回の変更点:

- C ABI `ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2_fp8q` を追加した。
- Rust FFI `cached_prefix_attn_fp8_e4m3_flash2_fp8q` とCPU/HIPテストを追加した。
- `runtime-cached-prefix-attn-smoke` に `cached_prefix_flash2_fp8q` executorを追加した。
- `tools/run-runtime-cached-prefix-sweep.py` に `cached_prefix_flash2_fp8q` を追加した。
- 既存 `cached_prefix_flash2` のHIPRTC kernelを、f32 QとFP8 Qの両方を受け取れる形に拡張した。
- R9700で `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256` を比較した。

| executor | Q dtype | q bytes | wall ms mean | input tok/s | sampled max abs diff |
| --- | --- | ---: | ---: | ---: | ---: |
| cached_prefix_flash2 | F32 | 262144 | 3.856106 | 4149.263890 | 0.000000611 |
| cached_prefix_flash2_fp8q | FP8 E4M3 | 65536 | 4.123693 | 3880.016943 | 0.000002176 |
| cached_prefix_rocwmma_fp8 | FP8 E4M3 | 65536 | 20.403873 | 784.164837 | 0.000000719 |

観察:

- FP8 Q化だけなら、scalar flash2では約 `1.07x` の遅化に留まった。
- rocWMMA版は同じFP8 Q/K/V入力でも `cached_prefix_flash2_fp8q` より約 `4.95x` 遅い。
- したがって現時点のrocWMMA版の主因はFP8 Q復号ではなく、value groupごとのQK/softmax再計算とK/V tile再利用不足を含むkernel構造側だと考える。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c14-fp8q-flash2-baseline-v1.md` に保存した。

次の行動:

1. `cached_prefix_flash2_fp8q` をFP8 Q入力の短期baselineとして使う。
2. rocWMMA版は、QK/softmax再計算を減らすだけでなく、複数query row/blockでK/V tileを再利用するFlashAttention2-like構造へ寄せる。
3. SQ候補評価では、FP8 Q復号単体を主ボトルネック扱いせず、attention kernel構造の未成熟と分けて評価する。

### 2026-07-08 progress: RDNA4 FP8 rocWMMA value group heuristic v1

前回の要点:

- `cached_prefix_flash2_fp8q` により、FP8 Q復号そのものは主ボトルネックではないと切り分けた。
- rocWMMA版の残りの問題は、value groupごとのQK/softmax再計算とblock並列性のバランスだった。

今回の変更点:

- `cached_prefix_rocwmma_fp8` のvalue group幅をruntime選択にした。
- `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH={16,32,64,128,256}` で明示指定できる。
- env未指定時は、`new_tokens < 64` なら16、そうでなければ64を選ぶ。
- kernelにはvalue group幅を引数で渡すため、同じHIPRTC moduleでshapeごとの選択ができる。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c15-rocwmma-value-group-heuristic-v1.md` に保存した。

観察:

- `L=4096,M=16` ではvalue group幅16が最速だったが、scalar `cached_prefix_flash2_fp8q` にはまだ大きく負ける。
- `L=4096,M=128` では `cached_prefix_rocwmma_fp8` が `15.409175ms`、scalar `cached_prefix_flash2` が `26.629428ms`、`cached_prefix_flash2_fp8q` が `28.595555ms` だった。
- `L=4096,M=512` では `cached_prefix_rocwmma_fp8` が `70.911643ms`、scalar `cached_prefix_flash2` が `103.509898ms`、`cached_prefix_flash2_fp8q` が `113.356638ms` だった。
- つまりrocWMMAはdecode-likeな短いchunkではまだ不利だが、SQ評価で重要な数百token prefill chunkではscalar flash2を上回り始めた。

次の行動:

1. short chunkでは `cached_prefix_flash2_fp8q`、larger prefill chunkでは `cached_prefix_rocwmma_fp8` を比較baselineとして扱う。
2. rocWMMA側はvalue group調整だけではなく、複数query token tileへ拡張してFlashAttention2-likeなK/V tile再利用に寄せる。
3. SQ候補評価では、`M=128/512` 以上のcached prefix prefillを必ず含める。

### 2026-07-08 progress: rocWMMA value group sweep axis v1

前回の要点:

- `cached_prefix_rocwmma_fp8` は短いchunkと大きいprefill chunkで有利なvalue group幅が違った。
- 手書きshell loopでは再現性と結果schemaが弱いため、SQ候補評価のgridへ混ぜにくかった。

今回の変更点:

- `tools/run-runtime-cached-prefix-sweep.py` に `--rocwmma-value-group-widths auto|16|32|64|128|256` を追加した。
- `auto` はenv未指定のruntime heuristic、数値指定は `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH` をcaseごとに設定する。
- JSONLの `required_env` と `workload.rocwmma_value_group_width` に設定を保存する。
- summary markdownへ `rocWMMA value group` 列を追加した。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c16-rocwmma-value-group-sweep-axis-v1.md` に保存した。

次の行動:

1. 長prefix gridで `cached_prefix_rocwmma_fp8` を測るときは、少なくとも `auto,16,64` を比較できるようにする。
2. 次のkernel構造変更では、このsweep結果をbaselineとして使う。

### 2026-07-08 progress: rocWMMA long-prefix grid v1

前回の要点:

- `cached_prefix_rocwmma_fp8` のvalue group幅をsweep軸にした。
- 次の判断には、`L={4096,16384,65536}`、`M={16,128,512}` の長prefix gridで、scalar flash2系と同一shape比較する必要があった。

今回の変更点:

- R9700で `cached_prefix_rocwmma_fp8` の `value_group={auto,16,64}` を長prefix gridで測定した。
- 同じ `q_heads=16,kv_heads=1,head_dim=256,value_dim=256` で `cached_prefix_flash2` と `cached_prefix_flash2_fp8q` も測定した。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c17-rocwmma-long-prefix-grid-v1.md` に保存した。

観察:

- `M=16` ではscalar flash2/fp8qがまだ圧倒的に速く、rocWMMAは約 `0.19-0.24x` 程度に留まる。
- `M=128` では全prefix長でrocWMMAがscalar flash2を上回り、`1.47-1.85x` 程度速かった。
- `M=512` では `L=4096` と `L=65536` でrocWMMAが明確に速く、`L=16384` ではF32-Q scalar flash2に対してほぼ同等、FP8-Q scalar flash2には勝った。
- runtime heuristicはこのgridで最速または最速近傍だった。

次の行動:

1. SQ cached-prefix評価では、短chunk/decode-likeは `cached_prefix_flash2_fp8q`、`M>=128` は `cached_prefix_rocwmma_fp8` を主要baselineとして扱う。
2. 次のkernel変更は、`M=16` の弱さを埋めるためのmulti-query-token tile化を検討する。

### 2026-07-08 progress: RDNA4 FP8 cached-prefix auto executor v1

前回の要点:

- 長prefix gridでは、`M=16` はscalar `cached_prefix_flash2_fp8q` が有利で、`M=128/512` は `cached_prefix_rocwmma_fp8` が有利だった。
- SQ候補評価では、この分岐を毎回手で選ぶより、単一executorとして扱える方が測定が安定する。

今回の変更点:

- `runtime-cached-prefix-attn-smoke` に `cached_prefix_rdna4_fp8_auto` を追加した。
- `new_prefill_tokens < 64` は `cached_prefix_flash2_fp8q`、`new_prefill_tokens >= 64` は `cached_prefix_rocwmma_fp8` に解決する。
- smoke出力に `resolved_executor` を追加し、autoが実際にどちらを使ったかを見えるようにした。
- `tools/run-runtime-cached-prefix-sweep.py` もauto executorに対応し、JSONLの `workload.resolved_executor` とsummaryの `resolved` 列に保存する。
- autoの `M<64` はvalue-group sweepを展開せず、`M>=64` のときだけ `--rocwmma-value-group-widths` を適用する。
- 結果は `benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c18-rdna4-fp8-auto-executor-v1.md` に保存した。

観察:

- R9700の `L=4096,q_heads=16,kv_heads=1,head_dim=256,value_dim=256` smokeで、`M=16` はautoが `cached_prefix_flash2_fp8q` に解決し、明示flash2_fp8qと同等だった。
- `M=64` と `M=128` はautoが `cached_prefix_rocwmma_fp8` に解決し、scalar FP8-Q flash2より速かった。
- これにより、SQ候補測定では `cached_prefix_rdna4_fp8_auto` をR9700 FP8 cached-prefixの暫定default executorとして使える。

次の行動:

1. SQ候補のcached-prefix測定では、まず `cached_prefix_rdna4_fp8_auto` を使い、必要に応じて `resolved_executor` で結果を分解する。
2. explicit比較やthreshold調整が必要な場合は、`cached_prefix_flash2_fp8q` と `cached_prefix_rocwmma_fp8` を併走させる。
3. 次のkernel最適化は、短chunkを改善するmulti-query-token tile化として扱う。

## 2026-07-08 current plan update: move to SQ candidate prototype

前回の要点:

- cached-prefix attentionは、FlashAttention2-style scalar pathとRDNA4 FP8 rocWMMA pathを組み合わせるところまで進んだ。
- `L=65536,M=128` の代表runは約1秒級であり、SQ候補評価を始める暫定速度としては十分と判断する。
- 追加のmulti-query-token tilingは有用だが、SQ candidate prototypeを止めるほどの未解決事項ではない。

今回の変更点:

- 次の主タスクを、attention kernel追加からSQ候補プロトタイプへ戻す。
- 最初のSQ候補は `sq-fp8-w8a16-r9700-v0` とする。
- cached-prefix component測定では `cached_prefix_rdna4_fp8_auto` を暫定defaultにする。
- `cached_prefix_flash2_fp8q` と `cached_prefix_rocwmma_fp8` は、threshold調整や原因分解用の明示比較軸として残す。
- T0/T1/T2を先に進め、AQ4 baseline、FP8 SQ候補、vLLM参考結果を同じschemaで比較できる状態を作る。

次の行動:

1. T0として、R9700 device index、AQ4 baseline commit/package、FP8 artifact path、result directory、JSONL schemaを固定する。
2. T1として、batch throughput runnerのJSONL集約、VRAM/KV cache bytes記録、`resolved_executor` 記録を整える。
3. T2として、`sq-fp8-w8a16-r9700-v0` のpayload writer、metadata、runtime load path、short prompt guardを実装する。
4. T3として、FP8候補をprefill/decode runnerへ接続し、cold prefill、cached prefix、decodeの代表gridを保存する。
5. T5でAQ4 latest baselineとFP8候補を同一schemaで比較し、T6/T7でvLLM比較に進む。

## 2026-07-08 T0-T2 plan update: SQ candidate scaffolding

前回の要点:

- cached-prefix attentionは、SQ候補評価を始めるための暫定速度に到達した。
- SQ候補評価では、attention executorの改善だけを続けるより、FP8 SQ候補のartifact、result schema、batch throughput記録を先にそろえる段階へ移る。
- 最初の候補はR9700専用の `sq-fp8-w8a16-r9700-v0` とし、V620/RDNA2向けdequant pathは後続へ回す。

今回の変更点:

- T0は完了扱いにする。R9700 runtime device index、AQ4 baseline package、AQ4 prompt-suite summary、result schema、SQ候補ID、cached-prefix default executorを `benchmarks/results/2026-07-08/sq-r9700-state-freeze-v0.1.*` に固定する。
- T1は「比較行に必要な情報を落とさない」段階まで進める。`inference-benchmark-result-v0.1` と `batch-throughput-workload-v0.1` で、prefill mode、cached prefix token数、新規prefill token数、total context token数、推定attention work、KV cache bytes、requested/resolved executorを保持する。
- T2はartifact境界を先に作る。`sq-fp8-w8a16-r9700-v0` のmanifest仕様と、safetensors modelからFP8 E4M3 payload + F32 scale metadataを生成するwriterを追加する。
- T2のruntime load pathは、`sq_manifest.json` 読込と選択tensor行のFP8 E4M3 + F32 scale materialize smokeから、既存package model loadへSQ overlayを差し込む段階まで進める。
- short prompt guardは、1 tensorの `q_proj` SQ FP8 overlay + AQ4 fallbackではtop1一致まで通った。
- その後、同一layerのself-attention `q/k/v/o_proj` とMLP `gate/up/down_proj` の7 tensor overlayでも、短い3ケースでtop1がAQ4 baselineと一致した。ただしfull FP8 SQ候補ではないため、T2全体はまだ完了扱いにしない。
- 複数self-attention layerへ広げた `layers=3,7` では、attention-only、MLP-only、attention+MLPのいずれもtop1がAQ4 baselineから入れ替わった。AQ4 top1はSQ top8内に残るため壊滅的崩壊ではないが、full-targetへ進む前にfamily/scale/許容基準の切り分けが必要である。
- family別切り分けでは、`q`、`v`、`down` が単独でtop1を動かし、`k`、`o`、`gate`、`up` は単独ではtop1を保った。さらに `k/o/gate/up` を同時にFP8化したsafe subsetは短い3 promptでtop1一致、`q/v/down` のrisk subsetはtop1不一致だった。
- safe subsetを `layers=3,7,11,15` へ広げると、短い3 prompt中 `2 / 3` はtop1一致したが、case_aでtop1が入れ替わった。`layers=3,7,11`、layer `15` 単体、`layers=3,7,15` ではcase_aがtop1一致したため、4 self-attention layer時の累積または組み合わせdriftとして扱う。
- `row_block` scaleを追加した。risk familyでは、`q` はrow-block32、`down` はrow-block64でtop1一致に戻ったが、`v` はblock16/32/64/128でもtop1不一致だった。`v` をfallbackし、`q/k/o/gate/up/down` をrow-block32 FP8にした混合候補は `layers=3,7` の短い3 promptでtop1一致した。

現在のT0-T2状態:

| task | status | artifact |
| --- | --- | --- |
| T0 state freeze | done | `benchmarks/results/2026-07-08/sq-r9700-state-freeze-v0.1.json`, `.md` |
| T1 result schema preservation | partial done | `docs/specs/batch-throughput-workload-v0.1.md`, `docs/specs/inference-benchmark-result-v0.1.md`, `tools/run-external-benchmark.py` |
| T1 real batch/total throughput runner | partial done with package-backed component batch grid, logical full-package grid, hybrid model-loop smoke, and token-id selected-layer real-prefill bridge | `tools/run-package-prefill-component-workload.py`, `benchmarks/workloads/r9700-aq4-package-prefill-component-real-batch-smoke.json`, `phase-t1-package-prefill-component-runner-v1.md`, `phase-t1-package-prefill-component-batch-grid-v1.md`, `benchmarks/workloads/r9700-aq4-full-package-logical-batch-small-grid.json`, `phase-t1-full-package-logical-batch-small-grid-v1.md`, `phase-t1-model-loop-hybrid-throughput-smoke-v1.md`, `phase-t1-token-id-model-loop-hybrid-smoke-v1.md`, `phase-t1-token-id-model-loop-real-prefill-smoke-v1.md`; real full-package request-batch prefill/decode/end-to-end total throughput is still not done. |
| T2 FP8 SQ artifact manifest | done | `docs/specs/sq-fp8-artifact-v0.1.md` |
| T2 FP8 SQ artifact writer | partial done with policy artifact verified | `tools/build-sq-fp8-w8a16-artifact.py` accepts `--policy-json`; actual `kup6_gate5_down5` payload artifact generated under `/tmp` with `22` FP8 tensors and `753` passthrough tensors. |
| T2 runtime load path | partial done with policy artifact materialize and selected-layer model-loop bridge verified | `crates/ullm-engine/src/sq.rs`, `Qwen3PackageSqOverlay`, `sq-fp8-materialize-smoke`, `sq-fp8-token-ids-logits-smoke`, `sq-fp8-token-ids-model-loop-smoke`, `tools/run-sq-fp8-overlay-logits-guard.py`, row-block scale materialization; policy artifact materialize smoke and token-id model-loop SQ overlay bridge verified on R9700. |
| T2 short prompt guard | partial done with six-layer prompt bundle subset found; stricter model-loop guard now blocks promotion | one-tensor and layer-3 projection-set guards passed top1; row-block scale produced a `v`-fallback mixed candidate; later six-layer split narrowed the safe subset to `k/up` over layers `3,7,11,15,19,23` plus at most two of `o/gate/down` over layers `3,7,11,15,19`; `kup6_gate5_down5` passes direct logits len4/case_a/case_b strict top1, but the six-layer token-id model-loop prompt bundle fails `len4` and `case_a`. Coverage reduction shows k/up row-block32 still fails `case_a` even at layer3 only; `up_proj` layer3 row-block32 and `k_proj` layer3 row-block16 pass individually, but k/up layer3 row-block16 fails when combined. Guard artifacts include `benchmarks/results/2026-07-08/sq-fp8-qproj-overlay-logits-guard-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-layer3-projection-overlay-logits-guard-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-layers3-7-overlay-quality-boundary-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-layers3-7-family-split-guard-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-safe-subset-layer-scaling-guard-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-rowblock-scale-risk-guard-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-six-layer-family-boundary-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-six-layer-per-layer-combination-boundary-v0.1.md`, `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup6-gate5-down5-prompt-bundle-v0.1.md`, `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1.md`, and `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-coverage-reduction-v1.md`. |

次の行動:

1. `kup6_gate5_down5` はdirect logits regression subsetとしては残すが、token-id model-loop prompt bundleで崩れるためSQ quality policyへは昇格しない。
2. `kup6_ogatedown5` はstrict-top1 failureかつnear-miss diagnosticとして残す。
3. T2は、model-loop prompt bundleを昇格gateにして、次はper-family/per-tensor scale-layout対応を入れて `k_proj` row-block16 + `up_proj` row-block32 のような混合scale policyを試す。
4. T1 real batch runnerをpackage-backed componentからfull package total throughputへ広げ、JSONL変換後に `prefill_total_input_tps`、`decode_total_generated_tps`、`end_to_end_total_tps`、KV cache bytes、requested/resolved executor、VRAM peakが失われないことを確認する。
5. FP8 SQ候補のthroughput評価では、overlay load timingを使わない。native FP8、materialization-aware path、または明示的にmaterialized working-setを保存したruntime pathだけを速度比較対象にする。
6. T3の追加prefill kernel作業は、SQ候補比較で不足が見えたcaseだけに限定する。現時点ではcached-prefix/cold-prefill component速度はSQ評価へ進む前提として十分と扱う。
7. T5でAQ4 latest baselineとFP8 SQ候補を同じworkload gridで測り、quality、VRAM、resident bytes、working-set bytes、prefill/decode/end-to-end total throughputを比較する。
8. T6/T7のvLLM比較は、uLLM側でR9700 `batch=1/4/8` のFP8/AQ4結果が揃ってから実施する。

2026-07-08 runtime loader smoke result:

- Added `crates/ullm-engine/src/sq.rs`.
- Added `ullm-engine sq-fp8-materialize-smoke`.
- Verified a 4x8 FP8 artifact fixture on CPU device `0` and R9700 device `2`.
- R9700 smoke selected `gate_proj`, materialized two rows from FP8 E4M3 + F32 row scale, copied them to runtime memory, read them back, and reported `roundtrip_max_abs_diff=0`.

2026-07-08 SQ FP8 overlay logits guard result:

- Added a package load overlay path that can materialize exact-name SQ FP8 tensors and fall back to the existing AQ4 package tensors for the rest.
- Added `ullm-engine sq-fp8-token-ids-logits-smoke`.
- Generated a one-tensor artifact for `model.language_model.layers.3.self_attn.q_proj.weight`.
- R9700 short guard with token IDs `1,2,3,4` matched AQ4 baseline top1 token `55020`.
- Top8 common tokens were `7 / 8`, so this is a useful boundary guard, not yet a full SQ quality result.
- Result: `benchmarks/results/2026-07-08/sq-fp8-qproj-overlay-logits-guard-v0.1.md`.

2026-07-08 SQ FP8 layer projection-set guard result:

- Generated a 7 tensor artifact covering layer 3 self-attention `q/k/v/o_proj` and MLP `gate/up/down_proj`.
- R9700 short guard bundle used three token-ID sequences.
- AQ4 and SQ overlay top1 matched in `3 / 3` cases.
- Top8 common tokens were `7 / 8`, `5 / 8`, and `4 / 8`.
- The SQ path spent about `18.3-18.7 s` in `layer_load` because v0.1 still materializes FP8 to host F32 before runtime copy. This is not a native FP8 speed result.
- Result: `benchmarks/results/2026-07-08/sq-fp8-layer3-projection-overlay-logits-guard-v0.1.md`.

2026-07-08 SQ FP8 multi-layer quality boundary:

- Generated layers `3,7` overlay artifacts for attention-only, MLP-only, and attention+MLP projection sets.
- Layer 7 alone still matched top1 for token IDs `1,2,3,4`.
- With layers `3,7` together, attention-only, MLP-only, and attention+MLP overlays all changed top1.
- AQ4 top1 remained inside the SQ top8 in all three multi-layer overlays, so the issue is ranking drift, not total logits collapse.
- Result: `benchmarks/results/2026-07-08/sq-fp8-layers3-7-overlay-quality-boundary-v0.1.md`.

2026-07-08 SQ FP8 family split guard:

- Added `tools/run-sq-fp8-overlay-logits-guard.py` to automate artifact generation, AQ4/SQ logits smoke runs, and top-k comparison JSON.
- In layers `3,7`, individual `q`, `v`, and `down` overlays changed top1.
- Individual `k`, `o`, `gate`, and `up` overlays preserved top1.
- Combined `k/o/gate/up` preserved top1 in `3 / 3` short prompts, while combined `q/v/down` changed top1 and moved AQ4 top1 to rank `5`.
- Result: `benchmarks/results/2026-07-08/sq-fp8-layers3-7-family-split-guard-v0.1.md`.

2026-07-08 SQ FP8 safe subset layer scaling guard:

- Expanded `k/o/gate/up` from layers `3,7` to larger self-attention layer sets.
- `layers=3,7` preserved top1 in `3 / 3` short prompts.
- `layers=3,7,11,15` preserved top1 in `2 / 3` short prompts but failed case_a.
- case_a still passed for `layers=3,7,11`, layer `15` alone, and `layers=3,7,15`, so the failure is cumulative or interaction-driven.
- Result: `benchmarks/results/2026-07-08/sq-fp8-safe-subset-layer-scaling-guard-v0.1.md`.

2026-07-08 SQ FP8 row-block scale risk guard:

- Added `row_block` scale generation and runtime materialization.
- `q` recovered top1 with row-block32.
- `down` recovered top1 with row-block64.
- `v` did not recover top1 for block16/32/64/128.
- A mixed candidate with `v` fallback and `q/k/o/gate/up/down` row-block32 FP8 passed `3 / 3` short prompts for layers `3,7`.
- Result: `benchmarks/results/2026-07-08/sq-fp8-rowblock-scale-risk-guard-v0.1.md`.

## 2026-07-08 current plan update: SQ evaluation phase

前回の要点:

- FlashAttention2-style cached-prefix/cold-prefill componentは、R9700でSQ候補評価を始める前提速度として一旦十分と判断した。
- SQ FP8候補は、row-block32とfallbackの組み合わせで品質境界をかなり狭めた。
- 現在の短期候補 `kup6_gate5_down5` は、len4/case_a/case_bの6層prompt bundleでstrict top1を維持した。

今回の変更点:

- 追加のFlashAttention2-like最適化を主タスクから外し、SQ候補評価の阻害要因になったcaseだけに限定する。
- SQ策定フェーズの主順序を、T2品質境界固定、T1 real batch throughput runner、T5 AQ4/FP8比較、T6/T7 vLLM比較に並べ直す。
- T2では、`kup6_gate5_down5` を6層strict-top1 regression subsetとして扱う。ただしcase_aのtop8 overlapは低いため、full SQ policyには昇格しない。選択FP8/fallback方針は `sq-fp8-policy-v0.1` として保存した。
- T1では、component prefill real-batch smoke outputを `inference-benchmark-result-v0.1` JSONLへ変換できる `ullm-component-prefill` parserを追加した。
- T1では、`.ullm.d` package-backed prefill component smokeを同じJSONL経路へ流すrunnerを追加した。ただしfull package total throughputではない。
- T2では、`kup6_gate5_down5` policyから実FP8 payload artifactを生成し、runtime materialize smokeまで確認した。
- throughput評価では、SQ overlayのhost-side materialize/load timingを使わない。速度比較はnative FP8 path、materialization-aware runtime path、またはworking-setを明示したpathに限定する。
- SQ候補の採用判断では、`prefill_total_input_tps`、`decode_total_generated_tps`、`end_to_end_total_tps`、quality、VRAM、resident bytes、materialized working-set bytesを同じ表で見る。

次の行動:

1. `sq-fp8-kup6-gate5-down5-policy-v0.1.json` を `--policy-json` で渡して、次のSQ artifact生成とfallback理由記録を固定する。
2. real batch runnerをfull package component pathからrequest batchとdecode/end-to-end pathへ広げ、`batch=1/4/8` のprefill/decode/end-to-end total throughputを保存する。
3. FP8 SQ候補1とAQ4 latest baselineを同じworkload gridで測る。
4. cold prefill、cached prefix、decodeの代表gridを埋める。cached-prefixでは `cached_prefix_rdna4_fp8_auto` と `resolved_executor` を使う。
5. uLLM側のR9700結果が揃った後、vLLMを同じgridで測る。R9700 FP8がunsupportedなら、unsupported reason付きの比較行として残す。

## 2026-07-08 current plan update: SQ format evaluation execution order

前回の要点:

- R9700/RDNA4のcached-prefix/cold-prefill componentは、FlashAttention2-style scalar path、FP8-Q path、rocWMMA path、auto executorまで進み、SQ候補評価を始める前提速度としては十分と判断する。
- `kup6_gate5_down5` は6層prompt bundleでstrict top1を維持するが、case_aのtop8 overlapが低いため、full SQ policyではなくregression subsetとして扱う。
- T1はpackage-backed component real-batch JSONLまでは進んだが、full packageのrequest-batch prefill、decode、end-to-end total throughputはまだ未完了である。

今回の変更点:

- 次の主線を、追加のattention kernel開発ではなくSQ format evaluationへ戻す。
- self-attention componentのrequest-boundary検証は有用だが、SQ評価の必須経路ではfull package total throughput runnerを優先する。flattened component gridはkernel/schema sanityに限定し、最終性能比較には使わない。
- SQ候補1は引き続き `sq-fp8-w8a16-r9700-v0` とし、R9700だけで実装・計測する。
- throughput比較では、SQ overlayのhost-side materialize/load timingを速度結果として使わない。native FP8 path、materialization-aware runtime path、またはmaterialized working-setを明示したpathだけを比較対象にする。
- fixed batch v0.1の評価軸は、`batch=1/4/8`、cold prefill、cached prefix、decode、end-to-end、VRAM、resident bytes、working-set bytes、quality guardに限定する。
- continuous batching、tensor parallel、API/server integration、V620/RDNA2 dequant pathは、SQ候補1のR9700評価後へ回す。
- vLLM比較は、uLLM側のR9700 AQ4/FP8行が同じschemaで揃った後に行う。R9700 FP8がvLLM側でunsupportedなら、unsupported reason、backend、dtype、quantizationを比較行に残す。

次の行動:

1. T1aとして、full package request-batch runnerを作る。必須出力は `prefill_total_input_tps`、`decode_total_generated_tps`、`end_to_end_total_tps`、`batching.mode=real`、`concurrent_requests`、`prompt_tokens_per_request`、`generated_tokens_per_request`、KV cache bytes、VRAM peakである。
2. T1bとして、AQ4 packageで `batch=1/4/8` のsmall gridを通し、component gridではなくfull package rowとしてJSONLへ保存する。
3. T2aとして、`sq-fp8-kup6-gate5-down5-policy-v0.1.json` をruntime pathへ接続し、selected FP8/fallback familyが速度計測時にもmanifestから追跡できるようにする。
4. T2bとして、SQ pathのquality guardをbatch pathにも接続する。v0.1の昇格条件はstrict top1一致のままにし、top-k overlap、AQ4 top1 rank、logit gapは診断値として保存する。
5. T3として、FP8 SQ候補1のcold prefill、cached prefix、decode、end-to-end rowsを保存する。cached prefixは `cached_prefix_rdna4_fp8_auto` を既定にし、`resolved_executor` を必ず残す。
6. T4として、AQ4 baselineとFP8 SQ候補1を同一workload gridで比較する。比較表にはquality、VRAM、resident bytes、materialized working-set bytes、prefill/decode/end-to-end total throughputを同じ行に入れる。
7. T5として、SQ候補1が品質または速度で不十分ならformat iterationへ進む。候補はrow-block幅、scale dtype/layout、fallback family、FP8値+FP8 scale、またはより保守的なhybrid policyである。
8. T6として、uLLM側の比較表が固まった後にvLLMを同じgridで測る。vLLM側のunsupported/fallbackは失敗ではなく比較条件として記録する。
9. T7として、R9700 SQ候補1の採否判断を行う。採用条件は、AQ4比のVRAM/working-set削減が説明でき、total throughputが大きく悪化せず、quality guardが通ることである。

## 2026-07-08 progress: T1 full-package logical batch small grid

前回の要点:

- T1にはpackage-backed component real-batch rowとflattened component batch gridがある。
- ただし、これらはkernel/schema sanityであり、full packageのprefill/decode/end-to-end throughputではなかった。
- SQ候補比較へ進むには、少なくともfull package rowで `prefill_total_input_tps`、`decode_total_generated_tps`、`end_to_end_total_tps`、KV cache bytes、VRAM、correctnessが落ちないことを確認する必要がある。

今回の変更点:

- AQ4 full-package logical batch small grid workloadを追加した。
  - `benchmarks/workloads/r9700-aq4-full-package-logical-batch-small-grid.json`
- R9700で `batch=1/4/8`、`prompt_tokens=4`、`generated_tokens=2` を実測した。
  - `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-full-package-logical-batch-small-grid-v1.md`
- 3行すべて `status=ok`、`correctness.verified_all=true` だった。
- JSONLには `prefill_total_input_tokens_per_second`、`decode_total_generated_tokens_per_second`、`end_to_end_total_tokens_per_second`、`memory.kv_cache_bytes_total`、VRAM peak/consumedが残った。
- ただし全行 `batching.mode=logical`、`prefill_real_batch=false`、`decode_real_batch=false`、`runtime_reused_across_requests=false`、`weights_reloaded_per_request=true` である。

実測値:

| batch | prefill total tok/s | decode generated tok/s | end-to-end tok/s | KV cache bytes | VRAM consumed bytes | verified |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 33.574674811 | 68.797769301 | 2.172326744 | 393216 | 4279500800 | true |
| 4 | 58.744142319 | 69.070493674 | 2.433498426 | 1572864 | 4206096384 | true |
| 8 | 67.008573726 | 69.071288010 | 2.533759132 | 3145728 | 4279500800 | true |

次の行動:

1. このlogical full-package gridは、schema/control-plane guardとして扱う。SQ/vLLM比較の最終性能判断には使わない。
2. T1の次の実装は、full package pathへreal request-batch prefill/decode executorを接続し、同じ `batch=1/4/8` gridで `batching.mode=real` の行を出すことに集中する。
3. real batch化後も同じJSONL schemaを使い、logical/realの差分が比較表で混ざらないようにする。

## 2026-07-08 progress: T1 model-loop hybrid throughput smoke

前回の要点:

- logical full-package gridはschema/control-plane guardとして有効だが、real request-batch性能ではない。
- 既存の `package-self-attn-mlp-block-model-loop-smoke` は、selected layer stackでschedulerとdecode ready batchを使う足場を持っていた。
- ただし、stdoutにtimed total-throughput fieldsがなく、JSONL parserにも未接続だった。

今回の変更点:

- `package-self-attn-mlp-block-model-loop-smoke` に次のkey-value fieldsを追加した。
  - `prefill_total_input_tokens`
  - `decode_total_generated_tokens`
  - `end_to_end_total_tokens`
  - `prefill_wall_ms`
  - `decode_wall_ms`
  - `total_wall_ms`
  - `prefill_total_input_tps`
  - `decode_total_generated_tps`
  - `end_to_end_total_tps`
  - `layers_csv`
  - `prompt_tokens_csv`
  - `max_new_tokens_csv`
  - `total_tokens_csv`
  - `generated_tokens_csv`
  - `decode_batch_ready_counts_csv`
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` を追加し、model-loop key-value stdoutを `inference-benchmark-result-v0.1` JSONLへ変換できるようにした。
- R9700で layers `3,7`、sequence_len `3` のsmokeを実行した。
  - `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-model-loop-hybrid-throughput-smoke-v1.md`
- この行は `batching.mode=hybrid`、`prefill_real_batch=false`、`decode_real_batch=true`、`decode_executor_request_parallelism=2` を保存した。

実測値:

| layers | requests | prefill real | decode real | decode request parallelism | prefill total tok/s | decode generated tok/s | end-to-end tok/s | verified |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `3,7` | 3 | false | true | 2 | 78.702126 | 78.214266 | 78.492300 | true |

次の行動:

1. このhybrid rowはselected-layer stack guardとして扱う。full language-model SQ/vLLM比較には使わない。
2. 次はtoken-id full package pathとmodel-loop stack runnerの接続点を作り、embedding、all selected runtime layers、final norm/lm_head、quality guardを同じrequest-batch scheduler上で扱う。
3. prefillもrequest-batch化できた段階で `batching.mode=real` へ昇格する。decodeだけrealの間は `hybrid` として区別する。

## 2026-07-08 progress: T1 token-id model-loop bridge

前回の要点:

- selected-layer model-loop smokeはscheduler、KV cache、decode ready batch、throughput key-value stdoutを通せるようになった。
- ただし入力はsynthetic residualであり、SQ候補のdriftを見るためのtoken-id embedding入力とfinal lm_head guardには未接続だった。
- full-package real request-batch throughputはまだ未完了なので、selected-layer bridgeは中間gateとして扱う必要があった。

今回の変更点:

- `package-token-ids-model-loop-smoke` を追加した。
- prompt token ID batchを `model.language_model.embed_tokens.weight` のrowへ変換し、model-loop schedulerの初期residualとして使う。
- decode側は固定のsynthetic future token IDをembedding rowとして追加する。これはgreedy generationではなく、scheduler、selected runtime layers、final lm_head guardを接続するための中間gateである。
- final hiddenにfinal RMSNormとlm_head top-kをかけ、requestごとのfinal top1 tokenをstdoutとJSONLへ保存した。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` は `input_source`、`resolved_prefill_executor`、`final_top1_tokens` を保持するようにした。
- R9700 AQ4 packageで layers `3,7`、`batch=2`、prompt `2`、generated `1` のsmokeを実行し、`batching.mode=hybrid`、`decode_real_batch=true`、`final_top1_tokens=155793,23175`、`verified=true` を保存した。
- 結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-token-id-model-loop-hybrid-smoke-v1.md` に保存した。

次の行動:

1. このrowはselected-layer T1/T2 bridgeとして扱う。full LM throughputやSQ最終性能判断には使わない。
2. 次はSQ overlayまたはcandidate policyをこのtoken-id model-loop pathへ接続し、AQ4/SQのfinal top1、top-k overlap、logit gap、throughputを同じscheduler pathで比較する。
3. prefillのrequest-batch化は未完了なので、`batching.mode=hybrid` のまま区別する。full-package real batch runnerはT1aとして継続する。

## 2026-07-09 progress: T1 token-id model-loop real-prefill bridge

前回の要点:

- token-id model-loop bridgeは、token ID embedding入力、selected runtime layers、decode ready batch、final lm_head top1 guardを接続できた。
- ただしprefillはrequestごとの逐次実行であり、`prefill_real_batch=false`、`batching.mode=hybrid` のままだった。
- SQ候補のbatch throughput評価では、prefillもrequest batchとして流れる行が必要である。

今回の変更点:

- decoder layer runnerにprefill batch input helperとprefill batch runner APIを追加した。
- `package-token-ids-model-loop-smoke` のprefillを、layerごと・timestepごとに実行可能requestをまとめる `stack_prefill_request_batch_step` へ変更した。
- stdoutとJSONLに `prefill_batch_request_counts_csv` を追加し、parserは `batching.prefill_batch_request_counts` として保持する。
- R9700 AQ4 packageで layers `3,7`、`batch=2`、prompt `2`、generated `1` のsmokeを再実行した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-token-id-model-loop-real-prefill-smoke-v1.md` に保存した。

実測値:

| layers | requests | prefill real | decode real | prefill request parallelism | decode request parallelism | prefill total tok/s | decode generated tok/s | end-to-end tok/s | final top1 | verified |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `3,7` | 2 | true | true | 2 | 2 | 85.722441 | 84.560571 | 85.331620 | `155793,23175` | true |

次の行動:

1. このrowはselected-layer T1/T2 bridgeとして扱う。full LM throughputやSQ最終性能判断には使わない。
2. 次はSQ overlayまたはcandidate policyをこのtoken-id model-loop pathへ接続し、AQ4/SQのfinal top1、top-k overlap、logit gap、throughputを同じscheduler pathで比較する。
3. full-package real batch runnerはT1aとして継続し、最終的なAQ4/SQ/vLLM比較にはfull-package real batch行を使う。

## 2026-07-09 progress: T2 SQ FP8 token-id model-loop bridge

前回の要点:

- T1ではtoken-id model-loop pathがrequest-batch prefillとdecode ready batchを通すようになった。
- T2では `kup6_gate5_down5` policy artifactをruntimeへ渡す経路はmaterialize/logits smokeで確認済みだった。
- まだSQ FP8 policy artifactをtoken-id model-loop pathへ接続していなかった。

今回の変更点:

- `sq-fp8-token-ids-model-loop-smoke` を追加した。
- `Qwen3PackageModelRuntime::load_with_sq_overlay` を使い、`/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` をselected-layer model-loop pathへ接続した。
- stdoutとJSONLに `sq_overlay`、`sq_candidate`、`sq_artifact`、`sq_fp8_tensor_count`、`sq_passthrough_tensor_count`、`sq_row_chunk` を保存するようにした。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` はSQ overlay metadataを `workload` に保持する。
- R9700で layers `3,7`、`batch=2`、prompt `2`、generated `1` のSQ FP8 selected-layer smokeを実行し、AQ4と同じfinal top1 `155793,23175` を確認した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-real-prefill-smoke-v1.md` に保存した。

実測値:

| row | final top1 | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | verified |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| AQ4 real-prefill bridge | `155793,23175` | true | true | 85.722441 | 84.560571 | 85.331620 | true |
| SQ FP8 real-prefill bridge | `155793,23175` | true | true | 99.526151 | 99.900180 | 99.650516 | true |

注意:

- このrowはselected-layer bridgeであり、full LM throughputではない。
- 内部tok/sはlayer loadとSQ artifact materializationを含まない。wrapper elapsedは `33.426s` だった。
- token数が小さいので、速度の最終判断ではなく、同じscheduler pathでSQ品質guardとthroughput schemaを保存できたことを成果とする。

次の行動:

1. 既存のlen4/case_a/case_b prompt bundleをこのSQ model-loop pathに接続し、AQ4/SQのtop1、top-k overlap、AQ4 top1 rank、logit gapを保存する。
2. prompt-bundle rowでも `batching.mode=real` と `prefill_real_batch=true` を維持する。
3. full-package real batch runnerはT1aとして継続し、最終比較にはfull-package real batch行を使う。

## 2026-07-08 current plan update: SQ format design phase v1

前回の要点:

- R9700/RDNA4のcached-prefix/cold-prefill componentは、SQ候補評価を始める前提速度として一旦十分と判断した。
- T1はlogical full-package gridとselected-layer hybrid model-loop smokeまで進んだが、real full-package request-batch throughputはまだ未完了である。
- T2は `kup6_gate5_down5` の6層strict-top1 regression subsetと、実FP8 payload artifactのmaterialize smokeまで進んだ。

今回の変更点:

- 以後の主線を、追加attention kernel開発ではなくSQ format design/evaluationへ移す。
- full-package real batch runnerは最終性能比較に必要だが、SQ候補探索の開始blockerにはしない。
- SQ候補探索は、品質境界、format候補、速度/メモリ測定の3本を並行して進める。
- `kup6_gate5_down5` はbaseline candidateではなく「現在壊れていない最小regression subset」として扱う。これを起点に、より広いfamily/layer coverageへ広げるか、別scale/layoutへ進むかを判断する。
- SQ formatの候補軸を次のように固定する。
  - `sq-fp8-w8a16-r9700-v0`: FP8 weight + BF16/F32 activation、row-block F32 scale。現在の基準候補。
  - `sq-fp8-w8a16-r9700-v1-scale16`: FP8 weight + FP16/BF16 scale。F32 scaleのbyte/帯域を減らせるかを見る候補。
  - `sq-fp8-w8a16-r9700-v1-scale8`: FP8 weight + FP8 scale。品質劣化とscale復号overheadが許容できるかだけを見る実験候補。
  - `sq-fp8-w8a8-r9700-v0`: FP8 weight + FP8 activation。R9700 native経路でthroughputが伸びる場合だけ進める候補。
  - `sq-fp8-hybrid-r9700-v0`: risky family/layerをAQ4またはhigher precision fallbackに残す保守候補。
- 品質guardは当面strict top1を正式条件にする。top-k overlap、AQ4 top1 rank、logit gap、短文生成結果は診断として残すが、strict top1 failureを自動承認しない。
- speed評価では、overlay host materialize/load timingをSQ速度として読まない。native FP8、materialization-aware runtime path、またはmaterialized working-setを明示したpathだけを比較対象にする。

次の行動:

1. `sq-fp8-kup6-gate5-down5-policy-v0.1.json` を基準に、候補matrixを機械可読なmanifestへ落とす。各候補にはquantized tensor family、fallback family、scale dtype/layout、row-block幅、resident bytes、working-set bytesを必ず持たせる。
2. 品質探索を優先して、`kup6_gate5_down5` から広げる方向と、scale/layoutを強める方向を分けて試す。少なくともlen4/case_a/case_bに加え、もう少し長いtoken-id prompt bundleを追加する。
3. selected-layer stackでは、token-id embedding入力、selected runtime layers、final norm/lm_head、quality guardを同じscheduler pathへ接続する。これはfull LMではないが、SQ候補のdriftとthroughputを同じ経路で見るための中間gateにする。
4. full-package real batch runnerはT1aとして継続する。`batch=1/4/8`、prompt `512`、generated `128` のAQ4/FP8比較行を最初の保存gridにする。
5. prefill/cached-prefixは、既存の `cached_prefix_rdna4_fp8_auto` とcold causal flash2系の結果を基準にする。SQ候補のformat差が見えないcaseだけ、追加kernel最適化へ戻る。
6. throughput比較表は、AQ4、SQ候補、必要ならvLLM参考行を同じ列で並べる。列はquality、resident bytes、working-set bytes、VRAM peak、prefill total input tok/s、decode total generated tok/s、end-to-end tok/s、batching mode、executor/resolved executorにする。
7. SQ候補1の採否は、AQ4比でVRAM/working-set削減が説明でき、quality guardが通り、decode/prefill throughputが大きく悪化しないことを条件にする。満たせない場合は、FP8 scale、fallback増加、row-block幅変更、または別SQ候補へ進む。

## 2026-07-08 progress: SQ FP8 format candidate matrix v0.1

前回の要点:

- SQ format design phaseでは、`kup6_gate5_down5` を起点に候補matrixを機械可読manifestへ落とす必要があった。
- full-package real batch runnerは最終性能比較には必要だが、SQ候補探索の開始blockerにはしない方針にした。
- overlay host materialize/load timingはSQ速度として読まない方針にした。

今回の変更点:

- `tools/build-sq-fp8-candidate-matrix.py` を追加した。
- 現在の `sq-fp8-kup6-gate5-down5-policy-v0.1.json` と `sq-fp8-kup6-gate5-down5-policy-artifact-v0.1.json` から、候補matrixを再生成できるようにした。
- 生成物は `benchmarks/results/2026-07-08/sq-fp8-format-candidate-matrix-v0.1.json` と `.md` に保存した。
- matrixには次の候補を入れた。
  - `sq-fp8-w8a16-r9700-v0`
  - `sq-fp8-w8a16-r9700-v1-scale16`
  - `sq-fp8-w8a16-r9700-v1-scale8`
  - `sq-fp8-w8a8-r9700-v0`
  - `sq-fp8-hybrid-r9700-v0`
- matrixは、strict top1をquality promotion ruleとして保存し、top-k overlap、AQ4 top1 rank、logit gap、短文生成healthを診断専用にした。
- matrixは、`full_package_real_batch_required_for_final_comparison=true` と `full_package_real_batch_blocks_candidate_exploration=false` を同時に保存する。
- `sq-r9700-state-freeze-v0.1.{json,md}` に現在の候補matrixとして追記した。

次の行動:

1. `scale16` を実験する場合は、artifact builderへscale dtype optionを追加し、同じstrict top1 prompt bundleを再実行する。
2. `scale8` はscale16の品質とruntime overheadが見えてから、risk probeとして実施する。
3. `W8A8` はW8A16のquality guardとselected-layer throughput pathが安定してから進める。
4. 先にselected-layer stackへtoken-id embedding、final norm/lm_head、quality guardを接続し、SQ候補のdriftとthroughputを同じscheduler pathで見る。

## 2026-07-09 progress: T2 SQ FP8 token-id model-loop prompt bundle

前回の要点:

- T2では `sq-fp8-token-ids-model-loop-smoke` を追加し、`kup6_gate5_down5` SQ FP8 artifactをtoken-id model-loop request-batch prefill pathへ接続した。
- 直前のsmokeは layers `3,7`、batch `2`、prompt `2` の小さい接続確認で、AQ4/SQのfinal top1は一致していた。
- 次は既存の `len4`、`case_a`、`case_b` prompt bundleを同じscheduler pathへ流し、top1だけでなくtop-k overlap、AQ4 top1 rank、logit gapを保存する必要があった。

今回の変更点:

- `PackageModelLoopSmokeRun` のstdoutに `final_topk_tokens_csv` と `final_topk_logits_csv` を追加した。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` は、それらを `workload.final_topk_tokens` と `workload.final_topk_logits` として保持する。
- R9700でAQ4/SQを同じ条件で実行した。条件は layers `3,7,11,15,19,23`、batch `3`、top-k `8`、LM head chunk rows `4096`、prompt bundle `len4/case_a/case_b` である。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1.md` と `comparison.json` に保存した。

実測値:

| row | batching | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | wrapper elapsed s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AQ4 | real | true | true | 33.044260 | 32.565648 | 32.981036 | 5885833216 | 5.223 |
| SQ FP8 W8A16 | real | true | true | 32.377649 | 32.021248 | 32.330712 | 5885886464 | 92.056 |

Quality:

| case | AQ4 top1 | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common |
| --- | ---: | ---: | --- | ---: | ---: |
| len4 | 110784 | 102446 | false | 3 | `6 / 8` |
| case_a | 237950 | 111791 | false | 2 | `4 / 8` |
| case_b | 182949 | 182949 | true | 1 | `6 / 8` |

判断:

- このselected-layer model-loop guardは、以前のdirect logits prompt-bundle guardより厳しい。token-id embedding、6 selected runtime layers、request-batch prefill、decode ready batch、final norm、lm_headを一つの経路で通すと、`kup6_gate5_down5` は `len4` と `case_a` でstrict top1を維持しなかった。
- AQ4 top1は3ケースすべてSQ top8内に残っているので、regression subsetとしては有用である。
- 現行のstrict top1 promotion ruleでは、`kup6_gate5_down5` はSQ quality policyへ昇格しない。
- SQ wrapper elapsedはartifact read/materializationを含む。内部tok/sはload後のmodel-loop区間なので、速度比較ではこの2つを分けて読む。

次の行動:

1. `kup6_gate5_down5` はselected-layer regression subsetとして維持し、promoted SQ policyとは扱わない。
2. 次のT2品質探索では、model-loop top1 driftを起こすFP8 coverageを削る方向、またはscale/layout候補を変える方向を優先する。
3. 以後のAQ4/SQ比較では `final_topk_tokens` / `final_topk_logits` を保存し、top1だけで判断しない。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop coverage reduction

前回の要点:

- `kup6_gate5_down5` はdirect logits prompt bundleでは通ったが、token-id model-loop prompt bundleでは `len4` と `case_a` が崩れた。
- 現在の正式な昇格条件は、direct logitsではなくtoken-id model-loop上のstrict top1である。
- 次は、どのFP8 coverageがmodel-loop top1 driftを起こすかを切り分ける必要があった。

今回の変更点:

- k/up row-block32のcoverageを `6 -> 5 -> 4 -> 3 -> 2 -> 1` layerへ削り、R9700で同じ6-layer token-id model-loop prompt bundleを再実行した。
- layer3について `k_proj` 単体、`up_proj` 単体、`k_proj` row-block16、`k/up` row-block16を追加評価した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-coverage-reduction-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kup6-rowblock32` | 12 | 2 / 3 | 110784 | 193706 | 4 | 182949 |
| `kup5-rowblock32` | 10 | 2 / 3 | 110784 | 193706 | 5 | 182949 |
| `kup4-rowblock32` | 8 | 2 / 3 | 110784 | 193706 | 5 | 182949 |
| `kup3-rowblock32` | 6 | 2 / 3 | 110784 | 193706 | 4 | 182949 |
| `kup2-rowblock32` | 4 | 2 / 3 | 110784 | 193706 | 3 | 182949 |
| `kup1-layer3-rowblock32` | 2 | 2 / 3 | 110784 | 124170 | 4 | 182949 |
| `k-layer3-rowblock32` | 1 | 2 / 3 | 110784 | 111791 | 2 | 182949 |
| `up-layer3-rowblock32` | 1 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `k-layer3-rowblock16` | 1 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `kup1-layer3-rowblock16` | 2 | 2 / 3 | 110784 | 193706 | 3 | 182949 |

判断:

- k/up row-block32は、coverageをlayer3だけまで削っても `case_a` が崩れる。
- `up_proj` layer3 row-block32と `k_proj` layer3 row-block16は単体ではstrict top1を維持する。
- ただし `k/up` layer3 row-block16は組み合わせると `case_a` が崩れるため、単体probeのpassをそのままpolicy passとは扱えない。
- 次のT2では、artifact builder/runtime metadataにper-family/per-tensor scale-layoutを持たせ、`k_proj` row-block16 + `up_proj` row-block32のような混合scale policyをmodel-loop guardで試す。
- このselected-layer model-loopのtok/sとVRAMは診断用であり、full-package SQ throughputではない。wrapper elapsedにはartifact read/materializationが含まれる。

次の行動:

1. model-loop prompt bundleをSQ quality promotion gateとして継続する。
2. per-family/per-tensor scale-layoutをartifact manifestとbuilderに追加する。
3. 混合scale policyを作った後、`k_proj` と `up_proj` の組み合わせを最小coverageから再評価する。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop mixed scale

前回の要点:

- coverage削減では、k/up row-block32はlayer3だけでも `case_a` が崩れた。
- `up_proj` layer3 row-block32と `k_proj` layer3 row-block16は単体ではstrict top1を維持した。
- `k/up` layer3 row-block16は組み合わせると崩れたため、同一artifact内でtensorごとにscale block幅を変える必要があった。

今回の変更点:

- `tools/build-sq-fp8-w8a16-artifact.py` がpolicy `scale.overrides[]` を読み、tensorごとに異なるscale layoutをmanifestへ保存できるようにした。
- mixed layoutではcandidate-level `scale_granularity=mixed`、`scale_layout=per_tensor` を保存し、各 `fp8_tensors[]` entryの `scale_granularity` と `scale_block_cols` をauthoritativeにした。
- `docs/specs/sq-fp8-artifact-v0.1.md` と `docs/words.txt` にper-tensor scale layoutを追記した。
- R9700で `k_proj` row-block16 + `up_proj` row-block32を、layer3とlayers `3,7` の2条件でmodel-loop prompt bundleにかけた。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-mixed-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `kup1-layer3-k16-up32` | 2 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `kup2-k16-up32` | 4 | 2 / 3 | 110784 | 193706 | 2 | 182949 |

判断:

- mixed `k16/up32` は、layer3単体では前回の `k/up` row-block16およびrow-block32の崩れを回復した。
- 同じmixed scaleをlayers `3,7` へ広げると `case_a` が崩れる。AQ4 top1はSQ top8 rank 2に残る。
- 現在の境界は、単一tensorのscale幅ではなくlayer coverageと累積interaction側に移った。
- 次はlayer7単体の `k16/up32`、またはlayer7の追加fallback/別scaleを試して、layer3とlayer7のどちらが主因かを分ける。

次の行動:

1. `kup1-layer3-k16-up32` はpassing mixed-scale probeとして保持するが、promoted SQ policyにはしない。
2. `kup2-k16-up32` をfailure guardとして残し、layer coverage interactionを次のT2対象にする。
3. 次の候補はlayer7単体 `k16/up32`、またはlayer7 `k_proj` / `up_proj` の片側fallbackである。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 isolation

前回の要点:

- `kup1-layer3-k16-up32` はstrict top1を `3 / 3` 維持した。
- `kup2-k16-up32` は、同じmixed scaleをlayers `3,7` へ広げると `case_a` が崩れた。
- 次はlayer7単体と、layer3 passing probeへlayer7の片側だけを足す切り分けが必要だった。

今回の変更点:

- `layer7 k16/up32`、`layer3 k16/up32 + layer7 k16`、`layer3 k16/up32 + layer7 up32` の3条件を作った。
- すべて同じR9700 six-layer token-id model-loop prompt bundleで評価した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-isolation-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-k16-up32` | 2 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `layer3-kup-plus-layer7-k16` | 3 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `layer3-kup-plus-layer7-up32` | 3 | 2 / 3 | 110784 | 193706 | 3 | 182949 |

判断:

- layer7単体の `k16/up32` はstrict top1を維持する。
- layer3 passing probeへlayer7 `k_proj` row-block16だけを足してもstrict top1を維持する。
- layer3 passing probeへlayer7 `up_proj` row-block32を足すと `case_a` が崩れる。
- 現在のT2境界は、layer7 `up_proj` とlayer3 k/up mixed-scale probeのinteractionに絞られた。
- `case_a` のAQ4 top1はSQ top8 rank `3` に残るが、strict top1 promotion ruleではfailure guardとして扱う。

次の行動:

1. `layer7-k16-up32` と `layer3-kup-plus-layer7-k16` はpassing probesとして保持する。
2. `layer3-kup-plus-layer7-up32` を現在のfailure guardとして残す。
3. 次はlayer7 `up_proj` のrow-block16、row-block64、またはfallbackを、layer3 k16/up32 + layer7 k16固定で試す。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 up scale

前回の要点:

- layer7 isolationでは、layer7単体の `k16/up32` と、layer3 k16/up32 passing probeへlayer7 `k_proj` row-block16だけを足した条件はstrict top1を維持した。
- layer3 k16/up32 passing probeへlayer7 `up_proj` row-block32を足すと `case_a` が崩れた。
- 直近の境界はlayer7 `up_proj` のscaleまたはfallbackだった。

今回の変更点:

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` について、fallback、row-block16、row-block64の3条件をpolicy artifact化した。
- 同じR9700 six-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-up-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | layer7 up policy | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-up-fallback` | fallback | 3 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `layer7-up16` | row-block16 | 4 | 2 / 3 | 110784 | 193706 | 2 | 182949 |
| `layer7-up64` | row-block64 | 4 | 2 / 3 | 110784 | 193706 | 2 | 182949 |

判断:

- layer7 `up_proj` fallbackは、layer3 k16/up32 + layer7 k16のpassing subsetを維持した。
- layer7 `up_proj` row-block16とrow-block64は、どちらも `case_a` でAQ4 top1 `237950` からSQ top1 `193706` へ入れ替わった。
- どちらの失敗でもAQ4 top1はSQ top8 rank `2` に残るため、壊滅的崩壊ではなくranking driftとして扱う。
- 現在のT2 policy boundaryでは、layer3 `k_proj` row-block16、layer3 `up_proj` row-block32、layer7 `k_proj` row-block16をpassing subsetとして保持し、layer7 `up_proj` はfallbackに残す。

次の行動:

1. `layer7-up-fallback` を現在のpassing boundaryとして保持する。
2. `layer7-up16` と `layer7-up64` をfailure guardsとして残す。
3. 次はこのpassing subsetを基準に、追加family/layerを1つずつ戻して `case_a` driftが再発する境界を探す。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 add family

前回の要点:

- layer7 `up_proj` scale probeでは、layer3 k16/up32 + layer7 k16のpassing subsetを維持するにはlayer7 `up_proj` fallbackが必要だった。
- 次のT2対象は、このpassing boundaryにlayer7の追加familyを1つずつ戻し、`case_a` driftが再発する境界を探すことだった。

今回の変更点:

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` はfallbackのまま、layer7 `o_proj` row-block32、`gate_proj` row-block32、`down_proj` row-block64を個別に追加した。
- `o32` と `gate32` が個別に通ったため、追加で `o32+gate32` の組み合わせも測った。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-add-family-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-plus-o32` | 4 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `layer7-plus-gate32` | 4 | 3 / 3 | 110784 | 237950 | 1 | 182949 |
| `layer7-plus-down64` | 4 | 2 / 3 | 110784 | 111791 | 2 | 182949 |
| `layer7-plus-o32-gate32` | 5 | 2 / 3 | 110784 | 193706 | 2 | 182949 |

判断:

- layer7 `o_proj` row-block32は単独追加で `3 / 3` strict top1を維持した。
- layer7 `gate_proj` row-block32も単独追加で `3 / 3` strict top1を維持した。
- layer7 `down_proj` row-block64は `case_a` で `237950` から `111791` へ入れ替わった。
- layer7 `o32+gate32` は単独pass同士の組み合わせだが、`case_a` で `193706` へ入れ替わった。
- 現在のboundaryでは、layer7 `up_proj` と `down_proj` はfallback維持、`o_proj` と `gate_proj` は片方ずつならpassing、同時追加はfailure guardである。

次の行動:

1. `layer7-plus-o32` と `layer7-plus-gate32` はpassing probesとして保持する。
2. `layer7-plus-down64` と `layer7-plus-o32-gate32` はfailure guardsとして残す。
3. 次は `o32+gate32` の組み合わせをより強いscale/layoutで回復できるか試すか、`o32` または `gate32` の片側branchでcoverageを広げる。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 o/gate scale

前回の要点:

- layer7 add-family probeでは、layer7 `o_proj` row-block32と `gate_proj` row-block32は個別に `3 / 3` strict top1を維持した。
- ただし `o32+gate32` の同時追加は `case_a` で崩れた。
- 次の確認は、`o/gate` の同時追加がrow-block幅の強化で回復するかを見ることだった。

今回の変更点:

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` と `down_proj` はfallbackのまま、`o/gate` の組み合わせを `o16+gate32`、`o32+gate16`、`o16+gate16` で評価した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-ogate-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-ogate-o16-gate32` | 5 | 2 / 3 | 110784 | 193706 | 2 | 182949 |
| `layer7-ogate-o32-gate16` | 5 | 2 / 3 | 110784 | 193706 | 2 | 182949 |
| `layer7-ogate-o16-gate16` | 5 | 2 / 3 | 110784 | 193706 | 2 | 182949 |

判断:

- `o16+gate32`、`o32+gate16`、`o16+gate16` はすべて `case_a` が `193706` へ入れ替わった。
- 失敗時もAQ4 top1 `237950` はSQ top8 rank `2` に残るため、壊滅的崩壊ではなくranking driftである。
- row-block16化だけでは、layer7 `o_proj` と `gate_proj` の同時追加は回復しない。
- 現在のT2境界では、`o_proj` と `gate_proj` は片方ずつのbranch候補として扱い、同時追加はfailure guardに残す。

次の行動:

1. `o+gate` 同時追加は現行W8A16/F32 row-block scaleではfailure guardとして保持する。
2. 次は `o32` branchまたは `gate32` branchのどちらかを選び、coverageを広げる。
3. `o+gate` 同時追加を回復する場合は、row-block幅ではなく別scale layout、別dtype、またはtext-level acceptance guardの導入後に再評価する。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 o32 branch layer11

前回の要点:

- layer7 add-family probeでは、layer7 `o_proj` row-block32と `gate_proj` row-block32は個別に `3 / 3` strict top1を維持した。
- `o32+gate32` は、row-block16化しても `case_a` のranking driftを回復しなかった。
- `case_a` のtop1 marginは `o32` branchの方が `gate32` branchより広かったため、coverage拡張は `o32` branchから進める判断だった。

今回の変更点:

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16 + layer7 `o_proj` row-block32を固定した。
- layer11 `k_proj` row-block16を追加した5 tensor policyと、layer11 `k_proj` row-block16 + `o_proj` row-block32を追加した6 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer11-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-o32-plus-layer11-k16` | 5 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 33.188099 | 32.853465 |
| `layer7-o32-plus-layer11-k16-o32` | 6 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 31.492675 | 32.453352 |

判断:

- layer7 `o32` branchにlayer11 `k_proj` row-block16を足した5 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- layer11 `o_proj` row-block32も追加した6 tensor policyも、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32` まで広げられる。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 6 tensor版をpassing branchとして保持し、5 tensor版はrollback guardとして残す。
2. 次は同じ `o32` branchでlayer15の `k_proj` row-block16、必要なら `o_proj` row-block32を追加して、どこでstrict top1が崩れるかを見る。
3. layer7 `gate32` branchや `o32+gate32` 回復は、layer方向の広がりを一度見た後に戻る。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 o32 branch layer15

前回の要点:

- layer7 `o32` branchにlayer11 `k_proj` row-block16と `o_proj` row-block32を足した6 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- 次のT2対象は、同じ `o32` branchでlayer15 `k_proj` row-block16、必要ならlayer15 `o_proj` row-block32を追加して、strict top1の境界を見ることだった。

今回の変更点:

- layer15 `k_proj` row-block16を追加した7 tensor policyを作成した。
- さらにlayer15 `o_proj` row-block32も追加した8 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer15-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-o32-layer11-o32-plus-layer15-k16` | 7 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 28.249777 | 28.294170 |
| `layer7-o32-layer11-o32-plus-layer15-k16-o32` | 8 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 32.938634 | 29.980802 |

判断:

- layer15 `k_proj` row-block16を足した7 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- layer15 `o_proj` row-block32も追加した8 tensor policyも、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32` + layer15 `k16/o32` まで広げられる。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 8 tensor版をpassing branchとして保持し、7 tensor版はrollback guardとして残す。
2. 次は同じ `o32` branchでlayer19の `k_proj` row-block16、必要なら `o_proj` row-block32を追加して、どこでstrict top1が崩れるかを見る。
3. layer7 `gate32` branchや `o32+gate32` 回復は、layer方向の広がりを一度見た後に戻る。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 o32 branch layer19

前回の要点:

- layer7 `o32` branchにlayer15 `k_proj` row-block16と `o_proj` row-block32を足した8 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- 次のT2対象は、同じ `o32` branchでlayer19 `k_proj` row-block16、必要ならlayer19 `o_proj` row-block32を追加して、strict top1の境界を見ることだった。

今回の変更点:

- layer19 `k_proj` row-block16を追加した9 tensor policyを作成した。
- さらにlayer19 `o_proj` row-block32も追加した10 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer19-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16` | 9 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 33.194584 | 32.799897 |
| `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16-o32` | 10 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 33.076310 | 32.841953 |

判断:

- layer19 `k_proj` row-block16を足した9 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- layer19 `o_proj` row-block32も追加した10 tensor policyも、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32` + layer15 `k16/o32` + layer19 `k16/o32` まで広げられる。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 10 tensor版をpassing branchとして保持し、9 tensor版はrollback guardとして残す。
2. 次は同じ `o32` branchでlayer23の `k_proj` row-block16、必要なら `o_proj` row-block32を追加して、どこでstrict top1が崩れるかを見る。
3. layer7 `gate32` branchや `o32+gate32` 回復は、layer方向の広がりを一度見た後に戻る。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop layer7 o32 branch layer23

前回の要点:

- layer7 `o32` branchにlayer19 `k_proj` row-block16と `o_proj` row-block32を足した10 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- 次のT2対象は、同じ `o32` branchでlayer23 `k_proj` row-block16、必要ならlayer23 `o_proj` row-block32を追加して、strict top1の境界を見ることだった。

今回の変更点:

- layer23 `k_proj` row-block16を追加した11 tensor policyを作成した。
- さらにlayer23 `o_proj` row-block32も追加した12 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer23-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16` | 11 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 32.943496 | 32.707575 |
| `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16-o32` | 12 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 33.056640 | 32.555004 |

判断:

- layer23 `k_proj` row-block16を足した11 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- layer23 `o_proj` row-block32も追加した12 tensor policyも、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/up32` + layer7/11/15/19/23 `k16/o32` まで広げられる。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 12 tensor版をpassing branchとして保持し、11 tensor版はrollback guardとして残す。
2. 次はlayer3 `o_proj` row-block32を追加して、selected-layer `k/o` branchの穴を埋められるかを見る。
3. その後にMLP familyをlayer単位またはfamily単位で戻す。layer7 `up/down/gate` と `o+gate` combined failureは引き続きfailure guardとして残す。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer3 o32

前回の要点:

- layer7 `o32` branchはlayer23まで `k_proj` row-block16と `o_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- ただしlayer3は `k_proj` と `up_proj` のみで、selected-layer `k/o` branchとしてはlayer3 `o_proj` が穴として残っていた。

今回の変更点:

- current 12 tensor branchにlayer3 `o_proj` row-block32を追加した13 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-o32-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | case_a AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `selected-layer-ko-plus-layer3-o32` | 13 | 3 / 3 | `110784,237950,182949` | 1 | 32.766710 | 30.752791 | 32.489193 |

判断:

- layer3 `o_proj` row-block32を足した13 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32` + layer7/11/15/19/23 `k16/o32` まで広げられる。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 13 tensor版をcurrent passing branchとして保持する。
2. 次はlayer3 `gate_proj` row-block32を追加して、layer3 MLP coverageを `up` から `up+gate` へ広げられるかを見る。
3. layer7 `up/down/gate` と `o+gate` combined failureは引き続きfailure guardとして残す。
4. full-package real batch throughputは引き続きT1aとして別に進める。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer3 gate scale

前回の要点:

- layer3 `o_proj` row-block32を追加した13 tensor branchは、3 promptすべてでAQ4 top1を維持した。
- 次のT2対象は、layer3 `gate_proj` を追加してlayer3 MLP coverageを `up` から `up+gate` へ広げられるかを見ることだった。

今回の変更点:

- current 13 tensor branchにlayer3 `gate_proj` row-block32を追加した14 tensor policyを作成した。
- `gate32` がlen4でstrict top1に失敗したため、layer3 `gate_proj` だけrow-block16へ狭めた `gate16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-gate-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-plus-layer3-o32-gate32` | 14 | 2 / 3 | `102446,237950,182949` | 3 | 1 | 1 | 28.610392 | 28.109230 |
| `selected-layer-ko-plus-layer3-o32-gate16` | 14 | 2 / 3 | `102446,237950,182949` | 3 | 1 | 1 | 33.121674 | 32.765115 |

判断:

- layer3 `gate_proj` row-block32を追加すると、len4のtop1が `110784` から `102446` に変わる。
- layer3 `gate_proj` row-block16でも同じlen4 failureは回復しない。
- case_a/case_bはAQ4 top1を維持し、len4でもAQ4 top1はSQ top8内の3位に残るが、T2 promotion ruleはstrict top1なのでpromoteしない。
- current passing branchは13 tensor版 `selected-layer-ko-plus-layer3-o32` のままとする。

次の行動:

1. 13 tensor版 `selected-layer-ko-plus-layer3-o32` をcurrent passing branchとして保持する。
2. layer3 `gate_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer3 `down_proj` row-block64を追加して、layer3 MLP output projectionをcurrent branchへ足せるかを見る。
4. gate coverageは、より強いscale/layoutまたはtext-level guardの扱いが決まるまでfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer3 down64

前回の要点:

- 13 tensor版 `selected-layer-ko-plus-layer3-o32` は `3 / 3` strict top1 passだった。
- layer3 `gate_proj` row-block32/16はどちらもlen4でstrict top1を壊した。
- 次のT2対象は、layer3 `down_proj` row-block64を追加してlayer3 MLP output projectionをcurrent branchへ足せるかを見ることだった。

今回の変更点:

- current 13 tensor branchにlayer3 `down_proj` row-block64を追加した14 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-down64-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-plus-layer3-o32-down64` | 14 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 33.091248 | 32.876952 |

判断:

- layer3 `down_proj` row-block64を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer7/11/15/19/23 `k16/o32` まで広げられる。
- layer3 `gate_proj` row-block32/16は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` をcurrent passing branchとして保持する。
2. layer3 `gate_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer11 `up_proj` row-block32を追加して、layer3以外のMLP入力側をcurrent branchへ足せるかを見る。
4. layer7 `up/gate/down` は既存failure guardがあるため、layer11以降のMLP familyを先に見る。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer11 up scale

前回の要点:

- 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` は `3 / 3` strict top1 passだった。
- 次のT2対象は、layer11 `up_proj` row-block32を追加してlayer3以外のMLP入力側をcurrent branchへ足せるかを見ることだった。

今回の変更点:

- current 14 tensor branchにlayer11 `up_proj` row-block32を追加した15 tensor policyを作成した。
- `up32` がlen4でstrict top1に失敗したため、layer11 `up_proj` だけrow-block16へ狭めた `up16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-up-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-plus-layer11-up32` | 15 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 28.741285 | 30.527447 |
| `selected-layer-ko-layer3-down64-plus-layer11-up16` | 15 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 32.863404 | 32.437353 |

判断:

- layer11 `up_proj` row-block32を追加すると、len4のtop1が `110784` から `102446` に変わる。
- layer11 `up_proj` row-block16でも同じlen4 failureは回復しない。
- case_a/case_bはAQ4 top1を維持し、len4でもAQ4 top1はSQ top8内の2位に残るが、T2 promotion ruleはstrict top1なのでpromoteしない。
- current passing branchは14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` のままとする。

次の行動:

1. 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` をcurrent passing branchとして保持する。
2. layer11 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer11 `down_proj` row-block64を追加して、layer11 MLP output projectionをcurrent branchへ足せるかを見る。
4. layer11 `up_proj` coverageは、より強いscale/layoutまたはtext-level guardの扱いが決まるまでfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer11 down64

前回の要点:

- 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` は `3 / 3` strict top1 passだった。
- layer11 `up_proj` row-block32/16はどちらもlen4でstrict top1を壊した。
- 次のT2対象は、layer11 `down_proj` row-block64を追加して同じMLP output projection branchを広げられるかを見ることだった。

今回の変更点:

- current 14 tensor branchにlayer11 `down_proj` row-block64を追加した15 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-down64-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-plus-layer11-down64` | 15 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 28.647323 | 28.333764 |

判断:

- layer11 `down_proj` row-block64を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layers 7/15/19/23 `k16/o32` まで広げられる。
- layer11 `up_proj` row-block32/16は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` をcurrent passing branchとして保持する。
2. layer11 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer15 `down_proj` row-block64を追加して、同じMLP output projection branchを広げられるかを見る。
4. layer7 `up/gate/down` と layer11 `up_proj` は既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer15 down64

前回の要点:

- 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` は `3 / 3` strict top1 passだった。
- current passing branchは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layers 7/15/19/23 `k16/o32` だった。
- 次のT2対象は、layer15 `down_proj` row-block64を追加して同じMLP output projection branchを広げられるかを見ることだった。

今回の変更点:

- current 15 tensor branchにlayer15 `down_proj` row-block64を追加した16 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer15-down64-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-down64` | 16 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 32.082053 | 27.504989 |

判断:

- layer15 `down_proj` row-block64を追加すると、`len4` のtop1がAQ4 `110784` からSQ `102446` に変わる。
- `case_a` と `case_b` はAQ4 top1を維持し、`len4` でもAQ4 top1はSQ top8内の2位に残る。
- T2 promotion ruleはstrict top1なので、この16 tensor branchはpromoteしない。
- current passing branchは15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` をcurrent passing branchとして保持する。
2. layer15 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer19 `down_proj` row-block64を追加して、同じMLP output projection branchを別レイヤーで広げられるかを見る。
4. layer7 `up/gate/down`、layer11 `up_proj`、layer15 `up/gate/down_proj` は既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer19 down64

前回の要点:

- 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` はcurrent passing branchとして維持している。
- layer15 `down_proj` row-block64は `len4` でstrict top1を壊したためfailure guardになった。
- 次のT2対象は、layer19 `down_proj` row-block64を追加して同じMLP output projection branchを別レイヤーで広げられるかを見ることだった。

今回の変更点:

- current 15 tensor branchにlayer19 `down_proj` row-block64を追加した16 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-down64-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-layer11-down64-plus-layer19-down64` | 16 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.060194 | 32.529504 |

判断:

- layer19 `down_proj` row-block64を追加すると、`len4` のtop1がAQ4 `110784` からSQ `102446` に変わる。
- `case_a` と `case_b` はAQ4 top1を維持し、`len4` でもAQ4 top1はSQ top8内の2位に残る。
- T2 promotion ruleはstrict top1なので、この16 tensor branchはpromoteしない。
- current passing branchは15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` をcurrent passing branchとして保持する。
2. layer19 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer23 `down_proj` row-block64を追加して、同じMLP output projection branchを別レイヤーで広げられるかを見る。
4. layer7 `up/gate/down`、layer11 `up_proj`、layer15 `up/gate/down_proj`、layer19 `up/gate/down_proj` は既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer23 down64

前回の要点:

- 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` はcurrent passing branchとして維持している。
- layer15/19 `down_proj` row-block64はいずれも `len4` でstrict top1を壊したためfailure guardになった。
- 次のT2対象は、layer23 `down_proj` row-block64を追加して同じMLP output projection branchを別レイヤーで広げられるかを見ることだった。

今回の変更点:

- current 15 tensor branchにlayer23 `down_proj` row-block64を追加した16 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer23-down64-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` | 16 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 29.307130 | 28.656054 |

判断:

- layer23 `down_proj` row-block64を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer23 `k16/o32/down64` + layers 7/15/19 `k16/o32` まで広げられる。
- layer15/19 `down_proj` row-block64は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` をcurrent passing branchとして保持する。
2. layer15/19 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer23 `up_proj` row-block32を追加し、必要ならrow-block16 recoveryを試す。
4. layer7 `up/gate/down`、layer11 `up_proj`、layer15/19 MLP familyは既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer23 up scale

前回の要点:

- 16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` は `3 / 3` strict top1 passだった。
- current passing branchは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer23 `k16/o32/down64` + layers 7/15/19 `k16/o32` だった。
- 次のT2対象は、layer23 `up_proj` row-block32を追加し、失敗時はrow-block16 recoveryを確認することだった。

今回の変更点:

- current 16 tensor branchにlayer23 `up_proj` row-block32を追加した17 tensor policyを作成した。
- `up32` がlen4でstrict top1に失敗したため、layer23 `up_proj` だけrow-block16へ狭めた `up16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer23-up-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer23-up32` | 17 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 31.149837 | 32.513167 |
| `selected-layer-ko-layer23-up16` | 17 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 32.062485 | 27.859754 |

判断:

- layer23 `up_proj` row-block32を追加すると、`len4` のtop1がAQ4 `110784` からSQ `102446` に変わる。
- layer23 `up_proj` row-block16でも同じlen4 failureは回復しない。
- `case_a` と `case_b` はAQ4 top1を維持し、`len4` でもAQ4 top1はSQ top8内の2位に残る。
- T2 promotion ruleはstrict top1なので、layer23 `up_proj` はpromoteしない。
- current passing branchは16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` をcurrent passing branchとして保持する。
2. layer23 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer23 `gate_proj` row-block32を追加して、layer23の残りMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up_proj`、layer15/19 MLP familyは既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer23 gate scale

前回の要点:

- 16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` はcurrent passing branchだった。
- layer23 `up_proj` row-block32/16はどちらもlen4でstrict top1を壊した。
- 次のT2対象は、layer23 `gate_proj` row-block32を追加して、layer23の残りMLP branchを確認することだった。

今回の変更点:

- current 16 tensor branchにlayer23 `gate_proj` row-block32を追加した17 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer23-gate-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer23-gate32` | 17 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 28.681630 | 28.382956 |

判断:

- layer23 `gate_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer23 `k16/o32/gate32/down64` + layers 7/15/19 `k16/o32` まで広げられる。
- layer23 `up_proj` row-block32/16は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer23 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer11 `gate_proj` row-block32を追加して、layer11側の残りMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up_proj`、layer15/19 MLP familyは既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer11 gate scale

前回の要点:

- 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer11 `up_proj` row-block32/16はどちらもlen4でstrict top1を壊した。
- 次のT2対象は、layer11 `gate_proj` row-block32を追加し、失敗時はrow-block16 recoveryを確認することだった。

今回の変更点:

- current 17 tensor branchにlayer11 `gate_proj` row-block32を追加した18 tensor policyを作成した。
- `gate32` がlen4でstrict top1に失敗したため、layer11 `gate_proj` だけrow-block16へ狭めた `gate16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-gate-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer11-gate32` | 18 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.163512 | 32.595261 |
| `selected-layer-ko-layer11-gate16` | 18 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 32.889847 | 32.551791 |

判断:

- layer11 `gate_proj` row-block32を追加すると、`len4` のtop1がAQ4 `110784` からSQ `102446` に変わる。
- layer11 `gate_proj` row-block16でも同じlen4 failureは回復しない。
- `case_a` と `case_b` はAQ4 top1を維持し、`len4` でもAQ4 top1はSQ top8内の2位に残る。
- T2 promotion ruleはstrict top1なので、layer11 `gate_proj` はpromoteしない。
- current passing branchは17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer11 `gate_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer15 `gate_proj` row-block32を追加して、layer15側のMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up/gate`、layer15/19 MLP family、layer23 `up_proj` は既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer15 gate scale

前回の要点:

- 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer15 `down_proj` row-block64はlen4でstrict top1を壊した。
- 次のT2対象は、layer15 `gate_proj` row-block32を追加して、layer15側のMLP branchを確認することだった。

今回の変更点:

- current 17 tensor branchにlayer15 `gate_proj` row-block32を追加した18 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer15-gate-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer15-gate32` | 18 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 28.562896 | 28.068268 |

判断:

- layer15 `gate_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer15 `k16/o32/gate32` + layer23 `k16/o32/gate32/down64` + layers 7/19 `k16/o32` まで広げられる。
- layer15 `down_proj` row-block64は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 18 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer15 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer15 `up_proj` row-block32を追加して、layer15の残りMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up/gate`、layer15 `down`、layer19 MLP family、layer23 `up_proj` は既存failure guardがあるためfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer15 up scale

前回の要点:

- 18 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-gate32-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer15 `gate_proj` row-block32はstrict top1を維持した。
- layer15 `down_proj` row-block64はlen4でstrict top1を壊したためfailure guardだった。

今回の変更点:

- current 18 tensor branchにlayer15 `up_proj` row-block32を追加した19 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer15-up-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer15-up32` | 19 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 33.142998 | 32.846824 |

判断:

- layer15 `up_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer15 `k16/o32/up32/gate32` + layer23 `k16/o32/gate32/down64` + layers 7/19 `k16/o32` まで広げられる。
- layer15 `down_proj` row-block64は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 19 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer15 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer19 `gate_proj` row-block32を追加して、layer19側のMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up/gate`、layer15 `down`、layer19 `up/down/gate`、layer23 `up_proj` は既存failure guardまたは未選択branchとしてfallbackに残す。

## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer19 gate scale

前回の要点:

- 19 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer19 `down_proj` row-block64はlen4でstrict top1を壊したためfailure guardだった。
- 次のT2対象は、layer19 `gate_proj` row-block32を追加して、layer19側のMLP branchを確認することだった。

今回の変更点:

- current 19 tensor branchにlayer19 `gate_proj` row-block32を追加した20 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-gate-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer19-gate32` | 20 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 28.118832 | 28.167761 |

判断:

- layer19 `gate_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer15 `k16/o32/up32/gate32` + layer19 `k16/o32/gate32` + layer23 `k16/o32/gate32/down64` + layer7 `k16/o32` まで広げられる。
- layer19 `down_proj` row-block64は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer19 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer19 `up_proj` row-block32を追加して、layer19の残りMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up/gate`、layer15 `down`、layer19 `up/down`、layer23 `up_proj` は既存failure guardまたは未選択branchとしてfallbackに残す。


## 2026-07-09 progress: T2 SQ FP8 model-loop selected-layer k/o layer19 up scale

前回の要点:

- 20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer19 `down_proj` row-block64はlen4でstrict top1を壊したためfailure guardだった。
- 次のT2対象は、layer19 `up_proj` row-block32を追加し、失敗時はrow-block16 recoveryを確認することだった。

今回の変更点:

- current 20 tensor branchにlayer19 `up_proj` row-block32を追加した21 tensor policyを作成した。
- row-block32がlen4でstrict top1を壊したため、layer19 `up_proj` row-block16 recoveryも実行した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-up-scale-v1.md` と `comparison.json` に保存した。

実測値:

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer19-up32` | 21 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.172968 | 29.662839 |
| `selected-layer-ko-layer19-up16` | 21 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.200339 | 32.691546 |

判断:

- layer19 `up_proj` row-block32/16はいずれもlen4でSQ top1が `102446` になり、strict top1を維持しなかった。
- AQ4 top1 `110784` はSQ top8内の2位に残るが、T2 promotion ruleはstrict top1なので、この21 tensor branchはpromoteしない。
- current passing branchは20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

次の行動:

1. 20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer19 `up_proj` row-block32/16とlayer19 `down_proj` row-block64をfailure guardとして残す。
3. 現在のselected-layer MLP probe setでは、追加でpromoteできる候補がほぼ尽きたため、次はT1 full-package real request-batch throughput runnerへ戻るか、T2をselected layer外へ広げる。

## 2026-07-09 progress: T1 self-attn stack real-batch small grid

前回の要点:

- T1のtoken-id model-loopはselected-layer bridgeとして動いていたが、full packageにはself-attention層とlinear-attention層が混在している。
- 既存runnerはmixed-attention full layer orderを推定できず、`all` をfull packageとして扱うことはできなかった。
- SQ throughput比較の前には、少なくともreal request-batch prefill/decode/end-to-end rowを増やす必要があった。

今回の変更点:

- `package-token-ids-model-loop-smoke` と `sq-fp8-token-ids-model-loop-smoke` に `all-self-attn` / `manifest-self-attn` aliasを追加した。
- aliasはmanifest内のself-attention `q_norm` / `k_norm` passthrough tensor集合からlayer indexを抽出する。
- R9700 AQ4 packageでmanifest self-attention 8層 `3,7,11,15,19,23,27,31` を `batch=1/4/8`、`prompt=4`、`generated=1` で測定した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-self-attn-stack-real-batch-small-grid-v1.md` に保存した。

実測値:

| batch | batching | prefill real batch | decode real batch | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `hybrid` | false | false | 74.066673 | 70.654751 | 73.358179 | 7435599872 |
| 4 | `real` | true | true | 73.537780 | 71.298348 | 73.078709 | 7435612160 |
| 8 | `real` | true | true | 73.326934 | 71.010893 | 72.851718 | 7563571200 |

判断:

- Self-attention stack上では、`batch=4/8` で `prefill_real_batch=true` と `decode_real_batch=true` を保存できた。
- この短い `prompt=4` 条件ではbatchを増やしてもtotal tok/sは伸びず、B=1からB=8までほぼ横ばいだった。
- これはmanifest self-attention層だけの中間rowであり、linear-attention層を含むQwen3.5-9B full mixed-attention LM throughputではない。
- 次のT1本命は、linear-attention層を含むfull mixed-attention package real-batch runnerである。

## 2026-07-09 progress: T1 full mixed layer kind inventory

前回の要点:

- self-attention stack real-batch rowは得られたが、full packageにはlinear-attention層が含まれる。
- full mixed-attention runnerを実装するには、manifest由来のlayer orderとlayer kindを先に固定する必要があった。
- logical full-package rowは既にあるが、`prefill_real_batch=false` / `decode_real_batch=false` のためSQ throughput比較には使えない。

今回の変更点:

- `package-layer-kind-inventory-smoke` を追加した。
- `manifest-all` aliasで、`.ullm.d` manifestからsupported layer indexを昇順に抽出できるようにした。
- `package-token-ids-logits-smoke`、`sq-fp8-token-ids-logits-smoke`、`package-token-ids-generate-smoke`、`package-batch-throughput-bench` も `manifest-all` を受け取れるようにした。
- R9700 AQ4 packageで32層連続、self-attention 8層、linear-attention 24層を確認した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-full-mixed-layer-kind-inventory-v1.md` に保存した。

Inventory:

| field | value |
| --- | --- |
| layers | `0..31` |
| contiguous | true |
| self-attention layers | `3,7,11,15,19,23,27,31` |
| linear-attention layers | `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30` |
| verified | true |

判断:

- `manifest-all` はfull mixed-attention runnerのlayer order入力として使える。
- 次の実装対象は、linear-attention層のper-request recurrent stateとcausal Conv1d historyを持つreal request-batch ownerである。
- このinventoryはthroughput rowではないため、SQ性能比較には直接使わない。

次の行動:

1. full mixed-attention runnerではmanifest orderをそのまま使う。
2. self-attention層では既存のpaged KV state / ready-batch decode入力を使う。
3. linear-attention層ではrequestごとのrecurrent stateとConv1d historyを保持する。
4. full packageで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを保存してからSQ候補比較へ進む。

## 2026-07-09 progress: T1 linear-attn request state owner

前回の要点:

- full mixed-attention layer orderは固定できた。
- 次の実装blockerは、linear-attention層のrecurrent stateとcausal Conv1d historyをrequestごとに分離することだった。
- 既存のlinear-attn resident step layerはsingle request向けだった。

今回の変更点:

- `PackageLinearAttnResidentStepBatchLayer` を追加した。
- `RequestId` からlinear-attn resident layer state slotへ解決するownerにした。
- 各slotは `PackageLinearAttnResidentStepLayer` を持ち、requestごとのrecurrent stateとConv1d historyを分離する。
- 空request listと重複request idを拒否するunit testを追加した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-linear-attn-request-state-owner-v1.md` に保存した。

判断:

- これはthroughput rowではなく、full mixed real-batch runnerへ接続するためのstate ownerである。
- 現時点ではweights共有ではなくrequest slotごとのresident layerを持つため、性能最終形ではshared resident weight + per-request state bufferへ寄せる必要がある。
- ただし、full mixed pathに必要な「linear-attn stateがrequest間で混ざらない」契約はコード上で固定できた。

次の行動:

1. full mixed-attention runnerのlayer enumへlinear-attn request-batch ownerを接続する。
2. 小さいB=2 / prompt=2 / generated=1で、full mixed pathの `prefill_real_batch=true` / `decode_real_batch=true` smokeを作る。
3. その後、weights共有とactual throughputの改善へ進む。

## 2026-07-09 progress: T1 linear-attn request state smoke

前回の要点:

- `PackageLinearAttnResidentStepBatchLayer` は追加済みだったが、unit test中心で、実package上の実行証拠はまだなかった。
- full mixed-attention runnerへ進むには、linear-attention層のrecurrent stateとcausal Conv1d historyがrequest間で混ざらないことを実行時にも確認する必要があった。

今回の変更点:

- `package-linear-attn-request-state-smoke` を追加した。
- R9700上で実packageのlinear-attention layer `0` を `request_count=2`、`sequence_len=2` でinterleaved実行した。
- batch ownerの出力を、requestごとに単体 `PackageLinearAttnResidentStepLayer` をロードし直したserial referenceと比較した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-linear-attn-request-state-smoke-v1.md` に保存した。

判断:

- `serial_reference_max_abs_diff=0.000000000`、`nonfinite_count=0`、`unknown_request_rejected=true` で、request state ownerが実package上でもstate分離guardとして動くことを確認した。
- このsmokeはthroughput rowではない。`interleaved_step_tps=80.056352` はsynchronous readback込みの小さいstate smoke値であり、SQ性能比較には使わない。
- 現在はrequest slotごとにresident weightsを複製するため、full package throughputの最終形ではshared resident weight + per-request state bufferへ寄せる必要がある。

次の行動:

1. full mixed-attention runnerのlayer enumにself-attention resident step layerとlinear-attn request-state ownerを並べる。
2. manifest order `0..31` の小さいB=2 / prompt=2 / generated=1 full mixed path smokeを作る。
3. full mixed smoke後に、weights共有とactual throughput改善へ進む。

## 2026-07-09 progress: T1 self-attn request state owner

前回の要点:

- linear-attention側のrequest-state ownerは実package smokeまで通った。
- full mixed-attention runnerではself-attention層とlinear-attention層を同じrequest-id dispatch形へ揃える必要がある。
- 既存の `PackageSelfAttnResidentStepLayer` はsingle request向けにpaged KV cache、written_len、block tableを内部保持していた。

今回の変更点:

- `PackageSelfAttnResidentStepBatchLayer` を追加した。
- `RequestId` からself-attn resident layer state slotへ解決するownerにした。
- 各slotは `PackageSelfAttnResidentStepLayer` を持ち、requestごとのpaged KV cache、written_len、block tableを分離する。
- request id slot index helperをlinear/selfで共通化し、self-attn側の空request listと重複request idを拒否するunit testを追加した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-self-attn-request-state-owner-v1.md` に保存した。

判断:

- これはthroughput rowではなく、full mixed real-batch runnerへ接続するためのstate ownerである。
- self-attention側とlinear-attention側のrequest-id dispatch形が揃った。
- 現時点ではweights共有ではなくrequest slotごとのresident layerを持つため、性能最終形ではshared resident weight + per-request state bufferへ寄せる必要がある。

次の行動:

1. full mixed-attention runnerのlayer enumに `PackageSelfAttnResidentStepBatchLayer` と `PackageLinearAttnResidentStepBatchLayer` を並べる。
2. 小さいB=2 / prompt=2 / generated=1で、manifest orderのfull mixed path smokeを作る。
3. full mixed smoke後に、weights共有とactual throughput改善へ進む。

## 2026-07-09 progress: T1 mixed request-state layer enum smoke

前回の要点:

- linear-attention側とself-attention側のrequest-state ownerは揃った。
- full mixed-attention runnerへ進むには、両ownerを同じlayer enumでdispatchし、linear-attn層からself-attn層へdevice bufferを渡せる必要があった。
- SQ throughput比較用のreal batch rowへ進む前に、小さいmixed pathでstateがrequest間に混ざらない境界を通す必要があった。

今回の変更点:

- `PackageMixedRequestStateLayer` を追加した。
- `package-token-ids-mixed-request-state-smoke` を追加した。
- R9700上で実packageのlayer `0,3` を `batch=2`、`prompt=2`、`generated=1` で実行した。
- token ID embedding入力、linear-attention request state、self-attention paged KV state、final RMSNorm、lm_head top1 guardまでを同じsmokeで接続した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-layer-enum-smoke-v1.md` に保存した。

実測値:

| field | value |
| --- | --- |
| layers | `0,3` |
| layer kinds | `linear_attention,self_attention` |
| batching mode | `request_state_interleaved` |
| throughput row | `false` |
| prefill real batch | `false` |
| decode real batch | `false` |
| prefill request counts | `2,2` |
| decode request counts | `2` |
| final top1 tokens | `151353,151353` |
| verified | `true` |

判断:

- mixed request-state layer enumは、小さいlinear-attn→self-attn pathでは動作した。
- これはthroughput rowではなく、full mixed real-batch runnerへ進む前のdispatch/state guardである。
- 現時点ではrequest slotごとにresident weightsとcache bufferを持つため、shared-weight final designではない。
- `prefill_real_batch=false` / `decode_real_batch=false` なので、SQ/vLLM throughput比較には使わない。

次の行動:

1. `0,3` guardから `manifest-all` へ広げ、full mixed layer orderで壊れないことを確認する。
2. full manifest smoke後に、shared resident weights + per-request state bufferへ寄せる。
3. その後、full packageで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを作る。

## 2026-07-09 progress: T1 mixed request-state manifest smoke

前回の要点:

- `PackageMixedRequestStateLayer` は `layers=0,3` の小さいlinear-attn to self-attn guardで動作した。
- full mixed-attention packageのmanifest order `0..31` が同じrequest-state dispatchで通るかは未確認だった。
- SQ throughput比較へ進むには、まずfull mixed layer orderが壊れないことを確認する必要があった。

今回の変更点:

- `package-token-ids-mixed-request-state-smoke` を `manifest-all` で実行した。
- R9700上でAQ4 full packageの32層をmanifest order通りに通した。
- linear-attention 24層とself-attention 8層を同じrequest-id dispatch境界でinterleaved実行した。
- final RMSNormとlm_head top1 guardまで到達することを確認した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-manifest-smoke-v1.md` に保存した。

実測値:

| field | value |
| --- | --- |
| layers | `0..31` |
| linear-attention layers | 24 |
| self-attention layers | 8 |
| batching mode | `request_state_interleaved` |
| throughput row | `false` |
| prefill real batch | `false` |
| decode real batch | `false` |
| final top1 tokens | `44370,5446` |
| layer load ms | `18416.054962` |
| total wall ms | `19055.161428` |
| verified | `true` |

判断:

- full mixed layer order `0..31` はrequest-state dispatchで通った。
- このrowはthroughput rowではない。
- 現在はrequest slotごとにresident weightsを複製しており、`layer_load_ms` が支配的で、real package throughputとは扱わない。
- `prefill_real_batch=false` / `decode_real_batch=false` のままなので、SQ/vLLM throughput比較には使わない。

次の行動:

1. request slotごとのresident weight複製をやめ、shared resident weights + per-request state/cache bufferへ寄せる。
2. full package pathで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを保存する。
3. その後、SQ FP8候補を同じworkload gridへ接続する。

## 2026-07-09 progress: T1 mixed request-state AQ4 payload sharing

前回の要点:

- `manifest-all` full mixed request-state smokeは32層全体で通った。
- ただしrequest slotごとにresident layerをロードしており、AQ4 payloadもslotごとに再ロードしていた。
- full package real throughputへ進むには、少なくとも同一layer内のrequest slot間でweight payloadを共有する必要があった。

今回の変更点:

- `PackageAq4ResidentMatvec::load` が、同じ `WeightRegistry` 内に既に同名tensorがある場合、そのloaded tensor bundleを再利用するようにした。
- `PackageLinearAttnResidentStepBatchLayer` と `PackageSelfAttnResidentStepBatchLayer` は、request slot生成時に同じ `WeightRegistry` を渡すようにした。
- smoke出力に `slot_aq4_payload_registry_shared=true` を追加した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-aq4-payload-sharing-v1.md` に保存した。

実測値:

| field | before | after |
| --- | ---: | ---: |
| final top1 tokens | `44370,5446` | `44370,5446` |
| layer load ms | 18416.054962 | 17828.586114 |
| total wall ms | 19055.161428 | 18452.035174 |
| slot AQ4 payload registry shared | n/a | `true` |
| verified | `true` | `true` |

判断:

- 同一layer内のrequest slot間でAQ4 index/scale/codebook runtime bufferを共有する最初の境界は通った。
- これは完全なshared resident weightsではない。RMSNorm/aux passthrough buffers、AQ4 `scale_values_buffer`、workspace、Conv1d history、recurrent state、paged KV cacheはまだslotごとに持っている。
- `prefill_real_batch=false` / `decode_real_batch=false` のままなので、SQ/vLLM throughput比較には使わない。

次の行動:

1. `scale_values_buffer` とpassthrough weight buffersをslot間で共有できるようにする。
2. workspace/state/cacheを分離したまま、weight-only resident bundleをlayerごとに1つへ寄せる。
3. request-batch stepをreal batch executorへ置き換えてfull package throughput rowを作る。

## 2026-07-09 progress: T1 mixed request-state scale/passthrough sharing

前回の要点:

- `manifest-all` full mixed request-state smokeは、AQ4 index/scale/codebook payload bufferのslot間共有まで通っていた。
- ただしAQ4 `scale_values_buffer`、row-scale buffer、RMSNorm/Conv1d/A_log/dt_biasなどのpassthrough weight bufferはまだslotごとに確保していた。
- full package real throughputへ進む前に、少なくとも不変weight bufferとrequest別state/cache/workspaceの境界を分ける必要があった。

今回の変更点:

- `PackageResidentSharedBufferRegistry` を追加し、同一batch layer内のrequest slot間でf32 runtime bufferを共有するようにした。
- `PackageAq4ResidentMatvec::load_with_shared_buffers` をbatch loaderから使い、AQ4 `scale_values_buffer` とrow-scale bufferもslot間共有へ寄せた。
- self-attention側はinput/q/k/post RMSNorm weight bufferを共有する。
- linear-attention側はinput/post RMSNorm、linear-attn norm、Conv1d weight、A_log、dt_bias bufferを共有する。
- smoke出力に `slot_aq4_scale_values_shared=true` と `slot_passthrough_weight_buffers_shared=true` を追加した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-scale-passthrough-sharing-v1.md` に保存した。

実測値:

| field | payload shared | scale/passthrough shared |
| --- | ---: | ---: |
| final top1 tokens | `44370,5446` | `44370,5446` |
| layer load ms | 17828.586114 | 16240.561131 |
| total wall ms | 18452.035174 | 16859.363507 |
| slot AQ4 payload registry shared | `true` | `true` |
| slot AQ4 scale values shared | n/a | `true` |
| slot passthrough weight buffers shared | n/a | `true` |
| verified | `true` | `true` |

判断:

- 同一layer内のrequest slot間でAQ4 payload、AQ4 scale/row-scale、主要passthrough weight bufferを共有する境界は通った。
- これはまだ完全なweight-only resident bundleではない。workspace、Conv1d history、recurrent state、paged KV cache、block tableはrequest slot別である。
- `prefill_real_batch=false` / `decode_real_batch=false` のままなので、SQ/vLLM throughput比較には使わない。

次の行動:

1. weight-only resident bundleをlayerごとに1つへ寄せ、slot別state/cache/workspaceとの境界を明示する。
2. request-batch stepをreal batch executorへ置き換えてfull package throughput rowを作る。
3. SQ候補をfull mixed pathへ接続する準備を続ける。

## 2026-07-09 progress: T1 mixed request-state weight-bundle sharing

前回の要点:

- `manifest-all` full mixed request-state smokeは、AQ4 payload、AQ4 scale/row-scale、主要passthrough weight bufferのslot間共有まで通っていた。
- ただしresident layer struct内ではweight fieldとstate/workspace fieldが混在しており、layerごとに1つのweight-only resident bundleという境界はまだ明示されていなかった。

今回の変更点:

- self-attention resident layerを `PackageSelfAttnResidentStepWeights` とrequest slot state/workspaceへ分けた。
- linear-attention resident layerを `PackageLinearAttnResidentStepWeights` とrequest slot state/workspaceへ分けた。
- batch layer loaderは1slot目でweight bundleを作り、2slot目以降は同じ `Arc<...Weights>` からstate/workspaceだけを作る。
- requestごとのpaged KV cache、block table、Conv1d history、recurrent state、workspaceは引き続きslot別に残した。
- smoke出力に `self_attn_weight_bundle_shared=true` と `linear_attn_weight_bundle_shared=true` を追加した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-weight-bundle-sharing-v1.md` に保存した。

実測値:

| field | scale/passthrough shared | weight bundle shared |
| --- | ---: | ---: |
| final top1 tokens | `44370,5446` | `44370,5446` |
| layer load ms | 16240.561131 | 10256.735321 |
| total wall ms | 16859.363507 | 10907.692026 |
| self-attn weight bundle shared | n/a | `true` |
| linear-attn weight bundle shared | n/a | `true` |
| verified | `true` | `true` |

判断:

- full mixed request-state pathで、layerごとに1つのweight-only resident bundleを作る境界は通った。
- これはまだreal batch throughput rowではない。実行は `batching_mode=request_state_interleaved` で、`prefill_real_batch=false` / `decode_real_batch=false` のままである。

次の行動:

1. request-state interleaved stepをreal request-batch executorへ置き換える。
2. full packageで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを保存する。
3. T2 SQ候補を同じfull mixed pathへ接続し、AQ4/SQ比較へ進む。

## 2026-07-09 progress: T2 SQ FP8 mixed request-state resident throughput

前回の要点:

- full mixed AQ4 `manifest-all` resident throughput baselineはB=1/4/8で取得済みだった。
- T2ではSQ FP8 candidateを同じfull mixed resident pathへ接続し、AQ4/SQのqualityとthroughputを同じschemaで比較する必要があった。

今回の変更点:

- `sq-fp8-token-ids-mixed-request-state-smoke` を追加した。
- full mixed request-state loaderへ `Qwen3PackageSqOverlay` を渡し、artifactに存在するtensorだけSQ FP8からF32 resident bufferへmaterializeするようにした。
- artifactに存在しないtensorは従来どおりAQ4 resident matvecへfallbackする。
- `PackageAq4ResidentMatvec` はAQ4 storageとSQ/F32 materialized storageを持てるようになった。
- stdoutに `sq_execution_mode=materialized_f32_fallback` を追加した。
- `run-external-benchmark.py` は `sq_execution_mode` をworkload metadataとして保持する。
- R9700で `kup6_gate5_down5` artifactをB=1/4/8のfull `manifest-all` で実行し、結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-mixed-request-state-resident-throughput-small-grid-v1.md` に保存した。

実測値:

| batch | mode | SQ prefill tok/s | SQ decode tok/s | SQ end-to-end tok/s | AQ4 end-to-end tok/s | AQ4 final top1 | SQ final top1 | top1 match |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `single` | 13.047930 | 24.609341 | 7.260317 | 8.926325 | `44370` | `44370` | `true` |
| 4 | `real` | 20.661755 | 24.890457 | 14.585119 | 24.096096 | `44370,5446,10701,25411` | `44370,1622,10701,25411` | `false` |
| 8 | `real` | 22.932277 | 25.066788 | 17.961835 | 34.577530 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,1622,10701,25411,21901,685,279,27973` | `false` |

判断:

- SQ FP8 candidateをfull mixed resident pathへ接続できた。
- B=4/B=8で2番目requestのfinal top1がAQ4 baselineからずれるため、`kup6_gate5_down5` はfull mixed quality guardを通過していない。
- 現在のSQ速度はmaterialized F32 fallbackを含むため、native SQ kernelの速度代表値ではない。

次の行動:

1. top1 driftが出ない保守的SQ candidateをfull mixed pathで再評価する。
2. SQ FP8 direct matvecまたは低遅延dequant matvecへ進む。
3. native SQ rowができたら、同じB=1/4/8 schemaでAQ4/SQ/vLLM比較へ戻る。

## 2026-07-09 progress: T2 SQ FP8 full mixed conservative candidate

前回の要点:

- `kup6_gate5_down5` はfull mixed B=4/B=8で2番目requestのfinal top1がAQ4 baselineからずれた。
- selected-layerで通ったcandidateでも、full mixed request-batch pathではstrict top1 guardを別途通す必要がある。

今回の変更点:

- R9700で `sq-fp8-w8a16-r9700-v0-k-layer3-rb16` をfull mixed `manifest-all` B=1/4/8で再評価した。
- `up-layer3` と `kup1-layer3-k16-up32` はB=4だけ確認し、full mixed expansionとしてrejectした。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-mixed-request-state-conservative-candidate-v1.md` と `results.jsonl` に保存した。

実測値:

| candidate | batch | SQ prefill tok/s | SQ decode tok/s | SQ end-to-end tok/s | AQ4 final top1 | SQ final top1 | top1 match |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `k-layer3-rb16` | 1 | 17.180309 | 80.069802 | 8.301199 | `44370` | `44370` | `true` |
| `k-layer3-rb16` | 4 | 48.691111 | 81.134326 | 25.144005 | `44370,5446,10701,25411` | `44370,5446,10701,25411` | `true` |
| `k-layer3-rb16` | 8 | 63.272579 | 81.162501 | 35.820128 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,5446,10701,25411,21901,685,279,27973` | `true` |
| `up-layer3` | 4 | 40.916537 | 64.087055 | 22.785325 | `44370,5446,10701,25411` | `44370,1622,10701,25411` | `false` |
| `kup1-layer3-k16-up32` | 4 | 36.929827 | 60.895996 | 20.726920 | `44370,5446,10701,25411` | `44370,1622,10701,25411` | `false` |

判断:

- 現在full mixed strict-top1 regression subsetとしてpromoteできる保守候補は、layer3 `k_proj` row-block16の1 tensorだけである。
- layer3 `up_proj` は単体でもB=4でtop1 driftを起こすため、`k+up` へ広げるcandidateはpromoteしない。
- B=4/B=8は `prefill_real_batch=true` / `decode_real_batch=true` で通っているため、full mixed request-batch guardとして扱う。
- 速度は引き続き `sq_execution_mode=materialized_f32_fallback` なので、SQ format本来の速度評価ではなくquality boundary確認として扱う。

次の行動:

1. `k-layer3-rb16` をfull mixed strict-top1 regression subsetとして固定する。
2. SQ FP8 direct matvecまたは低遅延dequant matvecを実装し、materialized F32 fallbackから外す。
3. native SQ rowができたら、B=1/4/8と長いprefill/prefix gridへ同じcandidateを流す。

## 2026-07-09 progress: T2 SQ FP8 direct matvec

前回の要点:

- full mixed strict-top1 regression subsetとしてpromoteできる保守候補は、layer3 `k_proj` row-block16の1 tensorだけだった。
- 直前の結果は `sq_execution_mode=materialized_f32_fallback` であり、SQ payloadをF32 resident bufferへ全展開していた。

今回の変更点:

- runtimeへ `ullm_runtime_sq_fp8_matvec_f32` を追加した。
- HIPRTC kernel `ullm_sq_fp8_matvec_f32_kernel` は、FP8 E4M3 payload byteとF32 scaleを直接読み、matvec内でdequantしてF32 outputへaccumulateする。
- `PackageAq4ResidentMatvec` に `SqFp8` storageを追加し、SQ overlay tensorはF32 materializationではなくpayload/scale resident bufferとして保持する。
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でR9700上のfull mixed B=1/4/8を実行し、HIP direct kernelが使われることを確認した。
- 結果は `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-direct-matvec-conservative-candidate-v1.md` と `results.jsonl` に保存した。

実測値:

| batch | SQ execution mode | SQ prefill tok/s | SQ decode tok/s | SQ end-to-end tok/s | F32 fallback end-to-end tok/s | AQ4 final top1 | SQ final top1 | top1 match |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `direct_fp8_dequant_matvec` | 20.373389 | 79.999546 | 9.083299 | 8.301199 | `44370` | `44370` | `true` |
| 4 | `direct_fp8_dequant_matvec` | 44.467487 | 81.012681 | 23.470577 | 25.144005 | `44370,5446,10701,25411` | `44370,5446,10701,25411` | `true` |
| 8 | `direct_fp8_dequant_matvec` | 63.262438 | 81.320426 | 35.598166 | 35.820128 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,5446,10701,25411,21901,685,279,27973` | `true` |

判断:

- SQ FP8 overlayの実行はF32 materialized bufferではなく、payload/scale resident bufferからのdirect dequant matvecへ移った。
- 保守候補のB=1/4/8 final top1はAQ4 baselineと一致した。
- 単一tensorだけの候補では、direct pathのend-to-end tok/sはmaterialized F32 fallbackとほぼ同等で、format性能の判断材料としてはまだ弱い。
- 速度改善には、SQ tensor数を増やしてもqualityが崩れない候補探索、またはSQ FP8 pair/triple/fused matvecへの拡張が必要である。

次の行動:

1. `SqFp8` storageをpair/triple/fused matvecへ広げ、AQ4 fused pathと同じ呼び出し粒度に近づける。
2. `up_proj` 系はrow-block幅、scale粒度、W8A8/activation scaleを変えて再探索する。
3. qualityが通るcandidate数が増えた段階で、B=1/4/8と長いprefill/prefix gridを再計測する。

## 2026-07-09 progress: T2 SQ FP8 direct matvec batch

前回の要点:

- SQ FP8 overlayはF32 materialized fallbackから、payload/scale resident bufferを読むdirect dequant matvecへ移った。
- ただし `SqFp8` storageは単発 `matvec` だけdirect pathで、`matvec_batch` はAQ4専用だった。
- SQ候補をprefill batch componentや将来のfull package batch runnerへ流すには、AQ4と同じbatch matvec API境界でSQ FP8を実行できる必要がある。

今回の変更点:

- runtimeへ `ullm_runtime_sq_fp8_matvec_batch_f32` を追加した。
- HIPRTC sourceへ `ullm_sq_fp8_matvec_batch_f32_kernel` を追加し、`grid.x=row`、`grid.y=batch` でbatch-major input/outputを処理する。
- Rust FFIへ `sq_fp8_matvec_batch_f32` を追加した。
- `PackageAq4ResidentMatvec::matvec_batch` は、`SqFp8` storageの場合にSQ FP8 batch direct kernelへdispatchする。
- CPU unit testと、`ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1` 付きHIP unit testでrow-block scaleのbatch出力を確認した。

判断:

- SQ FP8は単発matvecだけでなく、batch matvec API境界でもF32 materializeなしに実行できるようになった。
- これはpair/triple/fused projectionの完全対応ではないが、T1/T2のbatch throughput評価へSQ tensorを流すための足場になる。
- full mixed pathの速度改善には、次に `matvec_pair_with`、`matvec_triple_with`、MLP/linear-attn fused boundaryをSQ FP8 direct pathへ広げる必要がある。

次の行動:

1. `SqFp8` storageをpair/triple matvecへ広げ、self-attention Q/K/V projectionで単発kernel連打にならないようにする。
2. `matvec_silu_mul_with` などMLP fused境界は、qualityが通るSQ tensorが増えてから優先度を上げる。
3. SQ FP8 batch matvecを使うcomponentまたはpackage-level prefill rowを追加し、AQ4 batch matvecとの比較行を保存する。

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
