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
- 512 tokenまでのcomponent timingだけでは長コンテキストprefillの評価として不足するため、cold prefillとcached prefix付きprefillのworkload gridを追加する。
- 512 tokenの結果はshort sanityとlocal bottleneck検出には使えるが、SQ候補のprefill性能判断には使わない。prompt長、cached prefix長、新規chunk長、batch幅を変えた探索gridを追加し、結果に合わせてprefill kernelのtiling/blocking方針を更新する。

## 次の行動

1. R9700向けのbatch throughput result schemaを固定する。
2. logical batch runnerを先に作り、その後real batch kernelへ広げる。
3. FP8 SQ候補1のpackage/runtime prototypeを作る。
4. prefillをtoken-by-token実行からbatched/tiled実行へ移す。
5. long context向けに、`2^16` token級のcold prefillと、cached prefix `L` に対して新規chunk `M` を追加するcached prefillを測れるようにする。
6. 512 tokenを超える複数patternの結果から、causal attention、cached prefix attention、projection/MLP batchのどこを次に直すべきかを決める。
7. uLLM側でR9700のprefill/decodeが比較可能な速度になった後、vLLM baselineを同じworkload gridで測る。
8. FP8 K/V cacheの長prefix bottleneck解消として、R9700/RDNA4向けFlashAttention2-style tiled attentionを次の主タスクにする。

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

## 2026-07-08 next task: FlashAttention2 implementation

前回の要点:

- FP8 K/V cacheは専用変換命令により、R9700/RDNA4では変換そのもののoverheadをかなり削減できた。
- `L=4096` ではF32に近く、`L=16384` ではF32より速いが、`L=65536` ではまだF32より遅い。
- 残っている主因は、現cached-prefix kernelが長いprefixでK/Vをtile化せず、decoded K/Vやscoreを十分に再利用できていないことだと考える。

今回の変更点:

- 次の主タスクを、R9700/RDNA4向けFlashAttention2-style tiled attention実装にする。
- 対象はまずcached prefix prefillとcold prefillのattention componentに限定する。
- R9700/RDNA4では、FP8変換命令、BF16/FP16系演算、十分なLDS/波面制御を前提に、そのまま実装できるものとして進める。
- RDNA2/V620などBF16が使えない、またはFP8変換・scale変換の条件が異なる環境は今回の合否から外す。
- RDNA2向けには、必要ならFP32へのdequant機構、別accumulator path、またはfallback attention kernelを後続で設計する。今回は保留する。

次の行動:

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
