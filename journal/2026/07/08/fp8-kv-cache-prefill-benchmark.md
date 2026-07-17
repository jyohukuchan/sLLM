# FP8 KV cache cached-prefix prefill benchmark

## 前回の要点

- cached-prefix prefillは、既存KV cache長 `L` と新規chunk長 `M` を分けて計測する必要がある。
- 直前のcached-prefix attentionはF32 K/V cacheで計測していた。
- ユーザーは、KV cacheを標準でFP8にして同じprefill速度ベンチを取ることを許容した。

## 今回の変更点

- runtime C APIに `ullm_runtime_cached_prefix_attn_fp8_e4m3` を追加した。
- Rust FFIに `cached_prefix_attn_fp8_e4m3` を追加し、CPU/HIPテストを追加した。
- `runtime-cached-prefix-attn-smoke` に `kv_cache_dtype=fp8_e4m3|f32` を追加し、既定値を `fp8_e4m3` にした。
- FP8 K/V cacheはper-tensor scale付きE4M3 byte列、Q/outputはF32のままにした。
- R9700で `L={4096,16384,65536}`、`M={16,128,512}` のcached-prefix prefillをFP8/F32同一buildで測定した。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c4-fp8-e4m3-kv-cache-v1.md` に保存した。

## 観察

- FP8 K/V cacheは、同一shapeでF32比 `25%` のcache byteになった。
- sampled guardは全条件で通った。
- 速度は一様には改善しなかった。
  - `L=4096`: F32比 `0.90-0.93x`
  - `L=16384`: F32比 `1.10-1.29x`
  - `L=65536`: F32比 `0.49-0.58x`
- 長いprefixで遅い主因は、現cached-prefix kernelがFP8復号をK score計算とvalue accumulation内で繰り返すことだと考える。
- E4M3復号を `exp2f`、`switch`、FP32 bitcastで試した。`switch` は悪化し、bitcastは`exp2f`相当まで戻った。

## 次の行動

1. package decode state / paged decodeのK/V cacheはまだF32なので、必要なら別タスクでFP8化する。
2. cached-prefix FP8の速度改善は、decoded K/V tileの再利用、score再計算削減、FlashAttention系tilingを検討する。
3. SQ候補評価では、今回のFP8 cached-prefix結果を「byte削減は確認済み、速度はkernel構造依存」という基準線として扱う。

## 追記: FP8専用変換命令

### 前回の要点

- v1ではdevice側のFP8復号を手書きbit復号で行っていた。
- R9700/gfx1200にはFP8変換専用命令がある可能性が高かった。

### 今回の変更点

- `__builtin_amdgcn_cvt_f32_fp8` がgfx1200で `v_cvt_f32_fp8_e32` に落ちることを確認した。
- `__builtin_amdgcn_cvt_pk_f32_fp8` も `v_cvt_pk_f32_fp8_e32` に落ちるが、CK側にgfx12 compiler issueの注意があるため、今回は単体変換だけを使った。
- `__builtin_amdgcn_cvt_scalef32_f32_fp8` はROCm 7.2のgfx1200 targetで `fp8-cvt-scale-insts` feature不足により使えなかった。
- `runtime/src/ullm_runtime_hiprtc_sources.inc` のFP8 device変換を、gfx1200/gfx1201ではbuiltin、その他では従来のbit復号fallbackに変更した。
- R9700で同じ9条件をFP8/F32再測した。結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c4-fp8-e4m3-kv-cache-builtin-cvt-v2.md`。

### 観察

- FP8 pathはv1比で全条件改善した。
- `L=4096` はF32比 `0.98x` 付近まで戻った。
- `L=16384` はF32比 `1.25-1.37x` まで改善した。
- `L=65536` はまだF32比 `0.51-0.74x` に留まり、長prefixではkernel構造の問題が残る。

### 次の行動

1. 長prefixFP8の残りの遅さは、変換命令ではなくdecoded K/V再利用不足として扱う。
2. packed変換はCKの注意点を踏まえ、別途小さい正しさ検証とISA確認を行う。
3. cached-prefix attentionのtile化、score再計算削減、FlashAttention系実装を次の候補にする。

## 追記: 次タスクをFlashAttention2実装に設定

### 前回の要点

- FP8専用変換命令に置き換えても、`L=65536` の長prefixではF32より遅い状態が残った。
- 残りの問題は、K/V tileやscoreの再利用不足というkernel構造側にあると見ている。

### 今回の変更点

- `uLLM-project/docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` に、次の主タスクとしてFlashAttention2-style tiled attention実装を追記した。
- 対象はR9700/RDNA4に限定し、cached prefix prefillとcold prefillのattention componentから始める。
- RDNA2/V620などBF16が使えない環境やFP32 dequant fallbackは今回は保留にした。

### 次の行動

1. `runtime-cached-prefix-attn-smoke` にFlashAttention2 executorを追加する。
2. RDNA4向けにQ block x K/V tileのonline softmax kernelを実装する。
3. `L={4096,16384,65536}`、`M={16,128,512}` の保存gridでFP8/F32を再測する。

## 追記: cached-prefix flash2 FP8 kernel実装開始

### 前回の要点

- 公式/公開FlashAttention2はあるが、uLLMのFP8 K/V cacheとscale分離レイアウトにはそのまま合わない。
- 既存のFP8 cached-prefix kernelはonline softmaxだが、長prefixではK/V tile再利用不足が残っていた。

### 今回の変更点

- `ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2` を追加し、既存FP8 APIと横並びで比較できる形にした。
- HIPRTCに `ullm_cached_prefix_attn_fp8_e4m3_flash2_kernel` を追加した。
- kernelは1 blockを1 `(new token, q head)` に割り当て、64 token tileごとにscoreをshared memoryへ置き、online softmaxでmax、denominator、weighted valueを更新する。
- `runtime-cached-prefix-attn-smoke` に `cached_prefix_flash2` executorを追加した。

### 次の行動

1. `cargo fmt`、runtime-sysのtargeted tests、R9700 smokeで正しさを確認する。
2. 旧FP8 chunked executorと `cached_prefix_flash2` を同一shapeで比較する。
3. 遅い場合はvalue側の読み方、tile size、q head/token block割り当てを調整する。

## 追記: cached-prefix flash2 M=512 grid

### 前回の要点

- `cached_prefix_flash2` v1は `M=16` と `M=128` で旧FP8 `cached_prefix_chunked` より速かった。
- 保存gridとしては `M=512` が未測定だった。

### 今回の変更点

- R9700で `L={4096,16384,65536}, M=512` を `cached_prefix_chunked` と `cached_prefix_flash2` の両方で測定した。
- 旧FP8比のtok/sは以下:
  - `L=4096,M=512`: `1.242x`
  - `L=16384,M=512`: `1.234x`
  - `L=65536,M=512`: `1.361x`
- sampled guardは全て通り、最大diffは `0.000031497` だった。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c5-flash2-tiled-online-softmax-v1.md` に追記した。

### 次の行動

1. F32 cached-prefix flash2を足して、kernel構造改善とFP8 byte削減を分離する。
2. cold prefill causal attentionは既存kernelの通常shapeがすでにonline softmax 1-passなので、単純な移植よりQ/K tileや複数query並列化の設計が必要。
3. RDNA4向けのWMMA/MFMA化は、別の小さなprototypeで命令・layoutを確認してから入れる。

## 追記: cached-prefix flash2 F32 isolation

### 前回の要点

- FP8 flash2 v1は保存gridで旧FP8 `cached_prefix_chunked` より速かった。
- ただし、改善がFP8 byte削減によるものか、FlashAttention2-styleのkernel構造によるものかを分けて見る必要があった。

### 今回の変更点

- `ullm_runtime_cached_prefix_attn_f32_flash2` を追加した。
- Rust FFIに `cached_prefix_attn_f32_flash2` を追加し、CPU/HIPテストを追加した。
- `runtime-cached-prefix-attn-smoke` の `cached_prefix_flash2` executorを `f32` KV cacheでも使えるようにした。
- `tools/run-runtime-cached-prefix-sweep.py` に `--kv-cache-dtype f32|fp8_e4m3` とdtype別の必須HIP kernel envを追加した。
- R9700でF32 isolation sweepを実行し、旧F32比 `1.20x-1.49x` の改善を確認した。

| L | M | F32 flash2 / F32 baseline tok/s |
| ---: | ---: | ---: |
| 4096 | 16 | 1.203x |
| 4096 | 512 | 1.287x |
| 16384 | 16 | 1.322x |
| 16384 | 512 | 1.491x |
| 65536 | 16 | 1.362x |

### 次の行動

1. cold prefill causal attention側へFlashAttention2-style tile方針を展開する。
2. package self-attention prefillのattention componentに接続する。
3. RDNA4向けWMMA/MFMA化は、F32/FP8 cached-prefixの現v1を基準に別prototypeで進める。

## 追記: cold-prefill causal flash2 F32 v1

### 前回の要点

- cached-prefix flash2はFP8/F32の両方で改善した。
- 次はcold prefill causal attention側にtile online-softmax方針を展開する段階だった。
- 既存cold prefill kernelはすでにonline softmax 1-passなので、まず別API/別executorで比較するのが安全だった。

### 今回の変更点

- `ullm_runtime_causal_attn_f32_flash2` と `ullm_runtime_causal_attn_batch_f32_flash2` を追加した。
- Rust FFIに `causal_attn_f32_flash2` と `causal_attn_batch_f32_flash2` を追加し、CPU/HIPテストを追加した。
- `runtime-causal-attn-batch-smoke` に `flash2` executorを追加した。
- R9700で初期safe gridを測り、旧 `causal_attn_batch_f32` 比で `1.19x-1.24x` のinput tok/s改善を確認した。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-causal-attn/phase-c6-causal-flash2-tiled-online-softmax-v1.md` に保存した。

| batch | seq | head_dim | value_dim | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 64 | 64 | 1.188x |
| 4 | 128 | 64 | 64 | 1.232x |
| 1 | 256 | 64 | 64 | 1.236x |
| 1 | 128 | 256 | 256 | 1.188x |

### 次の行動

1. 長context prefill用のbenchmark harnessを追加し、8MiB smoke制限より大きいgridを安全に測る。
2. 複数query row/blockでK/V tileを再利用する本命のFlashAttention2-like構造へ進める。
3. RDNA4向けMFMA/WMMA layoutを小さいprototypeで検証する。

## 追記: RDNA4 FP8 WMMA最小プローブ

### 前回の要点

- cached-prefix/cold-prefillのFlashAttention2-style v1は改善したが、QK/V accumulationはまだscalar loopだった。
- 次はRDNA4向けWMMA/MFMAをattention本体へ入れる前に、HIPRTCからbuiltinを直接呼べるかを確認する必要があった。

### 今回の変更点

- `ullm_runtime_wmma_fp8_probe`、Rust FFI `wmma_fp8_probe`、CLI `runtime-wmma-fp8-probe-smoke` を追加した。
- HIPRTC kernel内で `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` を呼ぶ最小プローブを実装した。
- R9700/RDNA4 device index `2` では非0 markerで成功した。
- V620/RDNA2 device index `1` では0 markerで失敗扱いにし、非RDNA4を誤って成功扱いにしないことも確認した。

### 次の行動

1. Q/K tile layout、lane mapping、accumulator配置を小さいQK microkernelで固める。
2. cached-prefix flash2とcausal flash2のscalar dot部分を置き換える候補を作る。
3. tok/sとsampled diffを同時に見て、速度だけでなく出力品質の崩れも確認する。

## 追記: RDNA4 FP8 WMMA QK probe

### 前回の要点

- RDNA4ではHIPRTCからFP8 WMMA builtinを呼べることを確認した。
- ただし前回はmarker確認で、Q/K tileを入れた演算結果はまだ見ていなかった。

### 今回の変更点

- `ullm_runtime_wmma_fp8_qk_probe`、Rust FFI `wmma_fp8_qk_probe`、CLI `runtime-wmma-fp8-qk-probe-smoke` を追加した。
- Q/Kは16x16 FP8 E4M3 byte tile、outputは16x16 F32 accumulator tileに固定した。
- HIPRTC kernelで `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` を呼び、32 lane x 8 accumulatorを出力するようにした。
- 初期sanityとしてQ/KをFP8 1.0相当の `0x38` で埋め、R9700 device index `2` で `max_abs=16.000000000` を確認した。
- `layout` patternも追加し、非一様Q/K入力ではpreviewがCPU row-major Q*K^Tの先頭 `[0,1,2,...]` ではなく `[136,0,168,0,...]` になることを確認した。
- `layout 256` では最大値が `374` になった。CPU row-major Q*K^Tなら最大 `255` なので、現時点ではoutput順だけでなくA/B input register packingもrow-majorではない。
- つまり現時点ではWMMA arithmeticは動いているが、input packingとoutput accumulator orderはまだattention用row-major QKとして確定していない。
- V620/RDNA2とCPU CLIは失敗扱いにした。

## 追記: RDNA4 FP8 rocWMMA cached-prefix v1

### 前回の要点

- standalone rocWMMA attention probeで、FP8 QK tile、online softmax、V accumulationまでは確認済みだった。
- まだ `runtime-cached-prefix-attn-smoke` の実計測経路には接続されていなかった。

### 今回の変更点

- `ullm_runtime_cached_prefix_attn_fp8_e4m3_rocwmma` を追加した。
- Rust FFIに `cached_prefix_attn_fp8_e4m3_rocwmma` を追加し、CPU fallbackとRDNA4 HIP testを追加した。
- `runtime-cached-prefix-attn-smoke` に `cached_prefix_rocwmma_fp8` executorを追加した。
- このexecutorではQ/K/VをすべてFP8 byte列として扱い、Qのscaleも出力行へ `q_sequence_scale` として出す。
- sampled referenceはdecoded FP8 Q/K/Vを使うようにし、速度だけでなく出力品質の崩れも同時に見られる状態にした。
- `tools/run-runtime-cached-prefix-sweep.py` に `rocwmma_fp8` aliasを追加した。

### 観察

- 現時点の実装はRDNA4専用で、`head_dim=16`、`value_dim=16`、`q_heads/kv_heads` が16の倍数に限定している。
- R9700で `L=65536,M=512,q_heads=16,kv_heads=1,dim=16` が `14.627084ms`、`input tok/s=35003.559151`、`sampled_max_abs_diff=0.000000415` で通った。
- `q_heads=32,kv_heads=2` の複数KV head smokeも通り、strided K/V cache読み出しで `sampled_max_abs_diff=0.000000002` だった。
- 同じ `L=4096,M=16,dim=16` では、`cached_prefix_rocwmma_fp8` が `0.911433ms`、既存 `cached_prefix_flash2 fp8_e4m3` が `3.252488ms`、旧 `cached_prefix_chunked fp8_e4m3` が `3.778853ms` だった。
- ただし、この比較はrocWMMA経路だけQもFP8にしているため、純粋なkernel構造差だけではない。
- V620/RDNA2ではRDNA4必須エラーで拒否されることを確認した。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c11-rdna4-fp8-rocwmma-cached-prefix-v1.md` に保存した。

### 次の行動

1. `head_dim=16` 固定を外す前に、Q row groupingとK/V tile再利用の方針を詰める。
2. 実モデルに近いhead_dimへ広げる。
3. Q FP8化の影響とrocWMMA/FA2-like構造改善を分離して比較する。

## 追記: RDNA4 FP8 rocWMMA cached-prefix 16n dimension対応

### 前回の要点

- `cached_prefix_rocwmma_fp8` はR9700で動いたが、`head_dim=16,value_dim=16` 固定だった。
- 実モデルのhead dimensionに近づけるには、少なくとも16の倍数dimensionを扱える必要があった。

### 今回の変更点

- C API、Rust FFI、CLIの制約を `head_dim=16,value_dim=16` 固定から、`head_dim` と `value_dim` が16の倍数であることへ変更した。
- HIPRTC kernelではQKを `head_dim` 方向に16ずつrocWMMAで累積するようにした。
- `value_dim` は16列ずつ別blockで処理する形にし、既存のvalue tile並列性を残した。
- sweep toolにも `cached_prefix_rocwmma_fp8` 用のshape検証を追加し、無効なdimensionを実行前に拒否するようにした。

### 観察

- R9700で `q_heads=32,kv_heads=2,head_dim=32,value_dim=32` は `0.061021ms`、`sampled_max_abs_diff=0.000000004` で通った。
- R9700で `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256` は `17.222257ms`、`sampled_max_abs_diff=0.000000719` で通った。
- 同じ256次元shapeの既存 `cached_prefix_flash2 fp8_e4m3` は `3.952818ms` だったため、現rocWMMA拡張は性能面ではまだ負けている。
- 原因は、value tileごとにQKとonline softmaxを再計算していること。
- dynamic shared memoryに `16 * value_dim` のaccumulatorを置き、QK再計算を消す案も試したが、小さいprefixでも長いprefixでも悪化したため未採用に戻した。

### 次の行動

1. value tile並列を維持しながらQK/softmax再計算を減らすblock設計を検討する。
2. `head_dim=256,value_dim=256` で既存flash2を超えることを次の性能gateにする。
3. その後、cold-prefill causal attention側へ展開する。

## 追記: RDNA4 FP8 rocWMMA cached-prefix value group 64

### 前回の要点

- 16倍数dimensionには対応できた。
- ただし `value_dim=256` では16列value tileごとにQK/online softmaxを再計算するため、既存flash2より遅かった。

### 今回の変更点

- 1 blockが64列のvalue groupを担当する形に変更した。
- `value_dim=256` ではQK/online softmax再計算が16回から4回に減る。
- full-value dynamic shared accumulator案、32列group、128列groupも比較したが、長prefixでは64列groupが最も良かった。

### 観察

- R9700で `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256` は `15.438269ms`、`sampled_max_abs_diff=0.000000719` だった。
- phase C12の16列tile版は `17.222257ms` だったので改善はした。
- 既存 `cached_prefix_flash2 fp8_e4m3` は `3.952818ms` なので、まだ大きく負けている。
- full-value dynamic shared accumulatorはQK再計算を1回まで減らせたが、block並列性が落ちて `30ms` 台まで悪化した。
- つまり現段階では、QK/softmax再計算削減とV accumulationの並列性維持の両立が必要。

### 次の行動

1. QK/softmaxを1回だけ計算しつつ、V accumulationを十分並列化する2-stageまたはcooperative-group構造を検討する。
2. その前にcold-prefill causal attentionへrocWMMA QKを限定導入し、効果が出やすいshapeを探す案もある。

### 次の行動

1. 非一様Q/K inputでWMMA accumulatorのlane/register layoutを特定する。
2. CPU row-major Q*K^Tと比較できるようにoutput reorderを追加する。
3. cached-prefix flash2のQK dot部分を小さい条件で置き換え、tok/sとsampled diffを同時に見る。

## 追記: RDNA4 FP8 rocWMMA QK probe

### 前回の要点

- direct builtinのFP8 WMMA QK probeでは、`ones` は `16.0` で演算自体は動いていた。
- しかし `layout` patternはCPU row-major Q*K^Tと一致せず、input packingとaccumulator orderを自前で解く必要があった。

### 今回の変更点

- `ullm_runtime_rocwmma_fp8_qk_probe`、Rust FFI `rocwmma_fp8_qk_probe`、CLI `runtime-rocwmma-fp8-qk-probe-smoke` を追加した。
- HIPRTC kernelで `rocwmma::fragment`、`load_matrix_sync`、`mma_sync`、`store_matrix_sync` を使う16x16 FP8 QK probeを実装した。
- HIPRTC compile helperに追加include optionを渡せる経路を入れ、rocWMMA headerを読めるようにした。
- R9700 device index `2` で `ones` は `max_abs=16.0`、`layout` はpreview `0..63`、max `255` になった。
- V620 device index `1` はRDNA4必須として拒否されることを確認した。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-wmma/phase-c9-rdna4-fp8-rocwmma-qk-probe-v1.md` に保存した。

### 次の行動

1. raw builtin layout解析は主経路から外し、RDNA4 FlashAttention2-like実装の第一候補をrocWMMA fragment APIにする。
2. 16x16 FP8 QK tileを既存cached-prefix/cold-prefill flash2のQK dot部分へ小さい条件で組み込む。
3. tok/sだけでなくsampled diffを保存し、online softmaxとV accumulationの統合へ進める。

## 追記: RDNA4 FP8 rocWMMA attention probe

### 前回の要点

- rocWMMA QK probeで、FP8 Q*K^Tをrow-major出力として扱えることを確認した。
- 次はQK単体ではなく、online softmaxとV accumulationまでつないで壊れないかを見る必要があった。

### 今回の変更点

- `ullm_runtime_rocwmma_fp8_attn_probe`、Rust FFI `rocwmma_fp8_attn_probe`、CLI `runtime-rocwmma-fp8-attn-probe-smoke` を追加した。
- 固定shapeはQ `16x16` FP8、K `32x16` FP8、V `32x16` F32、output `16x16` F32。
- HIPRTC kernelでrocWMMA QKを2 tile実行し、per-row online softmaxでVを畳み込むstandalone probeにした。
- R9700 device index `2` で `ones` はCPU参照diff `0`、`layout` はCPU参照diff `0.000000119` だった。
- V620 device index `1` はRDNA4必須として拒否されることを確認した。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-wmma/phase-c10-rdna4-fp8-rocwmma-attn-probe-v1.md` に保存した。

### 次の行動

1. standalone probeをcached-prefix flash2のQK/softmax/V accumulationへ移植する。
2. その後cold-prefill causal flash2へcausal mask込みで移植する。
3. sampled diffを固定してからtok/sを測り、tile sizeとblock割り当てを調整する。

## 追記: cached-prefix flash2 FP8 Q baseline

### 前回の要点

- `cached_prefix_rocwmma_fp8` はFP8 Q/K/V入力で動くが、`head_dim=value_dim=256` では既存scalar `cached_prefix_flash2` より大幅に遅かった。
- 既存 `cached_prefix_flash2` はQだけF32だったので、rocWMMA版の遅さがFP8 Q復号由来かどうかを分離できていなかった。

### 今回の変更点

- `ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2_fp8q` を追加した。
- Rust FFIとCPU/HIPテストを追加した。
- `runtime-cached-prefix-attn-smoke` に `cached_prefix_flash2_fp8q` executorを追加した。
- `tools/run-runtime-cached-prefix-sweep.py` に `cached_prefix_flash2_fp8q` を追加した。
- 既存のHIPRTC flash2 FP8 kernelは、f32 QとFP8 Qを `q_is_fp8` で切り替えられるようにした。

### 観察

- R9700 device index `2`、`L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256` で測った。
- `cached_prefix_flash2`: `3.856106 ms`, `4149.263890 input tok/s`, Q bytes `262144`。
- `cached_prefix_flash2_fp8q`: `4.123693 ms`, `3880.016943 input tok/s`, Q bytes `65536`。
- `cached_prefix_rocwmma_fp8`: `20.403873 ms`, `784.164837 input tok/s`, Q bytes `65536`。
- FP8 Q化だけならscalar flash2では約 `1.07x` の遅化で済む。
- rocWMMA版の遅さはFP8 Q復号単体では説明できず、value groupごとのQK/softmax再計算やK/V tile再利用不足が主因だと考える。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c14-fp8q-flash2-baseline-v1.md` に保存した。

### 次の行動

1. `cached_prefix_flash2_fp8q` をFP8 Q入力の短期baselineとして使う。
2. rocWMMA版は単純なvalue group調整ではなく、複数query row/blockでK/V tileを再利用するFlashAttention2-like構造へ寄せる。
3. SQ候補評価では、FP8 Q復号とattention kernel構造を分けて評価する。

## 追記: RDNA4 FP8 rocWMMA value group heuristic

### 前回の要点

- FP8 Q復号そのものは、scalar `cached_prefix_flash2_fp8q` では小さいoverheadだった。
- rocWMMA版の遅さは、value groupごとのQK/softmax再計算とblock並列性のバランスが主因だった。

### 今回の変更点

- `cached_prefix_rocwmma_fp8` のvalue group幅をruntime引数化した。
- `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH={16,32,64,128,256}` でoverrideできる。
- env未指定では、`new_tokens < 64` なら16、そうでなければ64を使うheuristicにした。
- kernelを再compileせずにshapeごとのvalue group幅を変えられる。

### 観察

- R9700で `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256` のvalue group sweepを取った。
- width 16: `16.797953ms`, `952.497010 input tok/s`。
- width 64: `21.011732ms`, `761.479361 input tok/s`。
- width 256: `41.351419ms`, `386.927472 input tok/s`。
- 短いchunkではQK/softmax再計算削減よりblock並列性が重要だった。
- 一方、`M=128` ではrocWMMA heuristicが `15.409175ms`、scalar `cached_prefix_flash2` が `26.629428ms`、`cached_prefix_flash2_fp8q` が `28.595555ms` だった。
- `M=512` ではrocWMMA heuristicが `70.911643ms`、scalar `cached_prefix_flash2` が `103.509898ms`、`cached_prefix_flash2_fp8q` が `113.356638ms` だった。
- つまり、rocWMMAはdecode-likeな短いchunkではまだ不利だが、数百token prefill chunkではscalar flash2を上回り始めた。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c15-rocwmma-value-group-heuristic-v1.md` に保存した。

### 次の行動

1. `M=16` 付近は `cached_prefix_flash2_fp8q` を短期baselineとして維持する。
2. `M=128/512` 以上は `cached_prefix_rocwmma_fp8` をSQ候補評価用の本命attention baselineに近づける。
3. 次は複数query token tileへ広げ、FlashAttention2-likeなK/V tile再利用をさらに増やす。

## 追記: rocWMMA value group sweep軸を追加

### 前回の要点

- rocWMMA value group幅は、短いchunkと大きいprefill chunkで最適値が違った。
- 手書きshell loopだと結果schemaにvalue group幅やenv設定が残りにくかった。

### 今回の変更点

- `tools/run-runtime-cached-prefix-sweep.py` に `--rocwmma-value-group-widths auto|16|32|64|128|256` を追加した。
- `auto` はenv未指定、数値指定は `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH` をcaseごとに設定する。
- JSONLの `required_env` と `workload.rocwmma_value_group_width` に設定を残すようにした。
- summary markdownにも `rocWMMA value group` 列を追加した。

### 観察

- dry-runで `L=4096,M={16,128},width={auto,16,64}` の6ケースが展開できることを確認した。
- 実測でも6ケースすべてokだった。
- `M=16` はauto `17.804503ms`、width 16 `18.245964ms`、width 64 `20.665049ms`。
- `M=128` はauto `15.295860ms`、width 16 `48.100149ms`、width 64 `15.443554ms`。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c16-rocwmma-value-group-sweep-axis-v1.md` に保存した。

### 次の行動

1. 長prefix gridで `cached_prefix_rocwmma_fp8` を測るときは、`auto,16,64` を比較軸に入れる。
2. 次のmulti-query-token tile化では、このscript結果をbaselineとして使う。

## 追記: rocWMMA long-prefix grid

### 前回の要点

- rocWMMA value group幅をsweep scriptの正式な軸にした。
- 次は長prefix gridで、scalar flash2系と同じshapeで比較する必要があった。

### 今回の変更点

- R9700で `L={4096,16384,65536}`、`M={16,128,512}`、`q_heads=16,kv_heads=1,head_dim=256,value_dim=256` を測った。
- rocWMMAは `value_group={auto,16,64}` を測った。
- scalar baselineとして `cached_prefix_flash2` と `cached_prefix_flash2_fp8q` も同じshapeで測った。

### 観察

- `M=16` ではscalar flash2/fp8qがまだ圧倒的に速い。
  - `L=4096`: rocWMMA autoはscalar flash2比 `0.218x`。
  - `L=16384`: `0.235x`。
  - `L=65536`: `0.192x`。
- `M=128` では全prefix長でrocWMMAが勝った。
  - `L=4096`: scalar flash2比 `1.745x`。
  - `L=16384`: `1.848x`。
  - `L=65536`: `1.468x`。
- `M=512` では `L=4096` と `L=65536` でrocWMMAが明確に速く、`L=16384` でもF32-Q scalar flash2とほぼ同等、FP8-Q scalar flash2より速い。
- runtime heuristicは、value group `auto,16,64` の中で最速または最速近傍だった。
- 結果は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c17-rocwmma-long-prefix-grid-v1.md` に保存した。

### 次の行動

1. SQ cached-prefix評価では、短chunk/decode-likeは `cached_prefix_flash2_fp8q`、`M>=128` は `cached_prefix_rocwmma_fp8` をbaselineとして扱う。
2. 次のkernel変更は、`M=16` の弱さを埋めるmulti-query-token tile化を検討する。
