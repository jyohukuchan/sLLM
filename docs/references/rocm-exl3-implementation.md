# rocm_exl3 implementation reader

## 範囲とidentity

これは `CarouselAether/rocm_exl3` の実装方法を sLLM の設計判断へ渡すための reader 記録である。
調査対象は `reference/rocm_exl3` の HEAD `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`（2026-08-28）
と、そのclone作業ツリーで確認した未コミット差分である。今回、sLLMへ外部sourceをcopy、adapt、portしていない。

HEADのROCm実装は upstream `.cu/.cuh/.cpp`を変更せず、`setup.py:108-181,335-540` の source選択と
`rocm/hip_compat.hip.h` の force include、`rocm/cuda_shim` の先行includeで `hipcc` に直接渡す。
Torchのhipifyは `cudaKernelNodeParams` 等を変換しきれず、hipccでpristine sourceをビルドする必要がある
（`exllamav3/exllamav3_ext/rocm/README.md:29-56`）。ROCm最低版は7.2.4（`setup.py:229-271`）、
依存は `requirements_rocm.txt:41-60` のROCm Torchと `triton-rocm>=3.7.0` である。

upstream v1.4.4の実commit `17bc3923259ffd48aab742edd261a0ca45d55459` とfork HEADを実際に比較し、
ROCmディレクトリ外の `.cu/.cuh/.cpp/.h` 差分が空であることを確認した。cloneにはtagが付いていないため、
READMEの `git diff v1.4.4` はそのままでは失敗する。tagの指す上記commitを指定すれば再現できる。
またREADMEの「ROCm用ディレクトリ外は4ファイルだけ」という説明に対し、実差分は `.gitignore` と
`requirements_rocm.txt` を含む6ファイルだった。これは移植手法の否定ではなく、比較手順・説明の差異である。

## EXL3形式と量子化

Linearの論理重みは `(in_features, out_features)`。各16x16 tile（256要素）を `kbits`-bitのprocedural
codebook/trellisで表す。重みのlogical shapeを `(Kdim,Ndim)` とすると、未packed indexは
`(Kdim/16,Ndim/16,256)` int16、出力 `trellis` は `(Kdim/16,Ndim/16,256*kbits/16)` int16である
（`modules/quant/exl3_lib/quantize.py:488-514,927-946`、`exllamav3_ext/quant/pack.cu:9-93`）。
`kbits`は1..8、Ndimは128の倍数、Kdimは16の倍数を要求する。

量子化は、校正入力からHessianを集め、正則化block-LDLを計算し（`quantize.py:832-924`）、
出力channel RMS scale、ランダム符号、128幅の左右block Hadamard、tileごとのglobal scale searchを
適用する（`quantize.py:1125-1231`）。その後LDLQ補償付きViterbi tile量子化を行い、`suh/svh`をFP16で、
`mcg`または`mul1`をcodebook markerとして保存する（`quantize.py:1234-1432`）。loaderは
`.suh/.svh/.trellis`を読み、`K=trellis.shape[-1]/16`で `BC_LinearEXL3` を構築する
（`modules/linear.py:385-427`、`modules/quant/exl3.py:16-90`）。

rowsが144以下はEXL3 GEMM/GEMV、長いprefillはreconstruct+FP16 hgemmへ進む。
`rows>=1024`かつ両次元128整列なら、reconstruct時に両Hadamardとscaleを融合して元のbasisを出力する
（`modules/quant/exl3.py:114-218`）。したがって、短いdecodeだけの検証ではquant kernelを見られるが、
長いprefillの挙動は別経路として測る必要がある。

## HIP GEMM/GEMVとGPU境界

ROCm buildは upstream `quant/comp_units/`、`parallel/`、EXL3 GEMM/GEMV/reconstruct/quantize/MoE等を除外し、
`rocm/quant/*_rdna.*` と `comp_units_rdna/` を同じsymbol ABIで置き換える（`setup.py:116-159`）。
`parallel/`除外によりTensor Parallelは使用できない。

HEADのWMMA実装は `rdna_wmma.hip.h:1-30,200-270` でgfx1150/1151 wave32を前提とする。
GEMM本体はPTX `mma.sync.m16n8k16`、ldmatrix、cp.asyncをそれぞれRDNA WMMA、vector load、同期uint4 copyへ
再導出し、fragment layoutも作り直す（`quant/exl3_gemm_inner_rdna.hip.h:1-51,95-151,340-439`）。
測定済みfragmentは operand order `(B,A,C)`、Aはlaneの行、Bはlaneの列、Cはlaneごと8要素である。

RDNA shape tableは256 threads、K tile=16、N=128/256/384/512、LDS budget 64KB向けである
（`quant/exl3_kernel_map_rdna.hip.h:14-31,194-212`）。selectorはCUDAのccを使わず、K/Nの整列、runtime LDS、
occupancyを検査する（`quant/exl3_kernel_map_rdna.hip:35-162,179-255`）。不整合shapeを強制すると、
kernelが末尾列を捨てるため `:316-357` でfail-closedにする。

HEADの `setup.py:275-292` は gfx1100/1101/1102、gfx1150/1151、gfx1200/1201を列挙するが、
HEADのWMMA実証はgfx1151だけである。gfx1030は列挙されず、RDNA2はWMMA absenceによりprefill対応外と
`rocm/RDNA_NOTES.md:452-457` に明記される。gfx1201列挙はcompile target広告であり、HEAD単独の実機証拠ではない。

clone作業ツリーの未コミット差分では、gfx12の128-bit operandとC-fragment mappingを
`rdna_wmma.hip.h`へ追加し、`exl3_gemm_inner_rdna.hip.h`が `c_row/c_col` helperを通すよう変更している。
これはHEADとは別の実機検証済みlocal patchであり、gfx1201では `*_gfx12` builtin、normal `(A,B,C)` order、
column-major native Cを使う（差分 `rdna_wmma.hip.h`、`exl3_gemm_inner_rdna.hip.h`）。

さらにfull model loadで、`exl3_gemm_rdna.hip` 内4か所のautotune呼出しが固定 `SMEM_MAX=92160` を
渡すことが判明した。既存の `exl3_rdna_smem_budget(device)` に変更し、shape admissionと同じ64KiB上限を使う。
この5ファイルの局所patchによる実機結果と、bundled ROCr終了時segfaultへのhost ROCr preload回避は
[調査履歴](../history/2026/09/11-20/rocm-exl3-investigation.md)を参照する。

## Decode、attention、GDN

decodeのm=1は非cooperative GEMVを選び、入力Hadamard、dot、出力Hadamardを分離する
（`rocm/quant/exl3_gemv_rdna.hip:83-121,176-318`）。direct dot coreはdequant後のnative fragment layoutを
LDSへ往復せず、`fdot2`と最後のquad shuffleだけで出力を戻す（`exl3_gemv_kernel_rdna.hip.h:231-319`）。
K分割はnon-cooperativeで、4/8/16 warpsをshape依存で選ぶ（同`:391-474`、`exl3_gemv_rdna.hip:137-170`）。
MoE decodeはexpert axisをgrid.yへ置く4 kernelのmgemv pipelineであり、device parameter blockを使って
graph patch対象をprologueへ集約する（`rocm/quant/exl3_mgemv_rdna.hip:1-65,188-220,529-737`）。

AttentionはHIPではなくTriton Pythonである。paged update、GQA split-DV、long-query grouped pathは
`modules/attention_fn/triton_paged.py:36-176,342-643`、decode split/combineは`:1063-1211`、prefillの
online softmaxとlong-context split/combineは`:1247-1617,1650-1913` にある。ROCmは `get_device_capability()`
がgfx majorを返すため、Blackwell heuristicへ依存せず、head_dim<=128ではKV tile BN=32を明示選択する
（同`:1788-1809`）。

Qwen3.5-2Bのsplit GDNは qkv=6144、z/o=2048、b/a=16、conv=`[6144,1,4]`、A_log=F32[16]、
dt_bias=BF16[16]。qkv/z/oはEXL3、b/aはFP16のままにする必要がある（`modules/gated_delta_net.py:397-460,600-644`）。
split fast pathはEXL3 qkv/z/o、FP16 b/a、BF16 conv/dt_biasを検査し、b/aを `[32,2048]` のFP16へ結合して
`gdn_ba_gemv`を使う（同`:614-644,804-817`、`exllamav3_ext/gdn.cu:1466-1679`）。
recurrent stateはconv BF16 `[batch,6144,4+history]`、state F32 `[batch,history+1,16,128,128]`。
recurrent kernelは128x128専用分岐を持つ（`gdn.cu:829-968`）。

ROCm graph replayは既定無効で、`EXL3_ROCM_HIP_GRAPHS=1`だけがopt-inである。cross-job BC decode corruptionと
capture hangが測定され、通常はeager passthroughとなる（`rocm/graph_rdna.hip:12-59,76-113,130-138,211-281`）。
これはsLLMのHIP Graph採用判断へ直接一般化せず、runtimeごとに確認する必要がある。

## K=2 quantizer finding

実機で `quantize_rdna.hip:90` の `cudaFuncSetAttribute` がK=2に `invalid argument` を返した原因は、
K=2のcost bufferだけで `2*(65536>>2)*sizeof(half)=65536` bytesを使い、入力/reduction scratch 704 bytesを
足すと66240 bytesとなり、測定済みgfx1201 `shared_memory_per_block=65536`を超えることである。
clone作業ツリーの局所修正は `quantize_rdna.hip:84-92` と `quantize_tiles_kernel_rdna.hip.h:69-75` の両方で
共有costを `K>=3` に限定し、K=1/2は既存のglobal `temp_costs_ptr`へ戻す。launcherとkernelの閾値を必ず一致させる
必要があり、片側だけの変更はLDS overrunになる。この修正はsLLM側への移植時にも最小の再現条件として保持する。

## 検証入口とsLLMへの参考案（未採用）

Qwen3.5 smokeは `tests/models/smoke_qwen3_5_arch_.py:27-90`。`--model_dir --device`でconfig/graph、linear-attention
block、full-attention blockを確認できる。GDN oracleは `tests/test_gated_delta_rule.py:17-66,144-213` で、
history、非aligned sequence、FP32 state/BF16 activationを比較する。ただしdeviceは `cuda:1` 固定で、Qwen3.5-2Bの
Nv=16へ変更した `(1,15,16,16,128,128)` と `(2,17,16,16,128,128)` が適切である。

量子tile roundtripは `tests/test_quant_fn.py:64-128`（K=1..8、3 codebook、tail-biting、ideal decode/re-quant）、
layer quant/dequantは同`:131-183`だがLlama path `/mnt/str/...` と `cuda:2` 固定。GEMV oracleは
`rocm_tools/gemv_check.hip:378-420`、GEMM oracleは `rocm_tools/gemm_check.hip:27-191,272-280`。
既存テストはQwen split GDN、gfx1201、K=2 LDS limitを一括では証明しないため、smoke→WMMA/GEMV/GEMM→GDN→
Qwen 4bpw model generationの順にscopeを分ける。

今後EXL3を採用する場合の参考案として、既存のOCP MXFP/NVFP4 dtypeへ読み替えず、codebook、Hadamard/sign、trellis bit rate、padding、
packed layout、weight/activation execution contractを独立した quantization encoding descriptor として扱う設計が考えられる。
GGUFへ統一する場合は、EXL3 safetensorsのtensor metadataとper-tensor K/codebookをversioned converterで正規化し、
gfx1201 WMMA providerとgfx1030の別実装（packed-dequantまたは未対応判定）を分離する案が考えられる。これらは今回の採用決定・完了条件ではない。外部sourceを実装へ再利用する
場合は、対象commit、destination、reuse mode、ライセンスを `THIRD_PARTY_NOTICES.md` の import logへ記録する。
