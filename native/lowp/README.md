# lowp: 低精度HIP行列積ライブラリ

sLLMの低精度行列積・量子化・codecを独立したCMake libraryとして提供する。
`include/lowp/lowp.h` はsLLMやHIPのヘッダを必要としないC APIである。
現在の検証対象はexact `gfx1030` と `gfx1201`。gfx942の共有ソースは保持するが、
この切り出しで実機対応範囲を拡張しない。

## 対応形式

| 公開format | 重み | 活性化 | scale |
| --- | --- | --- | --- |
| MXFP8 E4M3 W8A8 | E4M3FN、block 32 | E4M3FN、block 32 | E8M0 |
| MXFP6 E3M2 W6A6 | packed E3M2、block 32 | packed E3M2、block 32 | E8M0 |
| MXFP8 W8A16 / MXFP6 W6A16 | 上記MX重み | BF16、M=1の対応shape | 重みE8M0 |
| NVFP4 W4A16 | E2M1、block 16 | BF16 | E4M3FN block scale＋FP32 tensor scale |
| NVFP4 W4A4 | E2M1、block 16 | E2M1、block 16 | 各operandのE4M3FN block scale＋FP32 tensor scale |
| FP8 outer W8A8 | E4M3FN | E4M3FN | FP32 row scale。gfx1030 softwareのみ |
| MXFP4 W4A8 v1 | E2M1、block 32 | E4M3FN、K方向block 32 | E8M0。**契約定義のみ、planは未対応として拒否** |

公開format番号4は予約値で、公開APIでは受理しない。
既存のMXFP4 W4A4はsLLMのMoE等が使う内部C++ APIにだけ残し、W4A8とは別に扱う。
BF16行列積、FP8 native、hipBLAS/hipBLASLt、KV/attention本体はこのライブラリの外にある。

## 演算と配置

`A[M,K] × W[N,K]^T → O[M,N]`。論理配置はrow-major、累積はFP32、出力はBF16 RNE。
value planeとscale planeは別pointerで渡す。GGUFやModelOpt等からの変換は呼び出し側で行う。

- MXFP8: 1 byte/value、scaleは各行のK方向32要素ごとに1 byte。
- MXFP6: 4 valueを3 byteに詰める。Kは32の倍数、scaleは同じblock 32配置。
- NVFP4 weight: 論理`N*K`要素を連続したnibble列へ詰める。下位nibbleが先。
  weight block scaleは`[N, ceil(K/16)]`。tensor scaleはdevice上のFP32 scalar。
- NVFP4 activation: 各行を`ceil(K/2)` byteに詰め、scaleは`[M, ceil(K/16)]`。
  既存baselineが受理する端数Kは保持する。MXFP8/MXFP6のK非整列は拒否する。
- FP8 outer: valueは`[M,K]`/`[N,K]` byte、scaleは`[M]`/`[N]` FP32。
- 正確なbyte数・activation scale offset・追加scratchはplanのfootprintを使う。
  値とscaleの間のpaddingを独自に推測しない。

`lowp_get_format_info` は形式の契約を返し、`lowp_supported_formats` はtargetの実装済み形式を返す。
実際のshapeが使えるかは `lowp_matmul_plan` の戻り値で判断する。
`provider`、`variant`、`tile`、`inner_product`は既存の監査IDを保持する。
`selector_supported`、`selector_enabled`、`adopted`は旧selectorの監査値であり、
planの実行可能性を表す`supported`とは区別する。たとえば端数KのNVFP4 baselineは
実行可能でも、最適化側のselector監査値がfalseになる場合がある。

## APIと所有権

1. requestの`struct_size`と`version=LOWP_ABI_VERSION`、形式・target・配置・M/N/Kを設定する。
2. `lowp_matmul_plan(&request, &plan)`で選択とfootprintを確定する。成功したplanを変更しない。
3. callerがdevice buffer・workspace・HIP streamを所有し、`lowp_matmul_launch`へ渡す。
   APIへ渡すstreamは `hipStream_t` を`void*`へ変換した値で、nullはHIP default stream。
   current deviceとstreamはplanのexact targetに一致させる。

通常のlaunchはBF16 activationをworkspaceへ量子化して行列積を実行する。
`lowp_quantize_activation`で先に量子化し、`LOWP_ACTIVATION_PREQUANTIZED`とvalue/scale pointerを
渡すこともできる。projection packはこの経路でactivation量子化を共有する。
BF16 activationを直接読むA16形式は量子化せず、workspaceも不要。

libraryはdevice allocation、同期、CPU fallbackを行わない。stream完了まで全bufferを有効に保つ。
FP16 stagingを使う既存の実験variantだけは、caller-owned staging bufferとGEMM callbackを要求する。
callbackのhandle・同期・resource管理もcallerの責任であり、libraryはBLASに依存しない。
戻り値はHIP互換のerror number。失敗時に別形式や別providerへ黙って切り替えない。

## C++連携ヘッダ

`include/lowp/detail/`は同じrepository内の利用者向けのC++連携ヘッダで、API・ABIの安定性は約束しない。
codec、形式契約、variant選択、kernel起動関数、内部MXFP4 W4A4の入口をここに置く。
利用者は`<lowp/detail/...>`としてincludeし、`src/`の実装ファイルへ直接依存しない。
`src/`の`.inc`はkernel本体であり、過去の研究用probeを除いて外から読まない。

`KernelVariant`は低精度providerの監査IDだけを持つ。値1（`Unspecialized`）は低精度の特化経路を
選ばなかったことを表す。値2、3、4、5、7、12、13、16、17、91は統合側（sLLMのBF16、hipBLAS、
FP8 native経路）が同じ監査ID空間で使うため、lowpでは再利用しない。
`lowp_logical_kernel_id`等はlowpの知らないIDに`nullptr`を返し、統合側が自分のIDを解決する。

## 単体ビルドと検査

ROCm 7.14 / LLVM 23を使う例。`gfx1201`でも同じ手順でtargetを置き換える。

```sh
cmake -S native/lowp -B /tmp/lowp-gfx1030 \
  -DLOWP_TARGET=gfx1030 -DROCM_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_BUILD_TYPE=Release \
  -DSLLM_ENABLE_LOW_PRECISION_CODEC_GPU_TEST=ON \
  -DSLLM_ENABLE_LOWP_MATMUL_GPU_TESTS=ON \
  -DLOWP_BUILD_HOST_SELECTION_TEST=ON
cmake --build /tmp/lowp-gfx1030 -j4
ctest --test-dir /tmp/lowp-gfx1030 -L host --output-on-failure
ROCR_VISIBLE_DEVICES=GPU-76a08c022586fed6 \
  ctest --test-dir /tmp/lowp-gfx1030 -L gpu --output-on-failure
```

HIP kernelはcode object v6、wave32、`-ffp-contract=off`で構築する。
sLLMからは `add_subdirectory` と `sllm_lowp` / `sllm_lowp_plan` targetを使う。
CPU-only親ビルドではplanのみを構築し、host testをGPU証拠として扱わない。
単体exampleは`tests/lowp_matmul_example.hip.cpp`、独立FP32 oracleは
`tests/lowp_matmul_oracle_gpu_test.hip.cpp`、microbenchは`tests/lowp_matmul_microbench.hip.cpp`。

## 来歴

MITライセンス。codec/providerと多くのkernelはsLLMの実装を移したもの。
llama.cpp由来部分の原典・許諾はソース冒頭と
[THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES.md)を参照する。
BF16 MMVF由来部分は従来の`native/hip`に残り、低精度kernelの来歴は新しいpathへ対応付けている。
別repoへの切り出し、配布・版管理、形式変換器は今回の対象外。
