# `SQ9_0` を V100 / RDNA1 向け保留 option へ訂正

## 前回の要点

- `e86c2e3c` は、V620 M=1 の `SQ9_0` 実測が `SQ8_0` 比 +6.069% に留まり、全 package + KV の
  採算条件 +7.29% を満たさないこと、容量・ISA・品質の比較が INT8 block-scale を支持することを
  保存したまま、`SQ9_0` を将来の compatibility implementation obligation と定義した。
- この host で既に確認済みの current target `gfx1030` / `gfx1100` / `gfx1201` / `gfx942` /
  `gfx950` には、INT8 dot またはそれ以上の INT8 matrix route がある。

## 今回の変更点

- `SQ9_0` を current runtime/artifact format から外し、**保留中の将来 option** に訂正した。
  packer、deterministic RNE quantizer、reader、validator、CPU oracle、generic dequant kernel、
  runtime selector、manifest handling はすべて保留であり、現在の実装対象ではない。
- uLLM の当面の format scope は `AQ4_0` / `SQ8_0` / `SQ8_1`、architecture scope は
  `gfx1030` / `gfx1100` / `gfx1201` / `gfx942` / `gfx950` と明記した。`SQ9_0` はこの五 target
  の format selector と manifest から選択不可である。
- ユーザー指定の future candidate は NVIDIA V100 と RDNA1 とした。ただし「INT8 dot がない」と
  一括しては記録しない。
  - V100 は FP16 Tensor Core を持ち、INT8 Tensor Core は持たないことを NVIDIA 資料で確認した。
    一方で `dp4a` / `IDP4A` の INT8 dot instruction は存在する。uLLM の shape でそれが実用的か、
    V100 に useful FP8 route がないか、`SQ9_0` が有利かは **未確認** である。
  - RDNA1 は exact GFX target を未指定のため世代単位では結論できない。GPU を使わない local
    ROCm 7.2.1 `llvm-mc` では `gfx1010` が `v_dot4c_i32_i8` / `v_dot4_i32_i8` を拒否し、
    `gfx1011` / `gfx1012` は両方を受理した。選択した FP8 WMMA mnemonic は三 target 全てで
    拒否されたが、これは scalar FP8、generated ISA、hardware behavior、性能の否定証明ではない。
- 上記は各 `gfx1010` / `gfx1011` / `gfx1012` に対して、次を CPU 上で実行した静的 probe である。
  NVIDIA target はこの local `llvm-mc` で検証できない。

  ```bash
  printf '%s\n' 'v_dot4c_i32_i8 v0, v1, v2' \
    | /opt/rocm/llvm/bin/llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=<arch> -filetype=obj -o /dev/null
  printf '%s\n' 'v_dot4_i32_i8 v0, v1, v2, v3' \
    | /opt/rocm/llvm/bin/llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=<arch> -filetype=obj -o /dev/null
  printf '%s\n' 'v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[10:11], v[0:7]' \
    | /opt/rocm/llvm/bin/llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=<arch> -filetype=obj -o /dev/null
  ```
- E5M3 の `q << 7` shift-only conversion が検討に値するのは、**specific target** で useful FP8
  route と practical INT8 matrix/dot route の双方が欠ける場合だけ、という方針にした。これは
  性能主張ではない。
- 過去の V620 M=1 +6.069%、+7.29% 条件、容量、static ISA、品質の数値は一切変更していない。
  今回は GPU、HIP runtime API、service、candidate、release、campaign、authorization、active
  manifest を使用・変更していない。

## 保留解除の着手条件

1. 実際の product/serving requirement が V100 または exact RDNA1 GPU/GFX を指定する。
2. その target に useful FP8 route と practical INT8 matrix/dot route がないことを、target 固有の
   toolchain/ISA と hardware evidence で確認する。V100 は NVIDIA 側で別検証が必要である。
3. `AQ4_0` / `SQ8_0` / `SQ8_1` との matched comparison が E5M3 route の必要性を示す。
4. CPU oracle、malformed input/tail、target differential、quality、benchmark gate を固定した
   新しい implementation plan が review 済みである。
5. ユーザーが実装と必要な GPU validation window を別途承認する。activation、campaign、authorization
   消費、active manifest はさらに別の明示承認を要する。

## 次の行動

1. `SQ9_0` の実装・GPU 実験・artifact/manifest/campaign 作成は行わない。
2. 現行 target の `AQ4_0` / `SQ8_0` / `SQ8_1` の各 gate を、それぞれの所有計画に従って進める。

## Evidence

- [NVIDIA V100 Tensor Cores](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [NVIDIA PTX `dp4a` ISA](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html)
- [NVIDIA Volta SASS `IDP4A` table](https://docs.nvidia.com/cuda/archive/11.4.4/cuda-binary-utilities/index.html)
- [NVIDIA TensorRT hardware support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/)
- [LLVM AMDGPU backend guide](https://llvm.org/docs/AMDGPUUsage.html)
- [SQ9_0 design input](../../../../docs/plans/sq9-format-design-input-v0.1.md)
