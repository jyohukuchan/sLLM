# (注:原稿、開発状況を反映したものではない)
# sLLM
- sLLM(serve LLM) is LLM inference engine for serving.
- every information is [here](https://example.com) (official site).

## 特徴
- サポートするモデル量子化フォーマット
  - weight/activation
  - BF16/BF16
  - FP8/FP8
  - MXFP8/MXFP8
  - MXFP6/MXFP6
  - MXFP4/MXFP6
  - NVFP4/NVFP4
  - EXL3(3bpw・4bpw・5bpw)/TBD
- サポートするKV Cache量子化フォーマット
  - Key/Value
  - FP16/FP16
  - FP8/FP8
  - TQ(TurboQuant) (K3・K4)/(V3・V4)
  - NVFP4/NVFP4
  - (MXFP8・MXFP6)/(MXFP8・MXFP6)

- サポートしているGPU
  - gfx1030
  - gfx1201
- PR・issueを許可するGPU(≈対応したい気持ちはあるGPU)
  - Tier1
    - RDNA2～RDNA4
    - CDNA1～CDNA5
    - gfx906
  - Tier2
    - Volta
    - Turing
    - Huawei Ascend
    - Sapphire Rapids～Clearwater Forest
    - zen4～zen6
    - Hygon GPU
    - Neoverse v3
- 許可しないGPU
  - 開発者が既に十分いる・メーカーのやる気がない・入手性が著しく悪いもの
  - Hopper～Rubin
  - intel GPU
  - Qualcomm
  - Google TPU


## Quick start

## あとがき
- coding agent等agent的な用途を重視、過度な量子化や遅いハードウェアに対応しない
- EXL3がVRAMあたり精度では優れているが、実装の単純さ・高バッチ時の合計tgでMXFP/NVFP4/FP8等ネイティブサポートがある形式が有利
