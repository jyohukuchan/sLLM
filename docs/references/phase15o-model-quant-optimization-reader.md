# Phase 15O model量子化path最適化 reader記録

## 固定identityと利用境界

| source | fixed revision | Phase 15Oでの扱い |
| --- | --- | --- |
| llama.cpp | `f5919bf458ef190468b5c329bb293f8a54a1e69c` | MIT。MMV/MMQ、Q8 activation、NVFP4 packed dotの構造候補。直接reuse時だけ個別provenanceとnoticeを追加する。 |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` | reader-only。技術的事実と選択軸だけを抽出し、source expressionをcopy、adapt、portしない。 |
| SGLang | `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` | reader-only。vLLMと同じ境界で独立cross-checkに使う。 |

完全SHA、取得状態、licenseの正本は[参照source固定manifest](source-lock.md)、reuse手順は
[provenance方針](../provenance/README.md)とする。以下は固定checkoutから抽出した設計上の事実であり、
実装へ渡すsource code表現を含まない。

## reader-only engineから固定する事実

- FP8 weightを保存していても、BF16 activationをW8A8 GEMMへ渡す直前のdynamic per-token/per-row量子化は
  独立した費用になる。vLLMでは`scaled_fp8_quant`とinput quantizer、SGLangではper-token/group quantizerを
  linear providerから選択している。
- activation量子化にはtensor/token/group、row-major/consumer向けlayout、OCP/FNUZ、static/dynamicという
  別contractがある。異なるscale layoutやencodingを同じcache identityへまとめない。
- producerとactivation量子化を融合する候補は存在する。固定vLLM treeのSilu-mul＋FP8量子化と、固定SGLang treeの
  activation/permute＋量子化は、BF16の中間round-tripとlaunchを減らせることのreader事実だけを示す。
  sLLMでの採否はQwen graphのconsumer共有、追加write、request lifetimeを独立に測って決める。
- GEMM providerはshape、target、scale方式、workspace、layoutで選択されるため、最初のheuristic solutionを
  decode M=1とprefill M>1へ共通適用する根拠はない。benchmarkもM/K/Nとbackendを固定して比較する。
- reader-only sourceにあるCUDA/Triton/engine固有のkernel構造、定数、dispatch順序、API wrapperは実装basisにしない。

確認pathは主に`reference/vLLM/vllm/_custom_ops.py`、
`reference/vLLM/vllm/model_executor/layers/quantization/input_quant_fp8.py`、
`reference/vLLM/vllm/model_executor/layers/quantization/utils/fp8_utils.py`、
`reference/vLLM/vllm/model_executor/kernels/linear/scaled_mm/`、
`reference/SGLang/python/sglang/srt/layers/quantization/fp8.py`、
`reference/SGLang/python/sglang/srt/layers/quantization/fp8_utils.py`、
`reference/SGLang/python/sglang/benchmark/one_batch.py`である。

## llama.cppから得る実装候補

- fixed treeはMMVQとMMQを分け、quantized weightとQ8 activationのdotをblock単位で行う。decodeとprefillで
  同じ一output-element kernelを共有せず、入力再利用とtile形状を分ける設計根拠になる。
- NVFP4はblock 16のpacked valueとscaleを直接読むvec-dotおよびMMQ tile loadを持つ。sLLMでもpacked residentを
  維持し、decodeでは一waveに複数output、prefillでは同じweight tileを複数M rowで共有する候補を優先する。
- Q8 activationのblock scale再利用、vector packed load、wave reductionは候補になるが、integer Q8 dotをFP8
  matrix engineと同一経路とは扱わない。FP8はAMD datatype/APIとdispatch evidenceで別に確認する。

確認pathは主に`reference/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`、
`reference/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`、
`reference/llama.cpp/ggml/src/ggml-cuda/mmq-load-tiles.cuh`、
`reference/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh`、
`reference/llama.cpp/ggml/src/ggml-cuda/quantize.cu`である。

## Phase 15Oの実装basis

- sLLMの実装は既存sidecar format、AMD/ROCmの公開datatype/API、現在のnative ABI、独立FP32 oracleを正本にする。
- 最初のcandidateは既存sLLM kernelからの独立変更とし、外部sourceを直接reuseしない。後でllama.cppの表現を
  reuseする方が明確に有利と判断した場合は、実装前に対象blob単位のprovenance/notice記録へ切り替える。
- NVFP4はdecode wave-GEMVとprefill M-row tiled provider、FP8はactivation量子化とM=1/M>1 solutionを別laneで
  評価する。reader sourceの性能値はsLLM candidateの採用証拠に使わない。
