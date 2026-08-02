# 参照source取得履歴

## 2026-08-02

- 公式GitHubの参照対象として、llama.cpp `b10227`、vLLM `v0.26.0`、SGLang `v0.5.16`、TensorRT-LLM `v1.2.1`、ROCm/ATOM `v0.1.5` を選定した。いずれも、この日に観測した `draft=false` かつ `prerelease=false` のreleaseであり、将来のlatest宣言ではない。
- 次の完全commit SHAをsource lockへ記録した。
  - llama.cpp: `f5919bf458ef190468b5c329bb293f8a54a1e69c`
  - vLLM: `568afb3a13806beb53bb2e6bd518269357b237c0`
  - SGLang: `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`
  - TensorRT-LLM: `376f7e1bd8ed543f75014309e3fd4b237e9b0e73`
  - ROCm/ATOM: `b0071c550ba3c99b1e9218debb91a6f81550da9a`
- SGLangのannotated tag object `d21f3c3a10606ba3c7bf43f981496da0a7d620cd` をcommit SHAと分けて記録した。
- 各local checkoutのorigin、path、top-level licenseを確認し、5件すべてで shallow、detached、clean、recursive submodule status空を確認した。
- TensorRT-LLMは `GIT_LFS_SKIP_SMUDGE=1` で取得し、4,121 LFS pointerを確認した。pointerが示すpayloadは取得していない。
- llama.cppの `models/` には語彙GGUF 19件、合計 `77556152` bytesが存在した。これはvocabulary fixtureであり、model weightではない。
- projectの `/reference/` ignoreを変更せず、source treeをGitへtrackしない方針を維持した。
- LMDeploy、MLC LLM、KTransformers、Candle、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIを未cloneの候補researchとして順位付けした。candidateのshort SHAはdiscovery identifierであり、lock値ではない。TGIは2026-03-21 archived、maintenance-onlyとして記録した。
- first tierは、LMDeployのscheduler/blocked KV/quantization/native kernels、MLC LLMのcompiler/runtime/multibackend、KTransformersのCPU-GPU/MoE/offload、CandleのRust tensor/loading/device/kernelとした。
- 固定sourceを配置したため、active CI計画の空directory・将来配置の記述をsource-lock manifest利用可能へ更新した。一方、固定exact revisionを一次sourceとして行うCI/test再調査は未完了のまま残した。
- 取得事実、再現コマンド、license/provenance境界、candidate調査を [source-lock manifest](../../../../references/source-lock.md) と [推論engine参照](../../../../references/inference-engines.md) に反映した。

[対応する計画](../../../../plans/archive/2026/08/1-10/reference-source-acquisition.md)
