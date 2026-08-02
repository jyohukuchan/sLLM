# 推論engine参照と今後の調査候補

## 位置付け

- ここで扱うsourceは、uLLMのarchitecture、scheduler、KV cache、量子化、native kernel、CI/testを調査するための参照である。uLLMの対応実績、性能証拠、実装の正しさの証明ではない。
- 固定された5件の取得事実と再現コマンドは [source-lock manifest](source-lock.md) を正とする。観測日は2026-08-02であり、ここで「release」と記すものは、その日に公式GitHubで `draft=false` かつ `prerelease=false` として観測した識別子である。将来のlatestを意味しない。
- vLLMおよびllama.cpp以外のengineはreader-onlyの参照とし、source codeのcopy・adapt・portを行わない。
- llama.cppからの直接reuseだけは、`docs/provenance/README.md` のexact import recordとnotice processを完了した場合に限り許可する。対象ファイルのlicense/copyright、upstream URL、完全SHA、source/local path、blob ID、hash、`exact`/`adapted`/`ported`区分、変更内容、import commitを記録し、`THIRD_PARTY_NOTICES.md` とsource-file headerを整備する。

## 固定している一次参照

| engine | official source | version / full commit SHA | 主な参照範囲 |
| --- | --- | --- | --- |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `b10227` / `f5919bf458ef190468b5c329bb293f8a54a1e69c` | tokenizer、GGUF/vocabulary、model parser、軽量baseline |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `v0.26.0` / `568afb3a13806beb53bb2e6bd518269357b237c0` | scheduler、paged/block KV、量子化、hardware別test構成 |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `v0.5.16` / `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` | serving scheduler、KV/radix系設計、test分割。tag objectは `d21f3c3a10606ba3c7bf43f981496da0a7d620cd` |
| TensorRT-LLM | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | `v1.2.1` / `376f7e1bd8ed543f75014309e3fd4b237e9b0e73` | engine build、GPU matrix、量子化、backend/stage別CI |
| ROCm/ATOM | [ROCm/ATOM](https://github.com/ROCm/ATOM) | `v0.1.5` / `b0071c550ba3c99b1e9218debb91a6f81550da9a` | AMD/ROCm向けnative kernel、shape guard、compile/cache運用 |

llama.cppのlocal `models/` にある19 vocabulary GGUF（合計 `77556152` bytes）は参照用fixtureであり、weightを取得したものではない。TensorRT-LLMはLFS smudgeをskipし、4,121 pointerのみを保持する。これらの特殊事情とlicense状態は [source-lock manifest](source-lock.md) に記録する。

## 今後の候補（未cloneのresearch snapshot）

下表は2026-08-02に行った候補調査の順位である。全て公式GitHub URLをorigin候補として記録したが、今回cloneしていない。SHAは調査時の **short discovery identifier** に過ぎず、lock値、再現可能なrevision、直接importの根拠ではない。固定して使う場合は、release/tagの存在確認、完全40桁SHA、license、clean shallow detached checkoutを別途取得して [source-lock manifest](source-lock.md) へ昇格させる。

| 順位 | candidate | official GitHub | observed version/ref | short discovery SHA | 調査対象 | 方針 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | LMDeploy | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | `v0.15.0` | `f4b8140` | scheduler、blocked KV、quantization、native kernels | first tier |
| 2 | MLC LLM | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | `main`（stable releaseは観測できず） | `2f78caa` | compiler、runtime、multi-backend | first tier |
| 3 | KTransformers | [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | `v0.6.4` | `924754a` | CPU-GPU協調、MoE、offload | first tier |
| 4 | Candle | [huggingface/candle](https://github.com/huggingface/candle) | `main` | `5447a87` | Rust tensor、loading、device、kernel | first tier |
| 5 | CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | `v4.8.1` | `0d8bcd3` | runtime、量子化、device abstraction | watch |
| 6 | OpenVINO GenAI | [openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai) | `2026.1.0.0` | `1dabb8c` | graph/runtime、heterogeneous device | watch |
| 7 | ONNX Runtime GenAI | [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) | `v0.13.1` | `db2baa9` | model graph、generation runtime、provider境界 | watch |
| 8 | TGI | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) | `v3.3.7` | `dfb3fbe` | serving/API、batching、deployment | archived 2026-03-21、maintenance-only |

### first tierの読む順序

1. **LMDeploy**: schedulerの待ち合わせ、blocked KV、quantization境界、native kernelの登録とfallbackを優先して読む。
2. **MLC LLM**: compiler/runtimeの分離、multi-backendのlowering、artifactとdevice contractを読む。`main`参照なので、stable releaseとして扱わない。
3. **KTransformers**: CPU-GPUの協調実行、MoEのoffload、weight/KV配置と実行計画を読む。
4. **Candle**: Rustでのtensor ownership、model loading、device abstraction、kernel境界を読む。uLLMのRust/C++境界の比較材料にする。

この4件は実装の直接移植先ではなく、uLLMのsemantic op contract、backend capability、KV layout、schedulerの設計判断を検証するためのreader対象である。CTranslate2、OpenVINO GenAI、ONNX Runtime GenAIはruntime/provider設計の比較、TGIは保守限定のserving/API比較に留める。候補sourceからの直接reuseは、候補のshort SHAだけでは開始せず、必ずfull SHAとprovenance/license reviewを完了してから判断する。

## 調査の受入条件

- 固定5件は [source-lock manifest](source-lock.md) のfull SHAとlocal checkout fact checkに一致する。
- candidateはshort SHAをlock値として扱わず、stable releaseの有無とmaintenance状態を明示する。
- 技術的要点はreaderとして抽出し、vLLMおよびllama.cpp以外のengineのcodeをimplementerへ直接渡したり、uLLMへcopy・adapt・portしたりしない。llama.cppの直接reuseは `docs/provenance/README.md` の記録・notice processに従う。
- 実際にimportする場合は、projectのprovenance方針に従い、license/copyright noticeと `THIRD_PARTY_NOTICES.md` の要否を判定する。
