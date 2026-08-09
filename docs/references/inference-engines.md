# 推論engine参照

## 位置付け

- ここで扱うsourceは、sLLMのarchitecture、scheduler、KV cache、量子化、native kernel、CI/testを調査するための参照である。sLLMの対応実績、性能証拠、実装の正しさの証明ではない。
- 固定された7件の取得事実と再現コマンドは [source-lock manifest](source-lock.md) を正とする。観測日は2026-08-02であり、ここで「release」と記すものは、その日に公式GitHubで `draft=false` かつ `prerelease=false` として観測した識別子である。将来のlatestを意味しない。
- llama.cpp以外のengineはreader-onlyの参照とし、source codeのcopy・adapt・portを行わない。
- llama.cppからの直接reuseだけは、`docs/provenance/README.md` のexact import recordとnotice processを完了した場合に限り許可する。対象ファイルのlicense/copyright、upstream URL、完全SHA、source/local path、blob ID、hash、`exact`/`adapted`/`ported`区分、変更内容、import commitを記録し、`THIRD_PARTY_NOTICES.md` とsource-file headerを整備する。

## 固定している一次参照

| engine | official source | version / full commit SHA | 主な参照範囲 |
| --- | --- | --- | --- |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `b10227` / `f5919bf458ef190468b5c329bb293f8a54a1e69c` | tokenizer、GGUF/vocabulary、model parser、軽量baseline |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `v0.26.0` / `568afb3a13806beb53bb2e6bd518269357b237c0` | scheduler、paged/block KV、量子化、hardware別test構成 |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `v0.5.16` / `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` | serving scheduler、KV/radix系設計、test分割。tag objectは `d21f3c3a10606ba3c7bf43f981496da0a7d620cd` |
| TensorRT-LLM | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | `v1.2.1` / `376f7e1bd8ed543f75014309e3fd4b237e9b0e73` | engine build、GPU matrix、量子化、backend/stage別CI |
| ROCm/ATOM | [ROCm/ATOM](https://github.com/ROCm/ATOM) | `v0.1.5` / `b0071c550ba3c99b1e9218debb91a6f81550da9a` | AMD/ROCm向けnative kernel、shape guard、compile/cache運用 |
| LMDeploy | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | `v0.15.0` / `f4b8140ba19cd823c541241cbb113cc32f854e6a` | schedulerの待ち合わせ、blocked KV、quantization境界、native kernelの登録とfallback |
| KTransformers | [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | `v0.6.4` / `924754a00bd8e5c6a2ad97929065c113f35782cf` | CPU-GPU協調実行、MoE offload、weight/KV配置、実行計画 |

llama.cppのlocal `models/` にある19 vocabulary GGUF（合計 `77556152` bytes）は参照用fixtureであり、weightを取得したものではない。TensorRT-LLMはLFS smudgeをskipし、4,121 pointerのみを保持する。KTransformersの4 gitlinkは未初期化であり、submoduleのsourceは取得していない。これらの特殊事情とlicense状態は [source-lock manifest](source-lock.md) に記録する。

## 今回未採用の調査済みsource

2026-08-02の候補調査で確認した次の6件は、今回正式sourceに採用しない。localへcloneせず、今後の採用予定にも置かない。表は過去の調査事実を簡潔に残すものであり、優先順位や将来の採用意思を示さない。

| source | official GitHub | 2026-08-02の観測 | 過去に想定した調査範囲 | 採用状態 |
| --- | --- | --- | --- | --- |
| MLC LLM | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | `main`。stable releaseは観測できず | compiler、runtime、multi-backend | 今回未採用。cloneしない。採用予定なし |
| Candle | [huggingface/candle](https://github.com/huggingface/candle) | `main` | Rust tensor、loading、device、kernel | 今回未採用。cloneしない。採用予定なし |
| CTranslate2 | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | `v4.8.1` | runtime、量子化、device abstraction | 今回未採用。cloneしない。採用予定なし |
| OpenVINO GenAI | [openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai) | `2026.1.0.0` | graph/runtime、heterogeneous device | 今回未採用。cloneしない。採用予定なし |
| ONNX Runtime GenAI | [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) | `v0.13.1` | model graph、generation runtime、provider境界 | 今回未採用。cloneしない。採用予定なし |
| TGI | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) | `v3.3.7`。2026-03-21 archived、maintenance-only | serving/API、batching、deployment | 今回未採用。cloneしない。採用予定なし |

## 調査の受入条件

- 固定7件は [source-lock manifest](source-lock.md) のfull SHAとlocal checkout fact checkに一致する。
- 今回未採用の6件をlocalへcloneせず、採用予定または優先調査対象として扱わない。
- 技術的要点はreaderとして抽出し、llama.cpp以外のengineのcodeをimplementerへ直接渡したり、sLLMへcopy・adapt・portしたりしない。llama.cppの直接reuseは `docs/provenance/README.md` の記録・notice processに従う。
- llama.cppから実際にimportする場合は、projectのprovenance方針に従い、license/copyright noticeと `THIRD_PARTY_NOTICES.md` の要否を判定する。

## 関連文書

- 今回の採用作業は [採用計画](../plans/archive/2026/08/1-10/reference-source-adoption.md) と [採用履歴](../history/2026/08/1-10/reference-source-adoption.md) を参照する。
- Qwen3.5 Phase 3の固定llama.cpp/vLLM reader結果は[Qwen3.5 Phase 3 reader記録](qwen3.5-phase3-reader.md)を参照する。
- Qwen3.5 full-model text path、state/cache、tensor分類、CLI/G3のreader結果は[Qwen3.5 Phase 3 full-model reader記録](qwen3.5-phase3-full-model-reader.md)を参照する。
