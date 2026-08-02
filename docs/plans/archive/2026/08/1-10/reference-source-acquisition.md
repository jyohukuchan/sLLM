# 参照source取得計画

## 状態

**完了（2026-08-02）**

この作業単位では、uLLMがCI、test、runtime、kernel、schedulerの設計を調査するための公式推論engine sourceを、再現可能なrevisionでlocal `reference/` に配置し、取得事実をmanifestへ固定する。sourceの実装利用、性能比較、CI/testのexact-revision再調査はこの計画の完了条件に含めない。

## 完了条件と結果

| 完了条件 | 結果 |
| --- | --- |
| 公式origin、local path、観測日、version、完全SHA、licenseを記録する | 5件を [source-lock manifest](../../../../../references/source-lock.md) に記録した |
| release選択の範囲を固定し、future latestを約束しない | 2026-08-02観測、`draft=false`・`prerelease=false` の公式releaseとして明記した |
| shallow detached clean、submoduleなしを確認する | 5件すべてで `HEAD (no branch)`、shallow、clean、recursive submodule status空を確認した |
| exact checkoutを再現するclone・検証コマンドを残す | manifestにorigin別clone、detach、40桁SHA、tag object、clean/shallow/submodule検証を記録した |
| 大きなsource/model payloadをtracked treeへ入れない | `/reference/` のignoreを維持し、sourceは未追跡のままとした。llama.cppは語彙GGUF 19件・`77556152` bytesのみ、TensorRT-LLMはLFS pointer 4,121件のみだった |
| 直接reuseと参照調査を分離する | license/provenance reviewが必要であること、候補sourceは未cloneでshort SHAをlock値にしないことを明記した |

## 取得対象

- llama.cpp `b10227` — `f5919bf458ef190468b5c329bb293f8a54a1e69c`
- vLLM `v0.26.0` — `568afb3a13806beb53bb2e6bd518269357b237c0`
- SGLang `v0.5.16` — commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`、annotated tag object `d21f3c3a10606ba3c7bf43f981496da0a7d620cd`
- TensorRT-LLM `v1.2.1` — `376f7e1bd8ed543f75014309e3fd4b237e9b0e73`
- ROCm/ATOM `v0.1.5` — `b0071c550ba3c99b1e9218debb91a6f81550da9a`

candidate researchは別記録としてLMDeploy、MLC LLM、KTransformers、Candleをfirst tierに、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIをwatch対象に順位付けした。candidateはcloneせず、short SHAをdiscovery identifierとしてのみ記録した。

## 引き継ぎ

- 固定sourceの技術的な読む順序とcandidate順位は [推論engine参照](../../../../../references/inference-engines.md) を正とする。
- active CI計画の「sourceを配置してから再調査する」という将来文は、source-lock manifestが利用可能になった状態へ更新した。ただし、固定exact revisionを一次sourceとして行うCI/test再調査自体は未完了であり、別のPhase 0作業として残る。
- 直接reuseを開始する場合は、対象ファイル、完全SHA、license、copyright、変更内容、import commit、noticeを含む独立したprovenance reviewを起票する。

[対応する履歴](../../../../../history/2026/08/1-10/reference-source-acquisition.md)
