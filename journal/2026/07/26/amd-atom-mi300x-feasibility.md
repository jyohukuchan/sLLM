# AMD ATOM MI300X feasibility research

Date: 2026-07-26

## 前回の要点

- `existing-engine-benchmark-plan-v0.1.md` は ROCm/ATOM を既存 engine の候補に置いていたが、MI300X×1 / Qwen3-14B-FP8 の実行可否と最適化対象かどうかは未調査だった。
- 現在は別プロセスが CPU 64 コアで F32 reference corpus を生成中であり、GPU、`ullm-openai.service`、served-model manifest、`/opt/ullm` を操作しない調査枠だった。

## 今回の変更点

- AMD 公式 `ROCm/ATOM`、GitHub releases、AMD Docker Hub、ROCm 7.2.4 compatibility/release notes、Qwen 公式 checkpoint config を確認した。ATOM は AITER 上の native vLLM-like engine であり、OOT vLLM plugin と SGLang model-implementation backend は別経路である。
- AMD が nightly CI 検証済みとして列挙するのは DeepSeek-R1-0528、GLM-5、GPT-OSS-120B、Kimi-K2 系、Qwen3-235B、Qwen3-Next である。Qwen3-14B と Qwen3-Coder-Next の exact checkpoint はそこにない。
- Qwen3-14B-FP8 は `Qwen3ForCausalLM`、Qwen3-Coder-Next-FP8 は `Qwen3NextForCausalLM` なので ATOM native registry/FP8 loader の architecture 候補には一致する。ただし前者の MI300X 実行と性能、後者の exact checkpoint / 単一 MI300X / hybrid GDN 実行は未確認と固定した。
- ROCm 7.2.4 / gfx942 は AMD 対応範囲であり、matching production image `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4`（約 16.61 GB）がある。native server は OpenAI-compatible completion/chat API を持つ。
- ただし初回 compile 約 10 分、image pull、threadpool model load、Coder-Next の約 80.38 GB weights が残り数時間と CPU critical path に不利なため、ATOM full benchmark は no-go とした。将来の Qwen3-14B native smoke は CPU job 完了・image cache 済み・別途 GPU 許可時だけの条件付き候補である。
- vLLM の hybrid block size 544 failure と同じ設定経路は native ATOM の Qwen3-Next/GDN source には見つからなかった（default KV block 16、GDN state は separate pool）。ただし gfx942 上の exact Coder-Next Triton 成功は未確認である。

## 保存状態と次の行動

- 根拠、tag/digest、起動参考案、未確認事項、no-go 理由は `docs/research/amd-atom-mi300x-feasibility-2026-07-26.md` に固定した。
- 親 benchmark plan の ROCm/ATOM 行から同調査へ参照を追加した。
- GPU、service、activation/campaign、served-model manifest、`/opt/ullm`、既存 benchmark result には変更を加えていない。ATOM 実機起動・Docker pull・model download・benchmark は実施していない。
