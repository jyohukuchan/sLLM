# SQ8 vLLM Same-Model Readiness

## 前回の要点

- SQ8_0計画のM10には、後半で `vLLM + Qwen3-14B-FP8` と比較する枠を追加済み。
- 既存の比較JSONLには、uLLM Qwen3.5-9B SQ8_0 smoke行、vLLM Qwen3-14B-FP8 smoke行、vLLM representative行が入っている。
- 現時点のuLLM行はQwen3.5-9Bなので、同一モデルのthroughput結論には使えない。

## 今回の変更点

- `Qwen3-14B-FP8` のローカルHugging Face directoryが存在し、configがQwen3 dense 40層FP8であることを確認した。
- `/tmp` とrepo内の `.ullm.d` を確認し、既存packageはQwen3.5-9B系が中心で、Qwen3-14BのuLLM packageは未確認と整理した。
- SQ8_0 artifact builder dry-runでは `281` FP8 tensor、`442` passthrough tensor、compact resident estimate `15557220864` bytesを確認した。
- `ullm-quant` package planner dry-runでは `723` total tensor、`280` supported tensor、`443` passthrough tensorを確認した。
- runtime側は `model.language_model.*` tensor名を前提にしているが、Qwen3-14B-FP8 safetensorsは `model.*` tensor名なので、same-model row前に名前解決が必要と判断した。

## 次の行動

- Qwen3 tensor namespace strategyを決める。短期はmanifest/package生成時rename、長期はruntime側のtensor namespace resolverが有力。
- Qwen3-14B-FP8のuLLM packageをbounded memoryで生成し、`manifest-all` 40層のSQ8_0 smoke rowを追加する。
- prompt guard bundleを添付し、vLLMのsmoke/representative行と同一モデル・同一workload条件で比較する。
