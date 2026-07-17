# SQ8_0 M10 vLLM FP8 Smoke Row

## 前回の要点

- uLLM SQ8_0側では、`prompt_tokens=16`、`generated_tokens=8`、`concurrent_requests=1`のsmoke-shape rowを保存した。
- vLLM baseline計画では、同じshapeで`Qwen3-14B-FP8`をR9700上で実行することになっている。
- モデルはuLLM側がQwen3.5-9B、vLLM側がQwen3-14B-FP8なので、same-model比較ではない。

## 今回の変更点

- `ROCR_VISIBLE_DEVICES=1`でvLLM smokeを実行した。
- vLLM rowは`status=ok`で、`benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl`へuLLM rowと同じJSONLに保存した。
- summaryとM10関連planに、vLLM smokeが成功したこととモデル差の制約を追記した。

## 次の行動

- representative `pp512/tg128/b1` のvLLM rowを追加する。
- same-model uLLM SQ8_0 rowが用意できるまでは、速度差を最終性能結論として扱わない。
