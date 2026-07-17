# SQ8_0 M10 vLLM FP8 Representative Row

## 前回の要点

- vLLM + Qwen3-14B-FP8 smoke rowはR9700で`status=ok`になった。
- uLLM SQ8_0側のsmoke-shape rowも同じJSONLに保存済みだが、モデルはQwen3.5-9Bでありsame-model比較ではない。
- M10 planにはrepresentative `prompt_tokens=512` / `generated_tokens=128` rowが後続として残っていた。

## 今回の変更点

- vLLM representative rowを`ROCR_VISIBLE_DEVICES=1`で実行した。
- `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl`へ3行目として保存した。
- summaryとM10関連planへ、prefill/decode/total tok/sとVRAM消費を追記した。

## 次の行動

- same-model uLLM SQ8_0 rowを用意するまで、vLLM行は外部FP8 feasibility baselineとして扱う。
