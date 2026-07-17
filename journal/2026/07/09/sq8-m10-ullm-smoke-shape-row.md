# SQ8_0 M10 uLLM Smoke-Shape Row

## 前回の要点

- M10にはvLLM + FP8との比較が含まれている。
- 比較前提として、uLLM SQ8_0側にも同じprompt/generation/concurrency形状のrowが必要だった。
- 既存のbehavioral prompt-suite guard bundleは、SQ8_0 rowへ添付できるようになっている。

## 今回の変更点

- `prompt_tokens=16`、`generated_tokens=8`、`concurrent_requests=1`のuLLM SQ8_0 rowを取得した。
- 結果は`benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl`へ保存した。
- `summary.md`とM10 planに、これはQwen3.5-9Bの測定経路rowであり、Qwen3-14B-FP8 vLLMとのsame-model比較ではないことを明記した。

## 次の行動

- vLLM smoke rowを同じ`results.jsonl`へ追加し、unsupportedまたはsuccessful baselineとして保存する。
