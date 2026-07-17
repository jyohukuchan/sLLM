# Pre-SQ Runtime TPS Plan v0.1

## 前回の要点

- hidden3994は `qwen35_9b_p4p46_hidden3994_v1` として名前付きpolicy化済み。
- CPU/R9700/V620の5 fixture gateでaccepted。
- 現状はlayer/prefix smoke中心で、sq format策定に必要な現実的TPSはまだ不足している。

## 今回の変更点

- `uLLM-project/docs/plans/pre-sq-runtime-tps-plan-v0.1.md` を追加した。
- sq format策定前に必要な作業を、token IDs入力のend-to-end runtime、長いprefill/decode、TPS harness、correctness guard、BF16/materialized AQ baselineに分けた。
- TP、batch、server API、tokenizer統合はpre-sq範囲から外した。

## 次の行動

1. T0でaccepted package、device IDs、benchmark schemaを固定する。
2. T1でtoken IDsからembedding、全decoder layer、final RMSNorm、lm_headまで通す。
3. T2/T3で `prompt_tokens=512`, `generated_tokens=256` のR9700/V620 TPSを保存できるCLIを作る。

## Estimate

- optimistic: `9 days`
- realistic: `2-3 weeks`
- if lm_head/full logits or VRAM handling needs major rework: `4 weeks`
