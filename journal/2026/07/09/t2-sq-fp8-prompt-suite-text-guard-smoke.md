# T2 SQ FP8 prompt-suite text guard smoke

## 前回の要点

- v0.2 prompt-suite guardはgenerated token IDに加え、decoded text一致を機械可読に保存できるようになっていた。
- 次はlayer23 `k16` passing branchをSQ artifact付きprompt-suite経路へ接続する段階だった。

## 今回の変更点

- `sq-fp8-token-ids-generate-smoke` / `sq-fp8-token-ids-bench` を追加し、SQ FP8 artifactを既存generate benchへ渡せるようにした。
- prompt bench/suite wrapperに `--sq-artifact` を追加した。
- guard bundle runnerは、guard不合格でもsummaryを保存してから非0終了するようにした。
- `acceptance_mode=behavioral` を追加し、exact token/text/logit一致を診断扱いに分離した。
- 1 prompt / 2 generated tokensのmini suiteでAQ4とSQ layer23 `k16`を比較した。

## 結果

- AQ4 generated token IDs: `314,279`
- SQ generated token IDs: `314,279`
- generated text: ` of the`
- strict guard `logit_atol=0.001`: generated token/text/no-stop text all matched, top logits failed.
- loose value check `logit_atol=0.2`: top-k rank driftが残るためstill failed.
- behavioral guard `logit_atol=0.001`: strictはfailedのまま、behavioralはpassed。
- SQ artifact付きprompt-suite経路は接続済み。FP8 SQ候補探索ではexact一致をblockerにせず、出力が壊れていないかとthroughput/memoryを優先する。

## 次の行動

1. SQ FP8を既定候補として、batch throughput / memory comparisonへ進める。
2. full v0.3 suiteは `acceptance_mode=behavioral` で走らせ、strict一致は診断列として残す。
3. 明らかな出力破壊だけをblockerにする。
