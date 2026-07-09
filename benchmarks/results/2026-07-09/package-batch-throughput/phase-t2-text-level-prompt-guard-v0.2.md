# T2 text-level prompt suite guard v0.2

## 前回の要点

- T2の短いtoken-id guardでは、text-level guardが未採用だったためstrict top1一致をpromotion ruleにしていた。
- 既存のprompt-suite guardは `generated_token_ids`、stop状態、top logitsを比較していたが、decoded text一致を独立した合格条件としては保存していなかった。

## 今回の変更点

- `tools/compare-package-token-prompt-suite.py` を `package-token-prompt-suite-generated-text-guard-v0.2` に更新した。
- `decoded_text.generated` と `decoded_text.generated_without_stop_sequence` の完全一致をcase単位で保存し、prompt-suite guardのpass条件に追加した。
- `tools/run-package-prompt-guard-bundle.py` もv0.2へ更新し、text match metricsをbundle summaryへ露出した。
- 既存R9700 AQ4 v0.3 prompt suiteをself-compareして、既存実測形式でv0.2 guardが通ることを確認した。

## Verification

| check | result |
| --- | --- |
| unit test | `python3 -m unittest tests.test_compare_package_guards` passed |
| self compare source | `benchmarks/results/2026-07-06/prompt-suite-aq4-matvec-add-rpb8-r9700-v0.3/summary.json` |
| schema | `package-token-prompt-suite-generated-text-guard-v0.2` |
| compared cases | 7 |
| generated token match | 7 / 7 |
| generated text match | 7 / 7 |
| no-stop generated text match | 7 / 7 |
| top logits match | 7 / 7 |
| passed | true |

Artifacts:

- `aq4-self-compare.json`
- `aq4-self-compare.md`

## 判断

- prompt-suiteでは、token ID一致だけでなくdecoded text一致も正式なguard条件として扱える。
- short token-id model-loop guardは引き続きstrict top1で候補境界を切る。
- full SQ policyへ昇格する段階では、v0.2 prompt-suite guard bundleのtoken/text/logit一致を要求する。

## 次の行動

1. layer23 `k16` passing branchまたは次のSQ candidateを、v0.2 prompt-suite guardへ接続する。
2. prompt-suiteを通したcandidateだけをbatch throughput / memory comparisonへ進める。
