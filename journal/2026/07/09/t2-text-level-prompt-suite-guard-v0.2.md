# T2 text-level prompt suite guard v0.2

## 前回の要点

- T2ではlayer23 `k16` までが現branchのpassing boundaryになった。
- layer23 `q4` / `v8` / `v4` は全てstrict top1を回復できなかった。
- 次の判断では、単発top1だけでなく、prompt-suite上の生成文字列が崩れないかを見るguardが必要だった。

## 今回の変更点

- `tools/compare-package-token-prompt-suite.py` を `package-token-prompt-suite-generated-text-guard-v0.2` に更新した。
- `generated_token_ids` に加えて、`decoded_text.generated` と `decoded_text.generated_without_stop_sequence` の完全一致を比較する。
- case reportには文字数、match flag、SHA256を保存する。
- summary metricsに `generated_text_match_count` と `generated_without_stop_text_match_count` を追加した。
- `tools/run-package-prompt-guard-bundle.py` もv0.2へ上げ、bundle summaryからtext match metricsを読めるようにした。

## 検証

- `python3 -m unittest tests.test_compare_package_guards` はpass。
- 既存R9700 AQ4 v0.3 prompt suiteをself-compareし、7/7でtoken/text/no-stop text/top logitsが一致した。
- self-compare artifact:
  - `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-text-level-prompt-guard-v0.2/aq4-self-compare.json`
  - `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-text-level-prompt-guard-v0.2/aq4-self-compare.md`

## 判断

- text-level guardはstrict top1 failureを緩和するためではなく、full SQ policy昇格前に実際の生成文字列が崩れていないことを確認する追加gateとして扱う。
- 次はlayer23 `k16` passing branchまたは次のSQ candidateをv0.2 prompt-suite guardへ接続する。
