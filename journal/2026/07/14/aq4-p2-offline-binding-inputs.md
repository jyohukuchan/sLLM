# AQ4 P2 オフライン binding input

## 前回の要点

P0/P1 の live-v4 artifact は存在するが、P2 の production path 実行と同一 artifact all-M=1 oracle は未実行だった。P1 の executor record には canonical model graph/state schema が含まれていた。

## 今回の変更点

- `uLLM-project/tools/generate-aq4-p2-offline-binding.py` を追加した。active manifest、P0 snapshot、P1 live-v4 executor record/trace を相互照合し、4-field `model_identity.json` を canonical 生成する。
- P1 canonical graph/state を `graph.json` と `state.json` へ lossless 抽出し、P1 trace の schema/source/digest と再計算 hash を検証する。
- duplicate JSON key、unknown field、identity/trace mismatch、canonical/hash tampering、source-oracle substitution を fail-closed する validator と tests を追加した。
- correctness threshold の既存 template/spec を監査した。必須5数値は全て未定義（null）であるため、値を発明せず `correctness-threshold-audit.json` を `BLOCKED` とした。
- `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/` に生成物、`hash-manifest.json`、`SHA256SUMS`、validation report を保存した。source oracle は独立 source として記録し、production path oracle へ代用していない。promotion eligible は常に false。

## 次の行動

P2 の実 production request boundary と same-artifact all-M=1 path oracle を別工程で実行し、review済み correctness threshold policy を binding するまでは promotion しない。

## 検証

- `python3 -m py_compile tools/generate-aq4-p2-offline-binding.py`
- `python3 -m unittest -v tests/test_aq4_p2_offline_binding.py`（7 tests、全て成功）
- 生成 bundle の `--validate`（成功、`promotion_eligible=false`、`path_oracle_status=not_run`、`correctness_threshold_status=blocked`）

## 独立QA修正

- validation reportをexact schema、固定semantic、artifact linkのpath/hash/typeまで独立検証するようにした。
- bundle directoryの実ファイル集合を8ファイルへ固定し、unexpected、missing、symlink、非regular fileを拒否するようにした。
- JSON、SHA256SUMS、外部inputを一つのfdからbounded readし、open前後のdevice、inode、size、mtime、ctimeを照合する。rename、append、同一size rewriteを決定的testで拒否した。
- output directory、input/output file、全親componentのsymlinkを拒否するようにした。
- `python3 -m unittest -v tests/test_aq4_p2_offline_binding.py`（16 tests、全て成功）
- artifact再生成後のstrict `--validate`、`py_compile`、`git diff --check`に成功した。

## Hardlink QA修正

- bundle内の全regular file、SHA256SUMS、外部inputを`st_nlink == 1`へ限定した。directoryにはhardlink数1の仮定を適用していない。
- fresh bundleのSHA256SUMS外部hardlink、graph artifact外部hardlink、active manifest/P1 executor record input hardlinkを拒否するtestを追加した。
- `python3 -m unittest -v tests/test_aq4_p2_offline_binding.py`（既存16 + 新規3 = 19 tests、全て成功）
