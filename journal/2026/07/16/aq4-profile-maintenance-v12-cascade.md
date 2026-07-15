# AQ4 profile maintenance v12 cascade

## 変更

- maintenance harness を launcher `7f961f8d`、capture helper `1aed601a`、execute-binding v9 `dc9c12b6` に更新した。
- launcher 経由で selection producer `c8becac6` と family classifier `e4f8583a` を消費する。
- current 出力を profile execute/evidence/capture/maintenance v9、ready/dry-run v12 に進めた。
- ready v7〜v11 と actual-v9 の sealed roots は履歴読み戻しだけに残し、current v9 の runtime/evidence/maintenance/capture が未生成であることを回帰テストにした。
- QA の exact test file blob と件数を更新し、12ファイル、639件、失敗0件へ固定した。

## 検証

- maintenance tests: 156 passed
- QA Python aggregate: 617 passed
- resident driver unit tests: 22 passed, 0 failed
- QA aggregate: 639 passed, 0 failed
- `python3.12 -m py_compile`: passed
- `git diff --check`: passed
- GPU、service、actual は実行していない。

## コミット境界

- `5ffdaafe`: v12 assertion と historical readback の先行固定
- maintenance source と本記録は、上記 test commit/blob を QA authority として後続コミットに固定する。
