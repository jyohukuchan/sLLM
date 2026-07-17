# Paged decode split production promotion

## 前回の要点

0245897時点では、generic split paged decodeはQwen3.5 AQ4 resident layerの実験環境変数でのみ選択され、canonical single planはQwen3.5 `.m1-gqa`を維持していた。

## 今回の変更点

- 実測crossoverに基づくgeneric typed production configを追加した。source tileは128、split thresholdはcache length 256とした。
- AQ4 resident layer loadでは、完全な実験環境変数pairを最優先し、pairがない場合はprobe済み `HipPagedDecodeAttentionSplit` featureでproduction configを選択する。featureなしではsingle-onlyを維持する。
- split configが選ばれた後のregistry/plan解決失敗はfail closedとした。canonical Qwen3.5 single planは常に既存 `.m1-gqa`を使用する。
- AQ4 workerとqwen35-9b-aq4 profileへ `ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL` を各1件追加した。SQ8 profile/pathと実験envは変更していない。

## 検証

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine qwen35_aq4_layer_runtime -- --test-threads=1`（17 passed）
- `cargo test -p ullm-engine aq4_worker_backend -- --test-threads=1`（4 passed）
- `cargo test -p ullm-engine --bin ullm-aq4-worker -- --test-threads=1`（11 passed）
- `cargo test -p ullm-engine --test worker_profile_snapshot -- --test-threads=1`（2 passed）
- `pytest -q tests/test_generate_served_model.py`（18 passed）
- `git diff --check`

## 次の行動

親エージェントが所有変更を統合して、production workerのisolated smokeとrocprofでsplit経路を確認する。
