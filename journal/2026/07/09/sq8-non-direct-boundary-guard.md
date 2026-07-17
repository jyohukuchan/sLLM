# SQ8_0 non-direct boundary guard

## 前回の要点

- SQ8_0 projection dispatchはsingle/batch/pair/tripleのdirect descriptorを選び、R9700では
  `sq8_0_matvec*_r9700_direct` を選択できる状態になっている。
- C++側の非direct kernel familyは未実装なので、Rust実行境界は`Direct` family以外をruntime
  kernelへ渡してはいけない。

## 今回の変更点

- `backend_dispatch`の内部テストに、非direct SQ8_0 matvec descriptor fixtureを追加した。
- single/batch/pair/tripleの各operationで、非direct fixtureが選択できても
  `sq8_0_projection_descriptor_family()` は `None` を返すことを確認した。
- `part_00.rs`の境界テストで、`SqFp8ProjectionDispatch::require_direct_family()` が
  single/batch/pair/tripleすべてを明示errorで拒否することを確認した。
- `tools/run-external-benchmark.py` の記録対象envに
  `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC*` を追加し、SQ8_0 direct-kernel requirementが
  JSONLの `artifacts.command` に残るようにした。
- `tools/run-package-token-prompt-bench.py --require-hip-kernels` に
  `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1` を追加し、SQ8_0 prompt-suite/bench経路でも
  batch matvec境界がfail-closedになるようにした。

## 検証

- `cargo fmt --all --check`
- `cargo test -p ullm-engine sq8_projection -- --test-threads=1`
- `cargo test -p ullm-engine sq_fp8_projection_dispatch_rejects_non_direct_family_for_single_batch_pair_and_triple_boundaries -- --test-threads=1`
- `python3 -m unittest tests.test_external_benchmark_batch_parser`
- `python3 -m py_compile tools/run-package-token-prompt-bench.py`
- `/tmp/ullm-external-benchmark-env-test` へのdummy行生成で
  `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` が `artifacts.command` に残ることを確認。

## 次の行動

- 実際の非direct/R9700最適化kernel familyを追加するまでは、このguardをfail-closed境界として維持する。
- M10の既存full 40-layer比較行はR9700 descriptor導入前の `*_rdna4_direct` 記録なので、必要なら
  次回以降に最新dispatchで再測定する。
