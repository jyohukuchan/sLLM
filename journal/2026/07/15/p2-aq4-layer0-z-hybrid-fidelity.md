# P2 layer0 Z one-at-a-time hybrid fidelity isolation

## 前回の要点

既存のCPU family isolationでは、ZのAQ4単独投影がBF16 source matmulに対して4族中で相対L2最大の診断候補だった。一方、recurrent stateを含むlayer実行順を推測したハイブリッド比較は未実装だった。

## 今回の変更点

- `package_linear_attn_mlp_block_sequence_run`に、既定値が`None`の診断専用入力/Z override引数を追加した。通常の呼び出しはラッパー経由で変更されない。
- Z overrideは本番AQ4 Z matvec出力を取得した直後、production `silu_mul_f32`へ渡す直前だけに適用した。QKV/A/B、conv、gate/beta、recurrent state、attention norm、out projection、post norm、MLP、残差加算は同じproduction runtime経路を使う。
- `package-linear-attn-z-hybrid-diagnostic`をCPU専用CLIとして追加した。既存の固定3行入力を同じ順序で使い、baseline（全AQ4）とhybrid（Zだけsource BF16出力）を別RuntimeContextでゼロstateから実行する。
- `build-aq4-layer0-z-source-sidecar.py`で、固定f32入力×BF16 source Z weightの明示f32 matmul sidecarを生成した。source tensor/index/inputのidentityをreportへ記録する。
- artifactは`benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-z-hybrid-fidelity-v0.1/`に固定した。thresholds=null、policy_not_evaluated、promotion=false、holdout=not_runを維持した。

## 結果

- AQ4 baseline Z対source BF16: aggregate relative L2 `0.0294115631`、cosine `0.9995795800`、max abs `0.4238085747`。
- Z source override後: source Zとのaggregate relative L2 `0`、cosine `1`、非有限値`0`。
- Z置換がattention-blockへ伝播した差: aggregate relative L2 `0.0006750630`、max abs `0.0225958824`、cosine `0.9999997722`。
- layer outputへの差: aggregate relative L2 `0.0006983384`、max abs `0.0240631104`、cosine `0.9999997562`。
- いずれも閾値判定・昇格判定・holdout実行はしていない。Z量子化誤差は下流へ伝播するが、Z単独で現在の広いfull-model fidelity失敗を説明するかは未判定である。

## 検証

- `cargo check -p ullm-engine --bin ullm-engine`: 成功
- `cargo test -p ullm-engine --bin ullm-engine -- --test-threads=1`: 26 passed
- `python3 -m py_compile tools/build-aq4-layer0-z-source-sidecar.py`: 成功
- `pytest -q tests/test_aq4_layer0_family_isolation.py tests/test_aq4_layer0_z_hybrid.py`: 7 passed
- `git diff --check`: 成功
- 既定`package-linear-attn-mlp-block-smoke`を2回実行し、出力SHA `9ac224cc444569bb9e5c4c493eacf4007c06c862c03466da31a058a123e4ad9b`が一致した。

## 次の行動

親へ通常commitを渡し、P2のZ寄与を「診断候補」として統合する。次のprecision候補はZ単独置換を採用決定せず、QKV/Z/A/Bのsource-aware量子化を同じlayer境界で比較したうえで選ぶ。
