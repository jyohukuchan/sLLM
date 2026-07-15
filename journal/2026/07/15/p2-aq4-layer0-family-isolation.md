# P2 layer0 AQ4 weight-family isolation

## 前回の要点

既存の CPU formal probe は AQ4 QKV の単独出力を固定していたが、QKV/Z/A/B 各重み族と BF16 source checkpoint の直接比較は未分離だった。

## 今回の変更点

- CPU-only Rust 診断 `ullm-aq4-layer0-family-isolation` を追加した。既存 production standalone AQ4 matvec を QKV/Z/A/B ごとに一回ずつ使い、入力 3 行を一度に保持して処理する。
- Python 比較 `compare-aq4-layer0-family-isolation.py` を追加した。source の BF16 tensor を一族ずつ `safe_open` し、同一 f32 入力に対して source weight を f32 へ明示変換して CPU matmul を行った。
- package manifest、各 AQ4 index/scale/codebook、source model index、source tensor payload、入力 SHA、shape/layout、行ごとの出力 SHA を report に固定した。
- `one_at_a_time_hybrid` は、production/source の recurrent state と layer 実行式を推測せず `not_implemented` とした。
- artifact: `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-family-isolation-v0.1/`

## 結果

3 行すべて有限で、policy threshold は設定せず、promotion=false、holdout=not_run とした。

| family | aggregate relative L2 | cosine | max abs |
|---|---:|---:|---:|
| QKV | 0.0256654451 | 0.9996858735 | 0.8943977356 |
| Z | 0.0294115631 | 0.9995795800 | 0.4238085747 |
| A | 0.0204237158 | 0.9997918242 | 0.3511457443 |
| B | 0.0185483902 | 0.9998365787 | 0.1360249519 |

相対 L2 の診断候補は Z、絶対最大誤差の候補は QKV だった。これは次の調査対象を示すだけで、品質判定や昇格判定ではない。

主要 SHA:

- QKV sidecar: `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473`
- Z sidecar: `7ed98f1c7f8988958377b548f44afe3a2ddc5180150d1e3191c7d0e2a408b286`
- A sidecar: `8ed8799b510e003f1f5c509656b9680ebb1beb5743ab30bc6527ca52f77d26be`
- B sidecar: `b8f1afead1f417b53c4db4c80efc271d0589312583feb2470bd26fa90f84ef66`
- comparison report: `8ac589a47783c2ea839249498f72e7ff7bdb2d2da3d742e7ed8d21fdc6658d90`

## 検証

- `cargo check -p ullm-engine --bin ullm-aq4-layer0-family-isolation`: 成功
- `python3 -m py_compile tools/compare-aq4-layer0-family-isolation.py`: 成功
- `pytest -q tests/test_aq4_layer0_family_isolation.py`: 4 passed
- 実測 probe と source 比較: 3 行 × 4 family、nonfinite 0、package/input identity の実行前後一致を確認

## 次の行動

この branch を親へ通常 commit として渡す。P2 policy は変更せず、Z/QKV の数値を次の fidelity 修正候補の入力にする。
