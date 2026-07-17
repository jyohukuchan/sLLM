# AQ4 final RMSNorm additive weight fix v0.1

## 前回の要点

- Phase 3d の CPU-only chain（decoder layer 0--31 + final RMSNorm + LM head fixed-row sample）では、decoder stack 内の相対 L2 はおおむね `0.075`--`0.171` で振動した一方、layer 31 output の `0.1278813307` から final RMSNorm の `0.5010330688` へ `3.917953x` 急増した。LM-head 34-row sample も `0.5860500940` であり、既知の最終値 `0.6151289249` と同じ規模だった。
- source/package の BF16 payload identity は既に一致しており、データ差ではなく AQ4 側の final norm 数式適用漏れが候補だった。package manifest の実名は `model.language_model.norm.weight` で、raw BF16 payload（4096 elements）の mean absolute value は今回も `1.139947243035` と確認した。従来の `< 0.75` 値ベース推測ではこの tensor を additive と判定できない。

## 今回の変更点

### source semantics と判定条件

- Transformers の Qwen3.5 source `modeling_qwen3_5.py` を確認した。`Qwen3_5RMSNorm` は weight を zero 初期化し、正規化後に必ず `output * (1.0 + self.weight.float())` を適用する（lines 736--750）。
  - decoder `input_layernorm` / `post_attention_layernorm` は同じ `Qwen3_5RMSNorm` を使う（lines 756--767）。
  - attention `q_norm` / `k_norm` も同じ class を使う（lines 650--670）。
  - text model の final `norm` も同じ class を使う（lines 1137--1147）。
  - 従って、今回の対象5種（input, post-attention, q, k, final）は一律 additive convention である。
- ただし、`linear_attn.norm` は別の `Qwen3_5RMSNormGated` である。これは weight を one 初期化し、正規化値へ raw `self.weight` を乗算してから gate を掛ける（lines 187--202）。したがって今回の additive 集合には含めない。
- 同じ suffix を持つ Qwen3 は `Qwen3RMSNorm` で weight を one 初期化し raw weight を乗算する（`modeling_qwen3.py` lines 49--65）。そのため全アーキテクチャ共通の「suffixだけなら additive」という判定にはせず、Qwen3.5 package/runtime path 専用の判定に分離した。

### 実装

- `crates/ullm-engine/src/loader.rs` に `effective_qwen35_rmsnorm_weight_values` を追加した。この helper は、Qwen3.5 path からのみ呼び、以下を tensor 名だけで無条件に `1.0 + raw_weight` にする。
  - `model.language_model.norm.weight`
  - `model.language_model.layers.*.input_layernorm.weight`
  - `model.language_model.layers.*.post_attention_layernorm.weight`
  - `model.language_model.layers.*.self_attn.q_norm.weight`
  - `model.language_model.layers.*.self_attn.k_norm.weight`
- `linear_attn.norm.weight` は name predicate から除外した。Qwen3.5 path では値ベース heuristic に依存しない。
- 既存の一般化 helper `effective_rmsnorm_weight_values` は legacy Qwen3 caller のために残し、明示的に legacy value-inference として分離した。Qwen3.5 AQ4 path はこの legacy helper を使わないため、final norm の `mean_abs < 0.75` 問題は回避される。
- unit test は対象5種すべてを `1.14`（旧閾値超過）で additive と確認し、`linear_attn.norm.weight` は raw のままと確認するよう拡張した。
- CPU diagnostic の `ullm-aq4-layer0-family-isolation`、Qwen3.5 AQ4 per-layer runtime、Qwen3.5 AQ4 model runtime（final norm）、および Qwen3.5 package smoke/main-parts の全 direct caller を新 helper へ切り替えた。従って CPU reference と GPU production runtime は同じ Qwen3.5 name-only semantics を参照する。GPU 実行は今回していない。

### SQ8_0 影響評価

- SQ8 modules は `effective_qwen35_rmsnorm_weight_values` / `effective_rmsnorm_weight_values` のどちらも呼ばない。SQ8 serving の tensor prefix は `model.layers.*`、final norm は `model.norm.weight` であり、独自の raw-BF16 reader を使う。
- SQ8 が対象とする Qwen3 source は raw-weight RMSNorm であることも source code で確認した。したがってこの commit は SQ8_0 のコード・tensor semantics・release gate を変更しない。SQ8 側に同種の問題があるかの判定や修正は本作業の範囲外である。

### CPU-only 検証

- `cargo check -p ullm-engine` — 成功（既存 C++ `subobject-linkage` warning のみ）。
- `cargo test -p ullm-engine loader::tests::effective_` — `2 passed`。
- `cargo test -p ullm-engine --lib` — `729 passed; 0 failed; 1 ignored`。loader/RMSNorm 関連を含む lib suite は成功した。
- `cargo build --package ullm-engine --bin ullm-aq4-layer0-family-isolation` — 成功（同じ既存 C++ warning のみ）。
- `cargo test -p ullm-engine` は test 実行前に、未変更の `examples/sq8_ck_serving_performance.rs` を standalone example として compile する際の既存エラー（`super` import / `main` / imports が不足、74 errors）で失敗した。SQ8_0 変更禁止の範囲なので修正せず、上記 `--lib` suite と targeted loader test で本変更を検証した。
- source commit に対する `git diff --check` は成功した。`cargo fmt --all -- --check` は本変更外の既存 formatting drift により repository-wide では成功しないため、既存箇所を整形目的で変更していない。

### 同一 fixture での再測定

新規 evidence root は `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-final-rmsnorm-fix-v0.1/` とし、既存の Phase 3d/Phase 3c/P3 evidence を上書きしていない。いずれも package、source model、3-context hybrid input、CPU-only binary を同一にして再実行した。各 compare artifact の `SHA256SUMS` は全件 `OK`、各 comparator は `status: valid` である。

| Phase 3d endpoint | 修正前 relative L2 | 修正後 relative L2 | 変化 |
| --- | ---: | ---: | ---: |
| decoder layer 31 output | 0.1278813307 | 0.1278813307 | 同一 |
| final RMSNorm（full hidden） | 0.5010330688 | 0.1688691127 | -66.30% |
| LM head fixed 34-row sample | 0.5860500940 | 0.0582126936 | -90.07% |

- final norm / layer31 ratio は `3.917953x` から `1.320514x` へ低下した。final norm はまだ layer31 と完全同一ではないが、`0.1` 台へ戻り、旧来の boundary spike は解消された。LM head は fixed-row sample であり、full-vocabulary fidelity gate の代替ではない。
- Phase 1 layer0 standalone は修正前後とも `0.04245138374421657` で完全一致した。
- Phase 2/2c layer 0--11 chain は12点すべて修正前の `observed_relative_l2` と完全一致した（layer0 `0.0424513837`、layer11 `0.0808269929`）。この fix による decoder layer 側の悪化は観測されない。
- 実行時間は phase1 `0:28.24`、phase2c `2:38.02`、phase3d `6:47.54`。すべて exit status `0`、major page faults `0` だった。

### Scope

- GPU、active production service、systemd unit、active manifest を変更・実行していない。
- 07/16 P3 harness および 07/17 Phase 3c の service-stop window tooling/evidence を変更していない。
- 正式な独立 holdout P2 fidelity gate は実行していない。

## 次の行動

1. 今回の CPU-only 改善をもって final RMSNorm additive 漏れの実装修正は確認できたが、GPU production path の実機再検証は別途明示承認後に行う。
2. GPU 再検証後も、正式 P2 fidelity gate は独立 holdout で別作業として実行する。fixed-row LM-head sample の改善だけで release/P2 gate を合格扱いにはしない。
3. final norm 後に残る `0.168869` の CPU relative L2 は、量子化誤差・既知の epsilon control 等と切り分ける対象として残す。ただし、今回の数式バグの再導入ではないことは layer31/final boundary の比較で確認できた。
