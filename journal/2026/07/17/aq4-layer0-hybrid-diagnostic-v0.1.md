# AQ4 layer0 hybrid diagnostic v0.1

## 前回の要点

- 07/15のfamily単体matvec比較では、BF16 sourceに対するAQ4相対L2はQKV=0.0256654451、Z=0.0294115631、A=0.0204237158、B=0.0185483902で、量子化単体としては妥当な範囲だった。
- ただし実際のlayer0 forwardを通した比較器は未実装であり、07/14のlayer0から始まる大きな差分をどの中間段で増幅するかは判定不能だった。

## 今回の変更点

- `ullm-aq4-layer0-family-isolation`にCPU専用の`--hybrid-input`経路を追加した。AQ4復号+row-scale済みQKV/Z/A/B、Conv1d後SiLU、Q/K L2正規化、runtimeの`runtime_host_linear_attn_recurrent_f32`、head RMSNorm/Silu(Z)、attention residual、post RMSNorm、SwiGLU MLP residualまでをlayer0の順序で再現する。
- Conv stateはruntimeと同じ`[kernel, channel]`のrotate-left/append契約、recurrent stateは`[value_head, key_dim, value_dim]`契約でゼロから各contextをreplayする。layer0はlinear attentionなのでRoPEは`not_applicable`として明示記録した。
- BF16 source側は必要なembedding行だけを`safe_open(..., device="cpu")`で読んだcontext fixtureから計算し、AQ4 stdout frameとstageごとに即時比較する。永続化するのは固定座標、relative L2/cosine/max absのみであり、full hidden/state/logitはartifactに残さない。LM headは34固定行のdiagnostic readoutだけで、最終モデルlogitとは区別する。
- 実contextは`fixture-prompt-0` step 0/1と`fixture-prompt-1` step 0（context hashは既存traceと一致）を使用した。artifactは`benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-hybrid-diagnostic-v0.1/`に保存した。

主なaggregate結果（9 replay token、diagnostic readoutは3 final context）:

| stage | relative L2 | max abs | cosine |
| --- | ---: | ---: | ---: |
| QKV dequant+row-scale | 0.025647 | 0.936214 | 0.999687 |
| Z dequant+row-scale | 0.029672 | 0.586211 | 0.999571 |
| recurrent state after | 0.038704 | 0.623254 | 0.999283 |
| attention residual | 0.033045 | 0.070312 | 0.999518 |
| post norm | 0.178438 | 0.166057 | 0.984072 |
| MLP up projection | 0.171038 | 0.100061 | 0.985317 |
| layer0 output hidden | 0.042451 | 0.069627 | 0.999107 |
| diagnostic LM-head readout (34 rows) | 0.026799 | 0.022404 | 0.999674 |

- 最初の明確な相対L2 jumpはattention residual (0.033045) からpost norm (0.178438) だった。AQ4 runtime reportのpost epsilonは`1e-5`、source configのpost epsilonは`1e-6`であるため、この境界が最優先の確認候補である。ただし本日はpolicy/runtimeを変更せず、Phase 2の修正には進んでいない。
- `cargo check --package ullm-engine --bin ullm-aq4-layer0-family-isolation`、`pytest -q tests/test_aq4_layer0_family_isolation.py`（6 passed）、実context CPU比較を完了した。

## 次の行動

1. Phase 1の結果として、post-norm epsilonを揃えた読み取り専用control比較が必要かをレビューする。この結果自体からproduction runtimeの変更は行わない。
2. このstage traceと07/14のlayer0 traceを照合し、次フェーズへ進む判断が明示されるまでPhase 2の介入・修正には着手しない。
3. artifactのchecksumとsource/package identityを保持したまま、必要なら独立holdout contextへ同じCPU-only診断を拡張する。
