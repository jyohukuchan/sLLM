# P2 layer0 QKV one-at-a-time hybrid fidelity isolation

## 前回の要点

Z one-at-a-time診断では、同一3-step入力とゼロ初期化stateでZだけをsource BF16投影へ置換したとき、layer outputのaggregate relative L2は`0.0006983384`だった。QKVは既存family isolationで大きなmax absを持っていたが、同じlayer境界での下流寄与は未測定だった。

## 今回の変更点

- Z診断と同じ`package_linear_attn_mlp_block_sequence_run_with_diagnostic_inputs`だけに、既定値が`None`のQKV projection overrideを追加した。通常wrapperとworkerからは到達不能である。
- overrideはproduction AQ4 QKV matvec完了後、production depthwise convへ渡す直前だけに適用した。Z/A/B、depthwise conv、Q/K splitと正規化、gate/beta、recurrent、attention norm/out、post norm、MLP、残差加算は同じproduction runtime経路を使う。
- `package-linear-attn-qkv-hybrid-diagnostic`をCPU専用CLIとして追加した。同一package、同一固定3行input、同一順序、baseline/hybridごとのゼロstateを使う。
- `build-aq4-layer0-qkv-source-sidecar.py`はsource BF16 `[8192,4096]`を256 output rowずつf32へ変換し、full f32 weightを保持せずsource QKV sidecarを生成する。
- QKV、recurrent output、attention-block、layer outputをbaseline/hybrid sidecarへ保存した。recurrent stateはproduction Q/K/V/gate/beta出力を同じゼロstateからhost f32でstep replayし、各stepのstate SHA-256をmetadataへ保存した。
- artifactは`benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-hybrid-fidelity-v0.1/`へ固定した。thresholdsは`null`、policyは`policy_not_evaluated`、promotionは`false`、holdoutは`not_run`である。

## 結果

- AQ4 baseline QKV対source BF16: relative L2 `0.0256654451`、cosine `0.9996858735`、max abs `0.8943977356`。
- QKV source override後: source QKVとのrelative L2 `0`、cosine `1`、非有限値なし。
- recurrent outputへの差: relative L2 `0.0132480712`、cosine `0.9999266820`、max abs `0.0065923780`。
- attention-blockへの差: relative L2 `0.0007618562`、cosine `0.9999997117`、max abs `0.0254900455`。
- layer outputへの差: relative L2 `0.0007890095`、cosine `0.9999996913`、max abs `0.0260486603`。
- QKVのlayer output relative L2はZの`0.0006983384`より約13.0%大きい。現時点のprecision候補順位ではQKVをZより先に調べる根拠になるが、閾値判定・採用判定・holdoutは行っていない。

## 検証

- `CARGO_BUILD_JOBS=1 cargo check -p ullm-engine --bin ullm-engine`: 成功。
- `package-linear-attn-qkv-hybrid-diagnostic ... 0 1048576 3`: CPUで成功。
- 既定`package-linear-attn-mlp-block-smoke`を2回実行し、両方のstdout SHA-256が`9ac224cc444569bb9e5c4c493eacf4007c06c862c03466da31a058a123e4ad9b`でbit-exactだった。
- GPU、service、P3、Gate、holdoutは実行していない。

## 次の行動

親agentが通常commitをmainへ統合する。precision候補はQKVを第一候補、Zを次候補として保持するが、P2の記述的診断を昇格判断へ流用しない。A/Bを同じone-at-a-time境界で測る場合も、別作業として同じ非昇格契約を維持する。
