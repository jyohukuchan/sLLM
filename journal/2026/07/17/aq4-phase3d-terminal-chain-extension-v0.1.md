# AQ4 Phase 3d terminal chain extension v0.1

## 前回の要点

- Phase 2cのCPU-only chainはlayer 0--11までを測定し、相対L2は0.08--0.13付近で非単調だった。layer 12--31、final norm、LM headは未測定だった。
- 0:11の実測はwall `157.13 s`、最大RSS `332008 KiB`、このプロセスのswap operation `0`だった。現行chainは埋め込み残差を開始入力にするため、start > 0 の分割実行は前層hiddenを欠き数学的に正しくない。

## 今回の変更点

- `ullm-aq4-layer0-family-isolation`のchain modeに`--chain-include-final-norm-lm-head`を追加した。このflagは完全な`0:31` chainでのみ許可し、final RMSNormをfull hiddenで、LM headを固定34 token rowで逐次出力する。
- Rust側は`model.language_model.norm.weight`をAQ4 runtimeと同じdirect weight handlingで適用し、LM headはpassthrough row readerで34行だけを読む。全語彙logitsと全層hidden/stateを保持しない。
- Python比較器はQwen3.5 source final RMSNormを明示的に`normalized * (1 + weight)`で計算し、final normのfull-hidden metricとLM-head固定row sample metricを成長曲線に追加した。
- source checkpointの`Qwen3_5RMSNorm`実装、package final normのpayload、AQ4 runtimeのweight handlingを読み取り専用で照合した。sourceは加算weight、AQ4 runtimeはfinal normを加算対象外として扱う。これは原因候補だが、この段階ではfixせず全32層の測定で確認する。
- `cargo check --package ullm-engine --bin ullm-aq4-layer0-family-isolation`、`python3 -m py_compile tools/compare-aq4-multilayer-accumulation.py`、`pytest -q tests/test_aq4_multilayer_accumulation.py tests/test_aq4_layer0_family_isolation.py`（16 passed）を完了した。
- 初回のCPU-only 0:31実行はdecoder計算後、LM headがpassthroughでなくAQ4 tensorであることを検出してterminal frame前に停止した。wall `405.62 s`、最大RSS `331188 KiB`、process swap `0`であり、fidelity結果としては無効として保存した。
- LM headを全語彙materializeせずAQ4の固定34行だけseek/dequantizeするreaderへ置換した。packed idx4 nibble順、u8 scale table、row-scale overrideを単体testで検証し、terminal chain/reportをschema v3へ更新した。この変更は診断の入力read/dequantize範囲のみで、production fixではない。
- 修正後の2回目はAQ4 binaryがfinal normとLM headの両terminal frameを正しく出力したが、Python比較器がstage一括順を期待していたため、per-timestepの`final_norm -> lm_head` stream順を拒否して無効終了した。wall `412.37 s`、最大RSS `334748 KiB`、process swap `0`だった。
- 比較器をproducer順へ合わせ、sourceのfixed-row logitsもcurrent timestepだけで計算するよう修正した。framed streamのterminal順を固定するtestを追加し、`pytest -q tests/test_aq4_multilayer_accumulation.py tests/test_aq4_layer0_family_isolation.py`は`17 passed`だった。
- 3回目のCPU-only 0:31+final norm+LM headはvalidに完走した。wall `408.41 s`、最大RSS `330744 KiB`、process swap `0`で、見積もりの約7分・512 MiB未満に収まった。32 decoder layer、self-attention `3,7,11,15,19,23,27,31`、full-hidden final norm、固定34 AQ4 LM-head rowをすべて確認した。
- decoder curveは非単調で、layer 31はrelative L2 `0.1278813307`（decoder最大はlayer 29の`0.1708747154`）だった。final RMSNormで`0.5010330688`へ`+0.3731517381` / `3.917953x`急増し、LM head固定row sampleは`0.5860500940`だった。LM head値はfull vocabulary L2ではない。
- source/packageのfinal norm BF16 payload bitは同一SHA-256 `44f7283137ae75c262c152f7e529b70c708ea13afc1bfaa565c8ea74b61ecf88`だった。一方source Qwen3.5は`normalized * (1 + raw_weight)`、AQ4 runtime/chainはfinal normをadditive suffix対象外として`normalized * raw_weight`を適用する。このためdominant boundaryはlinear/self/MLPではなくfinal RMSNorm weight interpretationに偏ると判断した。Phase 4 fixは実装していない。

## 次の行動

- Phase 3dの測定・境界特定は完了。`attempt-3/phase3d-analysis.md`、growth curve、weight semantics evidenceをreview対象にする。
- 次の判断が必要なら、今回のevidenceを入力にPhase 4の別タスクとして扱う。ここではfix、GPU、service、P3 harnessには進まない。
