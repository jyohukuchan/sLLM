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

## 次の行動

- 修正済みbinaryをbuildし、失敗attemptを上書きせず別ディレクトリでCPU-onlyの0:31+final norm+LM headを実行する。wall time/RSS/swapとterminal contractを記録する。
- 取得した成長曲線からboundaryとtensor familyを記録する。Phase 4のfix、GPU、service、P3 harnessには進まない。
