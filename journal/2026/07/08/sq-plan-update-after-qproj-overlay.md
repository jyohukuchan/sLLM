# SQ plan update after q_proj overlay guard

## 前回の要点

- SQ候補評価はR9700/RDNA4に絞り、最初の候補を `sq-fp8-w8a16-r9700-v0` とした。
- T0 state freeze、T1 schema preservation、T2 artifact writer/runtime materialize smokeまでは進んでいた。
- T2 short prompt guardは、SQ artifactを既存package model load pathへ接続できていないため未完了だった。

## 今回の変更点

- SQ FP8 artifactからexact-name tensorをmaterializeし、既存AQ4 package tensorへfallbackするoverlay pathを計画に反映した。
- `sq-fp8-token-ids-logits-smoke` の一部guardとして、layer 3 `q_proj` だけをSQ FP8 overlayにした結果を追加した。
- さらにlayer 3のself-attention `q/k/v/o_proj` とMLP `gate/up/down_proj` の7 tensor overlay guardを追加した。
- 7 tensor guardでは、短い3ケースでAQ4 baselineとSQ overlayのtop1がすべて一致した。top8共通数は `7 / 8`, `5 / 8`, `4 / 8` だった。
- layer 7単体もtoken IDs `1,2,3,4` ではtop1一致だった。
- layers `3,7` の複数layer overlayでは、attention-only、MLP-only、attention+MLPのいずれもtop1が入れ替わった。ただしAQ4 top1はSQ top8内に残った。
- family別guardを自動化する `tools/run-sq-fp8-overlay-logits-guard.py` を追加した。
- family別では、`q`、`v`、`down` が単独でtop1を動かし、`k`、`o`、`gate`、`up` はtop1を保った。
- `k/o/gate/up` を同時にFP8化したsafe subsetは短い3 promptでtop1一致、`q/v/down` のrisk subsetはtop1不一致だった。
- safe subsetを `layers=3,7,11,15` へ広げると、短い3 prompt中 `2 / 3` はtop1一致したが、case_aでtop1が入れ替わった。
- case_aは `layers=3,7,11`、layer `15` 単体、`layers=3,7,15` ではtop1一致だったため、4 layer時の累積または組み合わせdriftとして扱う。
- `row_block` scaleを追加し、`q` はblock32、`down` はblock64でtop1一致に戻った。
- `v` はblock16/32/64/128でもtop1不一致だった。
- `v` fallback + `q/k/o/gate/up/down` row-block32 FP8の混合候補は、`layers=3,7` の短い3 promptでtop1一致した。
- これはT2 short prompt guardの進捗であり、同時にfull SQ候補へ進む前の品質境界だと整理した。

## 次の行動

1. `v` fallback + `q/k/o/gate/up/down` row-block32 FP8を `layers=3,7,11,15` と `4layer_case_a` へ広げる。
2. `v` はFP8 row-blockではなく、高精度fallbackまたは別形式を検討する。
3. strict top1一致だけを合格にするか、top-k overlapやtext-level guardを許容するかを決める。
4. full-targetに近いSQ FP8 artifactでshort prompt guardを通し、T3のbatch/cold/cached-prefix/decode gridへ移る。
