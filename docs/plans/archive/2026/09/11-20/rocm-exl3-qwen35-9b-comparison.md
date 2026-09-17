# Qwen3.5-9BでのEXL3/llama.cpp比較

2026-09-17のユーザー依頼（5B以上で比較）により、同じ系列の固定Qwen3.5-9B BF16から比較する。

## 受入範囲

- source revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` のローカル重みをlockと照合。
- 前回検証済みrocm_exl3 patch/binaryとllama.cpp `bc52a12b38941b0a690ade65fbc5749715224e30`を使用。
- EXL3本体4bpw/head6/vision16、校正16×256の動作比較用artifactと、標準Q4_K_M/imatrixなしを生成する。
- R9700 gfx1201、単一要求、MTPなし、KV FP16、context4096。pp128/512/2048、decode128入力・64指定。
- 同じtoken ID配列とsampling profileでwarmup1＋measured3、中央値/範囲を記録。前回と同じphase境界差は明記。
- 両engineの生成、EXL3 finite logits、正常終了を確認し、数値・サイズ・制約を比較表へ記録。
- モデル・raw log・binaryはcheckout外。production実装への移植、commit/pushは対象外。

## 計測条件の補足

2,048入力の初回比較で、llama ubatch512（4分割）とEXL3 max_chunk_size2048（一括）の差を確認。
同じ入力/反復をllama ubatch2048でも測り、主表はchunk上限を合わせた比較、ubatch512は別欄に残す。
これは測定条件の帰属確認であり、新しい品質gateや必達倍率は設けない。

## 状態

完了。15 sourceファイルをlock照合し、両形式の生成、GDN6ケース、英日smoke、正常終了を確認。
共通16promptでEXL3とllama ubatch512/2048を各16request測定した。主表はchunk2048で揃え、
pp128/512/2048/decodeはEXL3 398.3/2938.5/4917.6/75.10、llama 1609.2/2986.5/3441.2/78.48 tok/s。
校正・timer境界・format精度配分・ばらつきの限界を記録。既存source再使用で新規の外部source importなし。

履歴: [9B比較結果](../../../../../history/2026/09/11-20/rocm-exl3-qwen35-9b-comparison.md)
