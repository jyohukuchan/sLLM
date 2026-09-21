# R9700 FP8 M3並列化のMTP退行原因

ユーザー指示: MTPありでFP8並列化が大幅に遅くなる理由を調べる。

- 保存済みserial/M1-only/M1+M3 reportで受理率・replay数・setupとdecode内訳を比較する。
- 同じfrozen binary、8192/128、R9700、scheduler=1のserial/forkをrocprofで比較し、kernel時間とGPU空白を分ける。
- 観測に基づいて必要な場合だけ診断対照実験を加え、退行を説明できる範囲と未特定の部分を明示する。
- 数値正しさはN0と既存logits証拠を参照。プロファイル速度を通常速度と混同しない。
- 本番の採用範囲は変更せず、生成物は `.local-artifacts/phase87/r9700-m3-cause/` に保持する。
- 結果を履歴へ記録し、原因を実測以上に断定しない。commit/pushなし。

## 完了

同一fork binaryの1 queue制限で22.0052→35.6604 token/sへ回復し、複数queue graph実行の同期・dispatchが主因と確認。
trace増分の68.51%はGPU空白。FP8ペアspanの悪化は約17µsで、合流と後続直列処理の待ちが広がる。
MTP受理率・計算量・生成token列は同じ。classicへの変更だけでは回復しない。本番変更なし。

履歴: [計測と原因の範囲](../../../../../history/2026/09/21-30/phase87-r9700-m3-cause.md)
