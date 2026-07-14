# AQ4 P2 fidelity holdout protocol

- 48-case production representative profileを8 strata（prompt 4種×baseline mode 2種）へ固定し、SHA256(case hash)順で各3 calibration/3 holdoutに分割する実装を追加した。
- fixture/indexのregular-file、hash、case identity、full-context step=0、attempt2除外、重複/欠落/非有限の拒否を実装した。
- policy JSONへ指標、二項率の95%片側Wilson下限、連続値の固定abs/relative margin、sample minimum=24、absolute floor/ceiling、relative-L2>1拒否、raw hidden max-abs診断専用、attempt2観測値の利用禁止、BF16 top1保持率品質指標を凍結した。
- freezeはcalibration 24件だけからactive-vs-BF16 envelopeを導出し、holdoutを未開始・一回だけ許可するreceiptを作る。モデル測定、GPU/service/raw evidence変更は行っていない。
- 実行: protocol unittest 3/3 OK。CPU処理はO(48)の目安1秒未満（未測定推定）、モデルCPU/GPU時間は未束縛。attempt3 preflight通常/lockedは別途rc0、strict本番は未実測No-Go。
