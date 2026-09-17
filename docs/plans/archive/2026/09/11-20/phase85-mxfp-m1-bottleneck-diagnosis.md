# MXFP M=1ボトルネックの切り分け

> 状態: 診断完了（2026-09-13、本番への最適化採用は別作業）
> 2026-09-13ユーザー指示: 「細かい検証を進めて何がボトルネックになっているのか特定して」

現在のローカルColumns2変更を保持し、BF16／MXFP8／MXFP6の実MTP M=1で帯域・命令処理・待ち時間の
寄与を切り分ける。最適化の追加や公開を先に行う作業ではない。主対象はgate/up K5120/N17408、
down K17408/N5120、退行・時間遷移を観測したo K6144/N5120とする。exact gfx1030/gfx1201で確認する。

## 方法と判断

1. 計時区間、実kernelのISA／VGPR・LDS・scratch、取得可能なRDNA counterを確認する。
2. 既存production kernelを同じbuffer条件で実行し、GPU event、kernel trace、hardware counterを区別して記録する。
   raw counterの定義・集約・単位を確認し、logical bytes/timeをDRAM帯域と呼ばない。
3. BF16／MX、旧M1／Columns2を比較し、独立したread-bandwidth基準、warm/steady、反復間隔・cache条件を
   必要な代表行で変えて因果を絞る。clock/電力/温度も記録し、以前の前半/後半の遷移を再確認する。
4. profilerで測った時間を通常推論の速度へ置換しない。演算子・draft・MTP全体の寄与を分け、
   特定できた律速と未確定部分、次に改善すべき箇所を根拠付きでまとめる。

HIP-only、exact UUID、非zero dispatch、数値oracle、cleanupを確認する。counter不在・取得失敗は測定成功にしない。
期間や倍率による新しいgateは設けず、実測で解ける代表行へ絞る。診断probe／raw traceは
`.local-artifacts/phase85-m1-bottleneck/`に置く。productionコード、モデル、既存APIの動作は変更しない。
R9700既存serviceは測定時だけ停止し、元のunit/binary/configとhealthへ復帰する。
clock状態を対照のためstock設定内で一時変更する場合は、元の設定を記録して復帰し、overclockは行わない。

## 診断結果

- GPU event／kernel trace／counter／stock clock対照を分離して確認した。
- V620ではMXFP8の復号制御をbranchless化すると実MTP gate重みのM1が721→365µs、
  MXFP6が441→378µs。読出し量を維持してSALU命令が大幅に減り、復号の制御処理が主要因と特定した。
- R9700はbranchless化だけでは小幅改善。scaleのlane0読出し＋wave broadcastを直接読出しへ置換すると
  追加で14〜16%短縮し、この共有・待ち経路の負担を確認した。残る命令発行／load／変換の内訳は未確定。
- exact両GPUでcodec全code、matmulの数値oracle、同形式の全出力一致を確認した。
  R9700の主要PMCはzero／欠落のため無効とし、帯域やVALU飽和の証拠には使用していない。
- 診断用コードはignored artifactに保持。本番selectorは変更せず、MTP全体の追加速度改善は未検証。

[診断履歴](../../../../../history/2026/09/11-20/phase85-mxfp-m1-bottleneck-diagnosis.md) /
[メイン計画](../../../../main-plan.md)
