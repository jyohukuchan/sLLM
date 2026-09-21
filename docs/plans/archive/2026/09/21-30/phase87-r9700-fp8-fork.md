# R9700 FP8単独並列化の検証

ユーザー指示: R9700にもFP8単独で適用して検証し、問題なければ採用する。

## 範囲と受入条件

- R9700のNVFP4は直列、V620の採用済み経路は保持する。FP8 GDN QKV/ZのM1/M3だけを候補とする。
- ROCm 7.14、gfx1201、従来scheduler=1を固定する。V620用classic回避を混ぜない。
- 実providerと一致する単体serial graph/fork graphで、独立数値oracle、finite、N0、cleanupと同一process AB/BAを確認する。
- 採否は既存Phase87原則の単体全round改善・通常TPOT換算1%以上・他条件で退行しないことに従い、必要ならMを限定する。
- 通常8192/128、MTPなし/あり、1 warmup+3 measuredで出力・速度を比較する。改善候補はABBAで確認する。
- 同じ固定128位置でBF16比KLDを取得する。既存BF16 referenceを再利用し、実graphのlogitsで比較する。
- 採用時はsource/CI hash/関連check/履歴を更新。不採用時は候補変更を戻し結果を残す。commit/pushはしない。

生成物: `.local-artifacts/phase87/r9700-fp8-fork/`。
baselineは直前のR9700最終binary `dc11e755...`。既存未commit変更は保持する。

## 実測を受けた同じ作業単位の再計画

- M1は1pair/48pairsだけのsynthetic graphでは退行するが、実モデルFP8単独は21.4768→21.8091 tok/s、全run N0一致。
  graph構成によるschedulerの差があり、合成graphをそのまま本番の寄与へ換算できない。
- M3は実モデルMTPありで35.6229→22.0052 tok/s（−38.23%）、N0は一致。M3は候補から外しM1だけに絞った。
- 単体の負値は保持する。追加の同一process比較は実際のwholegraphで行い、capture時のFP8 dependency更新だけを
  Aではno-op、Bでは通常処理にする。warmup B後、同一入力のfresh requestをA/B/B/A×2で測る。
  元のbenchmarkのrequest時間を使い、clone・追加instantiate・replay eventは入れない。
  この測定方法の追加はmainの判断で、R9700 FP8M1の採否に限定し、採否決定で終了する。
- 実captureはFP8 M1 48組のforkだけが変わり、NVFP4 56組と他の辺は不変と確認した。
- 合成bundleの初版はM混在と余分なtimestamp nodeを含み、GPU実行前に修正した。
  実行版のJSON loggerは配列終端の誤りがあり、rawを保存したまま値を変更せず正規化した。GPU処理/oracle/cleanupは終了したが、元runnerのFAILは保持する。

## 完了

M3は通常MTPあり−38.23%で不採用。M1も実wholegraph AB/BAの速度差が
+1.524/−0.085/−1.702/+1.694%、中央値+0.734%で安定した改善を確認できず既定採用を見送った。
全run N0、128位置のlogits bit一致、BF16比mean KLD 0.023446・変更差0。
本番sourceとCI hashは試行前へ復元し、候補rawは保持。commit/pushなし。

履歴: [測定・採否](../../../../../history/2026/09/21-30/phase87-r9700-fp8-fork.md)
