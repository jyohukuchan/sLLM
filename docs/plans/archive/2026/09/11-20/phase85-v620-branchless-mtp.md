# V620でのbranchless MXFP MTP検証

> 状態: 検証完了（ローカル分離build、本番未採用）
> 2026-09-13ユーザー指示: 「V620で診断用改善経路を使ってMTPが改善するか検証して」

診断で確認したbranchless復号だけを、現行Columns2 M1のgfx1030経路へ組み込んだ分離buildを作る。
本番source、selector C（MXFP6のOは旧ID20）、量子化recipe、KV、samplingは変更しない。

- exact V620 `GPU-76a08c022586fed6`、Qwen3.8-27B NVFP4本体、MXFP8 E4 KV。
- 8192入力／128出力、prefill chunk2048、MTP幅2、固定sampling seed123。
- 現行CのBF16／MXFP8／MXFP6を新規測定し、MX形式を分離candidateと比較する。
- 各構成1 warmup＋3 measured、通常auto clockでdecode tok/sとMAD、draft wall、TTFT／E2Eを記録する。
- 同形式の生成token列・採用数、HIP-only、fallbackなし、cleanup、演算子oracleを確認する。
- 候補の実MTP profileでID99/100への到達を確認し、profile時間は代表性能値へ使わない。
- 改善率やBF16超えを完了条件にせず、実測差と残差を報告する。本番採用・commit・pushは含めない。

source snapshot／build／raw outputは `.local-artifacts/phase85-branchless-mtp/` に保存する。
Qwenローカルservice停止を確認済み。R9700のserviceは測定対象外。

[メイン計画](../../../../main-plan.md) /
[起点の診断履歴](../../../../../history/2026/09/11-20/phase85-mxfp-m1-bottleneck-diagnosis.md)

## 結果

MXFP8 decodeは21.4158→22.6685 tok/s（+5.85%）、MXFP6は23.1281→23.4621 tok/s（+1.44%）。
同形式の生成token列・MTP countを維持した。BF16基準25.4097 tok/sには届かない。
draft wallは約29.0%／8.8%短縮。長いprefillを含むE2Eの観測差は約0.4〜0.5%にとどまる。
12演算子比較・限定8境界行・両形式profileを確認し、MXFP6 Oの旧ID20を維持した。

[検証履歴](../../../../../history/2026/09/11-20/phase85-v620-branchless-mtp.md)
