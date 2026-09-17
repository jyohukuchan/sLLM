# R9700のMXFP MTP改善実験

> 状態: 実験完了（分離build、本番未採用）
> 2026-09-13ユーザー指示: 「同様の統計情報は取得できないかもしれないが、R9700についても改善を試みて。」

現在のColumns2 Cを基準に、gfx1201 MXFP8 ID99のbranchless scale復号と直接scale読出し、
MXFP6 ID100のbranchless復号を分離buildで試す。前の診断で数値一致と効果を確認した経路を通常MTPへ接続する。
selector C、native E4M3変換、packed reader、FP32加算／reduction、本体NVFP4、KV、samplingを維持する。
本番checkoutへの採用・commit・pushは今回の範囲に含めない。

- exact R9700 gfx1201、UUID `GPU-a8e9ddefa2d60f55`。
- Qwen3.8-27B-NVFP4、MXFP8 E4 KV、MTP幅2、8192入力／128出力、chunk2048、state8320、固定seed123。
- BF16／現行MXFP8／現行MXFP6と候補を今回新規測定し、各1 warmup＋3 measuredでdecode tok/s、
  median/MAD、draft wall、prefill／TTFT／E2E、生成token列と採用数を比較する。
- 実MTP6形状×2形式の演算子前後oracle・digest一致と、selector／端数の限定境界を確認する。
- 候補profileは形式ごとに0+1、HIP dispatch／形状到達のみを確認し、性能代表値へ混ぜない。
- 主要PMCは前の診断でzero／欠落だったため今回の方法に含めず、GPU event／kernel trace／ISAを用いる。
- 全形状の必達倍率やBF16超えは設定せず、改善しない結果も記録する。結果が揃った後に無関係な全面検証を追加しない。

既存R9700 serviceへのactive接続がないことを確認して一時停止し、終了・失敗時とも元のunit／binary／run.sh hash、
healthz／readyzへ復帰する。raw source snapshot／build／traceは `.local-artifacts/phase85-r9700-mtp/` に保持する。

[メイン計画](../../../../main-plan.md) /
[診断履歴](../../../../../history/2026/09/11-20/phase85-mxfp-m1-bottleneck-diagnosis.md)

## 結果

MXFP8 decodeは32.9703→33.3046 tok/s（+1.01%）、MXFP6は30.6944→30.8218 tok/s（+0.41%）。
draft wallは約4.82%／4.31%短縮し、両形式とも全3反復のtoken列・MTP countを維持した。
BF16基準34.8683 tok/sには届かない。1 fixture／1+3での小幅な観測改善であり、広い条件への保証はしない。
実形状12比較・限定8境界比較と両形式profileをPASSし、元のR9700 service／auto clockへ復帰した。

[実験履歴](../../../../../history/2026/09/11-20/phase85-r9700-mtp-improvement.md)
