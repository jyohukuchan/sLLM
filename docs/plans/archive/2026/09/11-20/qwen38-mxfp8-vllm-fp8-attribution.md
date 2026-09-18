# Qwen3.8-27B MXFP8 / FP8 KLD差の原因調査

## 目的と固定条件

ユーザーの依頼は、sLLM MXFP8のBF16基準KLDがvLLM公式FP8より高い理由を、重みのBF16比誤差、量子化形式とアルゴリズム、実行経路に分けて調べること。既存の共通token-IDコーパス、全248,077有効語彙、FP64 KLD計算器を維持する。Qwen3.8-27B BF16原本、公式FP8、BF16原本から作ったMXFP8 GGUFの固定identityを混同しない。

1. 同じR9700 `gfx1201`で元のsLLM MXFP8を2,632位置取得し、既存のvLLM FP8と同じBF16基準で比較する。
2. 実際のtensor inventoryを照合し、共通量子化重みと片側だけの量子化重みを区別する。BF16原本から独立復元した重み誤差を行列群ごとに集計し、scaleと飽和の影響を検査する。
3. 同じR9700でvLLM BF16原本の短い対照を取得し、engine/KV差の大きさを測る。sLLM内ではGDN A/BだけBF16へ戻す診断用GGUFと、M=1活性値経路を比較する。
4. 実行kernel、活性値量子化、GDN、attention、logit保存精度をsourceと実測に照らして説明する。KLDは加法的な原因分解ではないため、未分離の効果を明記する。

正しさの証拠は実GPU成功、非有限値ゼロ、全語彙raw-logit、入力IDと位置の一致、対象GPU、数値oracleとする。失敗したrun、compile-only、CPU処理をGPU成功としない。実験用variantの既定動作とlegacy GGUF identityを保つ。commit/pushは依頼されていない。

結果と再現物は[対応する履歴](../../../../../history/2026/09/11-20/qwen38-mxfp8-vllm-fp8-attribution.md)へ記録する。完了時にこの計画を同日付のarchiveへ移す。

## 状態

2026-09-18に完了した。主因は重み・活性値のE8M0 scale選択による飽和で、診断opt-inの4条件比較で確認した。
既定の量子化方式は変更していない。
