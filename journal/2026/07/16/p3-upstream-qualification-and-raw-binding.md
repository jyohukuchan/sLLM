# P3 upstream qualification and raw binding

## 前回の要点

- P2 の実測校正は relative-L2 の事前拒否条件に触れ、holdout 未実行の NO-GO で終了した。
- P3 の診断経路と本番昇格経路を分離し、P2 の判定を型付き証跡として結合する必要があった。

## 今回の変更点

- upstream P2 qualification を `rejected_no_go` / `qualified_go` の厳密な直和型として追加した。
- `qualified_go` は既存 P2 公式スキーマの plan、calibration metrics、freeze、preflight、consumed ledger、passed holdout receipt の identity・path・SHA-256・attempt-id・残回数を再検証した場合だけ成立する。
- 実際の NO-GO package は `rejected_no_go` のみを生成し、promotion eligible は常に false になる。
- raw producer は promotion に `qualified_go`、one-case diagnostic に `rejected_no_go` を必須化した。
- selector は qualification file を独立再検証し、diagnostic raw を選考測定へ混入させず `no_eligible_candidate` にする。
- selector 出力は hard-link create-new で公開し、並行実行時にも既存出力を上書きしない。
- synthetic success/rejection fixture と unknown/missing/bool-as-int/hash/cross-variant/receipt-swap の fail-closed tests を追加した。

検証結果: P3 関連 144 tests passed。GPU と holdout は実行していない。

## 次の行動

- capability schema の全要求項目を exact contract として固定する。
- verified selection artifact だけで本番 direct route を許可する activation contract を追加する。
