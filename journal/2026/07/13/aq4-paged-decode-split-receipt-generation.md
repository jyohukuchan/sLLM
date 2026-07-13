# AQ4 paged-decode split receipt generation

## 前回の要点

AQ4 profileは、split paged-decode required guardを追加済みだったが、receipt参照は旧世代の `promotion.json` に固定されていた。

## 今回の変更点

- profileのreceipt参照を、同じproduct root内の `promotion-paged-decode-split-v1.json` へ分離した。
- 親エージェントが後続生成するevidenceの安定名を `resident-promotion-evidence-paged-decode-split-v1.json` とした。
- profile回帰テストで旧receiptとの別パス、experimental split env不在、split required guardの重複なしを固定した。

## 次の行動

親エージェントが新evidenceを上記名で保存し、receipt writerを `promotion-paged-decode-split-v1.json` 出力へ実行する。
