# 現在の進捗

- P3は未完了です。実GPUの一回限定試行はcapture失敗で終了し、本番サービスは正常復旧済みです。同じ認可では再実行しません。
- outer runnerとresident capture toolの固定契約が接続され、実capture toolと偽workerを使うsignal／timeoutの一本通し試験まで完了しました。関連115テストを通過しています。
- 次は独立統合監査で残り2件の解消を確認し、新しい候補・新しい認可を準備します。
