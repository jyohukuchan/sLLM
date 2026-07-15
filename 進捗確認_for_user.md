# 現在の進捗

- P3は未完了です。実GPUの一回限定試行はcapture失敗で終了し、本番サービスは正常復旧済みです。同じ認可では再実行しません。
- exact 6件の正式認可系譜manifestをbuilder、receipt、served manifest、Gate、runner、SHA256SUMSへ束縛する実装が完了し、関連133テストを通過しました。旧形式からのactual認可は拒否します。
- 次は実装commitを固定し、そのcommit向けmanifestとfresh unauthorized runtimeをcreate-newで再生成します。独立監査GOまではactual認可とGPU/service実行を禁止します。
