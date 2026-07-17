# OpenWebUI explicit model migration

- Manifest marker導入前のuLLMモデル行を安全に停止するため、明示IDとprevious served-model manifestの入力契約を追加した。
- 既存の`meta.ullm.managed=true`かつ同一base URLの行は従来どおり自動停止する。未marked行は明示されたIDだけを変更し、名前やID prefixでは推測しない。
- 明示入力は内部でmodel IDへ正規化し、重複、current modelとの衝突、legacy modeとの混在を拒否する。
- SQLiteテストでunmarked SQ8からAQ4への初回移行、AQ4からSQ8へのrollback、同一providerの無関係行と他provider行の完全不変を確認した。
- 実DBと稼働中コンテナには変更を加えていない。
