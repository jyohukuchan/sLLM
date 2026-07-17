# Inference APIと量子化処理の分離

## 開始時点

- AQ4/SQ8はOpenWebUIから利用できるが、worker protocol/runtimeに`Sq8*`依存が残る。
- gatewayとOpenWebUIへモデル契約が環境変数として重複している。
- AQ4 workerは要求ごとにCLI子processを起動する互換経路である。

## 計画

`docs/plans/inference-api-quantization-separation-plan-v0.1.md`を追加し、commit
`fa69c10`でP0を完了した。

最終条件は、共通`InferenceSession`、1モデル1`ullm.served_model.v1`、
SQ8/AQ4 session adapter、AQ4 resident化、atomic activation/rollback、
量子化分岐のAPI層からの除去である。

## 進捗

- P1: 量子化非依存の推論API型を`86c1fc3`で実装。
- P2: 共通`InferenceSession` driverとSQ8 adapterを`1f08863`で実装。
- P3: manifest仕様/loader/preflightを`c63f923`/`ed54fe2`で実装。
- P4: OpenWebUI同期とgateway manifest modeを`149c7de`/`22f990e`で実装。
- gatewayはRuff/mypyと202 tests、Rustは373 lib testsと両worker bin testsを通過。
- worker自身のmanifest受理、実product manifest生成、atomic activation/rollbackを並行実装中。
- 稼働サービスはlegacy AQ4 profileのまま維持し、未commit実装は配備していない。

## 稼働状態の追記

- gatewayは`172.20.0.1:8000`のbridge限定listenであり、ホストからのhealth確認はfirewallでtimeoutする。
- OpenWebUI container内からの`/healthz`は`{"status":"ok"}`、containerもhealthy。
- 1339/1360 tokenの長いpromptではAQ4 compatibility childが30秒間progressを返せず、gatewayのprogress deadlineでworkerを再起動した。AQ4 resident/streaming progress化で解消すべき実機再現条件として記録する。

## manifestとAQ4常駐化

- gatewayのmanifest modeを`22f990e`、共通protocol aliasを`a8f95f8`で保存。
- 実productからのstreaming hashでmanifestを生成するツールを`e7ea157`で保存。SQ8実productはvalidator通過、AQ4はpromotion receipt不在のためfail-closed。
- manifestのatomic replace、失敗時復元、DB等用rollback hookを`503fc55`で保存。
- Rust共通manifest loaderと両worker CLIを`bdf7954`、systemd/deploy連携を`b6c9244`で保存。AQ4 manifest modeはresident backend完成までfail-closed。
- `PackageAq4ResidentMatvec`とshared buffer/projectionを`aq4_package_runtime.rs`へ移し、`4907921`で保存。376 lib testsと36 bin testsを通過。
- worker profileを起動時に一度だけ確定し、protocol/reader/writer/inferenceが同一snapshotを使う構成を`928cb99`で保存。manifest値の環境変数書き戻しを削除。
- Qwen3.5 AQ4のself/linear attention resident layerを`qwen35_aq4_layer_runtime.rs`へ移し、`7ddba22`で保存。382 lib testsと30 main bin tests、両worker testsを通過。
- 次はembedding/final norm/lm_head抽出とlayer request-state reset APIを進める。
