# P3 profile diagnostic actual v6

## 前回の要点

- Quiet-window v11 は commit `5a79b3fe`、27 連続 clean、172.270818469 秒、reset 0 で GO になった。
- Operator manifest v6 は authority commit `db6ee0fa005d2b25c6b86996b6c2c8bcbe3a33af`、manifest file SHA-256 `4c3b7ba685ecc465e77169d9ae186c93bdc548357ef883c396bbd06a18d685ce`、exact argv 10、shell false、maximum invocation 1 を固定した。

## 今回の変更点

### 単一 actual 実行

- Manifest file/semantic self-hash、authority HEAD/tree、六 sealed roots の `SHA256SUMS`、5 selected tests、fresh 9、service/worker/lock、AMD/KFD owner、formal health、targeted external process 0を独立に確認した。
- 同一 PTY で sudo credential を確立した直後に dynamic preflight 13/13を再取得し、すべて PASS した。
- Manifest の cwd と exact argv 10を `shell=false` で1回だけ実行した。invocation count 1/1、再試行なし、return code 1だった。
- Operator stream filesystem boundary は start `1784123045886520126` ns、end `1784123098029562807` ns、elapsed `52143042681` nsだった。
- Operator stdout は254 bytes、SHA-256 `309420c9b009141b58ea87776d805cbbeb5f88520e893d4fd81f9698d00d5829`、stderr は0 bytes、SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` だった。
- Execution wrapper SHA-256 は `744a059baa0d0e957830b19e33a830fe298eb10b3566ad9cf1efab825aa1bed7` だった。実行中の介入と再試行は行っていない。

### READY candidate、process v2、failure boundary

- Validator は return code 0、stdout 113 bytes、stderr 0 bytesで PASS した。Capture tool、rocprof、profile launcher は各1回起動し、runner subprocess は return code 1になった。
- `capture-failure.json` は schema `ullm.aq4_p3_diagnostic_rocprof_failure.v2`、return code 1、timeout false、outer signal false、process-group cleanup complete、children remaining 0を固定した。
- READY candidate marker は valid、exact 1件だった。Secret-safe process audit は schema `ullm.aq4_p2_resident_driver.v2`、`event=ready`、`model_loads=1` の candidate を保持した。
- Audit validation は `ready_candidate_field_set_differs` で失敗した。`served_model_binding` が absent だったため、driver ready と model load 1は証明されず、GPU command/model load の実行状態は `unknown` とした。
- Runner stderr は2528 bytes、SHA-256 `91e07fe5be2bf3a8fb46e80ce73393d94b46ea1a0d73874951aeba1bcb476f94` で、`resident driver did not prove one model load` と READY candidate audit markerを保持した。
- ROCTx required range count 12に対し retained count 0、capture artifact と runner output rootも未生成だった。Measurement/promotion eligibleではない。

### Cleanup、integrity、restore、post-health

- Capture process-group cleanup、launcher cleanup、trusted lock substrate cleanup は PASS した。Trusted directory inode `793115`、lock inode `793116`、cleanup時 holder 0で、temporary lock/directoryは削除された。
- Package full content hashは exact 1回だった。7,700,872,459 bytes、1,045 files、SHA-256 `a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1` で PASS した。Tree identity `e105d6f242fc3792fb0e3b5c3dd7ea98449e1b451af0aec1f73c40571438d9a3` は final metadata recheckでも一致した。
- Restore は attempted/passed、6 polls、duration `15007853578` nsで absolute deadline 内に完了した。
- 独立 post-health は PASS した。Service main PID `193179` は active/running、`NRestarts=0`、worker PID `193294`、AMD/KFD owners `[193294]`、production lock holder `[193179]` だった。
- Formal container health identityは `51b37eda8771d36e154f6cd52a22be0bf2d33f2eafa0d90958afe57d14a7b82f`、targeted external/residual processesは0、secret material recordedはfalseだった。

### Immutable evidence

- Operator result SHA-256: `a0ef83f93af4b6f3a3af5ba209387df4d723632615ffea1558aed570e5d8876b`
- Actual audit SHA-256: `a111e922c663f98ecf630bb94b3a22e7598c710466faa741d2d310152212a611`
- Maintenance evidence SHA-256: `ae5a48af65245b65b6d0fc16c7b50c15c84b52aff296ff946c64c0cd1b2b3886`
- Launcher evidence SHA-256: `e33f4dc9b8a41a76ac9000f750f8b8236260d1eff0c479ca74e2b3693557036e`
- Runner target manifest SHA-256: `503e13e07fc5a1dea762652e1e866f16ff3a75df526f46be0237ee59c403c477`
- Capture failure SHA-256: `0a1b7896f4e86d0ef1ad029ba1a5b70bf4296a0cfc31adab40862b75137dd76c`
- Maintenance、launcher、capture、operator result、actual audit の全 `SHA256SUMS` entryが PASS した。各 root は mode `0555`、全 file は mode `0444`、nlink 1で封印した。

## 次の行動

- Operator v6 single-use authorization は失敗として消費済みであり、actual を再実行しない。
- 次版では READY event の exact field setに `served_model_binding` を含め、secret-safe candidate auditが driver ready/model load 1を証明できる契約に修正してから、新しい quiet-window と authorizationを作る。
