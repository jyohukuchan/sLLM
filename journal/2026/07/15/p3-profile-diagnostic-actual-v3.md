# P3 profile diagnostic actual v3

## 前回の要点

- operator v2 は validator が rocprof 環境を継承し、正常終了時にも stderr が非空になったため、runner 開始前に安全停止した。
- v3 authority commit `114034d733ed3e93e01c36d25549032d7a889e59` は validator と resident runner の計測境界を分離し、A4 restore を固定 absolute deadline 120秒、package full hash exact 1回へ変更した。
- quiet-window v7 commit `189becb0290d1e473b359f362c93ca043b49f7f3` は primary 27 samples / `192.447962750s` と confirmation 28 samples / `200.537321143s` で GO だった。

## 今回の変更点

### 単一実行契約

- operator manifest raw SHA-256: `073f61e9396245d2c11b49e0846e6868b21185d27c92253447c76fc3e3433c1b`
- canonical command SHA-256: `8249eddef3d10029e47f2b47986aa77202c32a080e2f2b05161f3849f37d3271`
- manifest の cwd と exact argv 10を `shell=false`、maximum invocation 1で実行した。
- 開始直前に authority HEAD、manifest/SHA256SUMS、profile-ready/SHA256SUMS、全 input hash と Git blob、fresh 5出力の不存在、quiet-window GO、current service epoch、worker、lock、AMD/KFD owner、formal health、targeted external process 0を同一PTYで再検証した。
- canonical start: `1784106963411740686` ns (`2026-07-15T18:16:03.411740686+09:00`)
- canonical end: `1784107023257623763` ns (`2026-07-15T18:17:03.257623763+09:00`)
- elapsed: `59845883077` ns。return code `1`。invocation count 1。再試行なし。operator stdoutは254 bytes、stderrは0 bytesだった。

### validator、FD-map、runner、driver

- trusted validator は profile 対象外で1回実行され、exit 0、report `prepared_not_executed`、stderr 0 bytesでPASSした。v2のvalidator境界問題は解消した。
- stopped live preflight は trusted lock inode `779219`、old workerなし、AMD/KFD owner空でPASSした。fresh target manifest raw SHA-256は `05bbf54602cf07730803be30eed6fe575a73aa23bc7cf9e3fb88e3fc0934d22f`、semantic SHA-256は `c3b8ecb876dd5a26b28456ef33e0a5f2fa0e9b9d4490ce62490b9775fa40061a`、target argv SHA-256は `2faa5b19a9491634737b2e6b8731e07d648a6982bf5e5bc91cff6bfe394900a1` だった。
- target manifest は code/control/lock を `pinned_fd`、package dataを `trusted_pre_post_guarded` として束縛した。runner はそのFD-mapを受理してdriverを PID `3212777` / process group `3212777` でspawnした。
- driver は ready 応答前に exit 1し、stderrへ `ullm-aq4-p2-resident-driver: served model rejected: served-model manifest traverses a symlink` を94 bytes出力した。runner failureは stage `ready`、kind `eof` だった。
- protocol は `ready_received=false`、warmup 0、measured 0、case begin/end 0、stdout event 0だった。runner cleanupはreap、stdin close、process group消滅を確認してPASSし、signal送信は不要だった。
- 原因はFD-mapとserved-model path安全検査の契約不整合だと考える。immutable trusted runner SHA-256 `0b3e55f250894403bf4ef7300a2e67422055dc7f43e82efe0fc75de7ecda0c1b` は `served_manifest` を `/proc/self/fd/<descriptor>` に置換する。一方、driverのserved-model loaderは全path componentのsymlinkを拒否し、`/proc/self` はsymlinkである。元の `/etc/ullm/served-models/active.json` とその親component自体にはsymlinkがなかった。

### rocprof、ROCTx、cleanup

- capture tool、rocprofv3、runnerは各1回だけ開始された。rocprofは runner をprofile対象にし、validatorとgatesはprofile対象外だった。timeout、outer signal、残存childはなく、capture process-group cleanupはPASSした。
- rocprofはrunner exit 1を受けて失敗した。stdoutは0 bytes、stderrはrunner failureと同じ69 bytesだった。capture artifact、kernel/API/memory record、selection raw/summaryは生成されず、measurement/promotion eligibleはいずれもfalseだった。
- runner はROCTx libraryのpinned FD load後にdriver spawnまで到達したが、driverがready前に終了したため、warmup/measured rangeは開始されず、ROCTx exact 12 marker artifactは生成されなかった。
- trusted lock substrate は directory inode `779218`、lock inode `779219` だった。runner終了後にholderなし、childrenなしを確認し、lockとdirectoryを削除してcleanup PASSした。

### A4 restoreとpost-health

- failure時restoreは固定 absolute monotonic deadline `1354096000482005` ns、timeout 120秒で実行された。開始 `1353976000482005` ns、完了 `1353991076532667` ns、duration `15076050662` ns、6 pollsでPASSした。
- package full content hashはexact 1回だけだった。7,700,872,459 bytes / 1,045 files、SHA-256 `a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1`、duration `17521940815` ns。tree identity `e105d6f242fc3792fb0e3b5c3dd7ea98449e1b451af0aec1f73c40571438d9a3` はfull hash前後と最終metadata recheckで一致し、restore中のfull rehashはなかった。
- service は明示的stop/startで復旧した。post service PID `3212819`、worker PID `3213208`、`NRestarts=0`、active/running、lock busy、AMD/KFD owners `[3213208]` で一致した。
- formal healthはgateway healthz/readyz/modelsとOpenWebUI healthが全てHTTP 200、container healthyだった。postflightのtargeted external remainingは空で、maintenance/capture/rocprof/runner/driverの残留processはなかった。

### immutable evidenceとSHA-256

- operator result `operator-result.json`: `0c74805df62a47f51aceb000a0fc61be86d22d913524dba921ba5af9ca50f39e`
- maintenance `launcher-evidence.json`: `e8a25ac31db046ee36372c41d8f03fdaea17c583f2f370c08eca1bb3dbd1c757`
- launcher `launcher-evidence.json`: `7e77a3b4750474dfa85317d0147423099f9a07748bb86309a5e575a184570143`
- runner `resident-batch.failure.json`: `2ae44b70d595f2278022852cec1148825ee0cc090b3ce84718f3407997542904`
- driver stderr: `0b588624e50438d9c61ffe32d73830bf081304a7bc23ba612f1d1349b9beaae7`
- capture `capture-failure.json`: `0f65f7b998016f4066e69bbc948f58e598b2612314f488f3ce85c3bc64c171db`
- operator / maintenance / launcher / runner / capture の `SHA256SUMS` SHA-256は順に `ba0ee7f592580c9897802a0a4d872d22a8ca6aa473d80b371ace83f4257d9cb9`、`c70a54cefe66e95d13cc1f70433410eb3a829517eeb573dc895def9434dbbb3c`、`9ae10e3800bd52a11b0c341c43a70b5b91e673711a9d07a5495739f2034f54f3`、`1b7c08ffd2212cfde0afc96ebe306e7d7b86bf7cd853a88bc32d0dfda2fdcb64`、`7809ff16a0754805f0deb160f3e0119181cfb1c424e792ced528f4c00787bd9a` で、全entryがPASSした。
- 5 evidence directoryは0555、全fileは0444で封印した。

## 次の行動

- このsingle-use authorizationは失敗証跡として消費済みであり、再実行しない。
- 次版ではserved-model loaderへ論理pathを渡したままFD identityを別経路で検証するか、loaderに明示的なtrusted inherited FD入力を追加し、`/proc/self/fd` path表現とno-symlink契約の衝突を解消する。
- 修正版はactual authorizationの前に、pinned served manifest経由でdriver readyまで到達するnegative/positive testを追加して検証する。
