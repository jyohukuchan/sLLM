# P3 profile diagnostic actual v4

## 前回の要点

- quiet-window v8は29連続clean、`137.175538188`秒、reset 0でGOだった。
- operator manifest v4はauthority `ea6fdd2ab7a77502f5dffd6ec77d5ce8c5cfea03`、raw SHA-256 `97f4ea10353575dedbe23c642fc0f6bd6391899ef969305eb5442cc30b38baad`で、exact argv 10、shell false、最大1回に固定されていた。
- v3ではdriverがready前にserved-manifestのsymlink検査で終了した。v4 authorityは新しいresident source/binaryと再生成済みの5成果物rootを固定した。

## 今回の変更点

### 単一actual実行

- 同一PTYでsudo credentialを確立し、manifest/SUMS/input/fresh 7/quiet/current service epoch/lock/AMD・KFD owner/formal health/targeted external 0を開始直前に再検証した。全gateはPASSした。
- exact cwdとargv 10をshell falseで1回だけ実行した。canonical startは`1784113603035575212` ns、endは`1784113655191061884` ns、elapsedは`52155486672` nsだった。
- return codeは`1`、operator stdoutは254 bytes、stderrは0 bytesだった。再試行は行っていない。

### failure boundary

- validatorはexit 0、stderr 0 bytesでPASSした。capture toolとrocprofv3は各1回起動した。
- resident runnerはexit 1となり、stderr 75 bytesに`AQ4 P2 resident batch failed: resident driver did not prove one model load`を記録した。
- FD-map transportを経由してrunner/driver-ready検証へ到達したが、exact ready event、model load 1、served-model bindingを証明できなかった。
- driver process v2のpartial evidenceはlauncherのfailure cleanupがrunner output rootを削除したため保持されなかった。ROCTx 12 range、capture artifact、runner summary/rawも生成・保持されていない。
- このためGPU commandとmodel loadの実行状態は`unknown`であり、成功や未実行とは判定しない。

### cleanup、restore、post-health

- capture process group cleanupはPASSし、children remainingは空だった。trusted lock substrate inodeはdirectory `785089`、lock `785090`で、holderなしを確認してcleanupした。
- package full hashはexact 1回だった。7,700,872,459 bytes、1,045 files、SHA-256 `a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1`でPASSした。
- restoreは固定absolute deadline `1360727886699005` ns内でPASSした。duration `15014218865` ns、poll 6回、final tree identity `e105d6f242fc3792fb0e3b5c3dd7ea98449e1b451af0aec1f73c40571438d9a3`だった。
- post serviceはactive/running、main PID `3707845`、worker PID `3708216`、AMD/KFD ownerは共にworkerだけだった。formal healthは全endpoint PASSで、targeted externalとresidual processは0だった。

### immutable evidence

- operator result SHA-256: `0ccee71396e7ee3a45f08f50e3fb28968f837a9d38d548b2b759149e65609c3d`
- maintenance evidence SHA-256: `7191f5f71c4aa29df644536ced0971921b003c7227081e2949889e414a9320e4`
- launcher evidence SHA-256: `3b97916689750e8cb85f1b9aec1a723fe014b7bc5672929194da59bd260dc77a`
- runner target manifest SHA-256: `34e3a037b9c75092cb374a43a28a1eef49f9d76d8e3fe4dadee0a2b6bd2be2b7`
- capture failure SHA-256: `58619cb05c13cac5fed392d587c7d9878a53bba6ed02ace15e1c37d5969e99c5`
- operator、maintenance、launcher、capture、actual auditの各rootをSUMS付きでimmutable封印する。

## 次の行動

- このsingle-use authorizationは失敗として消費済みであり、再実行しない。
- 次版ではdriverが出したready候補eventをfailure時にもimmutable保持し、model-load/served-bindingのどのfieldが不一致だったかを特定できるようにする。
- launcherのfailure cleanup前にresident process v2を別evidence rootへ退避し、driver-ready失敗でもFD-map、served binding、cleanupを欠落なく監査できる契約へ更新する。
