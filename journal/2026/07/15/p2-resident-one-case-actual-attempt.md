# P2 resident one-case actual attempt

## 前回の要点

- canonical operator commandはcommit `add661f3de93a97c60038814eb788366b177e9fc`の`command-manifest.json`に、cwdと9要素argv、`shell=false`、最大1回として固定されていた。
- operator manifestとready artifactのSHA-256/SHA256SUMS、`0444`/`0555`、`actual_executed=false`、3つのfresh outputを再確認した。
- read-only production preflightはformal health process counts `9/6/6`、全endpoint HTTP 200、service/worker/GPU/KFD/lock、production hashes、RAM、diskの全項目でPASSした。

## 今回の変更点

- 同一PTYでsudo credentialをprimeし、manifestから読んだcwdとargvを再構築・相対化せず、`shell=false`でcanonical commandを1回だけ実行した。profileは実行していない。
- canonical commandは`1784055794144931342` unix nsに開始し、`1784055846612596440` unix nsにreturn code `1`で終了した。elapsedは`52,467,665,098 ns`。再試行はしていない。
- harnessはservice stopとdurable marker作成までは成功したが、`stopped-gates`で`target GPU compute owners are not zero`となった。launcherは未開始、model loadは未実行、warmup/measuredは`0/0`で、runner raw/summaryとlauncher evidence/outputは作られていない。
- outer finallyによるrestoreはattempted/passed。serviceは新epochのmain PID `3900525`、worker PID `3900624`でactive/runningへ復帰し、`NRestarts=0`は不変だった。post formal healthは`9/6/6`、全endpoint HTTP 200、GPU/KFD ownersは新workerのみ、lockはserviceが保持し、production hashesも不変。actual関連childは残っていない。
- pre-stop時点のGPU/KFD ownerは旧worker PID `3090924`だけだった。systemdは04:03:38.461201にstopを開始し、gatewayは04:03:38.495038にworker stdout EOFを観測、04:03:38.688829にStoppedとなった。harnessのstop recordは04:03:38.693855、restore前sudo recordは04:03:39.558792で、stopped gatesはstop record後`0.865秒`以内に失敗した。
- old-worker pgrep gateは通過した後にAMD-SMI owner gateが失敗した。例外時のAMD-SMI process JSONは現行harnessに永続化されないため、失敗瞬間のPIDは確定できず、後段のKFD/lock gateも未観測である。pre/postにforeign ownerがないため、旧worker PID `3090924`のAMD-SMI側の遅延解放が最有力だと考える。
- immutable maintenance evidenceは`resident-one-case-smoke-maintenance-evidence-v1/`にあり、directory modeは`0555`、全file modeは`0444`、SHA256SUMS検証はPASSした。主要SHA-256は以下。
  - `launcher-evidence.json`: `b3c4e38712d67443f96aa7178f0a05a3972b628bf56cdf906956b2c16b2023ec`
  - `maintenance-marker.json`: `a067bd3b3804f860d0c061cb44c6ed4ab053eac5fc3a0e7e20f0d5e158d03005`
  - `SHA256SUMS`: `bf2d9c93239fda16c24ff12af7dbd46d9756f9da146ef36dbe41a4a21e785468`

## 次の行動

- このsingle-use outputは再利用せず、actualも再試行しない。
- 次のoperator artifactを新しいoutput pathで用意する前に、service stop後のAMD owner、KFD owner、lock freeをbounded pollで待ち、連続してPASSした時だけlauncherへ進むようにする。
- timeout時にはAMD-SMI/KFDのPID、`/proc/PID/cmdline`、VRAM、lock状態、観測時刻をfailure evidenceへ保存し、遅延解放とforeign ownerを判別できるようにする。
