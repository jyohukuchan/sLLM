# P2 resident one-case actual attempt v8

## 前回の要点

- v7はreadyとresident model loadまで成功したが、runner v3がprepared preflight linkをlive preflight linkで上書きし、case beginでdriverがschemaを拒否した。
- v8はrunner v4でprepared preflightとlive preflightを分離し、同じone-caseを2 warmup + 10 measuredまで進める契約だった。

## 今回の変更点

- commit `fba7d3cf`のoperator manifestだけをargv源とした。実行時HEAD `3a87874fafff2849e5847363c9a1316fd8afd868`は別のfidelity gate更新だけで、manifest inputsは変更されていない。manifest SHA-256は`b7a247e382b5545702af11a3b35761c2391c6f3fb0bbea0ad277f6ae1a644b9d`、canonical command SHA-256の独立再計算値は`31d761bf026f77fe022d0bbcf93ad01ea8557d233e3e1973d5cf33215510d6de`でrecordと一致した。
- operator/ready SHA256SUMS、Python/harness/ready artifactのhashと権限、harness latest commit `9fd24068b9a924348827b01b06a52271ec698d2f`、ready latest commit `cf8906902778de6fc4d56b960a25fc563a66f7ff`、ready semantic bindingを確認した。3 fresh outputは実行直前までABSENTだった。
- preflightはformal health `9/6/6`、全4 endpoint 200、service main PID `742209`、worker PID `742650`、`NRestarts=0`、AMD/KFD owner、busy lock、production hashesをPASSした。RAM availableは約88.6GB、disk availableは約2.58TBだった。
- 同一PTYでsudoをprimeし、manifestのcwdと9要素argvを`subprocess.run(..., shell=False)`へ変更せず1回だけ渡した。開始は`1784071927825827810` unix ns、終了は`1784072008390295024` unix ns、elapsedは`80,564,467,214 ns`、return codeは`1`。再試行とprofile実行はしていない。
- lock substrateはdirectory device `26`/inode `752400`、lock device `26`/inode `752401`で作成された。stopped gateは2 pollともstableで、poll SHA-256は`d922bb8fdbbec1e9542d3d3c2ed2d006c382cfa4824c4d615b88ff63ba452d02`と`759e17cd1845776b92e7e54a37f7a8f579928b3d2ae0edf190c23b0a248805de`。live preflightはAMD/KFD owner `[]`、VRAM used `0`、同じlock inode freeでPASSし、SHA-256は`583ca72a16208817d031255cf2e1df21e6184a75041bcf890e8da5a02930df96`だった。
- validatorはexit code `0`。runner PID `750600`は同じlock inode `752401`を取得し、driver PID/process group `750993`を起動した。driver readyは正常で`model_loads=1`、protocol v2、HIP/`gfx1201`/R9700/runtime index `1`、driver SHA-256 `6dc82558b79194b8d690d20213a48e4206cd8bc25a3f37a5b6ade26521ee22b8`を確認した。runner v4のprepared preflight分離もdriver validationを通過した。
- driverはcase beginへの応答前にexit code `1`で終了した。failureはstage `case_begin:p2-representative-full_model-cold_prefill-cold_batched-n128-m128-r9700-rdna4-aq4_0_target`、kind `eof`、reason `resident driver exited before response`。runner exit codeも`1`、launcher failureはstage `runner`、reason `execute runner subprocess failed`だった。
- driver stderrは74 bytesのcomplete record、mode `0444`、secret scan detected `false`、SHA-256 `0b102ea25a2c4e3b211c138c3047281c70d1acc4a938e69aada3f8b975f16513`。exact stderrは`ullm-aq4-p2-resident-driver: case workload/control/device binding differs\n`だった。
- exact mismatchはarchitecture vocabularyである。固定driver commit `084d2e71114857da77e4196061d18a1dfefd53e8`の`validate_case`はlines 771–775でruntime identity `gfx1201`を`RDNA4`へ変換し、line 816でcase device architectureと比較する。一方、bound target caseの`device.architecture`は`gfx1201`である。このためcase workload/control/device guardが必ずfailした。
- protocol evidenceはready received `true`、stdout event `1`、case begin/end complete `0/0`、warmup completed `0/2`、measured completed `0/10`。run commandへ進まず、raw case resultとsummaryは存在しない。readyによりresident model load `1`とHIP runtime identityは確定した。artifactの`model_load_executed`/`gpu_command_executed` safety分類は`unknown`、post-driver GPU ownerは`not_probed`のまま維持する。
- driver cleanupはPASSし、reaped `true`、process group alive final `false`、errors `[]`。launcher children remaining `[]`、maintenance runner children `[]`。substrate cleanupとouter restoreもPASSした。
- post service main PID `751803`、worker PID `752212`、`NRestarts=0`、formal health `9/6/6`、全4 endpoint 200、production hashes、AMD/KFD owner、busy lockは正常。systemd再作成後のdirectoryはdevice `26`/inode `752422`、lockはdevice `26`/inode `752432`でactual substrateとは別epochである。
- maintenance evidence SHA-256はlauncher `cecdfb1033a04c340448b62be9370931f20404f849482249c0a3ed6cab833acf`、marker `d97e17273d4cc9bd58be582e1c4539afee4f9160809d2d2df24144c0e22d77da`、poll 0 `d922bb8fdbbec1e9542d3d3c2ed2d006c382cfa4824c4d615b88ff63ba452d02`、poll 1 `759e17cd1845776b92e7e54a37f7a8f579928b3d2ae0edf190c23b0a248805de`、SUMS `76e89c99a1d6f54bd7c0f13f1fbb459c368e5bb8ca545bf66eec3d276104d719`。verificationはPASSした。
- launcher evidence SHA-256はlauncher `ed827504a601f137b179c86d92cf94505bd8deb0141cbf161804a041793a2a45`、live preflight `583ca72a16208817d031255cf2e1df21e6184a75041bcf890e8da5a02930df96`、runner stderr `d4e032c691321da93e28cbcaac192e4f6b0da28d5efc10bd445b9543fe51a01e`、validator stdout `7c463b16bab152c3554ee355938e1731b1ba65e3ea059adf22e0ccf329635c2a`、SUMS `5e4f9b3a3d434715e7ba125ddd6dd59331a2f68638fb3a193aaf4a016e8329a5`。verificationはPASSした。
- partial runner tree SHA-256は`ff24102d8e8e1794b3992e6f0f8f7c96b987b95659c080985f25b1b532c92a8f`。failure JSONは`6465fdc8403abc250ac1122a7fefc190e24697f7797c80d9375cf316fd894877`、lock ownerは`bcb36fa89c1ae97b4a155d768c38fcd9e2d15de3e74bba8a3dc83025a23f4282`、driver stderrは`0b102ea25a2c4e3b211c138c3047281c70d1acc4a938e69aada3f8b975f16513`。

## 次の行動

- v8のsingle-use outputは再利用せず、このversionは再試行しない。
- 次版ではdevice architectureのcanonical vocabularyを一つに固定する。今回のbound caseとruntime identityはともに`gfx1201`なので、driverがcase比較時だけ`RDNA4`へ変換しない方が既存identity bindingと整合すると考えます。
- case validationの各predicateを個別reason codeで保存し、architecture mismatchを単一の総称errorへ畳み込まない。
