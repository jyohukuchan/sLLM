# P2 resident one-case actual attempt v6

## 前回の要点

- v5はresident driverの最初のready応答より前にstdout EOFとなったが、旧runnerはdriver return codeとstderrを保存しなかったため、根本原因、model load、GPU commandは`unknown`だった。
- v6は同じsingle-use one-case契約をrunner v2へ更新し、ready前failureでもbounded/secret-scanned driver stderr、return code、protocol到達点、process cleanupを保存する契約だった。

## 今回の変更点

- commit `696d67198b8bf997b2ba59700009acd7e1099f18`のoperator manifestだけをargv源とした。manifest SHA-256は`b56197852c4d8e388f0172f26d6002463d98303e573b5733e13566fc5f224d1f`、canonical argv command SHA-256の独立再計算値は`be8a7e7c265834685f8f131bbb247b919624428341531119992be520bc06987e`で、recordと一致した。
- operator/ready SHA256SUMS、Python/harness/ready artifactのhashと権限、harness latest commit `7e597eb6e6ecbc61a727db91d68f7054101046c7`、ready artifact latest commit `b5e85b6032ef9cb42fda6baa57cd3662a99ad7c1`、ready semantic bindingを確認した。3 fresh outputは実行直前までABSENTだった。
- preflightはformal health `9/6/6`、全4 endpoint 200、service main PID `242454`、worker PID `242547`、`NRestarts=0`、AMD/KFD owner、busy lock、production hashesをPASSした。RAM availableは約90.9GB、disk availableは約2.58TBだった。
- 同一PTYでsudoをprimeし、manifestのworking directory `/home/homelab1/coding-local/ultimateLLM/uLLM-project`と9要素argvを`subprocess.run(..., shell=False)`へ変更せず1回だけ渡した。canonical開始は`1784067086977896891` unix ns、終了は`1784067155876358569` unix ns、elapsedは`68,898,461,678 ns`、return codeは`1`。再試行とprofile実行はしていない。
- service stop後のlock substrateはdirectory device `26`/inode `749084`、lock device `26`/inode `749085`で作成された。stopped gateは2 pollともstableで、poll SHA-256は`1795373f82af6a7f8365845a00cedf2959bbe4652db170b6979f03af530b0502`と`7ea64cbe7aa31e712981a3bb7375b711b33d6b9b45b9103554da26993b753bf3`。live preflightはAMD/KFD owner `[]`、VRAM used `0`、同じlock inode freeでPASSし、SHA-256は`cbeb1d86e18f924acea2c7d500a11e311e9ad0de37ee428fc2aeacb085d7e6fb`だった。
- validatorはexit code `0`。runner PID `437275`は同じlock device `26`/inode `749085`を取得し、次のresident driver exact argvを`shell=False`でspawnした。

```text
["/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/resident-driver", "--served-model-manifest", "/etc/ullm/served-models/active.json", "--device-index", "1", "--build-git-commit", "319d6187b29e877536aa5dfe80c02bde0c77ed7a"]
```

- driver PID `437613`/process group `437613`はspawn後、ready応答前にexit code `1`で終了した。failureはstage `ready`、kind `eof`、reason `resident driver exited before response`、signal `null`。runner exit codeも`1`、launcher failureはstage `runner`、reason `execute runner subprocess failed`だった。
- 新しい`resident-batch.failure.json`はdriver stderrを73 bytesのcomplete recordとして保存した。secret scanはperformed `true`/detected `false`、SHA-256は`4ede01ba2f794e6dc5b2c64d619c611241fba8d4a413a0fbe4374094738c1444`、exact stderrは`ullm-aq4-p2-resident-driver: hash input must be single-link regular file\n`だった。
- commit `319d6187b29e877536aa5dfe80c02bde0c77ed7a`のdriver startup順は、served manifest parse/validate、served manifest SHA、worker binary SHA、package tree SHA、runtime device query、`RealExecutor::load`である。served manifestはsingle-link、active worker binaryはhard link count `2`である。したがって、これはworker binary SHA guardが`nlink == 1`を要求したことによるfailureであり、package tree SHA、runtime device query、`RealExecutor::load`、ready eventには未到達だったと判断できる。
- protocol evidenceはready received `false`、stdout event `0`、case begin/end `0/0`、warmup completed `0/2`、measured completed `0/10`。raw case resultとsummaryは存在しない。artifactのsafety分類`model_load_executed`/`gpu_command_executed`は保守的に`unknown`、post-driver GPU ownerは`not_probed`のままである。一方、上記の固定binary source control flowから`RealExecutor::load`とprotocol run commandが未実行であることは確定できる。
- driver cleanupはPASSした。initial/final return code `1`、reaped `true`、process group alive final `false`、cleanup errors `[]`。launcher children remaining `[]`、maintenance runner children `[]`。追加確認でもmaintenance/runner/resident-driverの残存processはなかった。
- lock substrate cleanupは同じinodeをunlinkし、directoryをrmdirしてPASSした。outer restoreもPASSし、post service main PID `437627`、worker PID `438020`、`NRestarts=0`、formal health `9/6/6`、全4 endpoint 200、production hashes、AMD/KFD owner、busy lockは正常だった。systemdが再作成した現directoryはdevice `26`/inode `749097`、現lockはdevice `26`/inode `749107`で、actual substrateとは別epochである。
- maintenance evidenceのSHA-256は`launcher-evidence.json` `f7a98aed2fc6ae32ac34edcea61eb62f1da474d6226d92cd7627968f43a44487`、marker `05a75d899456f900bc285cee1057026cb1a92c1e5b7f19ed0bd1d0b5cede2766`、poll 0 `1795373f82af6a7f8365845a00cedf2959bbe4652db170b6979f03af530b0502`、poll 1 `7ea64cbe7aa31e712981a3bb7375b711b33d6b9b45b9103554da26993b753bf3`、SHA256SUMS `222c613981dcb6a8f0b4d7a1fa1c06b617984a83f12b4f7c14e7707e5b6882e0`。verificationはPASSした。
- launcher evidenceのSHA-256は`launcher-evidence.json` `8c7920e6310031494fe19fffcec2fcdab76ae0db9adef88c3017be1b5204cd73`、live preflight `cbeb1d86e18f924acea2c7d500a11e311e9ad0de37ee428fc2aeacb085d7e6fb`、runner stderr `d4e032c691321da93e28cbcaac192e4f6b0da28d5efc10bd445b9543fe51a01e`、runner stdout/validator stderrはempty SHA、validator stdout `7c463b16bab152c3554ee355938e1731b1ba65e3ea059adf22e0ccf329635c2a`、SHA256SUMS `950a236884de4ec6fe21a7bc7c2688367be7ef38db7c43cd8aac3efa6e334819`。verificationはPASSした。
- partial runner tree SHA-256は`d5755bb7630beea98b09d91f2c5aba66b689a9abbff22ab1f94d2245cce54106`。個別SHA-256はfailure JSON `c97fbcbbbc137a962d7ca1580e6bf9e50b90f4e618a7c60fac53e867c7e27af2`、lock owner `56161843ba4965c08268e9eca2e7589761ac991655ad65a7a121f373438000cc`、driver stderr `4ede01ba2f794e6dc5b2c64d619c611241fba8d4a413a0fbe4374094738c1444`。

## 次の行動

- v6のsingle-use outputは再利用せず、このversionは再試行しない。
- 次版を作る場合は、production workerが意図的にhard-link配置されている契約とdriverの`sha256_file()`単一link要求を整合させる。安全な案は、trusted path/identityを維持したままworker binaryだけexpected `nlink == 2`を明示的に受理し、hash前後でdevice/inode/size/mode/nlink/mtime/ctime不変を検証することだと思います。
- driver failure capture v2は今回、return code、complete secret-scanned stderr、protocol到達点、process group cleanupを保存できたため、この証拠形式を維持する。
