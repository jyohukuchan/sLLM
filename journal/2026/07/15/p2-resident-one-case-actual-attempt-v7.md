# P2 resident one-case actual attempt v7

## 前回の要点

- v6はdriver ready前にworker binary hard-link count `2`をsingle-link hash guardが拒否した。
- v7はproduction workerのexact hard-link fixtureをdriverへ固定し、runner v3、driver protocol v2、fresh execute output v3で2 warmup + 10 measuredを目標にした。

## 今回の変更点

- commit `45d79e975533798d129d4eaf09a4f3ee47c9274d`のoperator manifestだけをargv源とした。manifest SHA-256は`0f156997915546c9392dc9e8536fe13f751e68373bba82b1929a5a6c1c1b5f0a`、canonical command SHA-256の独立再計算値は`6576afbecfb33ecfd9665aab3fdbc3ddea48c54f747cb074f8a21d83fbee69a5`でrecordと一致した。
- operator/ready SHA256SUMS、Python/harness/ready artifactのhashと権限、harness latest commit `6dd40ac10f4fb009b419329df6977cd4ce7b8634`、ready latest commit `128ae3dcda528c9ea1a7c9735cfaac94c711bba1`、ready semantic bindingを確認した。3 fresh outputは実行直前までABSENTだった。
- preflightはformal health `9/6/6`、全4 endpoint 200、service main PID `437627`、worker PID `438020`、`NRestarts=0`、AMD/KFD owner、busy lock、production hashesをPASSした。RAM availableは約90.9GB、disk availableは約2.58TBだった。
- 同一PTYでsudoをprimeし、manifestのcwdと9要素argvを`subprocess.run(..., shell=False)`へ変更せず1回だけ渡した。開始は`1784070423451591486` unix ns、終了は`1784070503579732484` unix ns、elapsedは`80,128,140,998 ns`、return codeは`1`。再試行とprofile実行はしていない。
- lock substrateはdirectory device `26`/inode `751338`、lock device `26`/inode `751339`で作成された。stopped gateは2 pollともstableで、poll SHA-256は`1eb3f7d40afd24eb1887cbf86505b2ed324cc41a98852b7fde08e027a4c32595`と`1187e2c9317675202d190d06121991dd24161b7e12badc81824eefaf8fe6178d`。live preflightはAMD/KFD owner `[]`、VRAM used `0`、同じlock inode freeでPASSし、SHA-256は`f299e3cb39cfb5be371c0e646793303906df060217a29b082f336e51fc742c54`だった。
- validatorはexit code `0`。runner PID `646152`は同じlock inode `751339`を取得し、driver PID/process group `646804`を起動した。driver ready eventは正常に受信され、`model_loads=1`、protocol `ullm.aq4_p2_resident_driver.v2`、runtime deviceはHIP/`gfx1201`/R9700/runtime index `1`、driver binary SHA-256は`6dc82558b79194b8d690d20213a48e4206cd8bc25a3f37a5b6ade26521ee22b8`だった。worker hard-link guardとresident startup/model loadは通過した。
- driverは次のcase beginへの応答前にexit code `1`で終了した。failureはstage `case_begin:p2-representative-full_model-cold_prefill-cold_batched-n128-m128-r9700-rdna4-aq4_0_target`、kind `eof`、reason `resident driver exited before response`。runner exit codeも`1`、launcher failureはstage `runner`、reason `execute runner subprocess failed`だった。
- driver stderrは53 bytesのcomplete record、mode `0444`、secret scan performed `true`/detected `false`、SHA-256 `e888e43b164a0ae4fdb514a510ec213ff079751967fb4f1b00eb58a7ebd3feba`。exact stderrは`ullm-aq4-p2-resident-driver: preflight fields differ\n`だった。
- exact root causeはrunner v3のpreflight link上書きである。`trusted-runner.py`はprepared `preflight.json`から正しい7-field `preflight_link`を作るが、actual live preflight検証後のline 1632で`preflight_link = live_preflight_link`として別schemaのlive gate JSONへ上書きし、line 1698でそれをcase beginの`preflight` linkとして送信する。driverは`weights_bytes`、`persistent_state_bytes`、`kv_cache_bytes`、`workspace_bytes`、`temporary_bytes`、`vram_headroom_bytes`、`gpu_process_snapshot`のexact 7 fieldsを要求するため、live preflightのfield setを拒否した。
- protocol evidenceはready received `true`、stdout event `1`、case begin/end complete `0/0`、warmup completed `0/2`、measured completed `0/10`。run commandには進んでおらず、raw case resultとsummaryは存在しない。ready eventによりresident model load `1`とHIP runtime identityは確定した。artifactの`model_load_executed`/`gpu_command_executed` safety分類は保守的に`unknown`、post-driver GPU ownerは`not_probed`のまま維持する。
- driver cleanupはPASSし、exit code `1`、reaped `true`、process group alive final `false`、errors `[]`。launcher children remaining `[]`、maintenance runner children `[]`。追加確認でもmaintenance/runner/resident-driverの残存processはなかった。
- substrate cleanupとouter restoreはPASSした。post service main PID `647551`、worker PID `647632`、`NRestarts=0`、formal health `9/6/6`、全4 endpoint 200、production hashes、AMD/KFD owner、busy lockは正常。systemd再作成後のdirectoryはdevice `26`/inode `751363`、lockはdevice `26`/inode `751370`でactual substrateとは別epochである。
- maintenance evidence SHA-256はlauncher `43768c0698cfb329b3aa38ee90cbee85616b77aa08797f75401378f9bef1f6b5`、marker `8f4765f417dddff6b38761321b9b521ba0641518b5ca3774e31598d05c23b563`、poll 0 `1eb3f7d40afd24eb1887cbf86505b2ed324cc41a98852b7fde08e027a4c32595`、poll 1 `1187e2c9317675202d190d06121991dd24161b7e12badc81824eefaf8fe6178d`、SUMS `f3faae4e8e5864da373dedb24ea647a960a1da0a89fd1ba8c2c2ed88119549d2`。verificationはPASSした。
- launcher evidence SHA-256はlauncher `80bea86b39ebcb721f835355538e27a291b64de2a7fba8c722f9a1085eb345c9`、live preflight `f299e3cb39cfb5be371c0e646793303906df060217a29b082f336e51fc742c54`、runner stderr `d4e032c691321da93e28cbcaac192e4f6b0da28d5efc10bd445b9543fe51a01e`、validator stdout `7c463b16bab152c3554ee355938e1731b1ba65e3ea059adf22e0ccf329635c2a`、SUMS `318782b1fd17aee46cc1e2b36b5c19501246a72ef12b2bdcc9e0f64aa19e3cb2`。verificationはPASSした。
- partial runner tree SHA-256は`a437c482379541912e4ebf79f18d19455ee4b8c3c425384be1b92b2d595906b4`。failure JSONは`9c7761eea44f1b438183e1bf46bcbc43b76ff55cd3dc30bfb740a64b77d4bfa2`、lock ownerは`cd9485070e2fa31ef72c55f2874775d659ee2935ef53c256ff4f90dc948889ea`、driver stderrは`e888e43b164a0ae4fdb514a510ec213ff079751967fb4f1b00eb58a7ebd3feba`。

## 次の行動

- v7のsingle-use outputは再利用せず、このversionは再試行しない。
- 次版では`preflight_link`と`live_preflight_link`を別の型・変数として維持し、case beginにはprepared 7-field preflightだけを渡す。live preflightはrunner/lock/GPU gateの検証とraw resultのlive bindingにだけ使う。
- negative testでlive preflight schemaがcase beginの`preflight`へ流入しないことと、prepared preflight linkのpath/SHAがdriverへ届くことを固定する。
