# P3 profile ready v10

## 前回の要点

- profile-ready-v9とdry-run-v9はcommit `ed10d543bcc0d196b157fad306a6995782692fe9`でimmutable化した後、QA source manifest 12件のうちexecute launcher test authority 1件がstale pinであることが判明した。
- v8とv9は`invalid-preoperator`であり、downstream使用は禁止する。旧v7〜v9は削除や上書きをしない。

## 今回の変更点

- 全12 manifest authority修正後のmaintenance authorityをcommit `9926129547b8d8b9cc21bd64788ee19cc892f7d7`、tree `83cba2d41bbb26e2fd0cec97cb07a88f01c10a6e`、harness blob `608538c2fc6f5ccb144e8ce928fad2a841e4ce0f`、raw SHA-256 `217fed9eeff8139cdb34c542633f2893cfc185e439486cd3b8cd9543cf455114`へ固定した。
- maintenance test authorityはcommit `0c948a2cb519f620f4c27bdd584f96d45070ea7b`、blob `5eb7578c2307ea959dbdd4a4cf8ce27dd99a6378`。execute launcher test authorityはcommit `c5d17d7524608ea4438ba82a1fb51637b0876de6`、blob `5285531b3a6a952114ad3139a39a72a268dabb6a`で、12/12 manifest blobとcurrent sourceの一致を確認した。
- final HEAD aggregateは544/544 passedで、resident 382件、launcher 22件、capture 140件を含む。
- fresh profile-ready-v10を1回生成し、formal loaderでstatus `ready_for_one_case`、execution mode `profile_diagnostic`、actual eligible trueを確認した。
- fresh profile-ready-dry-run-v10を1回実行し、status `passed`、全process count 0、service操作、GPU command、model load、captureが未実行であることを確認した。
- ready binding JSON SHA-256は`cc4c9f76c7438c7e25a33db4bfa9c4b1de34ca2273f2b522de1dce52d3a65a61`、ready SUMS SHA-256は`59cc1c52d864040ba722ceb7a88bd4c0cf961b1d311912be79bffa55cccb4690`。
- dry-run evidence JSON SHA-256は`d3c9b2827dd7daa48be6018695282f822259f0667b37bc6eb82500970166675a`、dry-run SUMS SHA-256は`72f12e3c17ebf114dfa9a4eb2398b07f9a612b42909ac2c5abc73ad3cc61e7f9`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`のreadbackがpassedした。旧v7〜v9は不変。
- quiet、operator、GPU、service、actualは実行していない。

## 次の行動

- profile-ready-v10 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- operator sourceをprofile-ready-v10へrecascadeし、独立pre-operator GOを得るまではquiet-window採取とactualを実行しない。
