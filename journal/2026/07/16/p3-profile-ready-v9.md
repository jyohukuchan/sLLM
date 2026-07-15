# P3 profile ready v9

## 前回の要点

- profile-ready-v8とdry-run-v8はcommit `23140edc690f36527e6de147b5c5a2abbcab0392`でimmutable化した後、final HEAD aggregateに残存していた生成後absence assertionが1件failした。
- v8成果物自体のSUMSとauthority bindingは整合するが、QAは543/544相当なので`invalid-preoperator`であり、downstream使用は禁止する。削除や上書きはしない。

## 今回の変更点

- final-state-independent修正後のmaintenance authorityをcommit `b686be6823d07d1b26e5fdd8f0c1e259c8426fa2`、tree `b8ab87d6b0456cbfc29f06ffebe4842b001e1684`、harness blob `ce84595296d7230c8eef125e9babb1dee766993d`、raw SHA-256 `2ed0cbd8f49cf4288be71a17e9d25f91ecb84d903a68b3576c3cfd8b9eb684a5`へ固定した。
- maintenance test authorityはcommit `2d991c7abd08e4ff39a1418bfad64f9d176265c6`、blob `1a52d7220f4950ad273ac099410e3e6608bf17d9`、maintenance tests 156件。final HEAD aggregateは544/544 passedで、resident 382件、launcher 22件、capture 140件を含む。
- ready/dry artifact自身の存在に依存するabsence assertionを除去し、actual run、execute evidence、maintenance evidence、captureのfresh境界4件は維持した。
- fresh profile-ready-v9を1回生成し、formal loaderでstatus `ready_for_one_case`、execution mode `profile_diagnostic`、actual eligible trueを確認した。
- fresh profile-ready-dry-run-v9を1回実行し、status `passed`、全process count 0、service操作、GPU command、model load、captureが未実行であることを確認した。
- ready binding JSON SHA-256は`fc05c02cb0d3eabc91ef08d31ea643d582cd615ecc4558031cca2b3af8fc5c5d`、ready SUMS SHA-256は`c4ce3a86ad02c6252170b8f1b753d60a9dd011322141d92b20f2d7538c0c0570`。
- dry-run evidence JSON SHA-256は`2c5f85682d2fa3f8ea9742fbb8ef7a029bd2d7400c72e31d2ebbaa8a15c89f86`、dry-run SUMS SHA-256は`8e679f5816e79ccc39f23eda0662bb1fc80a1d7c5aad1f70b05d8d4c7e38a4c4`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`のreadbackがpassedした。旧v7とinvalid v8は不変。
- quiet、operator、actualは実行していない。

## 次の行動

- profile-ready-v9 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- operator sourceをprofile-ready-v9へrecascadeし、独立pre-operator GOを得るまではquiet-window採取とactualを実行しない。
