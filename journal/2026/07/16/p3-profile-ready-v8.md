# P3 profile ready v8

## 前回の要点

- profile-ready-v7は生成済みだったが、生成後のabsence assertionを含むQA attestationを最終treeで再現できなかったため、pre-operatorはNO-GOになった。
- 旧profile-ready-v7とprofile-ready-dry-run-v7はimmutable historyとして維持する。

## 今回の変更点

- recascade後のmaintenance final authorityをcommit `c0812f3bf9b85933a970e39290de54a8c4cca559`、tree `7a64bd584ddd2e3071554488cef1e64089015f4d`、harness blob `8d4135f31ed70ead93554f5c7c4b1481053e2a1c`、raw SHA-256 `c2423d252993f5948887b681cf9849ec401a98bef54b77dfbda9a86eb9c7d120`へ固定した。
- maintenance test authorityはcommit `41190f2c370f4abc6fbf6166aeed81950f5865a4`、blob `0f562642a92f8971a919e9ed380a1a113e4a9faf`、maintenance tests 156件、resident trust chain 382/382、aggregate 544で確定した。
- fresh profile-ready-v8を1回生成し、formal loaderでstatus `ready_for_one_case`、execution mode `profile_diagnostic`、actual eligible trueを確認した。
- fresh profile-ready-dry-run-v8を1回実行し、status `passed`、sudo、systemctl、launcher、rocprof、capture、Docker、health probeを含む全process countが0であることを確認した。
- dry-runではservice操作、GPU command、model load、captureを実行していない。
- ready binding JSON SHA-256は`81b3b47926962fede87a494b21341b8e0647bfdde645aa1cca445db0650ba800`、ready SUMS SHA-256は`66d7da120abe91dc34482627426ae57984a7ae3c3e5faec55afc3c236a6da7a1`。
- dry-run evidence JSON SHA-256は`5c095eb6797260728be099bfde224c5870a8a1f79346ada7b2dad2d144c09f34`、dry-run SUMS SHA-256は`6da2fa8c486afe4e2deb16e9a64089bfef8cdda233bbaf2516a685340c9907cb`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`のreadbackがpassedした。
- 旧profile-ready-v7とprofile-ready-dry-run-v7は`git diff --exit-code`で不変を確認した。quiet、operator、actualは実行していない。

## 次の行動

- profile-ready-v8 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- operator sourceをprofile-ready-v8へrecascadeし、独立pre-operator GOを得るまではquiet-window採取とactualを実行しない。
