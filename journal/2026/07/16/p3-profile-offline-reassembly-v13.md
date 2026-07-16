# P3 profile offline reassembly v13

## 前回の要点

- maintenance sourceはcommit `fd0b964d8467cd34ad7f8a012ee1f91869a71560`、tree `8ff15ce68c1b17b000d298d931535b69e939282a`、blob `906c310252205a75b6a8ee442f2cd8c1ba54c896`、raw SHA-256 `c857ebda0009d5c3ad7ba6aa01d9225e16c5e95a232ce7be2a78962b39041eb6`である。
- ready-v18はartifact commit `42856dbf80ca06b51a70994b224151320b0011ef`、repository tree `e06d1f99cdf64b9775f2c01daadd407bafd9d768`、ready root tree `78bc3006ec93405742239be4b99eeeb4023d1da8`、dry root tree `dbacd3dbb2e4987e13a5c93e5eed02b47c4afb04`で封印済みである。
- actual-v15はcommit `99faf0066b93eb021fa83bea1b1a0193d9a79fd4`の66 filesであり、offline-v13のsourceとしてだけ読み取る。

## 今回の変更点

- 独立QAではmaintenance test blob `91bc728cf543d2dd41a515ee6f105d7ed552d622`、launcher-v12 commit `780a68007d424e1cf3f53d4e60728161ce6d13d4`、execute-binding-v12 commit `9fdab4c5aa2c60813fbe9c0527ac0bdffa725044`、capture commit `418e507214b2a4c0352ac8867bf9689b81948ca4`を確認した。
- ready-v18／dry-v18の6 files、SUMS、root `0555`、members `0444`／single-link、Git committed bytesを確認した。dry process countsはすべて0だった。
- actual-v15 profilerは旧15 fieldsに対して`execution_closure`だけ、旧target environment 4 fieldsに対して`injected_fd_map_key`だけが追加され、missing／その他のcanonical差分は0だった。値は`ULLM_AQ4_PINNED_FD_MAP`である。
- archived target manifestのinput/runtime/control 27 entriesとcapture artifactの27 closure bindingsをrole、path、closure、method、SHA-256、保存identityで交差検証した。strict closure validator、capture helper closure、FD map schema/hash fieldsも合格した。
- historical ready-v17とoffline-v12をcurrent final stateに依存せず読み込んだ。actual-v15の66-file sealは生成前後で`members_sha256=c8a65103f533874da29e03b416b004fbf1f84169ff84f8a0dc11f4754036480a`のまま不変だった。
- QA manifestをjobs 1で再実行し、resident trust 426、resident driver 22、ROCTx 5、capture 84、selection producer 105、family exclusion 39、selector 26の合計707/707 testsがpassedした。
- GO判定後、actual future runtime-v12／execute-evidence-v12／capture-v12／maintenance-evidence-v13を未作成のまま、distinct offline-v13だけを生成した。
  - `p3/aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v13`: 40 files
  - `p2/resident-one-case-smoke-profile-maintenance-offline-reassembly-evidence-v13`: 2 files
- formal validatorは生成直後とcommit後に合格した。42 filesはroot `0555`、members `0444`／single-linkで、Git archive、Git blob、worktreeのbyte equalityは42/42、mismatch 0だった。
- offline evidence schemaは`ullm.aq4_p2_profile_maintenance_evidence.v13`である。主要SHA-256は次のとおりである。
  - capture artifact raw: `3e79c3fc61f978ca97f432fc958e7542c46182ee5f5298168e0a0ac877629654`
  - capture artifact semantic self hash: `bd65d676c0c284244dc6e51f435d7ade4190f29c45e6b2b2212d045cf908e645`
  - capture `SHA256SUMS`: `11c53ff23d8f6d8eeb99a934019b3dbf7fcecb0ab71a78eab8fdf5677ca98720`
  - offline evidence raw: `8a8c9c28fc0d79365ac5fa2088ea336de6c52273acddebc252e0fcae34662acb`
  - offline evidence semantic self hash: `c4a73242cfd68a5f0b5d1d96e90b9ae0f3caa66341767192c79bc76b5e647c22`
  - evidence `SHA256SUMS`: `681ab7018bd1219407412a5d74aa367d35e539e1286705d6186d30553224a663`
- artifact commitは`f1f92ad90834514f93ec92690f0285ea2b515c63`、repository treeは`5a7c3a0b822216c6a32241ca02ff091a359bd077`、capture root treeは`4bb4c5d777eb32d4e7b8a807e359a035ab97dce6`、evidence root treeは`a1fa9e733b3ccc2ccd67afc101634dd6824a61d4`である。
- execution counterはoffline assemble 1以外すべて0である。workload、rocprof、GPU、service、operator、actual、model loadは実行していない。

## 次の行動

- offline-v13 authorityとready-v18 authorityをoperator-v16のauthority bindingへ渡す。
- actualを実行する場合はoffline-v13を実行証拠として流用せず、ready-v18とfreshなactual future 4 rootsを使う。
