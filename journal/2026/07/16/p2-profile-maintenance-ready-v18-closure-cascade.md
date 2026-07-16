# P2 profile maintenance / ready v18 closure cascade

## 前回の要点

- launcher-v11、maintenance capture-v2、ready-v17、offline-v12は、それぞれのsource／artifact authorityで封印済みである。
- actual-v15はcommit `99faf0066b93eb021fa83bea1b1a0193d9a79fd4`、tree `0503c595c738ab66173918bd95986be613ddfc00`の66 filesとして封印され、capture-v11は`complete_diagnostic`になった。
- ready-v17、offline-v12、actual-v15はhistorical artifactとして不変に読み込む。

## 今回の変更点

- actual-v15の`capture-artifact.json`を実読し、profilerの旧15 fieldsに対する差分がtop-level `execution_closure`だけ、`target_environment`の旧4 fieldsに対する差分が`injected_fd_map_key`だけで、missingとその他のcanonical semantic diffが0であることを確認した。後者の厳密値は`ULLM_AQ4_PINNED_FD_MAP`である。
- maintenance success validatorはprofiler exact 16 fields、execution closure exact 9 fields、closure binding／identity／method／SHA schema、capture helper closure、injected FD map keyをfail-closeで検証する。unknown、missing、wrong valueの6負例とactual-v15 canonical再構成をtestsへ固定した。
- launcher-v12 source authorityをcommit `780a68007d424e1cf3f53d4e60728161ce6d13d4`、tree `bee76471ea0faee2a5c95aea0fa405f0620fe515`、blob `1f55f4ec02e1f41c04e988f18d3c92f9c01689d5`、raw SHA-256 `55977291b6300b9365e685b4482a3c5ba3c21eb7e5ce7eb777aa7440791dda8a`へ固定した。execute-binding-v12 artifact authorityはcommit `9fdab4c5aa2c60813fbe9c0527ac0bdffa725044`である。
- current namespaceをruntime-v12、execute-evidence-v12、capture-v12、maintenance-evidence-v13、ready/dry-v18へ進めた。offline laneはactual-v15をsourceとするcapture/evidence-v13へ進めたが、canonical offline-v13 artifactsは生成していない。capture authority `418e507214b2a4c0352ac8867bf9689b81948ca4`とschema v2は維持した。
- actual-v15 exact 66-file seal、historical ready-v17、historical offline-v12の最終状態非依存loaderを追加した。historical ready-v16、offline-v11、actual-v14のloaderも維持した。
- maintenance direct test authorityはcommit `def89e583737778531b7c1b03e61b54580f09afd`、tree `770efc498f577a0e84c818f7e015803845b4feeb`、blob `91bc728cf543d2dd41a515ee6f105d7ed552d622`で、181/181 testsがpassedした。
- maintenance source/QA authorityはcommit `fd0b964d8467cd34ad7f8a012ee1f91869a71560`、tree `8ff15ce68c1b17b000d298d931535b69e939282a`、blob `906c310252205a75b6a8ee442f2cd8c1ba54c896`、raw SHA-256 `c857ebda0009d5c3ad7ba6aa01d9225e16c5e95a232ce7be2a78962b39041eb6`である。
- QA manifestは13 distinct files、resident trust 426、driver 22、ROCTx 5、capture 84、selection producer 105、family exclusion 39、selector 26の合計707/707 testsをjobs 1でpassedした。QA attestation SHA-256は`15fa9f0c004a0c31fb872e1ed45ce29383c8108728b765f8329734cf174a32ea`である。
- ready-v18とdry-v18を各1回生成した。ready binding、harness trust、QA attestation、ready SUMSのSHA-256は順に`507bc4cd433769a7bc11b7cba033a81405f2d9db2dad2bce9c9dce990c74481a`、`87fa1090406dc11dcd4c640bc149445a0de80f8e6fac244e36e366d59fbd5c13`、`15fa9f0c004a0c31fb872e1ed45ce29383c8108728b765f8329734cf174a32ea`、`cc2c977428768ad2af6b92d1343857002eb4d953d7e5e299bac0ba4026d7cdf7`である。
- dry evidence／SUMSのSHA-256は`099ffb4d96b6cbab6eec8f9d40ad134424860e43fee20a39ee0c09e28a2b3552`／`b6458ed8bb4f52b6f482b3d7150f6123408a18e74bbfb359246d292bcd93a53f`で、全process count 0、service／GPU／model flagsはfalseである。
- ready artifact authorityはcommit `42856dbf80ca06b51a70994b224151320b0011ef`、repository tree `e06d1f99cdf64b9775f2c01daadd407bafd9d768`、ready root tree `78bc3006ec93405742239be4b99eeeb4023d1da8`、dry root tree `dbacd3dbb2e4987e13a5c93e5eed02b47c4afb04`である。両rootは`0555`、6 filesは`0444`である。GPU、service、model load、actualは実行していない。

## 次の行動

- ready-v18／dry-v18とmaintenance source authorityを下流へ渡す。
- actualを実行する場合はready-v18を単一authorityとし、runtime-v12／execute-evidence-v12／capture-v12／maintenance-evidence-v13を各1回だけ使用する。
- offline再構成が必要な場合だけ、distinct offline capture/evidence-v13へ生成し、actual-v15 sourceを不変のまま検証する。
