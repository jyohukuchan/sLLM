# P3 profile offline reassembly v11

## 前回の要点

- actual-v12 は commit `44617f7fd46c39f71f04502b248739cc116fe095`、tree `813c4ffc88fb58cf8764b91d3c80cea9ef351f0f` の 35 ファイルで封印済みだった。
- 最初の offline artifact commit `04e1886cfee45052329b8e1fd78af726d9fe9766` は、validator が generator identity を current HEAD と比較したため post-commit validation に失敗し、`cd54ac8824e3f131e941e89b7eab2e11f903a20c` で revert した。
- source-stable v10 commit `e6cc2aabbcad7e20893f0b1c7dfa89a29362847f` は post-commit validation に合格したが、offline sidecar が actual fresh output `resident-one-case-smoke-profile-maintenance-evidence-v11` と衝突したため、`e4673b595599e6f7fe2512a3791ce446d43f41a4` で revert した。v10 の capture root と journal も撤去済みである。
- definitive generator / validator authority は commit `c4fe279e6c0bf9a8899c2cd36642f45bf145fe8f`、tree `49685f2b9194d6128d8e92ad04d52c01540eed38`、tool blob `53ad6ab6eeec43eb77478397ad0fcd8c09caa45b`、raw SHA-256 `4330469041c664454165844e2f1de452f207ddd27814876d4f35caf9775698c4` だった。

## 今回の変更点

- actual-v12 の封印済み raw trace を byte-for-byte で複製し、新規 `aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v11` と distinct `resident-one-case-smoke-profile-maintenance-offline-reassembly-evidence-v11` を生成した。
- actual fresh output の `aq4-p3-diagnostic-rocprof-capture-v10` と `resident-one-case-smoke-profile-maintenance-evidence-v11` は absent のまま維持した。
- generic `hipMemcpyAsync` 14,084 件を correlation ID で排他的に結合した。memory copy exact-one は 10,845 件（H2D 6,438 件、D2H 4,407 件）、`__amd_rocclr_copyBuffer` kernel exact-one は 3,239 件（D2D）だった。missing、duplicate、ambiguous、overlap、unknown direction、other kernel はすべて 0 件だった。
- raw trace SHA-256 は HIP API `1b782186326bba54e369dc422ce750c7916db3a52226063dc637d583f165b531`、memory copy `2b3faa5a208e97eb983e698d503c1a99368fd12cf0630d2da363834d0aa2b07b`、kernel `936617bf9e855c388baeff379c01f3f6fe58195e1afc339eb822ff62f1d6964f` のまま変更していない。派生レコード SHA-256 は `742665e5491399af0ecf8f9460dc214e32a5e62f3b7c0fd194206d7157781ee4` だった。
- compatibility adapter は、generic API の方向結合と grouped child row の start/end timestamp による安定順序化だけを派生処理として記録した。D2D 推論は sealed actual-v12 限定である。
- offline root は 40 ファイル、evidence root は 2 ファイルだった。両 root は mode `0555`、全 member は mode `0444`・nlink `1` だった。
- offline `SHA256SUMS` は `5ae70242a6943e9ace4e4b64e0e5f0b81eac8d02335114ceb7c3a0b4c330bdbf`、evidence `SHA256SUMS` は `02972beff0ef06a310ad19ad9c7ddf62f69196ac09fa6a231a3e136ccb51f2c7` だった。capture artifact は `ea0c73aa03dc89e173a274790a050a48fb26384990d14f4bbb4b326aa0179e28`、offline evidence は `7b294b6c3d8beb28cb3e9682d852d23cb9e3c91a2433128aa380101c32639a36` だった。
- formal validator は生成時 1 回と commit 前の独立再検証 2 回で合格した。GPU command、rocprof process、workload process、service operation、operator invocation、model load はすべて 0 回だった。

## 次の行動

- ready-v16 / dry-v16 の先行 commit を待ち、この 2 root と本 journal だけを explicit path で artifact commit に保存する。
- current HEAD が artifact commit に進んだ状態で formal validator を再実行し、archive と Git object の byte equality、commit member 範囲、mode、nlink、`SHA256SUMS`、actual fresh output の absent、actual-v12 seal の不変性を再確認する。
- offline reassembly は measurement / promotion の対象にせず、次段階では immutable offline v11 と distinct evidence-v11 を入力 authority として扱う。
