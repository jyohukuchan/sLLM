# P3 profile offline reassembly v10

## 前回の要点

- actual-v12 は commit `44617f7fd46c39f71f04502b248739cc116fe095`、tree `813c4ffc88fb58cf8764b91d3c80cea9ef351f0f` の 35 ファイルで封印済みだった。
- 最初の offline artifact commit `04e1886cfee45052329b8e1fd78af726d9fe9766` は、validator が generator identity を current HEAD と比較していたため、artifact commit 後の formal validation に失敗した。この invalid commit は通常の revert commit `cd54ac8824e3f131e941e89b7eab2e11f903a20c` で取り消し、tree を ready artifact commit と同じ `4daa8f0cafe93274aeddd902bea58727633b3080` に戻した。
- definitive generator / validator authority は commit `dabb093074eb42bd7e117dbe83e9eea9a02f5f71`、tree `654aa2abcc8ef595f66414ff81b74fd7ddedf386`、tool blob `e3083bae2c655531128c2673867c88fc95f5e7b5`、raw SHA-256 `309c3fe4e34e1c85aa9b87f6659e605532061359ed6cbc13be7221e150607e30` だった。generator identity は source path の last-change authority を実読して固定する。

## 今回の変更点

- actual-v12 の封印済み raw trace を byte-for-byte で複製し、fresh distinct namespace `aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v10` と `resident-one-case-smoke-profile-maintenance-evidence-v11` を再生成した。canonical actual namespace `aq4-p3-diagnostic-rocprof-capture-v10` は absent のまま維持した。
- generic `hipMemcpyAsync` 14,084 件を correlation ID で排他的に結合した。memory copy exact-one は 10,845 件（H2D 6,438 件、D2H 4,407 件）、`__amd_rocclr_copyBuffer` kernel exact-one は 3,239 件（D2D）だった。missing、duplicate、ambiguous、overlap、unknown direction、other kernel はすべて 0 件だった。
- raw trace SHA-256 は HIP API `1b782186326bba54e369dc422ce750c7916db3a52226063dc637d583f165b531`、memory copy `2b3faa5a208e97eb983e698d503c1a99368fd12cf0630d2da363834d0aa2b07b`、kernel `936617bf9e855c388baeff379c01f3f6fe58195e1afc339eb822ff62f1d6964f` のまま変更していない。派生レコード SHA-256 は `742665e5491399af0ecf8f9460dc214e32a5e62f3b7c0fd194206d7157781ee4` だった。
- compatibility adapter は、generic API の方向結合と grouped child row の start/end timestamp による安定順序化だけを派生処理として記録した。D2D 推論は sealed actual-v12 限定である。
- offline root は 40 ファイル、evidence root は 2 ファイルだった。両 root は mode `0555`、全 member は mode `0444`・nlink `1` だった。
- offline `SHA256SUMS` は `6f061ff1fb2531642d593ca14a6b685588730302863fdc128a38dc5e9dcd013a`、evidence `SHA256SUMS` は `cde92e68ee3e4ced84cf10f50d0aa7579d87d4e4e65dfa7bc5cebafbf0f81849` だった。capture artifact は `9d989d3a671d96b7fc6050a012d2b3c4f77b46c8b03d5348e1af8cdf6fa0b5ff`、offline reassembly evidence は `89417d2f030f801e4d1353eb37f74f2c8592c9c306ccea1ec7bd01386bd73189` だった。
- formal validator は生成時 1 回と commit 前の独立再検証 2 回で合格した。GPU command、rocprof process、workload process、service operation、operator invocation、model load はすべて 0 回だった。

## 次の行動

- この 2 root と本 journal だけを artifact commit に保存し、current HEAD が artifact commit に進んだ状態でも formal validator が合格することを必須確認する。
- archive と Git object の byte equality、commit member 範囲、mode、nlink、`SHA256SUMS`、canonical actual root の absent、old seal の不変性を再確認する。
- offline reassembly は measurement / promotion の対象にせず、次の operator 段階ではこの immutable offline root と evidence-v11 を入力 authority として扱う。
