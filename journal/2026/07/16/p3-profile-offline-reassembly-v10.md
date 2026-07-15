# P3 profile offline reassembly v10

## 前回の要点

- actual-v12 は commit `44617f7fd46c39f71f04502b248739cc116fe095`、tree `813c4ffc88fb58cf8764b91d3c80cea9ef351f0f` の 35 ファイルで封印済みだった。
- actual-v12 の capture-v9 は、generic `hipMemcpyAsync` を未知の転送 API として拒否したため、実測処理は retry 禁止のまま offline reassembly が必要だった。
- 最終 generator / validator authority は commit `2167c33fe56c0efcbd3745055e6de8604aafd456`、tree `b76cdd6937d3f5f63565049596d8192ed6f87cd2`、tool blob `cf4fedca1912cc6cbe54ffbd63456c3ff1dbba53`、raw SHA-256 `f86f5be10968eab00f1fabae7827cd557514437098545049ac82def2ddbf2f0c` だった。

## 今回の変更点

- actual-v12 の封印済み raw trace を byte-for-byte で複製し、fresh distinct namespace `aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v10` と `resident-one-case-smoke-profile-maintenance-evidence-v11` を生成した。canonical actual namespace `aq4-p3-diagnostic-rocprof-capture-v10` は absent のまま維持した。
- generic `hipMemcpyAsync` 14,084 件を correlation ID で排他的に結合した。memory copy exact-one は 10,845 件（H2D 6,438 件、D2H 4,407 件）、`__amd_rocclr_copyBuffer` kernel exact-one は 3,239 件（D2D）だった。missing、duplicate、ambiguous、overlap、unknown direction、other kernel はすべて 0 件だった。
- raw trace SHA-256 は HIP API `1b782186326bba54e369dc422ce750c7916db3a52226063dc637d583f165b531`、memory copy `2b3faa5a208e97eb983e698d503c1a99368fd12cf0630d2da363834d0aa2b07b`、kernel `936617bf9e855c388baeff379c01f3f6fe58195e1afc339eb822ff62f1d6964f` のまま変更していない。派生レコード SHA-256 は `742665e5491399af0ecf8f9460dc214e32a5e62f3b7c0fd194206d7157781ee4` だった。
- compatibility adapter は、generic API の方向結合と grouped child row の start/end timestamp による安定順序化だけを派生処理として記録した。D2D 推論は sealed actual-v12 限定である。
- offline root は 40 ファイル、evidence root は 2 ファイルだった。両 root は mode `0555`、全 member は mode `0444`・nlink `1` だった。
- offline `SHA256SUMS` は `6f061ff1fb2531642d593ca14a6b685588730302863fdc128a38dc5e9dcd013a`、evidence `SHA256SUMS` は `133a03254ddedd493ee65b25c7cfbf505cc2d9dc21e628c2633a369e98ece089` だった。capture artifact は `9d989d3a671d96b7fc6050a012d2b3c4f77b46c8b03d5348e1af8cdf6fa0b5ff`、offline reassembly evidence は `6548386329bfbe33d770e21f24ebcc5aed7c90af8f0000ef1e0a899663bcf63c` だった。
- formal validator は生成時 1 回と独立再検証 2 回の計 3 回で合格し、両 root の `SHA256SUMS` 完全被覆も合格した。GPU command、rocprof process、workload process、service operation、operator invocation、model load はすべて 0 回だった。

## 次の行動

- この 2 root と本 journal だけを artifact commit に保存し、archive と Git object の byte equality、commit member 範囲、mode、nlink、`SHA256SUMS`、formal validator、canonical actual root の absent を再確認する。
- offline reassembly は measurement / promotion の対象にせず、次の ready / operator 段階ではこの immutable offline root と evidence-v11 を入力 authority として扱う。
