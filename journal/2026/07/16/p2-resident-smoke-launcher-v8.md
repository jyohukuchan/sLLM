# P2 resident smoke launcher v8

## 前回の要点

- actual-v8はcapture parserのembedded `device_lock` / `live_preflight`契約不一致によりfailure evidenceとしてcommit `4b651cd5c46212349b5a598b344da6ea11993d30`へ封印した。retryは禁止されたままである。
- 最終producer authorityはcommit `dac045244d7609c42c2db1ea0f91aa707ffb717b`、tree `c8138c2be5c54693e5c63140b9832f7e1c95f623`、blob `b838d92198f6eb69460ab40990aea893ec19d7ac`、raw SHA-256 `d0360a494f30c2bbac7ca1d043385dd6de9384fa2d81ab99881e54afeaaed934`で確定した。

## 今回の変更点

- launcherのproducer helper pinをfinal producer raw SHA-256へ更新した。
- execute-binding、通常execute run/output/evidence、profile run/output/evidence/captureをfresh v8 namespaceへ進めた。SDK ROCTx、runner、B、resident、validator等の既存pinsは変更していない。
- launcher source/test commitは`b81066dbf86857afbeb0dc7d41493fdef680266d`、tree `ba44559a4778504eaef37dc2cf4d052076fab838`、launcher blob `a9f0498d9dc51b276addda0410560b2d8e696859`、raw SHA-256 `bcd25ffa719e04d8535560ec179506f1bcae2ede417023d3d6303c864dadb5e3`。
- launcher CPU testsはsource commit前とartifact生成後に実行し、各79/79 passed。旧execute-binding-v7はhistorical sealed readbackとして検証した。
- fresh execute-binding-v8を1回生成し、status `blocked_pending_live_preflight_and_qa`、actual eligible false、launcher trust status `qa_pending`をformal loaderで確認した。
- execute-binding JSON SHA-256は`600d1dc231220303808eb8559ca09bd388cee386015374d077f999e58fb6fcc0`、launcher trust JSON SHA-256は`e33744c2db860c487fc22df87260f742604eeee8c4caaea7dba285a35409ed99`、SUMS SHA-256は`cb369d7eeab3aae6ab8370f956f8df564a1b3f8ac9ef6415c6d307d2c5ab7ca6`。
- artifact rootは`0555`、全filesは`0444`かつnlink 1で、`SHA256SUMS`がpassedした。
- 旧execute-binding-v7とprofile actual-v8 failure evidenceは不変。GPU、service、actualは実行していない。

## 次の行動

- launcher-v8 source authorityとexecute-binding-v8 artifactをmaintenance側へ渡し、capture-v8 final authorityとともにrecascadeする。
- fresh maintenance/ready chainと独立GOが確定するまではGPU、service、actualを実行しない。
