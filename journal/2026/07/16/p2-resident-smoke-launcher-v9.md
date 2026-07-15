# P2 resident smoke launcher v9

## 前回の要点

- actual-v9はcapture parserのunknown kernel family `__amd_rocclr_fillBufferAligned`によりfailure evidenceとしてcommit `00358807d7f400d621c11e20b942ecd4fbbd656f`へ封印した。retryは禁止されたままである。
- 旧launcher-v8 sourceはcommit `b81066dbf86857afbeb0dc7d41493fdef680266d`、execute-binding-v8はcommit `ee7333cdbc1da23f24295fe6d32462feebc6467f`で確定している。

## 今回の変更点

- family authorityをcommit `e4f8583a0fc710d2146f70d06b8b49eb42f04a16`、tree `be5ac39ea05b0b79223d974487c6cddda8d84f0c`、blob `8c318849838f85cf2f2a687aef260506bfa4097c`、raw SHA-256 `f8d32c340231e329f004d9e16192c02378f1fd58b8ab713e8efbbd3029b052d6`へ固定した。
- producer authorityをcommit `c8becac66551f216de47d0cd935929afe60b3b96`、tree `088ac662dc686741d3affafe9b4ecc58cccea638`、blob `b070361d992fddc5749dba677ecd9d81f4ac6c06`、raw SHA-256 `a589c3e644d36132fb6054afdb15b27543d8e8181e3c737dcbd071d7c52e3d20`へ固定した。
- execute-binding、通常execute run/output/evidence、profile run/output/evidence/captureをfresh v9 namespaceへ進め、通常・profile run IDもv9へ更新した。SDK ROCTx、selector、runner、B、resident、validator等の既存pinsは変更していない。
- launcher source/test commitは`7f961f8de75ccbb1080fcd35a5b274584d4e00f3`、tree `795061e58dba439b89eb51e904af1b4d8793792d`、launcher blob `1350c80360a366f6e517aaae083292b8aa990654`、raw SHA-256 `8f1dd30ebd39a9db6cf0bb31d5e15d6474648cfeb5842ca02a8a360f6e414bb8`。
- launcher CPU testsはsource commit前とartifact生成後に実行し、各79/79 passed。旧execute-binding-v8はhistorical sealed readbackとして検証した。
- fresh execute-binding-v9を1回生成し、status `blocked_pending_live_preflight_and_qa`、actual eligible false、launcher trust status `qa_pending`をformal loaderで確認した。
- execute-binding JSON SHA-256は`18663a22731a99bbb54a03d42232803c8636a6e650c6189f791b87faede50042`、launcher trust JSON SHA-256は`6ba6f5862243131f5edb1728e20f8661d89d86c0e1ad1507da36d4504c6253b3`、SUMS SHA-256は`b5764d5bdbcfd2fcd56e602c88081796913429fe6137711e20c249bc524a1532`。
- artifact rootは`0555`、全filesは`0444`かつnlink 1で、`SHA256SUMS`がpassedした。旧execute-binding-v8とprofile actual-v9 failure evidenceは不変。
- production serviceはread-only監視だけを行い、GPU、service、actualは実行していない。

## 次の行動

- launcher-v9 source authorityとexecute-binding-v9 artifactをmaintenance側へ渡し、capture-v9 final authorityとともにrecascadeする。
- fresh maintenance/ready/operator chainと独立GOが確定するまではGPU、service、actualを実行しない。
