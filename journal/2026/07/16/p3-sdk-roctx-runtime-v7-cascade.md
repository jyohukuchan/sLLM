# P3 SDK ROCTx runtime v7 cascade

## 前回の要点

- launcherはcommit `b0a9dfd1eda2b648c26149f027b7f8c6aed7e10d`でSDK ROCTxをpinした。
- 旧profile actual v6 failureはimmutable evidenceとして保持し、同じruntime/capture outputを再利用しない。

## 今回の変更点

- normal executeは既予約かつ未使用のrun/output/evidence v7を維持した。
- profile run ID、runner output、execute evidence、rocprof captureをfresh v7へ進めた。
- execute-binding rootをfresh `resident-one-case-smoke-execute-binding-v7`へ進めた。
- launcher source/tests authority commitは`60461d796ba64a7f0ba28353cb4f263d08d18dab`、tree `f3e461734923222dde178a75dbc50600689b9737`、launcher blob `98105c77f330f794ebb326d2fb19b70f2a21c2bc`、raw SHA-256 `65b6258cb07a053455c05e65c184a873a3d39c2b2fe1e237970bbd11147dc750`。
- runner、validator、binding B v6、resident binaryのpinsは変更していない。
- execute-binding-v7はnormal execute v7をblocked/QA pendingとして固定し、profile documentはSDK ROCTxとprofile runtime/capture v7を参照する。
- execute-binding JSON SHA-256は`152e35e64bac488c1a9dc0ce7cd821f68baff43076dc152772ac60624aa81266`、launcher-trust JSON SHA-256は`8413f923593d2f6ce2133e33b94fcfa4b096bac290ab6e52d18c81bb38ccfdcd`、SHA256SUMS SHA-256は`a76613511277ef2e8557277374d0e9a1d5b0f4a35a102d301819723c1f40a620`。
- artifact rootは`0555`、3 filesは`0444`かつnlink 1。formal loader、`sha256sum -c`、launcher CPU tests 79件がpassedした。
- old execute-binding-v6、profile runtime/capture v6は`git diff --exit-code`で不変を確認した。
- GPU、production service、actualは実行していない。

## 次の行動

- launcher commitとexecute-binding-v7 artifact commitをauthorityとしてmaintenance-v7へcascadeする。
- maintenance/artifact chainの独立QAが完了するまではactualを実行しない。
