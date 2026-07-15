# AQ4 P3 capture final producer pin

## 前回の要点

- producerはcommit `dac045244d7609c42c2db1ea0f91aa707ffb717b`、tree `c8138c2be5c54693e5c63140b9832f7e1c95f623`、blob `b838d92198f6eb69460ab40990aea893ec19d7ac`で確定した。
- producer SHA-256は`d0360a494f30c2bbac7ca1d043385dd6de9384fa2d81ab99881e54afeaaed934`。封印済みactual-v8のdevice lockとlive preflightを含むfull resident pairを受理し、negative matrixはfail-closeになった。

## 今回の変更点

- captureのproducer pinを最終SHAへ更新し、commit/tree/blob authorityを定数として固定した。
- PinnedPythonHelperが読み込むworktree bytes、authority commitのbytes、authority blob、current HEAD placementが同じproducerを指すことをGit objectとSHA-256の両方で検証するCPU testを追加した。
- capture fixtureをdiagnostic current contractのembedded live preflightへ更新した。legacy ref-only互換は追加していない。
- captureが実際にpinして読み込んだproducerで、封印済みactual-v8 raw SHA-256 `397f02a2cd87e5d30eb9eb569b5d022351b1f994358e71535f2ce697af5df25c`、summary SHA-256 `b82409bf997e207df5576ba7e38ebefddff363440c256250ffc8f7b521dcb3f5`、12 runs、device lock、raw/summary live preflight一致をfull validateした。
- capture全58件中57件はpassed、1件は既存の条件付きskip。`py_compile`と`git diff --check`もpassedした。
- capture source SHA-256は`d0d7093e2fe8575c1105432cabf801c04b3deee8b6772d792382486116657527`。GPU、service、actualは実行していない。

## 次の行動

- launcher/maintenance側はそれぞれの所有レーンでcapture source SHAとproducer authorityを再連鎖する。
- actual-v8は再実行せず、immutable failure evidenceとして保持する。
