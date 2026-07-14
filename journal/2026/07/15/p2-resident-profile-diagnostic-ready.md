# P2 resident profile diagnostic ready

## 前回の要点

base one-case chainはreadyだったが、runner `e93a2c1`のROCTx markerとP3 rocprof captureはlauncher/harness/artifactへ未固定だった。

## 今回の変更点

- runner→validator→B→launcher→harnessの順に再固定した。
- 通常one-case argvを変えず、profile modeだけROCTx libraryと12 rangeをexact透過・検証する。
- capture tool `489183a`がrocprofv3経由でmaintenance harnessを包むexact commandをprofile-readyへ固定した。
- base readyはactual one-case最大1回、profile-readyはdiagnostic actual最大1回で、両方ともpromotion不可・output no-reuseである。
- 主要回帰155、marker 55、capture 8 testsが通過した。手動marker境界15件も通過済みである。
- base/profile canonical dry-runは全actual process count 0である。
- actual service停止、GPU command、model load、rocprof captureは実行していない。

## 次の行動

actual実行は別の明示承認まで行わない。profile時はartifact内のexact capture commandとfresh outputだけを使用する。
