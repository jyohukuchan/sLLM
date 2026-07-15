# P3 SDK ROCTx launcher pin

## 前回の要点

- profile actual v7はrunner完了後に`ready_candidate_marker_absent`でfail-closedした。
- 通常リンクした旧`libroctx64.so.4`ではmarkerを取得できたため、`/proc/self/fd/N`からの遅延ロード経路をA/B比較した。

## 今回の変更点

- `/tmp/ullm-roctx-pinned-fd-ab-v1`で、実runnerと同じ`ctypes.CDLL('/proc/self/fd/3', RTLD_NOW | RTLD_LOCAL)`とdirect `roctxRangePushA/Pop`を使用した。
- Aの旧compat library `/opt/rocm-7.2.1/lib/libroctx64.so.4.1.70201`、SHA-256 `22bbc6946fdf5d7d8b1755cbd738c42a63f3795d18ac3ed1285b09cc772dee17`は、push/popとtargetが成功してもmarker fileが0件だった。
- BのSDK library `/opt/rocm-7.2.1/lib/librocprofiler-sdk-roctx.so.1.1.0`、SHA-256 `1a5831a3817eac29f63d1442dc348ba31b417202b7ce15f3aed9c09a8f4773c9`は、同一経路でmarker file 1件、range 1行を取得した。
- launcherのROCTx path/hash authorityをBへ変更した。canonical pathとrealpathは同一、SONAMEは`librocprofiler-sdk-roctx.so.1`、modeは`0644`、sizeは`456232` bytes。
- generic runner CLIの構造は変更せず、既存の`--profile-roctx-ranges --roctx-library PATH --roctx-library-sha256 SHA`のPATH/SHAだけを更新した。
- execute-binding-v6は旧launcher selfをpinしたままなので、新launcherに対してfail-closedする。binding/maintenance/artifactのcascadeはこの作業には含めない。
- A/Bは各1回だけで、GPU workload、production service操作、actual再実行は行っていない。

## 次の行動

- このlauncher commitをauthorityとして、execute binding、maintenance、ready artifactを別レーンで順にcascadeする。
- cascade前は既存execute-binding-v6をactualに使用しない。
