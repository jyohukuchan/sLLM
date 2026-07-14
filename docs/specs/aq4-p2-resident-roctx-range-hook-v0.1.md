# AQ4 P2 resident ROCTx run-range hook v0.1

## 前回の要点

P3 diagnostic captureは、同じrocprof sessionのexact 12 run rangeを使って2 warmupを除外し、10 measured traceを作る。従来runnerにはrun境界markerがなく、capture toolはmarker欠落をfail-closedとしていた。

## 今回の変更点

`tools/run-aq4-p2-resident-batch.py`へ明示的な`--profile-roctx-ranges`を追加した。flagがない通常runではROCTx libraryをloadせず、marker名の追加検証も行わない。flagはactual `--one-case-smoke`だけで使用でき、dry-run、84-case run、library optionだけの指定を拒否する。

```text
--profile-roctx-ranges
--roctx-library ABSOLUTE_INVOCATION_PATH
--roctx-library-sha256 EXPECTED_RESOLVED_FILE_SHA256
```

## library trust

invocation pathはabsoluteで`..`を含んではならない。全path componentを`lstat`し、ancestorとleafのsymlink chainについてpath、inode metadata、link targetを固定する。`resolve(strict=true)`後の実体はsingle-link regular fileとしてFD/path identityとSHA-256を検証する。expected SHAと異なる実体をloadしない。

resolved実体は`O_NOFOLLOW`でopenし、同じFDからSHAを計算する。`ctypes.CDLL`は`/proc/self/fd/<fd>`を`RTLD_NOW|RTLD_LOCAL`でloadし、path差し替え後の別DSO constructorを実行しない。`roctxRangePushA`と`roctxRangePop`の両symbolを必須とし、引数/戻り値型を固定する。load直後とsidecar発行直前にsymlink chain、resolved target、inode、SHAを再検証する。missing library、missing symbol、symlink差し替え、SHA差し替えはresident processを起動する前またはartifact発行前に拒否する。

## range契約

rangeはrunnerの同一PID・同一threadでだけ操作する。active rangeを一つに限定し、nested begin、activeなしpop、index飛び、kind違いを拒否する。

marker名は次のexact 1行である。

```text
ullm.aq4_p2.run.v1/run_id=<run_id>/session_id=<resident_session_id>/case_id=<case_id>/case_sha256=<64hex>/run_index=<0..11>/run_kind=<warmup|measured>
```

index 0,1は`warmup`、2..11は`measured`である。beginは`command=run`をresident stdinへ送る直前である。endは`_recv`結果を`validate_run`で検証し、run index/kind一致を確認した直後である。送信、受信、validation、OOM、timeoutのどこで例外が起きても`finally`でpopする。pop完了前に次beginを行わない。

成功sidecar `resident-batch.roctx-ranges.json`はexact 12 range、name/index/kind、push/pop結果、PID/thread、library invocation/resolved path、symlink chain、SHA、symbolsを保存する。self-hash fieldは`audit_sha256`である。sidecarは常に`measurement_eligible=false`、`promotion_eligible=false`で、profile overheadを含むlatencyのpromotion利用を認めない。不完全、unbalanced、12件未満ではsidecarとsuccessful summaryを発行しない。

## trust chainの後続更新

runner source変更後、live使用前に次を再pinする必要がある。

- prepared rootの`trusted-runner.py`
- `bundle.json`、`SHA256SUMS`、`trust-roots.json`
- B sidecarとrunner/validator SHA定数
- `tools/launch-aq4-p2-resident-smoke.py`のrunner/hash固定値
- execute binding、launcher trust、maintenance harnessのcanonical artifact

これらが新commitへ更新されるまで、既存launcherは変更runnerをactual trust chainとして受理しない。

## 次の行動

別作業でprepared bundle、validator、B、launcher、harnessのhashを再生成し、通常dry-run、fake marker、live-preflight gateの独立QAを行う。その後も明示された保守窓まではGPU/serviceを操作しない。
