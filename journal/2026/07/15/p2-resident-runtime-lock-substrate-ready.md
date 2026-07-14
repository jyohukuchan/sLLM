# P2 resident runtime lock substrate ready

## 前回の要点

actual v3 は KFD 停止確認を通過した後、`RuntimeDirectory=ullm` が service stop 時に `/run/ullm` を削除したため、停止 poll の lock 観測で失敗した。launcher、runner、binding、ready artifact は service 稼働中に存在する lock file を前提としていた。

## 今回の変更点

- service stop 後に `/run/ullm` の不在を確認し、SHA-256 固定済み `/usr/bin/install` を `sudo -n` で実行して、owner `homelab1`、mode `0750` の空 directory を再構築するようにした。lock は非 root で `O_CREAT|O_EXCL|O_NOFOLLOW`、mode `0600` で作成し、directory/file の owner、mode、nlink、device/inode と固定 argv を evidence に残す。
- 停止 poll は作成した substrate と同じ device/inode の lock だけを stable 候補にする。launcher の live preflight から runner まで同じ lock inode を拘束し、one-case actual の runner は lock を新規作成しない。
- outer finally は runner child と lock holder が 0 であることを確認し、同じ inode だけを unlink する。SHA-256 固定済み `/usr/bin/rmdir` で空 directory を削除し、cleanup が失敗しても service start と復旧検証を必ず試行する。cleanup failure 自体は run failure として保持する。
- KFD owner scanner は PID/queue/gpuid follow-up の ENOENT だけを bounded rescan し、EACCES、I/O error、不正 schema、symlink、identity 変更、許可集合外 owner を fail-closed にした。raw gpuid は保存せず、SHA-256、byte 数、source classification、root identity、retry diagnostic を evidence に残す。停止中は旧 worker PID、復旧後は新 worker PID だけを許可する。
- runner `c0480d1d`、validator pin `82c77957`、binding-v4 `9a98c67b`、launcher `180ab1be`、execute binding `0b88e5f8`、maintenance `9a3de269`/`c4c3d9f3`、base/profile ready と両 dry-run `0bb6d016` へ依存順に固定し直した。
- core trust-chain 198 tests、diagnostic capture 11 tests、capture 関連 93 testsが通過した。全6 artifact の `SHA256SUMS` と4 Python source の `py_compile` も通過し、両 canonical dry-run の全 process count は 0 だった。
- actual、sudo、service stop/start、GPU command、HTTP probe、model load、profiler capture は実行していない。unit test の旧 monkeypatch 名を見落とした初回回帰1件だけが KFD proc を読み、schema failure で終了したため、詳細 snapshot fake に修正して再発を防いだ。

## 次の行動

この ready chain を使う actual 再試行はまだ行わない。君が別途明示した場合だけ、fresh output、不在の `/run/ullm`、固定 tool SHA、同一 inode、stable 2、外側復旧を実行直前に再確認して一回だけ実行する。
