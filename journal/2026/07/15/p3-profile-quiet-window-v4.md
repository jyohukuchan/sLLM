# P3 profile quiet-window v4

## 前回の要点

- v3 は runtime、formal health、lock、AMD/KFD owner、端末集合が安定していたが、監視中に HEAD/tree が変化し、first-to-last sample span も 120 秒未満だったため NO-GO とした。

## 今回の変更点

- fresh path `resident-one-case-smoke-profile-quiet-window-v4` に、5 秒間隔で 27 点を取得した。
- sample 0 から sample 26 までの monotonic span は `130.975764708` 秒であり、130 秒以上の時間契約を満たした。
- runtime、formal health、lock、AMD/KFD owner、全 pts の process set は全標本で安定し、外部の systemctl、maintenance、profile capture、rocprof、GPU probe 対象プロセスは 0 件だった。
- サービスは MainPID `2634680`、worker PID `2635236`、NRestarts `0`、lock inode `772895` を維持した。actual、GPU command、service 操作は実行していない。
- 開始 HEAD/tree は `2e005c49000b1222b124fa4e89275f73b7ec669c` / `66afc0c41894b8481101eb19b948c69999e4b659`、終了 HEAD/tree は `446e50429da0c78acbc4ce67637ab04ba8297a4c` / `643f1e2868771eb1ba3ba98d44ba539a5717aca9` だった。
- 監視中に `2b211a3a4f58fde4e81bad92a7c4898c673eaca6`（P3 profile trust artifacts の再生成）と `446e50429da0c78acbc4ce67637ab04ba8297a4c`（canonical execute binding trust の更新）が入り、関連 worktree aggregate も `261a162d...` から `53b96ab7...` へ変化した。
- 契約どおり、HEAD/tree または関連ファイルの変化を 1 件でも許容しないため、判定は **NO-GO** とした。違反は sample 1〜26 の identity change 26 件、HEAD/tree change 1 件、relevant worktree change 1 件の計 28 件である。
- 固定 Git blob は runner `7adc7f9258f491c29a8f2d7842d5ece20488867f`、capture `9c0ae790011a614d6d44c39e33e5911f2ff358d3`、launcher `e9f6a11131faeb7d766b85e5ef512839f4071be9`、validator `13de5f3d2b96ef1936356e04c12773d674fa4488`、maintenance `32725371678755ba09e0e5686fbed8a371269034` として evidence に記録した。
- `quiet-window.json` の SHA-256 は `de3b8817ff7fa53bd3f771bf6b603479fe72b282bd5e3499e5f361c997b0a914`、`SHA256SUMS` の SHA-256 は `5faedce1466093d3c441080423c96966db302d245878480dbe2da7032fbe12e0` である。成果物は 0444、ディレクトリは 0555 で固定し、`sha256sum -c SHA256SUMS` は成功した。

## 次の行動

- この NO-GO を保持し、他作業のコミットと関連 trust artifact 更新が止まる時間帯を確保してから、別の fresh path で quiet-window を再実施する。
