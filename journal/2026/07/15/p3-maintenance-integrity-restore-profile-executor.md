# P3 maintenance integrity / restore / profile executor

## 前回の要点

- maintenance は稼働確認のたびに 7.2 GiB package 全体を再ハッシュしていた。
- 復旧は最大 120 回の相対 sleep で、絶対期限、probe timeout、最終 package tree identity 再確認を持たなかった。
- profile capture は launcher 全体を外側から包み、validator と live gate まで profile 対象に入る構造だった。

## 今回の変更点

- 稼働中 pre-stop で full content hash を厳密に 1 回だけ実施し、その前後で package tree metadata identity を固定した。
- identity は root を含む全 relative path の device / inode / mode / nlink / size / mtime_ns / ctime_ns を対象にし、directory と symlink も列挙する。
- full content hash と tree metadata identity、entry / regular / directory / symlink / special の各件数、総 bytes を一つの trusted package integrity identity に束ねた。ready binding が固定した基準値と pre-stop で一致しない stable な追加、空 directory、special file、symlink、metadata 差異も service stop 前に拒否する。
- 復旧 poll は軽量な動的 readiness のみにし、120 秒の monotonic absolute deadline、最大 10 秒の probe timeout、1 秒間隔を導入した。
- readiness 成功後に tree metadata を再列挙し、追加、削除、置換、内容変更相当、symlink 化、directory metadata 変更を fail-closed にした。
- explicit `systemctl stop/start` では MainPID と worker PID が変わり、`NRestarts`（自動再起動回数）は不変である契約を evidence 化した。
- launcher / capture commit `48cce1349eae0b58beac2851a05e40b2d522559e` の `profile_runner_executor` に接続した。順序は maintenance → launcher → validator / gates → capture → rocprof → runner であり、capture tool の callback は rocprof の `Popen` 成功後と runner 完了後を別々に通知する。
- profile capture tool は trust guard が検証した bytes を直接 compile / exec し、path を再度 open しない。success artifact と failure evidence は mode 0444、self-hash、exact nested schema、resident / target / profiler / environment / logical command の結合を検証する。
- artifact 内の全 file reference は canonical absolute path、`..` 不在、symlink component 不在、single-link regular file、非実行 mode、128 MiB 上限、streaming SHA-256 実 bytes 一致を要求する。外部 path、dotdot、symlink、hardlink、mode、hash の差し替えを self-hash 再生成後も拒否する。
- failure evidence は logical command と FD 化された effective command の digest を分離し、process-group cleanup と `children_state_known` / `children_remaining` を整合検証する。PID 一覧を持たない cleanup failure は unknown / empty placeholder とし、lock substrate は cleanup passed かつ known かつ empty の場合だけ unlink する。
- fake のみで 23 秒 hash、1 / 2 / 7 回の一時 health 失敗、永久失敗 120 poll / 120 秒、deadline crossing、6 種の metadata mutation、launcher rc=1 かつ runner 未起動の substrate cleanup を検証した。actual、GPU、service、HTTP は実行していない。
- actual capture module の `main` は subprocess を monkeypatch した unit integration として success、cleanup 成功 failure、children unknown failure、executor exception を検証した。GPU、service、HTTP は実行していない。

## 次の行動

- maintenance commit 後に ready artifact と QA attestation を maintenance の新しい commit / blob / SHA-256 へ再固定する。
- artifact 再生成担当が canonical readback / CLI を含む最終統合 test を実行する。
