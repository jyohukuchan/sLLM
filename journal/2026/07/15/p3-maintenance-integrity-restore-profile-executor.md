# P3 maintenance integrity / restore / profile executor

## 前回の要点

- maintenance は稼働確認のたびに 7.2 GiB package 全体を再ハッシュしていた。
- 復旧は最大 120 回の相対 sleep で、絶対期限、probe timeout、最終 package tree identity 再確認を持たなかった。
- profile capture は launcher 全体を外側から包み、validator と live gate まで profile 対象に入る構造だった。

## 今回の変更点

- 稼働中 pre-stop で full content hash を厳密に 1 回だけ実施し、その前後で package tree metadata identity を固定した。
- identity は root を含む全 relative path の device / inode / mode / nlink / size / mtime_ns / ctime_ns を対象にし、directory と symlink も列挙する。
- 復旧 poll は軽量な動的 readiness のみにし、120 秒の monotonic absolute deadline、最大 10 秒の probe timeout、1 秒間隔を導入した。
- readiness 成功後に tree metadata を再列挙し、追加、削除、置換、内容変更相当、symlink 化、directory metadata 変更を fail-closed にした。
- explicit `systemctl stop/start` では MainPID と worker PID が変わり、`NRestarts`（自動再起動回数）は不変である契約を evidence 化した。
- launcher commit `8593c38d7ba5739a49b4aedc16a9b6d1e8da2553` の `profile_runner_executor` に接続した。順序は maintenance → launcher → validator / gates → capture → rocprof → runner であり、capture tool の callback は rocprof の `Popen` 成功後だけ runner start を通知する。
- fake のみで 23 秒 hash、1 / 2 / 7 回の一時 health 失敗、永久失敗 120 poll / 120 秒、deadline crossing、6 種の metadata mutation、launcher rc=1 かつ runner 未起動の substrate cleanup を検証した。actual、GPU、service、HTTP は実行していない。

## 次の行動

- ready artifact と QA attestation は、launcher / capture / maintenance の新しい commit、SHA-256、動的 runner target manifest 契約へ再固定する。
- 再固定後に canonical readback / CLI を含む maintenance test 全件を再実行する。
