# P2 resident one-case invocation NO-GO

## 前回の要点

container health の実機件数を `docker=9`、`docker exec=6`、container curl total `6`、version `1`、endpoint `5` として契約へ反映し、HEAD `da5dd372` の ready artifact で base one-case actual を一度だけ実行する予定だった。

## 今回の変更点

read-only preflight は合格した。ready artifact の checksum と mode 0444、harness/launcher の固定内容、対象パス clean、三つの出力先未使用、service/worker/NRestarts、manifest/worker/package hash、GPU identity/単一 owner/lock、container namespace の全 health、RAM と disk を確認した。

同一 PTY で sudo cache を prime した後、maintenance harness を一度起動した。しかし `--ready-artifact` に相対パスを渡したため、immutable harness の最初の path identity 検査が `ready artifact path differs` で失敗した。service stop、launcher、runner、GPU command、model load、復旧 start は全て 0 であり、maintenance/launcher/runner の出力先は作成されていない。再試行禁止に従い、absolute path での再起動は行わなかった。sudo cache は `sudo -k` で明示的に破棄した。

post-check では production は preflight と同じ MainPID `3090367`、worker PID `3090924`、NRestarts `0`、active/running、hash 不変、対象 GPU の単一 owner と lock 保持、container health と全 endpoint HTTP 200 を維持した。

構造化 evidence は `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-invocation-no-go-v1/diagnosis.json` に保存した。

## 次の行動

この試行は NO-GO として終了する。次回の明示承認がある場合は、ready artifact と evidence output の両方を固定済み absolute path で指定した exact command を事前に文字列照合し、新規出力へ一度だけ実行する。
