# SQ8 P8-D non-stream gateway

日付: 2026-07-11

## 前回の要点

P8-Cのstandalone worker正式runは全gateに合格し、commit `051e5bd`で成功raw、stderr、独立validation、checksumを固定した。次はOpenWebUI接続の前段として、batchなし・active1/waiting0の非stream OpenAI gatewayを実装する段階だった。

## 今回の変更点

- `services/openai-gateway/`にPython 3.12専用packageと`uv.lock`を追加した。productとdevの直接・間接依存は固定し、vLLM環境から分離した。
- frozen Qwen3 tokenizerの5 file hash、Transformers `5.12.1`、tokenizer class、chat template hashを起動時に検証し、local-only/offlineで全履歴へ`add_generation_prompt=true`、`enable_thinking=false`を適用する。
- strict JSON、2 MiB上限、Bearer auth、role/content parts、null/unknown policy、sampling、4096 context予約、OpenAI error/usage shapeを実装した。P8-E前の`stream=true`は暗黙変換せず400にする。
- GPU singleton lockを保持してRust workerを1 processだけ起動し、Ready identity、M=128 progress、token/EOS、releaseを独立検証する。active slotはmatching `released(reset_complete=true)`だけで解放する。
- startup 600秒、request 180秒、no-progress 30秒、cancel-release 5秒、TERM 2秒後のKILL、fatal時のnonzero gateway exitを実装した。
- 独立レビューで見つかったevent-loop blocking、cancel/terminal race、shutdown例外時のworker残留、EOS後token、巨大JSON数値、progress cadence、trailing slash、bind範囲、access logの秘密漏えいを修正した。
- fake process、実tokenizer fixture、実OpenWebUI nonstream fixtureを含む85 tests、strict mypy、ruff、compile、lock check、dependency checkが合格した。

## 次の行動

1. P8-Eでstable incremental decodeとSSE chunk列を実装する。
2. disconnect/slow clientを既存のbounded cancel-releaseへ接続する。
3. 実stream/stop/recoveryを確認してからP8-Fのsystemd/OpenWebUI接続へ進む。

## R9700 HTTP acceptance

- source commit: `9b977d98ced759a43d076d673e346ab4e74202cf`
- worker SHA-256: `145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950`
- Ready: 2秒pollの13回目で200、`/v1/models`も200
- gateway PID/starttime: `4030912 / 99154355`
- worker PID/starttime: `4031291 / 99154532`、逐次request間で不変
- 日本語stop: 1.475147秒、`東京`、prompt 39 / completion 3
- 2件目: 3.357415秒、completion 64、処理中ready 31/31回が200
- collision: 0.001814秒で429、`Retry-After: 1`、先行requestは200
- collision後: 0.648689秒、`こんにちは！`、同一workerで200
- shutdown: gateway/worker/listener/R9700 processなし、lock再取得成功
- repo evidence: `benchmarks/results/2026-07-11/sq8-p8d-http-smoke-v0.1/summary.json`

P8-Dは完了した。非stream OpenAI API、常駐worker再利用、待ち行列なし、fatal/cleanup境界が実機で成立した。OpenWebUIの通常利用に必要なstreamingはP8-Eで追加する。
