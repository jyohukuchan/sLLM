# SQ8 P8-E SSE streaming gateway

日付: 2026-07-11

## 前回の要点

P8-Dでは、offline Qwen3 tokenizer、resident Rust worker supervisor、strictなOpenAI Chat Completions subset、非stream応答を実装した。R9700上で日本語応答、同一worker再利用、active中ready、衝突429、終了時のGPU/lock回収まで確認済みだった。P8-EにはSSE、incremental decode、切断とslow-clientの安全なcancelが残っていた。

## 今回の変更点

- `StableIncrementalDecoder`を追加し、日本語、emoji、combining character、code fenceをtoken単位で復元して、確定前のU+FFFDと既送信prefix変更を拒否する。
- OpenAI互換SSEをrole、非空content、finish reason、任意usage、`[DONE]`の固定順で実装した。streamとnonstreamは同じtoken列から同一本文を返す。
- worker pipe pumpはclientを待たず、32 tokenのbounded queueへnonblocking enqueueする。最初のoverflowで`slow_client` cancelを固定し、以後のtokenを捨てながらreleaseまでpipeをdrainする。
- ASGI 2.3 disconnect、blocked send、5秒send timeout、header commit前後のfatal、`[DONE]`送信中fatalを個別に扱う専用StreamingResponseを追加した。cleanupはAnyIO cancel scopeでshieldし、fatal error試行は200msに制限した。
- prompt用とstream decode用のtokenizer実体、executor、lockを分離し、同時requestのcontext検証がstream decodeを止めないようにした。
- validationとcontext判定の後にHTTP lifecycle gateを取得し、matching worker releaseだけでなく、JSON/SSEのterminal出力またはclose完了まで保持する。これにより旧responseのpost-release fatal ackが次世代requestを追い越さない。
- gateway起因fatalはtask schedule前に同期poisonし、readinessと次admissionを即時拒否する。fatal後は通常cancel/releaseを待たず、bounded error、transport close、TERM/KILLへ進む。
- 独立レビューで見つかった成功streamのglobal ack汚染、実ASGI切断のcleanup再cancel、blocked transport残留、header前fatalの200化、release後EOF、tokenizer競合、DONE後error、close前ack、同期poison、slow-client理由競合、decoder fatal時cancel hang、世代交差、post-claim error gate抜けを修正した。
- packageは114 tests、strict mypy、Ruff check/format、lock checkに合格し、最終独立レビューはP0/P1なしとなった。コードcommitは`3dcd1cb24ef7102abd0eecfd42ca47e47dc0d202`。

## R9700 acceptance

- Readyは2秒pollの17回目、model listは200。
- gateway PID/starttimeは`125204 / 99587014`、workerは`125590 / 99587237`。
- 日本語SSEはrole 34.6ms、最初の本文892.5ms、total 967.8msで、「こんにちは。」、`stop`、usage 22/3/25、`[DONE]`となった。同seedのnonstream本文も一致した。
- `max_tokens=1`は「One」、`length`、usage 20/1/21、`[DONE]`となった。
- active中の衝突は1.495msで429、`Retry-After: 1`。先行128-token streamは5.452秒で正常完了した。active readinessは20/20回200。
- 512-token設定のstreamを最初の本文受信後にsocket切断した。直後の3回は429、4回目が200となり、切断から回復応答完了まで914.5msだった。worker PID/starttimeは不変。
- 長い実streamは131 SSE records、`[DONE]`は1回かつ末尾、未要求usageは0、U+FFFDは0だった。
- 正常終了後はgateway、worker、port 8000 listener、R9700 ownerが残らず、singleton lockを再取得できた。
- 構造化証跡は`uLLM-project/benchmarks/results/2026-07-11/sq8-p8e-stream-smoke-v0.1/summary.json`。

## 次の行動

P8-Fへ進む。systemd unit、固定bridge bind/firewall、secret file、restart policyを実装した後、実OpenWebUI containerからmodel discovery、日本語・英語・code block・複数turn・Stop・再接続をend-to-endで確認する。
