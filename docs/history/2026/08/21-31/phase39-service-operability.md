# Phase 39 service operability・認証・observability

## 結果

2026-08-21にPhase 39のhost-side実装とintegrationを完了した。既存のstrict/OpenWebUI Chat Completions profile、
TLS/CORS/auth/metrics/replay無効default、通常SSE disconnect cancellationを維持し、次をadditiveに追加した。

- `/healthz` livenessと、lifecycle＋scheduler acceptanceを結合した`/readyz`。
- opt-in `/metrics`、authenticated `/props`、admin-only `/slots`、slot cancel、key reload。
- bounded model/request/token/TTFT/E2E/cancel/HTTP/scheduler/readiness/runtime-memory metrics。
- nonblocking Qwen/Gemma runtime allocator snapshot。固定categoryはmodel resident、request/KV、workspace/arena、total。
- SHA-256 digestだけを保持する複数user/admin credential、全entry constant-time比較、fail-closed atomic key reload。
- `sllm.resumable=true`の明示opt-in SSE、単調event ID、bounded process-local replay、`Last-Event-ID`再開。
- 最大32 exact originのCORS allowlist、Rustls TLS listener、certificate/keyの起動前検証、graceful drain。

## 互換性とsecurity境界

`/healthz`と`/readyz`はorchestratorがcredentialなしで利用できる。その他のread-only operationはuser/admin key、
slot/key mutationはadmin keyを要求する。credential未設定のlocal profileでもadmin surfaceは閉じたままである。
slot/props/metricsにはprompt、token ID、credential、request ID、backend error文字列を含めない。

key fileは`user:<token>`/`admin:<token>`、32 key、4,096 byte/token、64 KiB/fileに制限した。symlink、非regular、
Unix group/other permission、malformed/duplicateを拒否する。reloadはparse/validation後だけsnapshotを交換し、失敗時は旧keyを維持する。
TLS private keyもnon-symlink regular fileとprivate permissionを要求し、cert/key pairとPEMをGPU/model load前に検証する。

通常SSEは既存どおりclient disconnectでcancelする。resumableを要求したrequestだけはbounded replay producerがgenerationを保持する。
feature無効、`stream=false`、unknown session、古すぎるcursor、capacity不足はscheduler投入前または明示HTTP errorで拒否する。
CORSはscheme＋authorityだけの完全一致HTTP(S) originとし、wildcard、path、query、userinfoを拒否する。

## Observability境界

metric seriesは最大16 model aliasと固定enumの直積だけで構築する。memory値はsLLM runtime allocatorが追跡するdevice bytesであり、
driver全体のVRAM値ではない。backendはgeneration mutexを待たず`try_lock()`し、busy、shutdown、poisoned accounting時はzero snapshotを
返すため、scrapeがgenerationをblockしない。health endpointはbackend generationを呼ばず、readinessはfallbackを実行しない。

## Host verification

- `cargo fmt --all --check`: PASS
- `cargo test -p sllm-server --all-targets`: PASS（62 tests）
  - library 45
  - server binary 2
  - existing HTTP contract 10
  - Phase 39 operability integration 5
- `cargo clippy -p sllm-server --all-targets -- -D warnings`: PASS
- `git diff --check`: PASS

Phase 39 integrationはloopback HTTPとdeterministic fixture backendを使い、health/readiness、metrics opt-in/no-secret、props auth、
queued/active cancel、exact CORS preflight、resumable reconnect/重複なし/unknown/範囲外/disabled、credential role/key rotationを確認した。
GPUを使用しておらず、新しいGPU correctness、performance、hardware compatibilityのPASSは主張しない。MI300X実機作業はPhase 37/38の
deferred laneとして分離したままである。

[archived plan](../../../../plans/archive/2026/08/21-31/phase39-service-operability.md) /
[main plan](../../../../plans/main-plan.md) /
[OpenAI compatibility](../../../../api/openai-compatibility.md) /
[runtime architecture](../../../../architecture/runtime.md) /
[credentials](../../../../security/credentials.md)
