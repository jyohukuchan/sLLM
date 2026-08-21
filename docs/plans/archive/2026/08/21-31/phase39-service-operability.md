# Phase 39: service operability・認証・observability

> 状態: complete（host all-target 62件PASS、clippy warning 0。GPU PASS claimなし）
> 開始日: 2026-08-21

## 目的

現行のOpenAI Chat Completions profile v1を壊さず、長時間稼働する単一model serverに必要なliveness/readiness、
bounded metrics、redacted slot管理、明示opt-inのresumable SSE、TLS/CORS、複数user/admin credentialを追加する。
MI300X実機検証とは独立したhost-side Phaseとし、GPU PASSやGPU性能改善は主張しない。

## 固定した受入条件

1. 既存のTLS/CORS/auth/metrics/replay無効profileと通常SSE disconnect cancellationを維持する。
2. `/healthz`はGPU処理を起動せずprocess lifecycleを返し、`/readyz`はlifecycleとscheduler受付可否を
   fail-closedに結合する。
3. metricsは明示有効化時だけ公開し、model aliasと固定enumだけをlabelに用いる。prompt本文、token ID、credential、
   request IDを含めず、scheduler、request/token、TTFT/E2E、cancel、HTTP、model ready、runtime-tracked device memoryを返す。
4. props/slotsはrequest内容を持たないredacted snapshotとし、slot cancelとkey reloadはadmin keyだけを許す。
5. key fileは`user:<token>`/`admin:<token>`の最大32 keyとし、digestだけを保持して全entryをconstant-time比較する。
   symlink、非regular file、過剰権限、duplicate/malformed/oversizeを拒否し、reload失敗時は旧snapshotを維持する。
6. resumable SSEは`stream=true`かつ`sllm.resumable=true`だけで有効にし、単調event ID、bounded replay、
   `Last-Event-ID`の重複なし再開、unknown/範囲外の明示errorを実装する。event/sessionのbyte数も固定上限にし、通常SSEの意味は変更しない。
7. CORSは最大32件の完全一致HTTP(S) origin allowlistとし、wildcard、path/query/userinfoを拒否する。
   TLS cert/keyはpair必須、GPU load前にPEMを検証し、private keyのsymlink/非regular/過剰権限を拒否する。
8. loading/ready/draining/failed/shutdown、queue full、slow/disconnect、cancel、replay、key rotation、CORS、
   malformed configをhost unit/integration testで確認する。

## 実装work unit

- P39-A: atomic lifecycleとscheduler slot registry、非ゼロ単調slot ID、queued/active cancel。
- P39-B: credential digest store、role分離、fail-closed atomic key rotation。
- P39-C: bounded Prometheus registry、nonblocking backend memory snapshot、health/readiness/props/slots/admin endpoint。
- P39-D: bounded resumable store、SSE producer/replay route、strict request extension。
- P39-E: exact CORS、Rustls listener、graceful drain、CLI configuration。
- P39-F: host integration、compatibility/security/runtime文書、履歴、release checks。

## 非対象

- GPU correctness/performance、MI300X Phase 37/38の実機証拠。
- distributed tracing、external secret manager、mTLS、automatic certificate renewal、multi-process replay persistence。
- Phase 40以降のsampler/grammar、prefix/session checkpoint、dynamic model router、WebUI。

## 証拠境界

本PhaseのPASSはhost contract、HTTP integration、Rust compile/test/clippyに限定する。backend memoryはsLLM runtimeが追跡する
device allocationのcurrent/high-waterであり、driver全体のVRAM利用量ではない。backend busy時のscrapeはgenerationをblockせず、
memory snapshotを0として返す。

[roadmap](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[main plan](../../../../main-plan.md)
