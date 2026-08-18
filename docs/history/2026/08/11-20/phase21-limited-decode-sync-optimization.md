# Phase 21 limited decode segment synchronization optimization history

## 2026-08-17: 詳細計画作成

- ユーザー指示により、Phase 21を単一request、batch 1の通常text decodeに残るper-op completion/timing eventと
  completion queryの削減へ限定した。
- Phase 9/13のmodel-neutral prepared execution、same-stream segment、owner lifetime、terminal boundary、transactional state
  publicationを維持し、通常modeのper-op timing無効化と非空segment当たり最大1 terminal eventへの集約だけをcandidateとした。
- token/position H2D統合、Argmax transfer融合、KV publication変更、HIP Graph/command-list、event pool、registry lock、GEMV、
  batching、DeepSeek V4、TurboQuant、multi-GPUを明示的な非対象にした。
- public standalone completion ABI、normal/profile timing policy、terminal success/failure/pending/cancel/drop時のowner lifetime、
  audit exact countを固定contractとした。
- Qwen3.5-4B BF16 GGUFをprimary性能laneとし、canonical V620/R9700でfresh baselineとcounterbalanced candidateを比較する。
  採用はcase固有noise envelopeで判断し、改善しないcandidateはdefaultへ残さず、理由付き棄却でPhaseを完了できる条件とした。
- Gemma 4/Qwen3.5 MoEは共通adapterのhost回帰に限定し、Phase 21だけを理由に大型full-model GPU matrixを追加しない。
- 本時点ではsource、ABI、kernel、evidence schemaを変更しておらず、GPU測定も行っていない。

## 2026-08-18: 実装、dual-GPU評価、candidate棄却

### 実装

- public HIP ABIへ`PROFILED`/`DEFERRED` queue completion mode、untimed queue fence、
  same-context/same-queue fence成功後のeventless completion finalizeをadditiveに追加した。
- numeric operationはDEFERRED時だけper-op completion/timing eventを作らず、PROFILED既定では従来のquery/wait/timing contractを維持する。
- coreのbackend-neutral session/fence/submission contractと`ExecutionSegment`へterminal fence finalizeを追加し、
  semantic、KV、causal、linear、sparse-MoE ownerを同一contractで扱えるようにした。
- Qwen/Gemmaへのcandidate接続、stub/bindings、public symbol inventory、H3 schema/matrixを同期した。
  CLIのexact-session failureは元backend errorを保持するよう診断を改善した。

### correctness・review

- native fake-HIP testは17 RMSNorm owner、per-op event 0、fence event 1、fence前NOT_READY、record failure rollback、
  active中mode変更BUSY、finalize/release/accounting zeroを確認した。host CTestは3/3 PASSした。
- Rustは`cargo test -p sllm-core -p sllm-hip --offline`をPASSし、core 174、HIP 95に加えて関連bin/integration/doc testが全てPASSした。
- `cargo clippy`はPhase 21 findingを修正後、既存の`manual_contains`、`needless_borrow`、
  `collapsible_if`だけを明示allowしてPASSした。`cargo fmt --check`と`git diff --check`もPASSした。
- H3 public-runtime contract validator、JSON/schema validator、H3 contract/runner 65 testをPASSした。
  source inventoryの個別hashとaggregate hashをcurrent sourceへ同期した。
- exact `gfx1030`/`gfx1201` release buildをPASSした。R9700は複数GPU可視状態ではHIP loaderが
  `device kernel image is invalid`を返したが、canonical運用どおり`HIP_VISIBLE_DEVICES=2`で単独可視化するとPASSした。
- integration reviewでGemma static KV publicationを新規audit boundaryとして数え、boundary countを6から150へ変える差分を検出した。
  synchronizationは維持しつつ既存audit表現を保つよう修正し、V620 Gemma smokeでtoken `[236770,236770,236770]`、
  submission/kernel `3018/4002`、segment/boundary `6/6`、cleanup 0を確認した。

### 固定laneと採否

- source baselineはcommit `5fd22f6925ada3fe083620a1714044af2dc1f2f6`。
  modelは`phase20-audit-qwen35-bf16.gguf`、lock fingerprint
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`。
- promptはR9700 MTPの既存graph capacity制約を避けて両target共通の`Hello world`（2 token）へ固定し、
  greedy 3-token generation、各3 warmup + 10 measured、baseline/candidate交互順とした。
- V620は全runでtoken `[0,271,760]`、submission/kernel `1404/1476`、segment/boundary `27/27`、
  HIP-only、fallbackなし、cleanup 0だった。baseline/candidateの中央値は4.1463599465/4.1522215475秒、
  MADは0.029411482/0.0402361095秒で、candidateは0.14%遅かった。
- R9700も同じtoken、submission/kernel `936/984`、segment/boundary `18/18`、HIP-only、
  fallbackなし、cleanup 0だった。中央値は4.921334225/4.9301471625秒、MADは
  0.043910428/0.033448239秒で、candidateは0.18%遅かった。
- 17 ownerを1 fence eventへ集約する構造削減は成立したが、どちらのprimary wall metricも改善せず差はnoise内だった。
  固定した条件10/11に従いproduction candidateを棄却し、Qwen/GemmaはPROFILED defaultへ戻した。
  deferred ABI/core primitiveはfault-testedな実験基盤として残すが、Phase 21の性能成果とは表記しない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase21-limited-decode-sync-optimization.md)
