# Phase 45: adapter・control vector・dynamic model lifecycle

## 状態と目的

- 状態: complete (2026-08-22、host/API/CLIとV620/R9700 GPU evidenceを固定・検証済み)。exact gfx1030/gfx1201 full-model smokeとBroadcastAdd standalone oracleはPASSし、gfx942/MI300X runtimeだけをdeferredとする。
- 対象はPhase 39のservice lifecycle、Phase 41のidentity-safe prefix/checkpoint、既存Qwen model adapterを再利用したhost/runtime/API/CLIの機能拡張である。
- `llama.cpp`はtag `b10453`、commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`のbehavior referenceに限定する。直接reuseは行わず、行う場合はfile単位のMIT provenanceを別途固定する。
- model inputはverified GGUFとderived lock/artifactだけを受理する。download、URL、remote artifact、未検証cache、CPU/GPU fallbackは通常経路にしない。

## 固定profile

machine-readable contractは [`phase45_adapter_lifecycle_v1.json`](../../../../../../tests/fixtures/phase45_adapter_lifecycle_v1.json)、schemaは [`phase45-adapter-lifecycle-v1.schema.json`](../../../../../../ci/schema/phase45-adapter-lifecycle-v1.schema.json)、validatorは [`validate_phase45_profiles.py`](../../../../../../ci/tools/validate_phase45_profiles.py) を正とする。RDNA実機のcompact evidenceは [`phase45-adapter-lifecycle-gpu-summary-v1.json`](../../../../../../ci/matrix/phase45-adapter-lifecycle-gpu-summary-v1.json) に固定する。

- adapter/control execution v1はreviewed dense BF16 Qwen capabilityだけを対象とする。モデル、target、dtype、tensor capabilityの欠落・不一致は明示的なunsupported errorとし、別経路へfallbackしない。control vectorのlayer/range加算は既存elementwise provider/ABIへadditiveなBroadcastAddを追加して実装し、別backend・別ABI・CPU fallbackは作らない。
- LoRAはpreloaded artifactをverified base modelのreviewed tensorへbindする。target tensor名、A/B shape/orientation、rank、dtype、artifact digest、base lock、derived planをload前に検証する。rankは`1..=256`、preloaded adapterはmodelあたり最大8、request adapter setは最大4、scaleはfinite f32 `[-16,16]`、既定値`1.0`とする。
- control vectorはbase/derived identity、artifact digest、dtype、layer range、vector shapeをlockする。layer rangeは`[start,end)`のhalf-open、overlapは拒否、requestは最大4件、scaleはLoRAと同じ有限範囲とする。
- adapter/controlの順序はcanonical sorted uniqueを要求し、重複、順序違反、未知artifact、base違い、missing/extra tensor、shape/dtype/rank/range違反はGPU work前に拒否する。
- adapter無効時は既存の`adapter:none-v1` identityとbase logits/tokenを維持する。有効時はordered artifact IDとscaleをidentityへ含める。
- prefix/checkpoint identityはbase model lock fingerprint、derived plan digest、ordered adapter artifact IDs/scales、ordered control-vector IDs/scales、target semantics、renderer identity、tokenizer identityへ結合する。alias、path、cache directoryだけでidentityを構成しない。

## Registry・router contract

- registry stateは`unloaded`、`loading`、`ready`、`draining`、`failed`、`quarantined`とする。configured aliasは最大64、同時resident modelは最大16、resident bytesは設定quota以内とする。
- leaseの観測と遷移はlinearizableであり、同じ immutable identityへの同時loadはcoalesceする。loading中のrequestは503、unknown aliasは404、queue fullは429とする。
- `draining`は新規requestを拒否し、in-flight ownerを保持する。最後のownerが解放された後にだけshutdown/unloadする。active leaseを持つLRU entryはevictしない。
- load failureはpartial backend、GPU allocation、tokenizer/templateをregistryへpublishせず、`quarantined`へ遷移する。再試行は明示的なclear-quarantine actionの後だけ許可する。
- tokenizer/templateなどbase owner共有資産はadapterごとに複製しない。request-local adapter/control bindingはresident artifactを変更せず、cancel/error時はtransactionalに破棄する。
- registry manifestはaliasからverified model/derived/artifact identityを解決する。admin actionはaliasだけを引数に取り、path、URL、credentialをrequest JSONへ受け付けない。
- `--models` manifestをCLI/server起動面に追加し、`load`、`preload`、`unload`、`clear-quarantine`、`evict-idle`をCLIとadmin serverから利用可能にする。OpenAI Chat/Completions、Responses、Anthropic wire profileへ動的管理fieldを追加しない。
- requestのsLLM extensionは`sllm.adapters`と`sllm.control_vectors`のordered selectionだけを扱う。管理操作はadmin credential、推論は既存user credentialのrole境界を維持する。
- すべてのmanifest pathはregular-file、no-symlink、size/digest race checked、offline-onlyである。model lock、derived lock、artifact全体とtensor/range digestをpublish前に検証する。

## Work units

1. **P45-A0 profile・identity lock (complete)**: profile/schema/validator、LoRA/control derived-lock fields、canonical identity encoding、request/admin rejection matrix、offline/no-network boundaryを固定した。profile未登録やunknown fieldは黙認しない。
2. **P45-A1 adapter artifact・CPU oracle (complete)**: reviewed dense BF16 Qwen target mapへLoRA A/Bとcontrol vectorをbindするhost validatorとbounded slice contractを実装した。rank `1/3/17/255/256`、layer range境界、tail、zero/negative scale、NaN/Inf、disabled identityをprofileとhost testsへ固定した。
3. **P45-B1 dynamic registry (complete)**: alias table、load coalescing、linearizable lease、preload/lazy load、resident quota、LRU idle eviction、draining、quarantineを実装した。partial publication、in-flight early free、active evictionを拒否するhost concurrency testsを追加した。
4. **P45-B2 router・scheduler・observability (complete)**: alias/adapter/control requestをimmutable resolved identityへlowerし、readiness、queue/error semantics、redacted surfaceへ接続した。prompt、token、path、credential、artifact bytesは公開しない。
5. **P45-C1 CLI/server manifest・admin (complete)**: `--models` manifest、alias-only admin lifecycle、CLI管理コマンド、strict profile parser、offline path verificationを追加した。既存one-shot `generate`と既存OpenAI/Responses/Anthropic profileを維持した。
6. **P45-C2 Qwen production integration・Phase 41 identity (complete)**: Qwen dense BF16 production backendへrequest-local ordered adapter/control bindingを接続し、prefix/checkpoint identityへ結合した。未対応model/dtypeはadmission前に拒否する。
7. **P45-D1 RDNA smoke・closeout (complete)**: exact `gfx1030`/`gfx1201` release buildでQwen BF16 disabled/LoRA/control/combinedを各2回bitwise一致でPASSした。両targetでHIP-only、fallback=false、resident `8,411,592,192` bytes、request/workspace baseline復帰、pre/final allocation 0、retryable/quarantine 0を確認し、BroadcastAdd standalone (`M=1/3`, `H=17`, mismatch 0, cleanup PASS)もPASSした。gfx942はcompile-only/deferredに留める。

## Acceptance

- machine profile/schema/validatorはpin、limits、identity、state transition、security/no-network、work unit、positive/rejection matrixを一致させる。duplicate/unknown/wrong type/nonfinite、boundの両側、unsupported capabilityはGPU admission前に4xx/明示503で拒否する。
- wrong base、missing tensor、shape/dtype/rank mismatch、duplicate/order違反、scale境界、control range overlap/overflow、未対応model/low-bit dtypeをsilent fallbackせずrejectする。
- adapter/control disabled時はbase logits/token、renderer/tokenizer、existing prefix/checkpoint identityを維持する。有効時はordered artifact/scaleを含むidentity digestを再現できる。
- registryは同一identity loadをcoalesceし、linearizable leaseでin-flight ownerを保護する。drain後のlast-owner shutdown、idle-only LRU、resident quota、failed quarantine、clear後retryをhost concurrency testでPASSする。
- CLI/server manifestはoffline regular-file/digest lockを守り、admin lifecycleはalias-onlyである。request fieldsはordered selectionだけで、既存OpenAI/Responses/Anthropic semanticsを変更しない。
- synthetic slice oracleはadapter delta、control vector composition、BroadcastAdd host ABI、disabled identity、canonical digestを独立oracleへ一致させる。Qwen BF16 GPU smokeはV620 `gfx1030`（16,588 ms）とR9700 `gfx1201`（18,001 ms）でPASSし、compact summaryへmodel/plan/prompt/logit identity prefixとdispatch countを記録する。
- workspace affected tests、format、clippy、MSRV/dependency closure、Markdown/link、profile validatorをPASSする。MI300X実機はVM再確保後の別laneであり、本PhaseのPASS条件ではない。

## 検証行列

| lane | 主な確認 |
| --- | --- |
| H0 profile | JSON duplicate/nonfinite、schema const、positive/rejection case identity、pin/limit/state/identity mutation、validator unit |
| H0 artifact | offline regular file、lock/derived/artifact digest、tensor/range map、LoRA/control shape/rank/scale/layer oracle、BroadcastAdd host ABI/stride/range/finite oracle、secret/path redaction |
| H0 registry | alias 64/65、resident 16/17、coalesced load、linearizable lease、drain/cancel/error、LRU active/idle、quota、quarantine/retry、shared tokenizer/template |
| H1 API/CLI | `--models` manifest、alias-only admin actions、request extension、loading/draining/unknown/queue/auth error、legacy Chat/Responses/Anthropic regression |
| H2 identity | disabled `adapter:none-v1`、ordered set and scale digest、prefix/checkpoint fresh/reused identity and rollback |
| GPU RDNA | BroadcastAdd numeric oracleとQwen 4B BF16 disabled/LoRA/control/combinedをV620 `gfx1030`、R9700 `gfx1201`でPASS。HIP-only、fallback=false、bitwise 2x、cleanup 0、resident/request-workspace baseline復帰 |
| MI300X | real execution deferred; at most feature-pinned compile/selector evidence, never runtime PASS |

## 非対象・停止条件

- Phase 46のHF→GGUF/LoRA conversion、quantization、benchmark、quality/debug toolを前倒ししない。Phase45は既に検証済みのderived artifactを読むだけである。
- Phase 47の組込みtool/MCP、worker/sandbox、shell/network/filesystem、credential broker、human confirmationは追加しない。Phase 48 WebUIは作らない。
- model architecture、new dtype/KV format、new hardware、parallel/continuous batching、multi-GPU、MI300X tuning/provider/kernel変更は対象外である。
- artifact verificationがGPU admission後へ漏れる、in-flight ownerが早期解放される、異identityがsilent reuseされる、legacy wire profileが変わる、または同じlifecycle work unitが2回integration rejectされた場合は、同じ範囲を拡張せず再計画する。

## Closeout

完了時は本planをarchiveへ移し、matching history、main-planのPhase45 current state/llama gap row、Phase 37+ roadmap、model-lock、runtime architecture、必要なAPI/security文書、CI suite/path registrationを同期する。Phase45はhost/API/CLI/RDNA evidenceの1 commit/push単位とし、Phase46以降を同一commitへ混在させない。gfx942/MI300X real executionは別laneの入力として残す。

[Phase 37+ roadmap](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) / [main plan](../../../../main-plan.md)
