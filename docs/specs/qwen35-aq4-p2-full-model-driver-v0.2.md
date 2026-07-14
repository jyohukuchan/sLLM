# Qwen3.5 AQ4 P2 full-model driver v0.2

## 前回の要点

v0.1はAQ4 resident sessionを直接実行し、要求幅、実幅、timing、terminal auditを
sanitized JSONへ保存した。初版は成功時だけartifactを発行し、manifest workerと
benchmark driverのbinary identityを分離していなかった。

## 今回の変更点

driverは次の入力を必須とする。

    --served-model-manifest PATH
    --fixture PATH
    --case P2_CASE.json
    --identity P2_IDENTITY.json
    --preflight P2_PREFLIGHT.json
    --m 1|8|16|32|64|128
    --output RESULT.json

result schemaはullm.qwen35_aq4_p2.full_model_driver.v2である。rootのexact fieldsは次の
23 fieldsであり、追加・省略を認めない。

    schema_version, raw_target_schema_version, scope, status, immutable_status,
    case_id, case_sha256, identity, requested_m, resolved_m,
    actual_token_batch_width, actual_request_batch_width, timing, audit,
    lifecycle, reset, outcome, oom, fallback, preflight, failure, links, adapter

statusはok、failed、oomのいずれかである。failed/oomはimmutable_status=trueとし、
manifest load後のidentity、environment preflight、package preflight、model load、
request drive、audit、resetの失敗でも、指定outputへ一度だけatomic publishして
processは非zeroで終了する。oomはstageとreason_codeを保持する。producerの自由文errorは
保存しない。

okには次の全証拠を要求する。

- manifestが列挙するrequired_environmentの全値が実processで文字列1である。
- served-model loaderがmanifest workerのregular/executable/SHA-256を再検証する。
- 実行中benchmark binaryのpath/SHA-256を別identityとして再計算し、manifest workerの
  pathまたはdigestと同じ場合は拒否する。
- identity.runtime_deviceへ実際の--device-index、runtimeが観測したdevice_id、backend、
  name、gcn architectureを保存する。case.deviceのlogical device id、backend、name、
  architecture、runtime HIP ordinalとすべて一致しなければ拒否する。同じgfx architecture
  の別GPUもnameまたはdevice_idの不一致で拒否する。
- package rootの全regular fileをrelative path順にstreaming SHA-256 tree hashし、
  P2 identityのpackage_content_sha256と一致させる。package manifest hashをcontent
  identityの代用にしない。
- terminal OperationExecutionAuditが存在し、coverage_complete=trueである。
- terminal resetが1回完了している。
- terminalのresolved M、actual token/request width、lifecycleがすべて存在する。resetは
  attempted=1、complete=1、failed=0であり、lifecycleはprepare=commit+discardを満たす。
- native-M caseでunexpected fallbackがない。

fixtureおよびcase/identity/preflight JSONは、全parent path componentのsymlinkを拒否し、
open前後のdevice/inode/size identityを照合し、64 KiB chunkで上限まで読む。fixture schema
はunknown fieldを拒否し、全JSON objectはduplicate keyを拒否する。

caseはexact schemaとして読み、fixtureのprompt token count/step count、full_model scope、
cold_prefill phase、all_m1またはcold_batched mode、request/decode count、context、requested/
resolved M、greedy sampling、AQ4 target control、format/implementation、served identity、
runtime deviceを実行前に照合する。case_sha256 self-hashとcase file SHA-256 linkは別々に保持する。

fallback.countはterminal auditのimplementation invocation countから算出する。
fallback.reasonsはload-time ResolutionKind::Fallbackのunavailable primary、resolved
implementation、audited invocation countを保持する。auditがないresultはokにならない。

linksは次のexact fieldsを持つ。

    case, identity, preflight, timing, audit

case/identity/preflightはabsolute pathとfile SHA-256である。timingとauditはresult内の
JSON Pointerとdigestである。prompt本文、prompt token ID、生成token ID、生成本文は
resultに含めない。

### P2 raw v2 adapter handshake

adapter.target_schema_versionはullm.aq4_production_p2_raw_result.v2、
mapping_versionはullm.aq4_p2_full_model_to_raw.v1である。mappingは次の通りである。

- case_id、case_sha256、status、immutable_statusは同名fieldへ移す。
- timing.request_elapsed_msはexecution.elapsed_msへ移し、generation timingは
  measurement sidecarへ展開する。
- identity.package_root/package_content_sha256はdeclared_executionへ移す。
- preflight.inputはraw preflightへ移す。
- lifecycle/reset/audit/fallbackはstateおよびexecution-trace adapterの入力にする。
- links.case/identity/preflightのpath/SHA-256はraw linksと独立validatorへ渡す。

現行raw v2 runnerはdeclared executableをmanifest workerと同一に固定しており、実際に
実行したbenchmark driver binaryを別roleで表現できない。またtiming/auditのembedded
linkをmeasurement/stateへ展開するinterfaceがない。このためdriverは
raw_v2_requires_role_aware_adapter=trueを必ず記録する。adapter側が
executed_benchmark_driverとserved_identity_referenceを別々に検証するまで、driver
resultをraw v2として直接偽装してはならない。driverはP2 toolsを変更しない。

## 次の行動

P2 runner/adapter所有者が上記handshakeを実装し、worker identityの詐称なしにdriver
resultをraw v2へ変換する。その独立reviewが通るまでGPU/live実行を行わない。

現行expanded caseのprefill generated_tokens、implementation_id、device name/architectureは、
full-model driverの実際のrequest/session/runtime identityとexactに一致することをcase所有者が
事前に確認する。一致しない現行caseはidentity gateで拒否され、live実行に進めない。
