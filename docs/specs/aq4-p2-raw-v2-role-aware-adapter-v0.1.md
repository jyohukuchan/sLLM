# AQ4 P2 raw-v2 role-aware adapter v0.1

## 前回の要点

full-model driver v2はbenchmark driverとserved workerを別roleで記録するが、従来のP2
runnerは実行binaryをworker identityとして扱うため、driver artifactをraw-v2へ安全に
取り込めなかった。

## 今回の変更点

`tools/prefill_validation/aq4_p2_raw_v2_adapter.py` は、固定driver CLIをwarmup 2回、
measured 10回実行する。各driver artifactは別pathへ一度だけ保存し、schemaの全field、
duplicate/unknown field、case・identity・preflightのpath/hash、embedded timing/audit link、
model/package/worker、runtime device、request width、outcome、lifecycle、reset、fallback、OOMを
照合する。benchmark driverは`executed_benchmark_driver`、manifest workerは
`served_identity_reference`であり、pathまたはSHA-256が同じ場合は拒否する。

12件がすべて成功した場合だけmeasurement/state sidecarを生成する。audit、lifecycle、
resetの写像は`driver_lifecycle_input`であり、`not_a_production_execution_trace=true`を持つ。
これはP1 production execution traceではなく、`raw.links.trace`はnullのままである。
したがって、別途strict P1 trace bundleとのcase/model/worker/device/run associationが
成立するまでpromotion要件を満たさない。

driverがpreflight failure artifactを発行した場合、adapterはimmutable failure rawを保存して
非zero終了する。artifact未発行、部分schedule、artifact再利用、case/run swap、role swap、
reset/fallback不整合はfail-closeし、成功sidecarを発行しない。

binder、result builder、final validatorはplanning manifestを固定expanderで独立再展開する。
受領expandedを母集団として信頼しないため、case削除後にcount、stage count、canonical hashを
再計算した縮小matrixも拒否する。production identityはmanifestの保存pathではなく、公式
stage contract（smoke 84、representative 2,245、full 3,885、合計6,214）とcanonical
case-set digestを満たす必要がある。custom manifestは`--fixture-only`でだけbindでき、
`evidence_class=fixture_only`かつ`promotion_eligible=false`として隔離する。

## 次の行動

GPU/live実行は行わない。実driverを使う場合も、strict P1 trace bundleと完全matrixの独立検証が
揃うまではpromotionを許可しない。
