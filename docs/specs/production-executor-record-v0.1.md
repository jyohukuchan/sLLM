# Production executor record v0.1

この文書は、`ullm.production_execution_trace.v1` producerと独立validatorの間で使う、bounded sanitized executor-record sidecarの契約を定める。workerのJSONL wire protocolは変更しない。

## 目的

executor recordはproducerの`passed`やsummaryではなく、実行境界から観測した事実を保持する。traceはこのrecordから生成され、binding sidecarがtrace SHA-256とrecord SHA-256を結ぶ。

recordにはprompt本文、response本文、prompt/generated token ID、request ID、API key、HTTP header、OpenWebUI DB内容を含めない。`prompt_token_count`などの集計値は許可する。

## 必須root fields

```text
schema_version, trace_id, status, scope, graph, executor,
request_summary, phases, operator_resolutions, fallback, memory,
state_commit, server, failure
```

`schema_version`は`ullm.production_executor_record.v1`、`status`と`scope`はproduction execution traceと同じ値とする。`scope=production_server`では`server`をobjectにし、それ以外では`null`とする。

recordのrootは次のexact fieldsだけを持つ。

```text
schema_version, trace_id, status, scope, graph, executor, request_summary,
phases, operator_resolutions, fallback, memory, state_commit, server, failure
```

`scope=worker`または`scope=direct_worker`は入力側の互換ラベルとしてのみ許可し、producerはこれを`full_model`へ正規化する。production
serverは、ready/releaseを実際のserver boundaryで観測したrecordだけが宣言できる。direct workerのJSONL境界はserver boundaryではない。

`executor`、`request_summary`、`phases`、`operator_resolutions`、`fallback`、`memory`、`state_commit`、`server`、`failure`のnested fieldsは、trace仕様のexact field setと同一でなければならない。特に`memory.oom`は`null`または`stage`、`reason_code`、`planned_bytes`、`observed_peak_bytes`を持つobjectであり、booleanではない。graphはcanonical model/state schema入力と`compatibility_inputs`を含み、producerはcanonical JSONからdigestを再計算する。

## 記録する事実

- `phases`: 実際のphase種別、要求chunk幅、実token/request幅、context遷移、wall time。
- `operator_resolutions`: semantic operator、resolved implementation、format、shape bucket、workspace、invocation count。
- `fallback`: non-selected operatorごとの分類と理由。
- `memory`: capacity、resident、persistent state、temporary workspace、observed peak、observer完了状態、OOM。
- `state_commit`: prepare、commit、discard、cancel/error、resetの実数。
- `graph`: canonicalなmodel graph/state schema入力。producerがcanonical入力をSHA-256化し、traceにはdigestだけを残す。

全ての数値は非負かつJSON safe integer、durationは有限値とする。上限はtrace仕様の4 MiB、24階層、32,768 nodesに従う。

## Publication

producerは`.incomplete`へwrite、flush、fsyncした後にatomic renameする。既存artifactは上書きしない。binding sidecarは次のexact fieldsを持つ。

```json
{
  "schema_version": "ullm.production_executor_trace_binding.v1",
  "trace_id": "public-fixture-id",
  "trace_sha256": "64 lowercase hex",
  "executor_record_sha256": "64 lowercase hex"
}
```

P1のPython producerは、独立した未検証のworkerを実行した事実を示す
facts-emitter境界であり、`producer.binary_sha256`にはmanifestへbindingした
worker executable digestを入れる。これはPython script自体のハッシュをworker
identityとして偽装するものではない。独立validatorはmanifestのworker pathと
declared digestを再読込みし、`producer.binary_sha256`、
`identity.worker.binary_sha256`、実ファイルSHA-256が一致する場合だけ受け入れる。
manifest-relative pathは`..`とsymlink componentを拒否し、absolute pathを使う
production manifestでも同じregular-file/hash検査を行う。

独立validatorはproducerの`verified`を受け入れず、manifest、worker binary、facts、trace、bindingのhashと、phase/operator/memory/stateの算術を再計算する。traceの`independent_validation`が`not_run`のartifactは検証済みであってもproduction promotion eligibleではない。

P1のmechanics smokeはschema、binding、privacy、counter、publishの検証だけを示す。これは性能証拠ではない。P2では同じrecord/trace runnerを、active manifestと同一binary/packageを使う実際のproduction request boundaryへ接続し、full-model/full-server coverage、実際のbatch width、operator observer、OOM/resetを再取得する。P1-Dのread-only bottleneck auditは診断専用で、P2の実測前に候補を昇格させない。
