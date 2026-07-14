# AQ4 P3 selection raw producer v0.1

## 前回の要点

`select-aq4-p3-candidate.py`は、7件の代表prompt、M=128と別M、family exclusive時間、D2H・同期、full-model paired sampleからP3候補を選ぶ。一方、現行`ullm.aq4_p2_family_exclusive_profile.v1`はone-case診断用で、`measurement_eligible=false`、`promotion=false`であり、HIP APIのD2H・同期一次データを持たない。

## 今回の変更点

`tools/build-aq4-p3-selection-raw.py`は、hash-bound producer manifest、P2 identity、resident raw/summary、rocprofv3 kernel trace、rocprofv3 HIP API traceを読み、selector互換の`ullm.aq4_p2_candidate_selection_raw.v1`を生成する。kernel/APIのproducer集計値を入力せず、CSV一次行からfamily exclusive時間、D2H回数・union時間、stream/device同期回数・union時間を再計算する。

## 1. CLI

```text
python3 tools/build-aq4-p3-selection-raw.py \
  --manifest /absolute/path/producer-manifest.json \
  --output /new/path/candidate-selection-raw.json
```

入力fileと全parentのsymlinkを拒否し、regular fileのdevice、inode、mode、link count、size、mtime、ctimeをopen前後と出力直前に再検証する。入力は128 MiBを上限とする。出力先は上書きしない。

## 2. producer manifest

schemaは`ullm.aq4_p3_selection_raw_producer_input.v1`である。root exact fieldsは次の通りである。

```text
schema_version
status
measurement_eligible
smoke_only
promotion_eligible
manifest_sha256
candidate
identity
resident_summaries
representative_cases
full_model_pairs
```

### 2.1 promotion mode

```text
status = promotion_ready
measurement_eligible = true
smoke_only = false
promotion_eligible = true
```

代表caseはexactly 7件、各caseのprofile runはresident measured index 2..11のexactly 10件、full-model pairは2件以上を要求する。

### 2.2 one-case diagnostic mode

```text
status = one_case_diagnostic
measurement_eligible = false
smoke_only = true
promotion_eligible = false
```

代表caseはexactly 1件、profile runは1件、full-model pairは0件である。出力statusは`one_case_diagnostic`で、同じ非promotion flagを保持する。selectorはこのrawをpromotion inputとして拒否する。

### 2.3 manifest hash

`manifest_sha256`は自身をnullにし、次のarrayを意味順にsortしたcanonical JSONのSHA-256である。

- resident summary: SHA-256、path
- representative case: prompt ID、case SHA-256
- case内profile run: resident run index
- full-model pair: pair ID、case SHA-256

これにより入力array順序はoutputへ影響しない。各file referenceはexactly`path`と`sha256`を持ち、file bytesをhashで固定する。

## 3. identityとresident evidence

identityは`ullm.aq4_production_p2_identity.v2`、`status=bound`、self-hash一致を要求する。selector raw identityは次から導出する。

```text
identity_sha256            <- identity.identity_sha256
case_manifest_sha256       <- hash_binding.bound_case_manifest_sha256
binary_sha256              <- resident_driver_identity.binary_sha256
package_content_sha256     <- hash_binding.package_content_sha256
```

resident summaryは`ullm.aq4_p2_resident_batch.v1`、`status=complete`、2 warmup + 10 measured、`completed_cases=case_count`、identity file path/hash一致を要求する。run IDごとに一意でなければならない。

resident rawは`ullm.aq4_p2_resident_batch_raw.v1`で、次を要求する。

- `status=ok`、`immutable_status=false`、failureなし
- summaryへ存在するrun ID
- identity file path/hashとdriver identityの完全一致
- 1 model load
- scheduleが2 warmup + 10 measured、completed 12
- run index 0..11とwarmup/measured kindのexact order
- 全run成功、正のprefill時間、完全reset
- baseline identity、resident、device lock、workload、linksのexact fieldsとstrict型
- run timing、audit、state、lifecycle、reset、resource、terminalのexact fieldsとstrict型
- full-model、cold-prefill、manifest指定Mとcase ID/SHAの一致

整数fieldではboolやfloatによる代用を認めない。たとえば`reset.attempted=true`はJSON/Python上で`1`と等値に見える場合でも拒否する。driver identityの比較も、再帰的にJSON型を一致させる。

one-case smokeのraw/summaryはpromotion modeで拒否する。`smoke_only=true`、`execution_mode=one_case_smoke`、`promotion_eligible=false`、明示的`measurement_eligible=false`のいずれかがpromotion sourceに現れた場合はfail-closedとする。diagnostic modeでは逆にone-case smokeとpromotion不可の明示を必須とする。

## 4. rocprof run binding

各profile runは`ullm.aq4_p3_rocprof_run_binding.v1`で、exact fieldsは次の通りである。

```text
schema_version
case_id
case_sha256
identity_sha256
resident_run_index
measurement_eligible
clock_domain
kernel_trace_complete
hip_api_trace_complete
capture_capabilities
kernel_trace
hip_api_trace
```

`clock_domain=rocprofv3_monotonic_ns`、両trace complete=trueを要求する。promotion modeでは`measurement_eligible=true`、diagnostic modeではfalseである。case ID/SHA、identity SHA、resident measured run indexをrawへ一致させる。同じkernel traceまたはHIP API traceのSHA-256を別runで再利用してはならない。

trace completenessがfalse、必要columnがない、fileが空、API directionが不明な場合は0と推定せず入力を拒否する。`kernel_trace_complete`と`hip_api_trace_complete`はrun bindingの宣言であり、単独ではAPIの0件観測を証明しない。

### 4.1 capture capability manifest

各bindingの`capture_capabilities`はfile bytesをpath/SHA-256で固定し、schema `ullm.aq4_p3_rocprof_capture_capabilities.v1`を参照する。root exact fieldsは次である。

```text
schema_version
status
measurement_eligible
capability_sha256
tool
domains
rocprof_config
```

`capability_sha256`は自身をnullにしたcanonical JSONのself-hashである。`status=complete`、`tool.name=rocprofv3`、非空versionを要求する。promotionでは`measurement_eligible=true`、diagnosticではfalseである。

`domains`はexactly次のboolean fieldを持ち、すべてtrueでなければならない。

```text
kernel_dispatch
hip_api
d2h_memcpy
stream_synchronize
device_synchronize
```

`rocprof_config`はexactly次を持つ。

```text
kernel_trace = true
hip_api_trace = true
api_filter = all_functions
```

missing、unknown、false、型代用、self-hash不一致、file hash不一致はfail-closedである。同じcapability manifestは同一capture設定の複数runから参照できるが、kernel/API trace自体の再利用は禁止する。

## 5. kernel trace

受理するcolumn aliasは現行family profilerと同じである。

```text
dispatch: Dispatch_Id | Dispatch_ID | Index | dispatch_id
name:     Kernel_Name | KernelName | Name | kernel_name
start:    Start_Timestamp | BeginNs | start_ns
end:      End_Timestamp | EndNs | end_ns
phase:    Phase | phase
```

column aliasは各種類exactly 1個でなければならない。全rowは`phase=prefill`、dispatch ID一意、timestamp昇順、`0 <= start < end <= 2^63-1`を要求する。kernel名は既存`aq4_p2_kernel_family_mapping.v1`で分類し、unknownまたは複数family一致を拒否する。

同一familyの重複intervalはsweep-line unionで一度だけ数える。異なるfamilyが重なる区間はcross-family overlapとしてcandidate exclusive時間へ含めない。候補ごとの回収可能時間は次である。

- paged KV: `paged_validation.exclusive_ns`
- AQ4 register BM8: `aq4_projection.exclusive_ns`
- chunk execution: `attention.exclusive_ns + recurrent.exclusive_ns`
- fusion: `normalization.exclusive_ns`

10 profile runのcandidate exclusive millisecondsのmedianを`recoverable_family_exclusive_ms`にする。

## 6. HIP API trace

受理するcolumn aliasは次の通りである。

```text
correlation: Correlation_Id | Correlation_ID | Index | correlation_id
function:    Function | Api_Name | API_Name | Name | function
start:       Start_Timestamp | BeginNs | start_ns
end:         End_Timestamp | EndNs | end_ns
```

correlation ID一意、timestamp昇順、int64範囲を要求する。明示的に分類できるAPIは次である。

### D2H

```text
hipMemcpyDtoH
hipMemcpyDtoHAsync
```

### stream/device sync

```text
hipStreamSynchronize
hipDeviceSynchronize
```

明示的H2D/D2D/peer copyと`hipEventSynchronize`は別種として無視する。`hipMemcpyAsync`のように方向を証明できない名前、または未知の`*Synchronize*`は拒否する。API traceがheaderだけで空の場合も、0件を観測したとは扱わない。

`hipLaunchKernel`やH2Dだけを含みD2H/syncが0件のtraceは、hash-bound capability manifestが全HIP API domainと`api_filter=all_functions`を証明した場合にだけ0として受理する。capabilityがmissing、unknown、ambiguous、不完全なら0件を能力不足と区別できないため拒否する。

回数は一次row数である。時間は同種API intervalのunionで、重複時間を二重計上しない。selector rawの各代表promptには10 measured runの合計として次を保存する。

```text
d2h_count
d2h_time_ms
stream_sync_count
stream_sync_time_ms
```

## 7. baseline統計とfull-model pair

代表caseのbaselineはresident rawの10 measured `timing.prefill_ms`から再計算する。

```text
baseline_p50_ms = median
baseline_cv = sample standard deviation / mean
ci95_halfwidth_ms = t(0.975, df=9) * sample standard deviation / sqrt(10)
```

non-finiteと演算overflowを拒否する。

full-model pair exact fieldsは次の通りである。

```text
pair_id
case_id
case_sha256
run_index
baseline_raw
candidate_raw
```

baseline/candidateは異なるrun ID、同じidentity file、driver identity、case ID/SHA、workload、measured run indexでなければならない。同じraw内の別indexや別case、別identity、warmup index、同じrun IDへの差し替えを拒否する。出力値は対応するrunの`prefill_ms`である。

## 8. fail-closed条件

- schema/status/field/typeのmissing、unknown、duplicate、non-finite
- manifest、identity self-hash、file reference SHAの不一致
- input TOCTOUまたはtrace再利用
- incomplete/smoke/ineligible resident summary/rawからpromotion生成
- capture capability manifestの欠落、不完全なdomain/config、self-hash/file hash不一致
- resident raw nested fieldのbool/int/float代用またはunknown/missing field
- 7 prompt不足、prompt/case重複、M=128または別Mの欠落
- profile measured index 2..11の欠落・重複・順序差し替え
- kernel unknown、API transfer/sync unknown、trace column/clock/phase不正
- full-model pair不足、run/case/workload/identity不一致
- 統計または出力のnon-finite/overflow
- 既存outputへの上書き

これらはexit code 2で、outputを発行しない。

## 次の行動

実R9700 captureでは、各代表promptの10 measured runごとにkernel traceとHIP API traceを別fileへ保存し、run bindingへcase/identity/index/hashを固定する。one-case profileはdiagnostic manifestだけを生成し、7 promptとfull-model pairsが揃うまでpromotion rawへ昇格しない。生成したpromotion rawはselectorへ渡し、候補が確定してからP3 runtime実装へ進む。
