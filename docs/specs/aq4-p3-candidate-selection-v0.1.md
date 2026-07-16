# AQ4 P3 candidate selection v0.1

## 前回の要点

P3の最適化候補は、paged KV table検証、AQ4 register BM8、chunk実行、projection・norm・activation融合の順で仮説化されている。ただし、P2の実測前に候補を固定してはならない。現行の`ullm.aq4_p2_family_exclusive_profile.v1`はkernel区間の診断用であり、`measurement_eligible=false`、`promotion=false`で、D2H回数とstream同期回数を持たない。

## 今回の変更点

`tools/select-aq4-p3-candidate.py`は、hashで束縛されたP2 raw evidenceから候補別の回収可能時間比率`E`、ノイズ下限`N`、代表promptの支持数、M幅、full-model paired 95%信頼区間を再計算し、P3候補を機械的に選ぶ。producerが書いた合否や集計値は入力に持たせない。

## 1. CLI

```text
python3 tools/select-aq4-p3-candidate.py \
  --evidence /absolute/path/raw-1.json \
  [--evidence /absolute/path/raw-2.json ...] \
  --output /new/absolute/or/relative/path/selection.json
```

入力は複数指定できる。順序は意味を持たない。同じ候補・同じprompt ID、または同じ候補・同じpair IDが複数入力に現れた場合は拒否する。出力先が既に存在する場合は上書きしない。

## 2. 受理する入力schema

### 2.1 測定用raw evidence

schemaは`ullm.aq4_p2_candidate_selection_raw.v1`である。rootのexact fieldsは次の通りである。

```text
schema_version
status
measurement_eligible
smoke_only
promotion_eligible
promotion_ineligibility_reason
evidence_sha256
upstream_qualification
identity
capabilities
representative_prompt_count
measurements
full_model_pairs
```

必須値は次の通りである。

```text
schema_version = ullm.aq4_p2_candidate_selection_raw.v1
promotion: status=complete, measurement_eligible=true, smoke_only=false,
promotion_eligible=true, promotion_ineligibility_reason=null, representative_prompt_count=7

diagnostic: status=one_case_diagnostic, measurement_eligible=false, smoke_only=true,
promotion_eligible=false,
promotion_ineligibility_reason=upstream_p2_calibration_rejected_no_go,
representative_prompt_count=1
```

promotion rawは`qualified_go`、diagnostic rawは`rejected_no_go`のupstream qualification fileをpath、file SHA-256、qualification self-hash、status、reasonで結合する。selectorは元fileとP2 chainを再検証する。diagnostic rawの測定値は選考に投入せず、canonical `no_eligible_candidate`を生成する。missing field、unknown field、duplicate JSON key、NaN/Infinity、負値、不正な型も拒否する。

`identity`のexact fieldsは次の通りで、すべてlowercase SHA-256である。

```text
identity_sha256
case_manifest_sha256
binary_sha256
package_content_sha256
```

複数raw inputの`identity_sha256`は一致しなければならない。各measurementとfull-model pairの`identity_sha256`もroot identityと一致しなければならない。

`capabilities`のexact fieldsは次のbooleanである。

```text
family_exclusive_timing
d2h_count
d2h_bytes
d2h_time_ms
stream_sync_count
stream_sync_time_ms
```

同じ候補を複数raw inputへ分ける場合、その候補のmeasurementまたはfull-model pairを含む
全inputが必要capabilityをtrueにしなければならない。pairだけを供給するinputも集約対象であり、
falseまたはmissingはfail-closedとする。別候補のcapabilityを流用しない。

既存4固定候補はこの契約をそのまま使用する。候補Aの正式IDは
`sequence-output-direct-v1`で、familyは既存trace分類と互換な`attention_recurrent`である。
Aを含むrawだけは、次のA専用capabilityもexactに持つ。

```text
direct_sequence_output
d2d_bytes
d2d_copy_count
launch_count
component_latency
full_model_latency
workspace_bytes
peak_vram_bytes
fallback_reasons
alias_size_admission_safety
fidelity_binding
```

### 2.2 代表prompt measurement

`measurements`の各rowのexact fieldsは次の通りである。

```text
candidate_id
family
prompt_id
case_sha256
identity_sha256
resolved_m
baseline_p50_ms
baseline_cv
ci95_halfwidth_ms
recoverable_family_exclusive_ms
d2h_count
stream_sync_count
```

`baseline_cv`は比率であり、例えば2%は`0.02`とする。`ci95_halfwidth_ms`はbaseline p50と同じmillisecond単位である。`recoverable_family_exclusive_ms`は、同じ測定契約で得た非重複のfamily exclusive時間でなければならない。diagnostic profileの1回実行値と、別の2 warmup + 10 measured p50を混ぜて作らない。

`d2h_count`、`d2h_time_ms`、`stream_sync_count`、`stream_sync_time_ms`は、候補が不要とする場合だけnullを許す。paged KV候補では回数を非負整数、時間を非負のmillisecondとして一次traceから実測し、少なくともいずれかの回数が1件以上の代表promptで正値でなければならない。API traceが方向や同期種別を証明できない場合は0にせず証拠不足として扱う。

固定候補とfamilyの対応は次の通りである。

| candidate_id | family | 追加証拠 |
|---|---|---|
| `paged-kv-table-validation-v1` | `paged_validation` | D2H回数、stream同期回数 |
| `aq4-register-bm8-v1` | `aq4_projection` | family exclusive時間 |
| `chunk-execution-v1` | `attention_recurrent` | attention/recurrentの非重複合計 |
| `projection-norm-activation-fusion-v1` | `normalization` | family exclusive時間 |
| `sequence-output-direct-v1` | `attention_recurrent` | D2D bytes/copy、launch、component/full-model p50/p95、workspace/peak VRAM、fallback、alias/size/admission safety、direct/copy fidelity |

unknown candidate、candidateとfamilyの不一致、同じcandidate/prompt IDの重複は拒否する。

Aのmeasurement rowは、既存fieldsに加えて次のexact fieldsを持つ。

```text
baseline_d2d_bytes, candidate_d2d_bytes
baseline_d2d_copy_count, candidate_d2d_copy_count
baseline_launch_count, candidate_launch_count
baseline_component_p50_ms, baseline_component_p95_ms
candidate_component_p50_ms, candidate_component_p95_ms
baseline_full_model_p95_ms
candidate_full_model_p50_ms, candidate_full_model_p95_ms
baseline_workspace_bytes, candidate_workspace_bytes
baseline_peak_vram_bytes, candidate_peak_vram_bytes
baseline_fallback_count, candidate_fallback_count
baseline_fallback_reasons, candidate_fallback_reasons
direct_alias_safe, direct_size_safe, direct_admission_safe
direct_fidelity_binding_sha256, copy_fidelity_binding_sha256
```

Aのfull-model pairは、baseline/candidateのD2D bytes/copy count、launch count、workspace/peak
VRAM、fallback count/reasons、安全性3項目、fidelity binding 2項目を同様に必須とする。
measurementのD2D bytes、workspace bytes、peak VRAMは10件の非負整数sampleから得たmedian
なので、非負整数または半整数だけを許す。半整数を整数へ切り捨ててはならない。copy、launch、
fallbackのcount medianは整数だけを許し、それ以外の分数は証拠契約違反として拒否する。
浮動小数点でexactな半整数を表せない範囲も拒否する。

### 2.3 full-model paired sample

`full_model_pairs`の各rowのexact fieldsは次の通りである。

```text
candidate_id
pair_id
case_sha256
identity_sha256
baseline_ms
candidate_ms
```

同一条件で対応付けたbaselineとcandidateを2件以上30件以下与える。改善量は`baseline_ms - candidate_ms`である。selectorはsample mean、sample variance、Student tの両側95%信頼区間を再計算する。下限が厳密に0より大きい場合だけ合格する。下限が0と等しい場合は不合格である。

### 2.4 semantic hash

`evidence_sha256`は、自身をnullにしたroot全体をcanonical JSONへ変換してSHA-256を取る。canonical化ではobject keyを辞書順にし、`measurements`をcandidate ID、prompt ID、case SHA、Mの順、`full_model_pairs`をcandidate ID、pair ID、case SHAの順に並べる。このためarray順序はhashと選定結果へ影響しない。

fieldを書き換えてhashを更新しない場合は拒否する。hashを更新してもrowのidentityがroot identityと異なる場合は拒否する。

### 2.5 現行diagnostic profile

`ullm.aq4_p2_family_exclusive_profile.v1`もlineage確認のために入力できる。ただし、次を必須とする。

```text
status = profiled_diagnostic
measurement_eligible = false
promotion = false
timing_ns.prefill.families が存在する
```

このprofileは選定値を供給しない。raw evidenceがない場合、全候補をineligibleとする。特にpaged KV候補には次の不足理由を出す。

```text
eligible_raw_evidence_missing
paged_kv_d2h_capability_missing
paged_kv_stream_sync_capability_missing
```

kernel名からD2Hや同期を推測しない。profileのfamily exclusive時間だけでpaged KV候補を合格させない。

## 3. 選定式

各代表prompt `i`について次を再計算する。

```text
E_i = recoverable_family_exclusive_ms_i / baseline_p50_ms_i
N_i = max(
  0.05,
  3 * baseline_cv_i,
  2 * ci95_halfwidth_ms_i / baseline_p50_ms_i
)
```

`E_i > N_i`を厳密比較する。浮動小数点の丸め誤差だけで境界を越えないよう、近接値は同値として不合格にする。

候補全体の`E`と`N`は、それぞれ7件の`E_i`と`N_i`のmedianである。候補は次をすべて満たす場合だけeligibleになる。

1. 代表promptが重複なしでexactly 7件ある。
2. median `E > N`である。
3. `E_i > N_i`のpromptが7件中4件以上ある。
4. その支持promptに`resolved_m=128`がある。
5. その支持promptに`resolved_m!=128`がある。
6. full-model paired sampleが2件以上あり、95%信頼区間の下限が0より大きい。
7. 候補固有のcapabilityと追加証拠がある。

候補Aのmeasurement rowは、baseline/candidateそれぞれのD2D bytes、D2D copy count、
launch count、component/full-model p50/p95、workspace bytes、peak VRAM、fallback count/reasonsを
必須とする。candidateがD2D bytesまたはcopy countを減らすpromptが少なくとも1件あり、
他のpromptで増加しないこと、launch/workspace/peak VRAM/fallbackが増加しないこと、
alias/size/admission safetyが全てtrueであること、direct/copy fidelity binding SHA-256が一致することを
追加ゲートとする。full-model pairにも同じD2D・安全性・fidelityの証拠を要求する。

eligible候補が複数ある場合は、次の順で一意に選ぶ。

1. `E-N`の降順
2. 支持prompt数の降順
3. full-model paired 95%信頼区間下限の降順
4. candidate IDの辞書順

## 4. 出力

schemaは`ullm.aq4_p3_candidate_selection.v1`である。時刻を含めず、同じsemantic inputから同じJSONを生成する。rootは次を持つ。

```text
schema_version
status = selected | no_eligible_candidate
selected_candidate_id
eligible_candidate_ids
policy
input_binding
input_warnings
candidates
```

各candidateには`eligible`、sorted `reason_codes`、median `E`、median `N`、`E-N`、7件の再計算結果、M支持、paired 95%信頼区間、capabilityを記録する。入力file pathやprompt本文、token ID、生成本文は出力しない。

raw inputは順序不変のsemantic SHA-256で束縛する。diagnostic profileは現行schemaにself-hashがないため、file SHA-256で束縛する。複数入力のdigest listはsortする。

## 5. fail-closed条件

- 入力schema/version/statusが不正
- missing/unknown/duplicate fieldまたはduplicate JSON key
- non-finite、負値、不正型、0以下の時間
- semantic hash不一致、identity/case hash不正、複数identityの混在
- measurement/pair重複、unknown candidate、family不一致
- component/full-modelのp50が対応するp95を上回る
- 候補Aの整数由来medianが整数/半整数でない、またはpair専用inputの必要capabilityがfalse/missing
- smoke-only、promotion不可、measurement不可のraw evidence
- 31件以上のpaired sample
- 出力先の既存file
- 読み込み中または書き出し前の入力file identity変更

これらはexit code 2とし、新しいselection outputを発行しない。証拠が正しいschemaだが候補の統計条件を満たさない場合は、exit code 0で`status=no_eligible_candidate`と理由を発行する。

## 次の行動

P2 producerは、diagnostic profileを昇格させるのではなく、同一case・identity・測定scheduleへ束縛したcandidate-selection raw evidenceを生成する。最初の実測ではpaged KVのD2H回数とstream同期回数を必ず明示し、欠測時は候補を選ばない。selectorの出力が`selected`になった後にだけ、選ばれた候補のP3実装を開始する。
