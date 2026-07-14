# P3候補事前監査: paged KV block-table validationのD2H・stream同期削減

## 前回の要点

- P3候補はP2 R9700のoperator別wall time、実M、D2H/H2D、stream同期、workspace、fallbackを得るまで仮説に留める。
- 計画上の第一候補は、prefillのpaged KV chunk writer/readerで毎回実行されるblock-tableのD2H検証と同期の削減である。
- production trace/profileは、component/full_model/production_serverを混同せず、同一case identityへhash-boundしなければならない。

## 今回の変更点

コード、GPU、サービスは変更していない。以下は、現行実装を読んで整理したP2後の実装境界である。

## 1. 現行の呼出し経路と対象範囲

### C++ runtimeの対象関数

- `runtime/src/ullm_runtime_api_attention.inc:2566`付近の`ullm_runtime_paged_kv_write_chunk_f32`
  - HIP経路では、geometry/buffer検査の後に`validate_paged_chunk_block_table_hip`を呼ぶ。
  - 検証後に`paged_kv_write_chunk_f32_hip_kernel`を同じstreamへenqueueする。
- 同ファイル`paged_causal_gqa_chunk_f32_impl`（`2710`付近）
  - readerのHIP経路も同じ`validate_paged_chunk_block_table_hip`を呼ぶ。
  - plain/gatedの公開ABIはこの共通実装を使う。
- `runtime/src/ullm_runtime_parts/part_01.inc:4552`
  - `validate_paged_chunk_block_table_hip`が`table_entries * 4` bytesをD2Hし、`synchronize_hip_staging`でstreamを同期し、host側で全entryを範囲検査する。
  - したがって、chunk writerとchunk readerが各1回ずつ呼ばれると、1chunk・1self-attention layerあたりD2H 2回＋同期2回になる。
- `runtime/src/ullm_runtime_parts/part_01.inc:4395`以降
  - writer/readerのHIP kernel launcherはblock tableをdevice pointerとしてそのまま渡す。
  - `runtime/src/ullm_runtime_hiprtc_sources.inc:5556`以降のchunk kernelは、範囲外block IDを検出すると書き込みをreturnする。readerは出力を0にしてreturnするが、hostへエラーを返すstatus bufferはない。

### Rust FFI/ABI

- 公開C ABIは`runtime/include/ullm_runtime.h:995`以降の
  `ullm_runtime_paged_kv_write_chunk_f32`、
  `ullm_runtime_paged_causal_gqa_chunk_f32`、
  `..._sigmoid_gate_f32`である。
- FFI宣言と安全ラッパーは`crates/ullm-runtime-sys/src/lib_parts/part_00.rs:894`以降、`5882`以降にある。
- 現在の`RuntimeBuffer`はopaque handleで、device bufferのhost shadowは持たない。Rust側の`Vec<u32>`とdevice上のblock tableが一致していることをruntimeは知らない。
- `ULLM_RUNTIME_ABI_VERSION`は現在1。新しい公開関数または引数を追加する場合は、既存ABIを壊さない加算的変更か、ABI/identityの更新方針を先に固定する必要がある。

### AQ4 engine呼出し

- `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs`
  - `PackageSelfAttnResidentStepLayer`は、load時にblock tableの長さ・範囲をCPUで検査し、device bufferへ一度コピーする（`1763`、`1854`付近）。
  - sequence prefillは`run_device_step_sequence`内で、chunk writerを`3230`付近、chunk readerを`3311`付近から呼ぶ。ここが候補の主経路である。
  - M=1のdecode writer/fused writerと通常paged decodeは、現状chunk validatorを通らないため、候補1の直接対象ではない。
- `crates/ullm-engine/src/qwen35_aq4_model_runtime.rs:441`付近
  - product geometryは`context_length / block_size`からcache blocksを計算し、identity block table (`0..cache_blocks`)を生成する。
  - 現行Qwen3.5 AQ4 productでは`block_size=256`、context 4096なら16 entriesである。
- `crates/ullm-engine/src/decoder.rs:1940`付近の`PagedDecodeState::new`もhost側でblock tableを検査してからdeviceへuploadする。ただし、Qwen3.5 AQ4 resident sequenceの主経路はlayer runtime側の別bufferであるため、ここだけの変更では候補全体を覆えない。

### registry / state contract

- `crates/ullm-engine/src/backend_operation_registry.rs`
  - `OperationKind::PagedKvWrite`と`OperationKind::PagedCausalGqaRead`、`OperationGeometry::{PagedKvWrite,PagedCausalGqaRead}`、`StateEffect`、`OperationResolutionTrace`を所有する。
  - 現行descriptorはwriterが`InPlace`、readerが`ReadOnly`で、paged KV stateを明示する。
  - `OperationExecutionRecord`はimplementation ID/phase/statusだけを持ち、D2H bytesや同期回数を持たない。候補を採用する場合は、既存IDを黙って再利用せず、validated-table経路を識別できるimplementation/versionまたはexecutor sidecarの契約が必要である。

## 2. 推奨する実装境界

### 推奨案: immutable validated-table metadataをbufferへ束縛する

単純にHIP検証を削除するのは、foreign C callerや壊れたdevice tableをfail-openにするため不採用とする。次の二経路を保持する。

1. **untrusted/default経路**: 現行のD2H＋同期＋host範囲検査を残す。
2. **trusted/validated経路**: Rust側でhost block tableを完全検査し、device upload後にbufferへ「table entries、cache blocks、検証世代」を束縛する。chunk writer/readerは、geometryと束縛が一致する場合だけD2H検証を省略する。

この場合、`ullm_runtime_buffer`（`runtime/src/ullm_runtime_parts/part_00.inc:1524`付近）に小さな検証metadataを追加し、次のhost/device書込み操作でmetadataを無効化する必要がある。

- `ullm_runtime_buffer_copy_from_host`
- `ullm_runtime_buffer_copy`
- `ullm_runtime_buffer_zero`
- buffer destroy/再利用境界

Rust側からは、upload直後に`mark_validated_paged_block_table(table_entries, cache_blocks)`のような安全ラッパーを呼ぶ。`RuntimeBuffer`のraw pointerを外へ出さず、validated metadataはruntimeが管理する。

### 採用しない案: device status bufferだけで検証する

chunk kernel内にstatus flagを書かせる方式はD2Hを完全には消せるが、異常をhostがいつ観測するかが曖昧になる。少なくともstatusのD2Hまたは次の同期点が必要で、現行のfail-closed status契約と相性が悪い。P2で検証失敗の観測頻度が高い場合を除き、第一実装にはしない。

### 変更対象ファイルと責務

| 順序 | 対象 | 関数・型 | 責務 |
|---:|---|---|---|
| 1 | `runtime/include/ullm_runtime.h` | validated-table API（加算的に追加する場合） | ABI名、引数、ABI/identity方針を凍結 |
| 1 | `runtime/src/ullm_runtime_parts/part_00.inc` | `ullm_runtime_buffer`、buffer copy/zero | metadata保持と全書込み時の無効化 |
| 2 | `runtime/src/ullm_runtime_api_attention.inc` | chunk writer/reader公開wrapper | trusted metadata一致時だけvalidatorをskip。geometry検査は常に残す |
| 2 | `runtime/src/ullm_runtime_parts/part_01.inc` | `validate_paged_chunk_block_table_hip` | untrusted fallbackとして保持。必要なら共通判定helperを追加 |
| 2 | `runtime/src/ullm_runtime_hiprtc_sources.inc` | chunk writer/reader kernels | 通常は変更不要。status方式を採る場合だけ専用laneで変更 |
| 3 | `crates/ullm-runtime-sys/src/lib_parts/part_00.rs` | FFI宣言、`RuntimeBuffer` method | C ABIを安全なRust型へ束縛 |
| 3 | `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs` | resident load、sequence writer/reader | CPU検査→upload→markの一回化。request resetではtable metadataを変更しない |
| 3 | `crates/ullm-engine/src/decoder.rs` | `PagedDecodeState::new`（必要な範囲だけ） | generic paged stateでも同じ契約を使うかを判断。AQ4主経路との二重実装を避ける |
| 4 | `crates/ullm-engine/src/backend_operation_registry.rs` | descriptor、capability、trace | validated経路のimplementation/version、required feature、state effectを明示 |
| 5 | `tools/`、`tests/` | 新規P3 raw runner/validator | D2H、sync、launch、workspace、correctness、identityをrawから再計算 |

## 3. 依存順序

1. **P2 profile判定**: `paged_validation` familyのexclusive wall time、実M、D2H/H2D、同期、fallback、workspaceを確定する。現在の`tools/profile-aq4-p2-family-exclusive.py`はkernel timelineの診断専用で、`measurement_eligible=false`、D2H/sync countは収集しないため、candidate選抜にそのまま使わない。
2. **ABI/metadata契約の凍結**: trusted markerの生成条件、書込み時無効化、untrusted fallback、エラー/status、ABI version、implementation IDを決める。
3. **CPU oracle/state契約**: block tableの長さ・範囲、非恒等mapping、block boundary、cache position、reset後の再利用をM gridとcontext gridで固定する。
4. **runtime ABI/kernel**: C++ buffer metadataとchunk wrapperを実装し、旧経路のfail-closedテストを先に通す。
5. **Rust FFI/engine binding**: resident layerのupload後markとsequence writer/readerのtrusted dispatchを結ぶ。loader/constructorの初期同期は測定から除外する。
6. **registry/session統合**: descriptor/capability/implementation traceを直列で更新し、trusted経路が選択されたことを証跡へ出す。
7. **R9700 component測定**: M=1、2、8、16、32、64、128、contexts 16、128、512、1024、1339、2048、3584、境界chunkを一件ずつ測る。
8. **full-model offline → direct worker → production server**: 同一case/identityで順に昇格する。GPU実行は常に一件ずつで、失敗/OOMもimmutableに残す。

## 4. 並列可能な作業レーンと所有ファイル

ABI/descriptorを凍結した後に、次の三レーンまで並列可能である。

### P3-A runtime ABI/kernel（専有）

- 所有: `runtime/include/ullm_runtime.h`、`runtime/src/ullm_runtime_parts/part_00.inc`、`runtime/src/ullm_runtime_api_attention.inc`、`runtime/src/ullm_runtime_parts/part_01.inc`、必要時のみ`runtime/src/ullm_runtime_hiprtc_sources.inc`、runtime-sysのFFI専有部分。
- 目的: validated marker、無効化、trusted/untrusted dispatch、legacy fallback、C++ unit/FFI tests。
- 他laneと同じpublic header・Rust FFIを同時編集しない。ABIの最終差分をC laneから統合窓へ渡す。

### P3-B CPU oracle/state（専有）

- 所有: `crates/ullm-engine/src/cpu_reference_executor.rs`、新規CPU oracle module/test、既存paged cache packingの読み取り利用。
- 目的: M=1基準との逐次chunk差分、block-table permutation、境界、cache/state/reset、finite、hidden/logit/top-k/greedy。
- device ABIやproduction traceは生成しない。既存`decoder.rs`の共通型を変更する場合は、C laneと統合担当の承認後に直列化する。

### P3-D evidence（専有）

- 所有: `tools/`の新規P3 raw capture/validator、`tests/test_p3_*`、candidate schema文書。
- 目的: transfer/sync/launch/workspace/resource/correctness/state transitionをraw JSONLから再計算し、case/identity/trace SHAを束縛する。
- 既存P2 profilerとproduction trace validatorのsummaryを信頼せず、candidate専用sidecarを作る。prompt/response/token IDは保存しない。

### 直列のP3-C registry/engine integration

- 所有: `crates/ullm-engine/src/backend_operation_registry.rs`、`qwen35_aq4_layer_runtime.rs`、必要な`qwen35_aq4_model_runtime.rs`/`qwen35_aq4_session.rs`。
- P3-AとP3-Bが合格してから編集する。descriptor、capability、workspace、implementation ID、session接続を一つの統合点で更新する。
- その後のR9700、full-model、worker、production server、activationはすべて直列である。

## 5. 証拠スキーマの提案

現行`ullm.production_execution_trace.v1`/`ullm.production_executor_record.v1`のoperator recordは、implementation、shape、workspace、invocation countを持つが、D2H/H2Dとstream同期の実数を持たない。`profile-aq4-p2-family-exclusive.v1`も診断専用で、kernel intervalのfamily exclusive timeだけを保持する。

候補専用のbounded raw sidecar（例: `ullm.aq4_p3_candidate_measurement.v1`）へ次を追加する。

```json
{
  "schema_version": "ullm.aq4_p3_candidate_measurement.v1",
  "candidate_id": "paged-kv-table-validation-v1",
  "case_id": "...",
  "case_sha256": "...",
  "identity_sha256": "...",
  "variant": "baseline|candidate",
  "phase": "cold_prefill|cached_prefix|decode",
  "operator_kind": "PagedKvWrite|PagedCausalGqaRead",
  "implementation_id": "...",
  "geometry": {"m": 128, "cache_start": 0, "cached_prefix_len": 0, "block_size": 256, "cache_blocks": 16, "table_entries": 1},
  "block_table_contract": {"source": "cpu_validated", "table_sha256": "...", "validated_once": true, "mutation_epoch": 0},
  "transfer": {"d2h_count": 0, "d2h_bytes": 0, "h2d_count": 0, "h2d_bytes": 0},
  "sync": {"stream_sync_count": 0, "stream_sync_wait_ns": 0},
  "launch": {"kernel_count": 1, "kernel_names": ["..."]},
  "workspace": {"planned_bytes": 0, "observed_peak_bytes": 0},
  "correctness": {"finite": true, "max_abs_diff": 0.0, "cache_diff": 0.0, "greedy_match": true},
  "state": {"prepared": 1, "committed": 1, "discarded": 0, "reset_complete": true},
  "failure": null
}
```

実装上は1行1invocationのJSONLとし、validatorが`d2h_count`、`sync_count`、workspace算術、state遷移をraw eventから再計算する。producerの`passed`や集計値は採用根拠にしない。trusted markerが無効化された後にD2Hへ戻った場合も、fallback eventとして記録する。

## 6. 正しさ・性能ゲート

### 正しさ

- CPU/source/path oracleでM=1とM=2/8/16/32/64/128を比較する。
- context/cache gridは16、128、512、1024、1339、2048、3584とし、`cache_start`がblock境界の直前・境界・直後になるcaseを含める。
- identity tableだけでなく、`[2,0,3,...]`のような非恒等mapping、短いtable、範囲外ID、`table_entries`不足、cache position overflowを検査する。
- writer後のKV cache、reader output、finite、hidden/logit/top-k/greedyをall-M=1との差分で確認する。chunk writerは`InPlace`、readerは`ReadOnly`のstate effectを維持する。
- `copy_from_host`、buffer-to-buffer copy、zeroでvalidated metadataが無効化され、次の操作がuntrusted D2H経路またはfail-closed errorへ戻ることを確認する。
- cancel、publish failure、EOS/length、reset、次requestでKV stateがbaselineと同じ初期状態へ戻ることを確認する。invalid tableがkernelの「return/zero」だけで成功扱いになってはいけない。
- 旧APIを呼ぶforeign/untrusted callerは従来通りD2H検証され、異常時に`ULLM_STATUS_INVALID_ARGUMENT`または明示的runtime errorになる。

### 性能・資源

- trusted sequence pathで、chunk writer/readerの検証由来D2H bytesとstream sync countが0になることをraw evidenceで確認する。初回load/uploadの同期は測定区間外へ固定する。
- 追加metadataはconstant-sizeで、KV cache/workspace/VRAMを増やさない。resident loadのplanned/observed memory、OOM、peak headroomを再計算する。
- baselineとcandidateを同一binary/package/manifest/driver/power/warmup/repeatで比較し、prefill p50 5%超またはp95 10%超の回帰があれば停止する。
- family exclusive timeの改善だけでは昇格しない。full-model offline、direct worker、production serverでTTFT/prefill wall timeの改善が測定誤差を超え、短文脈decode p50が5%以内であることが必要である。
- P2の候補選抜では、recoverable share `E` がノイズ閾値 `N=max(5%, 3×baseline CV, 2×CI half-width/p50)`を上回り、代表7点のうち4点以上、M=128と別Mの双方で現れ、full-model paired 95% CIが0を跨がないことを要求する（P2結果で最終値を確定する）。

## 7. 代替候補へ切り替える条件

1. P2でpaged-validation familyの実wall timeまたはD2H/syncが観測できない、または`E < N`、代表点4/7未満、M=128または別Mが欠落する場合は候補1を実装しない。
2. trusted markerのABI追加がidentity/rollbackを壊す、foreign callerのfail-closedを保てない、またはinvalid tableが成功/zero outputへ漏れる場合は即時停止する。
3. full-model paired CIが0を跨ぐ、prefill改善が測定誤差内、VRAM/OOM/workspace policyを悪化させる場合はcandidateを不採用とする。
4. projection familyが`E`を上回る場合は候補2（AQ4 BM8/register shape coverage・scale residency）へ切り替える。主な境界は`runtime/src/ullm_runtime_api_aq4.inc`、`runtime/src/ullm_runtime_hiprtc_sources.inc`、専用kernel testである。
5. projectionが支配的でなく、linear/recurrentまたはdense self-attention chunkのwall timeが支配的なら候補4（chunk execution）へ切り替える。`qwen35_aq4_layer_runtime.rs`のsequence dispatchとstate transactionを中心に、paged table候補とは別familyとして測る。
6. launch/syncが広いfusion境界で支配的なら候補3（projection/norm/activation/residual fusion）へ切り替える。ただし数値差・alias・rollbackリスクが高いため、P2で支配性が明示された場合だけ着手する。

## 次の行動

- 親エージェントはP2 R9700 profileで、chunk writer/readerの実D2H、stream同期、family wall time、実M、workspace、fallbackを再計算する。
- 候補1の`E/N`条件とfull-model paired CIを満たす場合だけ、ABI/validated metadata契約を凍結してP3-A/B/Dを並列開始する。
- 条件を満たさない場合は、上記の切替条件に従ってBM8/registerまたはchunk/fusion候補を選び、同じ証拠スキーマで比較する。
