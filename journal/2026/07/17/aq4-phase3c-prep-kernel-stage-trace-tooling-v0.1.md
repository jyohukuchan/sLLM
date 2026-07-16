# AQ4 Phase 3c-prep kernel stage trace tooling v0.1

## 前回の要点

- Phase 3bで07/14の差はproduction packageを直接loadしたM=1/cold診断のlayer 0から始まると分かり、warm state、M>1、RoPE、paged KVというH6は棄却された。
- fused HIP kernelとCPU参照の静的レビューでは、有効なAQ4 payloadで07/14規模の差を直接説明する高確信度の算術欠落は未発見だった。従ってH5を実production M=1の中間値で局所化する必要がある。
- GPU windowは未承認である。この変更ではGPUを列挙・初期化・実行しない。

## 今回の変更点

### production M=1の診断read-back

- `PackageLinearAttnResidentStepLayer::visit_intermediate_trace_buffers` を追加し、完了済みのresident linear-attention layerが保持するdevice bufferを診断用に列挙できるようにした（`crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs`）。追加kernel、fusionの切替、HIP/C API/ABIの変更はない。
- `Qwen35Aq4IntermediateTraceObserver` に明示的opt-inのstage callbackを追加した（`crates/ullm-engine/src/qwen35_aq4_model_runtime.rs`）。既存observerはdefaultでopt-outのため、通常のproduction実行と従来traceの出力は変えない。
- 既存streamの`copy_to_host`と同期だけを使い、layer 0の次の10 stageを256 KiB chunkでD2Hする。recurrent state全体をhost常駐させず、有限値・厳密なbuffer byte長・連続chunk順を検証する。

  1. `qkv_dequant_row_scale`
  2. `z_dequant_row_scale`
  3. `recurrent_gate`
  4. `recurrent_beta`
  5. `recurrent_state_after`
  6. `recurrent_output`
  7. `attention_residual`
  8. `post_norm`
  9. `mlp_activation`
  10. `layer_output`

### trace artifactと比較器

- `ullm-aq4-differential-trace` に、既存payloadを変えない明示的な`--enable-intermediate-trace --enable-linear-stage-trace`を追加した。
- opt-in時だけ`kernel-stages.jsonl`（固定座標のsummary）と`kernel-stages.f32le`（Phase 1 CPU `StageEmitter`と同じframed f32le protocol）を出す。各rowは最大16 MiBで、terminal frameを必須にする。manifestと`SHA256SUMS`にsidecarのschema・stage順・byte数・guard identityを記録する。
- traceがpublication前に失敗した場合は、`OUTPUT.incomplete-PID` を削除せず残し、エラーにそのpathを明記する。これは再試行用ではなくfailure evidenceであり、terminal frame/manifest/`SHA256SUMS`を満たさない不完全rootは比較・promotionに使わない。
- stage modeは、R9700のglobal device index 1、`HIP_VISIBLE_DEVICES=1`/`ULLM_HIP_VISIBLE_DEVICES=1`、`gfx1201`、layer 0で必要な7つのHIP fusion guardをfail-closedで要求する。`ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING`と`ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA`は拒否し、意図せずfusionを外した比較を防ぐ。
- `tools/compare-aq4-layer0-cpu-gpu-stage-stream.py` を追加した。既存Phase 1 CPU stage streamの3 context・最終timestepだけを、GPU sidecarの同一identityとfull f32 element単位で突き合わせる。出力はstageごとのmax abs、relative L2、cosineを含むhash付きJSONであり、入力frameを順次破棄する。
- `tools/verify-aq4-layer0-package-embedding-fixture.py` を追加した。既存hybrid inputの全context rowをpackageのBF16 passthrough embedding行とraw f32 bytesで照合するCPU-only preflightである。AQ4のlayer 0入力差をGPU差と取り違えない。

### CPU-only検証

以下はすべてGPUを実行せずに完了した。

```text
cargo fmt --check
cargo build -p ullm-engine --bin ullm-aq4-differential-trace
cargo test -p ullm-engine --bin ullm-aq4-differential-trace -- --test-threads=1
cargo test -p ullm-runtime-sys cpu_aq4_matvec_ --lib -- --test-threads=1
python3 -m py_compile tools/verify-aq4-layer0-package-embedding-fixture.py tools/compare-aq4-layer0-cpu-gpu-stage-stream.py
pytest -q tests/test_aq4_phase3c_stage_tooling.py
git diff --check
```

- Rust trace unit test: 11件成功。新CLIの明示opt-in、stage sidecar protocol、guard、既存payload不変を含む。
- runtime CPU AQ4 matvec test: 10件成功。
- Python tooling test: 2件成功。synthetic framed streamのstage差検出とBF16 embeddingのbit-exact照合を含む。
- build時には既存の`-Wsubobject-linkage` warningのみが出た。新規warning/errorはない。

## 次の行動

1. Phase 3c実行runbookに、今回のtooling commit、固定fixture、R9700 lock、全production HIP guard、単発実行・evidence保存規則を明記する。
2. ユーザーがGPU windowを明示承認した後だけ、runbookのCPU input identity preflight、CPU stage stream、GPU sidecar、比較器を一回ずつ実行する。
3. 比較で最初に有意差が出たstageを根拠として、Phase 4の修正候補を別途承認にかける。今回の変更にはkernel fixを含めない。
