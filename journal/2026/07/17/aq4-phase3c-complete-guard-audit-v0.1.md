# AQ4 Phase 3c complete guard audit v0.1

## 前回の要点

- `aq4-phase3c-trace-binary-nlink-staging-v0.1.md` の第5回 service-stop window は、停止・既存lock取得・R9700 guard・trace staging・service復旧まで成功したが、trace は layer 0 load の capability gate で fail-closed した。
- 直接原因は driver が `ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1` と `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1` を渡しておらず、linear M=1 load が要求する4 RuntimeFeature のうち2件を prove できなかったことである。service は約3秒で正常復旧している。
- V620、07/16に停止したP3 harnessのlock/root/artifact/environmentにはアクセス・変更していない。

## 今回の変更点

### 機械抽出と実package確認

- 次の機械的抽出を実施した。`crates`、`runtime`、`tools` の Rust/C++ sourceから `ULLM_REQUIRE_HIP_` literal を抽出した結果は50件であり、`ULLM_REQUIRE_HIP_UNKNOWN_KERNEL` はworker負試験用のsentinelで、実運用名は49件である。

  ```bash
  rg -o --no-filename --glob '*.rs' --glob '*.inc' --glob '*.cpp' --glob '*.h' \
    'ULLM_REQUIRE_HIP_[A-Z0-9_]+' crates runtime tools | sort -u
  ```

  ```text
  ULLM_REQUIRE_HIP_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_TOP1_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL
  ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL
  ULLM_REQUIRE_HIP_AQ4_ROW_KERNEL
  ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL
  ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_F32_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_F32_FLASH2_KERNEL
  ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL
  ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_DEPTHWISE_CONV1D_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL
  ULLM_REQUIRE_HIP_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL
  ULLM_REQUIRE_HIP_ROPE_KERNEL
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL
  ULLM_REQUIRE_HIP_SILU_MUL_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL
  ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL
  ULLM_REQUIRE_HIP_TOP1_KERNEL
  ULLM_REQUIRE_HIP_TOP1_PAIRS_KERNEL
  ULLM_REQUIRE_HIP_UNKNOWN_KERNEL
  ```

- trace binaryは `layer_indices: None`、`prefill_chunk_tokens(1)` であり、model runtime はmanifestの全layerをloadしてM=1でdispatchする。read-onlyで実package manifestを確認すると、32層中linear-attentionは24層（0,1,2,4,...,30）、self-attentionは8層（3,7,11,15,19,23,27,31）であった。全self層のQ projectionは`[8192,4096]`、hiddenは4096なので、sourceの `q_rows == 2 * hidden` 分岐により全てQwen3.5 gated layoutである。

### 確定したguard集合

- **layer 0 production M=1 linear pathの完全集合は9件**である。

  ```text
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL
  ```

  `PackageLinearAttnResidentStepLayer::load_with_registry` のfeature gateはrecurrent、qkv prepare、AQ4 batch、qkv-prepare batchの4件を要求する。`run_device_step` はRMSNorm、QKV/Z/gate/beta fused AQ4、prepare、recurrent、segmented RMSNorm-SiLU、AQ4 matvec-add、AQ4 fused MLPを実行する。これにより既存7件へ今回の2件を足した9件がlayer 0だけの完全集合である。

- **固定Phase 3c traceを全層load・実行し、staging fallbackを許さず完走する集合は16件**である。上記9件に、gated self-attention loadの5件と、実packageのBF16 embedding / full-logit top-1の2件を加える。

  ```text
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL
  ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
  ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
  ULLM_REQUIRE_HIP_TOP1_KERNEL
  ```

- pair/triple AQ4 guards、register-BM8、plain paged-KV-write、paged-decode-split、sequence recurrent等はこの固定M=1/gated branchには不要である。特にself QKV tripleが失敗しても、pair→generic AQ4へfallbackし、`ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1`がgeneric pathをHIP-onlyにする。pair/tripleを強制すると有効なgeneric HIP fallbackまで不必要にfail-closedにするため、16件には含めない。workerの30件をそのまま流用すると14件の不要なsupersetとなり、追加scratch probeやdispatch変更を招くため採用しない。

### 起動前自己診断

- `qwen35_aq4_layer_runtime.rs` にlayer0の4 RuntimeFeatureと9件のguard定数を明文化し、load時の`require_features`もその定数を使用するようにした。
- `ullm-aq4-differential-trace` に `--print-phase3c-trace-guard-requirements` を追加した。この経路は環境値だけを確認し、HIP runtime context、stream、device memory、kernel launchを作らない。必要16件、可視device指定、固定branchを崩す15変数を一括検査し、欠落・誤値をすべてJSONで列挙してexit 1する。
- CPU-only検証結果:

  ```text
  CARGO_BUILD_JOBS=1 cargo test -p ullm-engine --bin ullm-aq4-differential-trace -- --test-threads=1
  # 13 passed

  CARGO_BUILD_JOBS=1 cargo test -p ullm-engine m1_linear_stage_guard_set_covers_every_load_feature_and_step_guard -- --test-threads=1
  # 1 passed (727 filtered)

  CARGO_BUILD_JOBS=1 cargo check -p ullm-engine --bin ullm-aq4-differential-trace
  # success
  ```

- 未設定の診断はrequired 16件すべてをJSON errorに列挙し、全16件を`=1`、固定branchを崩す変数をunsetにした診断は`status=valid`となることをCPU-onlyで確認した。

## 次の行動

1. service-stop driverとrunbookを、16件・固定branch unset・新規v0.6 output root・起動前診断に更新する。
2. source診断を先にcommitし、そのcommitをtrace tooling identityとしてrunbook/driverに固定してcommitする。
3. 新binaryをCPU-onlyでbuild/stageし、service稼働中にdiagnosticと既存R9700 HIP+amd-smi guard chainをリハーサルする。
4. 上記が安定して成功した場合だけ、一度だけservice-stop windowを実行し、成否にかかわらず直ちにserviceを復旧する。trace失敗時の同一window内再試行はしない。
