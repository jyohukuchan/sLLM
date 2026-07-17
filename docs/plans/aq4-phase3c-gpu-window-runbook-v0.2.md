# AQ4 Phase 3c GPU window runbook v0.2 — Register-BM8 load admission

このrunbookは v0.1 を履歴として残したまま、`service-stop-window-v0.7-register-bm8-load-admission` 専用に更新したものだ。ここでいう M=1 は実行時の token dispatch のことであり、モデルをロードする際の AQ4 batch-plan admission とは別である。

## v0.6 から分かったこと

v0.6 は stop → lock → R9700 guard → trace → post guard → service 復旧を完走し、停止時間は約4秒だった。`gpu-trace.stderr` の失敗は次だった。

```text
failed to load Qwen3.5 AQ4 linear layer 0:
unsupported backend operation Aq4MatvecBatch for phase ColdPrefill
```

これは GPU identity、権限、PATH、nlink、または service lifecycle の失敗ではない。

- trace は `with_prefill_chunk_tokens(1)` を使い、長さ1では `dispatch_token_for_phase` を選ぶ。M=1 の `run_device_step` は `Aq4MatvecBatch` を実行しない。
- ただし通常の full-model loader は、実行前に M=2..=128 の AQ4 batch plan を全 phase について eager resolve する。gfx1201 の該当 matrix では M=2..7 が `Aq4MatvecBatch`、M=8..128 が `Aq4RegisterBm8` を必要とする。
- v0.6 は前者を要求した一方で `ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL` を unset していた。そのため loader の ColdPrefill admission が失敗した。registry 未登録や M=1 が batch dispatch へ逸脱した事実はない。

従って `Aq4MatvecBatch` を外してはならない。M=1 実行経路を変えず、通常 loader が production と同じ batch-plan cache を admission できるよう、BM8 を17番目の trace guard として加える。

## 固定値

```bash
REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
TRACE_TOOLING_COMMIT=a7cb46e252b2f1a3e045278fe112bce21af05d32
PACKAGE=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package
HYBRID_INPUT="$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-hybrid-diagnostic-v0.1/input/hybrid-input.jsonl"
OUT="$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission"
HIP_GUARD_BINARY="$OUT/query-hip-device-identity"
```

`TRACE_TOOLING_COMMIT` は trace の identity contract 用であり、v0.7 driver はこの commit に対して runtime、trace、staging tooling の差分を fail-closed で検査する。runtime/backend registry は変更しない。

## Required trace guards (17)

次の17個は trace child にだけ `=1` で与える。BM8 は M=1 execution guard ではなく、normal loader の eager plan admission guard である。

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
ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL
ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL
ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL
ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL
ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL
ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL
ULLM_REQUIRE_HIP_BF16_ROW_KERNEL
ULLM_REQUIRE_HIP_TOP1_KERNEL
```

`ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL` は disallowed/unset set に入れない。driver は同じ17要素を `--print-phase3c-trace-guard-requirements` の JSON と照合してから service を停止する。

## service 稼働中の CPU-only 準備

ここまでは service、lock、HIP guard の**実行**、`amd-smi`、trace を実行しない。V620 と 07/16 P3 harness の root/lock/artifact/environment は対象外である。

```bash
test ! -e "$OUT"
mkdir -m 700 "$OUT"
test -x "$REPO/tools/run-aq4-phase3c-service-window.sh"

g++ -std=c++20 -Wall -Wextra -Werror -D__HIP_PLATFORM_AMD__ \
  -I/opt/rocm/include \
  "$REPO/tools/query-hip-device-identity.cpp" \
  -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib \
  -o "$HIP_GUARD_BINARY"
test -x "$HIP_GUARD_BINARY"
sha256sum "$HIP_GUARD_BINARY" > "$OUT/query-hip-device-identity.sha256"

(
  cd "$REPO"
  CARGO_BUILD_JOBS=1 ULLM_BUILD_GIT_COMMIT="$TRACE_TOOLING_COMMIT" \
    cargo build --release -p ullm-engine \
      --bin ullm-aq4-differential-trace \
      --bin ullm-aq4-layer0-family-isolation
)

TRACE_SOURCE_BIN="$REPO/target/release/ullm-aq4-differential-trace"
CPU_BIN="$REPO/target/release/ullm-aq4-layer0-family-isolation"
TRACE_STAGE_DIR="$OUT/trace-binary-staging"
test -x "$TRACE_SOURCE_BIN"
test -x "$CPU_BIN"

python3 "$REPO/tools/stage-aq4-phase3c-trace-binary.py" \
  --source "$TRACE_SOURCE_BIN" \
  --output "$TRACE_STAGE_DIR" \
  --trace-tooling-commit "$TRACE_TOOLING_COMMIT" \
  > "$OUT/trace-binary-staging-create.json"
python3 "$REPO/tools/stage-aq4-phase3c-trace-binary.py" \
  --verify \
  --source "$TRACE_SOURCE_BIN" \
  --output "$TRACE_STAGE_DIR" \
  --trace-tooling-commit "$TRACE_TOOLING_COMMIT" \
  > "$OUT/trace-binary-staging-verify-pre-stop.json"
TRACE_BIN="$TRACE_STAGE_DIR/ullm-aq4-differential-trace"
test -x "$TRACE_BIN"
```

次の自己診断は環境だけを読み、HIP runtime/context/stream/kernel を作らない。`env -i` により、worker の余分な guard や dispatch を変える変数を継承しない。

```bash
TRACE_DIAGNOSTIC_ENV=(
  "HOME=/home/homelab1"
  "PATH=$PATH"
  "HIP_VISIBLE_DEVICES=1"
  "ULLM_HIP_VISIBLE_DEVICES=1"
  "ULLM_SERVED_MODEL_MANIFEST=/etc/ullm/served-models/active.json"
  "ULLM_BUILD_GIT_COMMIT=$TRACE_TOOLING_COMMIT"
  "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4"
  "ULLM_AQ4_MATVEC_SILU_MUL_RPB=8"
  "ULLM_AQ4_MATVEC_ADD_RPB=8"
  "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1"
  "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1"
  "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL=1"
  "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1"
  "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1"
  "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1"
  "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1"
  "ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1"
  "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL=1"
  "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL=1"
  "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1"
  "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL=1"
  "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL=1"
  "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL=1"
  "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1"
  "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1"
  "ULLM_REQUIRE_HIP_TOP1_KERNEL=1"
)
env -i "${TRACE_DIAGNOSTIC_ENV[@]}" \
  "$TRACE_BIN" --print-phase3c-trace-guard-requirements \
  > "$OUT/trace-guard-diagnostic-preflight.json" \
  2> "$OUT/trace-guard-diagnostic-preflight.stderr"
python3 - "$OUT/trace-guard-diagnostic-preflight.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["schema_version"] == "ullm.qwen35_aq4_phase3c_trace_guard_diagnostic.v1"
assert payload["status"] == "valid"
assert len(payload["required_environment"]) == 17
assert payload["linear_stage_guard"]["required_environment"] == {
    name: "1" for name in payload["required_environment"]
}
PY

python3 "$REPO/tools/verify-aq4-layer0-package-embedding-fixture.py" \
  --package "$PACKAGE" \
  --hybrid-input "$HYBRID_INPUT" \
  --output "$OUT/cpu-input-identity.json"
env -i "HOME=/home/homelab1" "PATH=$PATH" \
  "$CPU_BIN" \
    --package "$PACKAGE" \
    --hybrid-input "$HYBRID_INPUT" \
    --output "$OUT/cpu-reference" \
    --stage-stream-stdout \
    > "$OUT/cpu-stages.f32le" \
    2> "$OUT/cpu-reference.stderr"
```

## 実行直前に親エージェントが行うこと

CPU-only 準備を確認した後にだけ、親エージェントが service 稼働中に既存の root-only guard rehearsal を行う。今回はこの runbook の作業者は実行しない。rehearsal evidence は final `OUT` と分離した create-new leaf に保存する。

```bash
GUARD_REHEARSAL_OUT="$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/guard-chain-rehearsal-v0.7-register-bm8-load-admission/attempt-1"
test ! -e "$GUARD_REHEARSAL_OUT"
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py \
  --output "$GUARD_REHEARSAL_OUT" \
  --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission/query-hip-device-identity \
  --health-phase pre-window
```

guard rehearsal が成功し、`OUT` の staged trace と guard binary が存在することを確認してから、下の **一度だけ** の最終 command を親エージェントが直接実行する。

```bash
sudo /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-service-window.sh \
  /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission \
  /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission/query-hip-device-identity \
  --confirm-single-window
```

この driver は既存の `ullm-openai.service` だけを停止・復旧し、stop 前の staging verify と17 guard JSON check を再実行する。pre-stop の検査に失敗した場合は service を停止せず終了する。
