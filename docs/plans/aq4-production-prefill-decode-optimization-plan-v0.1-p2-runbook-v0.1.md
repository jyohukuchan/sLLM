# AQ4 production prefill/decode optimization P2 runbook v0.1

Status: CPU-only preparation complete; R9700/service execution is intentionally pending the parent operator

This runbook implements the P2 section of `aq4-production-prefill-decode-optimization-plan-v0.1.md`.  It is a new-current-identity baseline plan, not a promotion of historical BM8, paged-split, tiled-GEMM, wave-softmax, or pre-07/17 fidelity evidence.

## Frozen preparation

Use only this preparation root:

```text
/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.4
```

| Binding | Value |
| --- | --- |
| clean source commit | `f1a3cf4c86978b3b8900396a0b6a8caff90b97f1` |
| preparation SHA-256 | `1289b145c65340f7a790113f7bbc7db60135c2870aa12a0618a9ec6739fcef49` |
| frozen identity SHA-256 | `e682b50a7c34edd288d759cac146c07679132bbfc40948efb90d42f405f038a2` |
| resident driver SHA-256 | `daf6f12a4d4aaad11b0ef5ffe717372d47b7174271671ae1b9e2f4daf1288753` |
| calibration driver SHA-256 | `b4bbdd6f57169326f269bcccc069538a378b308d61a247de2d77766bb539d641` |
| staged R9700 guard SHA-256 | `0964f145bc2a931a4270d89715e2b86c1d8043d088630da52d74d05f1f40aa1f` |

The active manifest does not expose a product-promotion source commit.  Therefore the preparation records `separated_not_comparable`, rather than asserting that the deployed active binary and clean source binary are identical.  A service window rechecks the active manifest/package/worker hashes and must remain a separate baseline if those differ.

The older `baseline-v0.1`, `baseline-v0.2`, and `baseline-v0.3` directories are superseded preparation receipts only; they contain no P2 GPU measurement evidence. Do not use them for an execution command or comparison.

## Required llama.cpp comparison-baseline exclusion

`llama-qwen35-udq4.service` is the manual Qwen3.5-9B UD-Q4_K_XL comparison
baseline and is boot-disabled as of 2026-07-26. Boot-disabled does not prove
that a unit started manually is inactive. Before every R9700 GPU window,
including the guard rehearsal, stop it and record the required inactive state:

```bash
sudo systemctl stop llama-qwen35-udq4.service
systemctl is-active llama-qwen35-udq4.service
# expected: inactive
```

Do this before the service-window driver stops `ullm-openai.service` or
acquires the R9700 lock. Do not restart the llama.cpp service as part of the
window teardown; its lifecycle is separate from the driver.

Historical P3 prefill `982 tok/s` / decode `56.6%` measurements were captured
while this service was boot-resident, leaving R9700 at junction `85°C`, with
the core clock pinned at maximum and `5.3 GB` of VRAM held by llama-server.
The fixed clock kept relative-run variation small, but these values require the
thermal-condition caveat for absolute reproducibility. P2 and all subsequent
GPU windows must record the stopped-baseline temperature and clock condition.

## Matrix and safe window split

- Cold prefill: prompt `128, 512, 1011, 1024, 1339, 2048, 3584`, with M `1, 8, 16, 32, 64, 128`.  The physical M=1 route is explicitly `all_m1`; its separately labelled production M=1 record is retained in the same run root.
- Decode: start context `16, 128, 512, 1024, 1339, 2048, 3584`, 64 generated tokens, with the same M grid. M applies to the cold context-prefill that establishes each decode state; every subsequent decode iteration remains physically width one. Requested M, resolved M, and fallback status are recorded for every case.
- Cached-prefix chunked: all 42 prefill/M combinations are recorded as `unsupported`; they are never counted as a successful benchmark.
- Every resident case has exactly two warmups and ten measured runs.  A terminal failure stops that window and no output path is reused.

Do not put the entire matrix in one outage.  The frozen plan has 28 serialized single-use R9700 windows:

| Window family | Count | Split rationale |
| --- | ---: | --- |
| normal prefill | 7 | one prompt length per window; each contains all-M=1 plus M-grid production rows |
| normal decode | 7 | one start context and its complete M grid per window |
| detailed rocprof | 6 | `prefill 128/M1`, `1024/M128`, `2048/M64`, `3584/M128`, `decode 16`, `decode 3584` |
| full-vector target path oracle | 8 | one all-M=1 anchor/model load per window |

The detailed-profile timings are diagnostic only and are never mixed into normal p50/p95. `rocprof` raw CSV members, parser output, executor raw JSONL, and sanitized sidecar are hash-bound separately. After all normal windows, a CPU-only reducer seals their measured sanitized rows into one immutable `baseline-measurements.jsonl` with its own manifest and SHA-256 list.

## CPU-only preflight already completed

The following were completed without service/systemd/lock/GPU execution:

- clean detached source build and two nlink=1 staged binaries;
- nlink=1 staged host-only guard build (the guard executable itself was not run);
- preparation/staging verification and normal-window dry run;
- CPU-source-oracle preflight with all CUDA/HIP visibility variables set to `-1`.

Repeat these read-only checks before handing off a service window if any input changed:

```bash
P2=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.4
SRC=/home/homelab1/coding-local/ultimateLLM/uLLM-p2-baseline-source-f1a3cf4c
COMMIT=f1a3cf4c86978b3b8900396a0b6a8caff90b97f1

python3 tools/prepare-aq4-p2-production-baseline.py --output "$P2" --verify
python3 tools/stage-aq4-p2-production-baseline-binaries.py --output "$P2/staging/baseline-binaries" --preparation "$P2" --source-commit "$COMMIT" --verify
python3 tools/stage-aq4-p2-r9700-guard.py --output "$P2/guard/r9700-guard-staging" --preparation "$P2" --source-commit "$COMMIT" --verify
python3 tools/run-aq4-p2-production-baseline-window.py --preparation "$P2" --staging "$P2/staging/baseline-binaries" --window prefill-n128 --output "$P2/windows/prefill-n128" --dry-run
```

Before target path-oracle windows, capture the independent source vector sidecars on CPU.  This can be long and memory-intensive; it neither stops the service nor uses HIP.  Its explicit confirmation is intentional.

```bash
env CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 ULLM_HIP_VISIBLE_DEVICES=-1 \
  python3 tools/capture-aq4-p2-production-source-oracle.py \
    --preparation "$P2" \
    --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
    --output "$P2/source-oracle/source-full" \
    --confirm-cpu-source-capture --threads 1
```

## Parent-only R9700 rehearsal

The following is the explicit guard rehearsal command for the parent operator in a root shell.  It performs the established R9700-only HIP/ASIC read-only guard; it does not stop the service or acquire/create the runtime lock.

```bash
P2=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.4
GUARD="$P2/guard/r9700-guard-staging/query-hip-device-identity"

sudo /usr/bin/python3 tools/run-aq4-phase3c-r9700-guard.py \
  --output "$P2/guard/r9700-guard-rehearsal-v0.1" \
  --guard-bin "$GUARD" \
  --health-phase p2-production-baseline-rehearsal
```

Stop here if the guard is not `valid`, if the BDF is not `0000:47:00.0`, or if its filtered HIP ordinal is not 0.  Do not substitute a V620 device or retry into the same evidence directory.

## Parent-only final single-window command

This is the first normal baseline window.  It is intentionally one service stop/restore and one R9700-only capture; run subsequent window IDs one at a time only after examining its result.

```bash
P2=/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-production-prefill-decode-baseline-v0.4
SRC=/home/homelab1/coding-local/ultimateLLM/uLLM-p2-baseline-source-f1a3cf4c
COMMIT=f1a3cf4c86978b3b8900396a0b6a8caff90b97f1
GUARD="$P2/guard/r9700-guard-staging/query-hip-device-identity"

sudo tools/run-aq4-p2-production-baseline-service-window.sh \
  "$P2" "$GUARD" "$SRC" "$COMMIT" prefill-n128 --confirm-single-window
```

The driver requires `RuntimeDirectoryPreserve=yes`, verifies the complete active AQ4 guard set, probes the pre-existing lock only after the service is stopped, passes a locked inherited FD to the non-root executor, and always attempts one restore.  It sets `HIP_VISIBLE_DEVICES=1` and `ULLM_HIP_VISIBLE_DEVICES=1`; filtered ordinal 0 is therefore the R9700 only.  It does not contain a V620 or SQ8 path.

For detailed profiling, substitute one of the six `profile-*` IDs from `window-plan.json`; the same driver automatically wraps only those windows in `rocprofv3` with kernel, HIP-runtime, and memory-copy traces.  For a target path anchor after the CPU source capture, use exactly one anchor per invocation:

```bash
sudo tools/run-aq4-p2-production-path-oracle-service-window.sh \
  "$P2" "$GUARD" "$SRC" "$COMMIT" "$P2/source-oracle/source-full" \
  p2-oracle-anchor-prefill-all-m1-n128-m1 --confirm-single-window
```

Both service-window drivers intentionally reject untracked or modified tool code and an unclean detached source worktree.  Satisfy those gates in the execution worktree before entering a root window.  They also reject any pre-existing output for the requested ID, so a failed window is evidence, not a retry target.

## Post-window CPU analysis

For every successful target anchor, compare one source/target pair without retaining a full logit matrix:

```bash
CASE=p2-oracle-anchor-prefill-all-m1-n128-m1
python3 tools/compare-aq4-p2-production-oracles.py \
  --source "$P2/source-oracle/source-full" \
  --target "$P2/source-oracle/target/$CASE" \
  --case-id "$CASE" \
  --output "$P2/source-oracle/comparisons/$CASE.json"
```

After all 14 normal windows have completed, seal the immutable baseline JSONL. This rejects a partial matrix, verifies every normal window's `SHA256SUMS`, raw-trace/sidecar binding, M resolution, and exactly ten measured runs per planned case.

```bash
python3 tools/seal-aq4-p2-production-baseline-jsonl.py \
  --preparation "$P2" \
  --windows-root "$P2/windows" \
  --output "$P2/windows/baseline-measurements.jsonl"

python3 tools/seal-aq4-p2-production-baseline-jsonl.py \
  --output "$P2/windows/baseline-measurements.jsonl" \
  --verify
```

After all six detailed-profile windows have also completed, build the report:

```bash
python3 tools/build-aq4-p2-production-bottleneck-report.py \
  --preparation "$P2" \
  --windows-root "$P2/windows" \
  --output "$P2/bottleneck-report.json"
```

The report ranks wall-time p50/p95, detailed-profile kernel families, launch/sync, and transfer evidence.  It deliberately remains blocked if current-identity workspace or semantic-fallback evidence is still `not_observed`; do not select an optimizer family or call P2 complete until those blockers are resolved from a new trace/sidecar rather than inferred as zero.
