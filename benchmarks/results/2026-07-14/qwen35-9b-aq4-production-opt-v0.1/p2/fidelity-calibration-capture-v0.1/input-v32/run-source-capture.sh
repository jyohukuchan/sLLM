#!/usr/bin/env bash
set -euo pipefail

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
ROOT="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2"
MODEL_DIR=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B
SPLIT_ROOT="$ROOT/fidelity-holdout-split-v0.1"
INPUT="$ROOT/fidelity-calibration-capture-v0.1/input-v32"
LEGACY_ORACLE="$ROOT/source-oracle-v2"
RUN="$ROOT/fidelity-calibration-capture-v0.1/attempts/source-attempt-v32-20260714T180609Z"
OUTPUT="$RUN/source-full"
STDOUT_LOG="$RUN/source.stdout.log"
STDERR_LOG="$RUN/source.stderr.log"
TIME_LOG="$RUN/time-v.txt"
VMSTAT_LOG="$RUN/vmstat.log"
EXPECTED_CASES_SHA=53f256bc8f5ed4036cfb1a9a98c0c9d9197bb980e1ef91d7ff01cf73001369a8
EXPECTED_PLAN_SHA=1b4f8c244e922ab73c0bb026216d8333a9cfe57c23e6695c4141554d117693c0

for path in "$OUTPUT" "$STDOUT_LOG" "$STDERR_LOG" "$TIME_LOG" "$VMSTAT_LOG"; do
  python3 - "$path" <<'PY'
import os
import sys
if os.path.lexists(sys.argv[1]):
    raise SystemExit(f"refusing to reuse existing attempt path: {sys.argv[1]}")
PY
done
test -x /usr/bin/time
command -v timeout >/dev/null
command -v vmstat >/dev/null
test -f "$INPUT/cases.json"
test -f "$INPUT/plan.json"
test -d "$MODEL_DIR"
test -d "$LEGACY_ORACLE"
mkdir -p "$RUN"

python3 - "$INPUT/cases.json" "$INPUT/plan.json" "$LEGACY_ORACLE" "$SPLIT_ROOT" "$EXPECTED_CASES_SHA" "$EXPECTED_PLAN_SHA" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

cases_path, plan_path, legacy_root, split_root, expected_cases, expected_plan = sys.argv[1:]
def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
if digest(cases_path) != expected_cases or digest(plan_path) != expected_plan:
    raise SystemExit("fixed input SHA differs")
plan = json.loads(open(plan_path, encoding="utf-8").read())
if plan["source"]["threads"] != 32 or plan["execution_contract"]["source_torch_threads"] != 32 or "--threads 32" not in plan["source"]["command_template"]:
    raise SystemExit("fixed input plan thread contract differs")
if os.path.realpath(plan["source"]["cases"]) != os.path.realpath(cases_path) or os.path.realpath(plan["active"]["cases"]) != os.path.realpath(cases_path):
    raise SystemExit("fixed input plan cases path differs")
split = json.loads(open(pathlib.Path(split_root) / "split-manifest.json", encoding="utf-8").read())
excluded = split.get("attempt2_exclusions", {}).get("case_ids", [])
legacy_ids = {json.loads(line)["case_id"] for line in (pathlib.Path(legacy_root) / "payload.jsonl").read_text(encoding="utf-8").splitlines() if line}
if not legacy_ids or not legacy_ids.issubset(set(excluded)):
    raise SystemExit(f"legacy row overlap is not bound by split exclusion: {sorted(legacy_ids - set(excluded))}")
print(json.dumps({"status":"preflight_identity_ok", "legacy_case_ids":sorted(legacy_ids), "excluded_case_ids":sorted(excluded)}, sort_keys=True))
PY

CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
python3 - "$MODEL_DIR" "$RUN" <<'PY'
import json
import math
import os
import pathlib
import shutil
import sys
import torch
model = pathlib.Path(sys.argv[1]); run = pathlib.Path(sys.argv[2])
if torch.cuda.is_available() or torch.cuda.device_count() != 0:
    raise SystemExit("GPU is visible during CPU-only preflight")
index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = sorted(set(index["weight_map"].values()))
checkpoint_bytes = sum((model / name).stat().st_size for name in shards)
meminfo = {}
for line in pathlib.Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
    if ":" in line:
        key, rest = line.split(":", 1); fields = rest.split()
        if key in {"MemTotal", "MemAvailable"} and fields: meminfo[key] = int(fields[0]) * 1024
required_headroom_bytes = math.ceil(checkpoint_bytes * 2.0)
expected_vector_bytes = 24 * (4096 + 248320) * 4
required_free_bytes = math.ceil(expected_vector_bytes * 1.2)
free_bytes = shutil.disk_usage(run.parent).free
if meminfo.get("MemAvailable", 0) < required_headroom_bytes or free_bytes < required_free_bytes:
    raise SystemExit("memory/disk preflight failed")
print(json.dumps({"status":"passed", "gpu":{"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"hip_visible_devices":os.environ.get("HIP_VISIBLE_DEVICES"),"rocr_visible_devices":os.environ.get("ROCR_VISIBLE_DEVICES"),"torch_cuda_available":torch.cuda.is_available(),"torch_cuda_device_count":torch.cuda.device_count()}, "threads":{"requested":32,"omp":os.environ.get("OMP_NUM_THREADS"),"mkl":os.environ.get("MKL_NUM_THREADS")}, "checkpoint":{"shards":len(shards),"bytes":checkpoint_bytes}, "memory":{"mem_total_bytes":meminfo.get("MemTotal"),"mem_available_bytes":meminfo.get("MemAvailable"),"required_headroom_bytes":required_headroom_bytes,"headroom_factor":2.0}, "disk":{"expected_vector_bytes":expected_vector_bytes,"required_free_bytes":required_free_bytes,"free_bytes":free_bytes}}, sort_keys=True))
PY

vmstat 1 >"$VMSTAT_LOG" 2>&1 &
VMSTAT_PID=$!
cleanup() { kill "$VMSTAT_PID" 2>/dev/null || true; wait "$VMSTAT_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PYTHONUNBUFFERED=1 \
/usr/bin/time -v -o "$TIME_LOG" timeout --signal=TERM --kill-after=60s 2h \
python3 "$REPO/tools/export-qwen35-aq4-source-calibration.py" \
  --model-dir "$MODEL_DIR" --split-root "$SPLIT_ROOT" --cases "$INPUT/cases.json" --output "$OUTPUT" --legacy-oracle "$LEGACY_ORACLE" \
  --expected-split-manifest-sha256 966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887 \
  --expected-policy-sha256 302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03 \
  --expected-calibration-cases-sha256 20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f \
  --expected-cases-sha256 53f256bc8f5ed4036cfb1a9a98c0c9d9197bb980e1ef91d7ff01cf73001369a8 \
  --chunk-elements 65536 --top-k 10 --threads 32 \
  >"$STDOUT_LOG" 2>"$STDERR_LOG"
