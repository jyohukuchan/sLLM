#!/usr/bin/env bash
# Capture a paired unprofiled/rocprof AQ4_0 decode accounting run.
#
# This script intentionally refuses to run while the served worker is active and
# holds /run/ullm/r9700.lock for the entire GPU window.  It is a reproducibility
# artifact, not a production deployment command.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
output_dir="$1"
profile_binary="/home/homelab1/coding-local/ultimateLLM/uLLM-aq4-p3-deployment-build-target-c4c9a9b3/release/ullm-aq4-decode-step-profile"
worker_binary="/opt/ullm/aq4-p3-deployment-v0.1/releases/aq4-p3-c4c9a9b3/ullm-aq4-worker"
active_manifest="/etc/ullm/served-models/active.json"
package_dir="/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package"
module_probe_source="$repo_root/benchmarks/results/2026-07-27/aq4-decode-walltime-accounting/hip-module-launch-overhead.cpp"
module_probe_kernel="$repo_root/benchmarks/results/2026-07-27/aq4-decode-walltime-accounting/hip-module-launch-overhead-kernel.hip"

if [[ -e "$output_dir" ]]; then
    echo "refusing to overwrite existing output directory: $output_dir" >&2
    exit 2
fi
if [[ ! -x "$profile_binary" || ! -x "$worker_binary" || ! -r "$active_manifest" ]]; then
    echo "required active AQ4_0 P3 input is missing or unreadable" >&2
    exit 1
fi
if [[ ! -r "$module_probe_source" || ! -r "$module_probe_kernel" ]]; then
    echo "module-launch probe sources are missing" >&2
    exit 1
fi

mkdir -p "$output_dir"

# This preflight is deliberately outside the lock.  If another owner wins the
# race, flock below fails rather than taking or waiting for its lock.
fuser -v /run/ullm/r9700.lock >"$output_dir/gpu-lock-before.txt" 2>&1 || true
# Run the required command but retain only PIDs: another task's full prompt
# can be arbitrarily large and is not useful provenance for this capture.
pgrep -af 'ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model' \
    | awk -v self="$$" '$1 != self {print $1}' \
    >"$output_dir/gpu-processes-before-pids.txt" || true
systemctl show ullm-openai.service -p ActiveState -p NRestarts \
    >"$output_dir/ullm-openai-service-before.txt"
if ! rg -qx 'ActiveState=inactive' "$output_dir/ullm-openai-service-before.txt"; then
    echo "ullm-openai.service is not inactive; refusing GPU capture" >&2
    exit 1
fi
if [[ -s "$output_dir/gpu-lock-before.txt" ]]; then
    echo "R9700 lock was held at preflight; refusing GPU capture" >&2
    exit 75
fi

exec 9>/run/ullm/r9700.lock
if ! flock -n 9; then
    echo "R9700 lock became held before capture; no GPU work was started" >&2
    exit 75
fi

metric_pid=""
cleanup() {
    if [[ -n "$metric_pid" ]] && kill -0 "$metric_pid" 2>/dev/null; then
        kill "$metric_pid" 2>/dev/null || true
        wait "$metric_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

fuser -v /run/ullm/r9700.lock >"$output_dir/gpu-lock-held.txt" 2>&1 || true
systemctl show ullm-openai.service -p ActiveState -p NRestarts \
    >"$output_dir/ullm-openai-service-held.txt"
if ! rg -qx 'ActiveState=inactive' "$output_dir/ullm-openai-service-held.txt"; then
    echo "ullm-openai.service became active before GPU work; refusing capture" >&2
    exit 75
fi
# Compile after the non-blocking lock succeeds.  It is CPU-only, but keeping
# it in the same exclusive window prevents another R9700 user from winning the
# several-second compile-to-capture race.  No HIP runtime is initialized until
# after this point.
probe_dir="$output_dir/module-launch-probe"
mkdir -p "$probe_dir"
hipcc --genco --offload-arch=gfx1201 "$module_probe_kernel" -o "$probe_dir/noop-gfx1201.co"
hipcc -O3 -std=c++20 "$module_probe_source" -o "$probe_dir/hip-module-launch-overhead"
sha256sum "$module_probe_source" "$module_probe_kernel" "$probe_dir/noop-gfx1201.co" \
    "$probe_dir/hip-module-launch-overhead" >"$output_dir/module-launch-probe-sha256.txt"

sha256sum "$active_manifest" "$worker_binary" "$profile_binary" >"$output_dir/input-sha256-before.txt"
jq '{format, worker: {binary: .worker.binary, binary_sha256: .worker.binary_sha256, required_environment: .worker.required_environment}}' \
    "$active_manifest" >"$output_dir/active-manifest-identity-before.json"
git -C /home/homelab1/coding-local/ultimateLLM/uLLM-aq4-p3-deployment-source-c4c9a9b3 rev-parse HEAD \
    >"$output_dir/profile-source-commit.txt"
sha256sum "$package_dir/manifest.json" >"$output_dir/package-manifest-sha256.txt"
amd-smi static -g 2 -a --json >"$output_dir/r9700-static.json"
amd-smi metric -g 2 -t -c -p --json >"$output_dir/r9700-metrics-before.json"
amd-smi metric -g 2 -t -c -p --json -w 1 -W 1200 --file "$output_dir/r9700-metrics-watch.json" \
    >"$output_dir/r9700-metrics-watch.log" 2>&1 &
metric_pid=$!

export HIP_VISIBLE_DEVICES=1
export ULLM_HIP_VISIBLE_DEVICES=1
while IFS= read -r required_name; do
    export "$required_name=1"
done < <(jq -r '.worker.required_environment[]' "$active_manifest")
env | rg '^(HIP_VISIBLE_DEVICES|ULLM_HIP_VISIBLE_DEVICES|ULLM_REQUIRE_HIP_)=' | sort \
    >"$output_dir/capture-environment.txt"

"$probe_dir/hip-module-launch-overhead" "$probe_dir/noop-gfx1201.co" 1024 31 \
    >"$output_dir/module-launch-overhead-unprofiled.json"
"$profile_binary" 1339 --warmup 6 --measured 32 \
    >"$output_dir/current-unprofiled.stdout" 2>"$output_dir/current-unprofiled.stderr"

rocprofv3 -d "$output_dir/rocprof" -f csv \
    --hip-trace --kernel-trace --marker-trace --memory-copy-trace --log-level error -- \
    "$profile_binary" 1339 --warmup 6 --measured 32 \
    >"$output_dir/current-rocprof.stdout" 2>"$output_dir/current-rocprof.stderr"

rocprofv3 -d "$output_dir/module-launch-probe-rocprof" -f csv \
    --hip-trace --kernel-trace --log-level error -- \
    "$probe_dir/hip-module-launch-overhead" "$probe_dir/noop-gfx1201.co" 1024 31 \
    >"$output_dir/module-launch-overhead-rocprof.json" \
    2>"$output_dir/module-launch-overhead-rocprof.stderr"

amd-smi metric -g 2 -t -c -p --json >"$output_dir/r9700-metrics-after.json"
sha256sum "$active_manifest" "$worker_binary" "$profile_binary" >"$output_dir/input-sha256-after.txt"
systemctl show ullm-openai.service -p ActiveState -p NRestarts \
    >"$output_dir/ullm-openai-service-after.txt"
if ! cmp -s "$output_dir/input-sha256-before.txt" "$output_dir/input-sha256-after.txt"; then
    echo "active manifest, worker, or profile binary changed during capture; refusing mixed provenance" >&2
    exit 1
fi
if ! rg -qx 'ActiveState=inactive' "$output_dir/ullm-openai-service-after.txt"; then
    echo "ullm-openai.service changed state during capture; refusing concurrent timing evidence" >&2
    exit 1
fi

kernel_trace="$(find "$output_dir/rocprof" -type f -name '*kernel_trace.csv' -print -quit)"
hip_trace="$(find "$output_dir/rocprof" -type f -name '*hip_api_trace.csv' -print -quit)"
marker_trace="$(find "$output_dir/rocprof" -type f -name '*marker_api_trace.csv' -print -quit)"
if [[ -z "$kernel_trace" || -z "$hip_trace" || -z "$marker_trace" ]]; then
    echo "rocprof did not produce all required trace CSVs" >&2
    exit 1
fi

python3 "$repo_root/tools/analyze-aq4-decode-walltime-accounting.py" \
    --kernel-trace "$kernel_trace" \
    --hip-api-trace "$hip_trace" \
    --marker-trace "$marker_trace" \
    --profile-stdout "$output_dir/current-rocprof.stdout" \
    --expected-cache-start 1339 --expected-steps 32 \
    --output "$output_dir/current-profiled-walltime-accounting.json"

unprofiled_wall_ms="$(python3 -c '
import json, sys
samples = []
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    if row.get("event") == "measured_decode_step":
        samples.append(row["elapsed_seconds"])
if len(samples) != 32:
    raise SystemExit(f"expected 32 measured samples, found {len(samples)}")
print(sum(samples) / len(samples) * 1000.0)
' "$output_dir/current-unprofiled.stdout")"
python3 "$repo_root/tools/analyze-aq4-decode-walltime-accounting.py" \
    --kernel-trace "$kernel_trace" \
    --hip-api-trace "$hip_trace" \
    --marker-trace "$marker_trace" \
    --wall-ms "$unprofiled_wall_ms" \
    --expected-cache-start 1339 --expected-steps 32 \
    --output "$output_dir/current-unprofiled-walltime-accounting.json"
python3 "$repo_root/tools/derive-aq4-projection-roofline.py" \
    --package-manifest "$package_dir/manifest.json" \
    --accounting "$output_dir/current-unprofiled-walltime-accounting.json" \
    --output "$output_dir/current-projection-roofline.json"

date -Is >"$output_dir/capture-complete-at.txt"
