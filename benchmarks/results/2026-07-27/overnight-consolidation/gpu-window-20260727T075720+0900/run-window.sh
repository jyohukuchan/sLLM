#!/usr/bin/env bash
# One bounded R9700-only validation window for the immutable AQ4_0 consolidated worker.
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo 'run as root' >&2
  exit 64
fi

repo=/home/homelab1/coding-local/ultimateLLM/uLLM-project
out=$repo/benchmarks/results/2026-07-27/overnight-consolidation/gpu-window-20260727T075720+0900
service=ullm-openai.service
llama_service=llama-qwen35-udq4.service
lock=/run/ullm/r9700.lock
active=/etc/ullm/served-models/active.json
expected_active=3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7
manifest=/opt/ullm/aq4-overnight-consolidation-v0.1/manifests/aq4-consolidated-840a1c7a-5a274733.manifest.json
worker=/opt/ullm/aq4-overnight-consolidation-v0.1/releases/aq4-consolidated-840a1c7a-5a274733/ullm-aq4-worker
decode=/tmp/ullm-overnight-consolidation-target-840a1c7a/release/ullm-aq4-decode-step-profile
prefill=/tmp/ullm-overnight-consolidation-target-840a1c7a/release/ullm-aq4-e2e-prefill-timing
amd_smi=/opt/rocm/bin/amd-smi

mkdir -p "$out"/{service,thermal,decode,prefill}
umask 027
service_stopped=0
lock_held=0

record_state() {
  local label=$1
  systemctl show "$service" -p ActiveState -p SubState -p Result -p NRestarts -p MainPID \
    -p StartLimitBurst -p StartLimitIntervalUSec >"$out/service/${label}-ullm-openai.txt" 2>&1 || true
  systemctl show "$llama_service" -p ActiveState -p SubState -p UnitFileState \
    >"$out/service/${label}-llama-qwen35-udq4.txt" 2>&1 || true
  fuser -v "$lock" >"$out/service/${label}-fuser.txt" 2>&1 || true
}

service_tree_pids() {
  local root_pid=$1 pid child
  local -a queue=("$root_pid")
  while ((${#queue[@]})); do
    pid=${queue[0]}
    queue=("${queue[@]:1}")
    printf '%s\n' "$pid"
    while IFS= read -r child; do
      [[ -n "$child" ]] && queue+=("$child")
    done < <(ps -o pid= --ppid "$pid" 2>/dev/null | tr -d ' ' || true)
  done
}

lock_is_service_only() {
  local main_pid holders pid
  main_pid=$(systemctl show "$service" -p MainPID --value)
  [[ "$main_pid" =~ ^[1-9][0-9]*$ ]] || return 1
  holders=$(fuser "$lock" 2>/dev/null || true)
  [[ -n "$holders" ]] || return 0
  local service_pids
  service_pids=$(service_tree_pids "$main_pid")
  for pid in $holders; do
    grep -qx "$pid" <<<"$service_pids" || return 1
  done
}

metric() {
  "$amd_smi" metric -g 2 -t -c -p --json
}

thermal_gate() {
  local attempt edge
  for attempt in $(seq 1 18); do
    metric >"$out/thermal/gate-${attempt}.json"
    edge=$(jq -r '.gpu_data[0].temperature.edge.value // empty' "$out/thermal/gate-${attempt}.json")
    if [[ "$edge" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN { exit !($edge <= 45.0) }"; then
      printf '%s\tthermal-pass edge_c=%s limit_c=45\n' "$(date --iso-8601=seconds)" "$edge" >>"$out/events.tsv"
      return 0
    fi
    printf '%s\tthermal-wait edge_c=%s attempt=%s limit_c=45\n' "$(date --iso-8601=seconds)" "${edge:-unknown}" "$attempt" >>"$out/events.tsv"
    sleep 5
  done
  echo 'thermal gate did not reach edge <= 45 C' >&2
  return 1
}

run_as_user() {
  local -a environment=(
    'PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin'
    'HOME=/home/homelab1'
    'XDG_CACHE_HOME=/home/homelab1/.cache'
    'HIP_VISIBLE_DEVICES=1'
    'ULLM_HIP_VISIBLE_DEVICES=1'
    'ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1'
  )
  local name
  while IFS= read -r name; do
    [[ "$name" =~ ^[A-Z_][A-Z0-9_]*$ ]] || return 65
    environment+=("${name}=1")
  done < <(jq -r '.worker.required_environment[]' "$manifest")
  runuser -u homelab1 -- env -i "${environment[@]}" "$@"
}

cleanup() {
  local status=$?
  trap - EXIT
  set +e
  if ((lock_held)); then
    flock -u 9
    exec 9>&-
    lock_held=0
    printf '%s\tr9700-lock-released\n' "$(date --iso-8601=seconds)" >>"$out/events.tsv"
  fi
  metric >"$out/thermal/after-window.json" 2>&1 || true
  if ((service_stopped)); then
    printf '%s\tservice-restore-attempt\n' "$(date --iso-8601=seconds)" >>"$out/events.tsv"
    if systemctl start "$service" >"$out/service/restore.stdout" 2>"$out/service/restore.stderr"; then
      printf 'start\n' >"$out/service/restore-commands.txt"
    elif [[ "$(systemctl show "$service" -p Result --value)" == 'start-limit-hit' ]] && \
         systemctl reset-failed "$service" >>"$out/service/restore.stdout" 2>>"$out/service/restore.stderr" && \
         systemctl start "$service" >>"$out/service/restore.stdout" 2>>"$out/service/restore.stderr"; then
      printf 'start\nreset-failed\nstart\n' >"$out/service/restore-commands.txt"
    else
      printf 'restore-failed\n' >"$out/service/restore-commands.txt"
      status=1
    fi
    for _ in $(seq 1 20); do
      systemctl is-active --quiet "$service" && break
      sleep 1
    done
    systemctl is-active --quiet "$service" || status=1
  fi
  record_state after-restore
  chown -R homelab1:homelab1 "$out" || status=1
  exit "$status"
}
trap cleanup EXIT

for path in "$manifest" "$worker" "$decode" "$prefill" "$amd_smi"; do
  [[ -x "$path" || -r "$path" ]] || { echo "missing required input: $path" >&2; exit 65; }
done
[[ $(sha256sum "$active" | awk '{print $1}') == "$expected_active" ]] || { echo 'active manifest drifted before window' >&2; exit 75; }
[[ $(jq -r '.format.format_id' "$manifest") == 'AQ4_0' ]] || exit 65
[[ $(jq -r '.worker.identity.device' "$manifest") == 'gfx1201' ]] || exit 65
[[ $(jq -r '.worker.execution.paged_decode_attention.kernel' "$manifest") == 'aq4_gqa_grouped_split' ]] || exit 65
[[ $(jq -r '.worker.execution.paged_decode_attention.split_tile' "$manifest") == '128' ]] || exit 65
if jq -r '.worker.required_environment[]' "$manifest" | grep -Eq '^(ULLM_KV_CACHE_DTYPE|ULLM_KV_CACHE_TYPE_K|ULLM_KV_CACHE_TYPE_V)$'; then
  echo 'candidate would select a non-F32 KV mode' >&2
  exit 65
fi

record_state before-stop
printf '%s\tpreflight-complete\n' "$(date --iso-8601=seconds)" >>"$out/events.tsv"
if [[ $(systemctl is-active "$llama_service" || true) != inactive || $(systemctl is-enabled "$llama_service" || true) != disabled ]]; then
  echo "$llama_service must remain inactive and disabled" >&2
  exit 75
fi
systemctl is-active --quiet "$service" || { echo 'gateway was not active' >&2; exit 75; }
lock_is_service_only || { echo 'R9700 lock is not held only by gateway service' >&2; exit 75; }

printf '%s\tservice-stop-attempt\n' "$(date --iso-8601=seconds)" >>"$out/events.tsv"
systemctl stop "$service" >"$out/service/stop.stdout" 2>"$out/service/stop.stderr"
service_stopped=1
for _ in $(seq 1 30); do
  if ! systemctl is-active --quiet "$service" && ! fuser "$lock" >/dev/null 2>&1; then break; fi
  sleep 1
done
record_state after-stop
if systemctl is-active --quiet "$service" || fuser "$lock" >/dev/null 2>&1; then
  echo 'service or R9700 lock remained active after stop' >&2
  exit 75
fi

exec 9<>"$lock"
flock -n 9 || { echo 'R9700 lock was acquired by another owner' >&2; exit 75; }
lock_held=1
printf '%s\tr9700-lock-acquired\n' "$(date --iso-8601=seconds)" >>"$out/events.tsv"
"$amd_smi" static -g 2 -a --json >"$out/thermal/r9700-static.json"
metric >"$out/thermal/before-window.json"
thermal_gate

printf '%s\n' 'throughput_source=full-model wall-clock JSON from profile binaries; no profiler range is used as throughput' >"$out/throughput-method.txt"
printf '%s\n' 'KV dtype selection: env -i omits ULLM_KV_* selectors; runtime default is F32.' >"$out/kv-dtype-check.txt"
printf '%s\n' 'HIP_VISIBLE_DEVICES=1' 'ULLM_HIP_VISIBLE_DEVICES=1' 'ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1' >"$out/runtime-environment.txt"
jq -r '.worker.required_environment[] + "=1"' "$manifest" >>"$out/runtime-environment.txt"

run_as_user "$decode" 1339 --warmup 6 --measured 32 >"$out/decode/run-a.jsonl" 2>"$out/decode/run-a.stderr"
metric >"$out/thermal/after-decode-a.json" || true
run_as_user "$decode" 1339 --warmup 6 --measured 32 >"$out/decode/run-b.jsonl" 2>"$out/decode/run-b.stderr"
metric >"$out/thermal/after-decode-b.json" || true
run_as_user "$prefill" >"$out/prefill/p2048-m128.json" 2>"$out/prefill/p2048-m128.stderr"
metric >"$out/thermal/after-prefill.json" || true
printf '%s\tprofiles-complete\n' "$(date --iso-8601=seconds)" >>"$out/events.tsv"
