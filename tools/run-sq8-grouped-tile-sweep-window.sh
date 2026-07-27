#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One exclusive R9700 window for the SQ8_0 grouped source-tile sweep.
set -Euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 RESULT_DIR" >&2; exit 2; }
root=$(git rev-parse --show-toplevel)
out=$(realpath -m "$1")
cf=/home/homelab1/coding-local/ultimateLLM/uLLM-sq8-production-switch-cf/benchmarks/results/2026-07-27/sq8-production-switch
service=ullm-openai.service; llama=llama-qwen35-udq4.service; lock=/run/ullm/r9700.lock
artifact=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
serving="$root/target/release/examples/sq8_ck_serving"
steady="$root/target/release/examples/sq8_0_paged_decode_steady_bench"
moe="$root/target/release/ullm-qwen35-moe-aq4-generate"
moe_package=/home/homelab1/datapool/ullm/product/qwen35-35b-a3b-aq4_0-g8-moe-v0.2/package
moe_tokenizer=/home/homelab1/datapool/ai_models/safetensors/Qwen3.5-35B-A3B-BF16
# Rebuilt after the Qwen3.5 gated-Q layout inference fix.  The window refuses
# to run a stale binary so the recorded shape admission is auditable.
moe_sha256=bad1b58c566b3464e1b840b1107be85cebee918dbfac148e919641f7087ac25b
worker=/home/homelab1/coding-local/ultimateLLM/cf-sq8-build-release-20260727/ullm-sq8-worker
gateway="$root/services/openai-gateway/.venv/bin/ullm-openai-gateway"
suite="$cf/quality/prompt-suite-extended.json"
direct_cases="$cf/quality/direct/capture/cases"
port=18081
[[ ! -e "$out/window-events.tsv" ]] || { echo "refusing to reuse result directory" >&2; exit 2; }
mkdir -p "$out"/{bench,moe,numeric,quality/capture,quality/baseline,service,telemetry,provenance}
event() { printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >>"$out/window-events.tsv"; }
sudo_systemctl() { printf '%s\n' Threadripper | sudo -S -p '' systemctl "$@"; }
metric() { amd-smi metric --gpu 2 --temperature --clock --power --violation --json; }
state() {
  local name=$1
  fuser -v "$lock" >"$out/service/$name-fuser.txt" 2>&1 || true
  pgrep -af 'codex exec' >"$out/service/$name-codex.txt" || true
  systemctl show "$service" -p ActiveState -p SubState -p NRestarts -p Result -p MainPID >"$out/service/$name-service.txt"
  systemctl show "$llama" -p ActiveState -p UnitFileState >"$out/service/$name-llama.txt"
}
lock_held_only_by_service() {
  local main holders pid
  main=$(systemctl show "$service" -p MainPID --value)
  [[ "$main" =~ ^[1-9][0-9]*$ ]] || return 1
  holders=$(fuser "$lock" 2>/dev/null || true)
  for pid in $holders; do
    [[ "$pid" == "$main" ]] || pgrep -P "$main" -a | awk '{print $1}' | grep -qx "$pid" || return 1
  done
}
thermal() {
  local label=$1 edge
  for i in $(seq 1 120); do
    metric | tee -a "$out/telemetry/$label.jsonl" >"$out/telemetry/$label-latest.json"
    edge=$(jq -r '.gpu_data[0].temperature.edge.value // empty' "$out/telemetry/$label-latest.json")
    if [[ "$edge" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN{exit !($edge<=45)}"; then event "thermal-pass $label edge=$edge"; return; fi
    sleep 5
  done
  echo "thermal gate timeout" >&2; return 1
}
stopped=0; held=0; gateway_pid=
release_lock() { if ((held)); then flock -u 9 || true; exec 9>&-; held=0; event lock-released; fi; }
stop_gateway() { if [[ -n ${gateway_pid:-} ]]; then kill "$gateway_pid" 2>/dev/null || true; wait "$gateway_pid" 2>/dev/null || true; gateway_pid=; event isolated-gateway-stopped; fi; }
restore() {
  local status=$?; trap - EXIT; stop_gateway; release_lock
  if ((stopped)); then
    event production-restore-start
    sudo_systemctl start "$service" >"$out/service/restore.stdout" 2>"$out/service/restore.stderr" || status=1
    for d in 1 2 4 8 16 24; do systemctl is-active --quiet "$service" && break; sleep "$d"; done
    systemctl is-active --quiet "$service" || status=1
  fi
  state after-restore; metric >"$out/telemetry/after-restore.json" 2>&1 || true
  if systemctl is-active --quiet "$service"; then
    python3 - "$root/tools/lightweight_promotion.py" >"$out/service/restore-response.json" 2>"$out/service/restore-response.stderr" <<'PY' || status=1
import importlib.util, json, sys
from pathlib import Path
s=importlib.util.spec_from_file_location("p",sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules["p"]=m; s.loader.exec_module(m)
t=m.read_token(Path("/etc/ullm/openai-api-key"))
code,response,err=m._http_json("http://172.20.0.1:8000/v1/chat/completions",token=t,payload={"model":"ullm-qwen3.5-9b-aq4","messages":[{"role":"user","content":"Reply only: restored"}],"max_completion_tokens":8},timeout_seconds=45,gateway_container="open-webui")
if code != 200 or response is None or err: raise SystemExit(f"restore response failed {code} {err}")
print(json.dumps({"http_status":code,"content":m._extract_completion(response)},ensure_ascii=False))
PY
  fi
  event "window-finished status=$status"; exit "$status"
}
trap restore EXIT
printf 'timestamp\tevent\n' >"$out/window-events.tsv"
state before-stop
sha256sum /etc/ullm/served-models/active.json "$serving" "$steady" "$moe" "$worker" "$gateway" "$suite" >"$out/provenance/inputs.sha256"
cp -a -- "$direct_cases/." "$out/quality/baseline/"
cp -- "$cf/quality/direct/manifest.json" "$out/quality/direct-manifest-readonly-copy.json"
cp -- "$cf/quality/gqa-grouped-tile20/manifest.json" "$out/quality/grouped-tile20-manifest-readonly-copy.json"
python3 - "$cf/quality/gqa-grouped-tile20/manifest.json" "$out/quality/manifest-tile128.json" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text()); d["public"]["name"]="uLLM Qwen3 14B SQ8 grouped tile-128 CK candidate"; d["public"]["description"]="SQ8_0 GQA-grouped split-tile-128 candidate for CK."; d["worker"]["execution"]["paged_decode_attention"]["split_tile"]=128
Path(sys.argv[2]).write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
PY
python3 "$root/tools/validate-served-model.py" --manifest "$out/quality/manifest-tile128.json" >"$out/quality/manifest-validation.json"
metric >"$out/telemetry/before-stop.json"
[[ -x "$serving" && -x "$steady" && -x "$moe" && -x "$worker" && -x "$gateway" ]] || { echo "missing binary" >&2; exit 1; }
grep -qx 'ActiveState=inactive' "$out/service/before-stop-llama.txt" && grep -qx 'UnitFileState=disabled' "$out/service/before-stop-llama.txt" || { echo "llama is not inactive/disabled" >&2; exit 1; }
systemctl is-active --quiet "$service" || { echo "production service inactive" >&2; exit 75; }
lock_held_only_by_service || { echo "R9700 lock holder is outside production service; refusing to take it" >&2; exit 75; }
event production-stop; sudo_systemctl stop "$service" >"$out/service/stop.stdout" 2>"$out/service/stop.stderr"; stopped=1
for i in $(seq 1 30); do ! systemctl is-active --quiet "$service" && ! fuser "$lock" >/dev/null 2>&1 && break; sleep 1; done
state after-stop
! systemctl is-active --quiet "$service" && ! fuser "$lock" >/dev/null 2>&1 || { echo "service or lock still busy" >&2; exit 75; }
exec 9<>"$lock"; flock -n 9 || { echo "lock raced busy" >&2; exit 75; }; held=1; event lock-acquired
metric >"$out/telemetry/after-lock.json"

# Phase 2 is deliberately self-contained.  Its exit status is recorded, but
# cannot suppress the independent tile-128 Phase 3 that follows.
run_moe_phase() {
  local pid= status=1 sampled=0
  event moe-phase-start
  thermal moe-phase || { echo "thermal gate failed before MoE" >"$out/moe/error.txt"; return 1; }
  sha256sum "$moe" >"$out/moe/release-binary.sha256"
  grep -q "^$moe_sha256  " "$out/moe/release-binary.sha256" || {
    echo "stale MoE binary: expected $moe_sha256" >"$out/moe/error.txt"; return 1;
  }
  env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_KV_CACHE_DTYPE=f16 \
    ULLM_REQUIRE_HIP_AQ4_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 \
    ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL=1 \
    ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL=1 \
    ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL=1 \
    ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL=1 \
    ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL=1 ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL=1 \
    ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 \
    ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1 \
    ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1 \
    ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 \
    ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL=1 \
    ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL=1 \
    ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1 \
    ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL=1 \
    ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1 ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL=1 \
    ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 \
    ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL=1 \
    ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_TOP1_KERNEL=1 \
    "$moe" --package "$moe_package" --prompt-token-ids 248045,846,198,20206,303,799,2716,6163,11316,25,1092,3520,264,58377,5902,30,248046,198,248045,74455,198,248068,198 \
    --new-tokens 24 --context-length 262144 --kv-block-size 256 --device-index 1 --hold-seconds 20 \
    --output "$out/moe/generation.json" >"$out/moe/generation.stdout" 2>"$out/moe/generation.stderr" &
  pid=$!
  for _ in $(seq 1 180); do
    if [[ -s "$out/moe/generation.json" ]]; then
      amd-smi metric --gpu 2 --mem-usage --json >"$out/moe/vram-during-residency.json" 2>&1 || true
      amd-smi process --gpu 2 --json >"$out/moe/process-during-residency.json" 2>&1 || true
      metric >"$out/moe/telemetry-during-residency.json" 2>&1 || true
      sampled=1
      break
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  wait "$pid"; status=$?
  printf '%s\n' "$status" >"$out/moe/exit-status.txt"
  printf '%s\n' "$sampled" >"$out/moe/residency-sampled.txt"
  if ((status != 0)); then event "moe-phase-failed status=$status"; return "$status"; fi
  python3 - "$out/moe/generation.json" "$moe_tokenizer" "$out/moe/decoded-generation.json" "$out/moe/router-validation.json" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
result = json.loads(Path(sys.argv[1]).read_text())
tok = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True, trust_remote_code=True)
generated = result["generation"]["generated_token_ids"]
Path(sys.argv[3]).write_text(json.dumps({"prompt_text": "Reply in one short English sentence: what makes a rollback safe?", "generated_token_ids": generated, "generated_text": tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)}, ensure_ascii=False, indent=2) + "\n")
routes = result["router_verification"]
summary = {"layer_count": len(routes), "layer_indices": [row["layer_index"] for row in routes], "tie_free_mismatches": result["router_verification_tie_free_mismatches"], "strict_order_match_values": [row["strict_order_match"] for row in routes], "all_tie_free_routes_match": not result["router_verification_tie_free_mismatches"], "prefill_tokens_per_second": len(result["generation"]["prompt_token_ids"]) / (result["generation"]["prompt_wall_ms"] / 1000.0), "decode_tokens_per_second": result["generation"]["decode_tokens_per_second"]}
if summary["layer_count"] != 40 or summary["layer_indices"] != list(range(40)) or not summary["all_tie_free_routes_match"]:
    raise SystemExit("40-layer raw-BF16 router verification is incomplete or mismatched")
Path(sys.argv[4]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
PY
  status=$?
  event "moe-phase-finished status=$status"
  return "$status"
}
# CO's tile-quality and hardware-measurement window does not need to repeat
# the independently completed MoE admission.  Keep the default historical
# behavior for callers that do need it, but make the skip explicit in the
# evidence rather than silently omitting the phase.
if [[ ${ULLM_SKIP_MOE_PHASE:-0} == 1 ]]; then
  printf '%s\n' 'skipped-by-ULLM_SKIP_MOE_PHASE' >"$out/moe/phase-status.txt"
  event moe-phase-skipped
elif run_moe_phase; then :; else printf '%s\n' "$?" >"$out/moe/phase-status.txt"; fi
run_route() {
  local tile=$1; shift
  local -a e=(-u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE -u ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT -u ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1 ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1)
  [[ "$tile" == direct ]] || e+=(ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE="$tile" ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1 ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1)
  env "${e[@]}" "$@"
}
if [[ ${ULLM_SKIP_EXISTING_SPEED_BENCHMARKS:-0} != 1 ]]; then
  for route in direct 20 128; do
    label=tile$route; [[ $route == direct ]] && label=direct
    thermal "bench-$label"; event "bench-start-$label"
    run_route "$route" "$steady" --artifact "$artifact" --package "$package" --output "$out/bench/$label.json" --prompt-tokens 1024 --warmup-steps 4 --measured-steps 16 --repeats 5 >"$out/bench/$label.stdout" 2>"$out/bench/$label.stderr"
    event "bench-finished-$label"
  done
else
  event bench-skipped-existing-speed-evidence
fi
if [[ ${ULLM_SKIP_NUMERIC_CAPTURE:-0} != 1 ]]; then
  for route in direct 20 128; do
    label=tile$route; [[ $route == direct ]] && label=direct
    # The serving runner owns creation of the capture directory and rejects a
    # pre-existing target.  Its target is the route directory itself (not an
    # `oracle` child), so leave numeric/<route> absent until this invocation.
    run_route "$route" "$serving" --artifact "$artifact" --package "$package" --prompt-lengths 512 --max-new-tokens 4 --prefill-mode m128-chunk128 --decode-oracle-capture-dir "$out/numeric/$label" --result-json "$out/numeric/$label-result.json" >"$out/numeric/$label.stdout" 2>"$out/numeric/$label.stderr"
  done
  python3 - "$out/numeric" "$out/numeric/summary.json" <<'PY'
import json,math,struct,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
def items(p):
 with p.open("rb") as f:
  while b:=f.read(1048576): yield from struct.iter_unpack("<f",b)
def captures(route): return json.loads((root/f"{route}-result.json").read_text())["requests"][0]["decode_oracle_captures"]
base=captures("direct"); result={"schema_version":"ullm.sq8_grouped_tile_sweep_numeric.v1","reference_route":"direct","routes":{}}
for route in ("tile20","tile128"):
 maximum=0.; values=bad=0
 for a,b in zip(base,captures(route),strict=True):
  for key in ("final_hidden_file","logits_file"):
   # Capture file names are relative to numeric/, and already include the
   # route directory (for example direct/foo.f32le).  Do not prefix route a
   # second time: that made the completed capture comparison fail before the
   # quality phase could record a useful diagnostic.
   for (x,),(y,) in zip(items(root/a[key]),items(root/b[key]),strict=True):
    maximum=max(maximum,abs(x-y)); values+=1; bad+=not(math.isfinite(x) and math.isfinite(y))
 result["routes"][route]={"split_vs_direct_max_abs":maximum,"compared_f32_values":values,"nonfinite_values":bad}
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
PY
else
  # CO reuses the fresh CN numeric capture requested by the user; this window
  # is reserved for the still-missing text-quality and hardware evidence.
  cp -- "$root/benchmarks/results/2026-07-27/cn-moe-o-proj-tile128-window/numeric/summary.json" "$out/numeric/summary-reused-cn.json"
  event numeric-capture-skipped-reused-cn
fi
quality_status=0
if thermal quality-gateway; then
release_lock; event isolated-gateway-start
env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_SERVED_MODEL_MANIFEST="$out/quality/manifest-tile128.json" ULLM_GPU_LOCK_FILE="$lock" ULLM_BIND_HOST=127.0.0.1 ULLM_BIND_PORT="$port" "$gateway" >"$out/quality/gateway.stdout" 2>"$out/quality/gateway.stderr" & gateway_pid=$!
# The gateway receives the candidate manifest in its process environment, but
# the separate capture harness also reads that manifest to record request
# provenance.  Pass it explicitly here; the old command only set it for the
# background gateway and therefore failed before readiness with KeyError.
if ULLM_SERVED_MODEL_MANIFEST="$out/quality/manifest-tile128.json" \
python3 - "$root/tools/lightweight_promotion.py" "$suite" "$out/quality/capture" "$port" >"$out/quality/capture-harness.stdout" 2>"$out/quality/capture-harness.stderr" <<'PY'
import importlib.util,json,os,sys
from pathlib import Path
s=importlib.util.spec_from_file_location("p",sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules["p"]=m;s.loader.exec_module(m)
suite=m.load_suite(Path(sys.argv[2]));out=Path(sys.argv[3]);port=sys.argv[4];manifest=m.strict_object(Path(os.environ["ULLM_SERVED_MODEL_MANIFEST"]).read_bytes(),"manifest");token=m.read_token(Path("/etc/ullm/openai-api-key"))
ready=m.wait_for_live_gateway(base_url=f"http://127.0.0.1:{port}",token=token,model_id="ullm-qwen3-14b-sq8",timeout_seconds=120,gateway_container=None);m.write_json_new(out/"readiness.json",ready,"readiness")
rows=m.run_suite(suite=suite,model_id="ullm-qwen3-14b-sq8",manifest_document=manifest,base_url=f"http://127.0.0.1:{port}",token=token,request_timeout_seconds=120,output_dir=out/"cases",gateway_container=None)
m.write_json_new(out/"capture.json",{"schema_version":"ullm.lightweight_served_suite_capture.v1","case_count":len(rows),"passed":not any(r["analysis"]["blocking"] for r in rows),"blocking_findings":[f"{r['case_id']}:{x}" for r in rows for x in r["analysis"]["blocking"]]},"capture")
PY
then :; else quality_status=$?; fi
stop_gateway
exec 9<>"$lock"; flock -n 9; held=1; event lock-reacquired-after-quality
if ((quality_status == 0)); then
if python3 - "$root/tools/lightweight_promotion.py" "$suite" "$out/quality/baseline" "$out/quality/capture/cases" "$out/quality/comparison.json" "$out/quality/comparison.md" >"$out/quality/compare-harness.stdout" 2>"$out/quality/compare-harness.stderr" <<'PY'
import importlib.util,json,sys
from pathlib import Path
s=importlib.util.spec_from_file_location("p",sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules["p"]=m;s.loader.exec_module(m)
suite=m.load_suite(Path(sys.argv[2]))
def rows(d): return [json.loads((Path(d)/f"{case.case_id}.json").read_text()) for case in suite]
base,candidate=rows(sys.argv[3]),rows(sys.argv[4]);c=m.compare_suites(suite,base,candidate);m.write_json_new(Path(sys.argv[5]),c,"comparison");m.write_comparison_markdown(Path(sys.argv[6]),suite,base,candidate,c)
PY
then :; else quality_status=$?; fi
fi
else
  quality_status=$?
fi
printf '%s\n' "$quality_status" >"$out/quality/phase-status.txt"
event "quality-phase-finished status=$quality_status"

# Run the standalone R9700 hardware measurements while this same exclusive
# window still owns the device.  A separate destination avoids overwriting
# historical tile evidence; the wrapper rebuilds and ISA-audits its binary
# immediately before timing.  If this phase fails, the completed tile evidence
# remains intact and the EXIT trap still releases the lock before restoring AQ4.
if [[ -n ${ULLM_HW_MICROBENCH_RESULTS_DIR:-} ]]; then
  hw=$(realpath -m "$ULLM_HW_MICROBENCH_RESULTS_DIR")
  [[ ! -e "$hw/hw-microbench-gfx1201" ]] || { echo "hardware benchmark binary already exists: $hw" >&2; exit 2; }
  mkdir -p "$hw"
  thermal hw-microbench
  event hw-microbench-start
  hw_started=$(date --iso-8601=seconds)
  HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 \
    HW_MB_MEMORY_PEAK_GBPS=640 HW_MB_BF16_PEAK_TFLOPS=191 HW_MB_FP8_PEAK_TFLOPS=383 \
    "$root/tools/run-hw-microbench-rdna4-cdna3.sh" --arch gfx1201 --results-dir "$hw"
  hw_finished=$(date --iso-8601=seconds)
  printf 'started_at=%s\nfinished_at=%s\n' "$hw_started" "$hw_finished" >"$hw/window-wall-clock.txt"
  event hw-microbench-finished
fi
metric >"$out/telemetry/before-restore.json"; event measurements-complete
