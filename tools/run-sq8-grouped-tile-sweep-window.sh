#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One exclusive R9700 window for the SQ8_0 grouped source-tile sweep.
set -Eeuo pipefail
[[ $# == 1 ]] || { echo "usage: $0 RESULT_DIR" >&2; exit 2; }
root=$(git rev-parse --show-toplevel)
out=$(realpath -m "$1")
cf=/home/homelab1/coding-local/ultimateLLM/uLLM-sq8-production-switch-cf/benchmarks/results/2026-07-27/sq8-production-switch
service=ullm-openai.service; llama=llama-qwen35-udq4.service; lock=/run/ullm/r9700.lock
artifact=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
package=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
serving="$root/target/release/examples/sq8_ck_serving"
steady="$root/target/release/examples/sq8_0_paged_decode_steady_bench"
worker=/home/homelab1/coding-local/ultimateLLM/cf-sq8-build-release-20260727/ullm-sq8-worker
gateway="$root/services/openai-gateway/.venv/bin/ullm-openai-gateway"
suite="$cf/quality/prompt-suite-extended.json"
direct_cases="$cf/quality/direct/capture/cases"
port=18081
[[ ! -e "$out/window-events.tsv" ]] || { echo "refusing to reuse result directory" >&2; exit 2; }
mkdir -p "$out"/{bench,numeric,quality/capture,quality/baseline,service,telemetry,provenance}
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
sha256sum /etc/ullm/served-models/active.json "$serving" "$steady" "$worker" "$gateway" "$suite" >"$out/provenance/inputs.sha256"
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
[[ -x "$serving" && -x "$steady" && -x "$worker" && -x "$gateway" ]] || { echo "missing binary" >&2; exit 1; }
grep -qx 'ActiveState=inactive' "$out/service/before-stop-llama.txt" && grep -qx 'UnitFileState=disabled' "$out/service/before-stop-llama.txt" || { echo "llama is not inactive/disabled" >&2; exit 1; }
systemctl is-active --quiet "$service" || { echo "production service inactive" >&2; exit 75; }
lock_held_only_by_service || { echo "R9700 lock holder is outside production service; refusing to take it" >&2; exit 75; }
event production-stop; sudo_systemctl stop "$service" >"$out/service/stop.stdout" 2>"$out/service/stop.stderr"; stopped=1
for i in $(seq 1 30); do ! systemctl is-active --quiet "$service" && ! fuser "$lock" >/dev/null 2>&1 && break; sleep 1; done
state after-stop
! systemctl is-active --quiet "$service" && ! fuser "$lock" >/dev/null 2>&1 || { echo "service or lock still busy" >&2; exit 75; }
exec 9<>"$lock"; flock -n 9 || { echo "lock raced busy" >&2; exit 75; }; held=1; event lock-acquired
metric >"$out/telemetry/after-lock.json"
run_route() {
  local tile=$1; shift
  local -a e=(-u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE -u ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE -u ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT -u ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 ULLM_REQUIRE_HIP_ROPE_KERNEL=1 ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_ADD_KERNEL=1 ULLM_REQUIRE_HIP_SILU_MUL_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1 ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1 ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1)
  [[ "$tile" == direct ]] || e+=(ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE="$tile" ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1 ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1)
  env "${e[@]}" "$@"
}
for route in direct 20 128; do
  label=tile$route; [[ $route == direct ]] && label=direct
  thermal "bench-$label"; event "bench-start-$label"
  run_route "$route" "$steady" --artifact "$artifact" --package "$package" --output "$out/bench/$label.json" --prompt-tokens 1024 --warmup-steps 4 --measured-steps 16 --repeats 5 >"$out/bench/$label.stdout" 2>"$out/bench/$label.stderr"
  event "bench-finished-$label"
done
for route in direct 20 128; do
  label=tile$route; [[ $route == direct ]] && label=direct; mkdir -p "$out/numeric/$label"
  run_route "$route" "$serving" --artifact "$artifact" --package "$package" --prompt-lengths 512 --max-new-tokens 4 --prefill-mode m128-chunk128 --decode-oracle-capture-dir "$out/numeric/$label/oracle" --result-json "$out/numeric/$label/result.json" >"$out/numeric/$label.stdout" 2>"$out/numeric/$label.stderr"
done
python3 - "$out/numeric" "$out/numeric/summary.json" <<'PY'
import json,math,struct,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
def items(p):
 with p.open("rb") as f:
  while b:=f.read(1048576): yield from struct.iter_unpack("<f",b)
def captures(route): return json.loads((root/route/"result.json").read_text())["requests"][0]["decode_oracle_captures"]
base=captures("direct"); result={"schema_version":"ullm.sq8_grouped_tile_sweep_numeric.v1","reference_route":"direct","routes":{}}
for route in ("tile20","tile128"):
 maximum=0.; values=bad=0
 for a,b in zip(base,captures(route),strict=True):
  for key in ("final_hidden_file","logits_file"):
   for (x,),(y,) in zip(items(root/"direct"/a[key]),items(root/route/b[key]),strict=True):
    maximum=max(maximum,abs(x-y)); values+=1; bad+=not(math.isfinite(x) and math.isfinite(y))
 result["routes"][route]={"split_vs_direct_max_abs":maximum,"compared_f32_values":values,"nonfinite_values":bad}
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
PY
release_lock; event isolated-gateway-start
env HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ULLM_SERVED_MODEL_MANIFEST="$out/quality/manifest-tile128.json" ULLM_GPU_LOCK_FILE="$lock" ULLM_BIND_HOST=127.0.0.1 ULLM_BIND_PORT="$port" "$gateway" >"$out/quality/gateway.stdout" 2>"$out/quality/gateway.stderr" & gateway_pid=$!
python3 - "$root/tools/lightweight_promotion.py" "$suite" "$out/quality/capture" "$port" <<'PY'
import importlib.util,json,os,sys
from pathlib import Path
s=importlib.util.spec_from_file_location("p",sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules["p"]=m;s.loader.exec_module(m)
suite=m.load_suite(Path(sys.argv[2]));out=Path(sys.argv[3]);port=sys.argv[4];manifest=m.strict_object(Path(os.environ["ULLM_SERVED_MODEL_MANIFEST"]).read_bytes(),"manifest");token=m.read_token(Path("/etc/ullm/openai-api-key"))
ready=m.wait_for_live_gateway(base_url=f"http://127.0.0.1:{port}",token=token,model_id="ullm-qwen3-14b-sq8",timeout_seconds=120,gateway_container=None);m.write_json_new(out/"readiness.json",ready,"readiness")
rows=m.run_suite(suite=suite,model_id="ullm-qwen3-14b-sq8",manifest_document=manifest,base_url=f"http://127.0.0.1:{port}",token=token,request_timeout_seconds=120,output_dir=out/"cases",gateway_container=None)
m.write_json_new(out/"capture.json",{"schema_version":"ullm.lightweight_served_suite_capture.v1","case_count":len(rows),"passed":not any(r["analysis"]["blocking"] for r in rows),"blocking_findings":[f"{r['case_id']}:{x}" for r in rows for x in r["analysis"]["blocking"]]},"capture")
PY
stop_gateway
exec 9<>"$lock"; flock -n 9; held=1; event lock-reacquired-after-quality
python3 - "$root/tools/lightweight_promotion.py" "$suite" "$out/quality/baseline" "$out/quality/capture/cases" "$out/quality/comparison.json" "$out/quality/comparison.md" <<'PY'
import importlib.util,json,sys
from pathlib import Path
s=importlib.util.spec_from_file_location("p",sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules["p"]=m;s.loader.exec_module(m)
suite=m.load_suite(Path(sys.argv[2]))
def rows(d): return [json.loads((Path(d)/f"{case.case_id}.json").read_text()) for case in suite]
base,candidate=rows(sys.argv[3]),rows(sys.argv[4]);c=m.compare_suites(suite,base,candidate);m.write_json_new(Path(sys.argv[5]),c,"comparison");m.write_comparison_markdown(Path(sys.argv[6]),suite,base,candidate,c)
PY
metric >"$out/telemetry/before-restore.json"; event measurements-complete
