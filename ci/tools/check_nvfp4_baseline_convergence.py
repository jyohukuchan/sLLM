#!/usr/bin/env python3
"""Does forcing the NVFP4 W4A4 baseline make the two GPUs agree on target hidden?

gfx1201 selects Nvfp4W4A4PrefillGfx1201WmmaKahan (WMMA + Kahan compensation) for
the 27B MLP prefill; gfx1030 has no matrix cores and selects a compensated /
dp4a kernel instead. SLLM_NVFP4_W4A4_FORCE_BASELINE=1 removes that specialised
layer on both targets. If target_hidden then matches, the NVFP4 prefill kernel
specialisation -- of which WMMA is the gfx1201 half -- explains the divergence.
"""

from __future__ import annotations

import glob, csv, hashlib, http.client, json, os, pathlib, subprocess, sys, time

REPO = pathlib.Path("/home/homelab1/coding-local/sLLM")
UNIT = "sllm-qwen38-r9700.service"
OUT = REPO / ".local-artifacts/mtp-bench/wmma-check/baseline"
PREFIX = REPO / ".local-artifacts/mtp-bench/wmma-check/prefix-one.json"
MODEL = "/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4"
TARGETS = {
    "gfx1030": {"uuid": "GPU-76a08c022586fed6",
                "binary": REPO / ".local-artifacts/mtp-bench/claimb/bin/bench-gfx1030",
                "lease": False},
    "gfx1201": {"uuid": "GPU-a8e9ddefa2d60f55",
                "binary": REPO / ".local-artifacts/mtp-bench/claimb/bin/bench-gfx1201",
                "lease": True},
}


def health(e):
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
    try:
        c.request("GET", e); r = c.getresponse(); r.read(); return r.status
    except OSError:
        return 0
    finally:
        c.close()


def service_hashes():
    fs = [pathlib.Path("/home/homelab1/.config/systemd/user") / UNIT,
          REPO / ".local-artifacts/qwen38-r9700-server/run.sh",
          REPO / ".local-artifacts/qwen38-r9700-server/sllm-server"]
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in fs}


def active():
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0


def find(node, key):
    if isinstance(node, dict):
        if key in node: return node[key]
        for v in node.values():
            r = find(v, key)
            if r is not None: return r
    elif isinstance(node, list):
        for i in node:
            r = find(i, key)
            if r is not None: return r
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"schema_version": "nvfp4-baseline-convergence-v1", "state": "running",
           "flag": "SLLM_NVFP4_W4A4_FORCE_BASELINE=1 on both targets",
           "condition": "code-rust-bugfix-zh", "runs": {}, "service_stopped": False}

    def save():
        (OUT / "execution.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    save()
    try:
        for target, spec in TARGETS.items():
            if spec["lease"]:
                if subprocess.check_output(["ss","-Htn","state","established","( sport = :8000 )"],text=True).strip():
                    raise RuntimeError("active connections; not leasing")
                rep["service_hashes_before"] = service_hashes()
                rep["service_was_active"] = active()
                if rep["service_was_active"]:
                    if health("/readyz") != 200: raise RuntimeError("server not ready")
                    rep["service_stopped"] = True; save()
                    subprocess.run(["systemctl","--user","stop",UNIT],check=True)
                    for _ in range(60):
                        if not active(): break
                        time.sleep(1)
            d = OUT / target
            (d / "stage0").mkdir(parents=True, exist_ok=True)
            trace = d / "trace"; trace.mkdir(parents=True, exist_ok=True)
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("SLLM_")
                   and k not in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")}
            env.update({"ROCR_VISIBLE_DEVICES": spec["uuid"], "LD_LIBRARY_PATH": "/opt/rocm/lib",
                "SLLM_PHASE78_MODEL_PATH": MODEL, "SLLM_PHASE78_TARGET": target,
                "SLLM_PHASE78_DEVICE": "0", "SLLM_PHASE78_WARMUPS": "0", "SLLM_PHASE78_MEASURED": "1",
                "SLLM_PHASE78_CHUNK_CAPACITY": "2048", "SLLM_PHASE83_MODE": "coding8192",
                "SLLM_PHASE83_KV": "mxfp8", "SLLM_PHASE83_SAMPLING": "gpu-fixed",
                "SLLM_PHASE83_REPLAY": "0", "SLLM_PHASE83_MTP": "on", "SLLM_PHASE83_MTP_WIDTH": "2",
                "SLLM_PHASE83_STATE_CAPACITY": "10240", "SLLM_PHASE85_A16_STAGE0": "1",
                "SLLM_PHASE85_A16_PREFIX_FILE": str(PREFIX),
                "SLLM_PHASE85_A16_OUTPUT_DIR": str(d / "stage0"),
                "SLLM_PHASE85_A16_SERIES": "bf16",
                "SLLM_NVFP4_W4A4_FORCE_BASELINE": "1"})
            cmd = ["/usr/bin/rocprofv3","--kernel-trace","--output-format","csv",
                   "--output-directory", str(trace), "--", str(spec["binary"])]
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, env=env, cwd=str(REPO))
            (d / "run.stdout.json").write_bytes(p.stdout); (d / "run.stderr.log").write_bytes(p.stderr)
            entry = {"exit_code": p.returncode, "wall_seconds": round(time.time()-t0,1)}
            if p.returncode == 0:
                doc = json.loads(p.stdout.decode())
                ent = find(doc, "entries")[0]
                entry["target_hidden_sha256"] = ent["target_hidden_sha256"]
                entry["logits_sha256"] = ent["logits_sha256"]
                names = {}
                for f in glob.glob(str(trace/"**"/"*kernel_trace*.csv"), recursive=True):
                    for row in csv.DictReader(open(f, newline="")):
                        n = row.get("Kernel_Name") or row.get("Name") or ""
                        names[n] = names.get(n,0)+1
                w = {k:v for k,v in names.items() if "wmma" in k.lower()}
                entry["wmma_dispatch_count"] = sum(w.values()); entry["wmma_kernels"] = w
                entry["total_dispatches"] = sum(names.values())
            else:
                entry["error"] = p.stderr.decode(errors="replace")[-400:]
            rep["runs"][target] = entry
            save()
            print("%s exit=%d wmma=%s hidden=%s" % (target, p.returncode,
                  entry.get("wmma_dispatch_count"), str(entry.get("target_hidden_sha256"))[:24]), flush=True)
            if spec["lease"]:
                subprocess.run(["systemctl","--user","start",UNIT],check=False)
                for _ in range(120):
                    if active() and health("/healthz")==200 and health("/readyz")==200: break
                    time.sleep(1)
                rep["service_hashes_after"] = service_hashes()
                rep["service_restored"] = (active() and health("/healthz")==200 and health("/readyz")==200
                                           and rep["service_hashes_before"]==rep["service_hashes_after"])
                rep["service_stopped"] = False
        a = rep["runs"].get("gfx1030",{}).get("target_hidden_sha256")
        b = rep["runs"].get("gfx1201",{}).get("target_hidden_sha256")
        rep["target_hidden_match"] = (a is not None and a == b)
        rep["logits_match"] = (rep["runs"].get("gfx1030",{}).get("logits_sha256")
                               == rep["runs"].get("gfx1201",{}).get("logits_sha256"))
        rep["verdict"] = ("CONVERGED: forcing the NVFP4 baseline makes both targets agree"
                          if rep["target_hidden_match"] else
                          "STILL DIVERGENT: the NVFP4 prefill specialisation is not sufficient")
        rep["state"] = "complete"
        print("\n" + rep["verdict"], flush=True)
    except Exception as e:
        rep["state"] = "failed"; rep["error"] = str(e)[:600]; print("ERROR:", e, flush=True)
    finally:
        if rep.get("service_stopped"):
            subprocess.run(["systemctl","--user","start",UNIT],check=False)
            for _ in range(120):
                if active() and health("/healthz")==200 and health("/readyz")==200: break
                time.sleep(1)
            rep["service_restored"] = active() and health("/healthz")==200 and health("/readyz")==200
        rep["ended"] = time.time(); save()
        print("service_active=%s state=%s" % (active(), rep["state"]), flush=True)
    return 0 if rep["state"]=="complete" else 1


if __name__ == "__main__":
    sys.exit(main())
