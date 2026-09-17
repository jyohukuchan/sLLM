#!/usr/bin/env python3
"""Generate and freeze the mtp-bench-v1 reference committed token sequences.

Two reference series are frozen per condition: the BF16 companion and the
MXFP8 W8A8 companion. Keeping both lets a later comparison check that the sign
of an acceptance difference does not depend on which reference produced the
forced sequence.

A frozen sequence is a forced input, not a reproduction target: what matters is
that it is fixed, hashed, and shared by every compared series. The build and
device identity that produced it are recorded as provenance only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = REPO / "ci" / "fixtures" / "mtp-bench-v1"
SEQ_DIR = FIXTURE / "sequences"

REFERENCES = {
    "bf16": None,
    "w8a8": ".local-artifacts/phase84/mxfp8",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flatten(document: dict) -> dict:
    """Collect the generate report fields regardless of how deeply they nest."""
    collected: dict = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value)
                elif key not in collected:
                    collected[key] = value
            for key in ("generated_token_ids", "input_token_ids", "cleanup"):
                if key in node and key not in collected:
                    collected[key] = node[key]
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(document)
    for key in ("generated_token_ids", "input_token_ids", "cleanup"):
        found = find_key(document, key)
        if found is not None:
            collected[key] = found
    return collected


def find_key(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = find_key(item, key)
            if found is not None:
                return found
    return None


def run_one(binary: pathlib.Path, model: str, prompt: str, sidecar: str | None,
            device_index: int, target: str, seed: int, max_new: int,
            env: dict[str, str]) -> dict:
    cmd = [
        str(binary), "generate",
        "--qwen38-nvfp4", model,
        # The reviewed model is instruction tuned: a raw completion prompt makes
        # it emit EOS immediately, so every condition goes through the chat
        # template as a single user turn, which is also the real agent usage.
        "--message", f"user:{prompt}",
        "--max-new-tokens", str(max_new),
        "--device-index", str(device_index),
        "--target", target,
        "--temperature", "1.0",
        "--seed", str(seed),
        "--kv-cache-encoding", "kv-mxfp8-e4",
        "--mtp-draft-width", "2",
    ]
    if sidecar is not None:
        cmd += ["--mtp-weights", str(REPO / sidecar)]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, env=env)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"generate failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:600]}")
    document = json.loads(proc.stdout.decode())
    payload = flatten(document)
    payload["_wall_seconds"] = elapsed
    payload["_state"] = document.get("state")
    return payload


def check_contract(payload: dict) -> list[str]:
    """Fail-closed contract: never accept CPU fallback or a zero dispatch run."""
    problems = []
    if payload.get("fallback_used"):
        problems.append("fallback_used")
    if not payload.get("all_dispatches_hip", False):
        problems.append("not_all_dispatches_hip")
    if int(payload.get("kernel_dispatch_count", 0)) <= 0:
        problems.append("zero_dispatch")
    backend = payload.get("selected_backend")
    if backend is not None and backend != "hip":
        problems.append(f"backend={backend!r}")
    if payload.get("_state") not in (None, "PASS"):
        problems.append(f"state={payload.get('_state')!r}")
    cleanup = payload.get("cleanup")
    if isinstance(cleanup, dict) and any(int(v or 0) for v in cleanup.values()):
        problems.append(f"cleanup={cleanup}")
    if not payload.get("output_text"):
        problems.append("empty_output_text")
    if int(payload.get("completion_tokens", 0)) <= 0:
        problems.append("zero_completion_tokens")
    if payload.get("finish_reason") not in ("length", "stop"):
        problems.append(f"finish_reason={payload.get('finish_reason')!r}")
    if int(payload.get("completion_tokens", 0)) < 32:
        problems.append(f"too_short_completion={payload.get('completion_tokens')}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target", default="gfx1030")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--hip-visible-devices", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--only", default=None, help="comma separated cond_id subset")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True, help="provenance json output")
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    conditions = manifest["conditions"]
    if args.only:
        wanted = set(args.only.split(","))
        conditions = [c for c in conditions if c["cond_id"] in wanted]

    env = dict(os.environ)
    env.pop("SLLM_MX_WA_M1_A16", None)
    env.pop("SLLM_MX_WA_M1_FORCE_BASELINE", None)
    if args.hip_visible_devices is not None:
        env["HIP_VISIBLE_DEVICES"] = args.hip_visible_devices

    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    binary = pathlib.Path(args.binary).resolve()
    records, failures = [], []

    for condition in conditions:
        prompt_path = FIXTURE / condition["prompt_file"]
        prompt = prompt_path.read_text(encoding="utf-8")
        if sha256_bytes(prompt.encode("utf-8")) != condition["prompt_sha256"]:
            failures.append({"cond_id": condition["cond_id"], "error": "prompt_sha256 mismatch"})
            continue

        for ref_name, sidecar in REFERENCES.items():
            try:
                payload = run_one(binary, args.model, prompt, sidecar,
                                  args.device_index, args.target, args.seed,
                                  int(condition["output_tokens"]), env)
            except Exception as error:  # noqa: BLE001 - recorded, not hidden
                failures.append({"cond_id": condition["cond_id"], "reference": ref_name,
                                 "error": str(error)[:600]})
                print(f"FAIL {condition['cond_id']} {ref_name}: {str(error)[:200]}", flush=True)
                continue

            problems = check_contract(payload)
            if problems:
                failures.append({"cond_id": condition["cond_id"], "reference": ref_name,
                                 "error": "contract: " + ",".join(problems)})
                print(f"FAIL {condition['cond_id']} {ref_name}: {problems}", flush=True)
                continue

            # The Qwen3.8 NVFP4 CLI path reports text, not token ids, so the
            # frozen sequence is tokenize(output_text). The sequence is a forced
            # input rather than a reproduction of the original decode path, so a
            # tokenizer round trip that is not bit-identical does not invalidate
            # it; the derivation is recorded and the round trip is checked.
            text = payload["output_text"]
            token_ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=False).ids]
            round_trip = tokenizer.decode(token_ids)
            body = json.dumps({
                "schema_version": "mtp-bench-sequence-v1",
                "cond_id": condition["cond_id"],
                "reference": ref_name,
                "prompt_sha256": condition["prompt_sha256"],
                "derivation": "tokenize(output_text)",
                "output_text_sha256": sha256_bytes(text.encode("utf-8")),
                "tokenizer_round_trip_exact": round_trip == text,
                "reported_completion_tokens": int(payload.get("completion_tokens", 0)),
                "committed_token_ids": token_ids,
            }, ensure_ascii=False, indent=1) + "\n"
            out_path = SEQ_DIR / f"{condition['cond_id']}.{ref_name}.json"
            out_path.write_text(body, encoding="utf-8")

            records.append({
                "cond_id": condition["cond_id"],
                "reference": ref_name,
                "sequence_file": f"sequences/{out_path.name}",
                "sequence_sha256": sha256_bytes(body.encode("utf-8")),
                "committed_token_sha256": sha256_bytes(
                    ",".join(str(t) for t in token_ids).encode("utf-8")),
                "committed_token_count": len(token_ids),
                "reported_completion_tokens": int(payload.get("completion_tokens", 0)),
                "tokenizer_round_trip_exact": round_trip == text,
                "finish_reason": payload.get("finish_reason"),
                "prompt_tokens_rendered": int(payload.get("prompt_tokens", 0)),
                "prompt_tokens_raw_manifest": int(condition["prompt_tokens"]),
                "mtp_proposed_draft_tokens": int(payload.get("mtp_proposed_draft_tokens", 0)),
                "mtp_accepted_draft_tokens": int(payload.get("mtp_accepted_draft_tokens", 0)),
                "mtp_draft_proposal_blocks": int(payload.get("mtp_draft_proposal_blocks", 0)),
                "kernel_dispatch_count": int(payload.get("kernel_dispatch_count", 0)),
                "wall_seconds": round(float(payload["_wall_seconds"]), 3),
            })
            print("OK   %-30s %-5s %4d tok  rendered=%5d raw=%5d  acc=%3d/%3d  %6.1fs" % (
                condition["cond_id"], ref_name, len(token_ids),
                int(payload.get("prompt_tokens", 0)), int(condition["prompt_tokens"]),
                int(payload.get("mtp_accepted_draft_tokens", 0)),
                int(payload.get("mtp_proposed_draft_tokens", 0)),
                payload["_wall_seconds"]), flush=True)

    provenance = {
        "schema_version": "mtp-bench-sequence-provenance-v1",
        "state": "COMPLETE" if not failures else "INCOMPLETE",
        "generated_on": {"exact_target": args.target, "device_index": args.device_index,
                         "hip_visible_devices": args.hip_visible_devices},
        "binary_sha256": sha256_bytes(binary.read_bytes()),
        "model_root": args.model,
        "sampling": {"temperature": 1.0, "seed": args.seed, "mtp_draft_width": 2,
                     "kv_cache_encoding": "kv-mxfp8-e4",
                     "note": "the Qwen3.8 NVFP4 CLI fixes temperature=1.0 and forbids greedy"},
        "references": REFERENCES,
        "sequence_count": len(records),
        "failures": failures,
        "records": records,
    }
    pathlib.Path(args.out).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nfrozen={len(records)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
