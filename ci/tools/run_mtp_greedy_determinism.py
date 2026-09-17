#!/usr/bin/env python3
"""Greedy determinism test for speculative decoding (claims A and B).

Claim A -- "greedy speculative output == greedy non-speculative output" -- is only
exact in exact arithmetic. The target runs at M=w+1 when verifying and at M=1
when decoding sequentially, which can select a different kernel and a different
reduction order. Phase84.5 already measured such a difference on gfx1201.

Claim B -- "greedy speculative output does not depend on which draft is used" --
is the property the acceptance comparison actually relies on. It is stronger
than claim A because every draft variant runs the same target at the same
verify shape, and the rows that differ between variants are exactly the rows
whose output is discarded. It is still an empirical property, because a
different accept count changes which batch row and which block alignment a
given token position is computed in.

This runner measures both by comparing the committed token sequence across
configurations under greedy sampling.
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


def collect_all(node, key, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                out.append(v)
            else:
                collect_all(v, key, out)
    elif isinstance(node, list):
        for item in node:
            collect_all(item, key, out)
    return out


CONFIGS = [
    # name,              mtp,  companion subdir (None = bundled BF16 companion)
    ("nonspec",          "off", None),
    ("mtp-bf16",         "on",  None),
    ("mtp-mxfp8",        "on",  ".local-artifacts/phase84/mxfp8"),
    ("mtp-mxfp6",        "on",  ".local-artifacts/phase84/mxfp6"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hip-visible-devices", required=True)
    parser.add_argument("--measured", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = pathlib.Path(args.binary).resolve()

    base_env = dict(os.environ)
    for stale in ("SLLM_MX_WA_M1_A16", "SLLM_MX_WA_M1_FORCE_BASELINE"):
        base_env.pop(stale, None)
    base_env.update({
        "LD_LIBRARY_PATH": "/opt/rocm/lib",
        "HIP_VISIBLE_DEVICES": args.hip_visible_devices,
        "SLLM_PHASE78_MODEL_PATH": args.model,
        "SLLM_PHASE78_TARGET": args.target,
        "SLLM_PHASE78_DEVICE": "0",
        "SLLM_PHASE78_WARMUPS": str(args.warmups),
        "SLLM_PHASE78_MEASURED": str(args.measured),
        "SLLM_PHASE78_CHUNK_CAPACITY": "2048",
        "SLLM_PHASE83_MODE": "coding8192",
        "SLLM_PHASE83_KV": "mxfp8",
        "SLLM_PHASE83_SAMPLING": "greedy",
        "SLLM_PHASE83_REPLAY": "0",
        "SLLM_PHASE83_STATE_CAPACITY": "8320",
    })

    results = {}
    failures = []

    for name, mtp, companion in CONFIGS:
        env = dict(base_env)
        env["SLLM_PHASE83_MTP"] = mtp
        if mtp == "on":
            env["SLLM_PHASE83_MTP_WIDTH"] = "2"
        env.pop("SLLM_PHASE84_MTP_COMPANION_PATH", None)
        if companion is not None:
            env["SLLM_PHASE84_MTP_COMPANION_PATH"] = str(REPO / companion)

        started = time.time()
        proc = subprocess.run([str(binary)], capture_output=True, env=env)
        elapsed = time.time() - started
        log = out_dir / f"{args.target}.{name}.json"
        log.write_bytes(proc.stdout)
        (out_dir / f"{args.target}.{name}.err").write_bytes(proc.stderr)

        if proc.returncode != 0:
            failures.append({"config": name, "error": proc.stderr.decode(errors="replace")[:500]})
            print(f"FAIL {name}: exit {proc.returncode}", flush=True)
            continue

        document = json.loads(proc.stdout.decode())
        if document.get("state") != "PASS":
            failures.append({"config": name, "error": f"state={document.get('state')}"})
            print(f"FAIL {name}: state={document.get('state')}", flush=True)
            continue

        hashes = collect_all(document, "generated_tokens_sha256", [])
        tokens = collect_all(document, "generated_tokens", [])
        deterministic = find_key(document, "deterministic_generated_tokens")
        # Every measured repetition of one configuration must agree, otherwise
        # the configuration itself is non-deterministic and cross-config
        # comparison is meaningless.
        unique = sorted(set(hashes))
        if len(unique) != 1:
            failures.append({"config": name, "error": f"intra-config hashes differ: {unique}"})

        results[name] = {
            "mtp": mtp,
            "companion": companion,
            "companion_encoding": find_key(document, "mtp_weight_encoding")
                                  or find_key(document, "companion_encoding"),
            "generated_tokens_sha256": unique[0] if unique else None,
            "intra_config_hash_count": len(unique),
            "repetitions_reported": len(hashes),
            "deterministic_generated_tokens": deterministic,
            "token_count": len(tokens[0]) if tokens else None,
            "tokens": tokens[0] if tokens else None,
            "proposal_blocks": find_key(document, "proposal_blocks"),
            "accepted_draft_tokens": find_key(document, "accepted_draft_tokens"),
            "proposed_draft_tokens": find_key(document, "proposed_draft_tokens"),
            "wall_seconds": round(elapsed, 1),
        }
        print("OK   %-12s sha=%s tokens=%s reps=%d  %.0fs" % (
            name, (unique[0][:16] + "...") if unique else "NONE",
            results[name]["token_count"], len(hashes), elapsed), flush=True)

    def first_divergence(a, b):
        if not a or not b:
            return None
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return {"index": i, "left": x, "right": y}
        return {"index": min(len(a), len(b)), "left": None, "right": None} if len(a) != len(b) else None

    comparisons = []

    def compare(label, left, right, claim):
        if left not in results or right not in results:
            comparisons.append({"label": label, "claim": claim, "state": "SKIPPED"})
            return
        lo, ro = results[left], results[right]
        equal = (lo["generated_tokens_sha256"] == ro["generated_tokens_sha256"]
                 and lo["generated_tokens_sha256"] is not None)
        comparisons.append({
            "label": label, "claim": claim,
            "state": "IDENTICAL" if equal else "DIVERGED",
            "left": left, "right": right,
            "left_sha256": lo["generated_tokens_sha256"],
            "right_sha256": ro["generated_tokens_sha256"],
            "first_divergence": None if equal else first_divergence(lo["tokens"], ro["tokens"]),
        })

    compare("nonspec vs mtp-bf16", "nonspec", "mtp-bf16", "A")
    compare("mtp-bf16 vs mtp-mxfp8", "mtp-bf16", "mtp-mxfp8", "B")
    compare("mtp-bf16 vs mtp-mxfp6", "mtp-bf16", "mtp-mxfp6", "B")
    compare("mtp-mxfp8 vs mtp-mxfp6", "mtp-mxfp8", "mtp-mxfp6", "B")

    report = {
        "schema_version": "mtp-greedy-determinism-v1",
        "state": "COMPLETE" if not failures else "INCOMPLETE",
        "target": args.target,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "hip_visible_devices": args.hip_visible_devices,
        "sampling": "greedy (device argmax, no logits readback)",
        "fixture": "coding8192, 8192 prompt / 128 output, chunk 2048, state 8320, KV mxfp8, MTP width 2",
        "warmups": args.warmups,
        "measured": args.measured,
        "configs": results,
        "comparisons": comparisons,
        "failures": failures,
    }
    (out_dir / f"{args.target}.summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    for c in comparisons:
        print("  claim %s  %-26s %s" % (c["claim"], c["label"], c["state"]))
        if c.get("first_divergence"):
            print("      first divergence: %s" % c["first_divergence"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
