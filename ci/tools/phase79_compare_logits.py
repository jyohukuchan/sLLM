#!/usr/bin/env python3
"""Compare fixed-input Phase79 captures; report errors without granting quality PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def compare_logits(reference: list[float], candidate: list[float]) -> dict:
    if not reference or len(reference) != len(candidate):
        raise ValueError("logit vectors must be nonempty and have equal vocabulary size")
    if not all(math.isfinite(x) for x in reference + candidate):
        raise ValueError("nonfinite logits")
    def logsumexp(values):
        maximum = max(values)
        return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))
    rz, cz = logsumexp(reference), logsumexp(candidate)
    kl = math.fsum(math.exp(r-rz) * ((r-rz)-(c-cz)) for r, c in zip(reference, candidate))
    if kl < -1e-10:
        raise ValueError("negative KL beyond rounding error")
    err2 = math.fsum((r-c)**2 for r, c in zip(reference, candidate))
    energy = math.fsum(r*r for r in reference)
    top_r = max(range(len(reference)), key=reference.__getitem__)
    top_c = max(range(len(candidate)), key=candidate.__getitem__)
    return {"reference_top1": top_r, "candidate_top1": top_c,
            "top1_match": top_r == top_c, "kld_reference_to_candidate": max(0.0, kl),
            "max_abs_logit_error": max(abs(r-c) for r, c in zip(reference, candidate)),
            "rmse": math.sqrt(err2 / len(reference)),
            "nmse": err2 / energy if energy else (0.0 if not err2 else None),
            "exact_values": reference == candidate}


def compare(reference: dict, candidate: dict) -> dict:
    for value in (reference, candidate):
        if value.get("schema_version") != "phase79-gemma-fixed-logits-v1" or value.get("state") != "CAPTURED":
            raise ValueError("not a successful Phase79 logits capture")
        for key in ("cleanup_current_bytes", "cleanup_retryable", "cleanup_durable"):
            if value.get(key) != 0:
                raise ValueError(f"missing or failed cleanup: {key}")
        if not value.get("rows"):
            raise ValueError("zero selected rows")
        for row in value["rows"]:
            if row.get("fallback_used") is not False or row.get("kernel_dispatch_count", 0) <= 0:
                raise ValueError("missing HIP dispatch or fallback detected")
    for key in ("model_fingerprint", "target", "device_index", "kv_encoding", "fixture"):
        if key not in reference or reference[key] != candidate.get(key):
            raise ValueError(f"comparison identity differs: {key}")
    def positions(value):
        found = {}
        fixture = value["fixture"]
        if len(value["rows"]) != len(fixture):
            raise ValueError("row count differs from fixture")
        for row, case in zip(value["rows"], fixture):
            if row["id"] != case["id"] or len(row["positions"]) != 1 + len(case["continuation"]):
                raise ValueError("incomplete fixture capture")
            for i, pos in enumerate(row["positions"]):
                if pos["position"] != len(case["prompt"])-1+i:
                    raise ValueError("missing or unexpected teacher-forced position")
                key = (row["id"], pos["position"])
                if key in found:
                    raise ValueError("duplicate comparison position")
                found[key] = pos["logits"]
        return found
    rp, cp = positions(reference), positions(candidate)
    if rp.keys() != cp.keys():
        raise ValueError("comparison positions differ")
    rows = [{"case_id": key[0], "position": key[1], **compare_logits(rp[key], cp[key])} for key in rp]
    return {"schema_version": "phase79-logits-comparison-v1", "state": "COMPARED",
            "quality_verdict": "not_evaluated", "numerical_classification": "requires_analysis",
            "model_fingerprint": reference["model_fingerprint"], "target": reference["target"],
            "kv_encoding": reference["kv_encoding"], "positions": rows,
            "summary": {"count": len(rows), "top1_matches": sum(r["top1_match"] for r in rows),
                        "max_kld": max(r["kld_reference_to_candidate"] for r in rows),
                        "max_abs_logit_error": max(r["max_abs_logit_error"] for r in rows)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    report["captures"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in (args.reference, args.candidate)]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
