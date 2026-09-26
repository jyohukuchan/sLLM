#!/usr/bin/env python3
"""Turn held-out BF16 MTP activation observations into NVFP4 input scales.

The input reports are made by the Stage0 secondary path with the Stage3
calibration readback enabled. They must contain the same fixed, disjoint suite
and no quantized companion. This tool never reads mtp-bench-v1 evaluation
outputs to choose a scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

SITES = (
    "mtp.concat.output",
    "layer.64.input_rmsnorm.output",
    "layer.64.full.sigmoid_mul.output",
    "layer.64.post_attention_rmsnorm.output",
    "layer.64.mlp.silu_mul.output",
)
WEIGHT_SITE = {
    "mtp.fc.weight": SITES[0],
    "mtp.layers.0.self_attn.q_proj.weight": SITES[1],
    "mtp.layers.0.self_attn.k_proj.weight": SITES[1],
    "mtp.layers.0.self_attn.v_proj.weight": SITES[1],
    "mtp.layers.0.self_attn.o_proj.weight": SITES[2],
    "mtp.layers.0.mlp.gate_proj.weight": SITES[3],
    "mtp.layers.0.mlp.up_proj.weight": SITES[3],
    "mtp.layers.0.mlp.down_proj.weight": SITES[4],
}


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def require_positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def collect(
    report: dict, suite_sha: str, prompt_contract: dict[str, tuple[str, int, int]]
) -> dict[str, float]:
    case_ids = set(prompt_contract)
    if report.get("state") != "PASS" or report.get("companion_encoding") is not None:
        raise ValueError("calibration report must be a PASS with the BF16 companion")
    if report.get("target") not in ("gfx1030", "gfx1201"):
        raise ValueError("calibration report has an unexpected GPU target")
    if report.get("series") != "bf16-calibration" or report.get("device_index") != 0:
        raise ValueError("calibration report has an unexpected series or device")
    if report.get("cleanup", {}).get("zero") is not True:
        raise ValueError("calibration report did not clean up")
    secondary = report.get("secondary_mtp")
    if not isinstance(secondary, dict):
        raise ValueError("calibration report has no secondary MTP run")
    if secondary.get("suite_file_sha256") != suite_sha:
        raise ValueError("calibration suite hash differs")
    if secondary.get("warmups") != 0 or secondary.get("measured") != 1:
        raise ValueError("calibration secondary run count differs")
    if set(secondary.get("selected_cases", [])) != case_ids or secondary.get("seed") != 123:
        raise ValueError("calibration selection or seed differs")
    entries = secondary.get("entries")
    if not isinstance(entries, list) or {row.get("case_id") for row in entries} != case_ids:
        raise ValueError("calibration cases are missing, duplicated, or extra")
    if len(entries) != len(case_ids):
        raise ValueError("calibration case count differs")
    maxima = {site: 0.0 for site in SITES}
    for row in entries:
        expected_hash, expected_count, expected_output = prompt_contract[row["case_id"]]
        if (row.get("prompt_sha256") != expected_hash
                or row.get("prompt_token_count") != expected_count
                or row.get("output_tokens") != expected_output):
            raise ValueError("calibration prompt identity or output length differs")
        run = row.get("run")
        if not isinstance(run, dict):
            raise ValueError("calibration entry has no run")
        if len(run.get("generated_tokens", [])) != expected_output:
            raise ValueError("calibration generated-token count differs")
        audit = run.get("audit", {})
        mtp = run.get("mtp", {})
        allocation = run.get("allocation_after_request_drop", {})
        if (audit.get("selected_backend") != "hip"
                or audit.get("target") != report["target"]
                or audit.get("fallback_used") is not False
                or audit.get("all_dispatches_hip") is not True
                or audit.get("kernel_dispatch_count", 0) <= 0
                or not isinstance(mtp, dict)
                or mtp.get("draft_fallback_used") is not False
                or mtp.get("draft_all_dispatches_hip") is not True
                or allocation.get("request_state", {}).get("current_bytes") != 0
                or allocation.get("workspace", {}).get("current_bytes") != 0
                or allocation.get("poisoned") is not False):
            raise ValueError("calibration entry lacks HIP-only and cleanup proof")
        observations = run.get("phase87_stage3_calibration_amax")
        if not isinstance(observations, dict) or set(observations) != set(SITES):
            raise ValueError("calibration entry has incomplete activation sites")
        for site in SITES:
            maxima[site] = max(
                maxima[site],
                require_positive_finite(observations[site], f"{row['case_id']}:{site}"),
            )
    return maxima


def expected_prompts(
    suite: dict, manifest: dict, manifest_path: Path, suite_sha: str,
    tokenizer_path: Path,
) -> dict[str, tuple[str, int, int]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError("calibration requires the Python tokenizers package") from error
    if manifest.get("schema_version") != "phase87-stage3-calibration-manifest-v1" or manifest.get("state") != "PASS":
        raise ValueError("calibration input manifest is not a validated v1 fixture")
    if manifest.get("source", {}).get("stage0_suite", {}).get("sha256") != suite_sha.removeprefix("sha256:"):
        raise ValueError("calibration input manifest names a different Stage0 suite")
    max_new_tokens = manifest.get("capture_contract", {}).get("max_new_tokens")
    if max_new_tokens != 32:
        raise ValueError("calibration output-token contract differs")
    rendered_info = manifest.get("rendered_inputs", {})
    rendered_path = manifest_path.parent / rendered_info.get("path", "")
    if (not rendered_path.is_file() or
            sha256(rendered_path) != "sha256:" + rendered_info.get("sha256", "")):
        raise ValueError("calibration rendered inputs differ from manifest")
    records = [json.loads(line) for line in rendered_path.read_text().splitlines()]
    by_id = {record["case_id"]: record for record in records}
    suite_by_id = {case["id"]: case for case in suite["cases"]}
    manifest_by_id = {case["case_id"]: case for case in manifest["cases"]}
    if (len(records) != len(by_id) or set(by_id) != set(suite_by_id)
            or set(by_id) != set(manifest_by_id)):
        raise ValueError("calibration rendered case set differs")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    expected = {}
    for case_id, record in by_id.items():
        source = manifest_by_id[case_id]
        suite_case = suite_by_id[case_id]
        rendered = record.get("rendered_prompt")
        if (not isinstance(rendered, str)
                or record.get("message") != suite_case.get("message")
                or source.get("rendered_prompt_sha256") != hashlib.sha256(rendered.encode()).hexdigest()
                or source.get("max_new_tokens") != max_new_tokens
                or any(source.get("overlap", {}).get(key) for key in (
                    "committed_text_hash_overlap", "committed_token_hash_overlap",
                    "evaluation_hash_overlap", "normalized_text_overlap_paths",
                ))):
            raise ValueError(f"calibration rendered input or exclusion differs: {case_id}")
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        if not ids or any(token < 0 or token >= 248_320 for token in ids):
            raise ValueError(f"calibration prompt tokens are invalid: {case_id}")
        packed = b"".join(struct.pack("<i", token) for token in ids)
        expected[case_id] = (
            "sha256:" + hashlib.sha256(packed).hexdigest(), len(ids), max_new_tokens
        )
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    suite = json.loads(args.suite.read_text())
    case_ids = [row["id"] for row in suite["cases"]]
    if len(case_ids) != len(set(case_ids)) or not case_ids:
        raise ValueError("calibration suite has duplicate or no cases")
    suite_sha = sha256(args.suite)
    input_manifest = json.loads(args.input_manifest.read_text())
    prompts = expected_prompts(
        suite, input_manifest, args.input_manifest, suite_sha, args.tokenizer
    )
    combined = {site: 0.0 for site in SITES}
    sources = []
    seen_targets = set()
    model_identity = None
    for path in args.report:
        report = json.loads(path.read_text())
        target = report.get("target")
        if target in seen_targets:
            raise ValueError("one calibration report per GPU target is allowed")
        seen_targets.add(target)
        observed_identity = (report.get("model_root"), report.get("model_sha256"))
        if not all(isinstance(value, str) and value for value in observed_identity):
            raise ValueError("calibration model identity is missing")
        if model_identity is not None and observed_identity != model_identity:
            raise ValueError("calibration reports use different models")
        model_identity = observed_identity
        observed = collect(report, suite_sha, prompts)
        for site in SITES:
            combined[site] = max(combined[site], observed[site])
        sources.append({"target": target, "report_sha256": sha256(path)})
    if not seen_targets:
        raise ValueError("at least one calibration report is required")
    scales = {}
    for weight, site in WEIGHT_SITE.items():
        scale = f32(combined[site] / (6.0 * 448.0))
        scales[weight] = require_positive_finite(scale, f"input scale for {weight}")
    report_hashes = "\n".join(sorted(item["report_sha256"] for item in sources))
    payload = {
        "schema": "qwen38-mtp-nvfp4-activation-scale-v1",
        "scale_rule": "f32(max_abs_bf16_activation / (6 * 448))",
        "input_manifest_sha256": sha256(args.input_manifest),
        "suite_sha256": suite_sha,
        "source_report_sha256": "sha256:" + hashlib.sha256(report_hashes.encode()).hexdigest(),
        "source_reports": sorted(sources, key=lambda item: item["target"]),
        "activation_amax": combined,
        "scales": scales,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "scales": scales}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
