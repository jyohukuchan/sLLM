#!/usr/bin/env python3
"""Compare M4 expected speed across MTP widths on the same 26 prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase87_stage12_acceptance import Stage12IdentityError, load_identity
from phase87_stage12_m4 import _valid_sha256, paired_interval, sha256


def _read(path: Path, width: int, identity: dict) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage12IdentityError(f"cannot read width {width} M4 report: {error}") from error
    if (document.get("schema_version") != "phase87-stage12-m4-width-v2"
            or document.get("state") != "PASS"
            or document.get("width") != width
            or document.get("manifest_sha256") != identity["manifest_sha256"]
            or document.get("conditions") != 26):
        raise Stage12IdentityError(f"width {width} M4 identity differs")
    if not _valid_sha256(document.get("model_sha256")):
        raise Stage12IdentityError(f"width {width} M4 model identity is invalid")
    rows = document.get("per_prompt")
    if not isinstance(rows, list) or len(rows) != 26:
        raise Stage12IdentityError(f"width {width} M4 prompt rows differ")
    by_case = {row.get("case_id"): row for row in rows if isinstance(row, dict)}
    if set(by_case) != set(identity["conditions"]):
        raise Stage12IdentityError(f"width {width} M4 prompt set differs")
    document["_by_case"] = by_case
    return document


def calculate(width2: Path, width3: Path, width4: Path, manifest: Path | None = None) -> dict:
    identity = load_identity(manifest) if manifest is not None else load_identity()
    documents = {
        2: _read(width2, 2, identity),
        3: _read(width3, 3, identity),
        4: _read(width4, 4, identity),
    }
    targets = {document.get("target") for document in documents.values()}
    models = {document.get("model_sha256") for document in documents.values()}
    if len(targets) != 1 or len(models) != 1 or None in targets or None in models:
        raise Stage12IdentityError("width M4 reports have different target or model identities")
    rows = []
    for case_id in sorted(identity["conditions"]):
        speeds = {}
        for width, document in documents.items():
            speed = document["_by_case"][case_id].get("m4_tokens_per_second")
            if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed <= 0:
                raise Stage12IdentityError(f"width {width} M4 speed is invalid: {case_id}")
            speeds[width] = float(speed)
        rows.append({
            "case_id": case_id,
            "m4_tokens_per_second": {f"width{width}": speeds[width] for width in (2, 3, 4)},
            "relative_to_width2": {
                "width3": speeds[3] / speeds[2] - 1.0,
                "width4": speeds[4] / speeds[2] - 1.0,
            },
        })
    return {
        "schema_version": "phase87-stage12-width-m4-screen-v1",
        "state": "PASS",
        "target": documents[2]["target"],
        "model_sha256": documents[2]["model_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "input_report_sha256": {
            "width2": sha256(width2), "width3": sha256(width3), "width4": sha256(width4)
        },
        "relative_to_width2": {
            width: paired_interval([row["relative_to_width2"][width] for row in rows])
            for width in ("width3", "width4")
        },
        "note": "M4 screening only; WU-12D default selection also requires normal multi-prompt AB/BA",
        "conditions": len(rows),
        "per_prompt": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width2", type=Path, required=True)
    parser.add_argument("--width3", type=Path, required=True)
    parser.add_argument("--width4", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    report = calculate(args.width2, args.width3, args.width4, args.manifest)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['target']} conditions={report['conditions']}")


if __name__ == "__main__":
    main()
