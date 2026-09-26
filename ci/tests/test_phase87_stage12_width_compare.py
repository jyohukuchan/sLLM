from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from phase87_stage12_acceptance import Stage12IdentityError, load_identity  # noqa: E402
from phase87_stage12_width_compare import calculate  # noqa: E402


def _write_report(path: Path, width: int, identity: dict, speed: float, target: str = "gfx1030") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "phase87-stage12-m4-width-v2",
                "state": "PASS",
                "target": target,
                "width": width,
                "model_sha256": "a" * 64,
                "manifest_sha256": identity["manifest_sha256"],
                "conditions": 26,
                "per_prompt": [
                    {"case_id": case, "m4_tokens_per_second": speed}
                    for case in identity["conditions"]
                ],
            }
        )
        + "\n"
    )


def test_width_comparison_pairs_the_same_prompts_and_keeps_default_decision_separate(tmp_path: Path) -> None:
    identity = load_identity()
    files = {width: tmp_path / f"width{width}.json" for width in (2, 3, 4)}
    for width, speed in ((2, 100.0), (3, 115.0), (4, 95.0)):
        _write_report(files[width], width, identity, speed)
    result = calculate(files[2], files[3], files[4])
    assert result["conditions"] == 26
    assert result["relative_to_width2"]["width3"]["mean"] == pytest.approx(0.15)
    assert result["relative_to_width2"]["width4"]["mean"] == pytest.approx(-0.05)
    assert "AB/BA" in result["note"]

    _write_report(files[4], 4, identity, 95.0, target="gfx1201")
    with pytest.raises(Stage12IdentityError, match="target or model"):
        calculate(files[2], files[3], files[4])

    _write_report(files[4], 4, identity, 95.0)
    invalid = json.loads(files[4].read_text())
    invalid["model_sha256"] = "not-a-digest"
    files[4].write_text(json.dumps(invalid))
    with pytest.raises(Stage12IdentityError, match="model identity is invalid"):
        calculate(files[2], files[3], files[4])
