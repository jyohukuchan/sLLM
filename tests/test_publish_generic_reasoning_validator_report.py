from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/publish-generic-reasoning-validator-report.py"
SPEC = importlib.util.spec_from_file_location(
    "_test_publish_generic_reasoning_validator_report",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def immutable(path: Path, value: dict[str, Any]) -> Path:
    path.write_bytes(canonical(value))
    path.chmod(0o444)
    return path


def validator(schema: str, *, gate_eligible: bool = True):
    def validate(path: Path) -> dict[str, Any]:
        assert path.is_file()
        return {
            "schema_version": schema,
            "input_sha256": "a" * 64,
            "gate_eligible": gate_eligible,
            "reasons": [] if gate_eligible else ["fixture failure"],
        }

    return validate


@pytest.mark.parametrize(
    ("kind", "input_schema", "report_schema"),
    (
        (
            "release",
            "ullm.generic_reasoning_release_evidence.v1",
            "ullm.generic_reasoning_release_validator.v1",
        ),
        (
            "browser",
            "ullm.openwebui.reasoning_browser_smoke.v2",
            "ullm.openwebui.reasoning_browser_smoke_validator.v1",
        ),
    ),
)
def test_publish_aq4_bundle_v1_report(
    tmp_path: Path,
    kind: str,
    input_schema: str,
    report_schema: str,
) -> None:
    evidence = immutable(
        tmp_path / f"{kind}-evidence.json",
        {"schema_version": input_schema},
    )
    output = tmp_path / f"{kind}-validator.json"

    result = TOOL.publish(
        kind=kind,
        evidence=evidence,
        output=output,
        require_complete=True,
        validator=validator(report_schema),
    )

    assert result["gate_eligible"] is True
    assert result["output"] == os.fspath(output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.stat().st_nlink == 1
    assert json.loads(output.read_text(encoding="ascii"))["schema_version"] == (
        report_schema
    )
    with pytest.raises(TOOL.ReportPublicationError, match="already exists"):
        TOOL.publish(
            kind=kind,
            evidence=evidence,
            output=output,
            require_complete=True,
            validator=validator(report_schema),
        )


@pytest.mark.parametrize(
    ("kind", "schema"),
    (
        ("release", "ullm.generic_reasoning_release_evidence.v2"),
        ("browser", "ullm.openwebui.reasoning_browser_smoke.v5"),
    ),
)
def test_sq8_lineage_schema_cannot_enter_aq4_report(
    tmp_path: Path,
    kind: str,
    schema: str,
) -> None:
    evidence = immutable(tmp_path / "evidence.json", {"schema_version": schema})
    with pytest.raises(
        TOOL.ReportPublicationError,
        match="not an AQ4 bundle-v1 schema",
    ):
        TOOL.publish(
            kind=kind,
            evidence=evidence,
            output=tmp_path / "report.json",
            require_complete=True,
            validator=validator("unused"),
        )


def test_require_complete_rejects_failed_gate(tmp_path: Path) -> None:
    evidence = immutable(
        tmp_path / "evidence.json",
        {"schema_version": "ullm.generic_reasoning_release_evidence.v1"},
    )
    with pytest.raises(
        TOOL.ReportPublicationError,
        match="not gate eligible",
    ):
        TOOL.publish(
            kind="release",
            evidence=evidence,
            output=tmp_path / "report.json",
            require_complete=True,
            validator=validator(
                "ullm.generic_reasoning_release_validator.v1",
                gate_eligible=False,
            ),
        )
    assert not (tmp_path / "report.json").exists()


def test_report_schema_must_match_aq4_branch(tmp_path: Path) -> None:
    evidence = immutable(
        tmp_path / "evidence.json",
        {"schema_version": "ullm.openwebui.reasoning_browser_smoke.v2"},
    )
    with pytest.raises(TOOL.ReportPublicationError, match="schema differs"):
        TOOL.publish(
            kind="browser",
            evidence=evidence,
            output=tmp_path / "report.json",
            require_complete=True,
            validator=validator(
                "ullm.openwebui.reasoning_browser_smoke_validator.v2"
            ),
        )


def test_evidence_must_be_read_only_and_single_link(tmp_path: Path) -> None:
    writable = tmp_path / "writable.json"
    writable.write_bytes(
        canonical(
            {"schema_version": "ullm.generic_reasoning_release_evidence.v1"}
        )
    )
    with pytest.raises(TOOL.ReportPublicationError, match="identity differs"):
        TOOL.publish(
            kind="release",
            evidence=writable,
            output=tmp_path / "report.json",
            require_complete=True,
            validator=validator(
                "ullm.generic_reasoning_release_validator.v1"
            ),
        )

    writable.chmod(0o444)
    hardlink = tmp_path / "hardlink.json"
    os.link(writable, hardlink)
    with pytest.raises(TOOL.ReportPublicationError, match="identity differs"):
        TOOL.publish(
            kind="release",
            evidence=writable,
            output=tmp_path / "second-report.json",
            require_complete=True,
            validator=validator(
                "ullm.generic_reasoning_release_validator.v1"
            ),
        )
