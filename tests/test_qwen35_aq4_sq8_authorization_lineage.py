from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/qwen35_aq4_sq8_authorization_lineage.py"
SPEC = importlib.util.spec_from_file_location("sq8_authorization_lineage", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def publish(path: Path, value: object) -> str:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode("ascii")
    path.write_bytes(raw)
    path.chmod(0o444)
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    request = "sq8-promotion-" + "a" * 64
    sources = [
        {
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "verdict": "implementation_ready",
            "actual": "not_executed",
            "authorization": {"eligible_for_fresh_authorization_builder": True},
        },
        {
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "verdict": "implementation_no_go",
            "actual": "not_executed",
            "reason_codes": ["first"],
        },
        {
            "schema_version": TOOL.CAPTURE_AUDIT_SCHEMA,
            "verdict": "implementation_no_go",
            "actual": "not_executed",
            "reason_codes": ["second"],
        },
        {
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request,
            "actual": {"status": "failed", "request_id": request},
        },
        {
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request,
            "actual": {"status": "failed", "request_id": request},
        },
        {
            "schema_version": TOOL.RUNTIME_AUDIT_SCHEMA,
            "verdict": "implementation_no_go",
            "actual": "not_executed",
            "reason_code": "restore_retry_terminal_identity_not_fail_closed",
        },
    ]
    entries = []
    for index, source in enumerate(sources):
        path = (tmp_path / f"source-{index}.json").resolve()
        digest = publish(path, source)
        common = {
            "relation": TOOL.RELATIONS[index],
            "path": str(path),
            "sha256": digest,
            "schema_version": source["schema_version"],
            "consumed": index != 0,
            "reusable_as_runtime_authorization": False,
        }
        if index == 0:
            common.update(verdict=source["verdict"], actual=source["actual"])
        elif index in {1, 2}:
            common.update(
                verdict=source["verdict"], actual=source["actual"],
                reason_codes=source["reason_codes"],
            )
        elif index in {3, 4}:
            common.update(
                status=source["status"], actual_status="failed", request_id=request,
            )
        else:
            common.update(
                verdict=source["verdict"], actual=source["actual"],
                reason_code=source["reason_code"],
            )
        entries.append(common)
    document = {
        "schema_version": TOOL.MANIFEST_SCHEMA,
        "disposition": "authorization_input_not_yet_runtime_bound",
        "source": {"commit": "b" * 40, "tree_oid": "c" * 40, "archive_sha256": "d" * 64},
        "entries": entries,
    }
    manifest = (tmp_path / "lineage.json").resolve()
    publish(manifest, document)
    return manifest, document


def republish(path: Path, document: dict) -> None:
    path.chmod(0o644)
    publish(path, document)


def test_exact_manifest_and_runtime_reference_are_accepted(tmp_path: Path) -> None:
    manifest, document = fixture(tmp_path)
    validated = TOOL.validate_manifest(manifest, expected_source=document["source"])
    runtime = (tmp_path / "runtime.json").resolve()
    runtime.write_bytes(validated["raw"])
    runtime.chmod(0o444)
    reference = TOOL.make_reference(validated, runtime)
    assert TOOL.validate_reference(reference, expected_runtime_path=runtime) == reference


@pytest.mark.parametrize("mutation", ["unknown", "missing", "reorder", "duplicate", "hash"])
def test_manifest_shape_order_duplicate_and_hash_drift_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    manifest, document = fixture(tmp_path)
    if mutation == "unknown":
        document["unknown"] = True
    elif mutation == "missing":
        document.pop("disposition")
    elif mutation == "reorder":
        document["entries"][0], document["entries"][1] = (
            document["entries"][1], document["entries"][0]
        )
    elif mutation == "duplicate":
        document["entries"][1]["path"] = document["entries"][0]["path"]
        document["entries"][1]["sha256"] = document["entries"][0]["sha256"]
    else:
        document["entries"][0]["sha256"] = "0" * 64
    republish(manifest, document)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    manifest, _document = fixture(tmp_path)
    manifest.chmod(0o644)
    manifest.write_text(
        '{"schema_version":"%s","schema_version":"%s"}\n'
        % (TOOL.MANIFEST_SCHEMA, TOOL.MANIFEST_SCHEMA), encoding="ascii"
    )
    manifest.chmod(0o444)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def test_path_alias_symlink_hardlink_and_mode_fail_closed(tmp_path: Path) -> None:
    manifest, document = fixture(tmp_path)
    aliased = Path(str(manifest.parent / "child" / ".." / manifest.name))
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(aliased)

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(manifest)
    with pytest.raises((TOOL.LineageError, OSError)):
        TOOL.validate_manifest(symlink)

    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(manifest)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)
    hardlink.unlink()

    manifest.chmod(0o644)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)
    manifest.chmod(0o444)
    assert TOOL.validate_manifest(manifest)["document"] == document
