from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
            "relation": TOOL.V1_RELATIONS[index],
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
        "schema_version": TOOL.MANIFEST_SCHEMA_V1,
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
        % (TOOL.MANIFEST_SCHEMA_V1, TOOL.MANIFEST_SCHEMA_V1), encoding="ascii"
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


def v2_fixture(tmp_path: Path) -> tuple[Path, dict]:
    source_identity = {
        "commit": "b" * 40,
        "tree_oid": "c" * 40,
        "archive_sha256": "d" * 64,
    }
    specifications = [
        ("implementation_ready_current", TOOL.CAPTURE_AUDIT_SCHEMA, "implementation_ready", None, "b" * 40),
        ("capture_implementation_no_go", TOOL.CAPTURE_AUDIT_SCHEMA, "implementation_no_go", None, "1" * 40),
        ("capture_implementation_no_go", TOOL.CAPTURE_AUDIT_SCHEMA, "implementation_no_go", None, "2" * 40),
        ("actual_failure", TOOL.PROMOTION_SCHEMA, "actual_failed", "sq8-promotion-" + "3" * 64, "3" * 40),
        ("actual_failure", TOOL.PROMOTION_SCHEMA, "actual_failed", "sq8-promotion-" + "4" * 64, "4" * 40),
        ("actual_failure", TOOL.PROMOTION_SCHEMA, "actual_failed", "sq8-promotion-" + "5" * 64, "5" * 40),
        ("restore_implementation_no_go", TOOL.RUNTIME_AUDIT_SCHEMA, "implementation_no_go", "sq8-promotion-" + "6" * 64, "6" * 40),
    ]
    entries = []
    for sequence, (relation, schema, status, request_id, commit) in enumerate(specifications):
        if schema == TOOL.PROMOTION_SCHEMA:
            source = {
                "schema_version": schema,
                "status": status,
                "request_id": request_id,
                "source_commit": commit,
                "actual": {"status": "failed", "request_id": request_id},
            }
        else:
            source = {
                "schema_version": schema,
                "verdict": status,
                "actual": "not_executed",
                "audited_source": {"commit": commit},
            }
            if relation == "implementation_ready_current":
                source["authorization"] = {
                    "eligible_for_fresh_authorization_builder": True
                }
            if relation == "restore_implementation_no_go":
                source["fixed_request_id"] = request_id
                source["reason_code"] = (
                    "restore_retry_terminal_identity_not_fail_closed"
                )
        entry_path = (tmp_path / f"v2-source-{sequence}.json").resolve()
        digest = publish(entry_path, source)
        entries.append(
            {
                "sequence": sequence,
                "relation": relation,
                "path": str(entry_path),
                "sha256": digest,
                "schema_version": schema,
                "status": status,
                "request_id": request_id,
                "source_commit": commit,
            }
        )
    document = {
        "schema_version": TOOL.MANIFEST_SCHEMA,
        "disposition": "authorization_input_not_yet_runtime_bound",
        "source": source_identity,
        "predecessor": None,
        "entries": entries,
    }
    manifest = (tmp_path / "lineage-v2.json").resolve()
    publish(manifest, document)
    return manifest, document


def test_v2_manifest_and_typed_reference_are_accepted(tmp_path: Path) -> None:
    manifest, document = v2_fixture(tmp_path)
    current = document["entries"][0]
    validated = TOOL.validate_manifest(
        manifest,
        expected_source=document["source"],
        expected_current_implementation_audit={
            "path": current["path"], "sha256": current["sha256"]
        },
    )
    assert validated["authorization_eligible"] is True
    assert validated["entry_count"] == 7
    runtime = (tmp_path / "runtime-v2.json").resolve()
    runtime.write_bytes(validated["raw"])
    runtime.chmod(0o444)
    reference = TOOL.make_reference(validated, runtime)
    assert reference["entry_count"] == 7
    assert TOOL.validate_reference(reference, expected_runtime_path=runtime) == reference


def test_v1_is_diagnostic_only(tmp_path: Path) -> None:
    manifest, _document = fixture(tmp_path)
    validated = TOOL.validate_manifest(manifest)
    assert validated["authorization_eligible"] is False
    assert TOOL.make_reference(validated, manifest)["schema_version"] == TOOL.REFERENCE_SCHEMA_V1


def test_v2_fourth_failure_appends_without_schema_change(tmp_path: Path) -> None:
    previous_path, previous_document = v2_fixture(tmp_path)
    previous = TOOL.validate_manifest(previous_path)
    request_id = "sq8-promotion-" + "7" * 64
    source = {
        "schema_version": TOOL.PROMOTION_SCHEMA,
        "status": "actual_failed",
        "request_id": request_id,
        "source_commit": "7" * 40,
        "actual": {"status": "failed", "request_id": request_id},
    }
    source_path = (tmp_path / "fourth-failure.json").resolve()
    digest = publish(source_path, source)
    document = json.loads(json.dumps(previous_document))
    document["predecessor"] = {
        "path": str(previous_path),
        "sha256": previous["sha256"],
        "entries_sha256": previous["entries_sha256"],
        "entry_count": previous["entry_count"],
    }
    document["entries"].append(
        {
            "sequence": 7,
            "relation": "actual_failure",
            "path": str(source_path),
            "sha256": digest,
            "schema_version": TOOL.PROMOTION_SCHEMA,
            "status": "actual_failed",
            "request_id": request_id,
            "source_commit": "7" * 40,
        }
    )
    appended_path = (tmp_path / "lineage-v2-appended.json").resolve()
    publish(appended_path, document)
    appended = TOOL.validate_manifest(appended_path)
    assert appended["document"]["schema_version"] == TOOL.MANIFEST_SCHEMA
    assert appended["entry_count"] == 8


@pytest.mark.parametrize("mutation", ["delete", "replace", "reorder", "duplicate"])
def test_v2_append_only_history_mutation_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    previous_path, previous_document = v2_fixture(tmp_path)
    previous = TOOL.validate_manifest(previous_path)
    document = json.loads(json.dumps(previous_document))
    document["predecessor"] = {
        "path": str(previous_path),
        "sha256": previous["sha256"],
        "entries_sha256": previous["entries_sha256"],
        "entry_count": previous["entry_count"],
    }
    document["entries"].append(json.loads(json.dumps(document["entries"][-1])))
    document["entries"][-1]["sequence"] = 7
    if mutation == "delete":
        document["entries"].pop(1)
        for index, entry in enumerate(document["entries"]):
            entry["sequence"] = index
    elif mutation == "replace":
        document["entries"][1]["relation"] = "historical_implementation_audit"
    elif mutation == "reorder":
        document["entries"][1], document["entries"][2] = (
            document["entries"][2], document["entries"][1]
        )
        document["entries"][1]["sequence"] = 1
        document["entries"][2]["sequence"] = 2
    else:
        document["entries"][-1]["sequence"] = 7
    candidate = (tmp_path / f"mutated-{mutation}.json").resolve()
    publish(candidate, document)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(candidate)


def test_v2_fake_current_go_and_live_entry_drift_fail_closed(tmp_path: Path) -> None:
    manifest, document = v2_fixture(tmp_path)
    current = document["entries"][0]
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(
            manifest,
            expected_current_implementation_audit={
                "path": current["path"], "sha256": "0" * 64
            },
        )

    entry_path = Path(current["path"])
    entry_path.chmod(0o644)
    source = json.loads(entry_path.read_text(encoding="ascii"))
    source["audited_source"]["commit"] = "e" * 40
    publish(entry_path, source)
    with pytest.raises(TOOL.LineageError):
        TOOL.validate_manifest(manifest)


def test_live_file_toctou_identity_change_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _document = v2_fixture(tmp_path)
    real_fstat = TOOL.os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        result = real_fstat(descriptor)
        if calls != 2:
            return result
        fields = {
            name: getattr(result, name)
            for name in (
                "st_mode", "st_nlink", "st_size", "st_dev", "st_ino",
                "st_mtime_ns", "st_ctime_ns",
            )
        }
        fields["st_mtime_ns"] += 1
        return SimpleNamespace(**fields)

    monkeypatch.setattr(TOOL.os, "fstat", changing_fstat)
    with pytest.raises(TOOL.LineageError, match="changed while reading"):
        TOOL.validate_manifest(manifest)
