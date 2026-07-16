from __future__ import annotations

import importlib.util
import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRITER = load_tool(
    "sq8_overlay_receipt_writer",
    ROOT / "tools/write-qwen35-aq4-sq8-overlay-promotion-receipt.py",
)
GENERATOR = load_tool("served_model_generator", ROOT / "tools/generate-served-model.py")


def _write_json(path: Path, value: dict, mode: int = 0o644) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(mode)


def _immutable_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


REQUEST_ID = "sq8-promotion-" + "a" * 64
READY_BODY = '{"status":"ready"}'
READINESS = {
    "schema": "ullm.bridge_container_readiness.v1",
    "container": {
        "name": "open-webui",
        "id": "4" * 64,
        "image_id": "sha256:" + "5" * 64,
        "config_image": "ullm/open-webui:test",
    },
    "network": {
        "name": "open-webui-network",
        "id": "6" * 64,
        "driver": "bridge",
        "bridge_interface": "br-" + "6" * 12,
    },
    "endpoint": {
        "url": "http://172.20.0.1:8000/readyz",
        "path": "/readyz",
        "expected_status": 200,
        "expected_body": READY_BODY,
        "expected_body_sha256": hashlib.sha256(READY_BODY.encode("ascii")).hexdigest(),
        "timeout_seconds": 5,
    },
}


def telemetry_binding(value: dict, request_id: str = REQUEST_ID) -> dict:
    return {
        "schema_version": WRITER.TELEMETRY_BINDING_SCHEMA,
        "request_id": request_id,
        "hash_encoding": WRITER.TELEMETRY_HASH_ENCODING,
        "telemetry_sha256": hashlib.sha256(WRITER._canonical(value)).hexdigest(),
    }


def sq8_telemetry() -> dict:
    return {
        "schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry.v1",
        "projection": {
            "single_matvec_count": 0,
            "batch_matvec_count": 1,
            "pair_matvec_count": 1,
            "triple_matvec_count": 0,
            "fallback_count": 0,
        },
        "diagnostic_host_staging": {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        },
    }


def operator_audit_evidence() -> dict:
    return {
        "schema_version": WRITER.OPERATOR_AUDIT_SCHEMA,
        "hash_encoding": WRITER.AUDIT_HASH_ENCODING,
        "source_audit_sha256": "2" * 64,
        "deterministic_digest_sha256": "3" * 64,
        "physical_operation_invocations": 128,
        "total_steps": 129,
        "decode_steps": 1,
        "token_equivalent_operation_coverage": 8256,
        "implementation_counts": [
            {"kind": kind, "implementation_id": implementation, "count": count}
            for kind, implementation, count in WRITER.REQUIRED_OPERATOR_COUNTS
        ],
    }


def load_resolution_evidence() -> dict:
    records = []
    phases = ("cold_prefill", "cached_prefix_prefill", "decode")
    for implementation, (kind, layer_count) in WRITER.LOAD_IMPLEMENTATION_KINDS.items():
        start = 0 if layer_count == 24 else 24
        for layer in range(start, start + layer_count):
            for phase in phases:
                records.append(
                    {
                        "layer_position": layer,
                        "phase": phase,
                        "kind": kind,
                        "implementation_id": implementation,
                        "resolution": "selected",
                    }
                )
    return {
        "schema_version": WRITER.LOAD_RESOLUTIONS_SCHEMA,
        "hash_encoding": WRITER.AUDIT_HASH_ENCODING,
        "record_count": 192,
        "records_sha256": hashlib.sha256(WRITER._canonical(records)).hexdigest(),
        "records": records,
    }


def worker_error_summary() -> dict:
    message = b"worker command failed protocol validation"
    return {
        "schema_version": WRITER.WORKER_ERROR_SCHEMA,
        "event_type": "error",
        "stage": "worker_error",
        "request_id": REQUEST_ID,
        "request_id_matches": True,
        "code": "invalid_request",
        "recoverable": True,
        "canonical_event_hash_encoding": WRITER.WORKER_ERROR_HASH_ENCODING,
        "canonical_event_sha256": "a" * 64,
        "message": {
            "byte_count": len(message),
            "sha256": hashlib.sha256(message).hexdigest(),
            "prefix_text": None,
            "prefix_bytes": 0,
            "prefix_truncated": True,
            "redaction": "omitted_by_capture_privacy_policy",
        },
        "shutdown": {"attempted": True, "completed": True, "error": None},
    }


@pytest.mark.parametrize("field", ["physical_operation_invocations", "total_steps", "decode_steps", "token_equivalent_operation_coverage"])
def test_receipt_operator_audit_rejects_formula_tamper(field: str) -> None:
    value = operator_audit_evidence()
    value[field] -= 1
    with pytest.raises(WRITER.ReceiptError, match="operator audit"):
        WRITER._validate_sq8_operator_audit(value)


def test_receipt_load_resolution_rejects_rehashed_duplicate() -> None:
    value = load_resolution_evidence()
    value["records"][-1] = dict(value["records"][0])
    value["records_sha256"] = hashlib.sha256(
        WRITER._canonical(value["records"])
    ).hexdigest()
    with pytest.raises(WRITER.ReceiptError, match="load-resolution"):
        WRITER._validate_sq8_load_resolutions(value)


def trusted_components() -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path),
            "sha256": WRITER.sha256_file(path),
            "device": path.stat(follow_symlinks=False).st_dev,
            "inode": path.stat(follow_symlinks=False).st_ino,
        }
        for name, path in WRITER.TRUSTED_COMPONENT_PATHS.items()
    }


def _valid_actual_inputs(
    tmp_path: Path,
    fixture: dict[str, Path | dict],
    component_bindings: dict[str, dict[str, object]],
) -> tuple[Path, Path]:
    """Create the smallest complete maintenance/executor pair for a receipt chain."""

    binding = json.loads(Path(fixture["binding"]).read_text(encoding="utf-8"))
    package_sha = WRITER.sha256_file(Path(fixture["package"]))
    maintenance = {
        "schema_version": "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1",
        "promotion_request_id": REQUEST_ID,
        "status": "passed",
        "actual_run_count": 1,
        "failure": None,
        "capture": {"timeouts": dict(WRITER.EXECUTION_TIMEOUTS)},
        "trusted_components": component_bindings,
        "candidate_pre": {"identity": "unchanged"},
        "candidate_post": {"identity": "unchanged"},
        "stopped_observations": [
            {
                "service": {
                    "active": False,
                    "running": False,
                    "main_pid": 0,
                    "worker_pid": 0,
                    "lock_owned": False,
                },
                "owners": {"worker_pids": [], "amd_pids": [], "kfd_pids": []},
            }
        ]
        * 2,
        "lock": {
            "path": "/run/ullm/device-1.lock",
            "held": True,
            "released": True,
        },
        "restore": {"attempted": True, "passed": True},
    }
    maintenance_path = tmp_path / "maintenance-evidence.json"
    _write_json(maintenance_path, maintenance)
    telemetry = sq8_telemetry()
    executor_path = tmp_path / "executor-record.json"
    _write_json(
        executor_path,
        {
            "schema_version": "ullm.production_executor_record.v1",
            "status": "ok",
            "sq8_promotion_evidence": {
                "schema_version": "ullm.qwen35_aq4.sq8_promotion_executor.v1",
                "request_id": REQUEST_ID,
                "manifest_identity": {
                    "implementation_id": WRITER.IMPLEMENTATION_ID,
                    "execution_profile": "sq8-test",
                    "artifact_content_sha256": binding["content_sha256"],
                    "artifact_manifest_sha256": WRITER.sha256_file(
                        Path(fixture["binding"])
                    ),
                    "package_manifest_sha256": package_sha,
                },
                "telemetry": telemetry,
                "telemetry_binding": telemetry_binding(telemetry),
                "operator_audit": operator_audit_evidence(),
                "load_resolutions": load_resolution_evidence(),
                "output_identity": {
                    "token_count": 2,
                    "token_ids_sha256": "4" * 64,
                    "token_ids_recorded": False,
                },
            },
        },
    )
    return maintenance_path, executor_path


@pytest.mark.parametrize("failure_evidence", [False, True])
@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("projection", "batch_matvec_count", True),
        ("projection", "batch_matvec_count", 1.0),
        ("projection", "batch_matvec_count", -1),
        ("projection", "batch_matvec_count", WRITER.SAFE_INT + 1),
        ("diagnostic_host_staging", "read_count", False),
        ("diagnostic_host_staging", "read_count", 0.0),
        ("diagnostic_host_staging", "read_count", -1),
        ("diagnostic_host_staging", "read_count", WRITER.SAFE_INT + 1),
    ],
)
def test_receipt_success_and_failure_telemetry_require_safe_integer_counters(
    failure_evidence: bool, section: str, key: str, replacement: object
) -> None:
    value = sq8_telemetry()
    value[section][key] = replacement
    with pytest.raises(WRITER.ReceiptError):
        WRITER._validate_sq8_telemetry(
            value, require_promotion_thresholds=not failure_evidence
        )


def test_receipt_telemetry_accepts_safe_integer_upper_boundary() -> None:
    value = sq8_telemetry()
    value["projection"]["batch_matvec_count"] = WRITER.SAFE_INT
    value["projection"]["pair_matvec_count"] = WRITER.SAFE_INT
    assert WRITER._validate_sq8_telemetry(value) is value


def test_generator_import_executes_retained_bytes_after_path_replacement(
    tmp_path: Path,
) -> None:
    generator_path = tmp_path / "generator.py"
    retained = b"MARKER = 'original-generator'\n"
    generator_path.write_bytes(retained)
    generator_path.write_text(
        "MARKER = 'replacement-generator'\n", encoding="ascii"
    )

    generator = WRITER._load_generator(generator_path, retained)
    assert generator.MARKER == "original-generator"


def test_actual_retained_writer_generator_validator_chain_ignores_live_replacements(
    tmp_path: Path, fixture: dict[str, Path | dict]
) -> None:
    """Pin writer/generator/validator bytes before replacing every live path."""

    prepared_path = Path(fixture["profile"]).with_name("promotion.json")
    WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=prepared_path,
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )

    trusted_root = tmp_path / "trusted-tools"
    trusted_root.mkdir()
    component_bytes = {
        "maintenance_wrapper": b"maintenance\n",
        "executor_capture": b"capture\n",
        "served_model_generator": Path(GENERATOR.__file__).read_bytes(),
        "promotion_receipt_writer": Path(WRITER.__file__).read_bytes(),
    }
    component_paths: dict[str, Path] = {}
    for name, source in component_bytes.items():
        path = trusted_root / f"{name}.py"
        path.write_bytes(source)
        component_paths[name] = path.resolve()
    component_bindings = {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(component_bytes[name]).hexdigest(),
            "device": path.stat(follow_symlinks=False).st_dev,
            "inode": path.stat(follow_symlinks=False).st_ino,
        }
        for name, path in component_paths.items()
    }
    maintenance_path, executor_path = _valid_actual_inputs(
        tmp_path, fixture, component_bindings
    )

    writer_spec = importlib.util.spec_from_file_location(
        "_retained_sq8_writer", component_paths["promotion_receipt_writer"]
    )
    assert writer_spec is not None and writer_spec.loader is not None
    retained_writer = importlib.util.module_from_spec(writer_spec)
    sys.modules[writer_spec.name] = retained_writer
    exec(
        compile(
            component_bytes["promotion_receipt_writer"],
            str(component_paths["promotion_receipt_writer"]),
            "exec",
        ),
        retained_writer.__dict__,
    )
    retained_writer.TRUSTED_COMPONENT_PATHS = component_paths
    retained_writer.TRUSTED_COMPONENT_APPROVED_ROOT = trusted_root.resolve()

    component_paths["served_model_generator"].write_text(
        "raise RuntimeError('live generator replacement executed')\n", encoding="ascii"
    )
    component_paths["promotion_receipt_writer"].write_text(
        "raise RuntimeError('live writer replacement executed')\n", encoding="ascii"
    )
    output = tmp_path / "promotion-actual-receipt.json"
    value = retained_writer.write_actual_receipt(
        prepared_receipt_path=prepared_path,
        maintenance_evidence_path=maintenance_path,
        executor_record_path=executor_path,
        output_path=output,
        generator_path=component_paths["served_model_generator"],
        trusted_components=component_bindings,
        trusted_component_sources=component_bytes,
    )
    assert value["status"] == "actual_verified"

    retained_generator = retained_writer._load_generator(
        component_paths["served_model_generator"],
        component_bytes["served_model_generator"],
    )
    document = retained_generator.materialize(
        Path(fixture["profile"]),
        receipt_path_override=output,
        overlay_receipt_tool=retained_writer._receipt_validator_dependency(
            component_bytes
        ),
    )
    assert document["promotion"]["source_commit"] == "1" * 40


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Path | dict]:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    _write_json(tokenizer / "tokenizer_config.json", {"chat_template": "{{ messages }}"})

    worker = tmp_path / "ullm-aq4-worker"
    worker.write_bytes(b"worker\n")
    worker.chmod(0o555)

    product = tmp_path / "product"
    artifact_root = product / "artifacts" / "overlay"
    package_root = product / "package"
    artifact_root.mkdir(parents=True)
    package_root.mkdir(parents=True)
    package = package_root / "manifest.json"
    _write_json(package, {"schema_version": "package.v1", "files": []})
    content = "a" * 64
    tensor_set = "b" * 64
    binding = artifact_root / "binding.json"
    _write_json(
        binding,
        {
            "schema_version": "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2",
            "format_id": "AQ4_0",
            "overlay_format_id": "SQ8_0",
            "implementation_id": WRITER.IMPLEMENTATION_ID,
            "tensor_names": [f"tensor_{i:02d}" for i in range(48)],
            "content_sha256": content,
            "tensor_set_sha256": tensor_set,
            "package": {"manifest_sha256": WRITER.sha256_file(package)},
        },
    )
    _immutable_tree(product)

    profile_path = tmp_path / "profile.json"
    profile = {
        "schema_version": "ullm.served_model.profile.v1",
        "tokenizer": {
            "root": str(tokenizer),
            "files": ["tokenizer_config.json"],
            "transformers_version": "4.0",
            "class": "Qwen2Tokenizer",
            "template_options": {},
        },
        "worker": {
            "protocol": "ullm.worker.v1",
            "binary": str(worker),
            "arguments": [],
            "required_environment": [],
            "identity": {"device": "gfx1201", "execution_profile": "sq8-test"},
        },
        "product": {
            "root": str(product),
            "artifact": {
                "manifest_path": "artifacts/overlay/binding.json",
                "content_sha256_from_receipt": ["overlay", "content_sha256"],
            },
            "package": {"manifest_path": "package/manifest.json"},
        },
        "public": {"id": "qwen-test", "revision": "r1"},
        "generation": {"max_new_tokens": 1},
        "format": {"implementation_id": WRITER.IMPLEMENTATION_ID, "id": "AQ4_0"},
        "promotion": {
            "receipt": str(tmp_path / "promotion.json"),
            "source_commit_from_receipt": ["source_commit"],
            "required_schema_version": WRITER.RECEIPT_SCHEMA,
            "overlay_from_receipt": ["overlay"],
            "release_from_receipt": ["release"],
            "package_from_receipt": ["package"],
            "actual_evidence_from_receipt": ["actual"],
            "request_id_from_receipt": ["request_id"],
            "authorization_audit_from_receipt": ["authorization_audit"],
            "readiness_from_receipt": ["readiness"],
            "readiness": READINESS,
            "release_source_commit": "1" * 40,
        },
    }
    _write_json(profile_path, profile)
    served = tmp_path / "served-model.json"
    return {"profile": profile_path, "profile_value": profile, "product": product, "package": package, "binding": binding, "worker": worker, "served": served}


def test_pre_gpu_receipt_is_pending_and_create_new(fixture: dict[str, Path | dict]) -> None:
    receipt_path = Path(fixture["profile"]).with_name("promotion.json")
    value = WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=receipt_path,
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    assert value["status"] == "prepared_not_executed"
    assert value["actual"] == {"status": "pending", "required": True}
    assert value["execution_timeouts"] == WRITER.EXECUTION_TIMEOUTS
    metadata = receipt_path.lstat()
    assert stat.S_IMODE(metadata.st_mode) == 0o444 and metadata.st_nlink == 1
    with pytest.raises(WRITER.ReceiptError, match="already exists"):
        WRITER.write_receipt(
            profile_path=Path(fixture["profile"]),
            output_path=receipt_path,
            source_tree_sha256="2" * 40,
            source_archive_sha256="3" * 64,
            served_model_path=Path(fixture["served"]),
            request_id=REQUEST_ID,
        )
    with pytest.raises(GENERATOR.GenerationError, match="not executable"):
        GENERATOR.materialize(Path(fixture["profile"]))

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["execution_timeouts"]["ready_seconds"] = 899
    receipt_path.chmod(0o644)
    _write_json(receipt_path, tampered)
    with pytest.raises(WRITER.ReceiptError, match="pending"):
        WRITER._load_prepared_receipt(receipt_path)


def test_authorization_audit_is_explicit_and_bound_for_prepared_candidate(
    tmp_path: Path, fixture: dict[str, Path | dict]
) -> None:
    audit_path = tmp_path / "authorization-audit.json"
    _write_json(
        audit_path,
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
            "verdict": "implementation_ready",
        },
    )
    prepared = Path(fixture["profile"]).with_name("promotion.json")
    value = WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=prepared,
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
        authorization_audit_path=audit_path,
    )
    expected_audit = {
        "path": str(audit_path.resolve()),
        "sha256": WRITER.sha256_file(audit_path),
    }
    assert value["authorization_audit"] == expected_audit

    document = GENERATOR._materialize_profile_document(
        Path(fixture["profile"]),
        expected_manifest_path=Path(fixture["served"]),
        allow_prepared=True,
        prepared_only=True,
        overlay_receipt_tool=WRITER._receipt_validator_dependency(),
    )
    assert document["promotion"]["authorization_audit"] == expected_audit

    _write_json(audit_path, {"schema_version": "tampered"})
    with pytest.raises(GENERATOR.GenerationError, match="authorization audit SHA-256 differs"):
        GENERATOR._materialize_profile_document(
            Path(fixture["profile"]),
            expected_manifest_path=Path(fixture["served"]),
            allow_prepared=True,
            prepared_only=True,
        )

    # Restore the audited file, then prove that a receipt SHA mismatch is also
    # rejected even when the path remains unchanged.
    _write_json(
        audit_path,
        {
            "schema_version": "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1",
            "verdict": "implementation_ready",
        },
    )
    tampered = json.loads(prepared.read_text(encoding="utf-8"))
    tampered["authorization_audit"]["sha256"] = "f" * 64
    prepared.chmod(0o644)
    _write_json(prepared, tampered)
    with pytest.raises(GENERATOR.GenerationError, match="authorization audit SHA-256 differs"):
        GENERATOR._materialize_profile_document(
            Path(fixture["profile"]),
            expected_manifest_path=Path(fixture["served"]),
            allow_prepared=True,
            prepared_only=True,
        )


def test_authorization_audit_null_and_profile_mapping_cannot_be_weakened(
    fixture: dict[str, Path | dict]
) -> None:
    prepared = Path(fixture["profile"]).with_name("promotion.json")
    value = WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=prepared,
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    assert value["authorization_audit"] is None
    receipt = json.loads(prepared.read_text(encoding="utf-8"))
    receipt.pop("authorization_audit")
    prepared.chmod(0o644)
    _write_json(prepared, receipt)
    with pytest.raises(GENERATOR.GenerationError, match="authorization audit.*absent"):
        GENERATOR._materialize_profile_document(
            Path(fixture["profile"]),
            expected_manifest_path=Path(fixture["served"]),
            allow_prepared=True,
            prepared_only=True,
        )

    profile = json.loads(Path(fixture["profile"]).read_text(encoding="utf-8"))
    profile["promotion"]["authorization_audit_from_receipt"] = ["release"]
    weakened_output = Path(fixture["profile"]).with_name("promotion-2.json")
    profile["promotion"]["receipt"] = str(weakened_output)
    _write_json(Path(fixture["profile"]), profile)
    with pytest.raises(WRITER.ReceiptError, match="authorization audit binding differs"):
        WRITER.write_receipt(
            profile_path=Path(fixture["profile"]),
            output_path=weakened_output,
            source_tree_sha256="2" * 40,
            source_archive_sha256="3" * 64,
            served_model_path=Path(fixture["served"]),
            request_id=REQUEST_ID,
        )


def test_profile_weakening_and_live_inventory_change_are_rejected(
    fixture: dict[str, Path | dict]
) -> None:
    profile_path = Path(fixture["profile"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["promotion"]["evidence_from_receipt"] = ["evidence"]
    _write_json(profile_path, profile)
    with pytest.raises(WRITER.ReceiptError, match="contract is incomplete"):
        WRITER.write_receipt(
            profile_path=profile_path,
            output_path=profile_path.with_name("promotion.json"),
            source_tree_sha256="2" * 40,
            source_archive_sha256="3" * 64,
            served_model_path=Path(fixture["served"]),
            request_id=REQUEST_ID,
        )

    profile["promotion"].pop("evidence_from_receipt")
    _write_json(profile_path, profile)
    WRITER.write_receipt(
        profile_path=profile_path,
        output_path=profile_path.with_name("promotion.json"),
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    Path(fixture["binding"]).chmod(0o644)
    with pytest.raises(GENERATOR.GenerationError, match="not executable"):
        GENERATOR.materialize(profile_path)


def test_generate_rejects_symlink_output(tmp_path: Path, fixture: dict[str, Path | dict]) -> None:
    WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=Path(fixture["profile"]).with_name("promotion.json"),
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    target = tmp_path / "target.json"
    target.write_text("keep\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(GENERATOR.GenerationError, match="symlink"):
        GENERATOR.generate(Path(fixture["profile"]), link)
    assert target.read_text(encoding="utf-8") == "keep\n"


def test_actual_evidence_uses_maintenance_stable2(tmp_path: Path, fixture: dict[str, Path | dict]) -> None:
    profile = fixture["profile_value"]
    assert isinstance(profile, dict)
    binding = json.loads(Path(fixture["binding"]).read_text(encoding="utf-8"))
    package_sha = WRITER.sha256_file(Path(fixture["package"]))
    snapshot = {"identity": "unchanged"}
    maintenance = {
        "schema_version": "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1",
        "promotion_request_id": REQUEST_ID,
        "status": "passed",
        "actual_run_count": 1,
        "failure": None,
        "capture": {"timeouts": dict(WRITER.EXECUTION_TIMEOUTS)},
        "trusted_components": trusted_components(),
        "candidate_pre": snapshot,
        "candidate_post": snapshot,
        "stopped_observations": [
            {"service": {"active": False, "running": False, "main_pid": 0, "worker_pid": 0, "lock_owned": False}, "owners": {"worker_pids": [], "amd_pids": [], "kfd_pids": []}},
            {"service": {"active": False, "running": False, "main_pid": 0, "worker_pid": 0, "lock_owned": False}, "owners": {"worker_pids": [], "amd_pids": [], "kfd_pids": []}},
        ],
        "lock": {"path": "/run/ullm/device-1.lock", "held": True, "released": True},
        "restore": {"attempted": True, "passed": True},
    }
    maintenance_path = tmp_path / "maintenance-evidence.json"
    _write_json(maintenance_path, maintenance)
    executor_path = tmp_path / "executor-record.json"
    telemetry = {
        "schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry.v1",
        "projection": {
            "single_matvec_count": 0,
            "batch_matvec_count": 1,
            "pair_matvec_count": 1,
            "triple_matvec_count": 0,
            "fallback_count": 0,
        },
        "diagnostic_host_staging": {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        },
    }
    _write_json(
        executor_path,
        {
            "schema_version": "ullm.production_executor_record.v1",
            "status": "ok",
            "sq8_promotion_evidence": {
                "schema_version": "ullm.qwen35_aq4.sq8_promotion_executor.v1",
                "request_id": REQUEST_ID,
                "manifest_identity": {
                    "implementation_id": WRITER.IMPLEMENTATION_ID,
                    "execution_profile": "sq8-test",
                    "artifact_content_sha256": binding["content_sha256"],
                    "artifact_manifest_sha256": WRITER.sha256_file(Path(fixture["binding"])),
                    "package_manifest_sha256": package_sha,
                },
                "telemetry": telemetry,
                "telemetry_binding": telemetry_binding(telemetry),
                "operator_audit": operator_audit_evidence(),
                "load_resolutions": load_resolution_evidence(),
                "output_identity": {"token_count": 2, "token_ids_sha256": "4" * 64, "token_ids_recorded": False},
            },
        },
    )
    output = Path(fixture["profile"]).with_name("promotion-actual-receipt.json")
    WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=Path(fixture["profile"]).with_name("promotion.json"),
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    wrong_executor = json.loads(executor_path.read_text(encoding="utf-8"))
    wrong_executor["sq8_promotion_evidence"]["request_id"] = "sq8-promotion-" + "f" * 64
    _write_json(executor_path, wrong_executor)
    with pytest.raises(WRITER.ReceiptError, match="request ID differs"):
        WRITER.write_actual_receipt(
            prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
            maintenance_evidence_path=maintenance_path,
            executor_record_path=executor_path,
            output_path=output,
        )
    wrong_executor["sq8_promotion_evidence"]["request_id"] = REQUEST_ID
    wrong_executor["sq8_promotion_evidence"]["output_identity"]["token_count"] = 1
    _write_json(executor_path, wrong_executor)
    with pytest.raises(WRITER.ReceiptError, match="output identity differs"):
        WRITER.write_actual_receipt(
            prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
            maintenance_evidence_path=maintenance_path,
            executor_record_path=executor_path,
            output_path=output,
        )
    wrong_executor["sq8_promotion_evidence"]["output_identity"]["token_count"] = 2
    _write_json(executor_path, wrong_executor)
    value = WRITER.write_actual_receipt(
        prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
        maintenance_evidence_path=maintenance_path,
        executor_record_path=executor_path,
        output_path=output,
    )
    assert value["status"] == "actual_verified"
    assert value["actual"]["gpu_exclusive_preflight"]["mode"] == "maintenance_stable2"
    assert value["actual"]["telemetry_binding"] == telemetry_binding(telemetry)
    assert value["actual"]["trusted_components"] == trusted_components()
    assert GENERATOR.materialize(
        Path(fixture["profile"]), receipt_path_override=output
    )["promotion"]["source_commit"] == "1" * 40
    for index, invalid_digest in enumerate(("A" * 64, "z" * 64)):
        tampered = json.loads(output.read_text(encoding="utf-8"))
        tampered["actual"]["prepared_receipt"]["sha256"] = invalid_digest
        tampered_path = tmp_path / f"tampered-actual-{index}.json"
        _write_json(tampered_path, tampered)
        with pytest.raises(GENERATOR.GenerationError, match="lowercase hexadecimal"):
            GENERATOR.materialize(
                Path(fixture["profile"]), receipt_path_override=tampered_path
            )
    with pytest.raises(WRITER.ReceiptError, match="already exists"):
        WRITER.write_actual_receipt(
            prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
            maintenance_evidence_path=maintenance_path,
            executor_record_path=executor_path,
            output_path=output,
        )

    executor = json.loads(executor_path.read_text(encoding="utf-8"))
    executor["sq8_promotion_evidence"]["request_id"] = "sq8-promotion-" + "b" * 64
    _write_json(executor_path, executor)
    with pytest.raises(WRITER.ReceiptError, match="request ID"):
        WRITER.validate_actual_evidence(
            maintenance_path=maintenance_path,
            executor_path=executor_path,
            output_path=output,
            profile=profile,
            overlay={
                "binding_manifest_sha256": WRITER.sha256_file(Path(fixture["binding"])),
                "content_sha256": binding["content_sha256"],
            },
            package_sha256=package_sha,
            request_id=REQUEST_ID,
            prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
        )
    executor["sq8_promotion_evidence"]["request_id"] = REQUEST_ID
    executor["sq8_promotion_evidence"]["telemetry"]["projection"]["pair_matvec_count"] = 0
    _write_json(executor_path, executor)
    with pytest.raises(WRITER.ReceiptError, match="batch and pair"):
        WRITER.validate_actual_evidence(
            maintenance_path=maintenance_path,
            executor_path=executor_path,
            output_path=output,
            profile=profile,
            overlay={
                "binding_manifest_sha256": WRITER.sha256_file(Path(fixture["binding"])),
                "content_sha256": binding["content_sha256"],
            },
            package_sha256=package_sha,
            request_id=REQUEST_ID,
            prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
        )
    executor["sq8_promotion_evidence"]["telemetry"]["projection"]["pair_matvec_count"] = 1
    executor["sq8_promotion_evidence"]["telemetry_binding"]["telemetry_sha256"] = "0" * 64
    _write_json(executor_path, executor)
    with pytest.raises(WRITER.ReceiptError, match="telemetry binding differs"):
        WRITER.validate_actual_evidence(
            maintenance_path=maintenance_path,
            executor_path=executor_path,
            output_path=output,
            profile=profile,
            overlay={
                "binding_manifest_sha256": WRITER.sha256_file(Path(fixture["binding"])),
                "content_sha256": binding["content_sha256"],
            },
            package_sha256=package_sha,
            request_id=REQUEST_ID,
            prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
        )
    executor["sq8_promotion_evidence"]["telemetry_binding"] = telemetry_binding(
        executor["sq8_promotion_evidence"]["telemetry"]
    )
    executor["sq8_promotion_evidence"]["telemetry_binding"]["request_id"] = (
        "sq8-promotion-" + "b" * 64
    )
    _write_json(executor_path, executor)
    with pytest.raises(WRITER.ReceiptError, match="telemetry binding differs"):
        WRITER.validate_actual_evidence(
            maintenance_path=maintenance_path,
            executor_path=executor_path,
            output_path=output,
            profile=profile,
            overlay={
                "binding_manifest_sha256": WRITER.sha256_file(Path(fixture["binding"])),
                "content_sha256": binding["content_sha256"],
            },
            package_sha256=package_sha,
            request_id=REQUEST_ID,
            prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
        )
    executor["sq8_promotion_evidence"]["telemetry_binding"] = telemetry_binding(
        executor["sq8_promotion_evidence"]["telemetry"]
    )
    for invalid_digest in ("A" * 64, "z" * 64):
        executor["sq8_promotion_evidence"]["output_identity"]["token_ids_sha256"] = invalid_digest
        _write_json(executor_path, executor)
        with pytest.raises(WRITER.ReceiptError, match="lowercase hexadecimal"):
            WRITER.validate_actual_evidence(
                maintenance_path=maintenance_path,
                executor_path=executor_path,
                output_path=output,
                profile=profile,
                overlay={
                    "binding_manifest_sha256": WRITER.sha256_file(Path(fixture["binding"])),
                    "content_sha256": binding["content_sha256"],
                },
                package_sha256=package_sha,
                request_id=REQUEST_ID,
                prepared_receipt_path=Path(fixture["profile"]).with_name("promotion.json"),
            )


def test_failure_receipt_is_separate_and_request_bound(fixture: dict[str, Path | dict]) -> None:
    prepared = Path(fixture["profile"]).with_name("promotion.json")
    WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=prepared,
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    maintenance = Path(fixture["profile"]).with_name("failed-maintenance.json")
    _write_json(
        maintenance,
        {
            "schema_version": "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1",
            "promotion_request_id": REQUEST_ID,
            "status": "failed",
            "trusted_components": trusted_components(),
            "failure": {"reason": "capture failed"},
        },
    )
    output = Path(fixture["profile"]).with_name("promotion-failure-receipt.json")
    value = WRITER.write_failure_receipt(prepared, maintenance, output)
    assert value["status"] == "actual_failed"
    assert value["actual"]["trusted_components"] == trusted_components()
    tampered = json.loads(maintenance.read_text(encoding="utf-8"))
    tampered["trusted_components"]["executor_capture"]["sha256"] = "0" * 64
    tampered_path = maintenance.with_name("tampered-failed-maintenance.json")
    _write_json(tampered_path, tampered)
    tampered_output = maintenance.with_name("tampered-output")
    tampered_output.mkdir()
    with pytest.raises(WRITER.ReceiptError, match="trusted component"):
        WRITER.write_failure_receipt(
            prepared,
            tampered_path,
            tampered_output / "promotion-failure-receipt.json",
        )
    with pytest.raises(WRITER.ReceiptError, match="already exists"):
        WRITER.write_failure_receipt(prepared, maintenance, output)


def test_failure_receipt_strictly_validates_typed_worker_error(
    fixture: dict[str, Path | dict], tmp_path: Path
) -> None:
    prepared = tmp_path / "promotion.json"
    WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=prepared,
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    base_tool_error = {
        "validation": "valid",
        "schema_version": WRITER.CAPTURE_ERROR_SCHEMA,
        "status": "failed",
        "stage": "worker_error",
        "request_id": REQUEST_ID,
        "timeouts": {
            "ready_seconds": 900,
            "request_seconds": 240,
            "shutdown_seconds": 30,
        },
        "worker_error": worker_error_summary(),
        "observed_sq8_promotion_telemetry": None,
        "observed_sq8_promotion_telemetry_binding": None,
    }

    def maintenance_value(tool_error: dict) -> dict:
        return {
            "schema_version": "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1",
            "promotion_request_id": REQUEST_ID,
            "status": "failed",
            "trusted_components": trusted_components(),
            "capture_failure": {"capture_tool_error": tool_error},
        }

    output_dir = tmp_path / "typed-output"
    output_dir.mkdir()
    maintenance = output_dir / "maintenance-evidence.json"
    _write_json(maintenance, maintenance_value(base_tool_error))
    receipt = WRITER.write_failure_receipt(
        prepared, maintenance, output_dir / "promotion-failure-receipt.json"
    )
    assert receipt["status"] == "actual_failed"
    assert receipt["actual"]["maintenance_evidence"]["sha256"] == (
        WRITER.sha256_file(maintenance)
    )

    for index, mutate in enumerate(
        (
            lambda value: value["worker_error"].__setitem__(
                "code", "runtime_failed"
            ),
            lambda value: value["worker_error"].__setitem__("recoverable", 1),
            lambda value: value["worker_error"]["message"].__setitem__(
                "prefix_text", "secret"
            ),
            lambda value: value["worker_error"].__setitem__(
                "canonical_event_sha256", "0" * 63
            ),
            lambda value: value["worker_error"]["shutdown"].update(
                {"attempted": False, "completed": False, "error": None}
            ),
        )
    ):
        tampered = json.loads(json.dumps(base_tool_error))
        mutate(tampered)
        tampered_output = tmp_path / f"typed-output-tampered-{index}"
        tampered_output.mkdir()
        tampered_maintenance = tampered_output / "maintenance-evidence.json"
        _write_json(tampered_maintenance, maintenance_value(tampered))
        with pytest.raises(WRITER.ReceiptError, match="worker error evidence"):
            WRITER.write_failure_receipt(
                prepared,
                tampered_maintenance,
                tampered_output / "promotion-failure-receipt.json",
            )


def test_failure_receipt_rejects_unsafe_observed_telemetry_counter(
    fixture: dict[str, Path | dict],
) -> None:
    prepared = Path(fixture["profile"]).with_name("promotion.json")
    WRITER.write_receipt(
        profile_path=Path(fixture["profile"]),
        output_path=prepared,
        source_tree_sha256="2" * 40,
        source_archive_sha256="3" * 64,
        served_model_path=Path(fixture["served"]),
        request_id=REQUEST_ID,
    )
    telemetry = sq8_telemetry()
    telemetry["projection"]["batch_matvec_count"] = WRITER.SAFE_INT + 1
    maintenance = Path(fixture["profile"]).with_name("unsafe-maintenance.json")
    _write_json(
        maintenance,
        {
            "schema_version": "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1",
            "promotion_request_id": REQUEST_ID,
            "status": "failed",
            "trusted_components": trusted_components(),
            "capture_failure": {
                "capture_tool_error": {
                    "validation": "valid",
                    "schema_version": WRITER.CAPTURE_ERROR_SCHEMA,
                    "status": "failed",
                    "stage": "worker_exit",
                    "request_id": REQUEST_ID,
                    "timeouts": {
                        "ready_seconds": 900,
                        "request_seconds": 240,
                        "shutdown_seconds": 30,
                    },
                    "worker_error": None,
                    "observed_sq8_promotion_telemetry": telemetry,
                    "observed_sq8_promotion_telemetry_binding": telemetry_binding(
                        telemetry
                    ),
                }
            },
        },
    )
    output_root = Path(fixture["profile"]).with_name("unsafe-failure")
    output_root.mkdir()
    with pytest.raises(WRITER.ReceiptError, match="telemetry count"):
        WRITER.write_failure_receipt(
            prepared,
            maintenance,
            output_root / "promotion-failure-receipt.json",
        )
