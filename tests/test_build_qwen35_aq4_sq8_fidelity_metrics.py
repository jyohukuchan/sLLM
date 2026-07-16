from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_qwen35_aq4_sq8_fidelity_protocol import Sq8ProtocolTests


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sq8_metrics_builder_test", ROOT / "tools/build-qwen35-aq4-sq8-fidelity-metrics.py")
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def binding_fixture():
    case = Sq8ProtocolTests(methodName="test_actual_receipt_binds_plan_and_freeze_recomputes")
    case.setUp()
    try:
        plan_path = case._plan()
        plan = json.loads(plan_path.read_text())
        identity = plan["identity"]
        receipt = json.loads(Path(identity["sq8_receipt_path"]).read_text())
        binding = json.loads(Path(receipt["overlay"]["binding_manifest_path"]).read_text())
        served_path = Path(identity["served_model"]["path"])
        served = json.loads(served_path.read_text())
        served["worker"] = {"required_environment": [f"GUARD_{index:02d}" for index in range(35)], "identity": {"device": "gfx1201"}}
        served["public"] = {"revision": "sq8-fixture-revision"}
        served_path.write_text(json.dumps(served, sort_keys=True) + "\n")
        profile_path = Path(receipt["release"]["profile"]["path"])
        profile = json.loads(profile_path.read_text())
        profile["worker"]["identity"]["device"] = "gfx1201"
        profile_path.write_text(json.dumps(profile, sort_keys=True) + "\n")
        source_identity = {"model_id": "Qwen/Qwen3.5-9B", "model_revision": "upstream-fixture", "source_checkpoint": {"aggregate_sha256": "7" * 64, "root": "/models/source"}, "tokenizer": {"aggregate_sha256": "8" * 64, "root": "/models/source"}, "hidden_size": 4096, "vocab_size": 248320}
        source = {"manifest": {"schema_version": "ullm.qwen35_aq4_source_calibration.v1", "oracle_kind": "independent_source_full", "identity": source_identity, "cases": {"path": "/cases.json", "sha256": "9" * 64}}}
        runtime = {"name": "ullm-aq4-sq8-fidelity-capture", "one_model_load": True, "split_manifest_sha256": identity["split_manifest_sha256"], "policy_sha256": identity["policy_sha256"], "calibration_cases_sha256": identity["calibration_cases_sha256"], "served_model_manifest_sha256": sha(served_path), "package_manifest_sha256": identity["package"]["manifest_sha256"], "worker_binary_sha256": identity["worker"]["sha256"], "guard_sha256": builder._guard_sha(served["worker"]["required_environment"]), "upstream_model_revision": source_identity["model_revision"], "quantized_artifact_revision": served["public"]["revision"], "source_checkpoint_aggregate_sha256": source_identity["source_checkpoint"]["aggregate_sha256"], "tokenizer_aggregate_sha256": source_identity["tokenizer"]["aggregate_sha256"], "device": {"architecture": "gfx1201"}}
        target_identity = {**source_identity, "format_id": "AQ4_0", "implementation_id": "qwen35_aq4_sq8_linear_qkv_z_overlay_v1", "artifact": {"package_manifest_sha256": identity["package"]["manifest_sha256"], "artifact_manifest_sha256": receipt["overlay"]["binding_manifest_sha256"], "content_sha256": identity["overlay_content_sha256"], "tensor_set_sha256": identity["overlay_tensor_set_sha256"], "tensor_names": list(builder.PROTOCOL.SQ8_RUNTIME_TENSOR_NAMES)}, "package_manifest_sha256": identity["package"]["manifest_sha256"], "worker_binary_sha256": identity["worker"]["sha256"]}
        target = {"manifest": {"schema_version": "ullm.qwen35_aq4_target_calibration.v1", "oracle_kind": "aq4_sq8_target", "identity": target_identity, "runtime": {"runtime": runtime}, "cases": copy.deepcopy(source["manifest"]["cases"])}}
        yield plan, source, target
    finally:
        case.tearDown()


def test_exact_actual_served_profile_overlay_package_source_binding_passes(binding_fixture) -> None:
    plan, source, target = binding_fixture
    builder._bind_target(plan, source, target)


@pytest.mark.parametrize(
    ("label", "mutate"),
    (
        ("oracle", lambda target: target["manifest"].__setitem__("oracle_kind", "aq4_target")),
        ("content", lambda target: target["manifest"]["identity"]["artifact"].__setitem__("content_sha256", "0" * 64)),
        ("tensor-set", lambda target: target["manifest"]["identity"]["artifact"].__setitem__("tensor_set_sha256", "0" * 64)),
        ("package", lambda target: target["manifest"]["runtime" ]["runtime"].__setitem__("package_manifest_sha256", "0" * 64)),
        ("worker", lambda target: target["manifest"]["runtime"]["runtime"].__setitem__("worker_binary_sha256", "0" * 64)),
        ("served", lambda target: target["manifest"]["runtime"]["runtime"].__setitem__("served_model_manifest_sha256", "0" * 64)),
        ("profile-device", lambda target: target["manifest"]["runtime"]["runtime"]["device"].__setitem__("architecture", "gfx1100")),
        ("policy", lambda target: target["manifest"]["runtime"]["runtime"].__setitem__("policy_sha256", "0" * 64)),
        ("source-model", lambda target: target["manifest"]["identity"]["source_checkpoint"].__setitem__("aggregate_sha256", "0" * 64)),
        ("cases", lambda target: target["manifest"]["cases"].__setitem__("sha256", "0" * 64)),
    ),
)
def test_independent_binding_mutations_are_rejected(binding_fixture, label, mutate) -> None:
    plan, source, target = binding_fixture
    tampered = copy.deepcopy(target)
    mutate(tampered)
    with pytest.raises(builder.MetricsBuildError):
        builder._bind_target(plan, source, tampered)


@pytest.mark.parametrize(
    ("label", "mutate"),
    (
        ("unknown-field", lambda artifact: artifact.__setitem__("unknown", "0" * 64)),
        ("missing-field", lambda artifact: artifact.pop("artifact_manifest_sha256")),
        ("wrong-list-type", lambda artifact: artifact.__setitem__("tensor_names", tuple(artifact["tensor_names"]))),
        ("same-set-wrong-order", lambda artifact: artifact["tensor_names"].reverse()),
        ("duplicate-name", lambda artifact: artifact["tensor_names"].__setitem__(1, artifact["tensor_names"][0])),
        ("unknown-name", lambda artifact: artifact["tensor_names"].__setitem__(0, "model.language_model.layers.31.linear_attn.in_proj_qkv.weight")),
        ("scalar-type", lambda artifact: artifact.__setitem__("content_sha256", 0)),
    ),
)
def test_target_artifact_shape_type_order_and_membership_mutations_are_rejected(binding_fixture, label, mutate) -> None:
    plan, source, target = binding_fixture
    tampered = copy.deepcopy(target)
    mutate(tampered["manifest"]["identity"]["artifact"])
    with pytest.raises(builder.MetricsBuildError):
        builder._bind_target(plan, source, tampered)


def test_binding_manifest_membership_tamper_is_rejected_by_receipt_sha(binding_fixture) -> None:
    plan, source, target = binding_fixture
    receipt = json.loads(Path(plan["identity"]["sq8_receipt_path"]).read_text())
    binding_path = Path(receipt["overlay"]["binding_manifest_path"])
    binding = json.loads(binding_path.read_text())
    binding["tensor_names"][0] = "model.language_model.layers.31.linear_attn.in_proj_qkv.weight"
    binding_path.write_text(json.dumps(binding, sort_keys=True) + "\n")
    with pytest.raises(builder.MetricsBuildError, match="SHA differs"):
        builder._bind_target(plan, source, target)
