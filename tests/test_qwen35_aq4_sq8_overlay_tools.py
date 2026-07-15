from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_tool("qwen35_aq4_sq8_overlay_builder_test", "build-qwen35-aq4-sq8-overlay.py")
ORACLE = load_tool("qwen35_aq4_sq8_overlay_oracle_test", "run-qwen35-aq4-sq8-overlay-cpu-oracle.py")


def production_config() -> dict:
    return {
        "text_config": {
            "layer_types": [
                "full_attention" if index % 4 == 3 else "linear_attention"
                for index in range(32)
            ]
        }
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    source = tmp_path / "source"
    package = tmp_path / "product/package"
    artifact = tmp_path / "product/artifacts/overlay"
    source.mkdir()
    package.mkdir(parents=True)
    artifact.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps(production_config()), encoding="utf-8")
    (source / "model.safetensors.index.json").write_text('{"weight_map": {}}\n', encoding="utf-8")
    (package / "manifest.json").write_text('{"schema_version": "fixture"}\n', encoding="utf-8")
    names = BUILDER.exact_tensor_names(production_config())
    entries = []
    for index, name in enumerate(names):
        payload = artifact / f"fp8/{index}.bin"
        scale = artifact / f"scales/{index}.bin"
        payload.parent.mkdir(exist_ok=True)
        scale.parent.mkdir(exist_ok=True)
        payload.write_bytes(bytes([index]))
        scale.write_bytes(index.to_bytes(4, "little"))
        family, shape = BUILDER._expected_entry(name)
        entries.append(
            {
                "name": name,
                "family": family,
                "source_dtype": "BF16",
                "shape": shape,
                "payload_dtype": "fp8_e4m3",
                "payload_file": f"fp8/{index}.bin",
                "payload_bytes": 1,
                "payload_sha256": sha(payload),
                "scale_granularity": "row_block",
                "scale_block_cols": 256,
                "scale_dtype": "f32",
                "scale_file": f"scales/{index}.bin",
                "scale_bytes": 4,
                "scale_sha256": sha(scale),
            }
        )
    manifest = {
        "schema_version": BUILDER.SQ_MANIFEST_SCHEMA,
        "candidate": {"id": "SQ8_0", "format_id": "SQ8_0"},
        "storage": {"fp8_tensor_count": 48},
        "fp8_tensors": entries,
    }
    (artifact / "sq_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source, package, artifact, names


def test_exact_production_tensor_set_is_stable() -> None:
    names = BUILDER.exact_tensor_names(production_config())
    assert len(names) == 48
    assert names[0] == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert names[-1] == "model.language_model.layers.30.linear_attn.in_proj_z.weight"
    assert not any(".layers.3." in name for name in names)
    assert BUILDER.tensor_set_sha256(names) == "6fbf047fe19b27a6c9075f06a76fa4bf376ba08ff9d39c84da43461fdf606846"
    assert BUILDER.exact_include_regex(names).startswith("^(?:")
    assert BUILDER.exact_include_regex(names).endswith(")$")


def test_binding_and_overlay_promotion_receipt_are_create_new(tmp_path: Path) -> None:
    source, package, artifact, names = write_fixture(tmp_path)
    product = package.parent
    evidence = product / "evidence.json"
    evidence.write_text('{"verified": true}\n', encoding="utf-8")
    (product / BUILDER.BASE_PROMOTION_NAME).write_text(
        json.dumps(
            {
                "schema_version": BUILDER.PROMOTION_SCHEMA,
                "source_commit": "a" * 40,
                "evidence": {"path": evidence.name, "sha256": sha(evidence)},
            }
        ),
        encoding="utf-8",
    )
    binding = BUILDER.create_binding(artifact, source, package, names)
    assert binding["tensor_set_sha256"] == BUILDER.tensor_set_sha256(names)
    assert binding["content_sha256"] == BUILDER.validate_sq_manifest(artifact, names)[1]
    receipt_path, receipt = BUILDER.create_overlay_promotion_receipt(
        product, binding["content_sha256"]
    )
    assert receipt["overlay"] == {"content_sha256": binding["content_sha256"]}
    assert receipt_path.name == BUILDER.OVERLAY_PROMOTION_NAME
    with pytest.raises(BUILDER.BuildError, match="overwrite binding"):
        BUILDER.create_binding(artifact, source, package, names)
    with pytest.raises(BUILDER.BuildError, match="overwrite JSON"):
        BUILDER.create_overlay_promotion_receipt(product, binding["content_sha256"])


def test_sq_manifest_payload_tamper_is_rejected(tmp_path: Path) -> None:
    _, _, artifact, names = write_fixture(tmp_path)
    (artifact / "fp8/0.bin").write_bytes(b"tampered")
    with pytest.raises(BUILDER.BuildError, match="payload identity differs"):
        BUILDER.validate_sq_manifest(artifact, names)


def test_oracle_metrics_report_exact_and_perturbed_vectors() -> None:
    reference = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32)
    exact = ORACLE.metrics_with_rows(reference, reference)
    assert exact["aggregate"]["max_abs"] == 0.0
    assert exact["aggregate"]["relative_l2"] == 0.0
    candidate = reference.clone()
    candidate[1, 0] += 0.5
    changed = ORACLE.metrics_with_rows(candidate, reference)
    assert changed["aggregate"]["max_abs"] == 0.5
    assert changed["rows"][0]["max_abs"] == 0.0
    assert changed["rows"][1]["max_abs"] == 0.5


def test_oracle_f32_reader_rejects_wrong_geometry(tmp_path: Path) -> None:
    path = tmp_path / "values.f32le"
    path.write_bytes(b"\0" * 4)
    with pytest.raises(ORACLE.OracleError, match="geometry differs"):
        ORACLE.read_f32(path, 2)
