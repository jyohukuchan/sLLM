from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from ullm_openai_gateway.served_model import (
    MAX_MANIFEST_BYTES,
    MAX_STRING_BYTES,
    ServedModelError,
    load_served_model,
)


FIXTURES = Path(__file__).parent / "fixtures/served-model"


@pytest.mark.parametrize(
    ("name", "model_id", "format_id", "vocab_size", "has_artifact"),
    [
        ("sq8", "ullm-qwen3-14b-sq8", "SQ8_0", 151_936, True),
        ("aq4", "ullm-qwen3.5-9b-aq4", "AQ4_0", 248_320, False),
        (
            "sq8/served-model-fq6.json",
            "ullm-qwen3-14b-fq6-fixture",
            "FQ6_0",
            151_936,
            True,
        ),
    ],
)
def test_quantization_format_fixtures_use_the_same_loader(
    name: str,
    model_id: str,
    format_id: str,
    vocab_size: int,
    has_artifact: bool,
) -> None:
    path = (
        FIXTURES / name
        if name.endswith(".json")
        else FIXTURES / name / "served-model.json"
    )
    loaded = load_served_model(path)

    assert loaded.manifest_path == path.resolve()
    assert len(loaded.manifest_sha256) == 64
    assert loaded.public.id == model_id
    assert loaded.format.format_id == format_id
    assert loaded.generation.vocab_size == vocab_size
    assert (loaded.product.artifact is not None) is has_artifact
    assert loaded.worker.arguments == ("--served-model-manifest", "{manifest}")
    assert loaded.worker.binary.is_absolute()
    assert loaded.tokenizer.root.is_absolute()


def _copy_fixture(tmp_path: Path, name: str = "sq8") -> Path:
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target)
    return target / "served-model.json"


def _document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def test_virtual_format_changes_only_public_and_format_contracts() -> None:
    existing = _document(FIXTURES / "sq8/served-model.json")
    virtual = _document(FIXTURES / "sq8/served-model-fq6.json")

    assert existing["public"] != virtual["public"]
    assert existing["format"] != virtual["format"]
    for section in (
        "schema_version",
        "generation",
        "tokenizer",
        "worker",
        "product",
        "promotion",
    ):
        assert virtual[section] == existing[section]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("unknown", 1),
        lambda value: value.pop("format"),
        lambda value: value["public"].__setitem__("context_length", True),
        lambda value: value["generation"].__setitem__("eos_token_ids", "151645"),
        lambda value: value["worker"]["identity"].__setitem__("extra", "x"),
        lambda value: value.__setitem__("schema_version", "ullm.served_model.v2"),
    ],
)
def test_unknown_missing_wrong_type_and_schema_are_rejected(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    mutate(value)
    _write(path, value)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def test_duplicate_key_is_rejected(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace(
            '{\n  "schema_version"',
            '{\n  "schema_version":"duplicate",\n  "schema_version"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ServedModelError, match="strict JSON"):
        load_served_model(path)


@pytest.mark.parametrize("payload", [b"\xff", b'{"schema_version":NaN}', b"[]"])
def test_non_utf8_nonfinite_and_nonobject_json_are_rejected(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def test_manifest_size_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ServedModelError, match="size limit"):
        load_served_model(path)


@pytest.mark.parametrize(
    "payload",
    [
        ("[" * 17 + "0" + "]" * 17).encode("ascii"),
        json.dumps({"value": "x" * (MAX_STRING_BYTES + 1)}).encode("ascii"),
        json.dumps({"value": [0] * 16_385}).encode("ascii"),
    ],
)
def test_json_depth_string_and_node_counts_are_bounded(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    with pytest.raises(ServedModelError, match="bounds"):
        load_served_model(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["generation"].__setitem__("eos_token_ids", [151936]),
        lambda value: value["generation"].__setitem__(
            "eos_token_ids", [151645, 151645]
        ),
        lambda value: value["generation"].__setitem__("max_completion_tokens", 4097),
        lambda value: value["generation"]["sampling"].__setitem__("top_k", 151937),
        lambda value: value["generation"]["sampling"].__setitem__("temperature", False),
    ],
)
def test_generation_cross_contract_violations_are_rejected(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    mutate(value)
    _write(path, value)
    with pytest.raises(ServedModelError):
        load_served_model(path)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("worker", "binary_sha256", "A" * 64),
        ("worker", "binary_sha256", "0" * 64),
        ("promotion", "receipt_sha256", "0" * 64),
    ],
)
def test_malformed_or_mismatched_sha_is_rejected(
    tmp_path: Path, section: str, field: str, value: str
) -> None:
    path = _copy_fixture(tmp_path)
    document = _document(path)
    document[section][field] = value
    _write(path, document)
    with pytest.raises(ServedModelError):
        load_served_model(path)


def test_tokenizer_and_product_payload_hashes_are_verified(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    (path.parent / "tokenizer/tokenizer.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ServedModelError, match="SHA-256"):
        load_served_model(path)

    path = _copy_fixture(tmp_path, "aq4")
    (path.parent / "product/package/manifest.json").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ServedModelError, match="SHA-256"):
        load_served_model(path)


@pytest.mark.parametrize(
    ("field_path", "unsafe"),
    [
        (("tokenizer", "root"), "../sq8-tokenizer"),
        (("worker", "binary"), "../worker"),
        (("product", "root"), "../product"),
        (("promotion", "receipt"), "../promotion.json"),
    ],
)
def test_relative_roots_cannot_escape_manifest_directory(
    tmp_path: Path, field_path: tuple[str, str], unsafe: str
) -> None:
    path = _copy_fixture(tmp_path)
    document = _document(path)
    document[field_path[0]][field_path[1]] = unsafe
    _write(path, document)
    with pytest.raises(ServedModelError, match="relative path"):
        load_served_model(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("tokenizer", "files"),
        ("product", "package"),
    ],
)
def test_child_paths_cannot_escape_declared_root(
    tmp_path: Path, section: str, field: str
) -> None:
    path = _copy_fixture(tmp_path)
    document = _document(path)
    if section == "tokenizer":
        document[section][field] = {"../promotion.json": "0" * 64}
    else:
        document[section][field]["manifest_path"] = "../promotion.json"
    _write(path, document)
    with pytest.raises(ServedModelError, match="relative path"):
        load_served_model(path)


def test_manifest_and_resource_symlinks_are_rejected(tmp_path: Path) -> None:
    real = _copy_fixture(tmp_path)
    link = tmp_path / "manifest-link.json"
    link.symlink_to(real)
    with pytest.raises(ServedModelError, match="symlink"):
        load_served_model(link)

    tokenizer_file = real.parent / "tokenizer/tokenizer.json"
    replacement = real.parent / "tokenizer/replacement.json"
    replacement.write_bytes(tokenizer_file.read_bytes())
    tokenizer_file.unlink()
    tokenizer_file.symlink_to(replacement.name)
    with pytest.raises(ServedModelError, match="symlink"):
        load_served_model(real)


@pytest.mark.parametrize(
    "target", ["manifest", "worker", "tokenizer", "tokenizer_root", "package"]
)
def test_world_writable_manifest_and_resources_are_rejected(
    tmp_path: Path, target: str
) -> None:
    path = _copy_fixture(tmp_path)
    targets = {
        "manifest": path,
        "worker": path.parent / "worker",
        "tokenizer": path.parent / "tokenizer/tokenizer.json",
        "tokenizer_root": path.parent / "tokenizer",
        "package": path.parent / "product/package/manifest.json",
    }
    selected = targets[target]
    selected.chmod(selected.stat().st_mode | 0o002)
    with pytest.raises(ServedModelError, match="safe"):
        load_served_model(path)


def test_worker_launch_contract_is_strict(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    value = _document(path)
    value["worker"]["arguments"] = ["--manifest", "missing-placeholder"]
    _write(path, value)
    with pytest.raises(ServedModelError, match="manifest"):
        load_served_model(path)

    path = _copy_fixture(tmp_path, "aq4")
    value = _document(path)
    value["worker"]["required_environment"] = ["invalid-name"]
    _write(path, value)
    with pytest.raises(ServedModelError, match="required_environment"):
        load_served_model(path)


def test_worker_binary_must_be_executable(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    binary = path.parent / "worker"
    binary.chmod(0o644)
    with pytest.raises(ServedModelError, match="executable"):
        load_served_model(path)


def test_fixture_permissions_are_not_world_writable() -> None:
    for path in FIXTURES.rglob("*"):
        assert not path.stat().st_mode & 0o002
        if path.name == "worker":
            assert os.access(path, os.X_OK)
