from __future__ import annotations

from pathlib import Path

import pytest

from ullm_openai_gateway.settings import GatewaySettings, SettingsError, read_api_key
from ullm_openai_gateway.settings import LEGACY_MODEL_ENVIRONMENT
from ullm_openai_gateway.worker import WorkerConfig


MANIFEST_FIXTURES = Path(__file__).parent / "fixtures/served-model"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [(b"secret", b"secret"), (b"secret\n", b"secret"), (b"secret\r\n", b"secret")],
)
def test_api_key_accepts_one_optional_terminal_line_ending(
    tmp_path: Path, payload: bytes, expected: bytes
) -> None:
    path = tmp_path / "key"
    path.write_bytes(payload)
    assert read_api_key(path) == expected


@pytest.mark.parametrize("payload", [b"", b"\n", b"a\nb", b"a\r\nb\r\n"])
def test_api_key_rejects_empty_or_multiple_lines(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "key"
    path.write_bytes(payload)
    with pytest.raises(SettingsError):
        read_api_key(path)


def test_api_key_rejects_symlink_and_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"secret")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(SettingsError):
        read_api_key(link)
    with pytest.raises(SettingsError):
        read_api_key(tmp_path)


def test_settings_reject_unspecified_or_named_bind_host(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    worker.write_bytes(b"#!/bin/sh\n")
    worker.chmod(0o700)
    for name in ("artifact", "package", "tokenizer"):
        (tmp_path / name).mkdir()

    def configured(host: str) -> GatewaySettings:
        return GatewaySettings(
            worker_binary=worker,
            artifact_dir=tmp_path / "artifact",
            package_dir=tmp_path / "package",
            tokenizer_dir=tmp_path / "tokenizer",
            api_key_file=tmp_path / "key",
            gpu_lock_file=tmp_path / "lock",
            bind_host=host,
        )

    for host in ("0.0.0.0", "::", "localhost", "192.168.0.66"):
        with pytest.raises(SettingsError):
            configured(host).validate_paths()
    configured("127.0.0.1").validate_paths()
    configured("172.20.0.1").validate_paths()


def test_model_and_worker_contract_can_be_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "ULLM_MODEL_ID": "ullm-qwen3.5-9b-aq4",
        "ULLM_MODEL_REVISION": "aq4-revision",
        "ULLM_ARTIFACT_CONTENT_SHA256": "a" * 64,
        "ULLM_PACKAGE_MANIFEST_SHA256": "b" * 64,
        "ULLM_DEVICE": "gfx1201",
        "ULLM_EXECUTION_PROFILE": "rdna4_aq4",
        "ULLM_MODEL_CONTEXT_LENGTH": "32768",
        "ULLM_MAX_NEW_TOKENS": "1024",
        "ULLM_VOCAB_SIZE": "248320",
        "ULLM_EOS_TOKEN_IDS": "248044,248046",
        "ULLM_TOP_K": "40",
        "ULLM_HIP_VISIBLE_DEVICES": "0",
        "ULLM_HIP_GUARDS": "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
        "ULLM_WORKER_EXTRA_ARGS": "--temperature-floor 0.1",
        "ULLM_TOKENIZER_PROFILE": "qwen35-9b",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = GatewaySettings.from_env()
    config = WorkerConfig.from_settings(settings)

    assert settings.model_id == "ullm-qwen3.5-9b-aq4"
    assert settings.context_length == 32_768
    assert settings.vocab_size == 248_320
    assert settings.eos_token_ids == (248_044, 248_046)
    assert settings.tokenizer_profile == "qwen35-9b"
    assert config.command[-2:] == ("--temperature-floor", "0.1")
    assert config.environment["HIP_VISIBLE_DEVICES"] == "0"
    assert config.environment["ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL"] == "1"
    assert config.model_id == settings.model_id
    assert config.execution_profile == "rdna4_aq4"
    assert config.context_length == 32_768
    assert config.vocab_size == 248_320
    assert config.eos_token_ids == (248_044, 248_046)
    assert config.top_k == 40


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ULLM_ARTIFACT_CONTENT_SHA256", "not-a-digest"),
        ("ULLM_MODEL_CONTEXT_LENGTH", "0"),
        ("ULLM_EOS_TOKEN_IDS", "1,1"),
        ("ULLM_HIP_GUARDS", "INVALID-NAME"),
        ("ULLM_WORKER_EXTRA_ARGS", "'unterminated"),
        ("ULLM_TOKENIZER_PROFILE", "unknown"),
    ],
)
def test_invalid_model_contract_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(SettingsError):
        GatewaySettings.from_env()


@pytest.mark.parametrize(
    ("fixture", "model_id", "vocab_size", "artifact_digest"),
    [
        (
            "sq8",
            "ullm-qwen3-14b-sq8",
            151_936,
            "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
        ),
        (
            "aq4",
            "ullm-qwen3.5-9b-aq4",
            248_320,
            "6cd84661eb27933b61d960d4f3af3c5ff60c571997d47ab96b0065d8191e0ed6",
        ),
        (
            "sq8/served-model-fq6.json",
            "ullm-qwen3-14b-fq6-fixture",
            151_936,
            "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
        ),
    ],
)
def test_manifest_mode_builds_gateway_and_worker_contract(
    monkeypatch: pytest.MonkeyPatch,
    fixture: str,
    model_id: str,
    vocab_size: int,
    artifact_digest: str,
) -> None:
    for name in LEGACY_MODEL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    manifest = (
        MANIFEST_FIXTURES / fixture
        if fixture.endswith(".json")
        else MANIFEST_FIXTURES / fixture / "served-model.json"
    )
    monkeypatch.setenv("ULLM_SERVED_MODEL_MANIFEST", str(manifest))
    monkeypatch.setenv("ULLM_HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ULLM_REQUIRE_HIP_STALE_KERNEL", "1")

    settings = GatewaySettings.from_env()
    config = WorkerConfig.from_settings(settings)

    assert settings.served_model is not None
    assert settings.model_id == model_id
    assert settings.vocab_size == vocab_size
    assert settings.artifact_content_sha256 == artifact_digest
    assert settings.tokenizer_dir == settings.served_model.tokenizer.root
    assert config.command == (
        str(settings.served_model.worker.binary),
        "--served-model-manifest",
        str(manifest.resolve()),
    )
    assert config.worker_schema == "ullm.worker.v1"
    assert config.model_id == model_id
    assert config.vocab_size == vocab_size
    assert config.environment["HIP_VISIBLE_DEVICES"] == "0"
    assert "ULLM_REQUIRE_HIP_STALE_KERNEL" not in config.environment
    for guard in settings.served_model.worker.required_environment:
        assert config.environment[guard] == "1"
    settings.validate_paths()


@pytest.mark.parametrize("legacy_name", sorted(LEGACY_MODEL_ENVIRONMENT))
def test_manifest_mode_rejects_every_legacy_model_environment_setting(
    monkeypatch: pytest.MonkeyPatch, legacy_name: str
) -> None:
    for name in LEGACY_MODEL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "ULLM_SERVED_MODEL_MANIFEST",
        str(MANIFEST_FIXTURES / "sq8/served-model.json"),
    )
    monkeypatch.setenv(legacy_name, "legacy-value")
    with pytest.raises(SettingsError, match="cannot be mixed"):
        GatewaySettings.from_env()


def test_manifest_mode_wraps_manifest_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in LEGACY_MODEL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "served-model.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ULLM_SERVED_MODEL_MANIFEST", str(path))
    with pytest.raises(SettingsError, match="validation failed"):
        GatewaySettings.from_env()
