from __future__ import annotations

from pathlib import Path

import pytest

from ullm_openai_gateway.settings import GatewaySettings, SettingsError, read_api_key


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
