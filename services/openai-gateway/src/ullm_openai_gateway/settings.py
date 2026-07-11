"""Fail-closed product configuration."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PRODUCT_ROOT = Path(
    "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1"
)
DEFAULT_TOKENIZER_DIR = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
)


class SettingsError(RuntimeError):
    """Raised when service configuration cannot meet the product contract."""


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    worker_binary: Path
    artifact_dir: Path
    package_dir: Path
    tokenizer_dir: Path
    api_key_file: Path
    gpu_lock_file: Path
    bind_host: str = "127.0.0.1"
    bind_port: int = 8000

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        product_root = Path(os.environ.get("ULLM_PRODUCT_ROOT", DEFAULT_PRODUCT_ROOT))
        return cls(
            worker_binary=Path(
                os.environ.get(
                    "ULLM_WORKER_BINARY",
                    "/opt/ullm/bin/ullm-sq8-worker",
                )
            ),
            artifact_dir=Path(
                os.environ.get("ULLM_ARTIFACT_DIR", product_root / "artifact")
            ),
            package_dir=Path(
                os.environ.get("ULLM_PACKAGE_DIR", product_root / "package")
            ),
            tokenizer_dir=Path(
                os.environ.get("ULLM_TOKENIZER_DIR", DEFAULT_TOKENIZER_DIR)
            ),
            api_key_file=Path(
                os.environ.get("ULLM_API_KEY_FILE", "/etc/ullm/openai-api-key")
            ),
            gpu_lock_file=Path(
                os.environ.get("ULLM_GPU_LOCK_FILE", "/run/lock/ullm-r9700.lock")
            ),
            bind_host=os.environ.get("ULLM_BIND_HOST", "127.0.0.1"),
            bind_port=_parse_port(os.environ.get("ULLM_BIND_PORT", "8000")),
        )

    def validate_paths(self) -> None:
        for path, label in (
            (self.worker_binary, "worker binary"),
            (self.artifact_dir, "artifact directory"),
            (self.package_dir, "package directory"),
            (self.tokenizer_dir, "tokenizer directory"),
        ):
            if label == "worker binary":
                if not path.is_file() or not os.access(path, os.X_OK):
                    raise SettingsError(f"{label} is not an executable regular file")
            elif not path.is_dir():
                raise SettingsError(f"{label} is not a directory")

        if self.bind_host not in {"127.0.0.1", "172.20.0.1"}:
            raise SettingsError(
                "bind host must be loopback or the fixed OpenWebUI bridge"
            )


def _parse_port(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise SettingsError("bind port must be an integer") from error
    if not 1 <= value <= 65_535:
        raise SettingsError("bind port is outside 1..65535")
    return value


def read_api_key(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SettingsError("API key file is absent or unreadable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SettingsError("API key path is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = handle.read(65_537)
    finally:
        os.close(descriptor)
    if len(value) > 65_536:
        raise SettingsError("API key is too large")
    if value.endswith(b"\r\n"):
        value = value[:-2]
    elif value.endswith(b"\n"):
        value = value[:-1]
    if not value or b"\n" in value or b"\r" in value:
        raise SettingsError("API key file must contain exactly one nonempty line")
    return value
