"""Fail-closed product configuration."""

from __future__ import annotations

import os
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path

from .served_model import ServedModel, ServedModelError, load_served_model


DEFAULT_PRODUCT_ROOT = Path(
    "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1"
)
DEFAULT_TOKENIZER_DIR = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
)

DEFAULT_HIP_GUARDS = (
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
)

LEGACY_MODEL_ENVIRONMENT = frozenset(
    {
        "ULLM_WORKER_BINARY",
        "ULLM_PRODUCT_ROOT",
        "ULLM_ARTIFACT_DIR",
        "ULLM_PACKAGE_DIR",
        "ULLM_TOKENIZER_DIR",
        "ULLM_MODEL_ID",
        "ULLM_MODEL_NAME",
        "ULLM_MODEL_DESCRIPTION",
        "ULLM_MODEL_REVISION",
        "ULLM_ARTIFACT_CONTENT_SHA256",
        "ULLM_PACKAGE_MANIFEST_SHA256",
        "ULLM_DEVICE",
        "ULLM_EXECUTION_PROFILE",
        "ULLM_MODEL_CONTEXT_LENGTH",
        "ULLM_MAX_NEW_TOKENS",
        "ULLM_VOCAB_SIZE",
        "ULLM_EOS_TOKEN_IDS",
        "ULLM_TOP_K",
        "ULLM_HIP_GUARDS",
        "ULLM_WORKER_EXTRA_ARGS",
        "ULLM_TOKENIZER_PROFILE",
    }
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
    model_id: str = "ullm-qwen3-14b-sq8"
    model_revision: str = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
    artifact_content_sha256: str = (
        "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
    )
    package_manifest_sha256: str = (
        "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
    )
    device: str = "gfx1201"
    execution_profile: str = "rdna4_w8a8_block_ck"
    context_length: int = 4_096
    max_new_tokens: int = 512
    vocab_size: int = 151_936
    eos_token_ids: tuple[int, ...] = (151_645, 151_643)
    top_k: int = 20
    hip_visible_devices: str = "1"
    hip_guards: tuple[str, ...] = DEFAULT_HIP_GUARDS
    worker_extra_args: tuple[str, ...] = ()
    tokenizer_profile: str = "qwen3-14b"
    served_model: ServedModel | None = None

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        manifest = os.environ.get("ULLM_SERVED_MODEL_MANIFEST")
        if manifest is not None:
            if not manifest:
                raise SettingsError("ULLM_SERVED_MODEL_MANIFEST must be nonempty")
            mixed = sorted(LEGACY_MODEL_ENVIRONMENT.intersection(os.environ))
            if mixed:
                raise SettingsError(
                    "served-model manifest mode cannot be mixed with legacy model settings"
                )
            try:
                served_model = load_served_model(Path(manifest))
            except ServedModelError as error:
                raise SettingsError(
                    "served-model manifest validation failed"
                ) from error
            artifact = served_model.product.artifact
            artifact_dir = (
                served_model.product.root / Path(artifact.manifest_path).parent
                if artifact is not None
                else served_model.product.root
            )
            package_dir = (
                served_model.product.root
                / Path(served_model.product.package.manifest_path).parent
            )
            return cls(
                worker_binary=served_model.worker.binary,
                artifact_dir=artifact_dir,
                package_dir=package_dir,
                tokenizer_dir=served_model.tokenizer.root,
                api_key_file=Path(
                    os.environ.get("ULLM_API_KEY_FILE", "/etc/ullm/openai-api-key")
                ),
                gpu_lock_file=Path(
                    os.environ.get("ULLM_GPU_LOCK_FILE", "/run/lock/ullm-r9700.lock")
                ),
                bind_host=os.environ.get("ULLM_BIND_HOST", "127.0.0.1"),
                bind_port=_parse_port(os.environ.get("ULLM_BIND_PORT", "8000")),
                model_id=served_model.public.id,
                model_revision=served_model.public.revision,
                artifact_content_sha256=(
                    artifact.content_sha256
                    if artifact is not None
                    else served_model.product.package.manifest_sha256
                ),
                package_manifest_sha256=(served_model.product.package.manifest_sha256),
                device=served_model.worker.identity.device,
                execution_profile=(served_model.worker.identity.execution_profile),
                context_length=served_model.public.context_length,
                max_new_tokens=served_model.generation.max_completion_tokens,
                vocab_size=served_model.generation.vocab_size,
                eos_token_ids=served_model.generation.eos_token_ids,
                top_k=served_model.generation.sampling.top_k,
                hip_visible_devices=_required_text("ULLM_HIP_VISIBLE_DEVICES", "1"),
                hip_guards=served_model.worker.required_environment,
                tokenizer_profile="manifest",
                served_model=served_model,
            )
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
            model_id=_required_text("ULLM_MODEL_ID", "ullm-qwen3-14b-sq8"),
            model_revision=_required_text(
                "ULLM_MODEL_REVISION", "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
            ),
            artifact_content_sha256=_sha256(
                "ULLM_ARTIFACT_CONTENT_SHA256",
                "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
            ),
            package_manifest_sha256=_sha256(
                "ULLM_PACKAGE_MANIFEST_SHA256",
                "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
            ),
            device=_required_text("ULLM_DEVICE", "gfx1201"),
            execution_profile=_required_text(
                "ULLM_EXECUTION_PROFILE", "rdna4_w8a8_block_ck"
            ),
            context_length=_positive_integer("ULLM_MODEL_CONTEXT_LENGTH", "4096"),
            max_new_tokens=_positive_integer("ULLM_MAX_NEW_TOKENS", "512"),
            vocab_size=_positive_integer("ULLM_VOCAB_SIZE", "151936"),
            eos_token_ids=_integer_list("ULLM_EOS_TOKEN_IDS", "151645,151643"),
            top_k=_positive_integer("ULLM_TOP_K", "20"),
            hip_visible_devices=_required_text("ULLM_HIP_VISIBLE_DEVICES", "1"),
            hip_guards=_name_list("ULLM_HIP_GUARDS", DEFAULT_HIP_GUARDS),
            worker_extra_args=_shell_arguments("ULLM_WORKER_EXTRA_ARGS"),
            tokenizer_profile=_choice(
                "ULLM_TOKENIZER_PROFILE", "qwen3-14b", {"qwen3-14b", "qwen35-9b"}
            ),
        )

    def validate_paths(self) -> None:
        if self.served_model is None:
            for path, label in (
                (self.worker_binary, "worker binary"),
                (self.artifact_dir, "artifact directory"),
                (self.package_dir, "package directory"),
                (self.tokenizer_dir, "tokenizer directory"),
            ):
                if label == "worker binary":
                    if not path.is_file() or not os.access(path, os.X_OK):
                        raise SettingsError(
                            f"{label} is not an executable regular file"
                        )
                elif not path.is_dir():
                    raise SettingsError(f"{label} is not a directory")

        if self.bind_host not in {"127.0.0.1", "172.20.0.1"}:
            raise SettingsError(
                "bind host must be loopback or the fixed OpenWebUI bridge"
            )
        if self.max_new_tokens > self.context_length:
            raise SettingsError("maximum new tokens exceed the model context length")
        if any(token_id >= self.vocab_size for token_id in self.eos_token_ids):
            raise SettingsError("EOS token ID is outside the model vocabulary")


def _parse_port(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise SettingsError("bind port must be an integer") from error
    if not 1 <= value <= 65_535:
        raise SettingsError("bind port is outside 1..65535")
    return value


def _required_text(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    if not value or any(character in value for character in "\r\n\0"):
        raise SettingsError(f"{name} must be nonempty single-line text")
    return value


def _positive_integer(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise SettingsError(f"{name} must be an integer") from error
    if value <= 0:
        raise SettingsError(f"{name} must be positive")
    return value


def _integer_list(name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(name, default)
    try:
        values = tuple(int(item, 10) for item in raw.split(","))
    except ValueError as error:
        raise SettingsError(f"{name} must be a comma-separated integer list") from error
    if (
        not values
        or any(value < 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise SettingsError(f"{name} must contain unique nonnegative integers")
    return values


def _sha256(name: str, default: str) -> str:
    value = _required_text(name, default)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SettingsError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _name_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw:
        return ()
    values = tuple(raw.split(","))
    if any(not value or not value.replace("_", "").isalnum() for value in values):
        raise SettingsError(f"{name} must be a comma-separated environment-name list")
    return values


def _shell_arguments(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    try:
        return tuple(shlex.split(raw))
    except ValueError as error:
        raise SettingsError(f"{name} contains invalid shell quoting") from error


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = _required_text(name, default)
    if value not in choices:
        raise SettingsError(f"{name} is not a supported value")
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
