#!/usr/bin/env python3
"""Fail-closed, offline validation for model-lock-v1.

The lock fingerprint uses a deliberately small RFC 8785 JCS implementation.
The fingerprint domain is JSON values made from null, booleans, integers,
strings, arrays, and objects; Python floats are rejected so an accidental
binary floating-point value cannot silently become part of the lock identity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT  # noqa: E402


SCHEMA_PATH = ROOT / "ci/schema/model-lock-v1.schema.json"
LOCK_PATH = ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
REPO_ID = "Qwen/Qwen3.5-4B"
RESOLVED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN_ALIAS = "qwen3.5-4b-bf16"
JCS_SAFE_INTEGER_MIN = -(2**53 - 1)
JCS_SAFE_INTEGER_MAX = 2**53 - 1
SAFETENSORS_HEADER_MAX_BYTES = 100_000_000
UINT64_MAX = 2**64 - 1
UINT32_MAX = 2**32 - 1
MAX_LOCK_JSON_BYTES = 1024 * 1024
MAX_CONFIG_JSON_BYTES = 1024 * 1024
MAX_INDEX_JSON_BYTES = 1024 * 1024
MAX_TOKENIZER_JSON_BYTES = 16 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_COLLECTION_ITEMS = 1_000_000
MAX_JSON_STRING_BYTES = 1024 * 1024
UTC_RFC3339_Z = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$")
SAFETENSORS_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}
QWEN_CONFIG_ROOT_FIELDS = frozenset({
    "architectures", "image_token_id", "model_type", "text_config",
    "tie_word_embeddings", "transformers_version", "video_token_id",
    "vision_config", "vision_end_token_id", "vision_start_token_id",
})
QWEN_VISION_CONFIG = {
    "deepstack_visual_indexes": [],
    "depth": 24,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1024,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "model_type": "qwen3_5",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 2560,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
QWEN_TEXT_OPTIONAL_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "mtp_use_dedicated_embeddings": False,
    "use_cache": True,
}
QWEN_TEXT_ROPE_PARAMETERS = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default",
}


@dataclass(frozen=True)
class _QwenShapeInputs:
    text: dict[str, int]
    vision: dict[str, int]
    layer_types: tuple[str, ...]
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool
    tie_word_embeddings: bool


def _shape_positive(value: Any, *, field: str) -> int:
    if type(value) is not int or not 0 < value <= UINT64_MAX:
        raise ContractError(f"Qwen shape field {field} must be a positive u64 integer")
    return value


def _checked_shape_mul(left: int, right: int, *, field: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise ContractError(f"Qwen shape multiplication inputs are invalid: {field}")
    if right and left > UINT64_MAX // right:
        raise ContractError(f"Qwen shape arithmetic overflows u64: {field}")
    return left * right


def _checked_shape_add(left: int, right: int, *, field: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise ContractError(f"Qwen shape addition inputs are invalid: {field}")
    if left > UINT64_MAX - right:
        raise ContractError(f"Qwen shape arithmetic overflows u64: {field}")
    return left + right


def _qwen_shape_inputs(config: dict[str, Any], model: dict[str, Any]) -> _QwenShapeInputs:
    text_config = config.get("text_config")
    vision_config = config.get("vision_config")
    if type(text_config) is not dict or type(vision_config) is not dict:
        raise ContractError("Qwen shape inputs require text_config and vision_config objects")
    text_fields = (
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
        "full_attention_interval", "linear_conv_kernel_dim", "linear_key_head_dim",
        "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
        "mtp_num_hidden_layers",
    )
    text = {field: _shape_positive(text_config.get(field), field=f"text_config.{field}") for field in text_fields}
    expected_layers = model["architecture"]["layer_schedule"]["layer_types"]
    actual_layers = text_config.get("layer_types")
    if (
        type(actual_layers) is not list
        or tuple(actual_layers) != tuple(expected_layers)
        or tuple(actual_layers) != tuple(["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8)
        or text["num_hidden_layers"] != len(actual_layers)
        or text["full_attention_interval"] != 4
        or text["num_key_value_heads"] > text["num_attention_heads"]
    ):
        raise ContractError("Qwen shape inputs do not match the explicit reviewed layer schedule")

    vision_fields = (
        "depth", "hidden_size", "in_channels", "temporal_patch_size", "patch_size",
        "spatial_merge_size", "intermediate_size", "num_heads",
        "num_position_embeddings", "out_hidden_size",
    )
    vision = {
        field: _shape_positive(vision_config.get(field), field=f"vision_config.{field}")
        for field in vision_fields
    }
    if vision["depth"] != 24:
        raise ContractError("Qwen vision depth must produce the reviewed 297 tensors")
    if vision_config.get("deepstack_visual_indexes") != []:
        raise ContractError("Qwen vision deepstack_visual_indexes must be explicitly empty")

    mtp_dedicated = text_config.get("mtp_use_dedicated_embeddings")
    tied = config.get("tie_word_embeddings")
    if (
        text["mtp_num_hidden_layers"] != 1
        or type(mtp_dedicated) is not bool
        or mtp_dedicated
        or type(tied) is not bool
        or not tied
    ):
        raise ContractError(
            "Qwen MTP requires one layer, tied embeddings, and no dedicated embeddings"
        )
    return _QwenShapeInputs(
        text=text,
        vision=vision,
        layer_types=tuple(actual_layers),
        mtp_num_hidden_layers=text["mtp_num_hidden_layers"],
        mtp_use_dedicated_embeddings=mtp_dedicated,
        tie_word_embeddings=tied,
    )


def _fixed_json_value_matches(actual: Any, expected: Any) -> bool:
    """Compare reviewed JSON constants without Python numeric coercion.

    Rust's ``serde_json::Value`` distinguishes booleans, integer numbers, and
    floating-point numbers.  Python's ordinary equality does not: for example,
    ``True == 1`` and ``248056.0 == 248056``.  Keep the comparison recursive so
    fixed lists and maps have the same exact JSON type contract at every level.
    """

    if expected is None:
        return actual is None
    if type(expected) is bool:
        return type(actual) is bool and actual == expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return type(actual) is float and math.isfinite(actual) and actual == expected
    if type(expected) is str:
        return type(actual) is str and actual == expected
    if type(expected) is list:
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _fixed_json_value_matches(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    if type(expected) is dict:
        return (
            type(actual) is dict
            and all(type(key) is str for key in actual)
            and actual.keys() == expected.keys()
            and all(
                _fixed_json_value_matches(actual[key], expected[key])
                for key in expected
            )
        )
    return False


class JCSValidationError(ValueError):
    """Raised when a value is outside the restricted model-lock JCS domain."""


@dataclass
class _VerifiedCacheFile:
    """A hash-verified descriptor kept open through semantic validation."""

    relative: str
    file_descriptor: int
    identity: tuple[int, int, int, int, int, int]
    size_bytes: int
    sha256: str


QWEN_FILES: dict[str, dict[str, Any]] = {
    "LICENSE": {
        "size_bytes": 11544,
        "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
        "git_blob": "f938136e3adacfd92be087f6e113b5d6d97f678f",
        "lfs_oid": None,
    },
    "README.md": {
        "size_bytes": 77661,
        "sha256": "1406be1b6b8fd8a6545870da516912804756593628a1d0fb0a7965211e82a7bb",
        "git_blob": "7950a3aadf378cd13758097bc52f0ed849a59007",
        "lfs_oid": None,
    },
    "chat_template.jinja": {
        "size_bytes": 7756,
        "sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "git_blob": "a585dec894e63da457d9440ec6aa7caa16d20860",
        "lfs_oid": None,
    },
    "config.json": {
        "size_bytes": 3161,
        "sha256": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
        "git_blob": "557d961b205319c6a7da5f757f565b69b3967b7d",
        "lfs_oid": None,
    },
    "merges.txt": {
        "size_bytes": 3353259,
        "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "git_blob": "a494e019ca1502219fd0128658b979e5f05ae8e8",
        "lfs_oid": None,
    },
    "model.safetensors-00001-of-00002.safetensors": {
        "size_bytes": 5329398688,
        "sha256": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
        "git_blob": "d2e1801a99c9c1618c526812e779d9543df84891",
        "lfs_oid": "sha256:26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    },
    "model.safetensors-00002-of-00002.safetensors": {
        "size_bytes": 3990429408,
        "sha256": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
        "git_blob": "8fbeeec18827d3b20e3df1dc0abf94f7e85bcef1",
        "lfs_oid": "sha256:cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
    },
    "model.safetensors.index.json": {
        "size_bytes": 76196,
        "sha256": "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
        "git_blob": "fddda6039f7c1d17260c9e923b8a72fd025d9a86",
        "lfs_oid": None,
    },
    "preprocessor_config.json": {
        "size_bytes": 390,
        "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "git_blob": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
        "lfs_oid": None,
    },
    "tokenizer.json": {
        "size_bytes": 12807982,
        "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "git_blob": "a73a846725794819aa6e1c9e97d8dc9671c2006d",
        "lfs_oid": "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    },
    "tokenizer_config.json": {
        "size_bytes": 16710,
        "sha256": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        "git_blob": "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1",
        "lfs_oid": None,
    },
    "video_preprocessor_config.json": {
        "size_bytes": 385,
        "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        "git_blob": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
        "lfs_oid": None,
    },
    "vocab.json": {
        "size_bytes": 6722759,
        "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        "git_blob": "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
        "lfs_oid": None,
    },
}


def _qwen_tensor_catalog(inputs: _QwenShapeInputs) -> dict[str, tuple[str, str, tuple[int, ...]]]:
    """Return the reviewed 738-tensor Qwen namespace with class, dtype, shape.

    The catalog is generated from the locked explicit layer schedule, not from
    a prefix count.  It therefore rejects a wrong layer family, an added name,
    or a missing known-unconsumed vision/MTP tensor independently of any graph
    consumer edge that a later runtime may introduce.
    """

    catalog: dict[str, tuple[str, str, tuple[int, ...]]] = {}

    def add(name: str, classification: str, dtype: str, shape: tuple[int, ...]) -> None:
        if name in catalog:
            raise ContractError(f"internal duplicate Qwen tensor catalog entry: {name}")
        catalog[name] = (classification, dtype, shape)

    text = inputs.text
    vision = inputs.vision
    linear_projection_width = _checked_shape_mul(
        text["linear_num_value_heads"], text["linear_value_head_dim"],
        field="linear projection width",
    )
    linear_qkv_width = _checked_shape_add(
        _checked_shape_mul(
            2,
            _checked_shape_mul(
                text["linear_num_key_heads"], text["linear_key_head_dim"],
                field="linear qkv query/key width",
            ),
            field="linear qkv query/key width",
        ),
        _checked_shape_mul(
            text["linear_num_value_heads"], text["linear_value_head_dim"],
            field="linear qkv value width",
        ),
        field="linear qkv width",
    )
    full_q_width = _checked_shape_mul(
        2,
        _checked_shape_mul(
            text["num_attention_heads"], text["head_dim"], field="full query width"
        ),
        field="full query/gate width",
    )
    full_kv_width = _checked_shape_mul(
        text["num_key_value_heads"], text["head_dim"], field="full KV width"
    )
    full_output_width = _checked_shape_mul(
        text["num_attention_heads"], text["head_dim"], field="full output width"
    )

    add(
        "model.language_model.embed_tokens.weight", "text", "BF16",
        (text["vocab_size"], text["hidden_size"]),
    )
    add("model.language_model.norm.weight", "text", "BF16", (text["hidden_size"],))
    for layer, layer_type in enumerate(inputs.layer_types):
        prefix = f"model.language_model.layers.{layer}"
        add(f"{prefix}.input_layernorm.weight", "text", "BF16", (text["hidden_size"],))
        add(f"{prefix}.post_attention_layernorm.weight", "text", "BF16", (text["hidden_size"],))
        add(
            f"{prefix}.mlp.gate_proj.weight", "text", "BF16",
            (text["intermediate_size"], text["hidden_size"]),
        )
        add(
            f"{prefix}.mlp.up_proj.weight", "text", "BF16",
            (text["intermediate_size"], text["hidden_size"]),
        )
        add(
            f"{prefix}.mlp.down_proj.weight", "text", "BF16",
            (text["hidden_size"], text["intermediate_size"]),
        )
        if layer_type == "full_attention":
            add(f"{prefix}.self_attn.q_proj.weight", "text", "BF16", (full_q_width, text["hidden_size"]))
            add(f"{prefix}.self_attn.k_proj.weight", "text", "BF16", (full_kv_width, text["hidden_size"]))
            add(f"{prefix}.self_attn.v_proj.weight", "text", "BF16", (full_kv_width, text["hidden_size"]))
            add(f"{prefix}.self_attn.o_proj.weight", "text", "BF16", (text["hidden_size"], full_output_width))
            add(f"{prefix}.self_attn.q_norm.weight", "text", "BF16", (text["head_dim"],))
            add(f"{prefix}.self_attn.k_norm.weight", "text", "BF16", (text["head_dim"],))
        else:
            add(
                f"{prefix}.linear_attn.in_proj_qkv.weight", "text", "BF16",
                (linear_qkv_width, text["hidden_size"]),
            )
            add(
                f"{prefix}.linear_attn.in_proj_z.weight", "text", "BF16",
                (linear_projection_width, text["hidden_size"]),
            )
            for suffix in ("in_proj_a.weight", "in_proj_b.weight"):
                add(
                    f"{prefix}.linear_attn.{suffix}", "text", "BF16",
                    (text["linear_num_value_heads"], text["hidden_size"]),
                )
            add(
                f"{prefix}.linear_attn.conv1d.weight", "text", "BF16",
                (linear_qkv_width, 1, text["linear_conv_kernel_dim"]),
            )
            add(f"{prefix}.linear_attn.A_log", "text", "F32", (text["linear_num_value_heads"],))
            add(f"{prefix}.linear_attn.dt_bias", "text", "BF16", (text["linear_num_value_heads"],))
            add(
                f"{prefix}.linear_attn.norm.weight", "text", "F32",
                (text["linear_value_head_dim"],),
            )
            add(
                f"{prefix}.linear_attn.out_proj.weight", "text", "BF16",
                (text["hidden_size"], linear_projection_width),
            )
    for block in range(vision["depth"]):
        prefix = f"model.visual.blocks.{block}"
        qkv_width = _checked_shape_mul(3, vision["hidden_size"], field="vision qkv width")
        for suffix, shape in (
            ("attn.proj.weight", (vision["hidden_size"], vision["hidden_size"])),
            ("attn.proj.bias", (vision["hidden_size"],)),
            ("attn.qkv.weight", (qkv_width, vision["hidden_size"])),
            ("attn.qkv.bias", (qkv_width,)),
            ("mlp.linear_fc1.weight", (vision["intermediate_size"], vision["hidden_size"])),
            ("mlp.linear_fc1.bias", (vision["intermediate_size"],)),
            ("mlp.linear_fc2.weight", (vision["hidden_size"], vision["intermediate_size"])),
            ("mlp.linear_fc2.bias", (vision["hidden_size"],)),
            ("norm1.weight", (vision["hidden_size"],)),
            ("norm1.bias", (vision["hidden_size"],)),
            ("norm2.weight", (vision["hidden_size"],)),
            ("norm2.bias", (vision["hidden_size"],)),
        ):
            add(f"{prefix}.{suffix}", "vision", "BF16", shape)
    spatial_merge_area = _checked_shape_mul(
        vision["spatial_merge_size"], vision["spatial_merge_size"],
        field="vision spatial merge area",
    )
    merged_width = _checked_shape_mul(
        vision["hidden_size"], spatial_merge_area, field="vision merged width"
    )
    for suffix, shape in (
        ("merger.linear_fc1.weight", (merged_width, merged_width)),
        ("merger.linear_fc1.bias", (merged_width,)),
        ("merger.linear_fc2.weight", (vision["out_hidden_size"], merged_width)),
        ("merger.linear_fc2.bias", (vision["out_hidden_size"],)),
        ("merger.norm.weight", (vision["hidden_size"],)),
        ("merger.norm.bias", (vision["hidden_size"],)),
        (
            "patch_embed.proj.weight",
            (
                vision["hidden_size"], vision["in_channels"], vision["temporal_patch_size"],
                vision["patch_size"], vision["patch_size"],
            ),
        ),
        ("patch_embed.proj.bias", (vision["hidden_size"],)),
        ("pos_embed.weight", (vision["num_position_embeddings"], vision["hidden_size"])),
    ):
        add(f"model.visual.{suffix}", "vision", "BF16", shape)

    if inputs.mtp_num_hidden_layers != 1 or inputs.mtp_use_dedicated_embeddings or not inputs.tie_word_embeddings:
        raise ContractError("Qwen MTP shape conditions are not satisfied")
    mtp_q_width = _checked_shape_mul(
        2,
        _checked_shape_mul(text["num_attention_heads"], text["head_dim"], field="MTP query width"),
        field="MTP query/gate width",
    )
    mtp_kv_width = _checked_shape_mul(
        text["num_key_value_heads"], text["head_dim"], field="MTP KV width"
    )
    mtp_output_width = _checked_shape_mul(
        text["num_attention_heads"], text["head_dim"], field="MTP output width"
    )
    for suffix, shape in (
        ("fc.weight", (text["hidden_size"], _checked_shape_mul(2, text["hidden_size"], field="MTP fc width"))),
        ("layers.0.input_layernorm.weight", (text["hidden_size"],)),
        ("layers.0.post_attention_layernorm.weight", (text["hidden_size"],)),
        ("layers.0.mlp.gate_proj.weight", (text["intermediate_size"], text["hidden_size"])),
        ("layers.0.mlp.up_proj.weight", (text["intermediate_size"], text["hidden_size"])),
        ("layers.0.mlp.down_proj.weight", (text["hidden_size"], text["intermediate_size"])),
        ("layers.0.self_attn.q_proj.weight", (mtp_q_width, text["hidden_size"])),
        ("layers.0.self_attn.k_proj.weight", (mtp_kv_width, text["hidden_size"])),
        ("layers.0.self_attn.v_proj.weight", (mtp_kv_width, text["hidden_size"])),
        ("layers.0.self_attn.o_proj.weight", (text["hidden_size"], mtp_output_width)),
        ("layers.0.self_attn.q_norm.weight", (text["head_dim"],)),
        ("layers.0.self_attn.k_norm.weight", (text["head_dim"],)),
        ("norm.weight", (text["hidden_size"],)),
        ("pre_fc_norm_embedding.weight", (text["hidden_size"],)),
        ("pre_fc_norm_hidden.weight", (text["hidden_size"],)),
    ):
        add(f"mtp.{suffix}", "mtp", "BF16", shape)
    if len(catalog) != 738:
        raise ContractError(f"internal Qwen tensor catalog cardinality drifted: {len(catalog)}")
    return catalog


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-standard JSON number is forbidden: {value}")


def _parse_json_bytes(
    raw: bytes,
    path: Path,
    *,
    max_bytes: int = MAX_LOCK_JSON_BYTES,
    purpose: str = "JSON document",
) -> Any:
    if len(raw) > max_bytes:
        raise ContractError(f"{purpose} exceeds the parser byte limit: {path}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=_reject_constant)
        _validate_json_bounds(value, purpose=purpose)
        return value
    except ContractError:
        raise
    except (OSError, RecursionError, UnicodeError, ValueError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def read_json(
    path: Path,
    *,
    max_bytes: int = MAX_LOCK_JSON_BYTES,
    purpose: str = "JSON document",
) -> Any:
    """Parse a bounded regular JSON file through one no-follow descriptor.

    The descriptor is opened before any path metadata inspection, read with
    ``pread`` rather than a shared seek cursor, and bound back to the path both
    before and after the read.  This is used for both lock and schema paths.
    """

    path = Path(path)
    descriptor = -1
    try:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        cloexec = getattr(os, "O_CLOEXEC", 0)
        if not nofollow or not cloexec:
            raise ContractError("bound JSON reads require O_NOFOLLOW and O_CLOEXEC")
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NONBLOCK | nofollow | cloexec,
        )
        fd_before = os.fstat(descriptor)
        if not stat.S_ISREG(fd_before.st_mode):
            raise ContractError(f"{purpose} must be a regular non-symlink file: {path}")
        if fd_before.st_size < 0 or fd_before.st_size > max_bytes:
            raise ContractError(f"{purpose} exceeds the parser byte limit: {path}")
        path_before = os.lstat(path)
        if stat.S_ISLNK(path_before.st_mode) or _stat_identity(fd_before) != _stat_identity(path_before):
            raise ContractError(f"{purpose} path changed while opening: {path}")
        raw = _read_exact(descriptor, fd_before.st_size, offset=0, max_bytes=max_bytes)
        if len(raw) != fd_before.st_size:
            raise ContractError(f"{purpose} is truncated while reading: {path}")
        fd_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if (
            not stat.S_ISREG(fd_after.st_mode)
            or stat.S_ISLNK(path_after.st_mode)
            or _stat_identity(fd_before) != _stat_identity(fd_after)
            or _stat_identity(fd_after) != _stat_identity(path_after)
        ):
            raise ContractError(f"{purpose} changed while reading: {path}")
        return _parse_json_bytes(raw, path, max_bytes=max_bytes, purpose=purpose)
    except ContractError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_json_bounds(value: Any, *, purpose: str) -> None:
    """Bound parsed JSON without relying on Python's recursion limit."""

    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise ContractError(f"{purpose} exceeds the nesting-depth limit")
        if isinstance(current, str):
            if len(current.encode("utf-8", errors="strict")) > MAX_JSON_STRING_BYTES:
                raise ContractError(f"{purpose} contains an oversized JSON string")
        elif isinstance(current, list):
            if len(current) > MAX_JSON_COLLECTION_ITEMS:
                raise ContractError(f"{purpose} exceeds the collection-item limit")
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, dict):
            if len(current) > MAX_JSON_COLLECTION_ITEMS:
                raise ContractError(f"{purpose} exceeds the collection-item limit")
            for key, item in current.items():
                if not isinstance(key, str) or len(key.encode("utf-8", errors="strict")) > MAX_JSON_STRING_BYTES:
                    raise ContractError(f"{purpose} contains an oversized JSON object key")
                stack.append((item, depth + 1))


def _reject_surrogates(value: str, *, where: str) -> None:
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise JCSValidationError(f"lone UTF-16 surrogate in {where}")


def _jcs_string(value: str) -> str:
    _reject_surrogates(value, where="string")
    if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
        raise JCSValidationError("JCS string exceeds the parser limit")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _jcs_key(value: str) -> bytes:
    _reject_surrogates(value, where="object key")
    return value.encode("utf-16be", errors="strict")


def _jcs_value(value: Any, *, depth: int = 0) -> str:
    if depth > MAX_JSON_DEPTH:
        raise JCSValidationError("JCS value exceeds the nesting-depth limit")
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if not JCS_SAFE_INTEGER_MIN <= value <= JCS_SAFE_INTEGER_MAX:
            raise JCSValidationError(
                "integer is outside the RFC 8785/I-JSON safe range "
                f"[{JCS_SAFE_INTEGER_MIN}, {JCS_SAFE_INTEGER_MAX}]"
            )
        return str(value)
    if isinstance(value, float) or isinstance(value, Decimal):
        raise JCSValidationError("floating-point values are forbidden in model-lock JCS")
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, list):
        if len(value) > MAX_JSON_COLLECTION_ITEMS:
            raise JCSValidationError("JCS array exceeds the collection-item limit")
        return "[" + ",".join(_jcs_value(item, depth=depth + 1) for item in value) + "]"
    if isinstance(value, dict):
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise JCSValidationError("JSON object keys must be strings")
        if len(keys) > MAX_JSON_COLLECTION_ITEMS:
            raise JCSValidationError("JCS object exceeds the collection-item limit")
        ordered = sorted(keys, key=_jcs_key)
        return "{" + ",".join(
            _jcs_string(key) + ":" + _jcs_value(value[key], depth=depth + 1)
            for key in ordered
        ) + "}"
    raise JCSValidationError(f"unsupported JSON value in JCS: {type(value).__name__}")


def jcs_dumps(value: Any) -> str:
    """Return compact RFC 8785-style JSON for the restricted lock domain."""

    try:
        return _jcs_value(value)
    except RecursionError as exc:
        raise JCSValidationError("JCS recursion limit exceeded") from exc


def fingerprint_for_document(document: dict[str, Any]) -> str:
    """Hash only the schema version and complete model object."""

    target = {"schema_version": document["schema_version"], "model": document["model"]}
    canonical = jcs_dumps(target).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _validate_schema(document: dict[str, Any], schema_path: Path) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise ContractError("jsonschema is required for model lock validation") from exc
    try:
        schema = read_json(
            schema_path,
            max_bytes=MAX_LOCK_JSON_BYTES,
            purpose="model-lock schema",
        )
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
            key=lambda error: list(error.path),
        )
    except RecursionError as exc:
        raise ContractError("model lock schema validation exceeded the recursion limit") from exc
    if errors:
        detail = "; ".join(error.message for error in errors[:8])
        raise ContractError(f"model lock schema validation failed: {detail}")


def _safe_relative_path(value: str, *, name: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or _contains_forbidden_control(value)
        or "\\" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ContractError(f"unsafe {name}: {value!r}")


def _contains_forbidden_control(value: str) -> bool:
    return any(
        ord(character) <= 0x1F or 0x7F <= ord(character) <= 0x9F
        for character in value
    )


def _validate_lock_text(value: Any, *, field: str, max_bytes: int = MAX_JSON_STRING_BYTES) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8", errors="strict")) > max_bytes
        or _contains_forbidden_control(value)
    ):
        raise ContractError(f"invalid {field}: empty, oversized, or contains a control character")


def _validate_lock_strings(value: Any) -> None:
    """Reject control bytes from every model-lock string and key iteratively."""

    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise ContractError("model lock exceeds the nesting-depth limit")
        if isinstance(current, str):
            _validate_lock_text(current, field="model-lock string")
        elif isinstance(current, list):
            if len(current) > MAX_JSON_COLLECTION_ITEMS:
                raise ContractError("model lock exceeds the collection-item limit")
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, dict):
            if len(current) > MAX_JSON_COLLECTION_ITEMS:
                raise ContractError("model lock exceeds the collection-item limit")
            for key, item in current.items():
                _validate_lock_text(key, field="model-lock object key")
                stack.append((item, depth + 1))


def _validate_generated_at(value: Any) -> None:
    if not isinstance(value, str):
        raise ContractError("generated_at must be a canonical UTC RFC3339 string")
    match = UTC_RFC3339_Z.fullmatch(value)
    if match is None:
        raise ContractError("generated_at must use canonical UTC RFC3339 Z form")
    try:
        dt.datetime(*(int(component) for component in match.groups()), tzinfo=dt.timezone.utc)
    except ValueError as exc:
        raise ContractError("generated_at is not a real UTC calendar date-time") from exc


def _exact_url(url: str, *, repo_id: str, revision: str, kind: str, path: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.query or parsed.fragment or parsed.username:
        raise ContractError(f"{kind} is not an immutable Hugging Face HTTPS URL: {url}")
    expected = f"/{repo_id}/{'blob' if kind == 'source_page_url' else 'resolve'}/{revision}/{path}"
    if parsed.path != expected:
        raise ContractError(f"{kind} does not bind {path} to resolved revision {revision}: {url}")


def _validate_qwen_contract(document: dict[str, Any]) -> None:
    model = document["model"]
    if QWEN_ALIAS not in document["aliases"] and model["repo_id"] != REPO_ID:
        return
    if model["repo_id"] != REPO_ID or model["resolved_revision"] != RESOLVED_REVISION:
        raise ContractError("Qwen alias/repository is not bound to the reviewed immutable revision")
    if model["requested_revision"] != "main":
        raise ContractError("Qwen lock requested_revision must preserve the reviewed main resolution")
    if document["aliases"] != [QWEN_ALIAS]:
        raise ContractError("Qwen lock alias binding is not exact")
    if model["license"]["id"] != "Apache-2.0" or set(model["license"]["evidence_paths"]) != {"LICENSE", "README.md"}:
        raise ContractError("Qwen lock license evidence is incomplete")
    if model["base_models"] != [{"repo_id": "Qwen/Qwen3.5-4B-Base", "revision": None, "evidence_path": "README.md"}]:
        raise ContractError("Qwen base-model declaration drifted")
    files = model["files"]
    by_path: dict[str, dict[str, Any]] = {}
    for entry in files:
        path = entry["path"]
        if path in by_path:
            raise ContractError(f"duplicate locked file: {path}")
        by_path[path] = entry
    if set(by_path) != set(QWEN_FILES):
        missing = sorted(set(QWEN_FILES) - set(by_path))
        extra = sorted(set(by_path) - set(QWEN_FILES))
        raise ContractError(f"Qwen file set mismatch; missing={missing}, extra={extra}")
    if [entry["path"] for entry in files] != sorted(QWEN_FILES):
        raise ContractError("Qwen lock files must be in deterministic path order")
    for path, expected in QWEN_FILES.items():
        entry = by_path[path]
        for field in ("size_bytes", "sha256", "git_blob", "lfs_oid"):
            if entry[field] != expected[field]:
                raise ContractError(f"Qwen {path} {field} does not match reviewed metadata")
        _exact_url(entry["source_page_url"], repo_id=REPO_ID, revision=RESOLVED_REVISION, kind="source_page_url", path=path)
        _exact_url(entry["download_url"], repo_id=REPO_ID, revision=RESOLVED_REVISION, kind="download_url", path=path)
    if model["evidence_files"] != ["LICENSE", "README.md"]:
        raise ContractError("Qwen evidence file binding drifted")
    if model["generation_config"] != {"present": False, "path": None}:
        raise ContractError("Qwen generation_config must be explicitly absent")

    architecture = model["architecture"]
    if architecture["architectures"] != ["Qwen3_5ForConditionalGeneration"] or architecture["top_level_architecture"] != "Qwen3_5ForConditionalGeneration":
        raise ContractError("Qwen top-level architecture mismatch")
    if architecture["model_type"] != "qwen3_5" or architecture["text_model_type"] != "qwen3_5_text":
        raise ContractError("Qwen model_type/text model_type mismatch")
    if architecture["custom_code"] or architecture["converted"] or architecture["moe"]:
        raise ContractError("Qwen lock permits custom code, conversion, or MoE unexpectedly")
    text = architecture["text_config"]
    expected_layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
    expected_text = {
        "model_type": "qwen3_5_text", "hidden_size": 2560, "num_hidden_layers": 32,
        "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
        "intermediate_size": 9216, "dtype": "BF16", "rms_norm_eps": "1e-6",
        "full_attention_interval": 4, "tie_word_embeddings": True, "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
    }
    if text != {**expected_text, "layer_types": expected_layer_types}:
        raise ContractError("Qwen text config contract does not match the resolved config")
    schedule = architecture["layer_schedule"]
    if schedule != {
        "kind": "explicit", "num_hidden_layers": 32, "full_attention_interval": 4,
        "layer_types": expected_layer_types, "allowed_types": ["linear_attention", "full_attention"],
    }:
        raise ContractError("Qwen layer schedule contract drifted")
    for component, prefix, count in (("vision", "model.visual.", 297), ("mtp", "mtp.", 15)):
        value = architecture[component]
        if value != {"present": True, "tensor_prefix": prefix, "tensor_count": count, "phase3_status": "known-unconsumed"}:
            raise ContractError(f"Qwen {component} component classification drifted")

    tensor = model["tensor_contract"]
    if tensor["index_path"] != "model.safetensors.index.json" or tensor["indexed_tensor_count"] != 738:
        raise ContractError("Qwen safetensors index contract drifted")
    if tensor["shards"] != sorted(["model.safetensors-00001-of-00002.safetensors", "model.safetensors-00002-of-00002.safetensors"]):
        raise ContractError("Qwen shard contract drifted")
    expected_classes = [
        {"id": "text", "prefix": "model.language_model.", "tensor_count": 426, "phase3_status": "partially-consumed"},
        {"id": "vision", "prefix": "model.visual.", "tensor_count": 297, "phase3_status": "known-unconsumed"},
        {"id": "mtp", "prefix": "mtp.", "tensor_count": 15, "phase3_status": "known-unconsumed"},
    ]
    if tensor["classifications"] != expected_classes:
        raise ContractError("Qwen tensor classification drifted")

    slice_contract = model["slice_contract"]
    if slice_contract != {
        "tensor_name": "model.language_model.layers.0.input_layernorm.weight",
        "source_file": "model.safetensors-00002-of-00002.safetensors",
        "dtype": "BF16", "shape": [2560], "header_length_field_bytes": 8,
        "header_length_bytes": 79064, "data_buffer_start": 79072,
        "data_offset_basis": "data-buffer-relative",
        "data_offsets": [15360, 20480], "absolute_byte_range": [94432, 99552], "byte_size": 5120,
        "normalization": {
            "kind": "rmsnorm", "scale_mode": "offset-one", "effective_scale": "1 + raw_weight",
            "epsilon": "1e-6", "activation_dtype": "BF16", "weight_dtype": "BF16",
            "accumulation_dtype": "FP32", "output_dtype": "BF16",
        },
    }:
        raise ContractError("Qwen G2 slice/RMSNorm contract drifted")
    tokenizer = model["tokenizer_contract"]
    if tokenizer["vocab_size"] != 248320 or tokenizer["eos_token_id"] != 248044 or tokenizer["chat_template_path"] != "chat_template.jinja":
        raise ContractError("Qwen tokenizer contract drifted")
    if tokenizer["stop_identity"] != {
        "config_eos": {
            "token": "<|endoftext|>", "token_id": 248044, "source_file": "config.json",
        },
        "tokenizer_eos": {
            "token": "<|im_end|>", "token_id": 248046,
            "source_files": ["tokenizer_config.json", "tokenizer.json"],
        },
    }:
        raise ContractError("Qwen stop identity contract drifted")
    expected_stop_policy = {
        "version": 1,
        "stop_token_ids": [248046, 248044],
        "evaluation": "newly_generated_after_argmax",
        "prompt_evaluation": "never_stop",
        "stop_token": {
            "visible_output": False,
            "subsequent_decode_input": False,
        },
        "budget_boundary": "stop_token_wins",
        "max_new_tokens_zero": "max_new_tokens_before_decode",
        "reason_version": 1,
    }
    if not _fixed_json_value_matches(tokenizer["generation_stop_policy"], expected_stop_policy):
        raise ContractError("Qwen generation stop policy contract drifted")
    expected_special = {
        "vision_start": 248053, "vision_end": 248054, "vision_pad": 248055,
        "image_pad": 248056, "video_pad": 248057,
    }
    if tokenizer["special_token_ids"] != expected_special:
        raise ContractError("Qwen special-token contract drifted")


def _validate_generation_stop_policy(policy: Any) -> None:
    """Validate the fixed v1 policy with Python's strict JSON type semantics."""

    required = {
        "version", "stop_token_ids", "evaluation", "prompt_evaluation", "stop_token",
        "budget_boundary", "max_new_tokens_zero", "reason_version",
    }
    if type(policy) is not dict or policy.keys() != required:
        raise ContractError("generation stop policy fields are missing or unknown")
    if type(policy["version"]) is not int or policy["version"] != 1:
        raise ContractError("generation stop policy version must be exactly 1")
    if type(policy["reason_version"]) is not int or policy["reason_version"] != 1:
        raise ContractError("generation stop policy reason_version must be exactly 1")
    token_ids = policy["stop_token_ids"]
    if type(token_ids) is not list or not token_ids:
        raise ContractError("generation stop policy stop_token_ids must be non-empty")
    if any(type(token_id) is not int or not 0 <= token_id <= UINT32_MAX for token_id in token_ids):
        raise ContractError("generation stop policy stop_token_ids must be ordered u32 integers")
    if len(set(token_ids)) != len(token_ids):
        raise ContractError("generation stop policy stop_token_ids must be unique")
    for field, expected in (
        ("evaluation", "newly_generated_after_argmax"),
        ("prompt_evaluation", "never_stop"),
        ("budget_boundary", "stop_token_wins"),
        ("max_new_tokens_zero", "max_new_tokens_before_decode"),
    ):
        if type(policy[field]) is not str or policy[field] != expected:
            raise ContractError(f"generation stop policy {field} is not the exact v1 enum")
    stop_token = policy["stop_token"]
    if type(stop_token) is not dict or stop_token.keys() != {"visible_output", "subsequent_decode_input"}:
        raise ContractError("generation stop policy stop_token fields are missing or unknown")
    if (
        type(stop_token["visible_output"]) is not bool
        or stop_token["visible_output"] is not False
        or type(stop_token["subsequent_decode_input"]) is not bool
        or stop_token["subsequent_decode_input"] is not False
    ):
        raise ContractError("generation stop policy stop-token handling must be false/false")


def _validate_generic_contract(document: dict[str, Any]) -> None:
    model = document["model"]
    _validate_generation_stop_policy(model["tokenizer_contract"]["generation_stop_policy"])
    paths = {entry["path"] for entry in model["files"]}
    if len(paths) != len(model["files"]):
        raise ContractError("duplicate file path in model lock")
    if not set(model["evidence_files"]).issubset(paths):
        raise ContractError("evidence file is not present in files")
    if not set(model["license"]["evidence_paths"]).issubset(paths):
        raise ContractError("license evidence path is not present in files")
    for base_model in model["base_models"]:
        if base_model["evidence_path"] not in paths:
            raise ContractError("base-model evidence path is not present in files")
    if model["tensor_contract"]["index_path"] not in paths:
        raise ContractError("tensor index path is not present in files")
    if not set(model["tensor_contract"]["shards"]).issubset(paths):
        raise ContractError("tensor shard is not present in files")
    if model["slice_contract"]["source_file"] not in paths:
        raise ContractError("slice source is not present in files")
    if not set(model["tokenizer_contract"]["files"]).issubset(paths):
        raise ContractError("tokenizer contract references an unlocked file")
    if model["tokenizer_contract"]["chat_template_path"] not in paths:
        raise ContractError("chat template is not present in files")
    stop_identity = model["tokenizer_contract"]["stop_identity"]
    if stop_identity["config_eos"]["source_file"] not in paths:
        raise ContractError("config EOS source is not present in files")
    if not set(stop_identity["tokenizer_eos"]["source_files"]).issubset(paths):
        raise ContractError("tokenizer EOS source is not present in files")
    if any(entry["path"] in paths for entry in model["excluded_files"]):
        raise ContractError("excluded repository metadata is also locked as a runtime file")


def validate_document(document: dict[str, Any], *, schema_path: Path = SCHEMA_PATH) -> None:
    if not isinstance(document, dict):
        raise ContractError("model lock root must be an object")
    schema_path = Path(schema_path)
    _validate_schema(document, schema_path)
    _validate_json_bounds(document, purpose="model lock")
    _validate_lock_strings(document)
    _validate_generated_at(document["generated_at"])
    try:
        if fingerprint_for_document(document) != document["fingerprint"]:
            raise ContractError("model lock fingerprint does not match RFC 8785 JCS input")
    except (KeyError, JCSValidationError) as exc:
        raise ContractError(f"cannot compute model lock fingerprint: {exc}") from exc
    model = document["model"]
    if not SHA40.fullmatch(model["resolved_revision"]):
        raise ContractError("resolved_revision is not a lowercase full SHA-1")
    if len(set(document["aliases"])) != len(document["aliases"]):
        raise ContractError("duplicate model alias")
    _validate_generic_contract(document)
    for entry in model["files"]:
        _safe_relative_path(entry["path"], name="locked file path")
        if entry["lfs_oid"] is not None and entry["lfs_oid"] != "sha256:" + entry["sha256"]:
            raise ContractError(f"LFS OID does not equal the locked byte SHA-256: {entry['path']}")
        _exact_url(entry["source_page_url"], repo_id=model["repo_id"], revision=model["resolved_revision"], kind="source_page_url", path=entry["path"])
        _exact_url(entry["download_url"], repo_id=model["repo_id"], revision=model["resolved_revision"], kind="download_url", path=entry["path"])
    for entry in model["excluded_files"]:
        _safe_relative_path(entry["path"], name="excluded file path")
    _validate_qwen_contract(document)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _mount_is_read_only(path: Path) -> bool:
    flags = os.statvfs(path).f_flag
    return bool(flags & getattr(os, "ST_RDONLY", 1))


def _cache_root_stat(cache_dir: Path, *, require_trusted_read_only: bool) -> os.stat_result:
    try:
        root_stat = os.lstat(cache_dir)
    except OSError as exc:
        raise ContractError(f"cannot inspect cache root: {cache_dir}: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ContractError(f"cache directory must be an existing non-symlink directory: {cache_dir}")
    if require_trusted_read_only:
        mode = stat.S_IMODE(root_stat.st_mode)
        mount_read_only = _mount_is_read_only(cache_dir)
        if root_stat.st_mode & stat.S_IWUSR | root_stat.st_mode & stat.S_IWGRP | root_stat.st_mode & stat.S_IWOTH or not mount_read_only:
            raise ContractError(
                "trusted read-only cache rejected: "
                f"root_mode={mode:04o} mount_read_only={mount_read_only}"
            )
    return root_stat


def _cache_entries(cache_dir: Path, *, require_trusted_read_only: bool = False) -> dict[str, Path]:
    _cache_root_stat(cache_dir, require_trusted_read_only=require_trusted_read_only)
    result: dict[str, Path] = {}
    for path in cache_dir.rglob("*"):
        relative = path.relative_to(cache_dir).as_posix()
        path_stat = os.lstat(path)
        if stat.S_ISLNK(path_stat.st_mode):
            raise ContractError(f"cache contains a symlink: {relative}")
        if stat.S_ISDIR(path_stat.st_mode):
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise ContractError(f"cache entry is not a regular file: {relative}")
        if path_stat.st_nlink != 1:
            raise ContractError(f"cache entry has a hardlink and is rejected: {relative}")
        if require_trusted_read_only:
            mode = stat.S_IMODE(path_stat.st_mode)
            mount_read_only = _mount_is_read_only(path)
            if path_stat.st_mode & stat.S_IWUSR | path_stat.st_mode & stat.S_IWGRP | path_stat.st_mode & stat.S_IWOTH or path_stat.st_nlink != 1 or not mount_read_only:
                raise ContractError(
                    "trusted read-only cache rejected: "
                    f"file={relative} mode={mode:04o} nlink={path_stat.st_nlink} "
                    f"mount_read_only={mount_read_only}"
                )
        result[relative] = path
    return result


def _open_relative_read_only(cache_dir: Path, relative: str) -> int:
    """Open a cache file through O_NOFOLLOW directory handles."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | nofollow | cloexec | getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY | nofollow | cloexec
    components = relative.split("/")
    if not components or any(not component or component in {".", ".."} for component in components):
        raise ContractError(f"unsafe cache relative path: {relative}")
    root_fd = os.open(cache_dir, directory_flags)
    current_fd = root_fd
    try:
        for component in components[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return os.open(components[-1], file_flags, dir_fd=current_fd)
    except OSError as exc:
        raise ContractError(f"cannot open cache file without following links: {relative}: {exc}") from exc
    finally:
        os.close(current_fd)


def _sha256_fd(file_descriptor: int, size_bytes: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size_bytes:
        chunk = os.pread(file_descriptor, min(1024 * 1024, size_bytes - offset), offset)
        if not chunk:
            raise ContractError("cache file ended while hashing its verified descriptor")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _verify_cache_file(
    cache_dir: Path,
    relative: str,
    expected: dict[str, Any],
    *,
    require_trusted_read_only: bool,
) -> _VerifiedCacheFile:
    path = cache_dir / relative
    file_descriptor = -1
    try:
        path_before = os.lstat(path)
        file_descriptor = _open_relative_read_only(cache_dir, relative)
        fd_before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(fd_before.st_mode)
            or fd_before.st_nlink != 1
            or _stat_identity(path_before) != _stat_identity(fd_before)
        ):
            raise ContractError(f"cache path changed while opening {relative}")
        if require_trusted_read_only:
            mode = stat.S_IMODE(fd_before.st_mode)
            mount_read_only = _mount_is_read_only(path)
            if fd_before.st_mode & stat.S_IWUSR | fd_before.st_mode & stat.S_IWGRP | fd_before.st_mode & stat.S_IWOTH or fd_before.st_nlink != 1 or not mount_read_only:
                raise ContractError(
                    "trusted read-only cache rejected: "
                    f"file={relative} mode={mode:04o} nlink={fd_before.st_nlink} "
                    f"mount_read_only={mount_read_only}"
                )
        digest = _sha256_fd(file_descriptor, fd_before.st_size)
        fd_after = os.fstat(file_descriptor)
        path_after = os.lstat(path)
        if _stat_identity(fd_before) != _stat_identity(fd_after) or _stat_identity(fd_after) != _stat_identity(path_after):
            raise ContractError(f"cache file changed during verification: {relative}")
        if fd_after.st_size != expected["size_bytes"]:
            raise ContractError(f"cache size mismatch for {relative}: {fd_after.st_size} != {expected['size_bytes']}")
        if digest != expected["sha256"]:
            raise ContractError(f"cache SHA-256 mismatch for {relative}")
        if expected["lfs_oid"] is not None and expected["lfs_oid"] != "sha256:" + digest:
            raise ContractError(f"cache LFS OID mismatch for {relative}")
        verified = _VerifiedCacheFile(
            relative=relative,
            file_descriptor=file_descriptor,
            identity=_stat_identity(fd_after),
            size_bytes=fd_after.st_size,
            sha256=digest,
        )
        file_descriptor = -1
        return verified
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot verify cache file {relative}: {exc}") from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _verified_file(files: dict[str, _VerifiedCacheFile], relative: str) -> _VerifiedCacheFile:
    try:
        verified = files[relative]
    except KeyError as exc:
        raise ContractError(f"semantic validation referenced an unhashed cache file: {relative}") from exc
    return verified


def _assert_verified_fd_stable(verified: _VerifiedCacheFile, *, operation: str) -> os.stat_result:
    try:
        current = os.fstat(verified.file_descriptor)
    except OSError as exc:
        raise ContractError(f"cannot inspect hashed cache descriptor for {verified.relative}: {exc}") from exc
    if current.st_nlink != 1 or _stat_identity(current) != verified.identity:
        raise ContractError(f"hashed cache file changed during {operation}: {verified.relative}")
    return current


def _json_read_limit(relative: str) -> int:
    if relative.endswith("tokenizer.json"):
        return MAX_TOKENIZER_JSON_BYTES
    if relative.endswith("safetensors.index.json"):
        return MAX_INDEX_JSON_BYTES
    return MAX_CONFIG_JSON_BYTES


def _read_cache_json(files: dict[str, _VerifiedCacheFile], relative: str) -> Any:
    """Read JSON from the descriptor whose bytes were already hash-verified."""

    verified = _verified_file(files, relative)
    before = _assert_verified_fd_stable(verified, operation="JSON validation")
    limit = _json_read_limit(relative)
    raw = _read_exact(verified.file_descriptor, before.st_size, offset=0, max_bytes=limit)
    if len(raw) != before.st_size:
        raise ContractError(f"cache JSON is truncated: {relative}")
    _assert_verified_fd_stable(verified, operation="JSON validation")
    return _parse_json_bytes(raw, Path(relative), max_bytes=limit, purpose=f"cache JSON {relative}")


def _unique_pairs(items: list[tuple[str, Any]], source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key in safetensors header {source}: {key}")
        result[key] = value
    return result


def _read_exact(file_descriptor: int, size: int, *, offset: int, max_bytes: int) -> bytes:
    if size < 0 or size > max_bytes:
        raise ContractError("bounded descriptor read exceeds its byte limit")
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        chunk = os.pread(file_descriptor, min(1024 * 1024, remaining), position)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
        position += len(chunk)
    return b"".join(chunks)


def _checked_u64(value: Any, *, field: str, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > UINT64_MAX:
        raise ContractError(f"{field} is outside unsigned 64-bit range: {source}")
    return value


def _checked_add(left: int, right: int, *, field: str, source: str) -> int:
    if left > UINT64_MAX - right:
        raise ContractError(f"{field} overflows unsigned 64-bit range: {source}")
    return left + right


def _checked_mul(left: int, right: int, *, field: str, source: str) -> int:
    if right and left > UINT64_MAX // right:
        raise ContractError(f"{field} overflows unsigned 64-bit range: {source}")
    return left * right


def _read_safetensors_header(files: dict[str, _VerifiedCacheFile], shard_name: str) -> tuple[int, bytes, int]:
    verified = _verified_file(files, shard_name)
    before = _assert_verified_fd_stable(verified, operation="safetensors header validation")
    raw_length = _read_exact(verified.file_descriptor, 8, offset=0, max_bytes=8)
    if len(raw_length) != 8:
        raise ContractError(f"safetensors header length is truncated: {shard_name}")
    header_length = struct.unpack("<Q", raw_length)[0]
    if header_length < 8 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise ContractError(f"invalid safetensors header length: {shard_name}")
    raw_header = _read_exact(
        verified.file_descriptor,
        header_length,
        offset=8,
        max_bytes=MAX_SAFETENSORS_HEADER_BYTES,
    )
    if len(raw_header) != header_length:
        raise ContractError(f"safetensors header is truncated: {shard_name}")
    after = _assert_verified_fd_stable(verified, operation="safetensors header validation")
    if _stat_identity(before) != _stat_identity(after):
        raise ContractError(f"safetensors shard changed during header validation: {shard_name}")
    return header_length, raw_header, after.st_size


def _validate_safetensors_headers(
    files: dict[str, _VerifiedCacheFile],
    model: dict[str, Any],
    qwen_shape_inputs: _QwenShapeInputs | None = None,
) -> None:
    """Validate index/header metadata without loading tensor data into memory."""

    index = _read_cache_json(files, model["tensor_contract"]["index_path"])
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"} or not isinstance(index["metadata"], dict) or not isinstance(index["weight_map"], dict):
        raise ContractError("safetensors index has unknown or missing keys")
    if set(index["metadata"]) != {"total_size"} or not isinstance(index["metadata"]["total_size"], int) or isinstance(index["metadata"]["total_size"], bool) or index["metadata"]["total_size"] < 0 or index["metadata"]["total_size"] > UINT64_MAX:
        raise ContractError("safetensors index metadata must contain one non-negative integer total_size")
    weight_map = index["weight_map"]
    if len(weight_map) != model["tensor_contract"]["indexed_tensor_count"]:
        raise ContractError("safetensors index tensor count does not match lock")
    if not all(isinstance(name, str) and name for name in weight_map) or not all(isinstance(shard, str) and shard for shard in weight_map.values()):
        raise ContractError("safetensors index weight_map must map tensor names to shard names")
    if set(weight_map.values()) != set(model["tensor_contract"]["shards"]):
        raise ContractError("safetensors index shard set does not match lock")
    all_headers: dict[str, dict[str, Any]] = {}
    total_tensor_bytes = 0
    for shard_name in model["tensor_contract"]["shards"]:
        header_length, raw_header, shard_size = _read_safetensors_header(files, shard_name)
        try:
            header = _parse_json_bytes(
                raw_header,
                Path(shard_name),
                max_bytes=MAX_SAFETENSORS_HEADER_BYTES,
                purpose=f"safetensors header {shard_name}",
            )
        except ContractError as exc:
            raise ContractError(f"invalid safetensors header JSON: {shard_name}: {exc}") from exc
        if not isinstance(header, dict) or ("__metadata__" in header and not isinstance(header["__metadata__"], dict)):
            raise ContractError(f"invalid safetensors header object: {shard_name}")
        metadata = header.get("__metadata__", {})
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
            raise ContractError(f"safetensors __metadata__ values must be strings: {shard_name}")
        data_buffer_start = _checked_add(8, header_length, field="safetensors data buffer start", source=shard_name)
        shard_size = _checked_u64(shard_size, field="safetensors shard size", source=shard_name)
        if data_buffer_start > shard_size:
            raise ContractError(f"safetensors data buffer starts beyond shard: {shard_name}")
        spans: list[tuple[int, int, str]] = []
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not name or not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
                raise ContractError(f"invalid tensor metadata: {shard_name}:{name}")
            dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
            if not isinstance(dtype, str) or dtype not in SAFETENSORS_DTYPE_BYTES or not isinstance(shape, list) or not shape:
                raise ContractError(f"invalid dtype/shape metadata: {shard_name}:{name}")
            try:
                shape = [_checked_u64(item, field="safetensors shape dimension", source=f"{shard_name}:{name}") for item in shape]
                offsets = [_checked_u64(item, field="safetensors data offset", source=f"{shard_name}:{name}") for item in offsets]
            except TypeError as exc:
                raise ContractError(f"invalid dtype/shape metadata: {shard_name}:{name}") from exc
            if not isinstance(offsets, list) or len(offsets) != 2 or offsets[0] >= offsets[1]:
                raise ContractError(f"invalid data offsets: {shard_name}:{name}")
            element_count = 1
            for dimension in shape:
                element_count = _checked_mul(element_count, dimension, field="safetensors element count", source=f"{shard_name}:{name}")
            expected_bytes = _checked_mul(element_count, SAFETENSORS_DTYPE_BYTES[dtype], field="safetensors tensor byte size", source=f"{shard_name}:{name}")
            if offsets[1] - offsets[0] != expected_bytes:
                raise ContractError(f"tensor dtype/shape byte size mismatch: {shard_name}:{name}")
            absolute_start = _checked_add(data_buffer_start, offsets[0], field="safetensors tensor start", source=f"{shard_name}:{name}")
            absolute_end = _checked_add(data_buffer_start, offsets[1], field="safetensors tensor end", source=f"{shard_name}:{name}")
            if absolute_end > shard_size:
                raise ContractError(f"tensor data is outside shard: {shard_name}:{name}")
            if name in all_headers:
                raise ContractError(f"duplicate tensor in safetensors shards: {name}")
            all_headers[name] = {
                "shard": shard_name,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets,
                "absolute_byte_range": [absolute_start, absolute_end],
                "byte_size": expected_bytes,
                "header_length_field_bytes": 8,
                "header_length_bytes": header_length,
                "data_buffer_start": data_buffer_start,
                "data_offset_basis": "data-buffer-relative",
            }
            total_tensor_bytes = _checked_add(
                total_tensor_bytes,
                expected_bytes,
                field="safetensors total tensor byte size",
                source=shard_name,
            )
            spans.append((offsets[0], offsets[1], name))
        spans.sort()
        payload_size = shard_size - data_buffer_start
        cursor = 0
        for start, end, name in spans:
            if start != cursor:
                if start < cursor:
                    raise ContractError(f"overlapping tensor ranges in shard: {shard_name}")
                raise ContractError(f"gap in safetensors tensor ranges in shard: {shard_name}:{name}")
            cursor = end
        if cursor != payload_size:
            raise ContractError(f"uncovered safetensors payload bytes in shard: {shard_name}")
    if index["metadata"]["total_size"] != total_tensor_bytes:
        raise ContractError("safetensors index total_size does not match header tensor payload bytes")
    if set(all_headers) != set(weight_map) or any(all_headers[name]["shard"] != weight_map[name] for name in weight_map):
        raise ContractError("safetensors index and shard tensor names do not match exactly")
    classifications = model["tensor_contract"]["classifications"]
    classification_ids = [item["id"] for item in classifications]
    if len(set(classification_ids)) != len(classification_ids):
        raise ContractError("safetensors tensor classifications contain duplicate IDs")
    counts = {
        item["id"]: sum(name.startswith(item["prefix"]) for name in all_headers)
        for item in classifications
    }
    expected = {item["id"]: item["tensor_count"] for item in classifications}
    if sum(counts.values()) != len(all_headers) or counts != expected:
        raise ContractError(f"safetensors tensor classification mismatch: {counts} != {expected}")
    if model["repo_id"] == REPO_ID:
        if qwen_shape_inputs is None:
            raise ContractError("Qwen safetensors validation lacks parsed config shapes")
        catalog = _qwen_tensor_catalog(qwen_shape_inputs)
        if set(all_headers) != set(catalog):
            missing = sorted(set(catalog) - set(all_headers))
            extra = sorted(set(all_headers) - set(catalog))
            raise ContractError(
                "Qwen safetensors names do not match the reviewed exact catalog; "
                f"missing={missing[:4]} extra={extra[:4]}"
            )
        for name, header in all_headers.items():
            expected_class, expected_dtype, expected_shape = catalog[name]
            actual_class = next(
                (
                    item["id"]
                    for item in classifications
                    if name.startswith(item["prefix"])
                ),
                None,
            )
            if actual_class != expected_class:
                raise ContractError(f"Qwen tensor class differs from the reviewed catalog: {name}")
            if header["dtype"] != expected_dtype:
                raise ContractError(f"Qwen tensor dtype differs from the reviewed catalog: {name}")
            if tuple(header["shape"]) != expected_shape:
                raise ContractError(f"Qwen tensor shape differs from the reviewed catalog: {name}")
    slice_contract = model["slice_contract"]
    slice_header = all_headers.get(slice_contract["tensor_name"])
    if slice_header is None:
        raise ContractError("locked slice tensor is absent from safetensors headers")
    for field in (
        "dtype", "shape", "data_offsets", "absolute_byte_range", "byte_size",
        "header_length_field_bytes", "header_length_bytes", "data_buffer_start", "data_offset_basis",
    ):
        if slice_header[field] != slice_contract[field]:
            raise ContractError(f"locked slice does not match safetensors header: {field}")
    if slice_header["shard"] != slice_contract["source_file"]:
        raise ContractError("locked slice source shard does not match safetensors header")


def _validate_qwen_config(
    files: dict[str, _VerifiedCacheFile],
    model: dict[str, Any],
) -> _QwenShapeInputs:
    config = _read_cache_json(files, "config.json")
    if not isinstance(config, dict):
        raise ContractError("Qwen config must be a JSON object")
    if set(config) != QWEN_CONFIG_ROOT_FIELDS:
        raise ContractError("Qwen config root fields differ from the reviewed revision")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ContractError("Qwen config has no text_config object")
    architecture = model["architecture"]
    if (
        not _fixed_json_value_matches(config.get("architectures"), architecture["architectures"])
        or not _fixed_json_value_matches(config.get("model_type"), architecture["model_type"])
    ):
        raise ContractError("Qwen config architecture/model_type does not match lock")
    if (
        not _fixed_json_value_matches(config.get("tie_word_embeddings"), True)
        or not _fixed_json_value_matches(config.get("transformers_version"), "4.57.0.dev0")
    ):
        raise ContractError("Qwen config root tie/version fields differ from the reviewed revision")
    actual_token_ids = {
        "image_token_id": config.get("image_token_id"),
        "video_token_id": config.get("video_token_id"),
        "vision_start_token_id": config.get("vision_start_token_id"),
        "vision_end_token_id": config.get("vision_end_token_id"),
    }
    if not _fixed_json_value_matches(actual_token_ids, {
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
    }):
        raise ContractError("Qwen config vision token IDs differ from the reviewed revision")
    if not _fixed_json_value_matches(config.get("vision_config"), QWEN_VISION_CONFIG):
        raise ContractError("Qwen vision config differs from the reviewed revision")
    locked_text = architecture["text_config"]
    expected_text_fields = set(locked_text) | set(QWEN_TEXT_OPTIONAL_CONFIG) | {"eos_token_id", "rope_parameters"}
    if set(text) != expected_text_fields:
        raise ContractError("Qwen text config fields differ from the reviewed revision")
    for key in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "full_attention_interval", "layer_types", "tie_word_embeddings", "vocab_size", "mtp_num_hidden_layers"):
        if not _fixed_json_value_matches(text.get(key), locked_text[key]):
            raise ContractError(f"Qwen text config field differs from lock: {key}")
    layer_types = text.get("layer_types")
    expected_layers = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
    if not _fixed_json_value_matches(layer_types, expected_layers):
        raise ContractError("Qwen text config layer_types differs from the exact 32-layer schedule")
    for key, expected in QWEN_TEXT_OPTIONAL_CONFIG.items():
        if not _fixed_json_value_matches(text.get(key), expected):
            raise ContractError(f"Qwen optional text config field differs from the reviewed revision: {key}")
    if not _fixed_json_value_matches(text.get("rope_parameters"), QWEN_TEXT_ROPE_PARAMETERS):
        raise ContractError("Qwen text rope parameters differ from the reviewed revision")
    if not _fixed_json_value_matches(
        text.get("eos_token_id"),
        model["tokenizer_contract"]["stop_identity"]["config_eos"]["token_id"],
    ):
        raise ContractError("Qwen config EOS ID differs from the locked stop identity")
    if not _fixed_json_value_matches(text.get("dtype"), "bfloat16"):
        raise ContractError("Qwen text config dtype is not bfloat16")
    if not _fixed_json_value_matches(text.get("rms_norm_eps"), 0.000001):
        raise ContractError("Qwen text config rms_norm_eps differs from 1e-6")
    return _qwen_shape_inputs(config, model)


def _validate_stop_identity(files: dict[str, _VerifiedCacheFile], model: dict[str, Any]) -> None:
    """Compare the two distinct EOS identities across config and tokenizer JSONs."""

    stop_identity = model["tokenizer_contract"]["stop_identity"]
    config_eos = stop_identity["config_eos"]
    tokenizer_eos = stop_identity["tokenizer_eos"]
    config = _read_cache_json(files, config_eos["source_file"])
    if not isinstance(config, dict):
        raise ContractError("config EOS source must be a JSON object")
    config_text = config.get("text_config")
    if not isinstance(config_text, dict) or config_text.get("eos_token_id") != config_eos["token_id"]:
        raise ContractError("config EOS ID does not match the locked stop identity")

    source_files = set(tokenizer_eos["source_files"])
    tokenizer_config_path = next((path for path in source_files if path.endswith("tokenizer_config.json")), None)
    tokenizer_json_path = next((path for path in source_files if path.endswith("tokenizer.json")), None)
    if tokenizer_config_path is None or tokenizer_json_path is None:
        raise ContractError("tokenizer stop identity must bind tokenizer_config.json and tokenizer.json")
    tokenizer_config = _read_cache_json(files, tokenizer_config_path)
    if not isinstance(tokenizer_config, dict):
        raise ContractError("tokenizer_config.json must be a JSON object")
    if tokenizer_config.get("eos_token") != tokenizer_eos["token"]:
        raise ContractError("tokenizer_config EOS token does not match the locked stop identity")
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise ContractError("tokenizer_config added_tokens_decoder is missing")
    for token_id, token, label in (
        (config_eos["token_id"], config_eos["token"], "config"),
        (tokenizer_eos["token_id"], tokenizer_eos["token"], "tokenizer"),
    ):
        decoder_entry = decoder.get(str(token_id))
        if not isinstance(decoder_entry, dict) or decoder_entry.get("content") != token:
            raise ContractError(f"tokenizer_config {label} EOS token ID does not match the locked stop identity")
        if sum(
            isinstance(entry, dict) and entry.get("content") == token
            for entry in decoder.values()
        ) != 1:
            raise ContractError(f"tokenizer_config {label} EOS content is missing or duplicated")

    tokenizer = _read_cache_json(files, tokenizer_json_path)
    if not isinstance(tokenizer, dict):
        raise ContractError("tokenizer.json must be a JSON object")
    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise ContractError("tokenizer.json added_tokens is missing")
    for token_id, token, label in (
        (config_eos["token_id"], config_eos["token"], "config"),
        (tokenizer_eos["token_id"], tokenizer_eos["token"], "tokenizer"),
    ):
        id_matches = [item for item in added_tokens if isinstance(item, dict) and item.get("id") == token_id]
        content_matches = [item for item in added_tokens if isinstance(item, dict) and item.get("content") == token]
        exact_matches = [
            item for item in added_tokens
            if isinstance(item, dict) and item.get("id") == token_id and item.get("content") == token
        ]
        if len(id_matches) != 1 or len(content_matches) != 1 or len(exact_matches) != 1:
            raise ContractError(f"tokenizer.json {label} EOS token ID does not match the locked stop identity")


def _assert_cache_identity_stable(cache_dir: Path, files: dict[str, _VerifiedCacheFile]) -> None:
    """Bind the verified descriptors to the paths after semantic reads finish."""

    for relative, verified in files.items():
        fd_stat = _assert_verified_fd_stable(verified, operation="semantic validation")
        try:
            path_stat = os.lstat(cache_dir / relative)
        except OSError as exc:
            raise ContractError(f"cannot inspect cache path after semantic validation: {relative}: {exc}") from exc
        if _stat_identity(fd_stat) != _stat_identity(path_stat):
            raise ContractError(f"cache path changed during semantic validation: {relative}")


def validate_cache(
    document: dict[str, Any],
    cache_dir: Path,
    *,
    schema_path: Path = SCHEMA_PATH,
    require_trusted_read_only: bool = False,
) -> None:
    """Verify every locked byte offline, optionally requiring a trusted RO cache."""

    cache_dir = Path(cache_dir)
    validate_document(document, schema_path=schema_path)
    model = document["model"]
    expected = {entry["path"]: entry for entry in model["files"]}
    root_before = _cache_root_stat(cache_dir, require_trusted_read_only=require_trusted_read_only)
    actual = _cache_entries(cache_dir, require_trusted_read_only=require_trusted_read_only)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ContractError(f"cache file set mismatch; missing={missing}, extra={extra}")
    verified_files: dict[str, _VerifiedCacheFile] = {}
    try:
        for path, entry in expected.items():
            verified_files[path] = _verify_cache_file(
                cache_dir,
                path,
                entry,
                require_trusted_read_only=require_trusted_read_only,
            )
        root_after_hash = _cache_root_stat(cache_dir, require_trusted_read_only=require_trusted_read_only)
        if _stat_identity(root_before) != _stat_identity(root_after_hash):
            raise ContractError("cache root changed during hash verification")

        # Every semantic read below consumes only a descriptor whose complete
        # contents have already been hashed.  The descriptors stay open until
        # the final path/root identity checks have completed.
        _validate_stop_identity(verified_files, model)
        qwen_shape_inputs = None
        if model["repo_id"] == REPO_ID:
            qwen_shape_inputs = _validate_qwen_config(verified_files, model)
        _validate_safetensors_headers(verified_files, model, qwen_shape_inputs)

        root_after_semantic = _cache_root_stat(cache_dir, require_trusted_read_only=require_trusted_read_only)
        if _stat_identity(root_before) != _stat_identity(root_after_semantic):
            raise ContractError("cache root changed during semantic validation")
        _assert_cache_identity_stable(cache_dir, verified_files)
    finally:
        for verified in verified_files.values():
            os.close(verified.file_descriptor)


def validate_lock_file(
    lock_path: Path = LOCK_PATH,
    *,
    schema_path: Path = SCHEMA_PATH,
    cache_dir: Path | None = None,
    require_trusted_read_only: bool = False,
) -> dict[str, Any]:
    lock_path = Path(lock_path)
    schema_path = Path(schema_path)
    document = read_json(
        lock_path,
        max_bytes=MAX_LOCK_JSON_BYTES,
        purpose="model lock",
    )
    validate_document(document, schema_path=schema_path)
    if cache_dir is not None:
        validate_cache(
            document,
            cache_dir,
            schema_path=schema_path,
            require_trusted_read_only=require_trusted_read_only,
        )
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--require-trusted-read-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        document = validate_lock_file(
            args.lock,
            schema_path=args.schema,
            cache_dir=args.cache_dir,
            require_trusted_read_only=args.require_trusted_read_only,
        )
        print(f"model lock validation: PASS fingerprint={document['fingerprint']}")
        if args.cache_dir is not None:
            if args.require_trusted_read_only:
                print(f"offline cache content + trusted read-only validation: PASS path={args.cache_dir}")
            else:
                print(f"offline cache content-only validation: PASS path={args.cache_dir}")
        return 0
    except (ContractError, OSError, JCSValidationError, ValueError) as exc:
        print(f"model lock validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
