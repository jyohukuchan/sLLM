#!/usr/bin/env python3
"""Independently validate SQ8 OpenWebUI release evidence.

The legacy phase-1 mode validates the immutable bundle, lifecycle journal, and
resource measurement contract without publishing a release decision.  The full
mode additionally reconstructs every derived view from raw evidence, binds the
recorded source tree to Git, and exclusively publishes release-validation.json.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import IO, Any, BinaryIO, Iterable, Iterator, NoReturn, Sequence, cast

TOOLS_DIRECTORY = Path(__file__).resolve().parent
if os.fspath(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS_DIRECTORY))

from sq8_full_campaign_bundle import FileEvidence  # noqa: E402
from sq8_full_campaign_independent_metrics import (  # noqa: E402
    IndependentMetricsError,
    reconstruct_latency_results,
    reconstruct_soak_resource_results,
)
from sq8_full_campaign_independent_views import (  # noqa: E402
    IndependentViewError,
    SOAK_RESULTS_SCHEMA,
    canonical_json_bytes as independent_canonical_json_bytes,
    reconstruct_front_views,
)


SESSION_SCHEMA = "ullm.sq8.openwebui_release.raw.v1"
RESOURCE_SCHEMA = "ullm.sq8.release_measurement.raw.v1"
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
MATRIX_SCHEMA = "ullm.sq8.openwebui_release.matrix.v1"
PHASE1_REPORT_SCHEMA = "ullm.sq8.openwebui_release.validation.phase1.v1"
FULL_REPORT_SCHEMA = "ullm.sq8.openwebui_release.validation.v1"
ENVIRONMENT_SCHEMA = "ullm.sq8.full_campaign.environment.v1"
MODEL_IDENTITY_SCHEMA = "ullm.sq8.full_campaign.model_identity.v1"
PROMOTION_SCHEMA = "ullm.sq8_product_promotion.v1"
ARTIFACT_SCHEMA = "sq-fp8-artifact-v0.2"
PACKAGE_SCHEMA = "ullm-prototype-manifest-v0.1"
WORKER_PROTOCOL_SCHEMA = "ullm.worker.v1"
API_CONTRACT_MODEL_ID = "ullm-qwen3-14b-sq8"
API_CONTRACT_MAX_RESPONSE_BYTES = 1024 * 1024
API_CONTRACT_INVALID_KEY_MESSAGE = "The supplied API key is invalid."
API_CONTRACT_QUERY_MESSAGE = "Query parameters are not supported."
API_CONTRACT_INVALID_JSON_MESSAGE = "The request body is not valid JSON."
API_CONTRACT_UNSUPPORTED_MESSAGE = "The requested parameter is not supported."
API_CONTRACT_MODEL_NOT_FOUND_MESSAGE = "The requested model does not exist."

API_CONTRACT_CANONICAL_BODY = (
    b'{"messages":[{"content":"API contract preflight","role":"user"}],'
    b'"model":"ullm-qwen3-14b-sq8"}'
)
API_CONTRACT_MALFORMED_BODY = b'{"broken":'
API_CONTRACT_DUPLICATE_KEY_BODY = (
    b'{"model":"ullm-qwen3-14b-sq8","model":"ullm-qwen3-14b-sq8",'
    b'"messages":[{"role":"user","content":"API contract preflight"}]}'
)
API_CONTRACT_UNSUPPORTED_N_BODY = (
    b'{"messages":[{"content":"API contract preflight","role":"user"}],'
    b'"model":"ullm-qwen3-14b-sq8","n":2}'
)
API_CONTRACT_MISSING_MODEL_BODY = (
    b'{"messages":[{"content":"API contract preflight","role":"user"}],'
    b'"model":"missing"}'
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
BOOT_ID_RE = re.compile(r"[0-9a-f]{32}")
NETWORK_ID_RE = re.compile(r"[0-9a-f]{64}")
BDF_RE = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_IDENTITY_JSON_BYTES = 2 * 1024 * 1024
MAX_SESSION_RECORDS = 16_384
MAX_SSE_ITEMS_PER_RESPONSE = 2_048
MAX_SESSION_SSE_ITEMS = 32_768
MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_SSE_EVENT_BYTES = 2 * 1024 * 1024
MAX_SSE_FINISH_REASON_BYTES = 64
MAX_BROWSER_SELECTOR_BYTES = 4096
MAX_SESSION_IDENTIFIER_BYTES = 512
MAX_HTTP_RESPONSE_HEADER_COUNT = 128
MAX_HTTP_HEADER_NAME_BYTES = 256
MAX_HTTP_HEADER_VALUE_BYTES = 8192
MAX_HTTP_RESPONSE_HEADER_BYTES = 64 * 1024
SOURCE_COPY_CHUNK_BYTES = 64 * 1024
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_GIT_CONTROL_OUTPUT_BYTES = 4096
SOURCE_COMMAND_TIMEOUT_SECONDS = 15.0
U64_MAX = (1 << 64) - 1
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
RESOURCE_FIXTURE_INPUT_PATH = "collector/resource-chat-fixture.json"
CONTEXT_OVERFLOW_CONTENT = {
    "context_overflow_1": "one" + (" overflow" * 5000),
    "context_overflow_2": "two" + (" overflow" * 5000),
}

UPSTREAM_MODEL_ID = "Qwen/Qwen3-14B-FP8"
SERVED_MODEL_ID = "ullm-qwen3-14b-sq8"
MODEL_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
SERVICE_UNIT = "ullm-openai.service"
DOCKER_NETWORK_NAME = "open-webui-network"
DOCKER_NETWORK_SUBNET = "172.20.0.0/16"
DOCKER_NETWORK_GATEWAY = "172.20.0.1"
DOCKER_NETWORK_ID = "79bb7cfca31cb5d76978cbbb229c946662c137b93ea647b5ae6c205af9126dc8"
DEVICE_ARCHITECTURE = "gfx1201"
EXECUTION_PROFILE = "rdna4_w8a8_block_ck"
CONTEXT_LENGTH = 4096
MAX_COMPLETION_TOKENS = 512
VOCAB_SIZE = 151_936
PROMOTION_PLAN_COMMIT = "dfc63de"
SYSTEMD_MAJOR = 255
OPENWEBUI_VERSION = "0.9.4-ullm.1"
OPENWEBUI_SOURCE_REVISION = "f51d2b026f1b0e7283b15f093412be8b67d24770"
OPENWEBUI_BASE_IMAGE_DIGEST = (
    "sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff"
)
OPENWEBUI_BASE_IMAGE_ID = (
    "sha256:18247c4608796dd5e416ec1e82f20457837a219ed9c272a8d64b405a262b3399"
)
OPENWEBUI_DERIVED_IMAGE_ID = (
    "sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409"
)
OPENWEBUI_PATCHED_MIDDLEWARE_SHA256 = (
    "b8aa5524fac6971aa8326cbef024b6fe9bcea03b3a00d4e7b0fa559514e0c66a"
)

EXPECTED_GPU_IDENTITY = {
    "index": 2,
    "bdf": "0000:47:00.0",
    "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
    "kfd_gpu_id": 51_545,
    "node_id": 2,
    "partition_id": 0,
    "architecture": DEVICE_ARCHITECTURE,
}

HIP_GUARDS = (
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

# This table is intentionally independent from the identity producer. A parity
# test makes source-contract changes explicit without importing producer code here.
EXPECTED_SOURCE_ROLE_PATHS = {
    "identity_generator": "tools/sq8_full_campaign_identity.py",
    "product_promotion_validator": "tools/validate-sq8-product-promotion.py",
    "product_promotion_canonical": "tools/sq8_canonical_artifact.py",
    "release_validator": "tools/validate-sq8-openwebui-release.py",
    "release_collector": "tools/collect-sq8-openwebui-release.py",
    "campaign_journal": "tools/sq8_openwebui_campaign.py",
    "campaign_bundle": "tools/sq8_full_campaign_bundle.py",
    "campaign_independent_metrics": "tools/sq8_full_campaign_independent_metrics.py",
    "campaign_independent_views": "tools/sq8_full_campaign_independent_views.py",
    "campaign_orchestrator": "tools/run-sq8-full-openwebui-campaign.py",
    "campaign_production": "tools/sq8_full_campaign_production.py",
    "campaign_prepare": "tools/sq8_full_campaign_prepare.py",
    "campaign_operational": "tools/sq8_full_campaign_operational.py",
    "worker_acceptance": "tools/run-sq8-worker-acceptance.py",
    "campaign_resource": "tools/sq8_full_campaign_resource.py",
    "campaign_renderer": "tools/sq8_full_campaign_renderer.py",
    "campaign_views": "tools/sq8_full_campaign_views.py",
    "gate_api_contract": "tools/run-sq8-api-contract-gate.py",
    "ingest_api_contract": "tools/sq8_api_contract_gate_ingest.py",
    "gate_openwebui_stop": "tools/run-openwebui-stop-gate.py",
    "ingest_openwebui_stop": "tools/sq8_openwebui_stop_gate_ingest.py",
    "gate_openwebui_soak": "tools/run-openwebui-soak-gate.py",
    "ingest_openwebui_gate": "tools/sq8_openwebui_gate_ingest.py",
    "gate_direct_cancel": "tools/run-sq8-direct-cancel-gate.py",
    "gate_openwebui_failure": "tools/run-openwebui-failure-gate.py",
    "gate_openwebui_failure_hook": "tools/run-openwebui-failure-hook.py",
    "ingest_openwebui_failure": "tools/sq8_openwebui_failure_gate_ingest.py",
    "gate_http_latency": "tools/run-sq8-http-latency-gate.py",
    "ingest_http_latency": "tools/sq8_http_latency_gate_ingest.py",
    "http_client": "tools/sq8-openwebui-http-client.py",
    "browser_smoke": "deploy/openwebui/browser-smoke.cjs",
    "browser_stop": "deploy/openwebui/browser-stop-smoke.cjs",
    "browser_failure": "deploy/openwebui/browser-failure-smoke.cjs",
    "browser_soak": "deploy/openwebui/browser-soak.cjs",
    "openwebui_dockerfile": "deploy/openwebui/Dockerfile",
    "openwebui_compose": "deploy/openwebui/compose.yaml",
    "openwebui_configure": "deploy/openwebui/configure.py",
    "openwebui_patch": "deploy/openwebui/provider-stream-error.patch",
    "openwebui_image_validator": "deploy/openwebui/verify-derived-image.sh",
    "systemd_service": "deploy/systemd/ullm-openai.service",
    "systemd_environment_contract": "deploy/systemd/ullm-openai.env.example",
    "gateway_pyproject": "services/openai-gateway/pyproject.toml",
    "gateway_lock": "services/openai-gateway/uv.lock",
    "gateway_init": "services/openai-gateway/src/ullm_openai_gateway/__init__.py",
    "gateway_main": "services/openai-gateway/src/ullm_openai_gateway/__main__.py",
    "gateway_app": "services/openai-gateway/src/ullm_openai_gateway/app.py",
    "gateway_errors": "services/openai-gateway/src/ullm_openai_gateway/errors.py",
    "gateway_schemas": "services/openai-gateway/src/ullm_openai_gateway/schemas.py",
    "gateway_settings": "services/openai-gateway/src/ullm_openai_gateway/settings.py",
    "gateway_tokenizer": "services/openai-gateway/src/ullm_openai_gateway/tokenizer.py",
    "gateway_worker": "services/openai-gateway/src/ullm_openai_gateway/worker.py",
    "worker_cargo_manifest": "crates/ullm-engine/Cargo.toml",
    "worker_entrypoint": "crates/ullm-engine/src/bin/ullm-sq8-worker.rs",
    "worker_backend": "crates/ullm-engine/src/sq8_worker_backend.rs",
    "worker_protocol": "crates/ullm-engine/src/sq8_worker_protocol.rs",
    "worker_runtime": "crates/ullm-engine/src/sq8_worker_runtime.rs",
    "engine_library": "crates/ullm-engine/src/lib.rs",
    "workspace_lock": "Cargo.lock",
    "serving_fixture_manifest": "tests/fixtures/sq8-serving-v0.1/manifest.json",
    "chat_template_fixture_manifest": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/manifest.json"
    ),
    "runtime_oracle_validation": (
        "benchmarks/results/2026-07-10/sq8-serving-v0.1/runtime-oracle-validation.json"
    ),
    "spec_release": "docs/specs/sq8-openwebui-release-v0.1.md",
    "spec_openai_chat_subset": "docs/specs/openai-chat-subset-v0.1.md",
    "spec_worker_protocol": "docs/specs/sq8-worker-protocol-v0.1.md",
    "fixture_ttft_p0032": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p0032.json"
    ),
    "fixture_ttft_p0128": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p0128.json"
    ),
    "fixture_ttft_p0512": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p0512.json"
    ),
    "fixture_ttft_p2048": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p2048.json"
    ),
    "fixture_ttft_p3584": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p3584.json"
    ),
}

EXPECTED_SOURCE_GROUPS = {
    "gateway": (
        "gateway_pyproject",
        "gateway_lock",
        "gateway_init",
        "gateway_main",
        "gateway_app",
        "gateway_errors",
        "gateway_schemas",
        "gateway_settings",
        "gateway_tokenizer",
        "gateway_worker",
    ),
    "worker": (
        "worker_cargo_manifest",
        "worker_entrypoint",
        "worker_backend",
        "worker_protocol",
        "worker_runtime",
        "engine_library",
        "workspace_lock",
    ),
    "collector": ("release_collector", "campaign_journal"),
    "browser": (
        "browser_smoke",
        "browser_stop",
        "browser_failure",
        "browser_soak",
    ),
    "http_client": ("http_client",),
    "deployment": (
        "openwebui_dockerfile",
        "openwebui_compose",
        "openwebui_configure",
        "openwebui_patch",
        "openwebui_image_validator",
        "systemd_service",
        "systemd_environment_contract",
    ),
    "oracle": (
        "serving_fixture_manifest",
        "chat_template_fixture_manifest",
        "runtime_oracle_validation",
    ),
    "campaign": (
        "identity_generator",
        "product_promotion_validator",
        "product_promotion_canonical",
        "release_validator",
        "release_collector",
        "campaign_journal",
        "campaign_bundle",
        "campaign_independent_metrics",
        "campaign_independent_views",
        "campaign_orchestrator",
        "campaign_production",
        "campaign_prepare",
        "campaign_operational",
        "worker_acceptance",
        "campaign_resource",
        "campaign_renderer",
        "campaign_views",
        "gate_api_contract",
        "ingest_api_contract",
        "gate_openwebui_stop",
        "ingest_openwebui_stop",
        "gate_openwebui_soak",
        "ingest_openwebui_gate",
        "gate_direct_cancel",
        "gate_openwebui_failure",
        "gate_openwebui_failure_hook",
        "ingest_openwebui_failure",
        "gate_http_latency",
        "ingest_http_latency",
        "http_client",
        "browser_smoke",
        "browser_stop",
        "browser_failure",
        "browser_soak",
    ),
    "spec": ("spec_release", "spec_openai_chat_subset", "spec_worker_protocol"),
    "fixture": (
        "serving_fixture_manifest",
        "chat_template_fixture_manifest",
        "fixture_ttft_p0032",
        "fixture_ttft_p0128",
        "fixture_ttft_p0512",
        "fixture_ttft_p2048",
        "fixture_ttft_p3584",
    ),
    "all": tuple(EXPECTED_SOURCE_ROLE_PATHS),
}

EXPECTED_ARTIFACT_IDENTITY = {
    "schema_version": ARTIFACT_SCHEMA,
    "manifest_file": "artifact/sq_manifest.json",
    "manifest_bytes": 379_114,
    "manifest_sha256": "23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2",
    "content_sha256": "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
    "selected_pair_count": 280,
    "payload_bytes": 13_213_670_400,
    "file_count": 561,
    "payloads_hashed": True,
}
EXPECTED_PACKAGE_IDENTITY = {
    "schema_version": PACKAGE_SCHEMA,
    "manifest_file": "package/manifest.json",
    "manifest_bytes": 91_910,
    "manifest_sha256": "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
    "payload_count": 163,
    "payload_bytes": 3_112_499_200,
    "file_count": 164,
    "payloads_hashed": True,
}

EXPECTED_TOKENIZER_FILES = (
    (
        "config.json",
        896,
        "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793",
    ),
    (
        "generation_config.json",
        240,
        "231c22c0b89ffbbb785d0e68b2f3f922244f263487af79f6542fc82dbee37dbf",
    ),
    (
        "merges.txt",
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    (
        "model.safetensors.index.json",
        62_044,
        "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151",
    ),
    (
        "tokenizer.json",
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "tokenizer_config.json",
        9_732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    (
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)
EXPECTED_CHAT_TEMPLATE_IDENTITY = {
    "utf8_bytes": 4_168,
    "sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
}
EXPECTED_ORACLE_FILE_IDENTITIES: dict[str, dict[str, Any]] = {
    "serving_fixture_manifest": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["serving_fixture_manifest"],
        "bytes": 31_749,
        "sha256": "3b6362fd472debbbfb30fb5616325703dd52e90e82319280490a0d84fcd6bf83",
    },
    "chat_template_fixture_manifest": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["chat_template_fixture_manifest"],
        "bytes": 4_948,
        "sha256": "6324b74e2604b86d46bf2dfdc259c1ca68d8cc9a47e90bfb765919f4aa9d54e0",
    },
    "runtime_oracle_validation": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["runtime_oracle_validation"],
        "bytes": 17_334,
        "sha256": "2612a4b434b6f5a15ab0c94d4465ee232a999a854a56a374b88904bdc54524aa",
    },
}
EXPECTED_TTFT_FIXTURE_IDENTITIES: dict[str, dict[str, Any]] = {
    "fixture_ttft_p0032": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["fixture_ttft_p0032"],
        "bytes": 1_333,
        "sha256": "c660c7fb3c25d2a3e25693e2beb2abc10295a06935772d17d23cedab04f24c07",
    },
    "fixture_ttft_p0128": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["fixture_ttft_p0128"],
        "bytes": 2_776,
        "sha256": "f8fe81bacb8761f3aa10cce1c333a51f9a85d65b5bfc7b02499886fb9f550a37",
    },
    "fixture_ttft_p0512": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["fixture_ttft_p0512"],
        "bytes": 8_538,
        "sha256": "e2f53c514a228e9e10871fc0df1867394aae12416215c9716770d2b420a3480f",
    },
    "fixture_ttft_p2048": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["fixture_ttft_p2048"],
        "bytes": 31_581,
        "sha256": "cd04c3339542f07731074ac0e00740a83061e620f6caff9c2a7e5316df1ccdcf",
    },
    "fixture_ttft_p3584": {
        "path": EXPECTED_SOURCE_ROLE_PATHS["fixture_ttft_p3584"],
        "bytes": 54_622,
        "sha256": "e3cd6c722302f73d688492b73a182298f34cc0a1498def209c262e5e9aa92912",
    },
}
EXPECTED_VLLM_IDENTITY = {
    "async_scheduling": False,
    "backend": "vLLM",
    "device": {
        "compute_capability": [12, 0],
        "gfx": DEVICE_ARCHITECTURE,
        "name": "AMD Radeon Graphics",
        "total_memory_bytes": 34_208_743_424,
        "visible_device_index": 0,
    },
    "dtype": "bfloat16",
    "enable_prefix_caching": False,
    "enforce_eager": True,
    "max_num_seqs": 1,
    "package_version": "0.23.1rc1.dev618+g8cf7c4d8a.rocm723",
    "pipeline_parallel_size": 1,
    "python_version": "3.12.3",
    "rocr_visible_devices": "1",
    "runner": "LLM.generate",
    "source_revision_from_package_version": "8cf7c4d8a",
    "tensor_parallel_size": 1,
    "torch_git_version": "d0c8b1f364ecacff4dd8bc06a645d0fb9324cd37",
    "torch_hip_version": "7.2.53211",
    "torch_version": "2.11.0+gitd0c8b1f",
    "transformers_version": "5.12.1",
}

FIXTURE_IDS = (
    "exact-p0032",
    "exact-p0128",
    "exact-p0512",
    "exact-p2048",
    "exact-p3584",
)
CANCEL_PHASES = (
    "after_started_before_progress",
    "prefill_after_128",
    "prefill_after_2048",
    "decode_after_first_content",
    "openwebui_stop_after_visible_content",
)
PHASES = {
    "preflight",
    "api_contract",
    "openwebui",
    "cancellation",
    "resource_normal",
    "post_header_failure",
    "resource_restart",
    "latency",
    "final",
}
FULL_CAMPAIGN_PHASE_ORDER = (
    "preflight",
    "api_contract",
    "openwebui",
    "cancellation",
    "resource_normal",
    "post_header_failure",
    "resource_restart",
    "latency",
    "final",
)

SCHEDULE = {
    "openwebui_chats": 20,
    "cancel_phases": list(CANCEL_PHASES),
    "normal_warmups": 10,
    "normal_requests": 100,
    "sampled_normal_indices": list(range(5, 101, 5)),
    "restart_warmups": 10,
    "restart_requests": 20,
    "ttft_fixture_ids": list(FIXTURE_IDS),
    "latency_warmups_per_case": 2,
    "latency_measured_per_case": 10,
    "decode_warmups": 2,
    "decode_measured": 10,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
RESOURCE_SCHEDULE = {
    "normal_warmups": 10,
    "normal_requests": 100,
    "restart_warmups": 10,
    "restart_requests": 20,
    "idle_settle_ms": 5000,
    "samples_per_point": 5,
    "sample_interval_ms": 1000,
}
THRESHOLDS = {
    "ttft_seconds_maximum": {
        "exact-p0032": {"p50": Decimal("2.5"), "p95": 3},
        "exact-p0128": {"p50": 4, "p95": 5},
        "exact-p0512": {"p50": 10, "p95": 12},
        "exact-p2048": {"p50": 30, "p95": 35},
        "exact-p3584": {"p50": 50, "p95": 60},
    },
    "decode_p50_tokens_per_second_minimum": 15,
    "decode_p95_inter_content_seconds_maximum": Decimal("0.1"),
    "cancel_release_max_ns": 5_000_000_000,
    "final_delta_max_bytes": 67_108_864,
    "theil_sen_max_bytes_per_request": 262_144,
}

COMMANDS = {
    "systemd_version": "systemctl --version",
    "service_identity": (
        "systemctl show ullm-openai.service --property=ControlGroup "
        "--property=MainPID --no-pager"
    ),
    "cgroup_type": "stat -fc %T /sys/fs/cgroup",
    "host_memory": "cat /sys/fs/cgroup${ControlGroup}/memory.current",
    "proc_stat": "cat /proc/${PID}/stat",
    "proc_status": "cat /proc/${PID}/status",
    "proc_exe": "readlink /proc/${PID}/exe",
    "proc_fds": "find -P /proc/${PID}/fd -mindepth 1 -maxdepth 1 -printf '%f\\n'",
    "proc_children": "cat /proc/${PID}/task/${PID}/children",
    "amd_smi_version": "amd-smi version",
    "amd_smi_list": "amd-smi list --json",
    "amd_smi_process": "amd-smi process --gpu 2 --general --json",
    "amd_smi_metric": "amd-smi metric --gpu 2 --json",
    "kfd_proc_probe": "test -d /sys/class/kfd/kfd/proc",
    "kfd_processes": (
        "find -P /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f\\n'"
    ),
    "kfd_vram": "cat /sys/class/kfd/kfd/proc/${PID}/vram_51545",
}

EXPECTED_ROLES = {
    "environment.json": "environment",
    "model-identity.json": "model_identity",
    "raw-session-results.jsonl": "session_raw",
    "soak-resources.raw.jsonl": "resource_raw",
    "service-journal.raw.jsonl": "service_journal_raw",
    "amd-smi-metric-normal-before.json": "gpu_metric_raw",
    "amd-smi-metric-normal-after.json": "gpu_metric_raw",
    "amd-smi-metric-restart-before.json": "gpu_metric_raw",
    "amd-smi-metric-restart-after.json": "gpu_metric_raw",
    "sampling-results.json": "derived_view",
    "cancel-results.json": "derived_view",
    "prefill-latency-results.json": "derived_view",
    "api-contract-results.json": "derived_view",
    "openwebui-smoke.json": "derived_view",
    "soak-results.json": "derived_view",
    "browser/openwebui-stop-before.png": "browser_screenshot",
    "browser/post-header-failure.png": "browser_screenshot",
}
MATRIX_EXCLUDED = {
    "release-matrix.json",
    "release-validation.json",
    "summary.md",
    "SHA256SUMS",
}
BUNDLE_FILES = set(EXPECTED_ROLES) | {
    "release-matrix.json",
    "summary.md",
    "SHA256SUMS",
}

COMMON_SESSION_FIELDS = {
    "schema_version",
    "record_type",
    "sequence",
    "phase",
    "case_id",
}
SESSION_FIELDS = {
    "header": {
        "run_id",
        "started_utc",
        "clock",
        "boot_id",
        "identities",
        "input_files",
        "schedule",
        "thresholds",
    },
    "http_request": {
        "request_index",
        "request_key",
        "method",
        "target",
        "headers",
        "body_base64",
        "body_sha256",
        "body_bytes",
        "connect_completed_monotonic_ns",
        "write_started_monotonic_ns",
        "last_body_byte_sent_monotonic_ns",
    },
    "http_response_start": {
        "request_key",
        "status",
        "headers",
        "observed_monotonic_ns",
    },
    "http_body_chunk": {
        "request_key",
        "chunk_index",
        "body_base64",
        "body_sha256",
        "body_bytes",
        "observed_monotonic_ns",
    },
    "http_response_end": {
        "request_key",
        "outcome",
        "error",
        "body_bytes",
        "body_sha256",
        "observed_monotonic_ns",
    },
    "gateway_event": {
        "journal_cursor",
        "journal_monotonic_usec",
        "journal_pid",
        "message",
        "message_sha256",
        "event",
    },
    "api_journal_observation": {
        "observation_index",
        "journal_cursor",
        "journal_monotonic_usec",
        "journal_pid",
        "message_utf8_bytes",
        "message_sha256",
    },
    "lifecycle_quiet_check": {
        "quiet_sequence",
        "label",
        "checked_monotonic_ns",
        "observer_open",
        "observer_event_count",
        "new_journal_record_count",
        "journal_record_count",
        "journal_cursor",
    },
    "browser_action": {
        "browser_case",
        "action_index",
        "action",
        "selector",
        "input_sha256",
        "started_monotonic_ns",
        "completed_monotonic_ns",
        "result",
        "screenshot_file",
        "screenshot_sha256",
    },
    "lifecycle_probe": {
        "probe",
        "observed_monotonic_ns",
        "service_active",
        "ready_http_status",
        "control_group",
        "gateway_pid",
        "gateway_starttime_ticks",
        "worker_pid",
        "worker_starttime_ticks",
        "n_restarts",
    },
    "fault_injection": {
        "injection",
        "target_pid",
        "target_starttime_ticks",
        "signal",
        "command",
        "started_monotonic_ns",
        "completed_monotonic_ns",
    },
    "run_end": {
        "completed_utc",
        "completed_monotonic_ns",
        "final_git_commit",
        "final_git_status_raw",
        "final_git_status_sha256",
        "record_counts",
        "final_journal_cursor",
    },
}

LIFECYCLE_FIELDS = {
    "request_admitted": {
        "request_id",
        "completion_id",
        "stream",
        "prompt_tokens",
        "max_completion_tokens",
    },
    "request_started": {
        "request_id",
        "completion_id",
        "stream",
        "prompt_tokens",
        "admit_to_start_ns",
    },
    "request_progress": {
        "request_id",
        "completion_id",
        "phase",
        "processed_prompt_tokens",
        "prompt_tokens",
    },
    "request_first_token": {
        "request_id",
        "completion_id",
        "stream",
        "completion_tokens",
    },
    "request_cancel_requested": {
        "request_id",
        "completion_id",
        "stream",
        "reason",
        "admit_to_cancel_ns",
    },
    "request_released": {
        "request_id",
        "completion_id",
        "stream",
        "outcome",
        "cancel_reason",
        "prompt_tokens",
        "completion_tokens",
        "reset_complete",
        "admit_to_start_ns",
        "start_to_release_ns",
        "admit_to_release_ns",
    },
    "worker_fatal": {"request_id", "completion_id", "reason", "admit_to_fatal_ns"},
}

RESOURCE_HEADER_FIELDS = {
    "schema_version",
    "record_type",
    "service_unit",
    "commands",
    "tools",
    "probes",
    "schedule",
}
RESOURCE_SAMPLE_FIELDS = {
    "schema_version",
    "record_type",
    "segment",
    "phase",
    "request_index",
    "request_id",
    "release_outcome",
    "release_observed_monotonic_ns",
    "reset_complete",
    "idle_settle_started_monotonic_ns",
    "sample_index",
    "sample_monotonic_ns",
    "systemd",
    "host",
    "gateway",
    "worker",
    "gpu",
}
SYSTEMD_FIELDS = {
    "control_group_before",
    "control_group_after",
    "main_pid_before",
    "main_pid_after",
}
HOST_FIELDS = {"memory_current_bytes"}
PROCESS_FIELDS = {
    "pid",
    "ppid",
    "exe",
    "starttime_ticks_before",
    "starttime_ticks_after",
    "vmrss_kb",
    "vmrss_bytes",
    "threads",
    "fd_count",
    "children",
}
GPU_FIELDS = {
    "index",
    "bdf",
    "uuid",
    "kfd_gpu_id",
    "process_record_count",
    "worker_pid",
    "mem_usage",
    "kfd_vram_bytes",
    "unrelated_process_pids",
}
GPU_METRIC_FIELDS = {
    "schema_version",
    "record_type",
    "segment",
    "boundary",
    "captured_monotonic_ns",
    "gpu_index",
    "raw_output_file",
    "raw_output_sha256",
}


class ValidationError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    raise ValidationError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    fail(f"JSON contains a non-finite numeric constant: {value}")


def _validate_unicode(value: Any, label: str) -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError as error:
            fail(f"{label} contains an invalid Unicode scalar: {error}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_unicode(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_unicode(key, f"{label} key")
            _validate_unicode(item, f"{label}.{key}")


def decode_json_bytes(
    raw: bytes,
    label: str,
    *,
    allow_outer_whitespace: bool = False,
    require_object: bool = True,
) -> Any:
    if not raw or len(raw) > MAX_JSON_BYTES:
        fail(f"{label} has an invalid size")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        fail(f"{label} is not strict UTF-8: {error}")
    if not allow_outer_whitespace and (
        not text.startswith("{") or not text.endswith("}")
    ):
        fail(f"{label} must contain exactly one JSON object without outer whitespace")
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_float=Decimal,
            parse_constant=reject_json_constant,
        )
    except ValidationError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        fail(f"failed to decode {label}: {error}")
    _validate_unicode(value, label)
    if require_object and type(value) is not dict:
        fail(f"{label} must be an object")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JSON_BYTES:
            fail(f"{label} has an invalid size: {size}")
        raw = path.read_bytes()
    except OSError as error:
        fail(f"failed to read {label}: {error}")
    return decode_json_bytes(raw, label, allow_outer_whitespace=True)


def validate_json_document(path: Path, label: str) -> None:
    regular_file(path, label)
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_JSON_BYTES:
            fail(f"{label} has an invalid size: {size}")
        raw = path.read_bytes()
    except OSError as error:
        fail(f"failed to read {label}: {error}")
    decode_json_bytes(
        raw,
        label,
        allow_outer_whitespace=True,
        require_object=False,
    )


def iter_jsonl(path: Path, label: str) -> Iterator[tuple[int, dict[str, Any]]]:
    regular_file(path, label)
    try:
        handle: BinaryIO
        with path.open("rb") as handle:
            line_number = 0
            while True:
                raw = handle.readline(MAX_JSON_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > MAX_JSON_BYTES:
                    fail(f"{label} line {line_number} exceeds the size limit")
                if not raw.endswith(b"\n"):
                    fail(f"{label} line {line_number} is not LF-terminated")
                raw = raw[:-1]
                if raw.endswith(b"\r"):
                    fail(f"{label} line {line_number} uses CRLF")
                yield line_number, decode_json_bytes(raw, f"{label} line {line_number}")
            if line_number == 0:
                fail(f"{label} is empty")
    except ValidationError:
        raise
    except OSError as error:
        fail(f"failed to read {label}: {error}")


def exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(
            f"{label} field set differs: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def integer(value: Any, label: str, minimum: int = 0, maximum: int = U64_MAX) -> int:
    if type(value) is not int:
        fail(f"{label} must be an integer")
    if value < minimum or value > maximum:
        fail(f"{label} is outside {minimum}..={maximum}")
    return value


def boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        fail(f"{label} must be a boolean")
    return value


def string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        fail(f"{label} must be {'a non-empty ' if nonempty else 'a '}string")
    return value


def nullable_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return string(value, label)


def bounded_utf8_string(
    value: Any, label: str, *, maximum_bytes: int, nonempty: bool = True
) -> str:
    result = string(value, label, nonempty=nonempty)
    if len(result.encode("utf-8")) > maximum_bytes:
        fail(f"{label} exceeds its UTF-8 byte bound")
    return result


def sha256_value(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def git_commit(value: Any, label: str) -> str:
    if type(value) is not str or GIT_COMMIT_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase 40-hex Git commit")
    return value


def json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def reject_key_recursive(value: Any, forbidden: str, label: str) -> None:
    if type(value) is dict:
        if forbidden in value:
            fail(f"{label} contains forbidden key {forbidden!r}")
        for key, item in value.items():
            reject_key_recursive(item, forbidden, f"{label}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            reject_key_recursive(item, forbidden, f"{label}[{index}]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        fail(f"failed to hash {path}: {error}")
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        fail(f"failed to stat {label}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular non-symlink file")
    return path


def _absolute_without_resolution(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def safe_bundle_root(path: Path) -> Path:
    absolute = _absolute_without_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            fail(f"failed to stat bundle path component {current}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"bundle path contains a symlink component: {current}")
    if not absolute.is_dir():
        fail("bundle root must be a directory")
    return absolute


def safe_relative_file(root: Path, relative: str, label: str) -> Path:
    if type(relative) is not str or not relative or "\\" in relative:
        fail(f"{label} is not a safe relative path")
    pure = PurePosixPath(relative)
    lexical_parts = relative.split("/")
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in lexical_parts):
        fail(f"{label} is not a safe relative path")
    current = root
    for part in pure.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            fail(f"failed to stat {label}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains a symlink")
    return regular_file(current, label)


def _identity_canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        fail(f"identity document cannot be canonically serialized: {error}")
        raise AssertionError("unreachable")


def _read_canonical_identity(
    root: Path, relative: str, label: str
) -> tuple[dict[str, Any], bytes]:
    path = safe_relative_file(root, relative, label)
    try:
        size = path.stat().st_size
        if not 1 <= size <= MAX_IDENTITY_JSON_BYTES:
            fail(f"{label} has an invalid size: {size}")
        raw = path.read_bytes()
    except OSError as error:
        fail(f"failed to read {label}: {error}")
    value = cast(
        dict[str, Any],
        decode_json_bytes(raw, label, allow_outer_whitespace=True),
    )
    reject_key_recursive(value, "passed", label)
    if raw != _identity_canonical_bytes(value):
        fail(f"{label} is not canonical identity JSON")
    return value, raw


def _identity_timestamp(
    value: Any, label: str, *, require_utc_z: bool = False
) -> datetime:
    text = string(value, label)
    if require_utc_z and not text.endswith("Z"):
        fail(f"{label} must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError:
        fail(f"{label} is not ISO-8601")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        fail(f"{label} lacks a UTC offset")
    if require_utc_z and parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        fail(f"{label} is not UTC")
    return parsed


def _identity_file(
    value: Any,
    label: str,
    *,
    expected_path: str | None = None,
) -> dict[str, Any]:
    item = exact_fields(value, {"path", "bytes", "sha256"}, label)
    path = string(item["path"], f"{label}.path")
    if expected_path is not None and path != expected_path:
        fail(f"{label}.path differs")
    integer(item["bytes"], f"{label}.bytes", minimum=1)
    sha256_value(item["sha256"], f"{label}.sha256")
    return item


def _identity_process(value: Any, label: str) -> dict[str, Any]:
    process = exact_fields(
        value,
        {
            "pid",
            "ppid",
            "uid",
            "gid",
            "starttime_ticks",
            "executable",
            "executable_bytes",
            "executable_sha256",
            "children",
        },
        label,
    )
    integer(process["pid"], f"{label}.pid", minimum=1)
    integer(process["ppid"], f"{label}.ppid")
    integer(process["uid"], f"{label}.uid")
    integer(process["gid"], f"{label}.gid")
    integer(process["starttime_ticks"], f"{label}.starttime_ticks", minimum=1)
    executable = string(process["executable"], f"{label}.executable")
    if not Path(executable).is_absolute():
        fail(f"{label}.executable is not absolute")
    integer(process["executable_bytes"], f"{label}.executable_bytes", minimum=1)
    sha256_value(process["executable_sha256"], f"{label}.executable_sha256")
    children = process["children"]
    if type(children) is not list:
        fail(f"{label}.children is not an array")
    parsed = [
        integer(child, f"{label}.children[{index}]", minimum=1)
        for index, child in enumerate(children)
    ]
    if parsed != sorted(set(parsed)):
        fail(f"{label}.children is not ascending and unique")
    return process


def _identity_source_aggregate(
    entries_by_role: dict[str, dict[str, Any]], roles: Iterable[str]
) -> str:
    entries = [entries_by_role[role] for role in sorted(roles)]
    return hashlib.sha256(_identity_canonical_bytes(entries)).hexdigest()


def _validate_identity_source_contract(
    role_paths: dict[str, str] | None = None,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> None:
    checked_paths = EXPECTED_SOURCE_ROLE_PATHS if role_paths is None else role_paths
    checked_groups = EXPECTED_SOURCE_GROUPS if groups is None else groups
    if type(checked_paths) is not dict or not checked_paths:
        fail("source role contract differs")
    paths: list[str] = []
    for role, path in checked_paths.items():
        pure = PurePosixPath(path) if type(path) is str else None
        if (
            type(role) is not str
            or not role
            or pure is None
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or "\\" in path
        ):
            fail("source role or path contract differs")
        paths.append(path)
    if len(paths) != len(set(paths)):
        fail("source role paths are not unique")
    if (
        type(checked_groups) is not dict
        or not checked_groups
        or "all" not in checked_groups
    ):
        fail("source group contract differs")
    known_roles = set(checked_paths)
    semantic_roles: set[str] = set()
    for group, roles in checked_groups.items():
        if (
            type(group) is not str
            or not group
            or type(roles) is not tuple
            or not roles
            or len(roles) != len(set(roles))
            or any(type(role) is not str or role not in known_roles for role in roles)
        ):
            fail("source group roles differ")
        if group != "all":
            semantic_roles.update(roles)
    if checked_groups["all"] != tuple(checked_paths) or semantic_roles != known_roles:
        fail("source group coverage differs")


@dataclass(frozen=True)
class _SourceDigest:
    byte_count: int
    sha256: str
    raw: bytes | None = None


def _stop_source_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _run_bounded_source_command(
    command: tuple[str, ...], *, maximum: int, retain: bool
) -> _SourceDigest:
    if (
        type(command) is not tuple
        or not command
        or not 0 <= maximum <= MAX_SOURCE_BYTES
    ):
        fail("source command contract differs")
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    stdout: IO[bytes] | None = None
    raw = bytearray() if retain else None
    total = 0
    digest = hashlib.sha256()
    deadline = time.monotonic() + SOURCE_COMMAND_TIMEOUT_SECONDS
    command_environment = os.environ.copy()
    command_environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    command_environment["GIT_OPTIONAL_LOCKS"] = "0"
    command_environment["LC_ALL"] = "C"
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env=command_environment,
        )
        stdout = process.stdout
        if stdout is None:
            fail("source command stdout is unavailable")
            raise AssertionError("unreachable")
        selector = selectors.DefaultSelector()
        selector.register(stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fail("source command exceeded its deadline")
            if not selector.select(remaining):
                fail("source command exceeded its deadline")
            chunk = os.read(
                stdout.fileno(),
                min(SOURCE_COPY_CHUNK_BYTES, maximum + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                fail("source command stdout exceeded its bound")
            digest.update(chunk)
            if raw is not None:
                raw.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail("source command exceeded its deadline")
        if process.wait(timeout=remaining) != 0:
            fail("source command failed")
        return _SourceDigest(
            total, digest.hexdigest(), bytes(raw) if raw is not None else None
        )
    except ValidationError:
        if process is not None:
            _stop_source_process(process)
        raise
    except (OSError, subprocess.SubprocessError):
        if process is not None:
            _stop_source_process(process)
        fail("source command execution failed")
        raise AssertionError("unreachable")
    finally:
        if selector is not None:
            selector.close()
        if process is not None and process.stdout is not None:
            process.stdout.close()


def _source_directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for source validation")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _source_file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for source validation")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _source_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclass(frozen=True)
class _OpenedSource:
    descriptor: int
    parent_fd: int
    name: str
    identity: tuple[int, ...]


def _open_source_from_root(root_fd: int, relative: str) -> _OpenedSource:
    parts = relative.split("/")
    current_fd = -1
    try:
        current_fd = os.dup(root_fd)
        for component in parts[:-1]:
            next_fd = os.open(component, _source_directory_flags(), dir_fd=current_fd)
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                os.close(next_fd)
                fail("source path parent is not a directory")
            os.close(current_fd)
            current_fd = next_fd
        entry = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
        descriptor = os.open(parts[-1], _source_file_flags(), dir_fd=current_fd)
        identity = _source_stat_identity(entry)
        if _source_stat_identity(os.fstat(descriptor)) != identity:
            os.close(descriptor)
            fail("source file changed while opening")
        result = _OpenedSource(descriptor, current_fd, parts[-1], identity)
        current_fd = -1
        return result
    except ValidationError:
        raise
    except OSError:
        fail("failed to open a source file without following links")
        raise AssertionError("unreachable")
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _hash_worktree_source(root_fd: int, relative: str) -> _SourceDigest:
    opened: _OpenedSource | None = None
    try:
        opened = _open_source_from_root(root_fd, relative)
        before = os.fstat(opened.descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= MAX_SOURCE_BYTES
        ):
            fail("source file is not one bounded regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(opened.descriptor, SOURCE_COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_SOURCE_BYTES:
                fail("source file exceeded its streaming bound")
            digest.update(chunk)
        if (
            total != before.st_size
            or _source_stat_identity(os.fstat(opened.descriptor)) != opened.identity
            or _source_stat_identity(
                os.stat(opened.name, dir_fd=opened.parent_fd, follow_symlinks=False)
            )
            != opened.identity
        ):
            fail("source file changed while hashing")
        return _SourceDigest(total, digest.hexdigest())
    except ValidationError:
        raise
    except OSError:
        fail("failed to hash a source file")
        raise AssertionError("unreachable")
    finally:
        if opened is not None:
            os.close(opened.descriptor)
            os.close(opened.parent_fd)


def _git_source_digest(repo_root: Path, commit: str, relative: str) -> _SourceDigest:
    object_name = f"{commit}:{relative}"
    size_result = _run_bounded_source_command(
        (
            "git",
            "--no-pager",
            "-C",
            os.fspath(repo_root),
            "cat-file",
            "-s",
            object_name,
        ),
        maximum=MAX_GIT_CONTROL_OUTPUT_BYTES,
        retain=True,
    )
    if size_result.raw is None:
        fail("Git source blob size output is unavailable")
        raise AssertionError("unreachable")
    try:
        size_text = size_result.raw.decode("ascii", errors="strict")
        if not size_text.endswith("\n") or not size_text[:-1].isdecimal():
            fail("Git source blob size output differs")
        size = int(size_text[:-1], 10)
    except UnicodeError:
        fail("Git source blob size output differs")
    if not 1 <= size <= MAX_SOURCE_BYTES:
        fail("Git source blob size exceeds its bound")
    result = _run_bounded_source_command(
        (
            "git",
            "--no-pager",
            "-C",
            os.fspath(repo_root),
            "cat-file",
            "blob",
            object_name,
        ),
        maximum=size,
        retain=False,
    )
    if result.byte_count != size:
        fail("Git source blob size changed while reading")
    return result


def _validate_environment_identity(
    document: dict[str, Any], expected_commit: str
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    _validate_identity_source_contract()
    exact_fields(
        document,
        {
            "schema_version",
            "record_type",
            "captured_utc",
            "git",
            "sources",
            "source_sets",
            "deployment",
            "host",
            "service",
            "openwebui",
        },
        "environment.json",
    )
    if (
        document["schema_version"] != ENVIRONMENT_SCHEMA
        or document["record_type"] != "environment"
    ):
        fail("environment.json schema or record type differs")
    _identity_timestamp(
        document["captured_utc"], "environment.json.captured_utc", require_utc_z=True
    )
    git = exact_fields(
        document["git"], {"commit", "dirty", "status_sha256"}, "environment.json.git"
    )
    if git_commit(git["commit"], "environment.json.git.commit") != expected_commit:
        fail("environment.json Git commit differs from the trusted CLI anchor")
    dirty = boolean(git["dirty"], "environment.json.git.dirty")
    status_sha = sha256_value(
        git["status_sha256"], "environment.json.git.status_sha256"
    )
    if dirty == (status_sha == EMPTY_SHA256):
        fail("environment.json Git dirty flag and status SHA-256 disagree")

    sources = document["sources"]
    if type(sources) is not list or len(sources) != len(EXPECTED_SOURCE_ROLE_PATHS):
        fail("environment.json source list count differs")
    by_role: dict[str, dict[str, Any]] = {}
    paths: list[str] = []
    for index, raw in enumerate(sources):
        entry = exact_fields(
            raw,
            {"role", "path", "bytes", "sha256"},
            f"environment.json.sources[{index}]",
        )
        role = string(entry["role"], f"environment.json.sources[{index}].role")
        path = string(entry["path"], f"environment.json.sources[{index}].path")
        if (
            role not in EXPECTED_SOURCE_ROLE_PATHS
            or role in by_role
            or path != EXPECTED_SOURCE_ROLE_PATHS[role]
        ):
            fail("environment.json source role or path differs")
        integer(entry["bytes"], f"environment.json.sources[{index}].bytes", minimum=1)
        sha256_value(entry["sha256"], f"environment.json.sources[{index}].sha256")
        by_role[role] = entry
        paths.append(path)
    if set(by_role) != set(EXPECTED_SOURCE_ROLE_PATHS) or paths != sorted(
        paths, key=lambda item: item.encode("utf-8")
    ):
        fail("environment.json sources are not the exact bytewise-sorted set")
    for role, expected in EXPECTED_ORACLE_FILE_IDENTITIES.items():
        source = by_role[role]
        if any(source[key] != expected[key] for key in ("path", "bytes", "sha256")):
            fail(f"environment.json oracle source {role} differs")
    for role, expected in EXPECTED_TTFT_FIXTURE_IDENTITIES.items():
        source = by_role[role]
        if any(source[key] != expected[key] for key in ("path", "bytes", "sha256")):
            fail(f"environment.json TTFT fixture source {role} differs")

    source_sets_raw = exact_fields(
        document["source_sets"],
        set(EXPECTED_SOURCE_GROUPS),
        "environment.json.source_sets",
    )
    source_sets: dict[str, str] = {}
    for group, roles in EXPECTED_SOURCE_GROUPS.items():
        digest = sha256_value(
            source_sets_raw[group], f"environment.json.source_sets.{group}"
        )
        if digest != _identity_source_aggregate(by_role, roles):
            fail(f"environment.json source aggregate {group} differs")
        source_sets[group] = digest

    deployment = exact_fields(
        document["deployment"],
        {"service_unit_file", "environment_file", "configuration"},
        "environment.json.deployment",
    )
    unit_file = _identity_file(
        deployment["service_unit_file"], "environment.json service unit"
    )
    environment_file = _identity_file(
        deployment["environment_file"], "environment.json service environment"
    )
    for file_value, role in (
        (unit_file, "systemd_service"),
        (environment_file, "systemd_environment_contract"),
    ):
        if (
            not Path(file_value["path"]).is_absolute()
            or file_value["bytes"] != by_role[role]["bytes"]
            or file_value["sha256"] != by_role[role]["sha256"]
        ):
            fail("environment.json effective deployment differs from tracked source")
    configuration = exact_fields(
        deployment["configuration"],
        {
            "worker_binary",
            "product_root",
            "tokenizer_root",
            "api_key_file",
            "gpu_lock_file",
            "bind_host",
            "bind_port",
            "hip_visible_devices",
            "hip_guards",
        },
        "environment.json deployment configuration",
    )
    for key in (
        "worker_binary",
        "product_root",
        "tokenizer_root",
        "api_key_file",
        "gpu_lock_file",
    ):
        path = string(configuration[key], f"environment.json configuration {key}")
        if not Path(path).is_absolute():
            fail(f"environment.json configuration {key} is not absolute")
    if (
        configuration["bind_host"] != DOCKER_NETWORK_GATEWAY
        or configuration["bind_port"] != 8000
        or configuration["hip_visible_devices"] != "1"
        or not json_equal(configuration["hip_guards"], list(HIP_GUARDS))
    ):
        fail("environment.json runtime network or HIP configuration differs")

    host = exact_fields(
        document["host"],
        {"os", "kernel", "boot_id", "cgroup_fs_type", "tools", "gpu"},
        "environment.json.host",
    )
    os_value = exact_fields(
        host["os"], {"id", "version_id", "pretty_name"}, "environment.json host OS"
    )
    for key, value in os_value.items():
        string(value, f"environment.json host OS {key}")
    kernel = exact_fields(
        host["kernel"],
        {"sysname", "release", "version", "machine"},
        "environment.json host kernel",
    )
    for key, value in kernel.items():
        string(value, f"environment.json host kernel {key}")
    boot_id = string(host["boot_id"], "environment.json host boot_id")
    if BOOT_ID_RE.fullmatch(boot_id) is None:
        fail("environment.json boot ID differs")
    if host["cgroup_fs_type"] != "cgroup2fs":
        fail("environment.json cgroup filesystem differs")
    tools_value = exact_fields(
        host["tools"],
        {
            "systemd_major",
            "systemd_version_line",
            "python_version_line",
            "rustc_version_line",
            "cargo_version_line",
            "docker_version",
            "docker_api_version",
            "docker_os",
            "docker_arch",
            "docker_kernel_version",
            "amd_smi_tool",
            "amd_smi_library",
            "rocm_version",
            "amd_smi_version_line",
        },
        "environment.json host tools",
    )
    systemd_major = integer(
        tools_value["systemd_major"], "environment.json systemd major", minimum=1
    )
    for key, value in tools_value.items():
        if key != "systemd_major":
            string(value, f"environment.json host tools {key}")
    if (
        systemd_major != SYSTEMD_MAJOR
        or not tools_value["systemd_version_line"].startswith(
            f"systemd {SYSTEMD_MAJOR}"
        )
        or not tools_value["python_version_line"].startswith("Python ")
        or not tools_value["rustc_version_line"].startswith("rustc ")
        or not tools_value["cargo_version_line"].startswith("cargo ")
        or tools_value["docker_os"] != "linux"
        or tools_value["docker_kernel_version"] != kernel["release"]
    ):
        fail("environment.json tool or host kernel identity differs")
    gpu = exact_fields(
        host["gpu"],
        {
            "index",
            "bdf",
            "uuid",
            "kfd_gpu_id",
            "node_id",
            "partition_id",
            "architecture",
        },
        "environment.json host GPU",
    )
    integer(gpu["index"], "environment.json GPU index")
    bdf = string(gpu["bdf"], "environment.json GPU BDF")
    uuid = string(gpu["uuid"], "environment.json GPU UUID")
    if BDF_RE.fullmatch(bdf) is None or UUID_RE.fullmatch(uuid) is None:
        fail("environment.json GPU BDF or UUID differs")
    integer(gpu["kfd_gpu_id"], "environment.json KFD GPU ID", minimum=1)
    integer(gpu["node_id"], "environment.json GPU node ID")
    integer(gpu["partition_id"], "environment.json GPU partition ID")
    if not json_equal(gpu, EXPECTED_GPU_IDENTITY):
        fail("environment.json frozen GPU identity differs")

    service = exact_fields(
        document["service"],
        {
            "unit",
            "user",
            "group",
            "uid",
            "gid",
            "fragment_path",
            "control_group",
            "gateway",
            "worker",
            "n_restarts",
            "active_state",
            "sub_state",
        },
        "environment.json.service",
    )
    if (
        service["unit"] != SERVICE_UNIT
        or service["active_state"] != "active"
        or service["sub_state"] != "running"
        or service["fragment_path"] != unit_file["path"]
        or service["control_group"] != f"/system.slice/{SERVICE_UNIT}"
    ):
        fail("environment.json service state or deployment path differs")
    string(service["user"], "environment.json service user")
    string(service["group"], "environment.json service group")
    uid = integer(service["uid"], "environment.json service UID")
    gid = integer(service["gid"], "environment.json service GID")
    integer(service["n_restarts"], "environment.json service restart count")
    gateway = _identity_process(service["gateway"], "environment.json gateway")
    worker = _identity_process(service["worker"], "environment.json worker")
    if (
        gateway["ppid"] != 1
        or gateway["uid"] != uid
        or gateway["gid"] != gid
        or gateway["children"] != [worker["pid"]]
        or worker["ppid"] != gateway["pid"]
        or worker["uid"] != uid
        or worker["gid"] != gid
        or worker["children"] != []
        or Path(worker["executable"]).name != "ullm-sq8-worker"
        or worker["executable"] != configuration["worker_binary"]
    ):
        fail("environment.json gateway/worker process relationship differs")

    openwebui = exact_fields(
        document["openwebui"],
        {
            "version",
            "source_revision",
            "base_image_digest",
            "base_image_id",
            "derived_image_id",
            "Dockerfile_sha256",
            "patch_sha256",
            "patched_middleware_sha256",
            "network_name",
            "network_id",
            "network_subnet",
            "network_gateway",
        },
        "environment.json.openwebui",
    )
    string(openwebui["version"], "environment.json OpenWebUI version")
    string(openwebui["source_revision"], "environment.json OpenWebUI revision")
    for key in ("base_image_digest", "base_image_id", "derived_image_id"):
        image_id = string(openwebui[key], f"environment.json OpenWebUI {key}")
        if IMAGE_ID_RE.fullmatch(image_id) is None:
            fail(f"environment.json OpenWebUI {key} differs")
    for key in ("Dockerfile_sha256", "patch_sha256", "patched_middleware_sha256"):
        sha256_value(openwebui[key], f"environment.json OpenWebUI {key}")
    network_id = string(
        openwebui["network_id"], "environment.json OpenWebUI network ID"
    )
    if (
        openwebui["version"] != OPENWEBUI_VERSION
        or openwebui["source_revision"] != OPENWEBUI_SOURCE_REVISION
        or openwebui["base_image_digest"] != OPENWEBUI_BASE_IMAGE_DIGEST
        or openwebui["base_image_id"] != OPENWEBUI_BASE_IMAGE_ID
        or openwebui["derived_image_id"] != OPENWEBUI_DERIVED_IMAGE_ID
        or openwebui["patched_middleware_sha256"] != OPENWEBUI_PATCHED_MIDDLEWARE_SHA256
        or openwebui["Dockerfile_sha256"] != by_role["openwebui_dockerfile"]["sha256"]
        or openwebui["patch_sha256"] != by_role["openwebui_patch"]["sha256"]
        or openwebui["network_name"] != DOCKER_NETWORK_NAME
        or network_id != DOCKER_NETWORK_ID
        or openwebui["network_subnet"] != DOCKER_NETWORK_SUBNET
        or openwebui["network_gateway"] != DOCKER_NETWORK_GATEWAY
    ):
        fail("environment.json OpenWebUI source or network identity differs")
    return by_role, source_sets, configuration, service, openwebui


def _validate_model_identity(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    exact_fields(
        document,
        {
            "schema_version",
            "record_type",
            "model",
            "promotion_validation",
            "product",
            "tokenizer",
            "oracle",
            "worker",
        },
        "model-identity.json",
    )
    if (
        document["schema_version"] != MODEL_IDENTITY_SCHEMA
        or document["record_type"] != "model_identity"
        or not json_equal(
            document["model"],
            {
                "upstream_id": UPSTREAM_MODEL_ID,
                "served_id": SERVED_MODEL_ID,
                "revision": MODEL_REVISION,
            },
        )
    ):
        fail("model-identity.json schema, record type, or model differs")
    receipt = exact_fields(
        document["promotion_validation"],
        {
            "schema_version",
            "result_sha256",
            "validator_source_sha256",
            "canonical_source_sha256",
            "full_payloads",
            "read_only",
            "verified",
        },
        "model-identity.json promotion validation",
    )
    if (
        receipt["schema_version"] != PROMOTION_SCHEMA
        or receipt["full_payloads"] is not True
        or receipt["read_only"] is not True
        or receipt["verified"] is not True
    ):
        fail("model-identity.json promotion validation state differs")
    sha256_value(receipt["result_sha256"], "model-identity.json promotion result SHA")
    sha256_value(
        receipt["validator_source_sha256"],
        "model-identity.json promotion validator source SHA",
    )
    sha256_value(
        receipt["canonical_source_sha256"],
        "model-identity.json promotion canonical source SHA",
    )

    product = exact_fields(
        document["product"],
        {"root", "promotion", "artifact", "package"},
        "model-identity.json product",
    )
    product_root = string(product["root"], "model-identity.json product root")
    if not Path(product_root).is_absolute():
        fail("model-identity.json product root is not absolute")
    promotion = exact_fields(
        product["promotion"],
        {"file", "bytes", "sha256", "created_at", "plan_commit"},
        "model-identity.json promotion",
    )
    if (
        promotion["file"] != "promotion.json"
        or promotion["plan_commit"] != PROMOTION_PLAN_COMMIT
    ):
        fail("model-identity.json promotion file or plan commit differs")
    integer(promotion["bytes"], "model-identity.json promotion bytes", minimum=1)
    sha256_value(promotion["sha256"], "model-identity.json promotion SHA")
    _identity_timestamp(promotion["created_at"], "model-identity.json promotion time")
    artifact = exact_fields(
        product["artifact"],
        set(EXPECTED_ARTIFACT_IDENTITY),
        "model-identity.json artifact",
    )
    package = exact_fields(
        product["package"],
        set(EXPECTED_PACKAGE_IDENTITY),
        "model-identity.json package",
    )
    if not json_equal(artifact, EXPECTED_ARTIFACT_IDENTITY):
        fail("model-identity.json fixed artifact identity differs")
    if not json_equal(package, EXPECTED_PACKAGE_IDENTITY):
        fail("model-identity.json fixed package identity differs")

    expected_receipt = {
        "schema_version": PROMOTION_SCHEMA,
        "product_root": product_root,
        "created_at": promotion["created_at"],
        "model_revision": MODEL_REVISION,
        "artifact": {
            "manifest_sha256": artifact["manifest_sha256"],
            "content_sha256": artifact["content_sha256"],
            "selected_pair_count": artifact["selected_pair_count"],
            "payloads_hashed": artifact["payloads_hashed"],
        },
        "package": {
            "manifest_sha256": package["manifest_sha256"],
            "payload_count": package["payload_count"],
            "payload_bytes": package["payload_bytes"],
            "payloads_hashed": package["payloads_hashed"],
        },
        "read_only": receipt["read_only"],
        "full_payloads": receipt["full_payloads"],
        "verified": receipt["verified"],
    }
    expected_receipt_sha = hashlib.sha256(
        _identity_canonical_bytes(expected_receipt)
    ).hexdigest()
    if receipt["result_sha256"] != expected_receipt_sha:
        fail("model-identity.json promotion receipt SHA-256 differs")

    tokenizer = exact_fields(
        document["tokenizer"],
        {"root", "revision", "aggregate_sha256", "chat_template", "files"},
        "model-identity.json tokenizer",
    )
    tokenizer_root = string(tokenizer["root"], "model-identity.json tokenizer root")
    if (
        not Path(tokenizer_root).is_absolute()
        or tokenizer["revision"] != MODEL_REVISION
    ):
        fail("model-identity.json tokenizer root or revision differs")
    files = tokenizer["files"]
    if type(files) is not list or len(files) != len(EXPECTED_TOKENIZER_FILES):
        fail("model-identity.json tokenizer file set differs")
    parsed_files: list[dict[str, Any]] = []
    for index, (raw, (path, byte_count, digest)) in enumerate(
        zip(files, EXPECTED_TOKENIZER_FILES, strict=True)
    ):
        item = _identity_file(
            raw,
            f"model-identity.json tokenizer file {index}",
            expected_path=path,
        )
        if item["bytes"] != byte_count or item["sha256"] != digest:
            fail(f"model-identity.json tokenizer file {path} differs")
        parsed_files.append(item)
    aggregate = sha256_value(
        tokenizer["aggregate_sha256"], "model-identity.json tokenizer aggregate"
    )
    expected_aggregate = hashlib.sha256(
        _identity_canonical_bytes(parsed_files)
    ).hexdigest()
    if aggregate != expected_aggregate:
        fail("model-identity.json tokenizer aggregate differs")
    chat_template = exact_fields(
        tokenizer["chat_template"],
        set(EXPECTED_CHAT_TEMPLATE_IDENTITY),
        "model-identity.json tokenizer chat template",
    )
    if not json_equal(chat_template, EXPECTED_CHAT_TEMPLATE_IDENTITY):
        fail("model-identity.json tokenizer chat template differs")

    oracle = exact_fields(
        document["oracle"],
        {
            "serving_fixture_manifest",
            "chat_template_fixture_manifest",
            "runtime_oracle_validation",
            "vllm_identity",
        },
        "model-identity.json oracle",
    )
    for role, expected in EXPECTED_ORACLE_FILE_IDENTITIES.items():
        item = _identity_file(
            oracle[role],
            f"model-identity.json oracle {role}",
            expected_path=expected["path"],
        )
        if not json_equal(item, expected):
            fail(f"model-identity.json oracle {role} differs")
    if not json_equal(oracle["vllm_identity"], EXPECTED_VLLM_IDENTITY):
        fail("model-identity.json vLLM oracle identity differs")

    worker = exact_fields(
        document["worker"],
        {
            "binary",
            "binary_bytes",
            "binary_sha256",
            "source_sha256",
            "protocol_schema",
            "device_architecture",
            "execution_profile",
            "context_length",
            "max_completion_tokens",
            "vocab_size",
            "model_revision",
            "artifact_content_sha256",
            "package_manifest_sha256",
        },
        "model-identity.json worker",
    )
    binary = string(worker["binary"], "model-identity.json worker binary")
    if not Path(binary).is_absolute() or Path(binary).name != "ullm-sq8-worker":
        fail("model-identity.json worker binary path differs")
    integer(
        worker["binary_bytes"], "model-identity.json worker binary bytes", minimum=1
    )
    sha256_value(worker["binary_sha256"], "model-identity.json worker binary SHA")
    sha256_value(worker["source_sha256"], "model-identity.json worker source SHA")
    expected_worker = {
        "protocol_schema": WORKER_PROTOCOL_SCHEMA,
        "device_architecture": DEVICE_ARCHITECTURE,
        "execution_profile": EXECUTION_PROFILE,
        "context_length": CONTEXT_LENGTH,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "vocab_size": VOCAB_SIZE,
        "model_revision": MODEL_REVISION,
        "artifact_content_sha256": artifact["content_sha256"],
        "package_manifest_sha256": package["manifest_sha256"],
    }
    if any(
        not json_equal(worker[key], value) for key, value in expected_worker.items()
    ):
        fail("model-identity.json worker contract differs")
    return product, tokenizer, oracle, worker


@dataclass(frozen=True)
class IdentityData:
    environment: dict[str, Any]
    model_identity: dict[str, Any]
    environment_sha256: str
    model_identity_sha256: str
    expected_commit: str
    expected_worker_binary_sha256: str
    source_by_role: dict[str, dict[str, Any]]
    source_sets: dict[str, str]
    configuration: dict[str, Any]
    service: dict[str, Any]
    openwebui: dict[str, Any]
    model_worker: dict[str, Any]

    def validate_session_header(self, record: dict[str, Any]) -> None:
        if record.get("boot_id") != self.environment["host"]["boot_id"]:
            fail("raw-session header boot ID differs from environment.json")
        started = _identity_timestamp(
            record.get("started_utc"),
            "raw-session header.started_utc",
            require_utc_z=True,
        )
        captured = _identity_timestamp(
            self.environment["captured_utc"],
            "environment.json.captured_utc",
            require_utc_z=True,
        )
        if started < captured:
            fail("raw-session header predates environment capture")
        identities = exact_fields(
            record.get("identities"),
            {
                "environment_file",
                "environment_sha256",
                "model_identity_file",
                "model_identity_sha256",
                "openwebui",
                "docker_network_id",
                "gateway_source_sha256",
                "worker_source_sha256",
                "worker_binary_sha256",
            },
            "raw-session header.identities",
        )
        expected_openwebui = {
            key: self.openwebui[key]
            for key in (
                "version",
                "source_revision",
                "base_image_digest",
                "base_image_id",
                "derived_image_id",
                "Dockerfile_sha256",
                "patch_sha256",
                "patched_middleware_sha256",
            )
        }
        expected = {
            "environment_file": "environment.json",
            "environment_sha256": self.environment_sha256,
            "model_identity_file": "model-identity.json",
            "model_identity_sha256": self.model_identity_sha256,
            "openwebui": expected_openwebui,
            "docker_network_id": self.openwebui["network_id"],
            "gateway_source_sha256": self.source_sets["gateway"],
            "worker_source_sha256": self.source_sets["worker"],
            "worker_binary_sha256": self.expected_worker_binary_sha256,
        }
        if not json_equal(identities, expected):
            fail("raw-session header identities differ from campaign identity")

    def validate_initial_probe(self, record: dict[str, Any]) -> None:
        gateway = self.service["gateway"]
        worker = self.service["worker"]
        expected = {
            "service_active": True,
            "ready_http_status": 200,
            "control_group": self.service["control_group"],
            "gateway_pid": gateway["pid"],
            "gateway_starttime_ticks": gateway["starttime_ticks"],
            "worker_pid": worker["pid"],
            "worker_starttime_ticks": worker["starttime_ticks"],
            "n_restarts": self.service["n_restarts"],
        }
        if any(
            not json_equal(record.get(key), value) for key, value in expected.items()
        ):
            fail("initial lifecycle probe differs from environment.json")

    def validate_run_end(self, record: dict[str, Any]) -> None:
        if record.get("final_git_commit") != self.expected_commit:
            fail("run_end Git commit differs from campaign identity")
        status = string(
            record.get("final_git_status_raw"),
            "run_end.final_git_status_raw",
            nonempty=False,
        )
        digest = hashlib.sha256(status.encode("utf-8", errors="strict")).hexdigest()
        if (
            record.get("final_git_status_sha256") != digest
            or digest != self.environment["git"]["status_sha256"]
            or bool(status) != self.environment["git"]["dirty"]
        ):
            fail("run_end Git status differs from environment.json")

    def validate_header_source_inputs(self, input_files: Any) -> None:
        if type(input_files) is not list:
            fail("raw-session header input_files must be an array")
        by_path: dict[str, dict[str, Any]] = {}
        ordered_paths: list[str] = []
        for index, raw in enumerate(input_files):
            label = f"raw-session header.input_files[{index}]"
            item = exact_fields(raw, {"path", "bytes", "sha256"}, label)
            path = string(item["path"], f"{label}.path")
            pure = PurePosixPath(path)
            if (
                pure.is_absolute()
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or "\\" in path
                or path in by_path
            ):
                fail("raw-session header input source path differs")
            integer(item["bytes"], f"{label}.bytes", minimum=1)
            sha256_value(item["sha256"], f"{label}.sha256")
            by_path[path] = item
            ordered_paths.append(path)
        if ordered_paths != sorted(
            ordered_paths, key=lambda value: value.encode("utf-8")
        ):
            fail("raw-session header input sources are not bytewise sorted")
        for source in self.source_by_role.values():
            bound = by_path.get(source["path"])
            if bound is None or any(
                bound[key] != source[key] for key in ("path", "bytes", "sha256")
            ):
                fail("raw-session header lacks an exact campaign source input")


def validate_campaign_identity(
    bundle: Path,
    *,
    expected_commit: str,
    expected_worker_binary_sha256: str,
) -> IdentityData:
    trusted_commit = git_commit(expected_commit, "expected campaign Git commit")
    trusted_worker_sha = sha256_value(
        expected_worker_binary_sha256, "expected campaign worker binary SHA-256"
    )
    root = safe_bundle_root(bundle)
    environment, environment_raw = _read_canonical_identity(
        root, "environment.json", "environment.json"
    )
    model_identity, model_raw = _read_canonical_identity(
        root, "model-identity.json", "model-identity.json"
    )
    source_by_role, source_sets, configuration, service, openwebui = (
        _validate_environment_identity(environment, trusted_commit)
    )
    product, tokenizer, oracle, model_worker = _validate_model_identity(model_identity)

    service_worker = service["worker"]
    if (
        configuration["product_root"] != product["root"]
        or configuration["tokenizer_root"] != tokenizer["root"]
        or configuration["worker_binary"] != model_worker["binary"]
        or service_worker["executable"] != model_worker["binary"]
        or service_worker["executable_bytes"] != model_worker["binary_bytes"]
        or service_worker["executable_sha256"] != model_worker["binary_sha256"]
        or model_worker["binary_sha256"] != trusted_worker_sha
        or model_worker["source_sha256"] != source_sets["worker"]
    ):
        fail("environment/model worker or runtime binding differs")
    promotion_sources = {
        "validator_source_sha256": "product_promotion_validator",
        "canonical_source_sha256": "product_promotion_canonical",
    }
    for field, role in promotion_sources.items():
        if (
            model_identity["promotion_validation"][field]
            != source_by_role[role]["sha256"]
        ):
            source_label = role.removeprefix("product_promotion_").replace("_", " ")
            fail(f"promotion {source_label} source differs from environment.json")
    for role in EXPECTED_ORACLE_FILE_IDENTITIES:
        source = source_by_role[role]
        model_oracle = oracle[role]
        if any(source[key] != model_oracle[key] for key in ("path", "bytes", "sha256")):
            fail(f"environment/model oracle source {role} differs")
    return IdentityData(
        environment=environment,
        model_identity=model_identity,
        environment_sha256=hashlib.sha256(environment_raw).hexdigest(),
        model_identity_sha256=hashlib.sha256(model_raw).hexdigest(),
        expected_commit=trusted_commit,
        expected_worker_binary_sha256=trusted_worker_sha,
        source_by_role=source_by_role,
        source_sets=source_sets,
        configuration=configuration,
        service=service,
        openwebui=openwebui,
        model_worker=model_worker,
    )


@dataclass(frozen=True)
class SourceCheckoutData:
    git_commit: str
    source_count: int
    all_source_sha256: str


def validate_campaign_source_checkout(
    identity: IdentityData, *, repo_root: Path
) -> SourceCheckoutData:
    """Bind recorded source entries to one Git commit and its current worktree files."""

    if not isinstance(identity, IdentityData) or not isinstance(repo_root, os.PathLike):
        fail("campaign source checkout arguments differ")
    _validate_identity_source_contract()
    root = Path(os.path.abspath(repo_root))
    root_fd = -1
    try:
        root_fd = os.open(root, _source_directory_flags())
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            fail("campaign source checkout is not a directory")
        top_level = _run_bounded_source_command(
            (
                "git",
                "--no-pager",
                "-C",
                os.fspath(root),
                "rev-parse",
                "--show-toplevel",
            ),
            maximum=MAX_GIT_CONTROL_OUTPUT_BYTES,
            retain=True,
        )
        if top_level.raw is None:
            fail("campaign Git top-level output is unavailable")
            raise AssertionError("unreachable")
        if top_level.raw != os.fsencode(root) + b"\n":
            fail("campaign source checkout is not the Git top level")
        resolved_commit = _run_bounded_source_command(
            (
                "git",
                "--no-pager",
                "-C",
                os.fspath(root),
                "rev-parse",
                "--verify",
                f"{identity.expected_commit}^{{commit}}",
            ),
            maximum=MAX_GIT_CONTROL_OUTPUT_BYTES,
            retain=True,
        )
        if resolved_commit.raw is None:
            fail("trusted campaign Git commit output is unavailable")
            raise AssertionError("unreachable")
        if resolved_commit.raw != identity.expected_commit.encode("ascii") + b"\n":
            fail("trusted campaign Git commit does not resolve exactly")

        if set(identity.source_by_role) != set(EXPECTED_SOURCE_ROLE_PATHS):
            fail("campaign source identity role set differs")
        git_entries: dict[str, dict[str, Any]] = {}
        for role, relative in EXPECTED_SOURCE_ROLE_PATHS.items():
            source = identity.source_by_role[role]
            if set(source) != {"role", "path", "bytes", "sha256"}:
                fail("campaign source identity fields differ")
            git_digest = _git_source_digest(root, identity.expected_commit, relative)
            worktree_digest = _hash_worktree_source(root_fd, relative)
            expected = {
                "role": role,
                "path": relative,
                "bytes": git_digest.byte_count,
                "sha256": git_digest.sha256,
            }
            if source != expected or worktree_digest != git_digest:
                fail("campaign source differs from its trusted Git blob")
            git_entries[role] = expected

        if set(identity.source_sets) != set(EXPECTED_SOURCE_GROUPS):
            fail("campaign source aggregate set differs")
        for group, roles in EXPECTED_SOURCE_GROUPS.items():
            if identity.source_sets[group] != _identity_source_aggregate(
                git_entries, roles
            ):
                fail("campaign source aggregate differs from trusted Git blobs")
        return SourceCheckoutData(
            git_commit=identity.expected_commit,
            source_count=len(git_entries),
            all_source_sha256=identity.source_sets["all"],
        )
    except ValidationError:
        raise
    except OSError:
        fail("failed to validate the campaign source checkout")
        raise AssertionError("unreachable")
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def median(values: Iterable[int | Fraction]) -> Fraction:
    converted: list[Fraction] = []
    for value in values:
        if type(value) is int:
            converted.append(Fraction(value))
        elif type(value) is Fraction:
            converted.append(value)
        else:
            fail("median input contains a non-exact or non-finite value")
    ordered = sorted(converted)
    if not ordered:
        fail("median input must not be empty")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: Iterable[int | Fraction], probability: Fraction) -> Fraction:
    converted: list[Fraction] = []
    for value in values:
        if type(value) is int:
            converted.append(Fraction(value))
        elif type(value) is Fraction:
            converted.append(value)
        else:
            fail("percentile input contains a non-exact or non-finite value")
    ordered = sorted(converted)
    if (
        not ordered
        or type(probability) is not Fraction
        or probability < 0
        or probability > 1
    ):
        fail("percentile input or probability is invalid")
    rank = Fraction(len(ordered) - 1) * probability
    lower = rank.numerator // rank.denominator
    upper = lower if rank.denominator == 1 else lower + 1
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def theil_sen(values: list[Fraction]) -> Fraction:
    if len(values) < 2:
        fail("Theil-Sen input must contain at least two points")
    if any(type(value) is not Fraction for value in values):
        fail("Theil-Sen input contains a non-exact or non-finite value")
    slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    return median(slopes)


def fraction_json(value: Fraction) -> int | dict[str, int]:
    if value.denominator == 1:
        return value.numerator
    return {"numerator": value.numerator, "denominator": value.denominator}


def decode_base64(value: Any, label: str) -> bytes:
    text = string(value, label, nonempty=False)
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        fail(f"{label} is not canonical base64: {error}")


def validate_schedule(value: Any, label: str) -> dict[str, Any]:
    expected = SCHEDULE
    exact_fields(value, set(expected), label)
    if not json_equal(value, expected):
        fail(f"{label} differs from the frozen release schedule")
    return value


def validate_thresholds(value: Any, label: str) -> dict[str, Any]:
    exact_fields(value, set(THRESHOLDS), label)
    ttft = exact_fields(
        value["ttft_seconds_maximum"], set(FIXTURE_IDS), f"{label}.ttft_seconds_maximum"
    )
    for fixture_id in FIXTURE_IDS:
        exact_fields(
            ttft[fixture_id],
            {"p50", "p95"},
            f"{label}.ttft_seconds_maximum.{fixture_id}",
        )
    if not json_equal(value, THRESHOLDS):
        fail(f"{label} differs from the frozen release thresholds")
    return value


@dataclass(frozen=True)
class MatrixData:
    run_id: str
    schedule: dict[str, Any]
    thresholds: dict[str, Any]


def validate_bundle_layout(root: Path) -> None:
    actual_files: set[str] = set()
    saw_browser = False
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name == "browser":
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        fail(
                            "bundle browser entry must be a regular non-symlink directory"
                        )
                    saw_browser = True
                    with os.scandir(entry.path) as browser_entries:
                        for browser_entry in browser_entries:
                            relative = f"browser/{browser_entry.name}"
                            if browser_entry.is_symlink() or not browser_entry.is_file(
                                follow_symlinks=False
                            ):
                                fail(
                                    f"bundle contains a non-regular file or symlink: {relative}"
                                )
                            if relative not in BUNDLE_FILES:
                                fail(
                                    f"bundle contains an extra evidence file: {relative}"
                                )
                            actual_files.add(relative)
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    fail(f"bundle contains a non-regular file or symlink: {entry.name}")
                if entry.name not in BUNDLE_FILES:
                    fail(f"bundle contains an extra evidence file: {entry.name}")
                actual_files.add(entry.name)
    except ValidationError:
        raise
    except OSError as error:
        fail(f"failed to enumerate bundle layout: {error}")
    if not saw_browser:
        fail("bundle lacks the browser evidence directory")
    if actual_files != BUNDLE_FILES:
        fail(
            f"bundle file set differs: missing={sorted(BUNDLE_FILES - actual_files)} "
            f"extra={sorted(actual_files - BUNDLE_FILES)}"
        )
    if (root / "release-validation.json").exists() or (
        root / "release-validation.json"
    ).is_symlink():
        fail("release-validation.json must be absent before validation")


def validate_sha256sums(root: Path) -> str:
    path = safe_relative_file(root, "SHA256SUMS", "SHA256SUMS")
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii", errors="strict")
    except (OSError, UnicodeError) as error:
        fail(f"failed to read SHA256SUMS: {error}")
    if not text or not text.endswith("\n") or "\r" in text:
        fail("SHA256SUMS must be non-empty LF-terminated ASCII")
    expected_paths = sorted(
        BUNDLE_FILES - {"SHA256SUMS"}, key=lambda item: item.encode("utf-8")
    )
    lines = text.splitlines()
    if len(lines) != len(expected_paths):
        fail("SHA256SUMS entry count differs")
    observed_paths: list[str] = []
    for index, (line, expected_path) in enumerate(
        zip(lines, expected_paths, strict=True), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  ([!-~]+)", line)
        if match is None:
            fail(f"SHA256SUMS line {index} is invalid")
        digest, relative = match.groups()
        observed_paths.append(relative)
        if relative != expected_path:
            fail(
                f"SHA256SUMS paths are not exact bytewise ascending paths at line {index}"
            )
        artifact = safe_relative_file(root, relative, f"SHA256SUMS artifact {relative}")
        if sha256_file(artifact) != digest:
            fail(f"SHA256SUMS digest mismatch for {relative}")
    return hashlib.sha256(raw).hexdigest()


def validate_matrix(root: Path) -> MatrixData:
    path = safe_relative_file(root, "release-matrix.json", "release-matrix.json")
    matrix = read_json(path, "release-matrix.json")
    reject_key_recursive(matrix, "passed", "release-matrix.json")
    exact_fields(
        matrix,
        {"schema_version", "run_id", "files", "schedule", "thresholds"},
        "release-matrix.json",
    )
    if matrix["schema_version"] != MATRIX_SCHEMA:
        fail("release-matrix.json schema_version differs")
    run_id = string(matrix["run_id"], "release-matrix.json.run_id")
    files = matrix["files"]
    if type(files) is not list or len(files) != len(EXPECTED_ROLES):
        fail("release-matrix.json.files has the wrong cardinality")
    paths: list[str] = []
    for index, entry in enumerate(files):
        label = f"release-matrix.json.files[{index}]"
        exact_fields(entry, {"role", "path", "bytes", "sha256"}, label)
        relative = string(entry["path"], f"{label}.path")
        if relative not in EXPECTED_ROLES:
            fail(f"{label}.path is not a defined matrix input")
        if entry["role"] != EXPECTED_ROLES[relative]:
            fail(f"{label}.role differs for {relative}")
        size = integer(entry["bytes"], f"{label}.bytes")
        digest = sha256_value(entry["sha256"], f"{label}.sha256")
        artifact = safe_relative_file(root, relative, f"matrix input {relative}")
        if artifact.stat().st_size != size:
            fail(f"matrix size differs for {relative}")
        if sha256_file(artifact) != digest:
            fail(f"matrix SHA-256 differs for {relative}")
        paths.append(relative)
    expected_paths = sorted(EXPECTED_ROLES, key=lambda item: item.encode("utf-8"))
    if paths != expected_paths:
        fail(
            "release-matrix.json.files paths are not exact bytewise ascending unique paths"
        )
    schedule = validate_schedule(matrix["schedule"], "release-matrix.json.schedule")
    thresholds = validate_thresholds(
        matrix["thresholds"], "release-matrix.json.thresholds"
    )
    return MatrixData(run_id=run_id, schedule=schedule, thresholds=thresholds)


@dataclass
class RequestTrace:
    phase: str
    case_id: str
    completion_id: str
    events: list[dict[str, Any]]
    terminal: str | None = None


@dataclass(frozen=True)
class GatewayEvidence:
    cursor: str
    journal_monotonic_usec: int
    journal_pid: int
    message: str
    message_sha256: str
    event: dict[str, Any]
    phase: str


@dataclass(frozen=True)
class InputSeal:
    size: int
    sha256: str


@dataclass(frozen=True)
class HttpSseItem:
    chunk_index: int
    observed_monotonic_ns: int
    done: bool
    completion_id_utf8_bytes: int | None
    completion_id_sha256: str | None
    content_utf8_bytes: int | None
    content_sha256: str | None
    finish_reason: str | None
    usage_present: bool
    usage_is_object: bool | None
    completion_tokens: int | None


@dataclass(frozen=True)
class HttpSseMetadata:
    chunk_count: int
    first_chunk_monotonic_ns: int | None
    last_chunk_monotonic_ns: int | None
    items: tuple[HttpSseItem, ...]


@dataclass(frozen=True)
class HttpCompactResult:
    phase: str
    case_id: str
    request_index: int
    request_key: str
    method: str
    target: str
    status: int
    outcome: str
    request_body_bytes: int
    request_body_sha256: str
    response_body_bytes: int
    response_body_sha256: str
    connect_completed_monotonic_ns: int
    write_started_monotonic_ns: int
    last_body_byte_sent_monotonic_ns: int
    response_started_monotonic_ns: int
    response_end_monotonic_ns: int
    sse: HttpSseMetadata | None


@dataclass(frozen=True)
class BrowserActionData:
    phase: str
    case_id: str
    browser_case: str
    action_index: int
    action: str
    selector: str | None
    input_sha256: str | None
    started_monotonic_ns: int
    completed_monotonic_ns: int
    result_visible: bool | None
    result_enabled: bool | None
    result_text_utf8_bytes: int | None
    result_text_sha256: str | None
    screenshot_file: str | None
    screenshot_sha256: str | None


@dataclass(frozen=True)
class FaultInjectionData:
    phase: str
    case_id: str
    injection: str
    target_pid: int
    target_starttime_ticks: int
    signal: str
    command_utf8_bytes: int
    command_sha256: str
    started_monotonic_ns: int
    completed_monotonic_ns: int


class _CompactSseParser:
    """Retain only bounded timing and semantic metadata from one SSE response."""

    def __init__(self) -> None:
        self.line = bytearray()
        self.data_lines: list[bytes] = []
        self.items: list[HttpSseItem] = []
        self.previous_cr = False
        self.event_bytes = 0
        self.last_chunk_index = -1
        self.last_timestamp = -1
        self.first_timestamp: int | None = None

    def feed(self, raw: bytes, chunk_index: int, observed_ns: int) -> None:
        if (
            chunk_index != self.last_chunk_index + 1
            or observed_ns < self.last_timestamp
        ):
            fail("compact SSE chunk identity or timestamps regress")
        if self.first_timestamp is None:
            self.first_timestamp = observed_ns
        self.last_chunk_index = chunk_index
        self.last_timestamp = observed_ns
        for byte in raw:
            if self.previous_cr:
                self.previous_cr = False
                if byte == 0x0A:
                    continue
            if byte == 0x0D:
                self._finish_line(chunk_index, observed_ns)
                self.previous_cr = True
            elif byte == 0x0A:
                self._finish_line(chunk_index, observed_ns)
            else:
                self.line.append(byte)
                if len(self.line) > MAX_SSE_LINE_BYTES:
                    fail("compact SSE line exceeds its size bound")

    def finish(self, *, allow_incomplete: bool) -> HttpSseMetadata:
        self.previous_cr = False
        if self.line:
            if self.last_chunk_index < 0:
                fail("compact SSE response lacks a final chunk")
            self._finish_line(self.last_chunk_index, self.last_timestamp)
        if self.data_lines:
            if allow_incomplete:
                self.data_lines.clear()
                self.event_bytes = 0
            else:
                self._dispatch(self.last_chunk_index, self.last_timestamp)
        return HttpSseMetadata(
            chunk_count=self.last_chunk_index + 1,
            first_chunk_monotonic_ns=self.first_timestamp,
            last_chunk_monotonic_ns=(
                None if self.last_chunk_index < 0 else self.last_timestamp
            ),
            items=tuple(self.items),
        )

    def _finish_line(self, chunk_index: int, observed_ns: int) -> None:
        line = bytes(self.line)
        self.line.clear()
        if not line:
            self._dispatch(chunk_index, observed_ns)
            return
        if line.startswith(b":"):
            return
        field_name, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field_name == b"data":
            self.event_bytes += len(value) + (1 if self.data_lines else 0)
            if self.event_bytes > MAX_SSE_EVENT_BYTES:
                fail("compact SSE event exceeds its size bound")
            self.data_lines.append(value)

    def _dispatch(self, chunk_index: int, observed_ns: int) -> None:
        if not self.data_lines:
            self.event_bytes = 0
            return
        raw = b"\n".join(self.data_lines)
        self.data_lines.clear()
        self.event_bytes = 0
        if len(self.items) >= MAX_SSE_ITEMS_PER_RESPONSE:
            fail("compact SSE item count exceeds its bound")
        if raw == b"[DONE]":
            self.items.append(
                HttpSseItem(
                    chunk_index=chunk_index,
                    observed_monotonic_ns=observed_ns,
                    done=True,
                    completion_id_utf8_bytes=None,
                    completion_id_sha256=None,
                    content_utf8_bytes=None,
                    content_sha256=None,
                    finish_reason=None,
                    usage_present=False,
                    usage_is_object=None,
                    completion_tokens=None,
                )
            )
            return
        value = cast(dict[str, Any], decode_json_bytes(raw, "compact SSE data object"))
        completion_id_value = value.get("id")
        if completion_id_value is not None and type(completion_id_value) is not str:
            fail("compact SSE completion id is not a string")
        completion_id_raw = (
            completion_id_value.encode("utf-8")
            if type(completion_id_value) is str
            else None
        )
        content: str | None = None
        finish_reason: str | None = None
        choices = value.get("choices")
        if type(choices) is list and choices and type(choices[0]) is dict:
            first_choice = cast(dict[str, Any], choices[0])
            delta = first_choice.get("delta")
            if type(delta) is dict:
                content_value = delta.get("content")
                if content_value is not None and type(content_value) is not str:
                    fail("compact SSE content is not a string or null")
                if type(content_value) is str and content_value:
                    content = content_value
            finish_value = first_choice.get("finish_reason")
            if finish_value is not None:
                finish_reason = bounded_utf8_string(
                    finish_value,
                    "compact SSE finish_reason",
                    maximum_bytes=MAX_SSE_FINISH_REASON_BYTES,
                    nonempty=False,
                )
        usage_present = "usage" in value
        usage_value = value.get("usage")
        usage_is_object = type(usage_value) is dict if usage_present else None
        completion_tokens: int | None = None
        if type(usage_value) is dict and "completion_tokens" in usage_value:
            completion_tokens = integer(
                usage_value["completion_tokens"], "compact SSE completion_tokens"
            )
        content_raw = content.encode("utf-8") if content is not None else None
        self.items.append(
            HttpSseItem(
                chunk_index=chunk_index,
                observed_monotonic_ns=observed_ns,
                done=False,
                completion_id_utf8_bytes=(
                    len(completion_id_raw) if completion_id_raw is not None else None
                ),
                completion_id_sha256=(
                    hashlib.sha256(completion_id_raw).hexdigest()
                    if completion_id_raw is not None
                    else None
                ),
                content_utf8_bytes=(
                    len(content_raw) if content_raw is not None else None
                ),
                content_sha256=(
                    hashlib.sha256(content_raw).hexdigest()
                    if content_raw is not None
                    else None
                ),
                finish_reason=finish_reason,
                usage_present=usage_present,
                usage_is_object=usage_is_object,
                completion_tokens=completion_tokens,
            )
        )


@dataclass
class HttpBodyState:
    digest: Any
    byte_count: int
    next_index: int
    raw: bytearray
    last_observed_ns: int
    sse_parser: _CompactSseParser | None = None


@dataclass
class HttpValidationState:
    fixture_seal: InputSeal
    requests: dict[str, dict[str, Any]]
    response_started: set[str]
    response_ended: set[str]
    bodies: dict[str, HttpBodyState]
    ordered_keys: list[str]
    completed_results: dict[str, HttpCompactResult] = dataclass_field(
        default_factory=dict
    )
    active_key: str | None = None
    last_response_end_ns: int = -1
    fixture_model: str | None = None
    fixture_messages: list[Any] | None = None
    total_sse_items: int = 0


@dataclass(frozen=True)
class ApiContractCase:
    case_id: str
    method: str
    target: str
    body: bytes
    authorization_mode: str
    expected_status: int
    expected_code: str | None
    expected_param: str | None
    expected_message: str | None
    expect_models: bool = False


API_CONTRACT_CASES = (
    ApiContractCase(
        "models-valid",
        "GET",
        "/v1/models",
        b"",
        "valid_bearer",
        200,
        None,
        None,
        None,
        True,
    ),
    ApiContractCase(
        "models-missing-auth",
        "GET",
        "/v1/models",
        b"",
        "missing",
        401,
        "invalid_api_key",
        None,
        API_CONTRACT_INVALID_KEY_MESSAGE,
    ),
    ApiContractCase(
        "models-invalid-auth",
        "GET",
        "/v1/models",
        b"",
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
        API_CONTRACT_INVALID_KEY_MESSAGE,
    ),
    ApiContractCase(
        "models-query",
        "GET",
        "/v1/models?x=1",
        b"",
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        API_CONTRACT_QUERY_MESSAGE,
    ),
    ApiContractCase(
        "chat-malformed-missing-auth",
        "POST",
        "/v1/chat/completions",
        API_CONTRACT_MALFORMED_BODY,
        "missing",
        401,
        "invalid_api_key",
        None,
        API_CONTRACT_INVALID_KEY_MESSAGE,
    ),
    ApiContractCase(
        "chat-invalid-auth",
        "POST",
        "/v1/chat/completions",
        API_CONTRACT_CANONICAL_BODY,
        "invalid_bearer",
        401,
        "invalid_api_key",
        None,
        API_CONTRACT_INVALID_KEY_MESSAGE,
    ),
    ApiContractCase(
        "chat-malformed-valid-auth",
        "POST",
        "/v1/chat/completions",
        API_CONTRACT_MALFORMED_BODY,
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        API_CONTRACT_INVALID_JSON_MESSAGE,
    ),
    ApiContractCase(
        "chat-duplicate-key",
        "POST",
        "/v1/chat/completions",
        API_CONTRACT_DUPLICATE_KEY_BODY,
        "valid_bearer",
        400,
        "invalid_request_error",
        None,
        API_CONTRACT_INVALID_JSON_MESSAGE,
    ),
    ApiContractCase(
        "chat-unsupported-n",
        "POST",
        "/v1/chat/completions",
        API_CONTRACT_UNSUPPORTED_N_BODY,
        "valid_bearer",
        400,
        "unsupported_parameter",
        "n",
        API_CONTRACT_UNSUPPORTED_MESSAGE,
    ),
    ApiContractCase(
        "chat-missing-model",
        "POST",
        "/v1/chat/completions",
        API_CONTRACT_MISSING_MODEL_BODY,
        "valid_bearer",
        404,
        "model_not_found",
        "model",
        API_CONTRACT_MODEL_NOT_FOUND_MESSAGE,
    ),
)


@dataclass(frozen=True)
class ApiContractValidationResult:
    case_ids: tuple[str, ...]
    request_keys: tuple[str, ...]
    statuses: tuple[int, ...]
    cases: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LifecycleQuietCheckData:
    phase: str
    case_id: str
    quiet_sequence: int
    label: str
    checked_monotonic_ns: int
    observer_open: bool
    observer_event_count: int
    new_journal_record_count: int
    journal_record_count: int
    journal_cursor: str


@dataclass(frozen=True)
class ApiJournalObservationData:
    phase: str
    case_id: str
    observation_index: int
    journal_cursor: str
    journal_monotonic_usec: int
    journal_pid: int
    message_utf8_bytes: int
    message_sha256: str


@dataclass
class SessionData:
    run_id: str
    boot_id: str
    schedule: dict[str, Any]
    thresholds: dict[str, Any]
    traces: dict[str, RequestTrace]
    releases_by_phase: dict[str, list[dict[str, Any]]]
    journal_events: dict[str, GatewayEvidence]
    final_journal_cursor: str
    record_counts: Counter[str]
    http_requests: dict[str, dict[str, Any]]
    ordered_http_keys: list[str]
    probes: dict[str, dict[str, Any]]
    raw_order_projection: tuple[dict[str, Any], ...]
    http_results: tuple[HttpCompactResult, ...]
    browser_actions: tuple[BrowserActionData, ...]
    api_journal_observations: tuple[ApiJournalObservationData, ...]
    lifecycle_quiet_checks: tuple[LifecycleQuietCheckData, ...]
    fault_injection: FaultInjectionData | None
    full_campaign_order: FullCampaignOrderResult | None
    api_contract: ApiContractValidationResult | None


@dataclass(frozen=True)
class FullCampaignOrderResult:
    phases: tuple[str, ...]
    openwebui_successful_requests: int
    cancellation_phases: tuple[str, ...]
    normal_gateway_pid: int
    restart_gateway_pid: int
    normal_worker_pid: int
    restart_worker_pid: int
    restart_count_before: int
    restart_count_after: int


@dataclass
class _CampaignTrace:
    phase: str
    case_id: str
    completion_id: str
    journal_pid: int
    events: list[tuple[int, dict[str, Any]]]


def validate_hash_bound_bytes(
    encoded: Any, byte_count: Any, digest_value: Any, label: str
) -> bytes:
    raw = decode_base64(encoded, f"{label}.body_base64")
    if base64.b64encode(raw).decode("ascii") != encoded:
        fail(f"{label}.body_base64 is not canonical")
    if integer(byte_count, f"{label}.body_bytes") != len(raw):
        fail(f"{label}.body_bytes differs from decoded bytes")
    digest = sha256_value(digest_value, f"{label}.body_sha256")
    if hashlib.sha256(raw).hexdigest() != digest:
        fail(f"{label}.body_sha256 differs")
    return raw


def _validate_lifecycle_common(value: Any, label: str) -> tuple[str, int]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    event_name = string(value.get("event"), f"{label}.event")
    if event_name not in LIFECYCLE_FIELDS:
        fail(f"{label}.event is unknown")
    exact_fields(
        value,
        {"schema_version", "event", "observed_monotonic_ns"}
        | LIFECYCLE_FIELDS[event_name],
        label,
    )
    if value["schema_version"] != LIFECYCLE_SCHEMA:
        fail(f"{label}.schema_version differs")
    observed = integer(value["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
    return event_name, observed


def validate_lifecycle(value: Any, label: str) -> dict[str, Any]:
    event_name, _ = _validate_lifecycle_common(value, label)
    request_id_value = value.get("request_id")
    completion_id_value = value.get("completion_id")
    if event_name == "worker_fatal" and request_id_value is None:
        if completion_id_value is not None or value["admit_to_fatal_ns"] is not None:
            fail(f"{label} nullable worker_fatal fields must be null together")
    else:
        bounded_utf8_string(
            request_id_value,
            f"{label}.request_id",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        )
        bounded_utf8_string(
            completion_id_value,
            f"{label}.completion_id",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        )

    if event_name == "request_admitted":
        boolean(value["stream"], f"{label}.stream")
        integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        integer(
            value["max_completion_tokens"], f"{label}.max_completion_tokens", minimum=1
        )
    elif event_name == "request_started":
        boolean(value["stream"], f"{label}.stream")
        integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        integer(value["admit_to_start_ns"], f"{label}.admit_to_start_ns")
    elif event_name == "request_progress":
        string(value["phase"], f"{label}.phase")
        processed = integer(
            value["processed_prompt_tokens"],
            f"{label}.processed_prompt_tokens",
            minimum=1,
        )
        prompt = integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        if processed > prompt:
            fail(f"{label}.processed_prompt_tokens exceeds prompt_tokens")
    elif event_name == "request_first_token":
        boolean(value["stream"], f"{label}.stream")
        if integer(value["completion_tokens"], f"{label}.completion_tokens") != 1:
            fail(f"{label}.completion_tokens must equal one")
    elif event_name == "request_cancel_requested":
        boolean(value["stream"], f"{label}.stream")
        string(value["reason"], f"{label}.reason")
        integer(value["admit_to_cancel_ns"], f"{label}.admit_to_cancel_ns")
    elif event_name == "request_released":
        boolean(value["stream"], f"{label}.stream")
        outcome = string(value["outcome"], f"{label}.outcome")
        if outcome not in {"stop", "length", "cancelled"}:
            fail(f"{label}.outcome is invalid")
        cancel_reason = nullable_string(
            value["cancel_reason"], f"{label}.cancel_reason"
        )
        if (outcome == "cancelled") != (cancel_reason is not None):
            fail(f"{label}.cancel_reason does not match outcome")
        integer(value["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
        integer(value["completion_tokens"], f"{label}.completion_tokens")
        if boolean(value["reset_complete"], f"{label}.reset_complete") is not True:
            fail(f"{label}.reset_complete must be true")
        admit_to_start = integer(
            value["admit_to_start_ns"], f"{label}.admit_to_start_ns"
        )
        start_to_release = integer(
            value["start_to_release_ns"], f"{label}.start_to_release_ns"
        )
        admit_to_release = integer(
            value["admit_to_release_ns"], f"{label}.admit_to_release_ns"
        )
        if admit_to_release != admit_to_start + start_to_release:
            fail(f"{label}.admit_to_release_ns arithmetic differs")
    elif event_name == "worker_fatal":
        string(value["reason"], f"{label}.reason")
        if request_id_value is not None:
            integer(value["admit_to_fatal_ns"], f"{label}.admit_to_fatal_ns")
    return value


def decode_lifecycle_message(message: str, label: str) -> dict[str, Any]:
    raw = message.encode("utf-8")
    if raw.startswith(b"{"):
        payload = raw
    elif raw.startswith(b"INFO:     {"):
        payload = raw[len(b"INFO:     ") :]
    else:
        fail(f"{label} has a forbidden journal prefix")
    return validate_lifecycle(decode_json_bytes(payload, label), label)


def _validate_header(
    record: dict[str, Any],
    root: Path,
    matrix: MatrixData,
    expected_worker_sha256: str,
) -> tuple[str, str, InputSeal]:
    label = "raw-session header"
    if record["phase"] != "preflight" or record["case_id"] is not None:
        fail(f"{label} phase/case_id differs")
    run_id = string(record["run_id"], f"{label}.run_id")
    if run_id != matrix.run_id:
        fail(f"{label}.run_id differs from release matrix")
    string(record["started_utc"], f"{label}.started_utc")
    if record["clock"] != "python.time.monotonic_ns":
        fail(f"{label}.clock differs")
    boot_id = string(record["boot_id"], f"{label}.boot_id")

    identities = exact_fields(
        record["identities"],
        {
            "environment_file",
            "environment_sha256",
            "model_identity_file",
            "model_identity_sha256",
            "openwebui",
            "docker_network_id",
            "gateway_source_sha256",
            "worker_source_sha256",
            "worker_binary_sha256",
        },
        f"{label}.identities",
    )
    if (
        identities["environment_file"] != "environment.json"
        or identities["model_identity_file"] != "model-identity.json"
    ):
        fail(f"{label}.identities bundle filenames differ")
    for name, digest_key in (
        ("environment.json", "environment_sha256"),
        ("model-identity.json", "model_identity_sha256"),
    ):
        expected = sha256_value(
            identities[digest_key], f"{label}.identities.{digest_key}"
        )
        if sha256_file(safe_relative_file(root, name, name)) != expected:
            fail(f"{label}.identities.{digest_key} differs from {name}")
    openwebui = exact_fields(
        identities["openwebui"],
        {
            "version",
            "source_revision",
            "base_image_digest",
            "base_image_id",
            "derived_image_id",
            "Dockerfile_sha256",
            "patch_sha256",
            "patched_middleware_sha256",
        },
        f"{label}.identities.openwebui",
    )
    for key in ("version", "source_revision"):
        string(openwebui[key], f"{label}.identities.openwebui.{key}")
    for key in ("base_image_digest", "base_image_id", "derived_image_id"):
        value = string(openwebui[key], f"{label}.identities.openwebui.{key}")
        if not value.startswith("sha256:") or SHA256_RE.fullmatch(value[7:]) is None:
            fail(f"{label}.identities.openwebui.{key} is not a content image identity")
    for key in ("Dockerfile_sha256", "patch_sha256", "patched_middleware_sha256"):
        sha256_value(openwebui[key], f"{label}.identities.openwebui.{key}")
    network_id = string(
        identities["docker_network_id"], f"{label}.identities.docker_network_id"
    )
    if SHA256_RE.fullmatch(network_id) is None:
        fail(f"{label}.identities.docker_network_id is not a 64-hex ID")
    sha256_value(
        identities["gateway_source_sha256"], f"{label}.identities.gateway_source_sha256"
    )
    sha256_value(
        identities["worker_source_sha256"], f"{label}.identities.worker_source_sha256"
    )
    worker_sha = sha256_value(
        identities["worker_binary_sha256"], f"{label}.identities.worker_binary_sha256"
    )
    if worker_sha != expected_worker_sha256:
        fail(f"{label} worker binary differs from the trusted CLI anchor")

    input_files = record["input_files"]
    if type(input_files) is not list:
        fail(f"{label}.input_files must be an array")
    input_paths: list[str] = []
    input_seals: dict[str, InputSeal] = {}
    for index, item in enumerate(input_files):
        item_label = f"{label}.input_files[{index}]"
        exact_fields(item, {"path", "bytes", "sha256"}, item_label)
        relative = string(item["path"], f"{item_label}.path")
        pure = PurePosixPath(relative)
        lexical_parts = relative.split("/")
        if (
            pure.is_absolute()
            or any(part in {"", ".", ".."} for part in lexical_parts)
            or "\\" in relative
        ):
            fail(f"{item_label}.path is unsafe")
        size = integer(item["bytes"], f"{item_label}.bytes")
        digest = sha256_value(item["sha256"], f"{item_label}.sha256")
        input_paths.append(relative)
        input_seals[relative] = InputSeal(size, digest)
    if input_paths != sorted(set(input_paths), key=lambda item: item.encode("utf-8")):
        fail(f"{label}.input_files paths are not bytewise ascending and unique")
    required_inputs = {
        "collector/config.json",
        RESOURCE_FIXTURE_INPUT_PATH,
        "tools/collect-sq8-openwebui-release.py",
        "tools/sq8-openwebui-http-client.py",
    }
    if not required_inputs.issubset(input_seals):
        fail(f"{label}.input_files lacks fixed collector/client/fixture inputs")
    validate_schedule(record["schedule"], f"{label}.schedule")
    validate_thresholds(record["thresholds"], f"{label}.thresholds")
    if not json_equal(record["schedule"], matrix.schedule) or not json_equal(
        record["thresholds"], matrix.thresholds
    ):
        fail(f"{label} schedule/thresholds differ from release matrix")
    return run_id, boot_id, input_seals[RESOURCE_FIXTURE_INPUT_PATH]


def _canonical_fixture(model: str, messages: list[Any], label: str) -> bytes:
    for index, message in enumerate(messages):
        if type(message) is not dict or set(message) != {"role", "content"}:
            fail(f"{label}.messages[{index}] fields differ")
        if message["role"] not in {"system", "user", "assistant"}:
            fail(f"{label}.messages[{index}].role differs")
        string(message["content"], f"{label}.messages[{index}].content")
    try:
        return json.dumps(
            {"model": model, "messages": messages},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        fail(f"{label} cannot be canonically encoded: {error}")


def _validate_positive_resource_body(
    raw: bytes,
    record: dict[str, Any],
    state: HttpValidationState,
    label: str,
) -> str:
    value = decode_json_bytes(raw, f"{label}.body")
    exact_fields(
        value,
        {
            "model",
            "messages",
            "stream",
            "stream_options",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
        },
        f"{label}.body",
    )
    model = string(value["model"], f"{label}.body.model")
    messages = value["messages"]
    if type(messages) is not list or not messages:
        fail(f"{label}.body.messages must be a non-empty array")
    fixture_raw = _canonical_fixture(model, messages, f"{label}.body")
    if (
        len(fixture_raw) != state.fixture_seal.size
        or hashlib.sha256(fixture_raw).hexdigest() != state.fixture_seal.sha256
    ):
        fail(f"{label} model/messages differ from the header-bound resource fixture")
    if state.fixture_model is None:
        state.fixture_model = model
        state.fixture_messages = messages
    elif model != state.fixture_model or not json_equal(
        messages, state.fixture_messages
    ):
        fail(f"{label} resource fixture differs between requests")
    if (
        value["stream"] is not True
        or not json_equal(value["stream_options"], {"include_usage": True})
        or integer(value["max_tokens"], f"{label}.body.max_tokens") != 2
    ):
        fail(f"{label} resource streaming/max_tokens settings differ")

    phase = record["phase"]
    case_id = record["case_id"]
    if phase == "resource_normal":
        warmup = re.fullmatch(r"normal-warmup-([0-9]{2})", case_id)
        measured = re.fullmatch(r"normal-measured-([0-9]{3})", case_id)
        match = warmup or measured
        if match is None:
            fail(f"{label} normal resource case_id differs")
        index = int(match.group(1))
        maximum = 10 if warmup is not None else 100
        if index < 1 or index > maximum or record["request_index"] != index:
            fail(f"{label} normal resource request index differs")
        sampled = measured is not None and index in SCHEDULE["sampled_normal_indices"]
        expected_temperature: int | Decimal = Decimal("0.6") if sampled else 0
        expected_top_p: int | Decimal = Decimal("0.95") if sampled else 1
        expected_seed = index if sampled else 0
        kind = "normal_warmup" if warmup is not None else "normal_measured"
    elif phase == "resource_restart":
        warmup = re.fullmatch(r"restart-warmup-([0-9]{2})", case_id)
        measured = re.fullmatch(r"restart-measured-([0-9]{3})", case_id)
        match = warmup or measured
        if match is None:
            fail(f"{label} restart resource case_id differs")
        index = int(match.group(1))
        maximum = 10 if warmup is not None else 20
        if index < 1 or index > maximum or record["request_index"] != index:
            fail(f"{label} restart resource request index differs")
        expected_temperature, expected_top_p, expected_seed = 0, 1, 0
        kind = "restart_warmup" if warmup is not None else "restart_measured"
    else:
        fail(f"{label} positive resource body has a non-resource phase")
    if (
        not json_equal(value["temperature"], expected_temperature)
        or not json_equal(value["top_p"], expected_top_p)
        or integer(value["seed"], f"{label}.body.seed") != expected_seed
    ):
        fail(f"{label} resource sampling settings differ")
    return kind


def _require_malformed_json(raw: bytes, label: str) -> None:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        fail(f"{label} malformed JSON body is not strict UTF-8: {error}")
    try:
        json.loads(
            text,
            parse_constant=reject_json_constant,
        )
    except (ValidationError, json.JSONDecodeError, ValueError, RecursionError):
        return
    fail(f"{label} must contain malformed JSON")


def _validate_overflow_body(
    raw: bytes,
    state: HttpValidationState,
    case_name: str,
    label: str,
) -> None:
    value = decode_json_bytes(raw, f"{label}.body")
    exact_fields(
        value,
        {
            "model",
            "messages",
            "stream",
            "stream_options",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
        },
        f"{label}.body",
    )
    if state.fixture_model is None or state.fixture_messages is None:
        fail(f"{label} context overflow appears before the resource fixture")
    if (
        value["model"] != state.fixture_model
        or not json_equal(
            value["messages"],
            [
                {
                    "role": "user",
                    "content": CONTEXT_OVERFLOW_CONTENT[case_name],
                }
            ],
        )
        or value["stream"] is not True
        or not json_equal(value["stream_options"], {"include_usage": True})
        or not json_equal(value["max_tokens"], 2)
        or not json_equal(value["temperature"], 0)
        or not json_equal(value["top_p"], 1)
        or not json_equal(value["seed"], 0)
    ):
        fail(f"{label} context-overflow request shape differs")


def _validate_negative_resource_body(
    raw: bytes, record: dict[str, Any], state: HttpValidationState, label: str
) -> str:
    expected = {
        "negative-after-025-context_overflow_1": (25, "context_overflow"),
        "negative-after-050-malformed_json": (50, "malformed_json"),
        "negative-after-075-context_overflow_2": (75, "context_overflow"),
    }
    item = expected.get(record["case_id"])
    if record["phase"] != "resource_normal" or item is None:
        fail(f"{label} negative resource identity differs")
    request_index, kind = item
    if record["request_index"] != request_index:
        fail(f"{label} negative resource index differs")
    if kind == "malformed_json":
        _require_malformed_json(raw, f"{label}.body")
    else:
        case_name = (
            "context_overflow_1" if request_index == 25 else "context_overflow_2"
        )
        _validate_overflow_body(raw, state, case_name, label)
    return kind


def _parse_resource_sse(raw: bytes, label: str) -> str:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        fail(f"{label} SSE is not strict UTF-8: {error}")
    data_events: list[str] = []
    data_lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line == "":
            if data_lines:
                data_events.append("\n".join(data_lines))
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
    if data_lines:
        data_events.append("\n".join(data_lines))
    if not data_events or data_events[-1] != "[DONE]":
        fail(f"{label} lacks terminal [DONE]")
    completion_ids: set[str] = set()
    content_count = 0
    usage_count: int | None = None
    for index, payload in enumerate(data_events[:-1]):
        value = decode_json_bytes(payload.encode("utf-8"), f"{label} data {index}")
        if "id" in value:
            completion_ids.add(string(value["id"], f"{label} data {index}.id"))
        choices = value.get("choices")
        if type(choices) is list and choices:
            first = choices[0]
            if type(first) is dict and type(first.get("delta")) is dict:
                content = first["delta"].get("content")
                if type(content) is str and content:
                    content_count += 1
        usage = value.get("usage")
        if type(usage) is dict and "completion_tokens" in usage:
            count = integer(usage["completion_tokens"], f"{label} usage")
            if usage_count is not None:
                fail(f"{label} duplicates usage")
            usage_count = count
    if len(completion_ids) != 1 or content_count < 1 or usage_count != 2:
        fail(f"{label} completion identity/content/usage differs")
    return next(iter(completion_ids))


def _parse_error_envelope(raw: bytes, expected_code: str, label: str) -> None:
    value = decode_json_bytes(raw, label)
    exact_fields(value, {"error"}, label)
    error = exact_fields(
        value["error"], {"message", "type", "param", "code"}, f"{label}.error"
    )
    string(error["message"], f"{label}.error.message")
    if error["type"] != "invalid_request_error" or error["code"] != expected_code:
        fail(f"{label} semantic error class differs")
    if expected_code == "context_length_exceeded" and error["param"] != "messages":
        fail(f"{label} context overflow param differs")
    if expected_code == "invalid_request_error" and error["param"] is not None:
        string(error["param"], f"{label}.error.param")


def _validate_http_record(
    record: dict[str, Any],
    label: str,
    state: HttpValidationState,
) -> None:
    record_type = record["record_type"]
    if record_type == "http_request":
        integer(record["request_index"], f"{label}.request_index")
        key = bounded_utf8_string(
            record["request_key"],
            f"{label}.request_key",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        )
        if key in state.requests:
            fail(f"{label}.request_key is duplicated")
        if state.active_key is not None:
            fail(f"{label} overlaps another active raw HTTP request")
        phase = record["phase"]
        case_id = bounded_utf8_string(
            record["case_id"],
            f"{label}.case_id",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        )
        method = string(record["method"], f"{label}.method")
        target = string(record["target"], f"{label}.target")
        if phase == "api_contract":
            if (method, target) not in {
                ("GET", "/v1/models"),
                ("GET", "/v1/models?x=1"),
                ("POST", "/v1/chat/completions"),
            }:
                fail(f"{label} API contract method/target differs")
        elif method != "POST" or target != "/v1/chat/completions":
            fail(f"{label} method/target differs")
        headers = exact_fields(
            record["headers"],
            {"content_type", "content_length", "authorization_mode"},
            f"{label}.headers",
        )
        if headers["content_type"] != "application/json":
            fail(f"{label}.headers.content_type differs")
        content_length = integer(
            headers["content_length"], f"{label}.headers.content_length"
        )
        authorization_mode = string(
            headers["authorization_mode"], f"{label}.headers.authorization_mode"
        )
        allowed_authorization_modes = (
            {"valid_bearer", "missing", "invalid_bearer"}
            if phase == "api_contract"
            else {"valid_bearer"}
        )
        if authorization_mode not in allowed_authorization_modes:
            fail(f"{label}.headers.authorization_mode differs")
        raw = validate_hash_bound_bytes(
            record["body_base64"], record["body_bytes"], record["body_sha256"], label
        )
        if content_length != len(raw):
            fail(f"{label}.headers.content_length differs")
        if phase == "api_contract" and ((method == "GET") != (raw == b"")):
            fail(f"{label} API contract method/body shape differs")
        connect = integer(
            record["connect_completed_monotonic_ns"],
            f"{label}.connect_completed_monotonic_ns",
        )
        started = integer(
            record["write_started_monotonic_ns"], f"{label}.write_started_monotonic_ns"
        )
        sent = integer(
            record["last_body_byte_sent_monotonic_ns"],
            f"{label}.last_body_byte_sent_monotonic_ns",
        )
        if not connect <= started <= sent:
            fail(f"{label} request timing order differs")
        if connect < state.last_response_end_ns:
            fail(f"{label} begins before the prior HTTP response ended")
        if phase in {"resource_normal", "resource_restart"}:
            if record["case_id"].startswith("negative-after-"):
                kind = _validate_negative_resource_body(raw, record, state, label)
            else:
                kind = _validate_positive_resource_body(raw, record, state, label)
        elif phase == "api_contract":
            kind = "api_contract"
        else:
            kind = "other"
        state.requests[key] = {
            "phase": phase,
            "case_id": case_id,
            "request_index": record["request_index"],
            "kind": kind,
            "connect_ns": connect,
            "write_ns": started,
            "sent_ns": sent,
            "method": method,
            "target": target,
            "authorization_mode": authorization_mode,
            "request_body_bytes": len(raw),
            "request_body_sha256": hashlib.sha256(raw).hexdigest(),
        }
        state.ordered_keys.append(key)
        state.bodies[key] = HttpBodyState(hashlib.sha256(), 0, 0, bytearray(), sent)
        state.active_key = key
    elif record_type == "http_response_start":
        key = bounded_utf8_string(
            record["request_key"],
            f"{label}.request_key",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        )
        if (
            key not in state.requests
            or key in state.response_started
            or key != state.active_key
        ):
            fail(f"{label} response start has an unknown or duplicated request_key")
        status = integer(record["status"], f"{label}.status", minimum=100, maximum=599)
        headers = record["headers"]
        if type(headers) is not list:
            fail(f"{label}.headers must be an array")
        if len(headers) > MAX_HTTP_RESPONSE_HEADER_COUNT:
            fail(f"{label}.headers exceeds its count bound")
        parsed_headers: list[tuple[str, str]] = []
        header_bytes = 0
        for index, pair in enumerate(headers):
            if type(pair) is not list or len(pair) != 2:
                fail(f"{label}.headers[{index}] must be a two-string array")
            name = bounded_utf8_string(
                pair[0],
                f"{label}.headers[{index}][0]",
                maximum_bytes=MAX_HTTP_HEADER_NAME_BYTES,
            )
            value = bounded_utf8_string(
                pair[1],
                f"{label}.headers[{index}][1]",
                maximum_bytes=MAX_HTTP_HEADER_VALUE_BYTES,
                nonempty=False,
            )
            header_bytes += len(name.encode("utf-8")) + len(value.encode("utf-8"))
            if header_bytes > MAX_HTTP_RESPONSE_HEADER_BYTES:
                fail(f"{label}.headers exceeds its aggregate byte bound")
            parsed_headers.append((name, value))
        content_types = [
            value.split(";", 1)[0].strip().lower()
            for name, value in parsed_headers
            if name.lower() == "content-type"
        ]
        expected_media = (
            "application/json"
            if state.requests[key]["phase"] == "api_contract"
            else "text/event-stream"
            if status == 200
            else "application/json"
        )
        if content_types != [expected_media]:
            fail(f"{label} response Content-Type differs")
        observed = integer(
            record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns"
        )
        body_state = state.bodies[key]
        if observed < body_state.last_observed_ns:
            fail(f"{label} response start precedes request send")
        body_state.last_observed_ns = observed
        state.requests[key]["status"] = status
        if state.requests[key]["phase"] == "api_contract":
            state.requests[key]["response_headers"] = tuple(parsed_headers)
        state.requests[key]["response_started_ns"] = observed
        if expected_media == "text/event-stream":
            body_state.sse_parser = _CompactSseParser()
        state.response_started.add(key)
    elif record_type == "http_body_chunk":
        key = bounded_utf8_string(
            record["request_key"],
            f"{label}.request_key",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        )
        if key not in state.response_started or key in state.response_ended:
            fail(f"{label} chunk has no active response")
        body_state = state.bodies[key]
        if (
            integer(record["chunk_index"], f"{label}.chunk_index")
            != body_state.next_index
        ):
            fail(f"{label}.chunk_index is not contiguous")
        raw = validate_hash_bound_bytes(
            record["body_base64"], record["body_bytes"], record["body_sha256"], label
        )
        request = state.requests[key]
        response_limit = (
            API_CONTRACT_MAX_RESPONSE_BYTES
            if request["phase"] == "api_contract"
            else MAX_JSON_BYTES
        )
        if request["phase"] == "api_contract" and not raw:
            fail(f"{label} API contract body chunk is empty")
        if body_state.sse_parser is not None and not raw:
            fail(f"{label} SSE body chunk is empty")
        if body_state.byte_count + len(raw) > response_limit:
            fail(f"{label} complete response exceeds its size limit")
        body_state.digest.update(raw)
        body_state.raw.extend(raw)
        body_state.byte_count += len(raw)
        body_state.next_index += 1
        observed = integer(
            record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns"
        )
        if observed < body_state.last_observed_ns:
            fail(f"{label} body chunk timestamps regress")
        if body_state.sse_parser is not None:
            body_state.sse_parser.feed(raw, body_state.next_index - 1, observed)
        body_state.last_observed_ns = observed
    elif record_type == "http_response_end":
        key = bounded_utf8_string(
            record["request_key"],
            f"{label}.request_key",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        )
        if key not in state.response_started or key in state.response_ended:
            fail(f"{label} response end has no active response")
        outcome = string(record["outcome"], f"{label}.outcome")
        if outcome not in {"eof", "client_closed", "timeout", "error"}:
            fail(f"{label}.outcome differs")
        error = nullable_string(record["error"], f"{label}.error")
        if (outcome in {"eof", "client_closed"}) != (error is None):
            fail(f"{label}.error does not match outcome")
        body_state = state.bodies[key]
        if (
            integer(record["body_bytes"], f"{label}.body_bytes")
            != body_state.byte_count
        ):
            fail(f"{label}.body_bytes differs from chunks")
        if (
            sha256_value(record["body_sha256"], f"{label}.body_sha256")
            != body_state.digest.hexdigest()
        ):
            fail(f"{label}.body_sha256 differs from chunks")
        observed = integer(
            record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns"
        )
        if observed < body_state.last_observed_ns:
            fail(f"{label} response end timestamp regresses")
        request = state.requests[key]
        request["outcome"] = outcome
        needs_body_validation = request["phase"] == "api_contract" or request[
            "kind"
        ] in {
            "normal_warmup",
            "normal_measured",
            "restart_warmup",
            "restart_measured",
            "context_overflow",
            "malformed_json",
        }
        response_body = bytes(body_state.raw) if needs_body_validation else None
        if request["phase"] == "api_contract":
            if outcome != "eof" or error is not None:
                fail(f"{label} API contract response did not terminate at EOF")
            api_position = sum(
                state.requests[ordered_key]["phase"] == "api_contract"
                for ordered_key in state.ordered_keys
                if ordered_key in state.response_ended or ordered_key == key
            )
            if api_position not in range(1, len(API_CONTRACT_CASES) + 1):
                request["api_response"] = None
            else:
                case = API_CONTRACT_CASES[api_position - 1]
                expected_key = f"api-contract-{api_position:02d}-{case.case_id}"
                if key == expected_key:
                    assert response_body is not None
                    request["response_body"] = response_body
                    request["response_chunk_count"] = body_state.next_index
                    try:
                        request["api_response"] = _validate_api_contract_response(
                            case, request, label
                        )
                    finally:
                        request.pop("response_body", None)
        if request["kind"] in {
            "normal_warmup",
            "normal_measured",
            "restart_warmup",
            "restart_measured",
        }:
            if request.get("status") != 200 or outcome != "eof":
                fail(f"{label} positive resource HTTP outcome differs")
            assert response_body is not None
            request["completion_id"] = _parse_resource_sse(
                response_body, f"{label}.body"
            )
        elif request["kind"] in {"context_overflow", "malformed_json"}:
            if request.get("status") != 400 or outcome != "eof":
                fail(f"{label} negative resource HTTP outcome differs")
            expected_code = (
                "context_length_exceeded"
                if request["kind"] == "context_overflow"
                else "invalid_request_error"
            )
            assert response_body is not None
            _parse_error_envelope(response_body, expected_code, f"{label}.body")
        sse = (
            body_state.sse_parser.finish(allow_incomplete=outcome == "client_closed")
            if body_state.sse_parser is not None
            else None
        )
        if sse is not None:
            state.total_sse_items += len(sse.items)
            if state.total_sse_items > MAX_SESSION_SSE_ITEMS:
                fail("compact session SSE item count exceeds its bound")
        request["response_end_ns"] = observed
        status = integer(request.get("status"), f"{label}.status", 100, 599)
        state.completed_results[key] = HttpCompactResult(
            phase=string(request.get("phase"), f"{label}.phase"),
            case_id=string(request.get("case_id"), f"{label}.case_id"),
            request_index=integer(
                request.get("request_index"), f"{label}.request_index"
            ),
            request_key=key,
            method=string(request.get("method"), f"{label}.method"),
            target=string(request.get("target"), f"{label}.target"),
            status=status,
            outcome=outcome,
            request_body_bytes=integer(
                request.get("request_body_bytes"), f"{label}.request_body_bytes"
            ),
            request_body_sha256=sha256_value(
                request.get("request_body_sha256"), f"{label}.request_body_sha256"
            ),
            response_body_bytes=body_state.byte_count,
            response_body_sha256=body_state.digest.hexdigest(),
            connect_completed_monotonic_ns=integer(
                request.get("connect_ns"), f"{label}.connect_ns"
            ),
            write_started_monotonic_ns=integer(
                request.get("write_ns"), f"{label}.write_ns"
            ),
            last_body_byte_sent_monotonic_ns=integer(
                request.get("sent_ns"), f"{label}.sent_ns"
            ),
            response_started_monotonic_ns=integer(
                request.get("response_started_ns"), f"{label}.response_started_ns"
            ),
            response_end_monotonic_ns=observed,
            sse=sse,
        )
        state.response_ended.add(key)
        response_body = None
        body_state.raw.clear()
        del state.bodies[key]
        state.active_key = None
        state.last_response_end_ns = observed


def _api_contract_header_values(
    headers: tuple[tuple[str, str], ...], name: str
) -> list[str]:
    return [value for key, value in headers if key.lower() == name.lower()]


def _validate_api_contract_response(
    case: ApiContractCase, request: dict[str, Any], label: str
) -> dict[str, Any]:
    status = integer(request.get("status"), f"{label}.status", minimum=100, maximum=599)
    if status != case.expected_status:
        fail(f"{label} status differs from the frozen API contract")
    if request.get("outcome") != "eof":
        fail(f"{label} response outcome differs from the frozen API contract")
    headers_value = request.get("response_headers")
    if type(headers_value) is not tuple or any(
        type(pair) is not tuple
        or len(pair) != 2
        or type(pair[0]) is not str
        or type(pair[1]) is not str
        for pair in headers_value
    ):
        fail(f"{label} response headers are incomplete")
    headers = cast(tuple[tuple[str, str], ...], headers_value)
    content_types = _api_contract_header_values(headers, "Content-Type")
    if content_types != ["application/json"]:
        fail(f"{label} response Content-Type differs")
    authenticate = _api_contract_header_values(headers, "WWW-Authenticate")
    expected_authenticate = ["Bearer"] if status == 401 else []
    if authenticate != expected_authenticate:
        fail(f"{label} WWW-Authenticate header differs")
    if _api_contract_header_values(headers, "Retry-After"):
        fail(f"{label} non-busy response contains Retry-After")
    if _api_contract_header_values(headers, "Transfer-Encoding"):
        fail(f"{label} response unexpectedly uses Transfer-Encoding")

    body_value = request.get("response_body")
    if type(body_value) not in {bytes, bytearray}:
        fail(f"{label} response body is incomplete")
    body = cast(bytes, body_value)
    if not body or len(body) > API_CONTRACT_MAX_RESPONSE_BYTES:
        fail(f"{label} response body size differs")
    content_lengths = _api_contract_header_values(headers, "Content-Length")
    if content_lengths != [str(len(body))]:
        fail(f"{label} response Content-Length differs")
    if integer(
        request.get("response_chunk_count"), f"{label}.response_chunk_count"
    ) not in range(1, 126):
        fail(f"{label} response chunk count differs")

    value = decode_json_bytes(
        body,
        f"{label}.response_body",
        allow_outer_whitespace=True,
    )
    error_summary: dict[str, Any] | None = None
    if case.expect_models:
        expected_models = {
            "object": "list",
            "data": [
                {
                    "id": API_CONTRACT_MODEL_ID,
                    "object": "model",
                    "owned_by": "ullm",
                }
            ],
        }
        if not json_equal(value, expected_models):
            fail(f"{label} model list differs")
    else:
        envelope = exact_fields(value, {"error"}, f"{label}.response_body")
        error = exact_fields(
            envelope["error"],
            {"message", "type", "param", "code"},
            f"{label}.response_body.error",
        )
        message = string(error["message"], f"{label}.response_body.error.message")
        if (
            error["type"] != "invalid_request_error"
            or error["code"] != case.expected_code
            or error["param"] != case.expected_param
            or message != case.expected_message
        ):
            fail(f"{label} error message, type, code, or param differs")
        message_raw = message.encode("utf-8")
        error_summary = {
            "type": error["type"],
            "code": error["code"],
            "param": error["param"],
            "message_utf8_bytes": len(message_raw),
            "message_sha256": hashlib.sha256(message_raw).hexdigest(),
        }
    return {
        "content_type": content_types[0],
        "content_length": len(body),
        "www_authenticate": authenticate,
        "response_body_bytes": len(body),
        "response_body_sha256": hashlib.sha256(body).hexdigest(),
        "error": error_summary,
    }


def validate_api_contract_http(
    source: SessionData | HttpValidationState,
) -> ApiContractValidationResult:
    """Reconstruct the exact ten-case API contract from raw HTTP records.

    Phase-1 validation intentionally does not call this helper.  The full release
    validator must call it explicitly after ``validate_session`` so a partial
    resource-only bundle does not acquire a completed API-contract claim.
    """

    if isinstance(source, SessionData):
        requests = source.http_requests
        ordered_keys = source.ordered_http_keys
    else:
        requests = source.requests
        ordered_keys = source.ordered_keys
    api_keys = [key for key in ordered_keys if requests[key]["phase"] == "api_contract"]
    if len(api_keys) != len(API_CONTRACT_CASES):
        fail("API contract raw HTTP request count differs from the frozen schedule")

    statuses: list[int] = []
    case_results: list[dict[str, Any]] = []
    for case_index, (key, case) in enumerate(
        zip(api_keys, API_CONTRACT_CASES, strict=True), start=1
    ):
        request = requests[key]
        expected_key = f"api-contract-{case_index:02d}-{case.case_id}"
        label = f"API contract case {case_index} ({case.case_id})"
        if (
            key != expected_key
            or request.get("phase") != "api_contract"
            or request.get("case_id") != case.case_id
            or request.get("request_index") != case_index
            or request.get("kind") != "api_contract"
            or request.get("method") != case.method
            or request.get("target") != case.target
            or request.get("authorization_mode") != case.authorization_mode
            or request.get("request_body_bytes") != len(case.body)
            or request.get("request_body_sha256")
            != hashlib.sha256(case.body).hexdigest()
        ):
            fail(f"{label} request identity, order, authorization, or body differs")
        response_value = request.get("api_response")
        if type(response_value) is not dict:
            fail(f"{label} validated response summary is absent")
        response = cast(dict[str, Any], response_value)
        status = integer(request["status"], f"{label}.status")
        statuses.append(status)
        case_results.append(
            {
                "case_index": case_index,
                "case_id": case.case_id,
                "method": case.method,
                "target": case.target,
                "authorization_mode": case.authorization_mode,
                "request_body_bytes": len(case.body),
                "request_body_sha256": hashlib.sha256(case.body).hexdigest(),
                "connect_completed_monotonic_ns": integer(
                    request.get("connect_ns"), f"{label}.connect_ns"
                ),
                "write_started_monotonic_ns": integer(
                    request.get("write_ns"), f"{label}.write_ns"
                ),
                "last_body_byte_sent_monotonic_ns": integer(
                    request.get("sent_ns"), f"{label}.sent_ns"
                ),
                "status": status,
                "response_started_monotonic_ns": integer(
                    request.get("response_started_ns"),
                    f"{label}.response_started_ns",
                ),
                "response_end_monotonic_ns": integer(
                    request.get("response_end_ns"), f"{label}.response_end_ns"
                ),
                **response,
            }
        )

    return ApiContractValidationResult(
        case_ids=tuple(case.case_id for case in API_CONTRACT_CASES),
        request_keys=tuple(api_keys),
        statuses=tuple(statuses),
        cases=tuple(case_results),
    )


def validate_api_contract_quiet_checks(
    checks: Sequence[LifecycleQuietCheckData],
    journal_observations: Sequence[ApiJournalObservationData],
    http_results: Sequence[HttpCompactResult],
    expected_gateway_pid: int,
) -> tuple[LifecycleQuietCheckData, ...]:
    labels = [case.case_id for case in API_CONTRACT_CASES] + [
        "http-client-shutdown",
        "post-observer-close",
        "final-readiness-and-identity",
    ]
    if len(checks) != len(labels):
        fail("API contract lifecycle quiet-check count differs from the fixed schedule")
    if not journal_observations:
        fail("API contract lifecycle quiet checks lack journal observations")
    seen_cursors: set[str] = set()
    prior_journal_monotonic = -1
    for index, observation in enumerate(journal_observations):
        if (
            observation.phase != "api_contract"
            or observation.observation_index != index
            or observation.case_id != f"api-journal-{index + 1:02d}"
            or observation.journal_cursor in seen_cursors
            or observation.journal_monotonic_usec < prior_journal_monotonic
            or observation.journal_pid != expected_gateway_pid
        ):
            fail("API contract journal observation identity, order, or PID differs")
        seen_cursors.add(observation.journal_cursor)
        prior_journal_monotonic = observation.journal_monotonic_usec
    api_http = {
        result.case_id: result
        for result in http_results
        if result.phase == "api_contract"
    }
    if set(api_http) != {case.case_id for case in API_CONTRACT_CASES}:
        fail("API contract quiet checks lack their complete HTTP result set")

    prior_checked_ns = -1
    prior_journal_count = 0
    for sequence, (check, expected_label) in enumerate(
        zip(checks, labels, strict=True)
    ):
        if (
            check.phase != "api_contract"
            or check.case_id != expected_label
            or check.label != expected_label
            or check.quiet_sequence != sequence
            or check.observer_open is not (sequence <= 10)
            or check.observer_event_count != 0
            or check.checked_monotonic_ns < prior_checked_ns
            or check.journal_record_count <= 0
            or check.journal_record_count > len(journal_observations)
            or check.journal_record_count < prior_journal_count
            or check.new_journal_record_count
            != check.journal_record_count - prior_journal_count
        ):
            fail("API contract lifecycle quiet-check identity, order, or count differs")
        bound_observation = journal_observations[check.journal_record_count - 1]
        if (
            check.journal_cursor != bound_observation.journal_cursor
            or check.checked_monotonic_ns
            < bound_observation.journal_monotonic_usec * 1000
        ):
            fail("API contract lifecycle quiet check differs from its journal boundary")
        if sequence < len(API_CONTRACT_CASES):
            response_end_ns = api_http[expected_label].response_end_monotonic_ns
        else:
            response_end_ns = max(
                result.response_end_monotonic_ns for result in api_http.values()
            )
        if check.checked_monotonic_ns < response_end_ns:
            fail("API contract lifecycle quiet check precedes its HTTP boundary")
        prior_checked_ns = check.checked_monotonic_ns
        prior_journal_count = check.journal_record_count
    if prior_journal_count != len(journal_observations):
        fail(
            "API contract lifecycle quiet checks do not cover the journal observations"
        )
    return tuple(checks)


def _add_lifecycle_event(
    traces: dict[str, RequestTrace],
    completion_ids: dict[str, str],
    phase: str,
    case_id: str,
    event: dict[str, Any],
    label: str,
) -> None:
    name = event["event"]
    request_id = event["request_id"]
    if request_id is None:
        return
    completion_id = event["completion_id"]
    trace = traces.get(request_id)
    if name == "request_admitted":
        if trace is not None or completion_id in completion_ids:
            fail(f"{label} admitted request/completion ID is duplicated")
        trace = RequestTrace(
            phase=phase, case_id=case_id, completion_id=completion_id, events=[]
        )
        traces[request_id] = trace
        completion_ids[completion_id] = request_id
    elif trace is None:
        fail(f"{label} refers to a request before admission")
    assert trace is not None
    if (
        trace.phase != phase
        or trace.case_id != case_id
        or trace.completion_id != completion_id
    ):
        fail(f"{label} request correlation differs")
    if trace.terminal is not None:
        fail(f"{label} appears after terminal event {trace.terminal}")
    previous_time = trace.events[-1]["observed_monotonic_ns"] if trace.events else None
    if previous_time is not None and event["observed_monotonic_ns"] < previous_time:
        fail(f"{label} monotonic event order differs")
    names = [item["event"] for item in trace.events]
    if name == "request_started":
        if names != ["request_admitted"]:
            fail(f"{label} started event order differs")
        admitted = trace.events[0]
        if (
            event["stream"] is not admitted["stream"]
            or event["prompt_tokens"] != admitted["prompt_tokens"]
        ):
            fail(f"{label} started fields differ from admission")
    elif name == "request_progress":
        if (
            "request_started" not in names
            or "request_first_token" in names
            or "request_cancel_requested" in names
        ):
            fail(f"{label} progress event order differs")
        if event["prompt_tokens"] != trace.events[0]["prompt_tokens"]:
            fail(f"{label} progress prompt_tokens differs from admission")
        prior_progress = [
            item["processed_prompt_tokens"]
            for item in trace.events
            if item["event"] == "request_progress"
        ]
        if prior_progress and event["processed_prompt_tokens"] <= prior_progress[-1]:
            fail(f"{label} progress is not strictly increasing")
    elif name == "request_first_token":
        if (
            "request_started" not in names
            or "request_first_token" in names
            or "request_cancel_requested" in names
        ):
            fail(f"{label} first-token event order differs")
        if event["stream"] is not trace.events[0]["stream"]:
            fail(f"{label} first-token stream flag differs from admission")
    elif name == "request_cancel_requested":
        if "request_started" not in names or "request_cancel_requested" in names:
            fail(f"{label} cancel event order differs")
        if event["stream"] is not trace.events[0]["stream"]:
            fail(f"{label} cancel stream flag differs from admission")
    elif name == "request_released":
        if "request_started" not in names or "request_released" in names:
            fail(f"{label} release event order differs")
        admitted = trace.events[0]
        started = next(
            item for item in trace.events if item["event"] == "request_started"
        )
        if (
            event["stream"] is not admitted["stream"]
            or event["prompt_tokens"] != admitted["prompt_tokens"]
        ):
            fail(f"{label} release fields differ from admission")
        if event["admit_to_start_ns"] != started["admit_to_start_ns"]:
            fail(f"{label} release/start duration differs")
        maximum_completion = admitted["max_completion_tokens"]
        if event["completion_tokens"] > maximum_completion:
            fail(f"{label} completion count exceeds admission maximum")
        if (event["completion_tokens"] > 0) != ("request_first_token" in names):
            fail(f"{label} completion count and first-token event differ")
        if (
            event["outcome"] == "length"
            and event["completion_tokens"] != maximum_completion
        ):
            fail(f"{label} length outcome does not reach the admission maximum")
        if event["outcome"] == "cancelled" and "request_cancel_requested" not in names:
            fail(f"{label} cancelled release lacks cancellation event")
        if event["outcome"] != "cancelled" and "request_cancel_requested" in names:
            fail(f"{label} non-cancelled release follows cancellation")
        if event["outcome"] == "cancelled":
            cancel = next(
                item
                for item in trace.events
                if item["event"] == "request_cancel_requested"
            )
            if event["cancel_reason"] != cancel["reason"]:
                fail(f"{label} does not retain the cancellation reason")
        trace.terminal = name
    elif name == "worker_fatal":
        if "request_started" not in names:
            fail(f"{label} active worker_fatal precedes start")
        trace.terminal = name
    trace.events.append(event)


def validate_service_journal(
    root: Path,
    expected: dict[str, GatewayEvidence],
    boot_id: str,
    final_cursor: str,
    quiet_checks: Sequence[LifecycleQuietCheckData] = (),
    api_journal_observations: Sequence[ApiJournalObservationData] = (),
) -> None:
    remaining = dict(expected)
    observations_by_cursor: dict[str, ApiJournalObservationData] = {}
    for expected_observation in api_journal_observations:
        if expected_observation.journal_cursor in observations_by_cursor:
            fail("API journal observation cursor is duplicated")
        observations_by_cursor[expected_observation.journal_cursor] = (
            expected_observation
        )
    next_observation_index = 0
    observation_span_started = False
    quiet_by_cursor: dict[str, list[LifecycleQuietCheckData]] = defaultdict(list)
    for check in quiet_checks:
        quiet_by_cursor[check.journal_cursor].append(check)
    remaining_quiet_cursors = set(quiet_by_cursor)
    final_seen = False
    seen_cursors: set[str] = set()
    last_cursor: str | None = None
    last_monotonic = -1
    for line_number, record in iter_jsonl(
        root / "service-journal.raw.jsonl", "service-journal.raw.jsonl"
    ):
        label = f"service-journal.raw.jsonl line {line_number}"
        for field in (
            "__CURSOR",
            "__MONOTONIC_TIMESTAMP",
            "_BOOT_ID",
            "_PID",
            "_SYSTEMD_UNIT",
            "PRIORITY",
            "MESSAGE",
        ):
            if field not in record:
                fail(f"{label} lacks required field {field}")
        cursor = string(record["__CURSOR"], f"{label}.__CURSOR")
        if cursor in seen_cursors:
            fail(f"{label} journal cursor is duplicated")
        seen_cursors.add(cursor)
        last_cursor = cursor
        if api_journal_observations and not observation_span_started:
            observation_span_started = (
                cursor == api_journal_observations[0].journal_cursor
            )
        if (
            observation_span_started
            and next_observation_index < len(api_journal_observations)
            and cursor
            != api_journal_observations[next_observation_index].journal_cursor
        ):
            fail(f"{label} interrupts or reorders the copied API journal span")
        if record["_BOOT_ID"] != boot_id:
            fail(f"{label} boot ID differs")
        if record["_SYSTEMD_UNIT"] != "ullm-openai.service":
            fail(f"{label} systemd unit differs")
        monotonic_text = string(
            record["__MONOTONIC_TIMESTAMP"], f"{label}.__MONOTONIC_TIMESTAMP"
        )
        pid_text = string(record["_PID"], f"{label}._PID")
        if not monotonic_text.isdecimal() or not pid_text.isdecimal():
            fail(f"{label} numeric journal fields are invalid")
        monotonic = int(monotonic_text)
        if monotonic < last_monotonic:
            fail(f"{label} journal monotonic timestamps regress")
        last_monotonic = monotonic
        matching_quiet = quiet_by_cursor.get(cursor, ())
        if any(
            monotonic * 1000 > check.checked_monotonic_ns for check in matching_quiet
        ):
            fail(f"{label} quiet-check cursor was observed before its journal record")
        remaining_quiet_cursors.discard(cursor)
        string(record["PRIORITY"], f"{label}.PRIORITY")
        message = string(record["MESSAGE"], f"{label}.MESSAGE", nonempty=False)
        matched_observation = observations_by_cursor.get(cursor)
        if matched_observation is not None:
            message_raw = message.encode("utf-8", errors="strict")
            if (
                monotonic != matched_observation.journal_monotonic_usec
                or int(pid_text) != matched_observation.journal_pid
                or len(message_raw) != matched_observation.message_utf8_bytes
                or hashlib.sha256(message_raw).hexdigest()
                != matched_observation.message_sha256
            ):
                fail(f"{label} API journal observation differs from global journal")
            if matched_observation.observation_index != next_observation_index:
                fail(f"{label} API journal observation order differs")
            next_observation_index += 1
        if cursor == final_cursor:
            final_seen = True
        evidence = remaining.pop(cursor, None)
        if evidence is not None:
            if (
                monotonic != evidence.journal_monotonic_usec
                or int(pid_text) != evidence.journal_pid
            ):
                fail(f"{label} copied numeric journal fields differ")
            if (
                message != evidence.message
                or sha256_text(message) != evidence.message_sha256
            ):
                fail(f"{label} copied MESSAGE bytes/hash differ")
        payload_text: str | None = None
        if message.startswith("{"):
            payload_text = message
        elif message.startswith("INFO:     {"):
            payload_text = message[len("INFO:     ") :]
        if payload_text is not None:
            decoded = decode_json_bytes(
                payload_text.encode("utf-8"), f"{label}.MESSAGE"
            )
            if (
                type(decoded) is dict
                and decoded.get("schema_version") == LIFECYCLE_SCHEMA
            ):
                validate_lifecycle(decoded, f"{label}.MESSAGE")
                if cursor not in expected:
                    fail(
                        f"{label} lifecycle message is omitted from raw-session-results.jsonl"
                    )
    if remaining:
        fail(f"service journal lacks {len(remaining)} copied gateway event cursor(s)")
    if remaining_quiet_cursors:
        fail(
            "service journal lacks one or more API lifecycle quiet-check cursor records"
        )
    if next_observation_index != len(api_journal_observations):
        fail("service journal lacks one or more copied API journal observations")
    if not final_seen:
        fail("service journal lacks the run_end final journal cursor")
    if last_cursor != final_cursor:
        fail("service journal is not bounded exactly by the run_end final cursor")


def _probe_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["control_group"],
        record["gateway_pid"],
        record["gateway_starttime_ticks"],
        record["worker_pid"],
        record["worker_starttime_ticks"],
        record["n_restarts"],
    )


def _validate_probe_boundary(
    probes: dict[str, dict[str, Any]],
) -> tuple[int, int, int, int]:
    required = {
        "normal-segment-start",
        "post-header-restart-ready",
        "restart-segment-start",
        "final-service-ready",
    }
    if set(probes) != required:
        fail("raw session lifecycle probe set differs from phase-1")
    normal = probes["normal-segment-start"]
    post = probes["post-header-restart-ready"]
    restart = probes["restart-segment-start"]
    final = probes["final-service-ready"]
    for name, record in (
        ("normal", normal),
        ("post", post),
        ("restart", restart),
        ("final", final),
    ):
        if record["service_active"] is not True or record["ready_http_status"] != 200:
            fail(f"{name} lifecycle probe is not active and ready")
    if _probe_identity(post) != _probe_identity(restart) or _probe_identity(
        restart
    ) != _probe_identity(final):
        fail("post-restart lifecycle probe identities differ")
    if normal["control_group"] != restart["control_group"]:
        fail("lifecycle probe ControlGroup changes across restart")
    if (
        (normal["gateway_pid"], normal["gateway_starttime_ticks"])
        == (restart["gateway_pid"], restart["gateway_starttime_ticks"])
        or (normal["worker_pid"], normal["worker_starttime_ticks"])
        == (restart["worker_pid"], restart["worker_starttime_ticks"])
        or restart["n_restarts"] != normal["n_restarts"] + 1
    ):
        fail("lifecycle probe restart identity/count boundary differs")
    if not (
        normal["observed_monotonic_ns"]
        <= post["observed_monotonic_ns"]
        <= restart["observed_monotonic_ns"]
        <= final["observed_monotonic_ns"]
    ):
        fail("lifecycle probe timestamps regress across the restart boundary")
    return (
        normal["gateway_pid"],
        restart["gateway_pid"],
        post["observed_monotonic_ns"],
        restart["observed_monotonic_ns"],
    )


def _validate_gateway_event_pids(
    events: dict[str, GatewayEvidence],
    normal_pid: int,
    restart_pid: int,
    normal_epoch_end_ns: int,
    post_header_epoch_end_ns: int,
) -> None:
    normal_phases = {
        "preflight",
        "api_contract",
        "openwebui",
        "cancellation",
        "resource_normal",
    }
    restart_phases = {"resource_restart", "latency", "final"}
    for evidence in events.values():
        if evidence.phase in normal_phases:
            expected = {normal_pid}
        elif evidence.phase in restart_phases:
            expected = {restart_pid}
        elif evidence.phase == "post_header_failure":
            expected = {normal_pid, restart_pid}
        else:
            fail("gateway event phase lacks a process-identity epoch")
        if evidence.journal_pid not in expected:
            fail("gateway event journal PID differs from its lifecycle probe epoch")
        if evidence.event["event"] == "worker_fatal" and not (
            evidence.phase == "post_header_failure"
            and evidence.journal_pid == normal_pid
        ):
            fail("worker_fatal is outside the sole planned post-header failure")
        if (
            evidence.journal_pid == normal_pid
            and evidence.event["observed_monotonic_ns"] > normal_epoch_end_ns
        ):
            fail("normal gateway event exceeds the post-header restart boundary")
        if (
            evidence.phase == "post_header_failure"
            and evidence.event["observed_monotonic_ns"] > post_header_epoch_end_ns
        ):
            fail("post-header gateway event exceeds its lifecycle phase boundary")


def _campaign_terminal(trace: _CampaignTrace) -> tuple[int, dict[str, Any]]:
    terminal = [
        item
        for item in trace.events
        if item[1].get("event") in {"request_released", "worker_fatal"}
    ]
    if len(terminal) != 1 or terminal[0] != trace.events[-1]:
        fail("full campaign request trace terminal placement differs")
    return terminal[0]


def _campaign_successful_release(trace: _CampaignTrace) -> bool:
    _, terminal = _campaign_terminal(trace)
    return (
        terminal["event"] == "request_released"
        and terminal.get("outcome") in {"stop", "length"}
        and terminal.get("reset_complete") is True
        and not any(
            event.get("event") == "request_cancel_requested"
            for _, event in trace.events
        )
    )


def _classify_campaign_cancellation(
    trace: _CampaignTrace,
    browser_actions: list[tuple[str, int]],
) -> str:
    _, terminal = _campaign_terminal(trace)
    if (
        terminal["event"] != "request_released"
        or terminal.get("outcome") != "cancelled"
        or terminal.get("reset_complete") is not True
    ):
        fail("full campaign cancellation target lacks a cancelled/reset release")
    cancel_events = [
        (position, event)
        for position, event in trace.events
        if event.get("event") == "request_cancel_requested"
    ]
    if len(cancel_events) != 1:
        fail("full campaign cancellation target has the wrong cancel cardinality")
    cancel_position, cancel_event = cancel_events[0]
    cancel_observed = integer(
        cancel_event.get("observed_monotonic_ns"),
        "full campaign cancellation observed_monotonic_ns",
    )
    before_cancel = [
        event for position, event in trace.events if position < cancel_position
    ]
    if not any(event.get("event") == "request_started" for event in before_cancel):
        fail("full campaign cancellation target is cancelled before request_started")

    stop_actions = [
        (action, completed_ns)
        for action, completed_ns in browser_actions
        if action in {"wait_visible", "click_stop"}
    ]
    if stop_actions:
        wait_times = [
            completed_ns
            for action, completed_ns in stop_actions
            if action == "wait_visible"
        ]
        click_times = [
            completed_ns
            for action, completed_ns in stop_actions
            if action == "click_stop"
        ]
        if (
            len(wait_times) != 1
            or len(click_times) != 1
            or not wait_times[0] < click_times[0] < cancel_observed
        ):
            fail("full campaign OpenWebUI Stop action order differs")
        return "openwebui_stop_after_visible_content"

    if any(event.get("event") == "request_first_token" for event in before_cancel):
        return "decode_after_first_content"
    progress = [
        integer(
            event.get("processed_prompt_tokens"),
            "full campaign cancellation progress.processed_prompt_tokens",
        )
        for event in before_cancel
        if event.get("event") == "request_progress"
    ]
    if not progress:
        return "after_started_before_progress"
    if progress[-1] == 128 and max(progress) == 128:
        return "prefill_after_128"
    if progress[-1] == 2048 and max(progress) == 2048:
        return "prefill_after_2048"
    fail("full campaign cancellation progress boundary is not frozen")
    raise AssertionError("unreachable")


def validate_full_campaign_order(
    records: Iterable[dict[str, Any]],
) -> FullCampaignOrderResult:
    """Validate the full-run order before the remaining release gates are wired.

    The caller supplies already schema-validated raw session records in file order.
    Phase-1 validation deliberately does not call this helper because a phase-1
    bundle omits the browser, cancellation, and latency phases by definition.
    """

    materialized = list(records)
    if not materialized:
        fail("full campaign raw session is empty")
    phase_rank = {phase: index for index, phase in enumerate(FULL_CAMPAIGN_PHASE_ORDER)}
    observed_phases: list[str] = []
    last_rank = -1
    header_positions: list[int] = []
    run_end_positions: list[int] = []
    traces: dict[str, _CampaignTrace] = {}
    gateway_records: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    browser_actions: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    http_request_counts: Counter[tuple[str, str]] = Counter()
    probes: dict[str, dict[str, Any]] = {}
    probe_positions: dict[str, int] = {}
    faults: list[tuple[int, dict[str, Any]]] = []

    for position, record in enumerate(materialized):
        label = f"full campaign raw record {position}"
        if type(record) is not dict:
            fail(f"{label} must be an object")
        if record.get("schema_version") != SESSION_SCHEMA:
            fail(f"{label}.schema_version differs")
        if integer(record.get("sequence"), f"{label}.sequence") != position:
            fail(f"{label}.sequence is not contiguous from zero")
        record_type = string(record.get("record_type"), f"{label}.record_type")
        if record_type not in SESSION_FIELDS:
            fail(f"{label}.record_type is unknown")
        phase = string(record.get("phase"), f"{label}.phase")
        if phase not in phase_rank:
            fail(f"{label}.phase is invalid")
        rank = phase_rank[phase]
        if rank < last_rank:
            fail("full campaign GPU-mutating phase order regresses")
        if rank != last_rank:
            observed_phases.append(phase)
            last_rank = rank
        case_value = record.get("case_id")
        if record_type in {"header", "run_end"}:
            if case_value is not None:
                fail(f"{label}.case_id must be null")
            case_id: str | None = None
        else:
            case_id = string(case_value, f"{label}.case_id")

        if record_type == "header":
            header_positions.append(position)
        elif record_type == "run_end":
            run_end_positions.append(position)
        elif record_type == "http_request":
            assert case_id is not None
            http_request_counts[(phase, case_id)] += 1
        elif record_type == "browser_action":
            assert case_id is not None
            action = string(record.get("action"), f"{label}.action")
            completed_ns = integer(
                record.get("completed_monotonic_ns"),
                f"{label}.completed_monotonic_ns",
            )
            browser_actions[(phase, case_id)].append((action, completed_ns))
        elif record_type == "lifecycle_probe":
            probe = string(record.get("probe"), f"{label}.probe")
            if probe in probes:
                fail("full campaign lifecycle probe is duplicated")
            probes[probe] = record
            probe_positions[probe] = position
        elif record_type == "fault_injection":
            faults.append((position, record))
        elif record_type == "gateway_event":
            assert case_id is not None
            event_value = record.get("event")
            if type(event_value) is not dict:
                fail(f"{label}.event must be an object")
            event = cast(dict[str, Any], event_value)
            event_name = string(event.get("event"), f"{label}.event.event")
            if event_name not in LIFECYCLE_FIELDS:
                fail(f"{label}.event.event is unknown")
            integer(
                event.get("observed_monotonic_ns"),
                f"{label}.event.observed_monotonic_ns",
            )
            journal_pid = integer(
                record.get("journal_pid"), f"{label}.journal_pid", minimum=1
            )
            request_id = string(event.get("request_id"), f"{label}.event.request_id")
            completion_id = string(
                event.get("completion_id"), f"{label}.event.completion_id"
            )
            trace = traces.get(request_id)
            if trace is None:
                trace = _CampaignTrace(
                    phase=phase,
                    case_id=case_id,
                    completion_id=completion_id,
                    journal_pid=journal_pid,
                    events=[],
                )
                traces[request_id] = trace
            elif (
                trace.phase != phase
                or trace.case_id != case_id
                or trace.completion_id != completion_id
                or trace.journal_pid != journal_pid
            ):
                fail(
                    "full campaign request changes phase, case, completion, or gateway"
                )
            trace.events.append((position, event))
            gateway_records.append((position, record, event))

    if header_positions != [0] or run_end_positions != [len(materialized) - 1]:
        fail("full campaign header/run_end placement differs")
    if tuple(observed_phases) != FULL_CAMPAIGN_PHASE_ORDER:
        fail("full campaign GPU-mutating phase set/order differs")

    ordered_traces = sorted(traces.values(), key=lambda item: item.events[0][0])
    for trace in ordered_traces:
        if not trace.events or trace.events[0][1].get("event") != "request_admitted":
            fail("full campaign request trace does not begin with request_admitted")
        if (
            sum(event.get("event") == "request_admitted" for _, event in trace.events)
            != 1
        ):
            fail("full campaign request trace admission cardinality differs")
        observed_times = [
            integer(
                event.get("observed_monotonic_ns"),
                "full campaign lifecycle observed_monotonic_ns",
            )
            for _, event in trace.events
        ]
        if observed_times != sorted(observed_times):
            fail("full campaign request lifecycle timestamps regress")
        _campaign_terminal(trace)

    if any(trace.phase == "api_contract" for trace in ordered_traces):
        fail("full campaign API contract phase produced a worker lifecycle admission")

    openwebui_traces = [trace for trace in ordered_traces if trace.phase == "openwebui"]
    if (
        len(openwebui_traces)
        != 1 + integer(SCHEDULE["openwebui_chats"], "frozen OpenWebUI chat count")
        or len({trace.case_id for trace in openwebui_traces}) != len(openwebui_traces)
        or not all(_campaign_successful_release(trace) for trace in openwebui_traces)
    ):
        fail("full campaign OpenWebUI smoke/20-chat cardinality or outcome differs")

    cancellation_traces = [
        trace for trace in ordered_traces if trace.phase == "cancellation"
    ]
    if len(cancellation_traces) != 2 * len(CANCEL_PHASES) or len(
        {trace.case_id for trace in cancellation_traces}
    ) != len(cancellation_traces):
        fail("full campaign cancellation target/recovery cardinality differs")
    classified: list[str] = []
    for pair_index in range(len(CANCEL_PHASES)):
        target = cancellation_traces[pair_index * 2]
        recovery = cancellation_traces[pair_index * 2 + 1]
        target_actions = browser_actions.get((target.phase, target.case_id), [])
        classified.append(_classify_campaign_cancellation(target, target_actions))
        if not _campaign_successful_release(recovery):
            fail(
                "full campaign cancellation target lacks immediate successful recovery"
            )
        target_http_count = http_request_counts[(target.phase, target.case_id)]
        if pair_index < 4:
            if target_http_count != 1 or any(
                action == "click_stop" for action, _ in target_actions
            ):
                fail("full campaign direct cancellation transport differs")
        elif target_http_count != 0 or not any(
            action == "click_stop" for action, _ in target_actions
        ):
            fail("full campaign OpenWebUI Stop transport differs")
    if tuple(classified) != CANCEL_PHASES:
        fail("full campaign cancellation phase order differs")

    expected_probe_phases = {
        "normal-segment-start": "resource_normal",
        "post-header-restart-ready": "post_header_failure",
        "restart-segment-start": "resource_restart",
        "final-service-ready": "final",
    }
    for probe, expected_phase in expected_probe_phases.items():
        if probe not in probes or probes[probe].get("phase") != expected_phase:
            fail("full campaign lifecycle probe phase differs")
    (
        normal_gateway_pid,
        restart_gateway_pid,
        normal_epoch_end_ns,
        post_header_epoch_end_ns,
    ) = _validate_probe_boundary(probes)
    normal_probe = probes["normal-segment-start"]
    post_probe = probes["post-header-restart-ready"]
    restart_probe = probes["restart-segment-start"]

    evidence = {
        f"full-campaign-{position}": GatewayEvidence(
            cursor=f"full-campaign-{position}",
            journal_monotonic_usec=integer(
                event["observed_monotonic_ns"], "full campaign gateway timestamp"
            )
            // 1000,
            journal_pid=integer(
                record["journal_pid"], "full campaign gateway journal_pid", minimum=1
            ),
            message="",
            message_sha256="0" * 64,
            event=event,
            phase=string(record["phase"], "full campaign gateway phase"),
        )
        for position, record, event in gateway_records
    }
    _validate_gateway_event_pids(
        evidence,
        normal_gateway_pid,
        restart_gateway_pid,
        normal_epoch_end_ns,
        post_header_epoch_end_ns,
    )

    fatal_records = [
        (position, record, event)
        for position, record, event in gateway_records
        if event.get("event") == "worker_fatal"
    ]
    if len(fatal_records) != 1 or len(faults) != 1:
        fail("full campaign must contain exactly one fault and worker_fatal")
    fatal_position, fatal_record, fatal_event = fatal_records[0]
    fault_position, fault = faults[0]
    if (
        fault.get("phase") != "post_header_failure"
        or fault.get("injection") != "post_header_worker_kill"
        or fault.get("signal") != "SIGKILL"
        or fault_position >= fatal_position
        or fatal_record.get("phase") != "post_header_failure"
        or integer(fatal_record.get("journal_pid"), "full campaign fatal PID")
        != normal_gateway_pid
        or integer(fault.get("target_pid"), "full campaign fault target PID")
        != normal_probe["worker_pid"]
        or integer(
            fault.get("target_starttime_ticks"),
            "full campaign fault target starttime",
        )
        != normal_probe["worker_starttime_ticks"]
    ):
        fail("full campaign planned post-header fault identity/order differs")
    fault_started = integer(
        fault.get("started_monotonic_ns"), "full campaign fault start"
    )
    fault_completed = integer(
        fault.get("completed_monotonic_ns"), "full campaign fault completion"
    )
    fatal_observed = integer(
        fatal_event.get("observed_monotonic_ns"), "full campaign fatal timestamp"
    )
    if not (
        fault_started
        <= fault_completed
        <= fatal_observed
        <= post_probe["observed_monotonic_ns"]
    ):
        fail("full campaign post-header fault/restart timestamps differ")

    post_traces = [
        trace for trace in ordered_traces if trace.phase == "post_header_failure"
    ]
    if (
        len(post_traces) != 2
        or _campaign_terminal(post_traces[0])[1].get("event") != "worker_fatal"
        or not _campaign_successful_release(post_traces[1])
        or post_traces[0].case_id != string(fault.get("case_id"), "fault case_id")
        or post_traces[1].events[0][0] <= probe_positions["post-header-restart-ready"]
    ):
        fail("full campaign post-header failure/recovery order differs")

    post_rank = phase_rank["post_header_failure"]
    for position, record, _ in gateway_records:
        phase = string(record["phase"], "full campaign gateway phase")
        pid = integer(record["journal_pid"], "full campaign gateway PID", minimum=1)
        if phase_rank[phase] < post_rank:
            expected_pid = normal_gateway_pid
        elif phase_rank[phase] > post_rank:
            expected_pid = restart_gateway_pid
        else:
            expected_pid = (
                normal_gateway_pid
                if position <= fatal_position
                else restart_gateway_pid
            )
        if pid != expected_pid:
            fail(
                "full campaign gateway epoch changes outside the sole restart boundary"
            )

    return FullCampaignOrderResult(
        phases=tuple(observed_phases),
        openwebui_successful_requests=len(openwebui_traces),
        cancellation_phases=tuple(classified),
        normal_gateway_pid=normal_gateway_pid,
        restart_gateway_pid=restart_gateway_pid,
        normal_worker_pid=integer(
            normal_probe["worker_pid"], "normal lifecycle worker_pid", minimum=1
        ),
        restart_worker_pid=integer(
            restart_probe["worker_pid"], "restart lifecycle worker_pid", minimum=1
        ),
        restart_count_before=integer(
            normal_probe["n_restarts"], "normal lifecycle n_restarts"
        ),
        restart_count_after=integer(
            restart_probe["n_restarts"], "restart lifecycle n_restarts"
        ),
    )


def _expected_resource_http_cases() -> list[tuple[str, str, int, str]]:
    expected: list[tuple[str, str, int, str]] = []
    for index in range(1, 11):
        expected.append(
            ("resource_normal", f"normal-warmup-{index:02d}", index, "normal_warmup")
        )
    negatives = {
        25: ("negative-after-025-context_overflow_1", "context_overflow"),
        50: ("negative-after-050-malformed_json", "malformed_json"),
        75: ("negative-after-075-context_overflow_2", "context_overflow"),
    }
    for index in range(1, 101):
        expected.append(
            (
                "resource_normal",
                f"normal-measured-{index:03d}",
                index,
                "normal_measured",
            )
        )
        if index in negatives:
            case_id, kind = negatives[index]
            expected.append(("resource_normal", case_id, index, kind))
    for index in range(1, 11):
        expected.append(
            ("resource_restart", f"restart-warmup-{index:02d}", index, "restart_warmup")
        )
    for index in range(1, 21):
        expected.append(
            (
                "resource_restart",
                f"restart-measured-{index:03d}",
                index,
                "restart_measured",
            )
        )
    return expected


def _validate_resource_http_schedule(
    state: HttpValidationState, traces: dict[str, RequestTrace]
) -> None:
    resource_keys = [
        key
        for key in state.ordered_keys
        if state.requests[key]["phase"] in {"resource_normal", "resource_restart"}
    ]
    expected = _expected_resource_http_cases()
    if len(resource_keys) != len(expected):
        fail("resource raw HTTP request count differs from the frozen schedule")
    prior_release_ns = -1
    for position, (key, (phase, case_id, request_index, kind)) in enumerate(
        zip(resource_keys, expected, strict=True)
    ):
        request = state.requests[key]
        if (
            key != f"p8f-{case_id}"
            or request["phase"] != phase
            or request["case_id"] != case_id
            or request["request_index"] != request_index
            or request["kind"] != kind
        ):
            fail("resource raw HTTP order/identity differs from the frozen schedule")
        matching = [
            trace
            for trace in traces.values()
            if trace.phase == phase and trace.case_id == case_id
        ]
        if kind in {"context_overflow", "malformed_json"}:
            if matching:
                fail("negative resource request produced a worker admission lifecycle")
            if position + 1 >= len(resource_keys):
                fail("negative resource request lacks a following recovery request")
            following_phase, following_case_id, _, _ = expected[position + 1]
            following_traces = [
                trace
                for trace in traces.values()
                if trace.phase == following_phase and trace.case_id == following_case_id
            ]
            if len(following_traces) != 1:
                fail("negative resource request lacks one following lifecycle trace")
            quiet_end = following_traces[0].events[0]["observed_monotonic_ns"]
            if any(
                request["connect_ns"]
                <= trace.events[0]["observed_monotonic_ns"]
                < quiet_end
                for trace in traces.values()
            ):
                fail("negative resource request interval contains a worker admission")
            continue
        if len(matching) != 1:
            fail("positive resource HTTP request lacks exactly one lifecycle trace")
        trace = matching[0]
        if request.get("completion_id") != trace.completion_id:
            fail("resource SSE completion ID differs from gateway lifecycle")
        if trace.events[0]["observed_monotonic_ns"] < request["sent_ns"]:
            fail("resource admission precedes the final request-body send boundary")
        release = trace.events[-1]
        if (
            trace.terminal != "request_released"
            or release["outcome"] != "length"
            or release["completion_tokens"] != 2
            or release["reset_complete"] is not True
        ):
            fail("positive resource HTTP request lacks a length/two/reset release")
        if request["connect_ns"] < prior_release_ns:
            fail("next resource HTTP connection begins before the prior release")
        prior_release_ns = release["observed_monotonic_ns"]
    if state.fixture_model is None or state.fixture_messages is None:
        fail("resource HTTP schedule lacks its header-bound fixture")


def _compact_session_order_record(record: dict[str, Any]) -> dict[str, Any]:
    record_type = cast(str, record["record_type"])
    compact: dict[str, Any] = {
        "schema_version": record["schema_version"],
        "record_type": record_type,
        "sequence": record["sequence"],
        "phase": record["phase"],
        "case_id": record["case_id"],
    }
    if record_type == "gateway_event":
        event = cast(dict[str, Any], record["event"])
        compact["journal_pid"] = record["journal_pid"]
        compact["event"] = {
            key: event[key]
            for key in (
                "event",
                "observed_monotonic_ns",
                "request_id",
                "completion_id",
                "processed_prompt_tokens",
                "outcome",
                "reset_complete",
            )
            if key in event
        }
    elif record_type == "browser_action":
        compact["action"] = record["action"]
        compact["completed_monotonic_ns"] = record["completed_monotonic_ns"]
    elif record_type == "lifecycle_probe":
        compact.update(
            {
                key: record[key]
                for key in (
                    "probe",
                    "observed_monotonic_ns",
                    "service_active",
                    "ready_http_status",
                    "control_group",
                    "gateway_pid",
                    "gateway_starttime_ticks",
                    "worker_pid",
                    "worker_starttime_ticks",
                    "n_restarts",
                )
            }
        )
    elif record_type == "fault_injection":
        compact.update(
            {
                key: record[key]
                for key in (
                    "injection",
                    "target_pid",
                    "target_starttime_ticks",
                    "signal",
                    "started_monotonic_ns",
                    "completed_monotonic_ns",
                )
            }
        )
    return compact


def _claims_full_campaign(records: Iterable[dict[str, Any]]) -> bool:
    """Identify evidence produced by the browser/fault full-run orchestrator.

    The complete validator must reject a ``None`` order result.  This narrower
    discriminator preserves phase-1's foreign-phase resource-window negatives.
    """

    return any(
        record["record_type"] in {"browser_action", "fault_injection"}
        for record in records
    )


def _validate_browser_action_order(actions: Iterable[BrowserActionData]) -> None:
    prior_completed = -1
    prior_index: dict[tuple[str, str], int] = {}
    for action in actions:
        identity = (action.phase, action.browser_case)
        if action.started_monotonic_ns < prior_completed:
            fail("raw browser action timestamps regress or overlap")
        expected_index = prior_index.get(identity, -1) + 1
        if action.action_index != expected_index:
            fail("raw browser action indices are not contiguous per browser case")
        prior_index[identity] = action.action_index
        prior_completed = action.completed_monotonic_ns


def _validate_browser_action_data(
    record: dict[str, Any], phase: str, case_id: str, label: str
) -> BrowserActionData:
    browser_case = bounded_utf8_string(
        record["browser_case"],
        f"{label}.browser_case",
        maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
    )
    action_index = integer(record["action_index"], f"{label}.action_index")
    action = string(record["action"], f"{label}.action")
    if action not in {
        "navigate",
        "select_model",
        "submit_chat",
        "wait_visible",
        "click_stop",
        "wait_failed",
        "wait_ready",
    }:
        fail(f"{label}.action differs")
    selector = (
        None
        if record["selector"] is None
        else bounded_utf8_string(
            record["selector"],
            f"{label}.selector",
            maximum_bytes=MAX_BROWSER_SELECTOR_BYTES,
        )
    )
    input_sha256 = (
        None
        if record["input_sha256"] is None
        else sha256_value(record["input_sha256"], f"{label}.input_sha256")
    )
    started = integer(record["started_monotonic_ns"], f"{label}.started_monotonic_ns")
    completed = integer(
        record["completed_monotonic_ns"], f"{label}.completed_monotonic_ns"
    )
    if completed < started:
        fail(f"{label} browser timing order differs")
    result = exact_fields(
        record["result"],
        {"visible", "enabled", "text_utf8_bytes", "text_sha256"},
        f"{label}.result",
    )
    visible = (
        None
        if result["visible"] is None
        else boolean(result["visible"], f"{label}.result.visible")
    )
    enabled = (
        None
        if result["enabled"] is None
        else boolean(result["enabled"], f"{label}.result.enabled")
    )
    if result["text_utf8_bytes"] is None:
        if result["text_sha256"] is not None:
            fail(f"{label}.result text fields must be null together")
        text_utf8_bytes = None
        text_sha256 = None
    else:
        text_utf8_bytes = integer(
            result["text_utf8_bytes"], f"{label}.result.text_utf8_bytes"
        )
        text_sha256 = sha256_value(result["text_sha256"], f"{label}.result.text_sha256")
    screenshot_value = record["screenshot_file"]
    screenshot_sha_value = record["screenshot_sha256"]
    if screenshot_value is None:
        if screenshot_sha_value is not None:
            fail(f"{label} screenshot fields must be null together")
        screenshot_file = None
        screenshot_sha256 = None
    else:
        screenshot_file = string(screenshot_value, f"{label}.screenshot_file")
        if screenshot_file not in {
            "browser/openwebui-stop-before.png",
            "browser/post-header-failure.png",
        }:
            fail(f"{label}.screenshot_file differs")
        screenshot_sha256 = sha256_value(
            screenshot_sha_value, f"{label}.screenshot_sha256"
        )
    return BrowserActionData(
        phase=phase,
        case_id=case_id,
        browser_case=browser_case,
        action_index=action_index,
        action=action,
        selector=selector,
        input_sha256=input_sha256,
        started_monotonic_ns=started,
        completed_monotonic_ns=completed,
        result_visible=visible,
        result_enabled=enabled,
        result_text_utf8_bytes=text_utf8_bytes,
        result_text_sha256=text_sha256,
        screenshot_file=screenshot_file,
        screenshot_sha256=screenshot_sha256,
    )


def _validate_lifecycle_quiet_check_data(
    record: dict[str, Any], phase: str, case_id: str, label: str
) -> LifecycleQuietCheckData:
    quiet_sequence = integer(
        record["quiet_sequence"], f"{label}.quiet_sequence", maximum=12
    )
    check_label = bounded_utf8_string(
        record["label"],
        f"{label}.label",
        maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
    )
    if phase != "api_contract" or case_id != check_label:
        fail(f"{label} lifecycle quiet-check phase or identity differs")
    checked_ns = integer(
        record["checked_monotonic_ns"], f"{label}.checked_monotonic_ns"
    )
    observer_open = boolean(record["observer_open"], f"{label}.observer_open")
    observer_event_count = integer(
        record["observer_event_count"],
        f"{label}.observer_event_count",
        maximum=MAX_SESSION_RECORDS,
    )
    new_journal_count = integer(
        record["new_journal_record_count"],
        f"{label}.new_journal_record_count",
        maximum=MAX_SESSION_RECORDS,
    )
    journal_count = integer(
        record["journal_record_count"],
        f"{label}.journal_record_count",
        minimum=1,
        maximum=MAX_SESSION_RECORDS,
    )
    journal_cursor = bounded_utf8_string(
        record["journal_cursor"],
        f"{label}.journal_cursor",
        maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
    )
    return LifecycleQuietCheckData(
        phase=phase,
        case_id=case_id,
        quiet_sequence=quiet_sequence,
        label=check_label,
        checked_monotonic_ns=checked_ns,
        observer_open=observer_open,
        observer_event_count=observer_event_count,
        new_journal_record_count=new_journal_count,
        journal_record_count=journal_count,
        journal_cursor=journal_cursor,
    )


def _validate_api_journal_observation_data(
    record: dict[str, Any], phase: str, case_id: str, label: str
) -> ApiJournalObservationData:
    observation_index = integer(
        record["observation_index"],
        f"{label}.observation_index",
        maximum=MAX_SESSION_RECORDS - 1,
    )
    if phase != "api_contract" or case_id != f"api-journal-{observation_index + 1:02d}":
        fail(f"{label} API journal observation phase or identity differs")
    return ApiJournalObservationData(
        phase=phase,
        case_id=case_id,
        observation_index=observation_index,
        journal_cursor=bounded_utf8_string(
            record["journal_cursor"],
            f"{label}.journal_cursor",
            maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
        ),
        journal_monotonic_usec=integer(
            record["journal_monotonic_usec"],
            f"{label}.journal_monotonic_usec",
        ),
        journal_pid=integer(record["journal_pid"], f"{label}.journal_pid", minimum=1),
        message_utf8_bytes=integer(
            record["message_utf8_bytes"],
            f"{label}.message_utf8_bytes",
            maximum=MAX_JSON_BYTES,
        ),
        message_sha256=sha256_value(
            record["message_sha256"], f"{label}.message_sha256"
        ),
    )


def _validate_fault_injection_data(
    record: dict[str, Any], phase: str, case_id: str, label: str
) -> FaultInjectionData:
    if (
        record["injection"] != "post_header_worker_kill"
        or record["signal"] != "SIGKILL"
        or record["command"] != "signal.pidfd_send_signal"
    ):
        fail(f"{label} fault identity differs")
    target_pid = integer(record["target_pid"], f"{label}.target_pid", minimum=1)
    target_starttime_ticks = integer(
        record["target_starttime_ticks"],
        f"{label}.target_starttime_ticks",
        minimum=1,
    )
    command = string(record["command"], f"{label}.command")
    command_raw = command.encode("utf-8")
    started = integer(record["started_monotonic_ns"], f"{label}.started_monotonic_ns")
    completed = integer(
        record["completed_monotonic_ns"], f"{label}.completed_monotonic_ns"
    )
    if completed < started:
        fail(f"{label} fault timing order differs")
    return FaultInjectionData(
        phase=phase,
        case_id=case_id,
        injection="post_header_worker_kill",
        target_pid=target_pid,
        target_starttime_ticks=target_starttime_ticks,
        signal="SIGKILL",
        command_utf8_bytes=len(command_raw),
        command_sha256=hashlib.sha256(command_raw).hexdigest(),
        started_monotonic_ns=started,
        completed_monotonic_ns=completed,
    )


def validate_session(
    root: Path,
    matrix: MatrixData,
    expected_commit: str,
    expected_worker_sha256: str,
    identity: IdentityData | None = None,
) -> SessionData:
    if identity is not None and not isinstance(identity, IdentityData):
        fail("raw-session campaign identity argument differs")
    traces: dict[str, RequestTrace] = {}
    completion_ids: dict[str, str] = {}
    releases_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    journal_events: dict[str, GatewayEvidence] = {}
    http_state: HttpValidationState | None = None
    probes: dict[str, dict[str, Any]] = {}
    order_projection: list[dict[str, Any]] = []
    browser_actions: list[BrowserActionData] = []
    api_journal_observations: list[ApiJournalObservationData] = []
    lifecycle_quiet_checks: list[LifecycleQuietCheckData] = []
    fault_injection: FaultInjectionData | None = None
    counts: Counter[str] = Counter()
    run_id: str | None = None
    boot_id: str | None = None
    final_cursor: str | None = None
    declared_counts: dict[str, Any] | None = None
    last_gateway_observed = -1
    saw_run_end = False

    for line_number, record in iter_jsonl(
        root / "raw-session-results.jsonl", "raw-session-results.jsonl"
    ):
        label = f"raw-session-results.jsonl line {line_number}"
        if line_number > MAX_SESSION_RECORDS:
            fail("raw-session-results.jsonl exceeds its record-count bound")
        reject_key_recursive(record, "passed", label)
        record_type = record.get("record_type")
        if type(record_type) is not str or record_type not in SESSION_FIELDS:
            fail(f"{label}.record_type is unknown")
        exact_fields(record, COMMON_SESSION_FIELDS | SESSION_FIELDS[record_type], label)
        if record["schema_version"] != SESSION_SCHEMA:
            fail(f"{label}.schema_version differs")
        if integer(record["sequence"], f"{label}.sequence") != sum(counts.values()):
            fail(f"{label}.sequence is not contiguous from zero")
        phase = record["phase"]
        if type(phase) is not str or phase not in PHASES:
            fail(f"{label}.phase is invalid")
        case_value = record["case_id"]
        case_id: str | None
        if record_type in {"header", "run_end"}:
            if case_value is not None:
                fail(f"{label}.case_id must be null")
            case_id = None
        else:
            case_id = bounded_utf8_string(
                case_value,
                f"{label}.case_id",
                maximum_bytes=MAX_SESSION_IDENTIFIER_BYTES,
            )
        if saw_run_end:
            fail(f"{label} appears after run_end")
        counts[record_type] += 1

        if record_type == "header":
            if line_number != 1 or counts[record_type] != 1:
                fail("raw-session header must be the first and sole header")
            run_id, boot_id, fixture_seal = _validate_header(
                record, root, matrix, expected_worker_sha256
            )
            if identity is not None:
                identity.validate_session_header(record)
                identity.validate_header_source_inputs(record["input_files"])
            http_state = HttpValidationState(
                fixture_seal=fixture_seal,
                requests={},
                response_started=set(),
                response_ended=set(),
                bodies={},
                ordered_keys=[],
            )
        elif run_id is None:
            fail(f"{label} appears before header")
        elif record_type.startswith("http_"):
            assert http_state is not None
            _validate_http_record(record, label, http_state)
        elif record_type == "gateway_event":
            cursor = string(record["journal_cursor"], f"{label}.journal_cursor")
            if cursor in journal_events:
                fail(f"{label}.journal_cursor is duplicated")
            usec = integer(
                record["journal_monotonic_usec"], f"{label}.journal_monotonic_usec"
            )
            journal_pid = integer(
                record["journal_pid"], f"{label}.journal_pid", minimum=1
            )
            message = string(record["message"], f"{label}.message", nonempty=False)
            message_digest = sha256_value(
                record["message_sha256"], f"{label}.message_sha256"
            )
            if sha256_text(message) != message_digest:
                fail(f"{label}.message_sha256 differs")
            event = decode_lifecycle_message(message, f"{label}.message")
            if not json_equal(event, record["event"]):
                fail(f"{label}.event differs from exactly decoded MESSAGE")
            name, observed = _validate_lifecycle_common(event, f"{label}.event")
            if usec < observed // 1000:
                fail(f"{label} journal observation precedes the lifecycle timestamp")
            if observed < last_gateway_observed:
                fail(f"{label} gateway events are not in monotonic order")
            last_gateway_observed = observed
            _add_lifecycle_event(
                traces, completion_ids, phase, cast(str, case_id), event, label
            )
            if name == "request_released":
                releases_by_phase[phase].append(event)
            journal_events[cursor] = GatewayEvidence(
                cursor, usec, journal_pid, message, message_digest, event, phase
            )
        elif record_type == "browser_action":
            browser_action = _validate_browser_action_data(
                record, phase, cast(str, case_id), label
            )
            if browser_action.screenshot_file is not None:
                if (
                    sha256_file(
                        safe_relative_file(
                            root,
                            browser_action.screenshot_file,
                            browser_action.screenshot_file,
                        )
                    )
                    != browser_action.screenshot_sha256
                ):
                    fail(f"{label}.screenshot_sha256 differs")
            browser_actions.append(browser_action)
        elif record_type == "api_journal_observation":
            api_journal_observations.append(
                _validate_api_journal_observation_data(
                    record, phase, cast(str, case_id), label
                )
            )
        elif record_type == "lifecycle_quiet_check":
            lifecycle_quiet_checks.append(
                _validate_lifecycle_quiet_check_data(
                    record, phase, cast(str, case_id), label
                )
            )
        elif record_type == "lifecycle_probe":
            probe_name = string(record["probe"], f"{label}.probe")
            expected_probe_phases = {
                "normal-segment-start": "resource_normal",
                "post-header-restart-ready": "post_header_failure",
                "restart-segment-start": "resource_restart",
                "final-service-ready": "final",
            }
            if (
                record["case_id"] != probe_name
                or record["phase"] != expected_probe_phases.get(probe_name)
                or probe_name in probes
            ):
                fail(f"{label} lifecycle probe identity is duplicated or differs")
            integer(record["observed_monotonic_ns"], f"{label}.observed_monotonic_ns")
            boolean(record["service_active"], f"{label}.service_active")
            integer(
                record["ready_http_status"], f"{label}.ready_http_status", maximum=599
            )
            string(record["control_group"], f"{label}.control_group")
            for key in (
                "gateway_pid",
                "gateway_starttime_ticks",
                "worker_pid",
                "worker_starttime_ticks",
            ):
                integer(record[key], f"{label}.{key}", minimum=1)
            integer(record["n_restarts"], f"{label}.n_restarts")
            probes[probe_name] = record
        elif record_type == "fault_injection":
            if fault_injection is not None:
                fail(f"{label} fault_injection is duplicated")
            fault_injection = _validate_fault_injection_data(
                record, phase, cast(str, case_id), label
            )
        elif record_type == "run_end":
            saw_run_end = True
            if phase != "final" or counts[record_type] != 1:
                fail(f"{label} run_end placement differs")
            string(record["completed_utc"], f"{label}.completed_utc")
            completed_ns = integer(
                record["completed_monotonic_ns"], f"{label}.completed_monotonic_ns"
            )
            if completed_ns < last_gateway_observed:
                fail(f"{label}.completed_monotonic_ns precedes the final gateway event")
            if (
                git_commit(record["final_git_commit"], f"{label}.final_git_commit")
                != expected_commit
            ):
                fail(f"{label} final commit differs from trusted CLI anchor")
            status = string(
                record["final_git_status_raw"],
                f"{label}.final_git_status_raw",
                nonempty=False,
            )
            if sha256_text(status) != sha256_value(
                record["final_git_status_sha256"], f"{label}.final_git_status_sha256"
            ):
                fail(f"{label}.final_git_status_sha256 differs")
            declared_counts = exact_fields(
                record["record_counts"], set(counts), f"{label}.record_counts"
            )
            final_cursor = string(
                record["final_journal_cursor"], f"{label}.final_journal_cursor"
            )
            if identity is not None:
                identity.validate_run_end(record)
        order_projection.append(_compact_session_order_record(record))

    if (
        run_id is None
        or boot_id is None
        or not saw_run_end
        or final_cursor is None
        or declared_counts is None
    ):
        fail("raw-session-results.jsonl lacks header or run_end")
    assert run_id is not None
    assert boot_id is not None
    assert final_cursor is not None
    assert declared_counts is not None
    if any(
        type(value) is not int for value in declared_counts.values()
    ) or declared_counts != dict(counts):
        fail("run_end.record_counts differs from independently counted raw records")
    if http_state is None:
        fail("raw-session-results.jsonl lacks initialized HTTP validation state")
    assert http_state is not None
    if (
        set(http_state.requests) != http_state.response_ended
        or http_state.bodies
        or http_state.active_key is not None
    ):
        fail(
            "one or more raw HTTP requests lack a complete response start/end correlation"
        )
    for request_id_value, trace in traces.items():
        if trace.terminal is None:
            fail(f"request {request_id_value} lacks a terminal lifecycle event")
    ordered_traces = sorted(
        traces.values(), key=lambda trace: trace.events[0]["observed_monotonic_ns"]
    )
    for prior, following in zip(ordered_traces, ordered_traces[1:]):
        terminal_time = prior.events[-1]["observed_monotonic_ns"]
        following_admission = following.events[0]["observed_monotonic_ns"]
        if following_admission <= terminal_time:
            fail("a request is admitted before the prior lifecycle terminal event")
        prior_release = prior.events[-1]
        if (
            prior.phase == "cancellation"
            and prior_release["event"] == "request_released"
            and prior_release["outcome"] == "cancelled"
        ):
            following_terminal = following.events[-1]
            if (
                following.phase != "cancellation"
                or following_terminal["event"] != "request_released"
                or following_terminal["outcome"] == "cancelled"
            ):
                fail(
                    "a cancellation is not followed by a successful recovery lifecycle"
                )
    (
        normal_gateway_pid,
        restart_gateway_pid,
        normal_epoch_end_ns,
        post_header_epoch_end_ns,
    ) = _validate_probe_boundary(probes)
    if identity is not None:
        identity.validate_initial_probe(probes["normal-segment-start"])
    _validate_gateway_event_pids(
        journal_events,
        normal_gateway_pid,
        restart_gateway_pid,
        normal_epoch_end_ns,
        post_header_epoch_end_ns,
    )
    _validate_resource_http_schedule(http_state, traces)
    _validate_browser_action_order(browser_actions)
    full_campaign_order = (
        validate_full_campaign_order(order_projection)
        if _claims_full_campaign(order_projection)
        else None
    )
    if full_campaign_order is not None:
        api_contract = validate_api_contract_http(http_state)
        validated_quiet_checks = validate_api_contract_quiet_checks(
            lifecycle_quiet_checks,
            api_journal_observations,
            tuple(http_state.completed_results[key] for key in http_state.ordered_keys),
            normal_gateway_pid,
        )
    else:
        if lifecycle_quiet_checks or api_journal_observations:
            fail("partial release evidence contains full-campaign API journal evidence")
        api_contract = None
        validated_quiet_checks = ()
    validate_service_journal(
        root,
        journal_events,
        boot_id,
        final_cursor,
        validated_quiet_checks,
        tuple(api_journal_observations),
    )
    return SessionData(
        run_id=run_id,
        boot_id=boot_id,
        schedule=matrix.schedule,
        thresholds=matrix.thresholds,
        traces=traces,
        releases_by_phase=dict(releases_by_phase),
        journal_events=journal_events,
        final_journal_cursor=final_cursor,
        record_counts=counts,
        http_requests=http_state.requests,
        ordered_http_keys=http_state.ordered_keys,
        probes=probes,
        raw_order_projection=tuple(order_projection),
        http_results=tuple(
            http_state.completed_results[key] for key in http_state.ordered_keys
        ),
        browser_actions=tuple(browser_actions),
        api_journal_observations=tuple(api_journal_observations),
        lifecycle_quiet_checks=validated_quiet_checks,
        fault_injection=fault_injection,
        full_campaign_order=full_campaign_order,
        api_contract=api_contract,
    )


@dataclass(frozen=True)
class ResourcePoint:
    segment: str
    phase: str
    request_index: int | None
    request_id: str | None
    release_outcome: str | None
    release_observed_monotonic_ns: int | None
    idle_settle_started_monotonic_ns: int
    sample_monotonic_ns: tuple[int, ...]
    host_memory: Fraction
    primary_vram: Fraction
    gateway_rss: Fraction
    worker_rss: Fraction
    gateway_threads: Fraction
    gateway_fds: Fraction
    gateway_children: Fraction
    worker_threads: Fraction
    worker_fds: Fraction
    worker_children: Fraction


@dataclass(frozen=True)
class ResourceResult:
    segments: dict[str, dict[str, Any]]
    sample_count: int
    gpu_metric_count: int


def _expected_resource_records() -> Iterator[
    tuple[str, str, str | None, int | None, int | None]
]:
    yield "gpu_metric", "normal", "before", None, None
    for sample_index in range(5):
        yield "resource_sample", "normal", "baseline", None, sample_index
    for request_index in range(1, 101):
        for sample_index in range(5):
            yield (
                "resource_sample",
                "normal",
                "post_release",
                request_index,
                sample_index,
            )
    yield "gpu_metric", "normal", "after", None, None
    yield "gpu_metric", "restart", "before", None, None
    for sample_index in range(5):
        yield "resource_sample", "restart", "baseline", None, sample_index
    for request_index in range(1, 21):
        for sample_index in range(5):
            yield (
                "resource_sample",
                "restart",
                "post_release",
                request_index,
                sample_index,
            )
    yield "gpu_metric", "restart", "after", None, None


def _validate_resource_header(record: dict[str, Any], label: str) -> None:
    exact_fields(record, RESOURCE_HEADER_FIELDS, label)
    if record["schema_version"] != RESOURCE_SCHEMA or record["record_type"] != "header":
        fail(f"{label} schema/record type differs")
    if record["service_unit"] != "ullm-openai.service":
        fail(f"{label}.service_unit differs")
    exact_fields(record["commands"], set(COMMANDS), f"{label}.commands")
    if not json_equal(record["commands"], COMMANDS):
        fail(f"{label}.commands differs from the frozen commands")
    tools_value = exact_fields(
        record["tools"],
        {
            "systemd_major",
            "systemd_version_line",
            "amd_smi_tool",
            "amd_smi_library",
            "rocm",
            "amd_smi_version_output",
        },
        f"{label}.tools",
    )
    if integer(tools_value["systemd_major"], f"{label}.tools.systemd_major") != 255:
        fail(f"{label}.tools.systemd_major differs")
    version_line = string(
        tools_value["systemd_version_line"], f"{label}.tools.systemd_version_line"
    )
    if not version_line.startswith("systemd 255 "):
        fail(f"{label}.tools.systemd_version_line differs")
    expected_versions = {
        "amd_smi_tool": "26.2.2+e1a6bc5663",
        "amd_smi_library": "26.2.2",
        "rocm": "7.2.1",
    }
    for key, expected in expected_versions.items():
        if tools_value[key] != expected:
            fail(f"{label}.tools.{key} differs")
    version_output = string(
        tools_value["amd_smi_version_output"], f"{label}.tools.amd_smi_version_output"
    )
    for expected in expected_versions.values():
        if expected not in version_output:
            fail(f"{label}.tools.amd_smi_version_output lacks {expected}")
    probes = exact_fields(
        record["probes"],
        {
            "cgroup_fs_type",
            "kfd_proc_present",
            "gpu_index",
            "gpu_bdf",
            "gpu_uuid",
            "kfd_gpu_id",
        },
        f"{label}.probes",
    )
    expected_probes = {
        "cgroup_fs_type": "cgroup2fs",
        "kfd_proc_present": True,
        "gpu_index": 2,
        "gpu_bdf": "0000:47:00.0",
        "gpu_uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
        "kfd_gpu_id": 51545,
    }
    if not json_equal(probes, expected_probes):
        fail(f"{label}.probes differs from the frozen R9700 identity")
    exact_fields(record["schedule"], set(RESOURCE_SCHEDULE), f"{label}.schedule")
    if not json_equal(record["schedule"], RESOURCE_SCHEDULE):
        fail(f"{label}.schedule differs from the frozen resource schedule")


def _ascending_unique_pids(value: Any, label: str) -> list[int]:
    if type(value) is not list:
        fail(f"{label} must be an array")
    result = [
        integer(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(value)
    ]
    if result != sorted(set(result)):
        fail(f"{label} must be ascending and unique")
    return result


def _validate_process(value: Any, label: str) -> dict[str, Any]:
    process = exact_fields(value, PROCESS_FIELDS, label)
    integer(process["pid"], f"{label}.pid", minimum=1)
    integer(process["ppid"], f"{label}.ppid", minimum=1)
    exe = string(process["exe"], f"{label}.exe")
    if not exe.startswith("/"):
        fail(f"{label}.exe must be absolute")
    before = integer(
        process["starttime_ticks_before"], f"{label}.starttime_ticks_before", minimum=1
    )
    after = integer(
        process["starttime_ticks_after"], f"{label}.starttime_ticks_after", minimum=1
    )
    if before != after:
        fail(f"{label} starttime changed during sampling")
    rss_kb = integer(process["vmrss_kb"], f"{label}.vmrss_kb")
    rss_bytes = integer(process["vmrss_bytes"], f"{label}.vmrss_bytes")
    if rss_bytes != rss_kb * 1024:
        fail(f"{label}.vmrss_bytes differs from VmRSS kB")
    integer(process["threads"], f"{label}.threads")
    integer(process["fd_count"], f"{label}.fd_count")
    _ascending_unique_pids(process["children"], f"{label}.children")
    return process


def _validate_resource_sample(record: dict[str, Any], label: str) -> dict[str, Any]:
    reject_key_recursive(record, "passed", label)
    exact_fields(record, RESOURCE_SAMPLE_FIELDS, label)
    if (
        record["schema_version"] != RESOURCE_SCHEMA
        or record["record_type"] != "resource_sample"
    ):
        fail(f"{label} schema/record type differs")
    segment = record["segment"]
    phase = record["phase"]
    if (
        type(segment) is not str
        or segment not in {"normal", "restart"}
        or type(phase) is not str
        or phase not in {"baseline", "post_release"}
    ):
        fail(f"{label} segment/phase differs")
    integer(
        record["idle_settle_started_monotonic_ns"],
        f"{label}.idle_settle_started_monotonic_ns",
    )
    integer(record["sample_index"], f"{label}.sample_index", maximum=4)
    integer(record["sample_monotonic_ns"], f"{label}.sample_monotonic_ns")

    if record["phase"] == "baseline":
        for field in (
            "request_index",
            "request_id",
            "release_outcome",
            "release_observed_monotonic_ns",
            "reset_complete",
        ):
            if record[field] is not None:
                fail(f"{label}.{field} must be null for a baseline sample")
    else:
        integer(record["request_index"], f"{label}.request_index", minimum=1)
        string(record["request_id"], f"{label}.request_id")
        release_outcome = string(record["release_outcome"], f"{label}.release_outcome")
        if release_outcome not in {"stop", "length", "cancelled"}:
            fail(f"{label}.release_outcome differs")
        integer(
            record["release_observed_monotonic_ns"],
            f"{label}.release_observed_monotonic_ns",
        )
        if boolean(record["reset_complete"], f"{label}.reset_complete") is not True:
            fail(f"{label}.reset_complete must be true")

    systemd_value = exact_fields(record["systemd"], SYSTEMD_FIELDS, f"{label}.systemd")
    before_group = string(
        systemd_value["control_group_before"], f"{label}.systemd.control_group_before"
    )
    after_group = string(
        systemd_value["control_group_after"], f"{label}.systemd.control_group_after"
    )
    pure_group = PurePosixPath(before_group)
    if (
        not pure_group.is_absolute()
        or ".." in pure_group.parts
        or before_group != after_group
    ):
        fail(f"{label}.systemd control group is unsafe or changed")
    main_before = integer(
        systemd_value["main_pid_before"], f"{label}.systemd.main_pid_before", minimum=1
    )
    main_after = integer(
        systemd_value["main_pid_after"], f"{label}.systemd.main_pid_after", minimum=1
    )
    if main_before != main_after:
        fail(f"{label}.systemd MainPID changed during sampling")
    host = exact_fields(record["host"], HOST_FIELDS, f"{label}.host")
    integer(host["memory_current_bytes"], f"{label}.host.memory_current_bytes")
    gateway = _validate_process(record["gateway"], f"{label}.gateway")
    worker = _validate_process(record["worker"], f"{label}.worker")
    if gateway["pid"] != main_before:
        fail(f"{label} gateway PID differs from systemd MainPID")
    if worker["ppid"] != gateway["pid"] or worker["pid"] not in gateway["children"]:
        fail(f"{label} worker is not a direct gateway child")
    if Path(worker["exe"]).name != "ullm-sq8-worker":
        fail(f"{label} worker executable basename differs")

    gpu = exact_fields(record["gpu"], GPU_FIELDS, f"{label}.gpu")
    if (
        gpu["index"] != 2
        or gpu["bdf"] != "0000:47:00.0"
        or gpu["uuid"] != "a8ff7551-0000-1000-80e9-ddefa2d60f55"
        or gpu["kfd_gpu_id"] != 51545
    ):
        fail(f"{label}.gpu physical identity differs")
    if integer(gpu["process_record_count"], f"{label}.gpu.process_record_count") != 1:
        fail(f"{label}.gpu.process_record_count must equal one")
    if (
        integer(gpu["worker_pid"], f"{label}.gpu.worker_pid", minimum=1)
        != worker["pid"]
    ):
        fail(f"{label}.gpu.worker_pid differs")
    mem_usage = exact_fields(
        gpu["mem_usage"], {"value", "unit"}, f"{label}.gpu.mem_usage"
    )
    primary_vram = integer(mem_usage["value"], f"{label}.gpu.mem_usage.value")
    if mem_usage["unit"] != "B":
        fail(f"{label}.gpu.mem_usage.unit differs")
    if integer(gpu["kfd_vram_bytes"], f"{label}.gpu.kfd_vram_bytes") != primary_vram:
        fail(f"{label} AMD SMI and KFD VRAM differ")
    if _ascending_unique_pids(
        gpu["unrelated_process_pids"], f"{label}.gpu.unrelated_process_pids"
    ):
        fail(f"{label}.gpu.unrelated_process_pids is not empty")
    return record


def _resource_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    gateway = record["gateway"]
    worker = record["worker"]
    return (
        record["systemd"]["control_group_before"],
        gateway["pid"],
        gateway["ppid"],
        gateway["exe"],
        gateway["starttime_ticks_before"],
        worker["pid"],
        worker["ppid"],
        worker["exe"],
        worker["starttime_ticks_before"],
    )


def _point_from_samples(samples: list[dict[str, Any]], label: str) -> ResourcePoint:
    if len(samples) != 5:
        fail(f"{label} must contain exactly five samples")
    first = samples[0]
    stable_fields = {
        "segment",
        "phase",
        "request_index",
        "request_id",
        "release_outcome",
        "release_observed_monotonic_ns",
        "reset_complete",
        "idle_settle_started_monotonic_ns",
    }
    for index, sample in enumerate(samples):
        if sample["sample_index"] != index:
            fail(f"{label} sample indices differ")
        for field in stable_fields:
            if not json_equal(sample[field], first[field]):
                fail(f"{label}.{field} changes within the point")
        if _resource_identity(sample) != _resource_identity(first):
            fail(f"{label} process identity changes within the point")
    starts = [sample["sample_monotonic_ns"] for sample in samples]
    settle_start = first["idle_settle_started_monotonic_ns"]
    if starts[0] - settle_start < 5_000_000_000:
        fail(f"{label} idle settle is shorter than five seconds")
    for prior, current in zip(starts, starts[1:]):
        if current - prior < 1_000_000_000:
            fail(f"{label} sample interval is shorter than one second")
    if first["phase"] == "post_release":
        release_time = first["release_observed_monotonic_ns"]
        if settle_start < release_time:
            fail(f"{label} settle starts before release")
    return ResourcePoint(
        segment=first["segment"],
        phase=first["phase"],
        request_index=first["request_index"],
        request_id=first["request_id"],
        release_outcome=first["release_outcome"],
        release_observed_monotonic_ns=first["release_observed_monotonic_ns"],
        idle_settle_started_monotonic_ns=settle_start,
        sample_monotonic_ns=tuple(starts),
        host_memory=median(
            sample["host"]["memory_current_bytes"] for sample in samples
        ),
        primary_vram=median(sample["gpu"]["mem_usage"]["value"] for sample in samples),
        gateway_rss=median(sample["gateway"]["vmrss_bytes"] for sample in samples),
        worker_rss=median(sample["worker"]["vmrss_bytes"] for sample in samples),
        gateway_threads=median(sample["gateway"]["threads"] for sample in samples),
        gateway_fds=median(sample["gateway"]["fd_count"] for sample in samples),
        gateway_children=median(
            len(sample["gateway"]["children"]) for sample in samples
        ),
        worker_threads=median(sample["worker"]["threads"] for sample in samples),
        worker_fds=median(sample["worker"]["fd_count"] for sample in samples),
        worker_children=median(len(sample["worker"]["children"]) for sample in samples),
    )


def _validate_gpu_metric(root: Path, record: dict[str, Any], label: str) -> None:
    reject_key_recursive(record, "passed", label)
    exact_fields(record, GPU_METRIC_FIELDS, label)
    if (
        record["schema_version"] != RESOURCE_SCHEMA
        or record["record_type"] != "gpu_metric"
    ):
        fail(f"{label} schema/record type differs")
    segment = record["segment"]
    boundary = record["boundary"]
    if (
        type(segment) is not str
        or segment not in {"normal", "restart"}
        or type(boundary) is not str
        or boundary not in {"before", "after"}
    ):
        fail(f"{label} segment/boundary differs")
    integer(record["captured_monotonic_ns"], f"{label}.captured_monotonic_ns")
    if integer(record["gpu_index"], f"{label}.gpu_index") != 2:
        fail(f"{label}.gpu_index differs")
    expected_name = f"amd-smi-metric-{segment}-{boundary}.json"
    if record["raw_output_file"] != expected_name:
        fail(f"{label}.raw_output_file differs")
    digest = sha256_value(record["raw_output_sha256"], f"{label}.raw_output_sha256")
    path = safe_relative_file(root, expected_name, expected_name)
    if sha256_file(path) != digest:
        fail(f"{label}.raw_output_sha256 differs")
    validate_json_document(path, expected_name)


def _phase_release_trace(session: SessionData, request_id_value: str) -> RequestTrace:
    trace = session.traces.get(request_id_value)
    if trace is None or trace.terminal != "request_released":
        fail(f"resource request {request_id_value} lacks a released lifecycle trace")
    return trace


def _validate_resource_lifecycle(
    session: SessionData,
    segment: str,
    baseline: ResourcePoint,
    points: list[ResourcePoint],
) -> None:
    phase = "resource_normal" if segment == "normal" else "resource_restart"
    expected_measured = 100 if segment == "normal" else 20
    releases = session.releases_by_phase.get(phase, [])
    if len(releases) != expected_measured + 10:
        fail(
            f"{phase} must contain ten warmup and {expected_measured} measured releases"
        )
    phase_traces = [trace for trace in session.traces.values() if trace.phase == phase]
    if len(phase_traces) != len(releases) or any(
        trace.terminal != "request_released" for trace in phase_traces
    ):
        fail(f"{phase} contains an extra or non-released request lifecycle")
    release_ids = [event["request_id"] for event in releases]
    measured_ids = [point.request_id for point in points]
    if release_ids[10:] != measured_ids:
        fail(f"{phase} measured release order differs from resource points")
    if baseline.idle_settle_started_monotonic_ns < releases[9]["observed_monotonic_ns"]:
        fail(f"{phase} baseline settle starts before the tenth warmup release")
    all_admissions = sorted(
        trace.events[0]["observed_monotonic_ns"]
        for trace in session.traces.values()
        if trace.events and trace.events[0]["event"] == "request_admitted"
    )
    quiet_intervals = [
        (releases[9]["observed_monotonic_ns"], baseline.sample_monotonic_ns[-1])
    ] + [
        (event["observed_monotonic_ns"], point.sample_monotonic_ns[-1])
        for point, event in zip(points, releases[10:], strict=True)
    ]
    for interval_start, interval_end in quiet_intervals:
        if any(
            interval_start < admitted <= interval_end for admitted in all_admissions
        ):
            fail(
                f"{phase} has a request admission during a frozen idle/sample interval"
            )
    ordered_traces: list[RequestTrace] = []
    for event in releases:
        trace = _phase_release_trace(session, event["request_id"])
        ordered_traces.append(trace)
        if trace.phase != phase:
            fail(f"{phase} lifecycle trace phase differs")
        admitted = trace.events[0]
        if (
            admitted["event"] != "request_admitted"
            or admitted["stream"] is not True
            or admitted["max_completion_tokens"] != 2
        ):
            fail(f"{phase} resource request admission parameters differ")
        if event["reset_complete"] is not True:
            fail(f"{phase} resource release reset acknowledgement differs")
    for prior_event, next_trace in zip(releases, ordered_traces[1:]):
        if (
            next_trace.events[0]["observed_monotonic_ns"]
            < prior_event["observed_monotonic_ns"]
        ):
            fail(f"{phase} requests overlap")
    first_measured_admission = ordered_traces[10].events[0]["observed_monotonic_ns"]
    if first_measured_admission < baseline.sample_monotonic_ns[-1]:
        fail(f"{phase} admits the first measured request during baseline sampling")
    for point, event in zip(points, releases[10:], strict=True):
        if point.release_observed_monotonic_ns != event["observed_monotonic_ns"]:
            fail(f"{phase} release observation timestamp differs")
        if point.release_outcome != event["outcome"]:
            fail(f"{phase} release outcome differs")
    for point, next_trace in zip(points, ordered_traces[11:]):
        if (
            next_trace.events[0]["observed_monotonic_ns"]
            < point.sample_monotonic_ns[-1]
        ):
            fail(f"{phase} admits a request during post-release resource sampling")


def _segment_metrics(
    baseline: ResourcePoint, points: list[ResourcePoint], label: str
) -> dict[str, Any]:
    expected_points = 100 if label == "normal" else 20
    if len(points) != expected_points:
        fail(f"{label} resource point count differs")
    diagnostic_names = (
        "gateway_threads",
        "gateway_fds",
        "gateway_children",
        "worker_threads",
        "worker_fds",
        "worker_children",
    )
    for point in points:
        for name in diagnostic_names:
            if getattr(point, name) != getattr(baseline, name):
                fail(
                    f"{label} {name} median differs from its segment baseline at request {point.request_index}"
                )

    host_values = [point.host_memory for point in points]
    vram_values = [point.primary_vram for point in points]
    gateway_rss_values = [point.gateway_rss for point in points]
    worker_rss_values = [point.worker_rss for point in points]
    host_final_delta = host_values[-1] - baseline.host_memory
    vram_final_delta = vram_values[-1] - baseline.primary_vram
    host_slope = theil_sen(host_values)
    vram_slope = theil_sen(vram_values)
    maximum_delta = Fraction(THRESHOLDS["final_delta_max_bytes"])
    maximum_slope = Fraction(THRESHOLDS["theil_sen_max_bytes_per_request"])
    if host_final_delta > maximum_delta:
        fail(f"{label} final MemoryCurrent delta exceeds the release threshold")
    if vram_final_delta > maximum_delta:
        fail(f"{label} final process VRAM delta exceeds the release threshold")
    if host_slope > maximum_slope:
        fail(f"{label} MemoryCurrent Theil-Sen slope exceeds the release threshold")
    if vram_slope > maximum_slope:
        fail(f"{label} process VRAM Theil-Sen slope exceeds the release threshold")
    return {
        "point_count": len(points),
        "baseline": {
            "memory_current_bytes": fraction_json(baseline.host_memory),
            "process_vram_bytes": fraction_json(baseline.primary_vram),
            "gateway_rss_bytes": fraction_json(baseline.gateway_rss),
            "worker_rss_bytes": fraction_json(baseline.worker_rss),
            "gateway_threads": fraction_json(baseline.gateway_threads),
            "gateway_fds": fraction_json(baseline.gateway_fds),
            "gateway_children": fraction_json(baseline.gateway_children),
            "worker_threads": fraction_json(baseline.worker_threads),
            "worker_fds": fraction_json(baseline.worker_fds),
            "worker_children": fraction_json(baseline.worker_children),
        },
        "final_delta": {
            "memory_current_bytes": fraction_json(host_final_delta),
            "process_vram_bytes": fraction_json(vram_final_delta),
        },
        "theil_sen_bytes_per_request": {
            "memory_current": fraction_json(host_slope),
            "process_vram": fraction_json(vram_slope),
            "gateway_rss_diagnostic": fraction_json(theil_sen(gateway_rss_values)),
            "worker_rss_diagnostic": fraction_json(theil_sen(worker_rss_values)),
        },
    }


def validate_resources(root: Path, session: SessionData) -> ResourceResult:
    iterator = iter_jsonl(root / "soak-resources.raw.jsonl", "soak-resources.raw.jsonl")
    try:
        first_line, header = next(iterator)
    except StopIteration:
        fail("soak-resources.raw.jsonl is empty")
    if first_line != 1:
        fail("resource header line differs")
    reject_key_recursive(header, "passed", "resource header")
    _validate_resource_header(header, "resource header")

    expected = list(_expected_resource_records())
    point_samples: list[dict[str, Any]] = []
    baselines: dict[str, ResourcePoint] = {}
    points: dict[str, list[ResourcePoint]] = {"normal": [], "restart": []}
    identities: dict[str, tuple[Any, ...]] = {}
    sample_count = 0
    metric_count = 0
    metric_times: dict[tuple[str, str], int] = {}

    observed_count = 0
    for observed_count, (line_number, record) in enumerate(iterator, start=1):
        label = f"soak-resources.raw.jsonl line {line_number}"
        if observed_count > len(expected):
            fail(f"{label} is an extra resource record")
        (
            expected_type,
            expected_segment,
            expected_phase,
            expected_request,
            expected_sample,
        ) = expected[observed_count - 1]
        if record.get("record_type") != expected_type:
            fail(f"{label}.record_type violates the exact resource state machine")
        if record.get("segment") != expected_segment:
            fail(f"{label}.segment violates the exact resource state machine")
        if expected_type == "gpu_metric":
            if record.get("boundary") != expected_phase:
                fail(f"{label}.boundary violates the exact resource state machine")
            _validate_gpu_metric(root, record, label)
            metric_times[(expected_segment, expected_phase)] = record[
                "captured_monotonic_ns"
            ]
            metric_count += 1
            continue

        _validate_resource_sample(record, label)
        if (
            record["phase"] != expected_phase
            or record["request_index"] != expected_request
            or record["sample_index"] != expected_sample
        ):
            fail(f"{label} violates the exact resource sample state machine")
        identity = _resource_identity(record)
        previous_identity = identities.setdefault(expected_segment, identity)
        if identity != previous_identity:
            fail(
                f"{label} process identity changes within the {expected_segment} segment"
            )
        point_samples.append(record)
        sample_count += 1
        if len(point_samples) == 5:
            point = _point_from_samples(
                point_samples,
                f"{expected_segment} {expected_phase} point {expected_request}",
            )
            if expected_phase == "baseline":
                if expected_segment in baselines:
                    fail(f"{expected_segment} baseline is duplicated")
                baselines[expected_segment] = point
            else:
                points[expected_segment].append(point)
            point_samples = []

    if observed_count != len(expected):
        fail(
            f"resource record count differs: expected {len(expected) + 1} total records"
        )
    if (
        point_samples
        or sample_count != 610
        or metric_count != 4
        or set(baselines) != {"normal", "restart"}
    ):
        fail("resource 1+610+4 state machine is incomplete")
    normal_identity = identities["normal"]
    restart_identity = identities["restart"]
    if normal_identity[0] != restart_identity[0]:
        fail("systemd ControlGroup changes across the planned service restart")
    if (normal_identity[1], normal_identity[4]) == (
        restart_identity[1],
        restart_identity[4],
    ) or (normal_identity[5], normal_identity[8]) == (
        restart_identity[5],
        restart_identity[8],
    ):
        fail(
            "gateway and worker identities must both change across the planned restart"
        )

    for segment, probe_name in (
        ("normal", "normal-segment-start"),
        ("restart", "restart-segment-start"),
    ):
        identity = identities[segment]
        probe = session.probes[probe_name]
        if (
            identity[0] != probe["control_group"]
            or identity[1] != probe["gateway_pid"]
            or identity[4] != probe["gateway_starttime_ticks"]
            or identity[5] != probe["worker_pid"]
            or identity[8] != probe["worker_starttime_ticks"]
        ):
            fail(f"{segment} resource identity differs from its lifecycle probe")

    for segment in ("normal", "restart"):
        before = metric_times[(segment, "before")]
        after = metric_times[(segment, "after")]
        baseline = baselines[segment]
        lifecycle_phase = (
            "resource_normal" if segment == "normal" else "resource_restart"
        )
        first_release = session.releases_by_phase[lifecycle_phase][0]
        first_trace = session.traces[first_release["request_id"]]
        first_admission = first_trace.events[0]["observed_monotonic_ns"]
        if before > first_admission:
            fail(f"{segment} gpu metric-before occurs after the first warmup admission")
        if before > baseline.idle_settle_started_monotonic_ns:
            fail(f"{segment} gpu metric-before occurs after baseline settle start")
        if after < points[segment][-1].sample_monotonic_ns[-1]:
            fail(f"{segment} gpu metric-after occurs before the final resource sample")
        expected_trace_ids = {
            request_id_value
            for request_id_value, trace in session.traces.items()
            if trace.phase == lifecycle_phase
        }
        if any(
            trace.events[0]["observed_monotonic_ns"] <= after
            and trace.events[-1]["observed_monotonic_ns"] >= before
            and request_id_value not in expected_trace_ids
            for request_id_value, trace in session.traces.items()
        ):
            fail(f"{segment} resource metric window contains a foreign lifecycle trace")
        expected_http_keys = {
            key
            for key, request in session.http_requests.items()
            if request["phase"] == lifecycle_phase
        }
        if any(
            request["connect_ns"] <= after
            and request["response_end_ns"] >= before
            and key not in expected_http_keys
            for key, request in session.http_requests.items()
        ):
            fail(f"{segment} resource metric window contains a foreign HTTP request")
        _validate_resource_lifecycle(session, segment, baseline, points[segment])
    if metric_times[("restart", "before")] < metric_times[("normal", "after")]:
        fail("restart metric-before occurs before normal metric-after")

    reports = {
        segment: _segment_metrics(baselines[segment], points[segment], segment)
        for segment in ("normal", "restart")
    }
    return ResourceResult(
        segments=reports, sample_count=sample_count, gpu_metric_count=metric_count
    )


FULL_DERIVED_VIEW_PATHS = (
    "sampling-results.json",
    "cancel-results.json",
    "prefill-latency-results.json",
    "api-contract-results.json",
    "openwebui-smoke.json",
    "soak-results.json",
)
RELEASE_VALIDATION_FILE = "release-validation.json"


def _directory_anchor(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)


def _read_bounded_root_file(root: Path, relative: str, label: str) -> bytes:
    if "/" in relative or relative in {"", ".", ".."}:
        fail(f"{label} is not a root evidence filename")
    root_fd = -1
    descriptor = -1
    try:
        root_fd = os.open(root, _source_directory_flags())
        root_identity = _directory_anchor(os.fstat(root_fd))
        before = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= MAX_JSON_BYTES
        ):
            fail(f"{label} is not one bounded regular file")
        descriptor = os.open(relative, _source_file_flags(), dir_fd=root_fd)
        identity = _source_stat_identity(before)
        if _source_stat_identity(os.fstat(descriptor)) != identity:
            fail(f"{label} changed while opening")
        raw = bytearray()
        while True:
            chunk = os.read(
                descriptor,
                min(SOURCE_COPY_CHUNK_BYTES, MAX_JSON_BYTES + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_JSON_BYTES:
                fail(f"{label} exceeds its byte bound")
        if len(raw) != before.st_size:
            fail(f"{label} byte count changed while reading")
        if _source_stat_identity(os.fstat(descriptor)) != identity:
            fail(f"{label} changed while reading")
        after = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
        if _source_stat_identity(after) != identity:
            fail(f"{label} changed after reading")
        if _directory_anchor(os.fstat(root_fd)) != root_identity:
            fail("campaign bundle root changed while reading evidence")
        return bytes(raw)
    except ValidationError:
        raise
    except OSError as error:
        fail(f"failed to read {label}: {error}")
        raise AssertionError("unreachable")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if root_fd >= 0:
            os.close(root_fd)


def _independent_summary(
    run_id: str,
    schedule: dict[str, Any],
    *,
    forbidden_values: tuple[bytes, ...],
) -> bytes:
    schedule_raw = independent_canonical_json_bytes(
        schedule, forbidden_values=forbidden_values
    )
    paths = sorted(BUNDLE_FILES, key=lambda item: item.encode("utf-8"))
    lines = [
        "# SQ8 OpenWebUI full campaign",
        "",
        f"Run ID: `{run_id}`",
        "",
        f"Schedule: `{schedule_raw[:-1].decode('ascii', errors='strict')}`",
        "",
        "Artifacts:",
        *(f"- `{path}`" for path in paths),
        "",
    ]
    raw = "\n".join(lines).encode("ascii", errors="strict")
    if b"passed" in raw.lower() or b"verdict" in raw.lower():
        fail("summary.md contains a producer decision")
    if any(secret in raw for secret in forbidden_values):
        fail("summary.md contains forbidden cleartext")
    return raw


def _validate_full_derived_views(
    root: Path,
    session: SessionData,
    *,
    forbidden_values: tuple[bytes, ...],
) -> tuple[dict[str, FileEvidence], dict[str, Any], dict[str, Any]]:
    try:
        front = reconstruct_front_views(
            cast(Any, session), root, forbidden_values=forbidden_values
        )
        latency = reconstruct_latency_results(session)
        resource = reconstruct_soak_resource_results(root, session)
        soak = {
            "schema_version": SOAK_RESULTS_SCHEMA,
            "browser": {
                "chat_count": 20,
                "cases": front.browser_soak_cases,
            },
            **resource,
        }
        if set(front.canonical_bytes) != {
            "sampling-results.json",
            "cancel-results.json",
            "api-contract-results.json",
            "openwebui-smoke.json",
        }:
            fail("independent front derived-view filename set differs")
        expected = {
            "sampling-results.json": front.canonical_bytes["sampling-results.json"],
            "cancel-results.json": front.canonical_bytes["cancel-results.json"],
            "prefill-latency-results.json": independent_canonical_json_bytes(
                latency, forbidden_values=forbidden_values
            ),
            "api-contract-results.json": front.canonical_bytes[
                "api-contract-results.json"
            ],
            "openwebui-smoke.json": front.canonical_bytes["openwebui-smoke.json"],
            "soak-results.json": independent_canonical_json_bytes(
                soak, forbidden_values=forbidden_values
            ),
        }
    except (IndependentViewError, IndependentMetricsError) as error:
        fail(f"independent derived-view reconstruction failed: {error}")
        raise AssertionError("unreachable")
    if tuple(expected) != FULL_DERIVED_VIEW_PATHS:
        fail("independent derived-view filename set or order differs")
    evidence: dict[str, FileEvidence] = {}
    for relative in FULL_DERIVED_VIEW_PATHS:
        actual = _read_bounded_root_file(root, relative, relative)
        reconstructed = expected[relative]
        if actual != reconstructed:
            fail(f"{relative} differs from independent raw-evidence reconstruction")
        evidence[relative] = FileEvidence(
            len(actual), hashlib.sha256(actual).hexdigest()
        )
    return evidence, latency, resource


def _full_validation_report(
    *,
    matrix: MatrixData,
    identity: IdentityData,
    source: SourceCheckoutData,
    session: SessionData,
    resources: ResourceResult,
    derived: dict[str, FileEvidence],
    latency: dict[str, Any],
    reconstructed_resource: dict[str, Any],
    sums_sha256: str,
) -> dict[str, Any]:
    order = session.full_campaign_order
    api = session.api_contract
    if order is None or api is None:
        fail("raw evidence does not contain one complete full campaign")
    return {
        "schema_version": FULL_REPORT_SCHEMA,
        "release_status": "complete",
        "full_campaign_validated": True,
        "run_id": matrix.run_id,
        "trusted_anchors": {
            "git_commit": identity.expected_commit,
            "worker_binary_sha256": identity.expected_worker_binary_sha256,
        },
        "verified_sha256sums_sha256": sums_sha256,
        "gate_details": {
            "identity": {
                "environment_sha256": identity.environment_sha256,
                "model_identity_sha256": identity.model_identity_sha256,
            },
            "source_checkout": {
                "git_commit": source.git_commit,
                "source_count": source.source_count,
                "all_source_sha256": source.all_source_sha256,
            },
            "full_order": {
                "phases": list(order.phases),
                "openwebui_successful_requests": order.openwebui_successful_requests,
                "cancellation_phases": list(order.cancellation_phases),
                "normal_gateway_pid": order.normal_gateway_pid,
                "restart_gateway_pid": order.restart_gateway_pid,
                "normal_worker_pid": order.normal_worker_pid,
                "restart_worker_pid": order.restart_worker_pid,
                "restart_count_before": order.restart_count_before,
                "restart_count_after": order.restart_count_after,
            },
            "api_contract": {
                "case_count": len(api.cases),
                "quiet_check_count": len(session.lifecycle_quiet_checks),
            },
            "browser": {"action_count": len(session.browser_actions)},
            "latency": {"request_count": latency["request_count"]},
            "resources": {
                "session_sample_count": resources.sample_count,
                "session_gpu_metric_count": resources.gpu_metric_count,
                "reconstructed_sample_count": reconstructed_resource[
                    "resource_sample_count"
                ],
                "reconstructed_gpu_metric_count": reconstructed_resource[
                    "gpu_metric_count"
                ],
            },
            "derived_views": [
                {
                    "path": relative,
                    "bytes": derived[relative].bytes,
                    "sha256": derived[relative].sha256,
                }
                for relative in FULL_DERIVED_VIEW_PATHS
            ],
        },
    }


def _write_release_validation(root: Path, raw: bytes) -> FileEvidence:
    if not raw or len(raw) > MAX_JSON_BYTES:
        fail("release-validation.json byte size differs")
    root_fd = -1
    descriptor = -1
    created = False
    created_anchor: tuple[int, int] | None = None
    try:
        root_fd = os.open(root, _source_directory_flags())
        root_identity = _directory_anchor(os.fstat(root_fd))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if not hasattr(os, "O_NOFOLLOW"):
            fail("O_NOFOLLOW is required for release validation publication")
        flags |= os.O_NOFOLLOW
        descriptor = os.open(RELEASE_VALIDATION_FILE, flags, 0o600, dir_fd=root_fd)
        created = True
        created_metadata = os.fstat(descriptor)
        created_anchor = (created_metadata.st_dev, created_metadata.st_ino)
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(
                descriptor, view[offset : offset + SOURCE_COPY_CHUNK_BYTES]
            )
            if written <= 0:
                fail("release-validation.json write made no progress")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.fstat(root_fd).st_uid
            or metadata.st_gid != os.fstat(root_fd).st_gid
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(raw)
        ):
            fail("release-validation.json created file identity differs")
        identity = _source_stat_identity(metadata)
        os.close(descriptor)
        descriptor = -1
        observed = os.stat(
            RELEASE_VALIDATION_FILE, dir_fd=root_fd, follow_symlinks=False
        )
        if _source_stat_identity(observed) != identity:
            fail("release-validation.json changed after publication")
        if _directory_anchor(os.fstat(root_fd)) != root_identity:
            fail("campaign bundle root changed during validation publication")
        os.fsync(root_fd)
        return FileEvidence(len(raw), hashlib.sha256(raw).hexdigest())
    except BaseException as error:
        if created and root_fd >= 0:
            try:
                cleanup_anchor = created_anchor
                if cleanup_anchor is None and descriptor >= 0:
                    created_stat = os.fstat(descriptor)
                    cleanup_anchor = (created_stat.st_dev, created_stat.st_ino)
                current = os.stat(
                    RELEASE_VALIDATION_FILE,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                if cleanup_anchor != (current.st_dev, current.st_ino):
                    raise ValidationError(
                        "refusing to remove a replaced release-validation.json"
                    )
                os.unlink(RELEASE_VALIDATION_FILE, dir_fd=root_fd)
                os.fsync(root_fd)
            except FileNotFoundError:
                pass
            except (OSError, ValidationError) as cleanup_error:
                error.add_note(
                    "failed to remove the incomplete release-validation.json: "
                    f"{cleanup_error}"
                )
        if isinstance(error, OSError):
            raise ValidationError(
                f"failed to exclusively create release-validation.json: {error}"
            ) from error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if root_fd >= 0:
            os.close(root_fd)


def validate_full_release(
    bundle: Path,
    *,
    expected_commit: str,
    expected_worker_binary_sha256: str,
    repo_root: Path,
    forbidden_values: tuple[bytes, ...] = (),
) -> FileEvidence:
    root = safe_bundle_root(bundle)
    validate_bundle_layout(root)
    sums_sha256 = validate_sha256sums(root)
    matrix = validate_matrix(root)
    identity = validate_campaign_identity(
        root,
        expected_commit=expected_commit,
        expected_worker_binary_sha256=expected_worker_binary_sha256,
    )
    source = validate_campaign_source_checkout(identity, repo_root=repo_root)
    session = validate_session(
        root,
        matrix,
        identity.expected_commit,
        identity.expected_worker_binary_sha256,
        identity,
    )
    if session.full_campaign_order is None or session.api_contract is None:
        fail("raw evidence does not contain one complete full campaign")
    resources = validate_resources(root, session)
    derived, latency, reconstructed_resource = _validate_full_derived_views(
        root, session, forbidden_values=forbidden_values
    )
    expected_summary = _independent_summary(
        matrix.run_id, matrix.schedule, forbidden_values=forbidden_values
    )
    if _read_bounded_root_file(root, "summary.md", "summary.md") != expected_summary:
        fail("summary.md differs from independent reconstruction")
    report = _full_validation_report(
        matrix=matrix,
        identity=identity,
        source=source,
        session=session,
        resources=resources,
        derived=derived,
        latency=latency,
        reconstructed_resource=reconstructed_resource,
        sums_sha256=sums_sha256,
    )
    try:
        raw = independent_canonical_json_bytes(
            report, forbidden_values=forbidden_values
        )
    except IndependentViewError as error:
        fail(f"release validation report serialization failed: {error}")
        raise AssertionError("unreachable")
    return _write_release_validation(root, raw)


class FullCampaignIndependentValidator:
    """Orchestrator adapter for one fail-closed full campaign validation."""

    def __init__(
        self,
        *,
        expected_commit: str,
        expected_worker_binary_sha256: str,
        repo_root: Path | None = None,
        forbidden_values: tuple[bytes, ...] = (),
    ) -> None:
        self.expected_commit = git_commit(
            expected_commit, "expected campaign Git commit"
        )
        self.expected_worker_binary_sha256 = sha256_value(
            expected_worker_binary_sha256,
            "expected campaign worker binary SHA-256",
        )
        self.repo_root = (
            Path(__file__).resolve().parents[1]
            if repo_root is None
            else Path(repo_root)
        )
        try:
            independent_canonical_json_bytes({}, forbidden_values=forbidden_values)
        except IndependentViewError as error:
            fail(f"validator forbidden cleartext contract differs: {error}")
        self.forbidden_values = forbidden_values

    def validate(self, stage_path: Path) -> FileEvidence:
        return validate_full_release(
            stage_path,
            expected_commit=self.expected_commit,
            expected_worker_binary_sha256=self.expected_worker_binary_sha256,
            repo_root=self.repo_root,
            forbidden_values=self.forbidden_values,
        )


def validate_phase1(
    bundle: Path,
    *,
    expected_commit: str,
    expected_worker_binary_sha256: str,
) -> dict[str, Any]:
    trusted_commit = git_commit(expected_commit, "--expected-commit")
    trusted_worker_sha = sha256_value(
        expected_worker_binary_sha256, "--expected-worker-binary-sha256"
    )
    root = safe_bundle_root(bundle)
    validate_bundle_layout(root)
    sums_sha256 = validate_sha256sums(root)
    matrix = validate_matrix(root)
    session = validate_session(root, matrix, trusted_commit, trusted_worker_sha)
    resources = validate_resources(root, session)
    return {
        "schema_version": PHASE1_REPORT_SCHEMA,
        "release_status": "incomplete",
        "phase1_validated": True,
        "run_id": matrix.run_id,
        "trusted_anchors": {
            "git_commit": trusted_commit,
            "worker_binary_sha256": trusted_worker_sha,
        },
        "verified_sha256sums_sha256": sums_sha256,
        "raw_counts": {
            "session_records": sum(session.record_counts.values()),
            "gateway_events": len(session.journal_events),
            "resource_samples": resources.sample_count,
            "gpu_metrics": resources.gpu_metric_count,
        },
        "resource_segments": resources.segments,
        "unimplemented_release_gates": [
            "api_contract",
            "openwebui_browser_smoke_and_20_chat_soak",
            "five_phase_cancellation_and_recovery",
            "post_header_failure_presentation_and_restart",
            "http_sse_ttft_and_decode",
            "aggregate_view_reconstruction",
            "complete_identity_and_source_state",
            "exclusive_release_validation_publication",
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-worker-binary-sha256", required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Git top-level used to verify every recorded campaign source",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="validate implemented phase-1 gates and emit an explicitly incomplete report",
    )
    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if type(value) is Decimal:
        return str(value)
    raise TypeError(f"unsupported JSON output type: {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.phase1_only:
            report = validate_phase1(
                args.bundle,
                expected_commit=args.expected_commit,
                expected_worker_binary_sha256=args.expected_worker_binary_sha256,
            )
            output = _identity_canonical_bytes(report)
        else:
            validator = FullCampaignIndependentValidator(
                expected_commit=args.expected_commit,
                expected_worker_binary_sha256=args.expected_worker_binary_sha256,
                repo_root=args.repo_root,
            )
            validator.validate(args.bundle)
            root = safe_bundle_root(args.bundle)
            output = _read_bounded_root_file(
                root, RELEASE_VALIDATION_FILE, RELEASE_VALIDATION_FILE
            )
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
