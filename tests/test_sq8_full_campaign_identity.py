from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sq8_full_campaign_identity.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


IDENTITY = load_module("test_sq8_full_campaign_identity_module", MODULE_PATH)


def canonical(value: Any) -> bytes:
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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class IdentityFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.product = self.root / "product"
        self.tokenizer = self.root / "tokenizer"
        self.worker = self.root / "bin" / "ullm-sq8-worker"
        self.effective_unit = self.root / "etc" / "ullm-openai.service"
        self.effective_environment = self.root / "etc" / "openai-gateway.env"
        self.repo.mkdir()
        self.product.mkdir()
        self.tokenizer.mkdir()
        self.worker.parent.mkdir()
        self.effective_unit.parent.mkdir()
        self._build_sources()
        self._build_product()
        self._build_tokenizer_and_worker()
        self.inputs = IDENTITY.IdentityBuildInputs(
            repo_root=self.repo,
            product_root=self.product,
            tokenizer_root=self.tokenizer,
            worker_binary=self.worker,
            effective_service_unit=self.effective_unit,
            effective_environment_file=self.effective_environment,
            promotion_validation_result=self.validation_result,
            git_commit="c" * 40,
            git_status_raw=b"",
            captured_utc="2026-07-11T12:00:00Z",
            source_specs=IDENTITY.default_source_specs(self.repo),
        )
        self.openwebui = IDENTITY.OpenWebUIExpectation(
            version="0.9.4-ullm.1",
            source_revision="f" * 40,
            base_image_ref=("ghcr.io/open-webui/open-webui@sha256:" + "1" * 64),
            base_image_digest="sha256:" + "1" * 64,
            base_image_id="sha256:" + "2" * 64,
            derived_image_ref="ullm/open-webui:0.9.4-ullm.1",
            derived_image_id="sha256:" + "3" * 64,
            patch_sha256=sha256(
                (self.repo / IDENTITY.SOURCE_ROLE_PATHS["openwebui_patch"]).read_bytes()
            ),
            patched_middleware_sha256="4" * 64,
            docker_network_id="9" * 64,
        )
        self.hardware = IDENTITY.HardwareExpectation(
            gpu_index=2,
            gpu_bdf="0000:47:00.0",
            gpu_uuid="a8ff7551-0000-1000-80e9-ddefa2d60f55",
            kfd_gpu_id=51545,
            node_id=2,
            partition_id=0,
            systemd_major=255,
            amd_smi_tool="26.2.2+e1a6bc5663",
            amd_smi_library="26.2.2",
            rocm_version="7.2.1",
        )
        self.expectation = IDENTITY.LiveCaptureExpectation(
            service_unit=IDENTITY.SERVICE_UNIT,
            service_user="homelab1",
            service_group="homelab1",
            service_fragment_path=self.effective_unit,
            openwebui=self.openwebui,
            hardware=self.hardware,
        )
        self.probe = FakeProbe(self)
        self.live = IDENTITY.capture_live_identity(self.probe, self.expectation)

    def __enter__(self) -> IdentityFixture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.temporary.cleanup()

    def _write(self, path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def _runtime_environment(self) -> bytes:
        return (
            b"# fixed fake service environment\n"
            + f"ULLM_WORKER_BINARY={self.worker}\n".encode()
            + f"ULLM_PRODUCT_ROOT={self.product}\n".encode()
            + f"ULLM_TOKENIZER_DIR={self.tokenizer}\n".encode()
            + b"ULLM_API_KEY_FILE=/etc/ullm/openai-api-key\n"
            + b"ULLM_GPU_LOCK_FILE=/run/ullm/r9700.lock\n"
            + b"ULLM_BIND_HOST=172.20.0.1\n"
            + b"ULLM_BIND_PORT=8000\n"
        )

    def _build_sources(self) -> None:
        oracle_roles = {
            "serving_fixture_manifest",
            "chat_template_fixture_manifest",
            "runtime_oracle_validation",
        }
        for role, relative in IDENTITY.SOURCE_ROLE_PATHS.items():
            if role in oracle_roles:
                continue
            if role in IDENTITY.TTFT_FIXTURE_IDENTITIES:
                raw = (ROOT / relative).read_bytes()
            elif role == "systemd_service":
                raw = b"[Service]\nUser=homelab1\nUMask=0077\n"
            elif role == "systemd_environment_contract":
                raw = self._runtime_environment()
            elif role == "openwebui_patch":
                raw = b"--- middleware.py\n+++ middleware.py\n@@ fake patch\n"
            else:
                raw = f"fixture source for {role}\n".encode("ascii")
            self._write(self.repo / relative, raw)
        chat_manifest_raw = canonical(
            {"schema_version": "ullm.sq8.chat_template_fixtures.v1"}
        )
        self._write(
            self.repo / IDENTITY.SOURCE_ROLE_PATHS["chat_template_fixture_manifest"],
            chat_manifest_raw,
        )
        runtime_validation_raw = canonical(
            {"schema_version": "ullm.sq8.runtime_oracle_validation.v1"}
        )
        self._write(
            self.repo / IDENTITY.SOURCE_ROLE_PATHS["runtime_oracle_validation"],
            runtime_validation_raw,
        )
        template = b"fake-qwen3-chat-template-v1"
        serving_manifest = {
            "schema_version": "ullm.sq8.serving_fixtures.v1",
            "vllm_identity": {
                "async_scheduling": False,
                "backend": "vLLM",
                "device": {
                    "compute_capability": [12, 0],
                    "gfx": IDENTITY.DEVICE_ARCHITECTURE,
                    "name": "AMD Radeon Graphics",
                    "total_memory_bytes": 34_208_743_424,
                    "visible_device_index": 0,
                },
                "dtype": "bfloat16",
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "max_num_seqs": 1,
                "package_version": "0.23.1rc1.dev618+synthetic",
                "pipeline_parallel_size": 1,
                "python_version": "3.12.3",
                "rocr_visible_devices": "1",
                "runner": "LLM.generate",
                "source_revision_from_package_version": "synthetic",
                "tensor_parallel_size": 1,
                "torch_git_version": "synthetic-torch-git",
                "torch_hip_version": "7.2.53211",
                "torch_version": "2.11.0+synthetic",
                "transformers_version": "5.12.1",
            },
            "tokenizer_identity": {
                "chat_template_sha256": sha256(template),
                "chat_template_utf8_bytes": len(template),
                "files": [{"file": "tokenizer_config.json"}],
                "revision": IDENTITY.MODEL_REVISION,
                "tokenizer_class": "Qwen2Tokenizer",
            },
            "chat_template_fixture": {
                "directory": "chat-template",
                "exact_prompt_lengths": [32, 128, 512, 2048, 3584],
                "manifest_file": "chat-template/manifest.json",
                "manifest_sha256": sha256(chat_manifest_raw),
                "status": "ready_independent_recompute_passed",
                "validator": "tools/validate-sq8-chat-template-fixtures.py",
            },
        }
        self._write(
            self.repo / IDENTITY.SOURCE_ROLE_PATHS["serving_fixture_manifest"],
            canonical(serving_manifest),
        )
        self._write(
            self.effective_unit,
            (self.repo / IDENTITY.SOURCE_ROLE_PATHS["systemd_service"]).read_bytes(),
        )
        self._write(
            self.effective_environment,
            (
                self.repo / IDENTITY.SOURCE_ROLE_PATHS["systemd_environment_contract"]
            ).read_bytes(),
        )

    def _build_product(self) -> None:
        artifact_manifest = {
            "schema_version": IDENTITY.ARTIFACT_SCHEMA,
            "integrity": {"content_sha256": "a" * 64},
            "coverage": {"selected_pair_count": 2},
            "storage": {"total_payload_bytes": 1024},
        }
        package_manifest = {
            "schema_version": IDENTITY.PACKAGE_SCHEMA,
            "tensors": [],
            "passthrough_tensors": [
                {
                    "name": "lm_head.weight",
                    "payload_file": "passthrough/lm-head.raw",
                    "payload_bytes": 11,
                    "payload_sha256": "b" * 64,
                },
                {
                    "name": "model.embed_tokens.weight",
                    "payload_file": "passthrough/embed.raw",
                    "payload_bytes": 13,
                    "payload_sha256": "d" * 64,
                },
            ],
        }
        artifact_raw = canonical(artifact_manifest)
        package_raw = canonical(package_manifest)
        self._write(self.product / "artifact" / "sq_manifest.json", artifact_raw)
        self._write(self.product / "package" / "manifest.json", package_raw)
        promotion = {
            "schema_version": IDENTITY.PROMOTION_SCHEMA,
            "created_at": "2026-07-10T12:16:25+09:00",
            "plan_commit": "dfc63de",
            "model": {
                "id": IDENTITY.UPSTREAM_MODEL_ID,
                "revision": IDENTITY.MODEL_REVISION,
            },
            "artifact": {
                "source": "/tmp/fake-artifact",
                "destination": str(self.product / "artifact"),
                "schema_version": IDENTITY.ARTIFACT_SCHEMA,
                "manifest_bytes": len(artifact_raw),
                "manifest_sha256": sha256(artifact_raw),
                "content_sha256": "a" * 64,
                "selected_pair_count": 2,
                "payload_bytes": 1024,
                "file_count": 3,
                "verified": True,
            },
            "package": {
                "source": "/tmp/fake-package",
                "destination": str(self.product / "package"),
                "schema_version": IDENTITY.PACKAGE_SCHEMA,
                "manifest_bytes": len(package_raw),
                "manifest_sha256": sha256(package_raw),
                "payload_count": 2,
                "file_count": 3,
                "verified": True,
            },
            "copy": {
                "method": "rsync_archive_streaming",
                "source_and_destination_manifests_byte_identical": True,
                "destination_read_only": True,
            },
        }
        self._write(self.product / "promotion.json", canonical(promotion))
        self.validation_result = {
            "schema_version": IDENTITY.PROMOTION_SCHEMA,
            "product_root": str(self.product),
            "created_at": promotion["created_at"],
            "model_revision": IDENTITY.MODEL_REVISION,
            "artifact": {
                "manifest_sha256": sha256(artifact_raw),
                "content_sha256": "a" * 64,
                "selected_pair_count": 2,
                "payloads_hashed": True,
            },
            "package": {
                "manifest_sha256": sha256(package_raw),
                "payload_count": 2,
                "payload_bytes": 24,
                "payloads_hashed": True,
            },
            "read_only": True,
            "full_payloads": True,
            "verified": True,
        }

    def _build_tokenizer_and_worker(self) -> None:
        for name in IDENTITY.TOKENIZER_FILES:
            raw = (
                canonical({"chat_template": "fake-qwen3-chat-template-v1"})
                if name == "tokenizer_config.json"
                else f"tokenizer identity {name}\n".encode()
            )
            self._write(self.tokenizer / name, raw)
        self._write(self.worker, b"fake executable worker identity\n")
        os.chmod(self.worker, 0o755)


class FakeProbe:
    def __init__(self, fixture: IdentityFixture) -> None:
        self.fixture = fixture
        self.service_calls = 0
        self.service_drift = False
        self.starttime_drift = False
        self.bad_gpu = False
        self.bad_image = False
        worker_raw = fixture.worker.read_bytes()
        self.gateway = IDENTITY.ProcessSnapshot(
            pid=1200,
            ppid=1,
            uid=1000,
            gid=1000,
            starttime_ticks=10_000,
            executable="/usr/bin/python3.12",
            executable_bytes=42,
            executable_sha256="6" * 64,
            children=(1201,),
        )
        self.worker = IDENTITY.ProcessSnapshot(
            pid=1201,
            ppid=1200,
            uid=1000,
            gid=1000,
            starttime_ticks=10_001,
            executable=str(fixture.worker),
            executable_bytes=len(worker_raw),
            executable_sha256=sha256(worker_raw),
            children=(),
        )

    def os_release(self) -> bytes:
        return b'PRETTY_NAME="Ubuntu 24.04.4 LTS"\nVERSION_ID="24.04"\nID=ubuntu\n'

    def uname(self) -> tuple[str, str, str, str]:
        return "Linux", "6.17.0-35-generic", "#35-Ubuntu SMP", "x86_64"

    def boot_id(self) -> bytes:
        return b"5" * 32 + b"\n"

    def cgroup_fs_type(self) -> bytes:
        return b"cgroup2fs\n"

    def systemd_version(self) -> bytes:
        return b"systemd 255 (synthetic)\n+PAM +AUDIT\n"

    def python_version(self) -> bytes:
        return b"Python 3.12.3\n"

    def rustc_version(self) -> bytes:
        return b"rustc 1.96.0 (synthetic 2026-05-25)\n"

    def cargo_version(self) -> bytes:
        return b"cargo 1.96.0 (synthetic 2026-05-25)\n"

    def service_show(self, unit: str) -> bytes:
        assert unit == IDENTITY.SERVICE_UNIT
        self.service_calls += 1
        pid = 2200 if self.service_drift and self.service_calls > 1 else 1200
        fields = {
            "MainPID": str(pid),
            "NRestarts": "2",
            "ControlGroup": "/system.slice/ullm-openai.service",
            "User": "homelab1",
            "Group": "homelab1",
            "ActiveState": "active",
            "SubState": "running",
            "FragmentPath": str(self.fixture.effective_unit),
        }
        return "".join(f"{key}={value}\n" for key, value in fields.items()).encode()

    def account_ids(self, user: str) -> tuple[int, int]:
        assert user == "homelab1"
        return 1000, 1000

    def process(self, pid: int) -> Any:
        if pid == self.gateway.pid:
            return self.gateway
        if pid == self.worker.pid:
            return self.worker
        raise AssertionError(f"unexpected process {pid}")

    def process_starttime(self, pid: int) -> int:
        value = self.gateway if pid == self.gateway.pid else self.worker
        return int(value.starttime_ticks) + (
            1 if self.starttime_drift and pid == 1201 else 0
        )

    def docker_version(self) -> bytes:
        return canonical(
            {
                "Version": "29.6.0",
                "ApiVersion": "1.55",
                "Os": "linux",
                "Arch": "amd64",
                "KernelVersion": "6.17.0-35-generic",
            }
        )

    def docker_network(self, name: str) -> bytes:
        assert name == IDENTITY.DOCKER_NETWORK_NAME
        return canonical(
            [
                {
                    "Name": name,
                    "Id": self.fixture.openwebui.docker_network_id,
                    "Driver": "bridge",
                    "IPAM": {
                        "Config": [
                            {
                                "Subnet": IDENTITY.DOCKER_NETWORK_SUBNET,
                                "Gateway": IDENTITY.DOCKER_NETWORK_GATEWAY,
                            }
                        ]
                    },
                }
            ]
        )

    def docker_images(self, base_ref: str, derived_ref: str) -> bytes:
        expected = self.fixture.openwebui
        assert base_ref == expected.base_image_ref
        assert derived_ref == expected.derived_image_ref
        middleware = "0" * 64 if self.bad_image else expected.patched_middleware_sha256
        return canonical(
            [
                {
                    "Id": expected.base_image_id,
                    "RepoDigests": [expected.base_image_ref],
                },
                {
                    "Id": expected.derived_image_id,
                    "Config": {
                        "Labels": {
                            "org.opencontainers.image.version": expected.version,
                            "org.opencontainers.image.revision": expected.source_revision,
                            "org.opencontainers.image.base.digest": expected.base_image_digest,
                            "io.ullm.openwebui.base.image.id": expected.base_image_id,
                            "io.ullm.openwebui.patch.sha256": expected.patch_sha256,
                            "io.ullm.openwebui.middleware.sha256": middleware,
                        }
                    },
                },
            ]
        )

    def amd_smi_version(self) -> bytes:
        hardware = self.fixture.hardware
        return (
            f"AMDSMI Tool: {hardware.amd_smi_tool} | "
            f"AMDSMI Library version: {hardware.amd_smi_library} | "
            f"ROCm version: {hardware.rocm_version}\n"
        ).encode()

    def amd_smi_list(self) -> bytes:
        hardware = self.fixture.hardware
        bdf = "0000:00:00.0" if self.bad_gpu else hardware.gpu_bdf
        return canonical(
            [
                {
                    "gpu": hardware.gpu_index,
                    "bdf": bdf,
                    "uuid": hardware.gpu_uuid,
                    "kfd_id": hardware.kfd_gpu_id,
                    "node_id": hardware.node_id,
                    "partition_id": hardware.partition_id,
                }
            ]
        )


class FullCampaignIdentityTests(unittest.TestCase):
    def test_source_contract_has_exact_unique_group_coverage(self) -> None:
        self.assertEqual(len(IDENTITY.SOURCE_ROLE_PATHS), 68)
        self.assertEqual(
            IDENTITY.SOURCE_GROUPS["all"], tuple(IDENTITY.SOURCE_ROLE_PATHS)
        )
        semantic = set().union(
            *(
                set(roles)
                for group, roles in IDENTITY.SOURCE_GROUPS.items()
                if group != "all"
            )
        )
        self.assertEqual(semantic, set(IDENTITY.SOURCE_ROLE_PATHS))
        IDENTITY._validate_source_contract()

    def test_source_contract_rejects_duplicate_paths_and_bad_groups(self) -> None:
        duplicate_paths = dict(IDENTITY.SOURCE_ROLE_PATHS)
        duplicate_paths["campaign_views"] = duplicate_paths["campaign_renderer"]
        with self.assertRaisesRegex(IDENTITY.IdentityError, "paths are not unique"):
            IDENTITY._validate_source_contract(duplicate_paths, IDENTITY.SOURCE_GROUPS)

        mutations = []
        unknown = dict(IDENTITY.SOURCE_GROUPS)
        unknown["campaign"] = (*unknown["campaign"], "unknown_source")
        mutations.append(unknown)
        duplicate = dict(IDENTITY.SOURCE_GROUPS)
        duplicate["campaign"] = (*duplicate["campaign"], duplicate["campaign"][0])
        mutations.append(duplicate)
        missing = dict(IDENTITY.SOURCE_GROUPS)
        missing["all"] = missing["all"][:-1]
        mutations.append(missing)
        unclassified = dict(IDENTITY.SOURCE_GROUPS)
        unclassified["fixture"] = tuple(
            role for role in unclassified["fixture"] if role != "fixture_ttft_p3584"
        )
        mutations.append(unclassified)
        for groups in mutations:
            with self.subTest(groups=groups), self.assertRaises(IDENTITY.IdentityError):
                IDENTITY._validate_source_contract(IDENTITY.SOURCE_ROLE_PATHS, groups)

    def test_fake_live_capture_builds_both_strict_identity_artifacts(self) -> None:
        with IdentityFixture() as fixture:
            artifacts = IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)

        self.assertEqual(
            artifacts.environment["schema_version"], IDENTITY.ENVIRONMENT_SCHEMA
        )
        self.assertEqual(
            artifacts.model_identity["schema_version"],
            IDENTITY.MODEL_IDENTITY_SCHEMA,
        )
        self.assertEqual(
            len(artifacts.environment["sources"]), len(IDENTITY.SOURCE_ROLE_PATHS)
        )
        self.assertTrue(
            artifacts.model_identity["promotion_validation"]["full_payloads"]
        )
        self.assertEqual(
            artifacts.model_identity["promotion_validation"]["canonical_source_sha256"],
            next(
                item["sha256"]
                for item in artifacts.environment["sources"]
                if item["role"] == "product_promotion_canonical"
            ),
        )
        self.assertTrue(
            artifacts.model_identity["product"]["artifact"]["payloads_hashed"]
        )
        self.assertEqual(
            artifacts.environment["host"]["tools"]["python_version_line"],
            "Python 3.12.3",
        )
        self.assertTrue(
            artifacts.environment["host"]["tools"]["rustc_version_line"].startswith(
                "rustc 1.96.0 "
            )
        )
        self.assertTrue(
            artifacts.environment["host"]["tools"]["cargo_version_line"].startswith(
                "cargo 1.96.0 "
            )
        )
        template = b"fake-qwen3-chat-template-v1"
        self.assertEqual(
            artifacts.model_identity["tokenizer"]["chat_template"],
            {"utf8_bytes": len(template), "sha256": sha256(template)},
        )
        self.assertEqual(
            artifacts.model_identity["oracle"]["vllm_identity"]["backend"],
            "vLLM",
        )
        self.assertEqual(
            artifacts.environment["source_sets"]["oracle"],
            IDENTITY._source_aggregate(
                {item["role"]: item for item in artifacts.environment["sources"]},
                IDENTITY.SOURCE_GROUPS["oracle"],
            ),
        )
        self.assertEqual(
            artifacts.environment_bytes,
            IDENTITY.serialize_environment_document(artifacts.environment),
        )
        self.assertEqual(
            artifacts.model_identity_bytes,
            IDENTITY.serialize_model_identity_document(artifacts.model_identity),
        )

    def test_build_does_not_require_large_artifact_or_package_payload_files(
        self,
    ) -> None:
        with IdentityFixture() as fixture:
            self.assertEqual(
                list((fixture.product / "artifact").iterdir()),
                [fixture.product / "artifact" / "sq_manifest.json"],
            )
            self.assertEqual(
                list((fixture.product / "package").iterdir()),
                [fixture.product / "package" / "manifest.json"],
            )
            artifacts = IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)
        self.assertEqual(
            artifacts.model_identity["product"]["package"]["payload_bytes"], 24
        )

    def test_metadata_only_or_unhashed_promotion_receipts_are_rejected(self) -> None:
        mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
            lambda value: value.__setitem__("full_payloads", False),
            lambda value: value["artifact"].__setitem__("payloads_hashed", False),
            lambda value: value["package"].__setitem__("payloads_hashed", False),
            lambda value: value.__setitem__("verified", False),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), IdentityFixture() as fixture:
                result = copy.deepcopy(fixture.validation_result)
                mutation(result)
                inputs = dataclasses.replace(
                    fixture.inputs, promotion_validation_result=result
                )
                with self.assertRaises(IDENTITY.IdentityError):
                    IDENTITY.build_identity_artifacts(inputs, fixture.live)

    def test_promotion_manifest_and_receipt_mutations_are_rejected(self) -> None:
        with IdentityFixture() as fixture:
            artifact_path = fixture.product / "artifact" / "sq_manifest.json"
            value = json.loads(artifact_path.read_bytes())
            value["integrity"]["content_sha256"] = "0" * 64
            artifact_path.write_bytes(canonical(value))
            with self.assertRaises(IDENTITY.IdentityError):
                IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)
        with IdentityFixture() as fixture:
            result = copy.deepcopy(fixture.validation_result)
            result["package"]["payload_bytes"] = 25
            inputs = dataclasses.replace(
                fixture.inputs, promotion_validation_result=result
            )
            with self.assertRaises(IDENTITY.IdentityError):
                IDENTITY.build_identity_artifacts(inputs, fixture.live)

    def test_source_binding_mode_links_and_symlinks_are_fail_closed(self) -> None:
        for defect in ("sha", "hardlink", "symlink"):
            with self.subTest(defect=defect), IdentityFixture() as fixture:
                role = "browser_soak"
                path = fixture.repo / IDENTITY.SOURCE_ROLE_PATHS[role]
                if defect == "sha":
                    specs = tuple(
                        dataclasses.replace(spec, expected_sha256="0" * 64)
                        if spec.role == role
                        else spec
                        for spec in fixture.inputs.source_specs
                    )
                    inputs = dataclasses.replace(fixture.inputs, source_specs=specs)
                else:
                    original = path.read_bytes()
                    path.unlink()
                    outside = fixture.root / "outside-source"
                    outside.write_bytes(original)
                    if defect == "hardlink":
                        os.link(outside, path)
                    else:
                        path.symlink_to(outside)
                    inputs = fixture.inputs
                with self.assertRaises(IDENTITY.IdentityError):
                    IDENTITY.build_identity_artifacts(inputs, fixture.live)

    def test_ttft_fixture_source_mutation_is_rejected(self) -> None:
        with IdentityFixture() as fixture:
            role = "fixture_ttft_p0032"
            path = fixture.repo / IDENTITY.SOURCE_ROLE_PATHS[role]
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                IDENTITY.IdentityError, "TTFT fixture source differs"
            ):
                IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)

    def test_pinned_file_detects_same_byte_entry_replacement_at_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source"
            raw = b"stable source\n"
            path.write_bytes(raw)
            pinned = IDENTITY._PinnedFile(
                path,
                maximum=1024,
                forbidden_values=(),
                retain=True,
            )
            try:
                replacement = root / "replacement"
                replacement.write_bytes(raw)
                os.replace(replacement, path)
                with self.assertRaises(IDENTITY.IdentityError):
                    pinned.seal()
            finally:
                pinned.close()

    def test_secret_split_across_hash_chunks_is_rejected_and_not_serialized(
        self,
    ) -> None:
        secret = b"campaign-secret-value"
        with IdentityFixture() as fixture:
            source = fixture.repo / IDENTITY.SOURCE_ROLE_PATHS["browser_soak"]
            source.write_bytes(b"A" * (IDENTITY.COPY_CHUNK_BYTES - 2) + secret + b"\n")
            inputs = dataclasses.replace(fixture.inputs, forbidden_values=(secret,))
            with self.assertRaises(IDENTITY.IdentityError):
                IDENTITY.build_identity_artifacts(inputs, fixture.live)
        with IdentityFixture() as fixture:
            inputs = dataclasses.replace(fixture.inputs, forbidden_values=(secret,))
            artifacts = IDENTITY.build_identity_artifacts(inputs, fixture.live)
            self.assertNotIn(secret, artifacts.environment_bytes)
            self.assertNotIn(secret, artifacts.model_identity_bytes)

    def test_live_capture_rejects_service_process_image_and_gpu_drift(self) -> None:
        for defect in ("service", "starttime", "image", "gpu"):
            with self.subTest(defect=defect), IdentityFixture() as fixture:
                probe = FakeProbe(fixture)
                if defect == "service":
                    probe.service_drift = True
                elif defect == "starttime":
                    probe.starttime_drift = True
                elif defect == "image":
                    probe.bad_image = True
                else:
                    probe.bad_gpu = True
                with self.assertRaises(IDENTITY.IdentityError):
                    IDENTITY.capture_live_identity(probe, fixture.expectation)

    def test_live_worker_binary_and_effective_config_mismatch_are_rejected(
        self,
    ) -> None:
        with IdentityFixture() as fixture:
            changed_live = dataclasses.replace(
                fixture.live,
                worker=dataclasses.replace(
                    fixture.live.worker, executable_sha256="0" * 64
                ),
            )
            with self.assertRaises(IDENTITY.IdentityError):
                IDENTITY.build_identity_artifacts(fixture.inputs, changed_live)
        with IdentityFixture() as fixture:
            raw = fixture.effective_environment.read_bytes().replace(
                b"ULLM_BIND_PORT=8000", b"ULLM_BIND_PORT=8001"
            )
            fixture.effective_environment.write_bytes(raw)
            with self.assertRaises(IDENTITY.IdentityError):
                IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)

    def test_public_validators_reject_type_order_aggregate_and_passed_drift(
        self,
    ) -> None:
        with IdentityFixture() as fixture:
            artifacts = IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)
        environment_mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
            lambda value: value["service"].__setitem__("uid", False),
            lambda value: value["sources"].reverse(),
            lambda value: value["source_sets"].__setitem__("worker", "0" * 64),
            lambda value: value.__setitem__("passed", True),
        )
        for mutation in environment_mutations:
            with self.subTest(environment_mutation=mutation):
                value = copy.deepcopy(artifacts.environment)
                mutation(value)
                with self.assertRaises(IDENTITY.IdentityError):
                    IDENTITY.validate_environment_document(value)
        model_mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
            lambda value: value["promotion_validation"].__setitem__(
                "full_payloads", False
            ),
            lambda value: value["tokenizer"].__setitem__("aggregate_sha256", "0" * 64),
            lambda value: value["oracle"]["vllm_identity"].__setitem__(
                "max_num_seqs", 2
            ),
            lambda value: value["worker"].__setitem__(
                "package_manifest_sha256", "0" * 64
            ),
            lambda value: value.__setitem__("passed", True),
        )
        for mutation in model_mutations:
            with self.subTest(model_mutation=mutation):
                value = copy.deepcopy(artifacts.model_identity)
                mutation(value)
                with self.assertRaises(IDENTITY.IdentityError):
                    IDENTITY.validate_model_identity_document(value)

    def test_writer_creates_only_fresh_0600_artifacts_and_refuses_replacement(
        self,
    ) -> None:
        with IdentityFixture() as fixture:
            artifacts = IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)
            output = fixture.root / "output"
            output.mkdir()
            result = IDENTITY.write_identity_artifacts(
                output, artifacts, uid=os.getuid(), gid=os.getgid()
            )
            self.assertEqual(set(result), {"environment.json", "model-identity.json"})
            for name in result:
                info = (output / name).stat()
                self.assertEqual(stat.S_IMODE(info.st_mode), 0o600)
                self.assertEqual(
                    result[name]["sha256"], sha256((output / name).read_bytes())
                )
            with self.assertRaises(IDENTITY.IdentityError):
                IDENTITY.write_identity_artifacts(output, artifacts)

    def test_writer_preflights_both_names_before_creating_either_file(self) -> None:
        with IdentityFixture() as fixture:
            artifacts = IDENTITY.build_identity_artifacts(fixture.inputs, fixture.live)
            output = fixture.root / "output"
            output.mkdir()
            (output / "model-identity.json").write_bytes(b"existing\n")
            with self.assertRaises(IDENTITY.IdentityError):
                IDENTITY.write_identity_artifacts(output, artifacts)
            self.assertFalse((output / "environment.json").exists())
            self.assertEqual(
                (output / "model-identity.json").read_bytes(), b"existing\n"
            )


if __name__ == "__main__":
    unittest.main()
