from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROMOTION_PATH = ROOT / "tools/sq8_serving_promotion.py"
GENERATOR_PATH = ROOT / "tools/generate-served-model.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROMOTION = load_module("test_sq8_serving_promotion_module", PROMOTION_PATH)
GENERATOR = load_module("test_sq8_serving_promotion_generator", GENERATOR_PATH)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    path.chmod(mode)


class Fixture:
    def __init__(self, root: Path, *, relocate_build_release: bool = False) -> None:
        self.root = root.resolve()
        self.source = self.root / "source"
        self.product = self.root / "product"
        self.tokenizer = self.root / "tokenizer"
        self.build_release = self.root / "build-release"
        self.release = self.root / "promotion-work"
        self.tools = self.root / "tool-bin"
        self.build_release.mkdir()
        self.release.mkdir()
        self.tools.mkdir()
        self._write_source()
        self.commit = self._git("rev-parse", "HEAD")
        self.tree = self._git("rev-parse", "HEAD^{tree}")
        self.source_epoch = self._git(
            "show",
            "-s",
            "--format=%ct",
            self.commit,
        )
        subprocess.run(
            ["git", "-C", self.source, "checkout", "--detach", "-q", self.commit],
            check=True,
        )
        self._write_runtime_inputs()
        self.build_receipt = self._write_build_receipt()
        self.build_release_origin = self.build_release
        self.build_receipt_before_relocation = self.build_receipt.read_bytes()
        self.build_seal_before_relocation = (
            self.build_release / "SEALED.json"
        ).read_bytes()
        if relocate_build_release:
            self._relocate_build_release()
        self.profile = self._write_profile()
        self.ephemeral = self._write_ephemeral_manifest()
        self.cpu_cases = self._write_cpu_cases()
        self.product_validation = self._product_validation()
        self.evidence = self.product / "sq8-serving-evidence-v1.json"
        self.receipt = self.product / "sq8-serving-promotion-v1.json"

    def _relocate_build_release(self) -> None:
        source = self.build_release
        destination = self.root / "staged-build-release"
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        source.chmod(0o755)
        shutil.rmtree(source)
        self.build_release = destination
        self.worker = destination / PROMOTION.BUILD_WORKER_RELATIVE_PATH
        self.build_receipt = destination / "build-receipt.json"

    def _git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", self.source, *arguments],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def _write_source(self) -> None:
        self.source.mkdir()
        subprocess.run(["git", "-C", self.source, "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", self.source, "config", "user.name", "SQ8 Fixture"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", self.source, "config", "user.email", "sq8@example.invalid"],
            check=True,
        )
        paths = set(PROMOTION.BUILD_INPUTS_V2) | set(
            PROMOTION.EVIDENCE_SOURCE_PATHS
        )
        for relative in paths:
            destination = self.source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            repository_source = ROOT / relative
            if repository_source.is_file():
                shutil.copyfile(repository_source, destination)
            else:
                destination.write_text(f"fixture:{relative}\n", encoding="ascii")
        subprocess.run(["git", "-C", self.source, "add", "."], check=True)
        subprocess.run(
            ["git", "-C", self.source, "commit", "-q", "-m", "fixture"], check=True
        )

    def _write_runtime_inputs(self) -> None:
        self.worker = self.build_release / "ullm-sq8-worker"
        self.worker.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.worker.chmod(0o555)
        self.cargo = self.tools / "cargo"
        self.cargo.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.cargo.chmod(0o555)
        self.python = self.tools / "python"
        self.python.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.python.chmod(0o555)

        self.tokenizer.mkdir()
        write_json(self.tokenizer / "tokenizer.json", {"model": "fixture"})
        write_json(
            self.tokenizer / "tokenizer_config.json",
            {
                "chat_template": "{{ messages }}",
                "tokenizer_class": "Qwen2Tokenizer",
            },
        )

        (self.product / "artifact").mkdir(parents=True)
        (self.product / "package").mkdir()
        write_json(
            self.product / "promotion.json",
            {
                "schema_version": "ullm.sq8_product_promotion.v1",
                "fixture": True,
            },
            mode=0o444,
        )
        write_json(
            self.product / "artifact/sq_manifest.json",
            {"schema_version": "sq-fp8-artifact-v0.2", "fixture": True},
            mode=0o444,
        )
        write_json(
            self.product / "package/manifest.json",
            {"schema_version": "ullm-prototype-manifest-v0.1", "fixture": True},
            mode=0o444,
        )

    def _write_build_receipt(self) -> Path:
        inputs: list[dict[str, Any]] = []
        provenance_inputs: dict[str, dict[str, Any]] = {}
        for relative in sorted(
            PROMOTION.BUILD_INPUTS_V2, key=lambda value: value.encode("utf-8")
        ):
            digest = sha(self.source / relative)
            inputs.append({"path": relative, "sha256": digest})
            provenance_inputs[relative] = {
                "bytes": (self.source / relative).stat().st_size,
                "sha256": digest,
            }
        recorded_source = self.root / "retired-source-at-build"
        build_environment = {
            "CARGO_BUILD_JOBS": "1",
            "CARGO_INCREMENTAL": "0",
            "CARGO_TARGET_DIR": os.fspath(self.root / "target"),
            "CUDA_VISIBLE_DEVICES": "-1",
            "GPU_ARCH": "gfx1201",
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCM_PATH": "/opt/rocm",
            "ROCR_VISIBLE_DEVICES": "-1",
            "RUSTC_WRAPPER": None,
            "SOURCE_DATE_EPOCH": self.source_epoch,
            "ULLM_HIP_VISIBLE_DEVICES": "-1",
        }
        build_argv = [
            "/usr/bin/cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "ullm-engine",
            "--bin",
            "ullm-sq8-worker",
            "--features",
            "rocm-ck-gfx1201",
        ]
        worker = {
            "relative_path": PROMOTION.BUILD_WORKER_RELATIVE_PATH,
            "bytes": self.worker.stat().st_size,
            "sha256": sha(self.worker),
            "mode": "0555",
            "nlink": 1,
        }
        receipt = {
            "schema_version": PROMOTION.BUILD_RECEIPT_SCHEMA,
            "source": {
                "repository_root": os.fspath(recorded_source),
                "commit": self.commit,
                "tree": self.tree,
                "detached": True,
                "worktree_clean": True,
                "status_sha256": hashlib.sha256(b"").hexdigest(),
            },
            "build": {
                "argv": build_argv,
                "environment": build_environment,
                "result": "success",
            },
            "inputs": inputs,
            "worker": worker,
        }
        path = self.build_release / "build-receipt.json"
        write_json(path, receipt, mode=0o444)
        provenance = {
            "schema_version": PROMOTION.BUILD_PROVENANCE_SCHEMA,
            "source": {
                "repository_root": os.fspath(recorded_source),
                "commit": self.commit,
                "tree": self.tree,
                "detached": True,
                "tracked_clean": True,
                "untracked_clean": True,
                "inputs": provenance_inputs,
            },
            "build": {
                "argv": build_argv,
                "working_directory": os.fspath(recorded_source),
                "target_directory": build_environment["CARGO_TARGET_DIR"],
                "environment": build_environment,
                "ambient_environment_hermetic": False,
                "ambient_compile_overrides_rejected": list(
                    PROMOTION.BUILD_REJECTED_ENVIRONMENT_V2
                ),
                "started_unix_ns": 1,
                "finished_unix_ns": 2,
                "toolchain": {
                    name: {
                        "path": f"/fixture/toolchain/{name}",
                        "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
                        "version": f"{name} fixture",
                    }
                    for name in ("cargo", "rustc", "cxx", "hipcc")
                },
                "result": "success",
            },
            "worker": {
                **worker,
                "protocol": "ullm.worker.v2",
                "format_id": "SQ8_0",
                "model_id": "ullm-qwen3-14b-sq8",
            },
        }
        write_json(
            self.build_release / "build-provenance.json",
            provenance,
            mode=0o444,
        )
        readme = self.build_release / "README.md"
        readme.write_text("# fixture SQ8 worker release\n", encoding="ascii")
        readme.chmod(0o444)
        sums_raw = "".join(
            f"{sha(self.build_release / name)}  {name}\n"
            for name in PROMOTION.BUILD_SUMMED_MEMBERS
        ).encode("ascii")
        sums = self.build_release / "SHA256SUMS"
        sums.write_bytes(sums_raw)
        sums.chmod(0o444)
        write_json(
            self.build_release / "SEALED.json",
            {
                "schema_version": PROMOTION.BUILD_RELEASE_SEAL_SCHEMA,
                "source_commit": self.commit,
                "source_tree": self.tree,
                "worker_sha256": worker["sha256"],
                "build_receipt_sha256": sha(path),
                "build_provenance_sha256": sha(
                    self.build_release / "build-provenance.json"
                ),
                "sha256sums_sha256": hashlib.sha256(sums_raw).hexdigest(),
                "complete": True,
            },
            mode=0o444,
        )
        self.build_release.chmod(0o555)
        return path

    def reasoning(self) -> dict[str, Any]:
        return {
            "enabled_by_default": False,
            "dialect_id": "qwen3-thinking-v1",
            "start_token_ids": [151667],
            "end_token_ids": [151668],
            "forced_end_token_ids": [151668],
            "initial_phase": "reasoning",
            "eos_policy": "close",
            "effort_budgets": {"low": 32, "medium": 128, "high": 256},
            "max_budget_tokens": 256,
            "reserved_answer_tokens": 1,
            "history_reasoning_policy": "omit",
        }

    def _profile_document(self) -> dict[str, Any]:
        return {
            "schema_version": "ullm.served_model.profile.v1",
            "public": {
                "id": "ullm-qwen3-14b-sq8",
                "name": "SQ8 fixture",
                "description": "SQ8 promotion fixture.",
                "upstream_id": "Qwen/Qwen3-14B-FP8",
                "revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
                "context_length": 4096,
            },
            "generation": {
                "max_completion_tokens": 512,
                "vocab_size": 151936,
                "eos_token_ids": [151645, 151643],
                "sampling": {"top_k": 20, "temperature": True, "top_p": True},
            },
            "format": {
                "format_id": "SQ8_0",
                "implementation_id": "qwen3_sq8_rdna4_v1",
            },
            "tokenizer": {
                "root": os.fspath(self.tokenizer),
                "transformers_version": "5.12.1",
                "class": "Qwen2Tokenizer",
                "files": ["tokenizer.json", "tokenizer_config.json"],
                "template_options": {
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            },
            "worker": {
                "protocol": "ullm.worker.v2",
                "binary": os.fspath(self.worker),
                "arguments": ["--served-model-manifest", "{manifest}"],
                "required_environment": [],
                "identity": {
                    "device": "gfx1201",
                    "execution_profile": "rdna4_w8a8_block_ck",
                },
            },
            "reasoning": self.reasoning(),
            "product": {
                "root": os.fspath(self.product),
                "artifact": {
                    "manifest_path": "artifact/sq_manifest.json",
                    "content_sha256_from_receipt": [
                        "product",
                        "artifact_content_sha256",
                    ],
                },
                "package": {"manifest_path": "package/manifest.json"},
            },
            "promotion": {
                "receipt": os.fspath(self.product / "sq8-serving-promotion-v1.json"),
                "source_commit_from_receipt": ["source_commit"],
                "required_schema_version": "ullm.sq8_serving_promotion.v1",
                "evidence_from_receipt": ["evidence", "path"],
                "evidence_sha256_from_receipt": ["evidence", "sha256"],
            },
        }

    def _write_profile(self) -> Path:
        path = self.release / "profile.json"
        write_json(path, self._profile_document())
        return path

    def _write_ephemeral_manifest(self) -> Path:
        pre_receipt = self.release / "pre-receipt.json"
        write_json(
            pre_receipt,
            {
                "source_commit": self.commit,
                "product": {"artifact_content_sha256": "a" * 64},
            },
            mode=0o444,
        )
        temporary_profile = self.release / "pre-profile.json"
        profile = self._profile_document()
        profile["promotion"] = {
            "receipt": os.fspath(pre_receipt),
            "source_commit_from_receipt": ["source_commit"],
        }
        write_json(temporary_profile, profile)
        path = self.release / "ephemeral-served-model.json"
        GENERATOR.generate(
            temporary_profile,
            path,
            source_root=self.source,
        )
        path.chmod(0o444)
        return path

    def _write_cpu_cases(self) -> Path:
        environment = {
            "CARGO_INCREMENTAL": "0",
            "CARGO_NET_OFFLINE": "true",
            "CARGO_TARGET_DIR": os.fspath(self.root / "target"),
            "CARGO_TERM_COLOR": "never",
            "CUDA_VISIBLE_DEVICES": "-1",
            "GPU_DEVICE_ORDINAL": "-1",
            "HSA_VISIBLE_DEVICES": "-1",
            "HIP_VISIBLE_DEVICES": "-1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.fspath(self.source / "services/openai-gateway/src"),
            "PY_COLORS": "0",
            "ROCR_VISIBLE_DEVICES": "-1",
            "ULLM_HIP_VISIBLE_DEVICES": "-1",
        }
        tools = {
            "cargo": {
                "invocation_path": os.fspath(self.cargo),
                "resolved_path": os.fspath(self.cargo),
                "bytes": self.cargo.stat().st_size,
                "sha256": sha(self.cargo),
            },
            "python": {
                "invocation_path": os.fspath(self.python),
                "resolved_path": os.fspath(self.python),
                "bytes": self.python.stat().st_size,
                "sha256": sha(self.python),
            },
        }
        test_runs = []
        for framework, run_id, selector in PROMOTION._cpu_test_specs():
            stdout = (
                f"test {selector} ... ok\n".encode("utf-8")
                if framework == "cargo-test"
                else b"1 passed in 0.01s\n"
            )
            stderr = b""
            test_runs.append(
                {
                    "id": run_id,
                    "framework": framework,
                    "selector": selector,
                    "argv": PROMOTION._cpu_test_argv(
                        framework=framework,
                        selector=selector,
                        cargo_path=os.fspath(self.cargo),
                        python_path=os.fspath(self.python),
                    ),
                    "exit_code": 0,
                    "stdout": {
                        "bytes": len(stdout),
                        "sha256": hashlib.sha256(stdout).hexdigest(),
                        "base64": base64.b64encode(stdout).decode("ascii"),
                    },
                    "stderr": {
                        "bytes": 0,
                        "sha256": hashlib.sha256(stderr).hexdigest(),
                        "base64": "",
                    },
                    "result": "pass",
                }
            )
        case_run_map = PROMOTION._cpu_case_run_map()
        cases = [
            {
                "id": case_id,
                "result": "pass",
                "details": {"test_run_ids": case_run_map[case_id]},
            }
            for case_id in PROMOTION.CPU_CASE_IDS
        ]
        document = {
            "schema_version": PROMOTION.CPU_CASES_SCHEMA,
            "source_root": os.fspath(self.source),
            "source_commit": self.commit,
            "source_tree": self.tree,
            "served_model_manifest_sha256": sha(self.ephemeral),
            "worker_binary_sha256": sha(self.worker),
            "identity": {
                "format_id": "SQ8_0",
                "worker_protocol": "ullm.worker.v2",
                "reasoning_dialect": "qwen3-thinking-v1",
            },
            "tools": tools,
            "environment": environment,
            "test_runs": test_runs,
            "cases": cases,
            "summary": {
                "required_case_ids": list(PROMOTION.CPU_CASE_IDS),
                "test_run_count": len(test_runs),
                "pass_count": len(PROMOTION.CPU_CASE_IDS),
                "fail_count": 0,
                "all_pass": True,
            },
        }
        path = self.release / "cpu-cases.json"
        write_json(path, document, mode=0o444)
        return path

    def _product_validation(self) -> dict[str, Any]:
        return {
            "schema_version": "ullm.sq8_product_promotion.v1",
            "product_root": os.fspath(self.product),
            "created_at": "2026-07-10T12:16:25+09:00",
            "model_revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
            "artifact": {
                "manifest_sha256": sha(self.product / "artifact/sq_manifest.json"),
                "content_sha256": "a" * 64,
                "selected_pair_count": 280,
                "payloads_hashed": True,
            },
            "package": {
                "manifest_sha256": sha(self.product / "package/manifest.json"),
                "payload_count": 163,
                "payload_bytes": 1024,
                "payloads_hashed": True,
            },
            "read_only": True,
            "full_payloads": True,
            "verified": True,
        }

    def publish_evidence(self) -> dict[str, Any]:
        document = PROMOTION.build_evidence(
            build_receipt_path=self.build_receipt,
            source_root=self.source,
            profile_path=self.profile,
            ephemeral_manifest_path=self.ephemeral,
            cpu_cases_path=self.cpu_cases,
            product_validation=self.product_validation,
        )
        PROMOTION.publish_immutable_json(self.evidence, document)
        return document

    def publish_receipt(self) -> dict[str, Any]:
        return PROMOTION.write_receipt(
            profile_path=self.profile,
            evidence_path=self.evidence,
            output_path=self.receipt,
            source_root=self.source,
        )


def reseal_build_release(fixture: Fixture) -> None:
    release = fixture.build_release
    release.chmod(0o755)
    sums_raw = "".join(
        f"{sha(release / name)}  {name}\n"
        for name in PROMOTION.BUILD_SUMMED_MEMBERS
    ).encode("ascii")
    sums = release / "SHA256SUMS"
    sums.chmod(0o644)
    sums.write_bytes(sums_raw)
    sums.chmod(0o444)
    receipt = json.loads(fixture.build_receipt.read_text(encoding="ascii"))
    seal = release / "SEALED.json"
    seal.chmod(0o644)
    write_json(
        seal,
        {
            "schema_version": PROMOTION.BUILD_RELEASE_SEAL_SCHEMA,
            "source_commit": receipt["source"]["commit"],
            "source_tree": receipt["source"]["tree"],
            "worker_sha256": receipt["worker"]["sha256"],
            "build_receipt_sha256": sha(fixture.build_receipt),
            "build_provenance_sha256": sha(
                release / "build-provenance.json"
            ),
            "sha256sums_sha256": hashlib.sha256(sums_raw).hexdigest(),
            "complete": True,
        },
        mode=0o444,
    )
    release.chmod(0o555)


def test_sq8_evidence_sources_bind_the_worker_release_builder() -> None:
    assert "tools/build-sq8-worker-release.py" in PROMOTION.EVIDENCE_SOURCE_PATHS


def test_all_sq8_promotion_clis_pass_the_explicit_source_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = Path("/current/sealed-source")
    observed: list[tuple[str, dict[str, Any]]] = []

    class FakePromotion:
        REASONING_CONTRACT: dict[str, Any] = {}

        @staticmethod
        def prepare_ephemeral_manifest(**kwargs: Any) -> dict[str, str]:
            observed.append(("prepare", kwargs))
            return {
                "receipt_path": "/output/pre-receipt.json",
                "receipt_sha256": "1" * 64,
                "manifest_path": "/output/manifest.json",
                "manifest_sha256": "2" * 64,
            }

        @staticmethod
        def build_cpu_cases_report(**kwargs: Any) -> dict[str, Any]:
            observed.append(("cpu-build", kwargs))
            return {
                "schema_version": PROMOTION.CPU_CASES_SCHEMA,
                "source_commit": "1" * 40,
                "source_tree": "2" * 40,
                "served_model_manifest_sha256": "3" * 64,
                "worker_binary_sha256": "4" * 64,
                "summary": {"test_run_count": 1, "pass_count": 1},
            }

        @staticmethod
        def publish_immutable_json(_path: Path, _document: dict[str, Any]) -> str:
            return "5" * 64

        @staticmethod
        def validate_build_receipt(
            _path: Path, **kwargs: Any
        ) -> dict[str, Any]:
            observed.append(("build-validate", kwargs))
            return {"schema_version": PROMOTION.BUILD_RECEIPT_SCHEMA}

        @staticmethod
        def resolve_build_source_root(
            _build: dict[str, Any], selected: Path | None
        ) -> Path:
            assert selected is not None
            return selected

        @staticmethod
        def validate_cpu_cases(_path: Path, **kwargs: Any) -> dict[str, Any]:
            observed.append(("cpu-validate", kwargs))
            return {}

        @staticmethod
        def build_evidence(**kwargs: Any) -> dict[str, Any]:
            observed.append(("evidence-build", kwargs))
            return {"schema_version": PROMOTION.EVIDENCE_SCHEMA}

        @staticmethod
        def validate_evidence(_path: Path, **kwargs: Any) -> dict[str, Any]:
            observed.append(("evidence-validate", kwargs))
            return {
                "schema_version": PROMOTION.EVIDENCE_SCHEMA,
                "source": {"commit": "1" * 40},
                "worker": {"sha256": "2" * 64},
                "cpu_cases": {"case_count": 1},
            }

        @staticmethod
        def write_receipt(**kwargs: Any) -> dict[str, Any]:
            observed.append(("receipt-write", kwargs))
            return {"schema_version": PROMOTION.RECEIPT_SCHEMA}

    invocations = (
        (
            "prepare-sq8-serving-promotion-ephemeral.py",
            [
                "--build-receipt",
                "/input/build-receipt.json",
                "--source-root",
                os.fspath(source_root),
                "--profile",
                "/input/profile.json",
                "--receipt-output",
                "/output/pre-receipt.json",
                "--manifest-output",
                "/output/manifest.json",
            ],
        ),
        (
            "run-sq8-serving-promotion-cpu-cases.py",
            [
                "--build-receipt",
                "/input/build-receipt.json",
                "--source-root",
                os.fspath(source_root),
                "--ephemeral-manifest",
                "/input/manifest.json",
                "--cargo",
                "/usr/bin/cargo",
                "--python",
                "/usr/bin/python3",
                "--target-dir",
                "/output/target",
                "--output",
                "/output/cpu.json",
            ],
        ),
        (
            "run-sq8-serving-promotion-evidence.py",
            [
                "--build-receipt",
                "/input/build-receipt.json",
                "--source-root",
                os.fspath(source_root),
                "--profile",
                "/input/profile.json",
                "--ephemeral-manifest",
                "/input/manifest.json",
                "--cpu-cases",
                "/input/cpu.json",
                "--output",
                "/output/evidence.json",
            ],
        ),
        (
            "validate-sq8-serving-promotion-evidence.py",
            [
                "/input/evidence.json",
                "--source-root",
                os.fspath(source_root),
                "--profile",
                "/input/profile.json",
            ],
        ),
        (
            "write-sq8-serving-promotion-receipt.py",
            [
                "--source-root",
                os.fspath(source_root),
                "--profile",
                "/input/profile.json",
                "--evidence",
                "/input/evidence.json",
                "--output",
                "/output/receipt.json",
            ],
        ),
    )
    monkeypatch.setitem(sys.modules, "sq8_serving_promotion", PROMOTION)
    for index, (filename, arguments) in enumerate(invocations):
        module = load_module(
            f"test_sq8_promotion_cli_{index}",
            ROOT / "tools" / filename,
        )
        monkeypatch.setattr(module, "promotion", FakePromotion)
        assert module.main(arguments) == 0

    source_bound_calls = {
        name
        for name, kwargs in observed
        if kwargs.get("source_root") == source_root
    }
    assert source_bound_calls == {
        "prepare",
        "cpu-build",
        "build-validate",
        "cpu-validate",
        "evidence-build",
        "evidence-validate",
        "receipt-write",
    }


def test_sq8_evidence_receipt_and_generator_bind_end_to_end(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)

    evidence = fixture.publish_evidence()
    receipt = fixture.publish_receipt()
    candidate = fixture.release / "served-model-final.json"
    digest = GENERATOR.generate(
        fixture.profile,
        candidate,
        source_root=fixture.source,
    )

    assert evidence["schema_version"] == PROMOTION.EVIDENCE_SCHEMA
    assert evidence["source"]["commit"] == fixture.commit
    assert evidence["worker"]["sha256"] == sha(fixture.worker)
    assert receipt["schema_version"] == PROMOTION.RECEIPT_SCHEMA
    assert receipt["product"]["artifact_content_sha256"] == "a" * 64
    assert digest == sha(candidate)
    assert json.loads(candidate.read_text())["schema_version"] == "ullm.served_model.v2"
    for path in (fixture.evidence, fixture.receipt):
        metadata = path.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1


def test_relocated_build_release_drives_every_later_absolute_binding(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path, relocate_build_release=True)
    staged_worker = fixture.build_release / PROMOTION.BUILD_WORKER_RELATIVE_PATH

    assert not fixture.build_release_origin.exists()
    assert fixture.build_receipt.read_bytes() == fixture.build_receipt_before_relocation
    assert (
        fixture.build_release / "SEALED.json"
    ).read_bytes() == fixture.build_seal_before_relocation
    assert sha(fixture.build_receipt) == hashlib.sha256(
        fixture.build_receipt_before_relocation
    ).hexdigest()
    assert sha(fixture.build_release / "SEALED.json") == hashlib.sha256(
        fixture.build_seal_before_relocation
    ).hexdigest()

    release = PROMOTION.validate_build_release(
        fixture.build_release,
        source_root=fixture.source,
    )
    assert Path(release["worker_path"]) == staged_worker
    assert json.loads(fixture.profile.read_text())["worker"]["binary"] == os.fspath(
        staged_worker
    )
    assert json.loads(fixture.ephemeral.read_text())["worker"]["binary"] == os.fspath(
        staged_worker
    )
    assert json.loads(fixture.cpu_cases.read_text())[
        "worker_binary_sha256"
    ] == sha(staged_worker)

    evidence = fixture.publish_evidence()
    fixture.publish_receipt()
    candidate = fixture.release / "served-model-final.json"
    GENERATOR.generate(
        fixture.profile,
        candidate,
        source_root=fixture.source,
    )
    assert evidence["worker"]["binary"] == os.fspath(staged_worker)
    assert evidence["worker_build_receipt"]["path"] == os.fspath(
        fixture.build_receipt
    )
    assert json.loads(candidate.read_text())["worker"]["binary"] == os.fspath(
        staged_worker
    )


def test_v2_build_live_validation_requires_explicit_current_source_root(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)

    assert not Path(
        json.loads(fixture.build_receipt.read_text())["source"]["repository_root"]
    ).exists()
    with pytest.raises(PROMOTION.PromotionError, match="explicit source root"):
        PROMOTION.validate_build_receipt(fixture.build_receipt)
    with pytest.raises(PROMOTION.PromotionError, match="explicit source root"):
        PROMOTION.build_evidence(
            build_receipt_path=fixture.build_receipt,
            profile_path=fixture.profile,
            ephemeral_manifest_path=fixture.ephemeral,
            cpu_cases_path=fixture.cpu_cases,
            product_validation=fixture.product_validation,
        )


def test_v2_build_rejects_the_wrong_explicit_current_source_root(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    wrong_source = fixture.root / "wrong-source"
    shutil.copytree(fixture.source, wrong_source)
    changed = wrong_source / next(iter(sorted(PROMOTION.REQUIRED_BUILD_INPUTS)))
    changed.write_bytes(changed.read_bytes() + b"\nwrong source\n")
    subprocess.run(["git", "-C", wrong_source, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", wrong_source, "commit", "-q", "-m", "wrong source"],
        check=True,
    )

    with pytest.raises(PROMOTION.PromotionError, match="live checkout differs"):
        PROMOTION.validate_build_receipt(
            fixture.build_receipt,
            source_root=wrong_source,
        )


def test_generator_requires_explicit_current_source_root_for_sq8(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.publish_evidence()
    fixture.publish_receipt()

    with pytest.raises(GENERATOR.GenerationError, match="validation failed"):
        GENERATOR.materialize(fixture.profile)

    source_alias = fixture.root / "source-alias"
    source_alias.symlink_to(fixture.source, target_is_directory=True)
    with pytest.raises(GENERATOR.GenerationError, match="validation failed"):
        GENERATOR.materialize(
            fixture.profile,
            source_root=source_alias,
        )


def test_generator_preserves_v1_nested_build_receipt_source_fallback(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    build = json.loads(fixture.build_receipt.read_text())
    relative = build["worker"].pop("relative_path")
    build["worker"]["path"] = os.fspath(fixture.build_release / relative)
    build["schema_version"] = PROMOTION.BUILD_RECEIPT_SCHEMA_V1
    build["source"]["repository_root"] = os.fspath(fixture.source)
    fixture.build_release.chmod(0o755)
    fixture.build_receipt.chmod(0o644)
    write_json(fixture.build_receipt, build, mode=0o444)
    fixture.build_release.chmod(0o555)

    fixture.publish_evidence()
    fixture.publish_receipt()
    document = GENERATOR.materialize(fixture.profile)

    assert document["format"]["format_id"] == "SQ8_0"
    assert document["worker"]["binary"] == os.fspath(fixture.worker)


@pytest.mark.parametrize(
    "locator",
    [
        "",
        ".",
        "..",
        "../ullm-sq8-worker",
        "nested/ullm-sq8-worker",
        "/tmp/ullm-sq8-worker",
        r"nested\ullm-sq8-worker",
        "ullm-sq8-worker/",
    ],
)
def test_v2_build_receipt_rejects_every_nonexact_worker_locator(
    tmp_path: Path, locator: str
) -> None:
    fixture = Fixture(tmp_path)
    document = json.loads(fixture.build_receipt.read_text())
    document["worker"]["relative_path"] = locator
    fixture.build_release.chmod(0o755)
    fixture.build_receipt.chmod(0o644)
    write_json(fixture.build_receipt, document, mode=0o444)
    fixture.build_release.chmod(0o555)

    with pytest.raises(PROMOTION.PromotionError, match="relative locator differs"):
        PROMOTION.validate_build_receipt(
            fixture.build_receipt,
            source_root=fixture.source,
        )


def test_v2_build_receipt_rejects_a_moved_receipt_without_the_exact_filename(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    alias = fixture.root / "build-receipt-alias.json"
    shutil.copy2(fixture.build_receipt, alias)

    with pytest.raises(PROMOTION.PromotionError, match="filename differs"):
        PROMOTION.validate_build_receipt(alias, source_root=fixture.source)


@pytest.mark.parametrize(
    "audit_path",
    [
        "relative/source",
        "//host/source",
        "/retired/../source",
        "/retired//source",
        "/retired/./source",
    ],
)
def test_v2_build_receipt_rejects_noncanonical_audit_source_paths(
    tmp_path: Path, audit_path: str
) -> None:
    fixture = Fixture(tmp_path)
    document = json.loads(fixture.build_receipt.read_text())
    document["source"]["repository_root"] = audit_path
    fixture.build_release.chmod(0o755)
    fixture.build_receipt.chmod(0o644)
    write_json(fixture.build_receipt, document, mode=0o444)
    fixture.build_release.chmod(0o555)

    with pytest.raises(PROMOTION.PromotionError, match="audit path"):
        PROMOTION.validate_build_receipt(
            fixture.build_receipt,
            verify_live_source=False,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "root-mode",
        "extra-member",
        "missing-member",
        "worker-mode",
        "worker-bytes",
        "worker-symlink",
        "worker-hardlink",
        "receipt-hardlink",
        "provenance",
        "provenance-toolchain",
        "provenance-rejected-environment",
        "sha256sums",
        "seal",
    ],
)
def test_v2_complete_build_release_mutations_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    fixture = Fixture(tmp_path)
    release = fixture.build_release
    release.chmod(0o755)

    if mutation == "root-mode":
        pass
    elif mutation == "extra-member":
        (release / "unexpected").write_bytes(b"unexpected\n")
        release.chmod(0o555)
    elif mutation == "missing-member":
        (release / "README.md").unlink()
        release.chmod(0o555)
    elif mutation == "worker-mode":
        fixture.worker.chmod(0o755)
        release.chmod(0o555)
    elif mutation == "worker-bytes":
        fixture.worker.chmod(0o755)
        fixture.worker.write_bytes(fixture.worker.read_bytes() + b"changed\n")
        fixture.worker.chmod(0o555)
        release.chmod(0o555)
    elif mutation == "worker-symlink":
        external = fixture.root / "external-worker"
        shutil.copy2(fixture.worker, external)
        fixture.worker.unlink()
        fixture.worker.symlink_to(external)
        release.chmod(0o555)
    elif mutation == "worker-hardlink":
        external = fixture.root / "external-worker"
        shutil.copy2(fixture.worker, external)
        fixture.worker.unlink()
        os.link(external, fixture.worker)
        release.chmod(0o555)
    elif mutation == "receipt-hardlink":
        external = fixture.root / "external-receipt.json"
        shutil.copy2(fixture.build_receipt, external)
        fixture.build_receipt.unlink()
        os.link(external, fixture.build_receipt)
        release.chmod(0o555)
    elif mutation in {
        "provenance",
        "provenance-toolchain",
        "provenance-rejected-environment",
    }:
        provenance = release / "build-provenance.json"
        document = json.loads(provenance.read_text())
        if mutation == "provenance":
            document["schema_version"] = PROMOTION.BUILD_PROVENANCE_SCHEMA_V1
        elif mutation == "provenance-toolchain":
            document["build"]["toolchain"]["python"] = {
                "path": "/usr/bin/python3",
                "sha256": "0" * 64,
                "version": "python fixture",
            }
        else:
            document["build"]["ambient_compile_overrides_rejected"].pop()
        provenance.chmod(0o644)
        write_json(provenance, document, mode=0o444)
        release.chmod(0o555)
    elif mutation == "sha256sums":
        sums = release / "SHA256SUMS"
        sums.chmod(0o644)
        sums.write_bytes(sums.read_bytes() + b"0" * 64 + b"  unexpected\n")
        sums.chmod(0o444)
        release.chmod(0o555)
    else:
        seal = release / "SEALED.json"
        document = json.loads(seal.read_text())
        document["worker_sha256"] = "0" * 64
        seal.chmod(0o644)
        write_json(seal, document, mode=0o444)
        release.chmod(0o555)

    with pytest.raises(PROMOTION.PromotionError):
        PROMOTION.validate_build_release(
            release,
            source_root=fixture.source,
        )


def test_v2_release_rejects_resealed_source_epoch_forgery(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.build_release.chmod(0o755)
    receipt = json.loads(fixture.build_receipt.read_text(encoding="ascii"))
    provenance_path = fixture.build_release / "build-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="ascii"))
    forged_epoch = str(int(fixture.source_epoch) + 1)
    receipt["build"]["environment"]["SOURCE_DATE_EPOCH"] = forged_epoch
    provenance["build"]["environment"]["SOURCE_DATE_EPOCH"] = forged_epoch
    fixture.build_receipt.chmod(0o644)
    write_json(fixture.build_receipt, receipt, mode=0o444)
    provenance_path.chmod(0o644)
    write_json(provenance_path, provenance, mode=0o444)
    reseal_build_release(fixture)

    with pytest.raises(
        PROMOTION.PromotionError,
        match="SOURCE_DATE_EPOCH differs",
    ):
        PROMOTION.validate_build_release(
            fixture.build_release,
            source_root=fixture.source,
        )


def test_v2_release_rejects_resealed_provenance_input_size_forgery(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.build_release.chmod(0o755)
    provenance_path = fixture.build_release / "build-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="ascii"))
    relative = PROMOTION.BUILD_INPUTS_V2[0]
    provenance["source"]["inputs"][relative]["bytes"] += 1
    provenance_path.chmod(0o644)
    write_json(provenance_path, provenance, mode=0o444)
    reseal_build_release(fixture)

    with pytest.raises(
        PROMOTION.PromotionError,
        match="provenance input identity differs",
    ):
        PROMOTION.validate_build_release(
            fixture.build_release,
            source_root=fixture.source,
        )


def test_sq8_ephemeral_preparer_publishes_no_replace_scaffold_and_v2_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path)
    monkeypatch.setattr(
        PROMOTION,
        "_ephemeral_product_metadata",
        lambda _root: fixture.product_validation,
    )
    receipt = fixture.release / "ephemeral-receipt.json"
    manifest = fixture.release / "prepared-ephemeral.json"
    result = PROMOTION.prepare_ephemeral_manifest(
        build_receipt_path=fixture.build_receipt,
        source_root=fixture.source,
        profile_path=fixture.profile,
        receipt_output_path=receipt,
        manifest_output_path=manifest,
    )

    assert result["manifest_sha256"] == sha(manifest)
    assert result["receipt_sha256"] == sha(receipt)
    assert json.loads(manifest.read_text())["schema_version"] == "ullm.served_model.v2"
    assert json.loads(receipt.read_text())["source_commit"] == fixture.commit
    for path in (receipt, manifest):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_nlink == 1

    with pytest.raises(PROMOTION.PromotionError, match="already exists"):
        PROMOTION.prepare_ephemeral_manifest(
            build_receipt_path=fixture.build_receipt,
            source_root=fixture.source,
            profile_path=fixture.profile,
            receipt_output_path=receipt,
            manifest_output_path=manifest,
        )


def test_sq8_cpu_report_producer_runs_exact_gpu_hidden_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path)
    observed: list[tuple[list[str], dict[str, str]]] = []

    def execute(
        argv: list[str],
        *,
        source_root: Path,
        environment: dict[str, str],
    ) -> tuple[int, bytes, bytes]:
        assert source_root == fixture.source
        observed.append((list(argv), dict(environment)))
        selector = argv[argv.index("--lib") + 1] if "--lib" in argv else argv[-1]
        stdout = (
            f"test {selector} ... ok\n".encode()
            if "--lib" in argv
            else b"1 passed in 0.01s\n"
        )
        return 0, stdout, b""

    monkeypatch.setattr(PROMOTION, "_run_cpu_test", execute)
    target = fixture.root / "cpu-target"
    target.mkdir()
    report = PROMOTION.build_cpu_cases_report(
        build_receipt_path=fixture.build_receipt,
        source_root=fixture.source,
        ephemeral_manifest_path=fixture.ephemeral,
        cargo_path=fixture.cargo,
        python_path=fixture.python,
        target_dir=target,
    )

    assert report["summary"]["all_pass"] is True
    assert report["summary"]["test_run_count"] == len(PROMOTION._cpu_test_specs())
    assert len(observed) == report["summary"]["test_run_count"]
    assert all(
        environment["CUDA_VISIBLE_DEVICES"] == "-1"
        and environment["GPU_DEVICE_ORDINAL"] == "-1"
        and environment["HSA_VISIBLE_DEVICES"] == "-1"
        and environment["HIP_VISIBLE_DEVICES"] == "-1"
        and environment["ROCR_VISIBLE_DEVICES"] == "-1"
        and environment["ULLM_HIP_VISIBLE_DEVICES"] == "-1"
        for _, environment in observed
    )


def test_sq8_cpu_report_producer_stops_on_failed_exact_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path)

    def execute(
        argv: list[str],
        *,
        source_root: Path,
        environment: dict[str, str],
    ) -> tuple[int, bytes, bytes]:
        del argv, source_root, environment
        return 1, b"test failed\n", b"failure\n"

    monkeypatch.setattr(PROMOTION, "_run_cpu_test", execute)
    target = fixture.root / "cpu-target"
    target.mkdir()
    with pytest.raises(PROMOTION.PromotionError, match="test failed"):
        PROMOTION.build_cpu_cases_report(
            build_receipt_path=fixture.build_receipt,
            source_root=fixture.source,
            ephemeral_manifest_path=fixture.ephemeral,
            cargo_path=fixture.cargo,
            python_path=fixture.python,
            target_dir=target,
        )


def test_sq8_receipt_publication_is_one_time_no_replace(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.publish_evidence()
    fixture.publish_receipt()
    original = fixture.receipt.read_bytes()

    with pytest.raises(PROMOTION.PromotionError, match="already exists"):
        fixture.publish_receipt()

    assert fixture.receipt.read_bytes() == original


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["worker"].__setitem__("sha256", "0" * 64),
            "worker identities",
        ),
        (
            lambda value: value["reasoning"].__setitem__(
                "dialect_id", "qwen3.5-thinking-v1"
            ),
            "reasoning binding",
        ),
        (
            lambda value: value["product"]["artifact"].__setitem__(
                "content_sha256", "0" * 64
            ),
            "product evidence identity",
        ),
        (
            lambda value: value["ephemeral_manifest"].__setitem__(
                "semantic_sha256", "0" * 64
            ),
            "semantic identity",
        ),
        (
            lambda value: value["cpu_cases"]["report"]["summary"].__setitem__(
                "all_pass", False
            ),
            "embedded report",
        ),
    ],
)
def test_sq8_evidence_mutations_fail_closed(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    fixture = Fixture(tmp_path)
    value = fixture.publish_evidence()
    fixture.evidence.chmod(0o644)
    mutate(value)
    write_json(fixture.evidence, value, mode=0o444)

    with pytest.raises(PROMOTION.PromotionError, match=message):
        PROMOTION.validate_evidence(
            fixture.evidence,
            expected_profile_path=fixture.profile,
            source_root=fixture.source,
            require_receipt_absent=True,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source"].__setitem__("commit", "0" * 40),
        lambda value: value["source"].__setitem__("tree", "0" * 40),
        lambda value: value["source"]["evidence_files"][0].__setitem__(
            "sha256", "0" * 64
        ),
        lambda value: value["worker_build_receipt"].__setitem__("sha256", "0" * 64),
        lambda value: value["profile"].__setitem__("sha256", "0" * 64),
        lambda value: value["ephemeral_manifest"].__setitem__("sha256", "0" * 64),
        lambda value: value["product"]["receipt"].__setitem__("sha256", "0" * 64),
        lambda value: value["product"]["artifact"].__setitem__(
            "manifest_sha256", "0" * 64
        ),
        lambda value: value["product"]["package"].__setitem__(
            "manifest_sha256", "0" * 64
        ),
        lambda value: value["product"].__setitem__("validation_sha256", "0" * 64),
        lambda value: value["cpu_cases"].__setitem__("sha256", "0" * 64),
    ],
)
def test_sq8_every_evidence_sha_and_source_binding_fails_closed(
    tmp_path: Path, mutate: Any
) -> None:
    fixture = Fixture(tmp_path)
    value = fixture.publish_evidence()
    fixture.evidence.chmod(0o644)
    mutate(value)
    write_json(fixture.evidence, value, mode=0o444)

    with pytest.raises(PROMOTION.PromotionError):
        PROMOTION.validate_evidence(
            fixture.evidence,
            expected_profile_path=fixture.profile,
            source_root=fixture.source,
            require_receipt_absent=True,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("source_commit", "0" * 40),
        lambda value: value["evidence"].__setitem__("path", "../evidence.json"),
        lambda value: value["evidence"].__setitem__("sha256", "0" * 64),
        lambda value: value["product"]["receipt"].__setitem__("sha256", "0" * 64),
        lambda value: value["product"].__setitem__(
            "artifact_manifest_sha256", "0" * 64
        ),
        lambda value: value["product"].__setitem__("artifact_content_sha256", "0" * 64),
        lambda value: value["product"].__setitem__("package_manifest_sha256", "0" * 64),
    ],
)
def test_sq8_every_receipt_binding_fails_closed(tmp_path: Path, mutate: Any) -> None:
    fixture = Fixture(tmp_path)
    fixture.publish_evidence()
    fixture.publish_receipt()
    value = json.loads(fixture.receipt.read_text())
    fixture.receipt.chmod(0o644)
    mutate(value)
    write_json(fixture.receipt, value, mode=0o444)

    with pytest.raises(PROMOTION.PromotionError):
        PROMOTION.validate_receipt(
            fixture.receipt,
            expected_evidence_path=fixture.evidence,
            expected_profile_path=fixture.profile,
            source_root=fixture.source,
        )


def test_sq8_cpu_case_failure_is_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    value = json.loads(fixture.cpu_cases.read_text())
    value["cases"][3]["result"] = "fail"
    fixture.cpu_cases.chmod(0o644)
    write_json(fixture.cpu_cases, value, mode=0o444)

    with pytest.raises(PROMOTION.PromotionError, match="did not pass"):
        fixture.publish_evidence()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("source_tree", "0" * 40),
        lambda value: value["tools"]["cargo"].__setitem__("sha256", "0" * 64),
        lambda value: value["environment"].__setitem__("HIP_VISIBLE_DEVICES", "0"),
        lambda value: value["test_runs"][0]["argv"].append("--unexpected"),
        lambda value: value["test_runs"][0].__setitem__("exit_code", 1),
        lambda value: value["test_runs"][0]["stdout"].__setitem__("sha256", "0" * 64),
        lambda value: value["cases"][0]["details"].__setitem__(
            "test_run_ids", ["gateway-eos-reconcile"]
        ),
    ],
)
def test_sq8_cpu_report_command_and_result_mutations_fail_closed(
    tmp_path: Path, mutate: Any
) -> None:
    fixture = Fixture(tmp_path)
    value = json.loads(fixture.cpu_cases.read_text())
    fixture.cpu_cases.chmod(0o644)
    mutate(value)
    write_json(fixture.cpu_cases, value, mode=0o444)

    with pytest.raises(PROMOTION.PromotionError):
        fixture.publish_evidence()


def test_sq8_build_receipt_requires_clean_detached_live_source(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    (fixture.source / "untracked").write_text("dirty\n", encoding="ascii")

    with pytest.raises(PROMOTION.PromotionError, match="live checkout differs"):
        fixture.publish_evidence()


def test_generator_rejects_aq4_receipt_schema_for_sq8_format(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    profile = json.loads(fixture.profile.read_text())
    profile["promotion"]["required_schema_version"] = "ullm.aq4_resident_promotion.v1"
    write_json(fixture.profile, profile)

    with pytest.raises(GENERATOR.GenerationError, match="schema/format pairing"):
        GENERATOR.materialize(fixture.profile, source_root=fixture.source)


def test_generator_rejects_sq8_receipt_schema_for_aq4_format(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    profile = json.loads(fixture.profile.read_text())
    profile["format"]["format_id"] = "AQ4_0"
    write_json(fixture.profile, profile)

    with pytest.raises(GENERATOR.GenerationError, match="schema/format pairing"):
        GENERATOR.materialize(fixture.profile, source_root=fixture.source)


def test_generator_rejects_tampered_sq8_receipt_evidence_hash(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.publish_evidence()
    fixture.publish_receipt()
    fixture.receipt.chmod(0o644)
    receipt = json.loads(fixture.receipt.read_text())
    receipt["evidence"]["sha256"] = "0" * 64
    write_json(fixture.receipt, receipt, mode=0o444)

    with pytest.raises(GENERATOR.GenerationError, match="evidence SHA-256 differs"):
        GENERATOR.materialize(fixture.profile, source_root=fixture.source)


@pytest.mark.parametrize(
    "mutation",
    ["source", "worker", "product", "profile", "reasoning", "evidence-path"],
)
def test_generator_sq8_dispatch_revalidates_every_live_identity(
    tmp_path: Path, mutation: str
) -> None:
    fixture = Fixture(tmp_path)
    fixture.publish_evidence()
    fixture.publish_receipt()

    if mutation == "source":
        source = fixture.source / "tools/generate-served-model.py"
        source.write_bytes(source.read_bytes() + b"\n")
    elif mutation == "worker":
        fixture.worker.chmod(0o755)
        fixture.worker.write_bytes(fixture.worker.read_bytes() + b"\n")
        fixture.worker.chmod(0o555)
    elif mutation == "product":
        product = fixture.product / "artifact/sq_manifest.json"
        product.chmod(0o644)
        product.write_bytes(product.read_bytes() + b"\n")
        product.chmod(0o444)
    elif mutation in {"profile", "reasoning"}:
        profile = json.loads(fixture.profile.read_text())
        if mutation == "profile":
            profile["public"]["description"] = "mutated profile"
        else:
            profile["reasoning"]["reserved_answer_tokens"] = 2
        write_json(fixture.profile, profile)
    else:
        fixture.receipt.chmod(0o644)
        receipt = json.loads(fixture.receipt.read_text())
        receipt["evidence"]["path"] = "missing-evidence.json"
        write_json(fixture.receipt, receipt, mode=0o444)

    with pytest.raises(GENERATOR.GenerationError):
        GENERATOR.materialize(fixture.profile, source_root=fixture.source)


def test_generator_rejects_nonimmutable_sq8_receipt(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.publish_evidence()
    fixture.publish_receipt()
    fixture.receipt.chmod(0o644)

    with pytest.raises(GENERATOR.GenerationError, match="validation failed"):
        GENERATOR.materialize(fixture.profile, source_root=fixture.source)
