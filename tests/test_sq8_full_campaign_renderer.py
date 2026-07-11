from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RENDERER = load_module(
    "test_sq8_full_campaign_renderer_tool",
    TOOLS / "sq8_full_campaign_renderer.py",
)
COLLECTOR = load_module(
    "test_sq8_full_campaign_renderer_collector",
    TOOLS / "collect-sq8-openwebui-release.py",
)
VALIDATOR = load_module(
    "test_sq8_full_campaign_renderer_validator",
    TOOLS / "validate-sq8-openwebui-release.py",
)
ORCHESTRATOR = load_module(
    "test_sq8_full_campaign_renderer_orchestrator",
    TOOLS / "run-sq8-full-openwebui-campaign.py",
)
VIEW_FIXTURES = load_module(
    "test_sq8_full_campaign_renderer_view_fixtures",
    ROOT / "tests" / "test_sq8_full_campaign_views.py",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class CampaignFixture:
    def __init__(self, *, run_id: str = "20260711-sq8-full-campaign") -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.stage = Path(self.temporary.name) / "stage"
        self.stage.mkdir(mode=0o700)
        self.browser = self.stage / "browser"
        self.browser.mkdir(mode=0o700)
        self._populate_files()
        self.evidence = types.SimpleNamespace(
            preflight=types.SimpleNamespace(
                header_fields={
                    "run_id": run_id,
                    "schedule": copy.deepcopy(RENDERER.SCHEDULE),
                    "thresholds": copy.deepcopy(RENDERER.THRESHOLDS),
                    "unrelated_preflight_field": "retained-in-raw-only",
                }
            ),
            api_contract=types.SimpleNamespace(derived_view=VIEW_FIXTURES.api_view()),
            combined=types.SimpleNamespace(derived_view=VIEW_FIXTURES.combined_view()),
            direct_cancel=types.SimpleNamespace(
                derived_view=VIEW_FIXTURES.direct_cancel_view()
            ),
            stop=types.SimpleNamespace(derived_view=VIEW_FIXTURES.stop_view()),
            failure=types.SimpleNamespace(derived_view=VIEW_FIXTURES.failure_view()),
            latency=types.SimpleNamespace(derived_view=VIEW_FIXTURES.latency_view()),
            resource_normal=COLLECTOR.ResourceSegmentResult(
                segment="normal",
                identity=COLLECTOR.ProcessIdentity(
                    "/system.slice/ullm-openai.service",
                    100,
                    1000,
                    101,
                    1001,
                    2,
                ),
                warmup_requests=10,
                measured_requests=100,
                negative_requests=3,
                resource_samples=505,
                gpu_metrics=2,
                sampling_cases=tuple(VIEW_FIXTURES.sampling_cases()),
            ),
        )
        self.context = types.SimpleNamespace(
            stage_path=self.stage,
            evidence=self.evidence,
        )

    def _populate_files(self) -> None:
        for relative in sorted(RENDERER.EXISTING_PATHS):
            path = self.stage / relative
            if relative == "soak-resources.raw.jsonl":
                VIEW_FIXTURES.write_resource(path)
            else:
                raw = (relative + "\n").encode("ascii")
                if relative == "raw-session-results.jsonl":
                    raw += b"x" * (RENDERER.HASH_CHUNK_BYTES + 17)
                path.write_bytes(raw)
            os.chmod(path, 0o600)
        os.chmod(self.browser, 0o700)
        os.chmod(self.stage, 0o700)

    def render(self) -> dict[str, bytes]:
        return cast(
            dict[str, bytes], RENDERER.FullCampaignRenderer().render(self.context)
        )

    def close(self) -> None:
        self.temporary.cleanup()

    def __enter__(self) -> CampaignFixture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class FullCampaignRendererTests(unittest.TestCase):
    def test_frozen_contract_matches_collector_and_independent_validator(self) -> None:
        self.assertEqual(RENDERER.MATRIX_SCHEMA, COLLECTOR.MATRIX_SCHEMA)
        self.assertEqual(RENDERER.MATRIX_SCHEMA, VALIDATOR.MATRIX_SCHEMA)
        self.assertEqual(RENDERER.MATRIX_ROLES, COLLECTOR.EXPECTED_ROLES)
        self.assertEqual(RENDERER.MATRIX_ROLES, VALIDATOR.EXPECTED_ROLES)
        self.assertEqual(RENDERER.SCHEDULE, COLLECTOR.SCHEDULE)
        self.assertEqual(RENDERER.SCHEDULE, VALIDATOR.SCHEDULE)
        self.assertEqual(RENDERER.THRESHOLDS, COLLECTOR.THRESHOLDS)
        renderer_thresholds = json.loads(
            json.dumps(RENDERER.THRESHOLDS), parse_float=Decimal
        )
        self.assertEqual(renderer_thresholds, VALIDATOR.THRESHOLDS)
        self.assertEqual(RENDERER.PREVALIDATION_PATHS, COLLECTOR.BUNDLE_FILES)
        self.assertEqual(RENDERER.PREVALIDATION_PATHS, VALIDATOR.BUNDLE_FILES)

    def test_renders_exact_artifacts_matrix_summary_and_checksums(self) -> None:
        with CampaignFixture() as fixture:
            self.assertIsInstance(
                fixture.evidence.resource_normal, COLLECTOR.ResourceSegmentResult
            )
            rendered = fixture.render()

            self.assertEqual(set(rendered), RENDERER.RENDERED_PATHS)
            self.assertTrue(
                all(raw and raw.endswith(b"\n") for raw in rendered.values())
            )

            matrix = json.loads(rendered["release-matrix.json"])
            self.assertEqual(matrix["schema_version"], RENDERER.MATRIX_SCHEMA)
            self.assertEqual(matrix["run_id"], "20260711-sq8-full-campaign")
            self.assertEqual(matrix["schedule"], RENDERER.SCHEDULE)
            self.assertEqual(matrix["thresholds"], RENDERER.THRESHOLDS)
            self.assertEqual(
                [entry["path"] for entry in matrix["files"]],
                list(RENDERER.MATRIX_INPUT_PATHS),
            )
            self.assertEqual(len(matrix["files"]), 17)
            for entry in matrix["files"]:
                relative = entry["path"]
                raw = (
                    rendered[relative]
                    if relative in rendered
                    else (fixture.stage / relative).read_bytes()
                )
                self.assertEqual(entry["role"], RENDERER.MATRIX_ROLES[relative])
                self.assertEqual(entry["bytes"], len(raw))
                self.assertEqual(entry["sha256"], sha256(raw))

            summary = rendered["summary.md"].decode("ascii")
            self.assertIn("20260711-sq8-full-campaign", summary)
            self.assertNotIn("passed", summary.lower())
            self.assertNotIn("verdict", summary.lower())
            for relative in RENDERER.PREVALIDATION_PATHS:
                self.assertIn(f"`{relative}`", summary)

            lines = rendered["SHA256SUMS"].decode("ascii").splitlines()
            self.assertEqual(len(lines), 19)
            observed_paths = [line.split("  ", 1)[1] for line in lines]
            self.assertEqual(observed_paths, list(RENDERER.SHA256SUM_INPUT_PATHS))
            for line, relative in zip(
                lines, RENDERER.SHA256SUM_INPUT_PATHS, strict=True
            ):
                digest, observed = line.split("  ", 1)
                self.assertEqual(observed, relative)
                raw = (
                    rendered[relative]
                    if relative in rendered
                    else (fixture.stage / relative).read_bytes()
                )
                self.assertEqual(digest, sha256(raw))

    def test_orchestrator_context_passes_actual_collector_result_to_renderer(
        self,
    ) -> None:
        with CampaignFixture() as fixture:
            restart_resource = COLLECTOR.ResourceSegmentResult(
                segment="restart",
                identity=COLLECTOR.ProcessIdentity(
                    "/system.slice/ullm-openai.service",
                    200,
                    2000,
                    201,
                    2001,
                    3,
                ),
                warmup_requests=10,
                measured_requests=20,
                negative_requests=0,
                resource_samples=105,
                gpu_metrics=2,
                sampling_cases=(),
            )
            evidence = ORCHESTRATOR.CampaignEvidence(
                fixture.evidence.preflight,
                fixture.evidence.api_contract,
                fixture.evidence.combined,
                fixture.evidence.direct_cancel,
                fixture.evidence.stop,
                fixture.evidence.resource_normal,
                fixture.evidence.failure,
                restart_resource,
                fixture.evidence.latency,
                types.SimpleNamespace(),
            )
            context = ORCHESTRATOR.RenderContext(fixture.stage, evidence)
            rendered = RENDERER.FullCampaignRenderer().render(context)
            self.assertEqual(set(rendered), RENDERER.RENDERED_PATHS)
            for relative, raw in rendered.items():
                path = fixture.stage / relative
                path.write_bytes(raw)
                os.chmod(path, 0o600)
            actual_files = {
                path.relative_to(fixture.stage).as_posix()
                for path in fixture.stage.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual_files, RENDERER.PREVALIDATION_PATHS)
            checksum_lines = (
                (fixture.stage / "SHA256SUMS").read_text(encoding="ascii").splitlines()
            )
            self.assertEqual(len(checksum_lines), 19)
            for line, relative in zip(
                checksum_lines, RENDERER.SHA256SUM_INPUT_PATHS, strict=True
            ):
                digest, observed = line.split("  ", 1)
                self.assertEqual(observed, relative)
                self.assertEqual(
                    digest, sha256((fixture.stage / relative).read_bytes())
                )

    def test_matrix_and_checksum_sets_exclude_recursive_outputs(self) -> None:
        self.assertEqual(len(RENDERER.MATRIX_INPUT_PATHS), 17)
        self.assertTrue(
            {
                "release-matrix.json",
                "summary.md",
                "SHA256SUMS",
                "release-validation.json",
            }.isdisjoint(RENDERER.MATRIX_INPUT_PATHS)
        )
        self.assertIn("release-matrix.json", RENDERER.SHA256SUM_INPUT_PATHS)
        self.assertIn("summary.md", RENDERER.SHA256SUM_INPUT_PATHS)
        self.assertNotIn("SHA256SUMS", RENDERER.SHA256SUM_INPUT_PATHS)
        self.assertNotIn("release-validation.json", RENDERER.SHA256SUM_INPUT_PATHS)
        self.assertEqual(
            list(RENDERER.MATRIX_INPUT_PATHS),
            sorted(RENDERER.MATRIX_INPUT_PATHS, key=lambda item: item.encode("utf-8")),
        )
        self.assertEqual(
            list(RENDERER.SHA256SUM_INPUT_PATHS),
            sorted(
                RENDERER.SHA256SUM_INPUT_PATHS,
                key=lambda item: item.encode("utf-8"),
            ),
        )

    def test_one_byte_change_between_matrix_and_checksum_passes_is_rejected(
        self,
    ) -> None:
        with CampaignFixture() as fixture:
            target = fixture.stage / "environment.json"
            original = RENDERER._StageSnapshot.hash_file
            changed = False

            def mutate_after_first_hash(
                snapshot: Any, relative: str, *, expected: Any | None = None
            ) -> Any:
                nonlocal changed
                result = original(snapshot, relative, expected=expected)
                if relative == "environment.json" and expected is None and not changed:
                    raw = target.read_bytes()
                    target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
                    os.chmod(target, 0o600)
                    changed = True
                return result

            with mock.patch.object(
                RENDERER._StageSnapshot,
                "hash_file",
                new=mutate_after_first_hash,
            ):
                with self.assertRaisesRegex(
                    RENDERER.FullCampaignRendererError,
                    "changed between (rendering )?passes|changed while",
                ):
                    fixture.render()

    def test_one_byte_change_after_checksum_pass_is_rejected(self) -> None:
        with CampaignFixture() as fixture:
            target = fixture.stage / "environment.json"
            original = RENDERER._StageSnapshot.hash_file
            changed = False

            def mutate_after_second_hash(
                snapshot: Any, relative: str, *, expected: Any | None = None
            ) -> Any:
                nonlocal changed
                result = original(snapshot, relative, expected=expected)
                if (
                    relative == "environment.json"
                    and expected is not None
                    and not changed
                ):
                    raw = target.read_bytes()
                    target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
                    os.chmod(target, 0o600)
                    changed = True
                return result

            with mock.patch.object(
                RENDERER._StageSnapshot,
                "hash_file",
                new=mutate_after_second_hash,
            ):
                with self.assertRaisesRegex(
                    RENDERER.FullCampaignRendererError,
                    "changed before render completion",
                ):
                    fixture.render()

    def test_entry_replacement_between_lstat_and_open_is_rejected(self) -> None:
        with CampaignFixture() as fixture:
            target = fixture.stage / "environment.json"
            real_open = RENDERER.os.open
            replaced = False

            def replace_then_open(
                path: Any,
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                nonlocal replaced
                if path == "environment.json" and dir_fd is not None and not replaced:
                    target.unlink()
                    target.write_bytes(b"replacement\n")
                    os.chmod(target, 0o600)
                    replaced = True
                if dir_fd is None:
                    return cast(int, real_open(path, flags, mode))
                return cast(int, real_open(path, flags, mode, dir_fd=dir_fd))

            with mock.patch.object(RENDERER.os, "open", side_effect=replace_then_open):
                with self.assertRaisesRegex(
                    RENDERER.FullCampaignRendererError, "changed while opening"
                ):
                    fixture.render()

    def test_parent_directory_symlink_is_rejected(self) -> None:
        with CampaignFixture() as fixture:
            parent = Path(fixture.temporary.name)
            alias = parent / "stage-parent-alias"
            alias.symlink_to(parent, target_is_directory=True)
            fixture.context.stage_path = alias / "stage"
            with self.assertRaisesRegex(
                RENDERER.FullCampaignRendererError, "contains a symbolic link"
            ):
                fixture.render()

    def test_missing_extra_nonregular_mode_owner_and_hardlink_are_rejected(
        self,
    ) -> None:
        for defect in ("missing", "extra", "symlink", "mode", "owner", "hardlink"):
            with self.subTest(defect=defect), CampaignFixture() as fixture:
                target = fixture.stage / "environment.json"
                owner_patch = None
                if defect == "missing":
                    target.unlink()
                elif defect == "extra":
                    extra = fixture.stage / "producer-verdict.json"
                    extra.write_bytes(b"extra\n")
                    os.chmod(extra, 0o600)
                elif defect == "symlink":
                    target.unlink()
                    target.symlink_to(fixture.stage / "model-identity.json")
                elif defect == "mode":
                    os.chmod(target, 0o640)
                elif defect == "owner":
                    owner_patch = mock.patch.object(
                        RENDERER.os, "geteuid", return_value=os.geteuid() + 1
                    )
                else:
                    source = fixture.stage / "hardlink-source"
                    target.rename(source)
                    os.link(source, target)
                    source.unlink()
                    outside = Path(fixture.temporary.name) / "outside-link"
                    os.link(target, outside)
                manager = owner_patch if owner_patch is not None else _NullContext()
                with manager:
                    with self.assertRaises(RENDERER.FullCampaignRendererError):
                        fixture.render()

    def test_sampling_cases_must_be_explicit_and_exact(self) -> None:
        with CampaignFixture() as fixture:
            fixture.evidence.resource_normal = types.SimpleNamespace()
            with self.assertRaisesRegex(
                RENDERER.FullCampaignRendererError, "lacks sampling_cases"
            ):
                fixture.render()

        with CampaignFixture() as fixture:
            cases = list(fixture.evidence.resource_normal.sampling_cases)
            cases[0] = {**cases[0], "request_index": 6}
            fixture.evidence.resource_normal = dataclasses.replace(
                fixture.evidence.resource_normal,
                sampling_cases=tuple(cases),
            )
            with self.assertRaisesRegex(
                RENDERER.FullCampaignRendererError,
                "failed to build canonical full campaign views",
            ):
                fixture.render()

    def test_passed_is_rejected_and_secret_scanning_remains_upstream(self) -> None:
        with CampaignFixture() as fixture:
            fixture.evidence.preflight.header_fields["schedule"]["passed"] = True
            with self.assertRaisesRegex(
                RENDERER.FullCampaignRendererError, "forbidden key passed"
            ):
                fixture.render()

        secret = "private-release-token"
        with CampaignFixture(run_id=secret) as fixture:
            rendered = fixture.render()
            self.assertIn(secret.encode("ascii"), rendered["release-matrix.json"])
            self.assertIn(secret.encode("ascii"), rendered["summary.md"])
            self.assertFalse(
                any(
                    path.name == "release-validation.json"
                    for path in fixture.stage.iterdir()
                )
            )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
