"""Differential host/stdlib checks for all semantic G1 schemas."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, RefResolver

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import validate_rmsnorm_g1_contracts as contracts  # noqa: E402


SCHEMA_PATHS = tuple(sorted((ROOT / "ci/schema").glob("rmsnorm-semantic-g1-*.schema.json")))


def _stdlib_valid(document: object, schema: dict[str, object]) -> bool:
    return not contracts._closed_schema_errors(document, schema, schema, "<root>")


def _host_valid(document: object, schema: dict[str, object]) -> bool:
    return not list(Draft202012Validator(schema, format_checker=contracts._format_checker()).iter_errors(document))


def _fragment_valid(document: object, fragment: dict[str, object], root: dict[str, object]) -> bool:
    errors = contracts._closed_schema_errors(document, fragment, root, "<fragment>")
    resolver = RefResolver.from_schema(root)
    host_errors = list(Draft202012Validator(fragment, resolver=resolver, format_checker=contracts._format_checker()).iter_errors(document))
    if bool(errors) != bool(host_errors):
        raise AssertionError(f"stdlib/host disagreement: {errors!r} / {host_errors!r}")
    return not errors


def _schema_example(node: dict[str, object], root: dict[str, object]) -> object:
    """Build a small valid instance for the closed schema subset."""

    if "$ref" in node:
        return _schema_example(root["$defs"][str(node["$ref"])[8:]], root)  # type: ignore[index]
    if "const" in node:
        return copy.deepcopy(node["const"])
    if "oneOf" in node:
        return _schema_example(node["oneOf"][0], root)  # type: ignore[index]
    if "allOf" in node:
        result: object = _schema_example({key: value for key, value in node.items() if key != "allOf"}, root) if node.get("type") else {}
        for child in node["allOf"]:  # type: ignore[union-attr]
            value = _schema_example(child, root)  # type: ignore[arg-type]
            if isinstance(result, dict) and isinstance(value, dict):
                result.update(value)
            elif result == {}:
                result = value
        return result
    if "enum" in node:
        return copy.deepcopy(node["enum"][0])  # type: ignore[index]
    if "properties" in node and "type" not in node:
        properties = node["properties"]
        return {
            name: _schema_example(child, root)
            for name, child in properties.items()  # type: ignore[union-attr]
        }
    schema_type = node.get("type")
    if schema_type == "object":
        properties = node.get("properties", {})
        return {
            name: _schema_example(properties[name], root)  # type: ignore[index]
            for name in node.get("required", [])  # type: ignore[union-attr]
            if name in properties  # type: ignore[operator]
        }
    if schema_type == "array":
        prefix = [_schema_example(child, root) for child in node.get("prefixItems", [])]  # type: ignore[arg-type]
        items = node.get("items")
        minimum = int(node.get("minItems", 0))
        if isinstance(items, dict):
            prefix.extend(_schema_example(items, root) for _ in range(max(0, minimum - len(prefix))))
        return prefix
    if schema_type == "string":
        pattern = str(node.get("pattern", ""))
        if "date-time" in node.get("format", ""):
            return "2026-08-07T12:34:56Z"
        if "64" in pattern and "[0-9a-f]" in pattern:
            return "a" * 64
        if "40" in pattern and "[0-9a-f]" in pattern:
            return "a" * 40
        if "[0-9a-f]*" in pattern:
            return ""
        if "^/proc/" in pattern:
            return "/proc/1/exe"
        if "^/tmp/sllm-exact-input-view-" in pattern:
            return "/tmp/sllm-exact-input-view-example"
        if "^rows/" in pattern:
            return "rows/rmsnorm-semantic-g1-gfx1030/raw/case-0.bin.sha256" if "sha256" in pattern else "rows/rmsnorm-semantic-g1-gfx1030/raw/case-0.bin"
        if "^[^/" in pattern:
            return "relative"
        if pattern.startswith("^/"):
            return "/tmp/value"
        return "x"
    if schema_type == "integer":
        return max(1, int(node.get("minimum", 0)))
    if schema_type == "number":
        return max(1, int(node.get("minimum", 0)))
    if schema_type == "boolean":
        return True
    raise AssertionError(f"cannot build schema example for {node!r}")


class SemanticG1SchemaTests(unittest.TestCase):
    def test_all_four_schemas_are_valid_and_keyword_sets_have_differential_coverage(self) -> None:
        self.assertEqual(len(SCHEMA_PATHS), 4)
        used_keywords: set[str] = set()
        for path in SCHEMA_PATHS:
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            self.assertFalse(_stdlib_valid({}, schema))
            self.assertEqual(_stdlib_valid({}, schema), _host_valid({}, schema), path.name)
            for value in (None, [], "", 0, True, {"forged": True}):
                self.assertEqual(_stdlib_valid(value, schema), _host_valid(value, schema), f"{path.name}: {value!r}")

            def collect(node: object) -> None:
                if isinstance(node, dict):
                    used_keywords.update(node)
                    for child in node.values():
                        collect(child)
                elif isinstance(node, list):
                    for child in node:
                        collect(child)

            collect(schema)

        self.assertTrue({"$ref", "type", "const", "enum", "required", "properties", "additionalProperties", "pattern", "minLength", "minimum", "minItems", "maxItems", "prefixItems", "items", "allOf", "oneOf", "format"} <= used_keywords)

    def test_every_validation_keyword_has_positive_and_negative_differential_coverage(self) -> None:
        """Exercise each validation keyword through both validator implementations.

        The probes are deliberately small schemas: the production schemas are
        checked above, while these probes make the stdlib/host equivalence
        requirement observable for every validation branch used by them.
        """

        probes = {
            "type": ({"type": "string"}, "ok", 1),
            "const": ({"const": 1}, 1, 2),
            "enum": ({"enum": [1, 2]}, 1, 3),
            "required": ({"type": "object", "required": ["x"]}, {"x": 1}, {}),
            "properties": ({"type": "object", "properties": {"x": {"const": 1}}}, {"x": 1}, {"x": 2}),
            "additionalProperties": ({"type": "object", "properties": {"x": {}}, "additionalProperties": False}, {}, {"y": 1}),
            "pattern": ({"type": "string", "pattern": "^a+$"}, "aaa", "aba"),
            "minLength": ({"type": "string", "minLength": 2}, "aa", "a"),
            "maxLength": ({"type": "string", "maxLength": 2}, "aa", "aaa"),
            "minimum": ({"type": "number", "minimum": 1}, 1, 0),
            "maximum": ({"type": "number", "maximum": 1}, 1, 2),
            "minItems": ({"type": "array", "minItems": 2}, [1, 2], [1]),
            "maxItems": ({"type": "array", "maxItems": 2}, [1, 2], [1, 2, 3]),
            "prefixItems": ({"type": "array", "prefixItems": [{"const": 1}]}, [1], [2]),
            "items": ({"type": "array", "items": False}, [], [1]),
            "allOf": ({"allOf": [{"const": 1}]}, 1, 2),
            "oneOf": ({"oneOf": [{"const": 1}, {"const": 2}]}, 1, 3),
            "format": ({"type": "string", "format": "date-time"}, "2026-08-07T12:34:56Z", "2026-08-07 12:34:56Z"),
        }
        used_keywords: set[str] = set()
        for path in SCHEMA_PATHS:
            schema = json.loads(path.read_text(encoding="utf-8"))

            def collect(node: object) -> None:
                if isinstance(node, dict):
                    used_keywords.update(node)
                    for child in node.values():
                        collect(child)
                elif isinstance(node, list):
                    for child in node:
                        collect(child)

            collect(schema)
            for keyword in set(probes) & used_keywords:
                fragment, positive, negative = probes[keyword]
                self.assertTrue(_fragment_valid(positive, fragment, {"$defs": {}}), f"{path.name}:{keyword}:positive")
                self.assertFalse(_fragment_valid(negative, fragment, {"$defs": {}}), f"{path.name}:{keyword}:negative")
        self.assertTrue(set(probes) <= used_keywords)

    def test_draft_2020_12_overlap_boolean_and_array_semantics_are_differential(self) -> None:
        cases = (
            (
                "explicit-property-and-pattern-both-apply-valid",
                {"type": "object", "properties": {"x": {"const": 1}}, "patternProperties": {"^x$": {"const": 1}}, "additionalProperties": False},
                {"x": 1},
                True,
            ),
            (
                "explicit-property-and-pattern-both-apply-invalid",
                {"type": "object", "properties": {"x": {"const": 1}}, "patternProperties": {"^x$": {"const": 2}}, "additionalProperties": False},
                {"x": 1},
                False,
            ),
            (
                "multiple-patterns-all-apply-valid",
                {"type": "object", "patternProperties": {"^x": {"type": "integer"}, "x$": {"minimum": 2}}, "additionalProperties": False},
                {"x": 2},
                True,
            ),
            (
                "multiple-patterns-all-apply-invalid",
                {"type": "object", "patternProperties": {"^x": {"type": "integer"}, "x$": {"minimum": 2}}, "additionalProperties": False},
                {"x": 1},
                False,
            ),
            (
                "ref-sibling-valid",
                {"$defs": {"value": {"type": "integer", "minimum": 1}}, "$ref": "#/$defs/value", "maximum": 3},
                2,
                True,
            ),
            (
                "ref-sibling-invalid",
                {"$defs": {"value": {"type": "integer", "minimum": 1}}, "$ref": "#/$defs/value", "maximum": 3},
                4,
                False,
            ),
            ("unique-items-valid", {"type": "array", "uniqueItems": True}, [1, 2], True),
            ("unique-items-invalid", {"type": "array", "uniqueItems": True}, [1, 1], False),
            ("items-false-valid-after-prefix", {"type": "array", "prefixItems": [{"const": 1}], "items": False}, [1], True),
            ("items-false-invalid-after-prefix", {"type": "array", "prefixItems": [{"const": 1}], "items": False}, [1, 2], False),
            ("false-prefix-items-valid-when-absent", {"type": "array", "prefixItems": [False]}, [], True),
            ("false-prefix-items-invalid-when-present", {"type": "array", "prefixItems": [False]}, [1], False),
        )
        for label, schema, document, expected in cases:
            with self.subTest(label=label):
                custom_errors = contracts._closed_schema_errors(document, schema, schema, "<root>")
                draft_errors = list(Draft202012Validator(schema).iter_errors(document))
                self.assertEqual(not custom_errors, not draft_errors)
                self.assertEqual(not custom_errors, expected)

    def test_candidate_git_branches_and_record_keywords_match_host_and_stdlib(self) -> None:
        candidate = {
            "reviewed_sha": "a" * 40, "tested_sha": "a" * 40, "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40, "git_object_format": "sha1", "git_oid_width": 40,
            "worktree_clean": True, "revision_input": "full-sha",
        }
        record = {"path": "/bin/true", "resolved_path": "/usr/bin/true", "size_bytes": 1, "sha256": "c" * 64}
        for path in SCHEMA_PATHS:
            schema = json.loads(path.read_text(encoding="utf-8"))
            for name, value in (("candidate", candidate), ("record", record)):
                fragment = schema.get("$defs", {}).get(name)
                if not isinstance(fragment, dict):
                    continue
                self.assertTrue(_fragment_valid(value, fragment, schema), f"{path.name}:{name}")
                negative = copy.deepcopy(value)
                if name == "candidate":
                    negative["git_tree_oid"] = "b" * 39
                else:
                    negative["size_bytes"] = 0
                self.assertFalse(_fragment_valid(negative, fragment, schema), f"{path.name}:{name}:negative")
                if name == "candidate":
                    for width in (41, 63):
                        negative = copy.deepcopy(value)
                        negative["git_tree_oid"] = "b" * width
                        self.assertFalse(_fragment_valid(negative, fragment, schema), f"{path.name}:{name}:oid-{width}")

    def test_exact_action_transcripts_require_issuance_identity_and_consumption(self) -> None:
        digest = "a" * 64
        record = {"path": "/tmp/tool", "resolved_path": "/tmp/tool", "size_bytes": 1, "sha256": digest}
        directory = {"path": "/tmp", "resolved_path": "/tmp", "device": 1, "inode": 1}
        executable = {**record, "device": 1, "inode": 1, "seals": 15}
        manifest = {
            "schema_version": "exact-parent-action-manifest-v1",
            "action_id": digest,
            "manifest_digest": "b" * 64,
            "executable": executable,
            "argv0": "/proc/self/fd/198",
            "argv": ["-c", "source"],
            "cwd": directory,
            "environment": [["PATH", "/usr/bin:/bin"]],
            "inputs": [],
            "implicit": [],
            "response_files": [],
            "outputs": [],
            "target": "gfx1030",
            "occurrence_index": 0,
            "occurrence_limit": 1,
        }
        binding = {"pid": 1, "starttime": 2, "uid": 0, "gid": 0}
        result = {
            "pid": 3, "starttime": 4, "ppid": 1, "pgrp": 3, "exit_code": 0,
            "stdout_b64": "", "stderr_b64": "", "stdout_sha256": "c" * 64,
            "stderr_sha256": "c" * 64, "duration_ns": 1, "status": "ok",
            "timed_out": False, "crashed": False,
            "invocation": {"action_manifest": manifest, "materialized_outputs": [], "sealed_input_view": {"algorithm": "sealed-input-view-v1", "argv": ["/proc/self/fd/300"], "argv_sha256": "d" * 64, "inputs": [], "include_directories": [], "sealed": True}},
            "kernel_limits": {"address_space_bytes": 1, "process_count": 1, "rss_bytes": 1, "enforced_by": "/usr/bin/prlimit", "address_space_enforcement": "kernel", "process_count_enforcement": "kernel", "rss_enforcement": "parent"},
            "action_id": digest, "action_digest": "b" * 64,
            "exec_identity": {"pid": 3, "starttime": 4, "ppid": 1, "pgrp": 3, "exe_dev": 1, "exe_ino": 1, "sealed_dev": 1, "sealed_ino": 1, "exe_path": "/proc/3/exe", "argv_sha256": "d" * 64, "cwd": "/tmp", "exec_ready": True},
        }
        event = {
            "sequence": 0, "request_nonce": "e" * 64, "observation_nonce": "f" * 64,
            "client_observation": {"observation_nonce": "f" * 64, "argv": ["-c", "source"], "cwd": "/tmp", "environment_sha256": "1" * 64, "client_binding": binding},
            "client_binding": binding, "action_id": digest, "action_digest": "b" * 64,
            "action_manifest": manifest, "request_frame_sha256": "2" * 64,
            "response_frame_sha256": "3" * 64, "ack_frame_sha256": "4" * 64,
            "compiler_source_sha256": digest, "compiler": result,
            "started_at_ns": 1, "finished_at_ns": 2, "consumed": True, "acknowledged": True,
        }
        transcript = {
            "protocol": "parent-owned-exact-action-broker-v1",
            "event_protocol": "parent-issued-exact-action-v1",
            "actions": [{
                "recipe_key": "reviewed-recipe",
                "action_id": digest,
                "action_digest": "b" * 64,
                "state": "consumed",
                "issued_at_ns": 1,
                "consumed_at_ns": 2,
                "manifest": manifest,
            }],
            "source": record, "client": record, "exec_helper": record, "session": "5" * 64,
            "request_count": 1, "events_sha256": "6" * 64, "expected_recipe_keys": ["reviewed-recipe"],
            "closure": {"state": "closed", "build_root_pid": 1, "build_root_starttime": 1, "build_root_pgrp": 1, "build_tree_reaped": True, "listener_closed": True, "active_requests": 0, "quiescence_rounds": 3, "state_machine": "new-running-closing-closed-v1", "request_count": 1, "last_sequence": 0, "events_sha256": "6" * 64},
            "events": [event],
        }
        for path in SCHEMA_PATHS:
            schema = json.loads(path.read_text(encoding="utf-8"))
            fragment = schema.get("$defs", {}).get("exact_action_transcript")
            if not isinstance(fragment, dict):
                continue
            self.assertTrue(_fragment_valid(transcript, fragment, schema), f"{path.name}:valid")
            forged = copy.deepcopy(transcript)
            forged["events"][0]["consumed"] = False
            self.assertFalse(_fragment_valid(forged, fragment, schema), f"{path.name}:consumption")
            forged = copy.deepcopy(transcript)
            forged["actions"][0]["action_id"] = "not-an-opaque-id"
            self.assertFalse(_fragment_valid(forged, fragment, schema), f"{path.name}:action-id")
            forged = copy.deepcopy(transcript)
            forged["actions"][0]["manifest"]["executable"]["forged"] = True
            self.assertFalse(_fragment_valid(forged, fragment, schema), f"{path.name}:nested-executable-open")
            forged = copy.deepcopy(transcript)
            forged["events"][0]["client_observation"]["environment_sha256"] = "too-short"
            self.assertFalse(_fragment_valid(forged, fragment, schema), f"{path.name}:nested-observation-digest")
            forged = copy.deepcopy(transcript)
            forged["events"][0]["compiler"]["invocation"]["sealed_input_view"]["inputs"] = [
                {**record, "role": "source", "device": 1, "inode": 1, "view_fd": 2}
            ]
            self.assertFalse(_fragment_valid(forged, fragment, schema), f"{path.name}:standard-view-fd")

        artifact_schema = json.loads((ROOT / "ci/schema/rmsnorm-semantic-g1-artifact-v1.schema.json").read_text(encoding="utf-8"))
        aggregate_schema = json.loads((ROOT / "ci/schema/rmsnorm-semantic-g1-aggregate-v1.schema.json").read_text(encoding="utf-8"))
        report_schema = json.loads((ROOT / "ci/schema/rmsnorm-semantic-g1-report-v1.schema.json").read_text(encoding="utf-8"))
        for definition in ("exact_action_manifest", "exact_action_issuance", "exact_action_event", "exact_action_transcript"):
            self.assertEqual(report_schema["$defs"][definition], artifact_schema["$defs"][definition], definition)
            self.assertEqual(aggregate_schema["$defs"][definition], artifact_schema["$defs"][definition], definition)
        aggregate_row = {
            "row_id": "rmsnorm-semantic-g1-gfx1030", "target": "gfx1030", "state": "PASS",
            "report_sha256": "7" * 64, "binary_sha256": "8" * 64,
            "companion_sha256": "9" * 64, "loader_sha256": "a" * 64,
            "runtime_library_sha256": "b" * 64, "runtime_dependency_closure_sha256": "c" * 64,
            "raw_frame_sha256": "d" * 64,
            "response_evidence": [
                {
                    "path": f"rows/rmsnorm-semantic-g1-gfx1030/raw/case-{order}.bin",
                    "sidecar_path": f"rows/rmsnorm-semantic-g1-gfx1030/raw/case-{order}.bin.sha256",
                    "size_bytes": 1,
                    "sha256": "1" * 64,
                    "sidecar_sha256": "2" * 64,
                    "candidate_sha256": "3" * 64,
                    "row_id": "rmsnorm-semantic-g1-gfx1030",
                    "case_id": f"case-{order}",
                    "order": order,
                }
                for order in range(15)
            ],
            "resource_counts": {"allocation_count": 1, "copy_count": 1, "dispatch_count": 1, "kernel_count": 1},
            "compiler_execution_sha256": "e" * 64, "compiler_execution": transcript,
        }
        self.assertTrue(_fragment_valid(aggregate_row, aggregate_schema["$defs"]["aggregate_row"], aggregate_schema))
        aggregate_row["compiler_execution"]["events"][0]["acknowledged"] = False
        self.assertFalse(_fragment_valid(aggregate_row, aggregate_schema["$defs"]["aggregate_row"], aggregate_schema))

    def test_date_time_positive_negative_branch_is_equivalent(self) -> None:
        fragment = {"type": "string", "format": "date-time"}
        for value, expected in (("2026-08-07T12:34:56Z", True), ("2026-08-07T12:34:56+09:00", True), ("2026-08-07 12:34:56Z", False), ("2026-08-07T12:34:56", False)):
            self.assertEqual(_fragment_valid(value, fragment, {"$defs": {}}), expected)

    def test_every_object_boundary_is_closed_and_rejects_forged_keys(self) -> None:
        checked = 0
        for path in SCHEMA_PATHS:
            schema = json.loads(path.read_text(encoding="utf-8"))

            def walk(node: object, schema_path: str) -> None:
                nonlocal checked
                if isinstance(node, dict):
                    if node.get("type") == "object":
                        self.assertIs(node.get("additionalProperties"), False, f"{path.name}:{schema_path}")
                        example = _schema_example(node, schema)
                        self.assertTrue(_fragment_valid(example, node, schema), f"{path.name}:{schema_path}:example")
                        forged = copy.deepcopy(example)
                        self.assertIsInstance(forged, dict)
                        forged["__forged_semantic_g1_key__"] = True  # type: ignore[index]
                        self.assertFalse(_fragment_valid(forged, node, schema), f"{path.name}:{schema_path}:forged")
                        checked += 1
                    for name, child in node.items():
                        walk(child, f"{schema_path}/{name}")
                elif isinstance(node, list):
                    for index, child in enumerate(node):
                        walk(child, f"{schema_path}/{index}")

            walk(schema, "")
        self.assertGreater(checked, 40)


if __name__ == "__main__":
    unittest.main()
