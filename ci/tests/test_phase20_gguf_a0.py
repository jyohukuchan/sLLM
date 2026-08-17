from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ci" / "schema" / "phase20-gguf-a0-v1.schema.json"
MANIFEST_PATH = ROOT / "ci" / "matrix" / "phase20-gguf-a0-v1.json"

EXPECTED_HANDOFF_IDS = (
    "qwen35-4b-bf16-dense",
    "gemma4-12b-bf16-dense",
    "gemma4-12b-nvfp4-mixed",
    "qwen35-35b-a3b-mxfp4-moe",
)

EXPECTED_SOURCE_SHA256 = {
    "reference/llama.cpp/ggml/include/gguf.h": "e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4",
    "reference/llama.cpp/ggml/src/gguf.cpp": "615894b49182f94d711280d570a215860e1247ecc05d4289c9382295170b53bc",
    "reference/llama.cpp/gguf-py/gguf/constants.py": "1db4fa8c0defa3910dae4e8f706519ec46cd65c2eb54ebbdca91086173f7aae2",
    "reference/llama.cpp/gguf-py/gguf/gguf_reader.py": "f1784b59de0f6ef5454091acd0997214a0c3151c0f33ee1f6394eca8441a0002",
    "reference/llama.cpp/gguf-py/gguf/gguf_writer.py": "9216da38c4bc5d7d9f2693125708fbaf64868e5447d26f186dbd07e0d059e594",
    "reference/llama.cpp/gguf-py/gguf/quants.py": "2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b",
    "reference/llama.cpp/conversion/qwen.py": "65c61155458078232dd3f9d23284710fa39c29cf7ecc341194c825cf5334f43f",
    "reference/llama.cpp/conversion/gemma.py": "b43670f30a470a3099f6475cb639a7cdb327b37e98467c96d288c8db862bfa76",
}


def load_schema() -> dict[str, object]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validation_errors(schema: dict[str, object], document: dict[str, object]) -> list[object]:
    validator = Draft202012Validator(schema)
    return sorted(validator.iter_errors(document), key=lambda error: list(error.path))


class Phase20GGUFa0ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_schema()
        self.manifest = load_manifest()

    def test_schema_is_valid_draft202012(self) -> None:
        Draft202012Validator.check_schema(self.schema)

    def test_manifest_passes_schema_validation(self) -> None:
        self.assertEqual(validation_errors(self.schema, self.manifest), [])

    def test_standard_encoding_mapping_is_exact(self) -> None:
        encodings = {entry["semantic"]: entry for entry in self.manifest["tensor_encodings"]}
        expected = {
            "bf16": (30, 1, 2),
            "mxfp4-e2m1-block32-e8m0": (39, 32, 17),
            "nvfp4-e2m1-block16-e4m3fn-f32-outer": (40, 64, 36),
        }
        for semantic, (ggml_type, block_size, type_size) in expected.items():
            entry = encodings[semantic]
            self.assertEqual(entry["status"], "standard", semantic)
            self.assertEqual(
                (entry["ggml_type"], entry["block_size"], entry["type_size"]),
                (ggml_type, block_size, type_size),
                semantic,
            )

    def test_fp8_is_extension_required_without_substitution(self) -> None:
        encodings = {entry["semantic"]: entry for entry in self.manifest["tensor_encodings"]}
        fp8 = encodings["fp8-e4m3fn-channel-bf16-scale"]
        self.assertEqual(fp8["status"], "extension-required")
        self.assertIsNone(fp8["ggml_type"])
        self.assertIsNone(fp8["block_size"])
        self.assertIsNone(fp8["type_size"])
        self.assertEqual(fp8["conversion"], "not-defined-in-a0")
        self.assertIn(
            "no-dequantized-substitution",
            self.manifest["decisions"]["fp8"],
        )

    def test_handoff_ids_are_exactly_the_expected_unique_set(self) -> None:
        ids = [handoff["id"] for handoff in self.manifest["handoffs"]]
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, list(EXPECTED_HANDOFF_IDS))

    def test_descriptor_source_paths_exist(self) -> None:
        for handoff in self.manifest["handoffs"]:
            for source in handoff["descriptor_sources"]:
                self.assertTrue(
                    (ROOT / source).is_file(),
                    f"missing descriptor source {source!r} for handoff {handoff['id']!r}",
                )

    def test_source_files_are_repository_relative_and_unique(self) -> None:
        paths = [entry["path"] for entry in self.manifest["source"]["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertFalse(path.startswith("/"), path)
            self.assertFalse(Path(path).is_absolute(), path)
            self.assertNotIn("..", Path(path).parts, path)
            self.assertTrue(path.startswith("reference/llama.cpp/"), path)

    def test_source_file_identity_set_is_exact(self) -> None:
        actual = {
            entry["path"]: entry["sha256"]
            for entry in self.manifest["source"]["files"]
        }
        self.assertEqual(actual, EXPECTED_SOURCE_SHA256)

    def test_handoff_identities_and_core_inventory_are_frozen(self) -> None:
        handoffs = {entry["id"]: entry for entry in self.manifest["handoffs"]}
        self.assertEqual(
            handoffs["qwen35-4b-bf16-dense"]["semantic_fingerprint"],
            "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae",
        )
        self.assertEqual(
            handoffs["qwen35-4b-bf16-dense"]["inventory"]["physical_tensors"],
            738,
        )
        self.assertEqual(
            handoffs["gemma4-12b-bf16-dense"]["inventory"],
            {
                "physical_tensors": 677,
                "loadable_text_tensors": 666,
                "catalog_sha256": "24e705586f0bba5e1018951a9ee09aa02b1bfccd73f5c0a82e31e29fb7c2931f",
                "component_scope": "text-only",
            },
        )
        self.assertEqual(
            handoffs["gemma4-12b-nvfp4-mixed"]["inventory"]["recipe_digest"],
            "sha256:e64f38576cffd36fac5f55d5e7c47846afdc59ef8ef5aec24b66f090aa8522e2",
        )
        moe = handoffs["qwen35-35b-a3b-mxfp4-moe"]
        self.assertEqual(moe["inventory"]["text_tensors"], 62_053)
        self.assertEqual(moe["inventory"]["load_plan_entries"], 493)
        self.assertEqual(
            moe["inventory"]["load_plan_digest"],
            "sha256:f96a3389cfaca4ab947fe060ccd6f048d078946e704464277d87019a13fb7ae4",
        )

    def test_tampered_bf16_ggml_type_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        for entry in tampered["tensor_encodings"]:
            if entry["semantic"] == "bf16":
                entry["ggml_type"] = "30"
        self.assertGreater(len(validation_errors(self.schema, tampered)), 0)

    def test_tampered_source_release_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["source"]["release"] = "b99999"
        self.assertGreater(len(validation_errors(self.schema, tampered)), 0)

    def test_tampered_empty_descriptor_sources_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["handoffs"][0]["descriptor_sources"] = []
        self.assertGreater(len(validation_errors(self.schema, tampered)), 0)

    def test_tampered_duplicate_descriptor_sources_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["handoffs"][1]["descriptor_sources"].append(
            tampered["handoffs"][1]["descriptor_sources"][0]
        )
        self.assertGreater(len(validation_errors(self.schema, tampered)), 0)

    def test_tampered_duplicate_source_file_path_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        files = tampered["source"]["files"]
        files.append(copy.deepcopy(files[0]))
        self.assertGreater(len(validation_errors(self.schema, tampered)), 0)

    def test_tampered_standard_first_decision_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["decisions"]["standard_first"] = False
        self.assertGreater(len(validation_errors(self.schema, tampered)), 0)


if __name__ == "__main__":
    unittest.main()
