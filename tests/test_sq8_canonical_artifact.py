from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "sq8_canonical_artifact.py"
WEIGHT_NAME = "model.layers.0.self_attn.q_proj.weight"
SCALE_NAME = f"{WEIGHT_NAME}_scale_inv"


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_canonical_artifact", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SQ8 = load_module()


def source_weight() -> torch.Tensor:
    values = (torch.arange(259 * 257, dtype=torch.float32).reshape(259, 257) % 29 - 14) / 8
    return values.to(torch.float8_e4m3fn)


def source_scale() -> torch.Tensor:
    return torch.tensor(
        [
            [2.0**-8, 2.0**-7, 2.0**-6],
            [2.0**-5, 2.0**-4, 2.0**-3],
            [2.0**-2, 2.0**-1, 1.0],
        ],
        dtype=torch.bfloat16,
    )


def write_model(
    root: Path,
    *,
    weight: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    include_weight: bool = True,
    include_scale: bool = True,
) -> Path:
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    config = {
        "model_type": "qwen3",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.embed_tokens.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
    }
    if include_weight:
        tensors[WEIGHT_NAME] = weight if weight is not None else source_weight()
    if include_scale:
        tensors[SCALE_NAME] = scale if scale is not None else source_scale()
    save_file(tensors, model_dir / "model.safetensors")
    return model_dir


def read_region(region) -> bytes:
    with region.source_file.open("rb") as handle:
        handle.seek(region.offset)
        payload = handle.read(region.length)
    if len(payload) != region.length:
        raise AssertionError("failed to read source tensor region")
    return payload


def add_index(model_dir: Path, weight_map: dict[str, str]) -> Path:
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return index_path


def rewrite_manifest(artifact_dir: Path, mutate) -> dict:
    path = artifact_dir / "sq_manifest.json"
    manifest = SQ8.read_json(path)
    mutate(manifest)
    manifest.pop("integrity", None)
    manifest["integrity"] = {
        "content_sha256": SQ8.artifact_content_sha256(manifest),
    }
    SQ8.write_json(path, manifest)
    return manifest


def write_raw_safetensors(path: Path, header: dict, data: bytes) -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + data)


class Sq8CanonicalArtifactTests(unittest.TestCase):
    def test_synthetic_round_trip_is_byte_exact_and_reconstructs_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"

            manifest = SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
                copy_chunk_bytes=34,
            )
            verification = SQ8.verify_canonical_artifact(artifact_dir)
            inventory, _index = SQ8.collect_tensor_inventory(model_dir)
            entry = manifest["quantized_tensors"][0]
            artifact_weight = artifact_dir / entry["weight"]["file"]
            artifact_scale = artifact_dir / entry["scale"]["file"]

            self.assertTrue(verification["verified"])
            self.assertEqual(verification["selected_pair_count"], 1)
            self.assertEqual(artifact_weight.read_bytes(), read_region(inventory[WEIGHT_NAME]))
            self.assertEqual(artifact_scale.read_bytes(), read_region(inventory[SCALE_NAME]))
            self.assertEqual(entry["shape"], [259, 257])
            self.assertEqual(entry["scale"]["shape"], [3, 3])
            self.assertEqual(entry["scale"]["block_shape"], [128, 128])

            points = [
                (0, 0),
                (0, 128),
                (127, 256),
                (128, 0),
                (128, 128),
                (258, 256),
            ]
            reconstructed = SQ8.reconstruct_artifact_points_f32(
                artifact_dir,
                WEIGHT_NAME,
                points,
            )
            weight = source_weight().float()
            scale = source_scale().float()
            expected = [
                float(weight[row, col] * scale[row // 128, col // 128])
                for row, col in points
            ]
            self.assertEqual(
                [struct_bits(value) for value in reconstructed],
                [struct_bits(value) for value in expected],
            )

    def test_rebuild_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            first = root / "first"
            second = root / "second"

            SQ8.build_canonical_artifact(model_dir, first, tensor_names=[WEIGHT_NAME])
            SQ8.build_canonical_artifact(model_dir, second, tensor_names=[WEIGHT_NAME])

            self.assertEqual(
                (first / "sq_manifest.json").read_bytes(),
                (second / "sq_manifest.json").read_bytes(),
            )
            first_manifest = SQ8.read_json(first / "sq_manifest.json")
            for entry in first_manifest["quantized_tensors"]:
                for kind in ("weight", "scale"):
                    relative = entry[kind]["file"]
                    self.assertEqual(
                        (first / relative).read_bytes(),
                        (second / relative).read_bytes(),
                    )

    def test_index_must_account_for_every_tensor_and_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            shard = model_dir / "model.safetensors"
            indexed_name = "model-00001-of-00001.safetensors"
            shard.rename(model_dir / indexed_name)
            complete_map = {
                WEIGHT_NAME: indexed_name,
                SCALE_NAME: indexed_name,
                "model.embed_tokens.weight": indexed_name,
            }
            index_path = add_index(model_dir, complete_map)

            inventory, _index = SQ8.collect_tensor_inventory(model_dir)
            self.assertEqual(set(inventory), set(complete_map))

            incomplete = dict(complete_map)
            del incomplete["model.embed_tokens.weight"]
            index_path.write_text(
                json.dumps({"metadata": {}, "weight_map": incomplete}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SQ8.ArtifactError, "index/inventory tensor mismatch"):
                SQ8.collect_tensor_inventory(model_dir)

            add_index(model_dir, complete_map)
            save_file(
                {"unexpected.weight": torch.ones((2, 2), dtype=torch.bfloat16)},
                model_dir / "model-00002-of-00002.safetensors",
            )
            with self.assertRaisesRegex(SQ8.ArtifactError, "shards absent from the index"):
                SQ8.collect_tensor_inventory(model_dir)

    def test_index_rejects_wrong_shard_and_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            first_name = "model-00001-of-00002.safetensors"
            second_name = "model-00002-of-00002.safetensors"
            (model_dir / "model.safetensors").rename(model_dir / first_name)
            save_file(
                {"other.weight": torch.ones((2, 2), dtype=torch.bfloat16)},
                model_dir / second_name,
            )
            complete_map = {
                WEIGHT_NAME: first_name,
                SCALE_NAME: first_name,
                "model.embed_tokens.weight": first_name,
                "other.weight": second_name,
            }
            index_path = add_index(model_dir, complete_map)

            wrong_map = dict(complete_map)
            wrong_map[WEIGHT_NAME] = second_name
            index_path.write_text(
                json.dumps({"metadata": {}, "weight_map": wrong_map}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SQ8.ArtifactError, "wrong shard"):
                SQ8.collect_tensor_inventory(model_dir)

            unsafe_map = dict(complete_map)
            unsafe_map[WEIGHT_NAME] = "../outside.safetensors"
            index_path.write_text(
                json.dumps({"metadata": {}, "weight_map": unsafe_map}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SQ8.ArtifactError, "must be a basename"):
                SQ8.collect_tensor_inventory(model_dir)

    def test_safetensors_regions_must_be_contiguous_and_non_overlapping(self) -> None:
        cases = {
            "overlap": (
                {
                    WEIGHT_NAME: {
                        "dtype": "F8_E4M3",
                        "shape": [1, 2],
                        "data_offsets": [0, 2],
                    },
                    SCALE_NAME: {
                        "dtype": "BF16",
                        "shape": [1, 1],
                        "data_offsets": [0, 2],
                    },
                },
                b"\x00\x00",
                "overlap",
            ),
            "gap": (
                {
                    WEIGHT_NAME: {
                        "dtype": "F8_E4M3",
                        "shape": [1, 2],
                        "data_offsets": [0, 2],
                    },
                    SCALE_NAME: {
                        "dtype": "BF16",
                        "shape": [1, 1],
                        "data_offsets": [4, 6],
                    },
                },
                b"\x00" * 6,
                "gap",
            ),
            "trailing": (
                {
                    WEIGHT_NAME: {
                        "dtype": "F8_E4M3",
                        "shape": [1, 2],
                        "data_offsets": [0, 2],
                    },
                    SCALE_NAME: {
                        "dtype": "BF16",
                        "shape": [1, 1],
                        "data_offsets": [2, 4],
                    },
                },
                b"\x00" * 6,
                "complete data buffer",
            ),
        }
        for label, (header, data, expected_error) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "invalid.safetensors"
                write_raw_safetensors(path, header, data)
                with self.assertRaisesRegex(SQ8.ArtifactError, expected_error):
                    SQ8.parse_safetensors_header(path)

    def test_json_booleans_are_not_accepted_as_integer_shape_or_offsets(self) -> None:
        headers = [
            {
                WEIGHT_NAME: {
                    "dtype": "F8_E4M3",
                    "shape": [True, 2],
                    "data_offsets": [0, 2],
                }
            },
            {
                WEIGHT_NAME: {
                    "dtype": "F8_E4M3",
                    "shape": [1, 2],
                    "data_offsets": [False, 2],
                }
            },
        ]
        for header in headers:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "invalid.safetensors"
                write_raw_safetensors(path, header, b"\x00\x00")
                with self.assertRaises(SQ8.ArtifactError):
                    SQ8.parse_safetensors_header(path)

    def test_missing_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root, include_scale=False)
            with self.assertRaisesRegex(SQ8.ArtifactError, "missing scale tensor"):
                SQ8.build_canonical_artifact(model_dir, root / "artifact")

    def test_orphan_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root, include_weight=False)
            with self.assertRaisesRegex(SQ8.ArtifactError, "orphan scale tensors"):
                SQ8.build_canonical_artifact(model_dir, root / "artifact")

    def test_source_without_fp8_pairs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root, include_weight=False, include_scale=False)
            with self.assertRaisesRegex(SQ8.ArtifactError, "contains no complete"):
                SQ8.build_canonical_artifact(model_dir, root / "artifact")

    def test_wrong_scale_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wrong = torch.ones((3, 2), dtype=torch.bfloat16)
            model_dir = write_model(root, scale=wrong)
            with self.assertRaisesRegex(SQ8.ArtifactError, "scale shape mismatch"):
                SQ8.build_canonical_artifact(model_dir, root / "artifact")

    def test_wrong_scale_dtype_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root, scale=source_scale().float())
            with self.assertRaisesRegex(SQ8.ArtifactError, "must use BF16"):
                SQ8.build_canonical_artifact(model_dir, root / "artifact")

    def test_nonpositive_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = source_scale()
            scale[1, 1] = 0
            model_dir = write_model(root, scale=scale)
            with self.assertRaisesRegex(SQ8.ArtifactError, "non-positive"):
                SQ8.build_canonical_artifact(model_dir, root / "artifact", copy_chunk_bytes=10)

    def test_payload_tamper_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"
            manifest = SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
            )
            weight_path = artifact_dir / manifest["quantized_tensors"][0]["weight"]["file"]
            with weight_path.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 1]))

            with self.assertRaisesRegex(SQ8.ArtifactError, "SHA-256 mismatch"):
                SQ8.verify_canonical_artifact(artifact_dir)

    def test_manifest_contract_mutations_are_rejected_after_rehash(self) -> None:
        mutations = {
            "scope": (
                lambda manifest: manifest["coverage"].update(scope="garbage"),
                "coverage.scope",
            ),
            "scale_name": (
                lambda manifest: manifest["quantized_tensors"][0]["scale"].update(
                    name="unrelated.scale"
                ),
                "scale name mismatch",
            ),
            "index_pair": (
                lambda manifest: manifest["source"].update(
                    index_file=None,
                    index_sha256="c" * 64,
                ),
                "must both be present or absent",
            ),
            "block_shape": (
                lambda manifest: manifest["source"]["quantization"].update(
                    weight_block_shape=[64, 128]
                ),
                "weight_block_shape",
            ),
            "passthrough_elements": (
                lambda manifest: manifest["passthrough_tensors"][0].update(elements=999),
                "passthrough tensor element count mismatch",
            ),
            "boolean_elements": (
                lambda manifest: manifest["quantized_tensors"][0].update(elements=True),
                "weight element count mismatch",
            ),
        }
        for label, (mutate, expected_error) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                model_dir = write_model(root)
                artifact_dir = root / "artifact"
                SQ8.build_canonical_artifact(
                    model_dir,
                    artifact_dir,
                    tensor_names=[WEIGHT_NAME],
                )
                rewrite_manifest(artifact_dir, mutate)

                with self.assertRaisesRegex(SQ8.ArtifactError, expected_error):
                    SQ8.verify_canonical_artifact(artifact_dir)

    def test_shape_derived_payload_length_cannot_be_redeclared(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"
            manifest = SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
            )
            weight = manifest["quantized_tensors"][0]["weight"]
            weight_path = artifact_dir / weight["file"]
            weight_path.write_bytes(weight_path.read_bytes()[:-1])

            def redeclare_payload(mutated: dict) -> None:
                entry = mutated["quantized_tensors"][0]
                entry["weight"]["bytes"] -= 1
                entry["weight"]["sha256"] = SQ8.sha256_file(weight_path)
                mutated["storage"]["weight_payload_bytes"] -= 1
                mutated["storage"]["total_payload_bytes"] -= 1

            rewrite_manifest(artifact_dir, redeclare_payload)

            with self.assertRaisesRegex(SQ8.ArtifactError, "weight payload byte count mismatch"):
                SQ8.verify_canonical_artifact(artifact_dir)

    def test_failed_overwrite_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = source_scale()
            scale[2, 2] = 0
            model_dir = write_model(root, scale=scale)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            marker = artifact_dir / "keep.txt"
            marker.write_text("existing", encoding="utf-8")

            with self.assertRaises(SQ8.ArtifactError):
                SQ8.build_canonical_artifact(model_dir, artifact_dir, overwrite=True)

            self.assertEqual(marker.read_text(encoding="utf-8"), "existing")

    def test_source_and_output_paths_must_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)

            for output_dir in (model_dir, model_dir / "artifact", root):
                with self.subTest(output_dir=output_dir):
                    with self.assertRaisesRegex(SQ8.ArtifactError, "must not be equal or contain"):
                        SQ8.build_canonical_artifact(
                            model_dir,
                            output_dir,
                            overwrite=True,
                        )
            self.assertTrue((model_dir / "config.json").is_file())
            self.assertTrue((model_dir / "model.safetensors").is_file())

    def test_overwrite_rejects_arbitrary_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            output_dir = root / "not-an-artifact"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("unrelated", encoding="utf-8")

            with self.assertRaisesRegex(SQ8.ArtifactError, "only accepts an existing verified"):
                SQ8.build_canonical_artifact(model_dir, output_dir, overwrite=True)

            self.assertEqual(marker.read_text(encoding="utf-8"), "unrelated")

    def test_successful_overwrite_atomically_replaces_verified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"
            first = SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
            )

            changed_scale = source_scale()
            changed_scale[0, 0] *= 2
            save_file(
                {
                    "model.embed_tokens.weight": torch.arange(
                        16, dtype=torch.bfloat16
                    ).reshape(4, 4),
                    WEIGHT_NAME: source_weight(),
                    SCALE_NAME: changed_scale,
                },
                model_dir / "model.safetensors",
            )
            second = SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
                overwrite=True,
            )

            self.assertNotEqual(
                first["integrity"]["content_sha256"],
                second["integrity"]["content_sha256"],
            )
            self.assertTrue(SQ8.verify_canonical_artifact(artifact_dir)["verified"])
            self.assertEqual(
                list(root.glob(".artifact.tmp.*")),
                [],
            )

    def test_initial_promotion_race_preserves_appeared_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"
            marker = artifact_dir / "unrelated-marker.txt"
            original_noreplace = SQ8._rename_noreplace
            appeared_identity = None
            injected = False

            def create_empty_output_then_promote(left: Path, right: Path) -> None:
                nonlocal appeared_identity, injected
                if injected:
                    original_noreplace(left, right)
                    return
                injected = True
                artifact_dir.mkdir()
                metadata = artifact_dir.stat()
                appeared_identity = (metadata.st_dev, metadata.st_ino)
                try:
                    original_noreplace(left, right)
                finally:
                    marker.write_text("must survive", encoding="utf-8")

            with mock.patch.object(
                SQ8,
                "_rename_noreplace",
                side_effect=create_empty_output_then_promote,
            ):
                with self.assertRaisesRegex(
                    SQ8.ArtifactError,
                    "atomic artifact initial promotion failed: File exists",
                ):
                    SQ8.build_canonical_artifact(
                        model_dir,
                        artifact_dir,
                        tensor_names=[WEIGHT_NAME],
                    )

            metadata = artifact_dir.stat()
            self.assertEqual((metadata.st_dev, metadata.st_ino), appeared_identity)
            self.assertEqual(marker.read_text(encoding="utf-8"), "must survive")
            self.assertEqual(list(root.glob(".artifact.tmp.*")), [])

    def test_overwrite_race_never_deletes_replacement_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"
            SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
            )
            verified_backup = root / "verified-backup"
            original_exchange = SQ8._rename_exchange
            swapped = False

            def replace_output_then_exchange(left: Path, right: Path) -> None:
                nonlocal swapped
                if not swapped:
                    artifact_dir.rename(verified_backup)
                    artifact_dir.mkdir()
                    (artifact_dir / "unrelated-marker.txt").write_text(
                        "must survive",
                        encoding="utf-8",
                    )
                    swapped = True
                original_exchange(left, right)

            with mock.patch.object(
                SQ8,
                "_rename_exchange",
                side_effect=replace_output_then_exchange,
            ):
                with self.assertRaisesRegex(
                    SQ8.ArtifactError,
                    "changed during atomic overwrite; promotion was rolled back",
                ):
                    SQ8.build_canonical_artifact(
                        model_dir,
                        artifact_dir,
                        tensor_names=[WEIGHT_NAME],
                        overwrite=True,
                    )

            self.assertEqual(
                (artifact_dir / "unrelated-marker.txt").read_text(encoding="utf-8"),
                "must survive",
            )
            self.assertTrue(SQ8.verify_canonical_artifact(verified_backup)["verified"])
            self.assertEqual(list(root.glob(".artifact.tmp.*")), [])

    def test_overwrite_race_rolls_back_file_and_symlink_entries(self) -> None:
        for entry_kind in ("file", "symlink"):
            with self.subTest(entry_kind=entry_kind), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                model_dir = write_model(root)
                artifact_dir = root / "artifact"
                SQ8.build_canonical_artifact(
                    model_dir,
                    artifact_dir,
                    tensor_names=[WEIGHT_NAME],
                )
                verified_backup = root / "verified-backup"
                symlink_target = root / "unrelated-target.txt"
                original_exchange = SQ8._rename_exchange
                replacement_identity = None
                swapped = False

                def replace_output_then_exchange(left: Path, right: Path) -> None:
                    nonlocal replacement_identity, swapped
                    if not swapped:
                        artifact_dir.rename(verified_backup)
                        if entry_kind == "file":
                            artifact_dir.write_text("must survive", encoding="utf-8")
                        else:
                            symlink_target.write_text("must survive", encoding="utf-8")
                            artifact_dir.symlink_to(symlink_target)
                        replacement_identity = SQ8._entry_identity(
                            artifact_dir,
                            "test replacement output entry",
                        )
                        swapped = True
                    original_exchange(left, right)

                with mock.patch.object(
                    SQ8,
                    "_rename_exchange",
                    side_effect=replace_output_then_exchange,
                ):
                    with self.assertRaisesRegex(
                        SQ8.ArtifactError,
                        "changed during atomic overwrite; promotion was rolled back",
                    ):
                        SQ8.build_canonical_artifact(
                            model_dir,
                            artifact_dir,
                            tensor_names=[WEIGHT_NAME],
                            overwrite=True,
                        )

                self.assertEqual(
                    SQ8._entry_identity(artifact_dir, "restored test output entry"),
                    replacement_identity,
                )
                self.assertEqual(artifact_dir.read_text(encoding="utf-8"), "must survive")
                self.assertEqual(artifact_dir.is_symlink(), entry_kind == "symlink")
                self.assertTrue(SQ8.verify_canonical_artifact(verified_backup)["verified"])
                self.assertEqual(list(root.glob(".artifact.tmp.*")), [])

    def test_exchanged_cleanup_race_preserves_replacement_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"
            SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
            )
            moved_verified_artifact = root / "moved-verified-artifact"
            original_noreplace = SQ8._rename_noreplace
            replacement_path = None
            replacement_identity = None
            swapped = False

            def replace_before_cleanup(left: Path, right: Path) -> None:
                nonlocal replacement_identity, replacement_path, swapped
                if not swapped and right.name == "owned-entry":
                    replacement_path = left
                    left.rename(moved_verified_artifact)
                    left.mkdir()
                    (left / "unrelated-marker.txt").write_text(
                        "must survive",
                        encoding="utf-8",
                    )
                    replacement_identity = SQ8._directory_identity(
                        left,
                        "test cleanup replacement",
                    )
                    swapped = True
                original_noreplace(left, right)

            with mock.patch.object(
                SQ8,
                "_rename_noreplace",
                side_effect=replace_before_cleanup,
            ):
                with self.assertWarnsRegex(RuntimeWarning, "changed identity before cleanup"):
                    SQ8.build_canonical_artifact(
                        model_dir,
                        artifact_dir,
                        tensor_names=[WEIGHT_NAME],
                        overwrite=True,
                    )

            assert replacement_path is not None
            self.assertEqual(
                SQ8._directory_identity(replacement_path, "restored cleanup replacement"),
                replacement_identity,
            )
            self.assertEqual(
                (replacement_path / "unrelated-marker.txt").read_text(encoding="utf-8"),
                "must survive",
            )
            self.assertTrue(SQ8.verify_canonical_artifact(artifact_dir)["verified"])
            self.assertTrue(SQ8.verify_canonical_artifact(moved_verified_artifact)["verified"])
            self.assertFalse(
                any(".cleanup." in path.name for path in root.iterdir()),
            )

    def test_failed_build_cleanup_race_preserves_replacement_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scale = source_scale()
            scale[1, 1] = 0
            model_dir = write_model(root, scale=scale)
            artifact_dir = root / "artifact"
            moved_owned_temp = root / "moved-owned-temp"
            original_noreplace = SQ8._rename_noreplace
            replacement_path = None
            replacement_identity = None
            swapped = False

            def replace_before_cleanup(left: Path, right: Path) -> None:
                nonlocal replacement_identity, replacement_path, swapped
                if not swapped and right.name == "owned-entry":
                    replacement_path = left
                    left.rename(moved_owned_temp)
                    left.mkdir()
                    (left / "unrelated-marker.txt").write_text(
                        "must survive",
                        encoding="utf-8",
                    )
                    replacement_identity = SQ8._directory_identity(
                        left,
                        "test failed-build cleanup replacement",
                    )
                    swapped = True
                original_noreplace(left, right)

            with mock.patch.object(
                SQ8,
                "_rename_noreplace",
                side_effect=replace_before_cleanup,
            ):
                with self.assertWarnsRegex(RuntimeWarning, "changed identity before cleanup"):
                    with self.assertRaisesRegex(SQ8.ArtifactError, "non-positive"):
                        SQ8.build_canonical_artifact(
                            model_dir,
                            artifact_dir,
                            tensor_names=[WEIGHT_NAME],
                        )

            assert replacement_path is not None
            self.assertEqual(
                SQ8._directory_identity(replacement_path, "restored failed-build replacement"),
                replacement_identity,
            )
            self.assertEqual(
                (replacement_path / "unrelated-marker.txt").read_text(encoding="utf-8"),
                "must survive",
            )
            self.assertFalse(artifact_dir.exists())
            self.assertTrue(moved_owned_temp.is_dir())
            self.assertFalse(
                any(".cleanup." in path.name for path in root.iterdir()),
            )


def struct_bits(value: float) -> int:
    import struct

    return struct.unpack("<I", struct.pack("<f", value))[0]


if __name__ == "__main__":
    unittest.main()
