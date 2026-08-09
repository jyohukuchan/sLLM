from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, sha256_file  # noqa: E402
import validate_rmsnorm_g2_contracts as contracts  # noqa: E402


def slice_record() -> dict[str, object]:
    return {
        "schema_version": "rmsnorm-g2-model-slice-v1",
        "slice_id": "qwen3.5-4b-layer0-input-layernorm-v1",
        "source": {"model_lock_path": contracts.MODEL_LOCK_PATH, "model_lock_sha256": sha256_file(ROOT / contracts.MODEL_LOCK_PATH), "model_lock_fingerprint": contracts.MODEL_LOCK_FINGERPRINT, "resolved_revision": contracts.RESOLVED_REVISION},
        "tensor": {"name": contracts.TENSOR_NAME, "source_shard": contracts.SOURCE_SHARD, "dtype": "BF16", "shape": [2560], "header_length_field_bytes": 8, "header_length_bytes": contracts.HEADER_LENGTH, "data_buffer_start": contracts.DATA_BUFFER_START, "data_offset_basis": "data-buffer-relative", "data_offsets": list(contracts.DATA_OFFSETS), "absolute_byte_range": list(contracts.ABSOLUTE_RANGE), "byte_size": contracts.BYTE_SIZE, "epsilon": "1e-6", "scale_mode": "offset-one"},
        "recipe": {"version": "rmsnorm-g2-slice-recipe-v1", "extractor": "sllm-g2-synthetic-safetensors-extractor", "repository": "sLLM", "commit": "a" * 40, "script_path": "ci/tools/extract_rmsnorm_g2_slice.py", "script_sha256": sha256_file(ROOT / "ci/tools/extract_rmsnorm_g2_slice.py"), "arguments": ["--fixture", "synthetic-fixture"], "synthetic_fixture_only": True},
        "output": {"size_bytes": contracts.BYTE_SIZE, "sha256": "0" * 64},
        "storage": {"raw_slice_stored": False, "raw_slice_uploaded": False, "raw_model_stored": False, "raw_model_uploaded": False, "path_recorded": False},
    }


def fixture(path: Path, *, marker: str = contracts.SYNTHETIC_MARKER, trailing: int = 0) -> None:
    header_object = {"__metadata__": {"sllm_fixture": marker}, contracts.TENSOR_NAME: {"dtype": "BF16", "shape": [2560], "data_offsets": list(contracts.DATA_OFFSETS)}}
    raw = json.dumps(header_object, separators=(",", ":")).encode()
    if len(raw) > contracts.HEADER_LENGTH:
        raise AssertionError("fixture header unexpectedly exceeds lock")
    header = raw[:-1] + b" " * (contracts.HEADER_LENGTH - len(raw)) + b"}"
    data = bytearray(contracts.ABSOLUTE_RANGE[1] + trailing)
    data[0:8] = struct.pack("<Q", contracts.HEADER_LENGTH)
    data[8 : 8 + len(header)] = header
    data[contracts.ABSOLUTE_RANGE[0] : contracts.ABSOLUTE_RANGE[1]] = bytes(range(256)) * 20
    path.write_bytes(data)


class G2SliceTests(unittest.TestCase):
    def test_full_file_verification_does_not_materialize_the_model_file(self) -> None:
        contents = bytes(range(256)) * 32
        with tempfile.TemporaryDirectory(prefix="sllm-g2-full-file-") as directory:
            path = Path(directory) / "shard.safetensors"
            path.write_bytes(contents)
            descriptor = contracts.os.open(path, contracts.os.O_RDONLY)
            try:
                with patch.object(contracts.os, "pread", side_effect=AssertionError("full model pread")):
                    payload = contracts._read_verified_fd_slice(
                        descriptor,
                        file_size=len(contents),
                        absolute_start=0,
                        absolute_end=len(contents),
                        expected_sha256=hashlib.sha256(contents).hexdigest(),
                        label="test full-file verification",
                        capture_payload=False,
                    )
            finally:
                contracts.os.close(descriptor)
            self.assertEqual(payload, b"")

    def test_locked_recipe_and_synthetic_extraction_only_hashes_payload(self) -> None:
        record = slice_record()
        with tempfile.TemporaryDirectory(prefix="sllm-g2-synthetic-") as directory:
            path = Path(directory) / contracts.SOURCE_SHARD
            fixture(path)
            result = contracts.extract_synthetic_slice(path, record)
            self.assertEqual(result["output"]["size_bytes"], 5120)
            self.assertEqual(result["output"]["sha256"], hashlib.sha256(path.read_bytes()[contracts.ABSOLUTE_RANGE[0] : contracts.ABSOLUTE_RANGE[1]]).hexdigest())
            self.assertFalse(result["storage"]["raw_slice_stored"])

    def test_slice_rejects_header_marker_trailing_and_offset_overflow(self) -> None:
        record = slice_record()
        with tempfile.TemporaryDirectory(prefix="sllm-g2-synthetic-") as directory:
            path = Path(directory) / contracts.SOURCE_SHARD
            fixture(path, marker="not-real")
            with self.assertRaises(ContractError):
                contracts.extract_synthetic_slice(path, record)
            fixture(path, trailing=1)
            with self.assertRaises(ContractError):
                contracts.extract_synthetic_slice(path, record)
        changed = copy.deepcopy(record)
        changed["tensor"]["data_offsets"] = [0, (1 << 64)]
        with self.assertRaises(ContractError):
            contracts.validate_slice_record(changed)

    def test_slice_rejects_symlink_raw_substitution_and_truncation(self) -> None:
        record = slice_record()
        with tempfile.TemporaryDirectory(prefix="sllm-g2-synthetic-") as directory:
            root = Path(directory)
            fixture_path = root / contracts.SOURCE_SHARD
            fixture(fixture_path)
            link = root / "linked.safetensors"
            link.symlink_to(fixture_path)
            with self.assertRaises(ContractError):
                contracts.extract_synthetic_slice(link, record)
            raw = root / "raw-slice.bin"
            raw.write_bytes(b"x" * contracts.BYTE_SIZE)
            with self.assertRaises(ContractError):
                contracts.extract_synthetic_slice(raw, record)
            truncated = root / "truncated.safetensors"
            truncated.write_bytes(fixture_path.read_bytes()[:-1])
            with self.assertRaises(ContractError):
                contracts.extract_synthetic_slice(truncated, record)

    def test_slice_rejects_file_replacement_during_extraction(self) -> None:
        record = slice_record()
        with tempfile.TemporaryDirectory(prefix="sllm-g2-synthetic-") as directory:
            path = Path(directory) / contracts.SOURCE_SHARD
            fixture(path)
            original_fstat = contracts.os.fstat
            calls = 0

            def changed_fstat(fd: int) -> object:
                nonlocal calls
                calls += 1
                observed = original_fstat(fd)
                if calls == 2:
                    return SimpleNamespace(
                        st_dev=observed.st_dev,
                        st_ino=observed.st_ino,
                        st_size=observed.st_size,
                        st_mtime_ns=observed.st_mtime_ns + 1,
                    )
                return observed

            with patch.object(contracts.os, "fstat", side_effect=changed_fstat):
                with self.assertRaises(ContractError):
                    contracts.extract_synthetic_slice(path, record)

    def test_slice_extractor_rejects_model_cache_path_even_with_marker(self) -> None:
        record = slice_record()
        with tempfile.TemporaryDirectory(prefix="sllm-g2-synthetic-") as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            path = cache / contracts.SOURCE_SHARD
            fixture(path)
            with self.assertRaises(ContractError):
                contracts.extract_synthetic_slice(path, record)

    def test_slice_rejects_dtype_shape_epsilon_and_range_drift(self) -> None:
        record = slice_record()
        for field, value in (("dtype", "F16"), ("shape", [255]), ("epsilon", "1e-5"), ("scale_mode", "plain"), ("absolute_byte_range", [94431, 99551])):
            changed = copy.deepcopy(record)
            changed["tensor"][field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                contracts.validate_slice_record(changed)


if __name__ == "__main__":
    unittest.main()
