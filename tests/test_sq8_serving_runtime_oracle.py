import hashlib
import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-runtime-oracle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_serving_runtime_oracle", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32_file(path: Path, values) -> str:
    payload = b"".join(struct.pack("<f", value) for value in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def u32_file(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<I", value))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_source_fixture(module, fixture: Path) -> None:
    payloads = []
    for path in sorted(fixture.rglob("*")):
        if path.is_file():
            payloads.append(
                {
                    "file": path.relative_to(fixture).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload_manifest = fixture / "payload-manifest.json"
    payload_manifest.write_text(
        json.dumps(
            {
                "schema_version": "ullm.sq8.serving_oracle_payload_manifest.v1",
                "files": payloads,
            }
        ),
        encoding="utf-8",
    )
    payload_manifest_sha256 = sha256_file(payload_manifest)
    metadata = fixture / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "ullm.sq8.serving_oracle.v1",
                "status": "captured_real_vllm",
                "source_model": {
                    "identity": {
                        "artifact_content_sha256": module.EXPECTED_ARTIFACT_SHA256,
                        "package_manifest_sha256": module.EXPECTED_PACKAGE_SHA256,
                    }
                },
                "source_fixture": {"fixture_set_id": module.EXPECTED_SOURCE_FIXTURE_SET_ID},
                "payload_manifest": {"sha256": payload_manifest_sha256},
            }
        ),
        encoding="utf-8",
    )
    sha256sums = fixture / "SHA256SUMS"
    lines = [
        f"{entry['sha256']}  {entry['file']}" for entry in payloads
    ] + [
        f"{sha256_file(metadata)}  metadata.json",
        f"{payload_manifest_sha256}  payload-manifest.json",
    ]
    sha256sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    module.EXPECTED_SOURCE_METADATA_SHA256 = sha256_file(metadata)
    module.EXPECTED_SOURCE_PAYLOAD_MANIFEST_SHA256 = payload_manifest_sha256
    module.EXPECTED_SOURCE_SHA256SUMS_SHA256 = sha256_file(sha256sums)


class Sq8ServingRuntimeOracleTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_compare_f32_files_recomputes_metrics_and_stable_topk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual.f32le"
            reference = root / "reference.f32le"
            f32_file(actual, [1.0, 3.0, 3.0, -2.0])
            f32_file(reference, [1.0, 3.0, 3.0, -2.0])
            self.module.TOP_K = 3
            metrics = self.module.compare_f32_files(
                actual,
                reference,
                elements=4,
                top_k=True,
                label="test",
            )
            self.assertEqual(metrics["relative_l2"], 0.0)
            self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
            self.assertEqual(
                [item["token_id"] for item in metrics["actual_top_10"]],
                [1, 2, 0],
            )
            self.assertTrue(metrics["top_1_exact"])

    def test_compare_f32_files_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual.f32le"
            reference = root / "reference.f32le"
            f32_file(actual, [1.0, float("nan")])
            f32_file(reference, [1.0, 2.0])
            with self.assertRaisesRegex(self.module.ValidationError, "non-finite"):
                self.module.compare_f32_files(
                    actual,
                    reference,
                    elements=2,
                    top_k=False,
                    label="test",
                )

    def test_load_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(self.module.ValidationError, "duplicate JSON key"):
                self.module.load_json(path)

    def test_validate_source_oracle_rejects_unbound_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            (root / "payload-manifest.json").write_text("{}", encoding="utf-8")
            (root / "SHA256SUMS").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(self.module.ValidationError, "identity differs"):
                self.module.validate_source_oracle(root)

    def test_validate_recorded_top1_rejects_modified_logit(self):
        metrics = {
            "actual_top_10": [
                {"token_id": 7, "logit": struct.unpack("<f", struct.pack("<f", 12.5))[0]}
            ]
        }
        with self.assertRaisesRegex(self.module.ValidationError, "differs from the raw logits"):
            self.module.validate_recorded_top1(7, -12345.0, metrics, "top1")

    def test_reject_same_file_rejects_hard_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.f32le"
            alias = root / "alias.f32le"
            source.write_bytes(b"payload")
            os.link(source, alias)
            with self.assertRaisesRegex(self.module.ValidationError, "aliases"):
                self.module.reject_same_file(alias, source, "capture")

    def test_validate_p7_path_equivalence_recomputes_direct_metrics(self):
        self.module.HIDDEN_SIZE = 4
        self.module.VOCAB_SIZE = 6
        self.module.TOP_K = 3
        self.module.PATH_MIN_TOP_10_OVERLAP = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p7_dir = root / "p7"
            p8_dir = root / "p8"
            hidden = [1.0, 2.0, 3.0, 4.0]
            logits = [0.0, 8.0, 2.0, 7.0, 1.0, -1.0]
            p7_hidden = p7_dir / "hidden.f32le"
            p7_logits = p7_dir / "logits.f32le"
            p8_hidden = p8_dir / "hidden.f32le"
            p8_logits = p8_dir / "logits.f32le"
            p7_hidden_hash = f32_file(p7_hidden, hidden)
            p7_logits_hash = f32_file(p7_logits, logits)
            p8_hidden_hash = f32_file(p8_hidden, hidden)
            p8_logits_hash = f32_file(p8_logits, logits)
            manifest_path = p7_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "ullm.sq8.p7_first_step_capture.v1",
                        "prompt_token_ids": list(range(1, 9)),
                        "step_index": 0,
                        "output_token_id": 1,
                        "output_logit": 8.0,
                        "final_hidden_file": "hidden.f32le",
                        "final_hidden_f32_le_sha256": p7_hidden_hash,
                        "logits_file": "logits.f32le",
                        "logits_f32_le_sha256": p7_logits_hash,
                    }
                ),
                encoding="utf-8",
            )
            result = self.module.validate_p7_path_equivalence(
                manifest_path,
                {
                    "generated_token_id": 1,
                    "capture": {
                        "final_hidden_file": str(p8_hidden),
                        "final_hidden_sha256": p8_hidden_hash,
                        "logits_file": str(p8_logits),
                        "logits_sha256": p8_logits_hash,
                    },
                },
                (root / "source-oracle").resolve(),
            )
            self.assertEqual(result["final_hidden"]["relative_l2"], 0.0)
            self.assertTrue(result["logits"]["top_1_exact"])

    def test_validate_results_recomputes_four_prompt_gates(self):
        self.module.PROMPT_LENGTHS = (1, 2, 3, 4)
        self.module.HIDDEN_SIZE = 4
        self.module.VOCAB_SIZE = 8
        self.module.TOP_K = 3
        self.module.MIN_TOP_10_OVERLAP = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            result_paths = []
            for result_index, lengths in enumerate(((1, 2), (3, 4))):
                requests = []
                for prompt_length in lengths:
                    prompt_id = f"raw-p{prompt_length:04d}"
                    reference_dir = fixture / "prompts" / prompt_id
                    top1 = prompt_length
                    hidden = [1.0, 2.0, 3.0, float(prompt_length)]
                    logits = [float(index) for index in range(8)]
                    logits[top1] = 20.0
                    f32_file(reference_dir / "final-hidden.f32le", hidden)
                    f32_file(reference_dir / "prefill-logits.f32le", logits)
                    u32_file(reference_dir / "greedy-g1.u32le", top1)

                    capture_dir = root / f"capture-{prompt_length}"
                    hidden_path = capture_dir / "hidden.f32le"
                    logits_path = capture_dir / "logits.f32le"
                    hidden_hash = f32_file(hidden_path, hidden)
                    logits_hash = f32_file(logits_path, logits)
                    requests.append(
                        {
                            "request_id": f"request-{prompt_length}",
                            "prompt_token_ids": list(range(1, prompt_length + 1)),
                            "max_new_tokens": 1,
                            "generated_token_ids": [top1],
                            "prompt_progress_events": prompt_length - 1,
                            "execution_units": prompt_length,
                            "terminal_reason": "length",
                            "release_outcome": "length",
                            "request_seconds": 1.0,
                            "reset_seconds": 0.01,
                            "oracle_capture": {
                                "position": prompt_length - 1,
                                "top1_token_id": top1,
                                "top1_logit": 20.0,
                                "final_hidden_file": str(hidden_path),
                                "final_hidden_f32_le_sha256": hidden_hash,
                                "logits_file": str(logits_path),
                                "logits_f32_le_sha256": logits_hash,
                            },
                        }
                    )
                result = {
                    "schema_version": self.module.SCHEMA_VERSION,
                    "passed": False,
                    "requests": requests,
                    "cancelled_request": None,
                    "load_seconds": 1.0,
                    "artifact_content_sha256": self.module.EXPECTED_ARTIFACT_SHA256,
                    "package_manifest_sha256": self.module.EXPECTED_PACKAGE_SHA256,
                    "device": {
                        "device_id": 0,
                        "backend": "hip",
                        "name": "AMD Radeon Graphics",
                        "gcn_arch_name": "gfx1201",
                        "compute_major": 12,
                        "compute_minor": 0,
                        "total_global_mem": 32 * 1024**3,
                    },
                    "kv_cache_bytes": self.module.KV_CACHE_BYTES,
                    "cache_blocks": self.module.CACHE_BLOCKS,
                    "context_tokens": self.module.CONTEXT_TOKENS,
                    "post_reset_status": "ready",
                    "post_reset_active": 0,
                    "post_reset_waiting": 0,
                    "post_reset_allocated_blocks": 0,
                    "post_reset_cache_lengths_all_zero": True,
                }
                result_path = root / f"result-{result_index}.json"
                result_path.write_text(json.dumps(result), encoding="utf-8")
                result_paths.append(result_path)

            finalize_source_fixture(self.module, fixture)
            validation = self.module.validate_results(result_paths, fixture)
            self.assertTrue(validation["passed"])
            self.assertEqual(
                validation["source_oracle"]["payload_manifest_sha256"],
                self.module.EXPECTED_SOURCE_PAYLOAD_MANIFEST_SHA256,
            )
            self.assertEqual(
                [prompt["prompt_tokens"] for prompt in validation["prompts"]],
                [1, 2, 3, 4],
            )
            self.assertTrue(
                all(prompt["logits"]["top_1_exact"] for prompt in validation["prompts"])
            )


if __name__ == "__main__":
    unittest.main()
