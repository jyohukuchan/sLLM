import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "build-sq-fp8-w8a16-artifact.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_sq_fp8_w8a16_artifact", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildSqFp8ArtifactPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder_module()

    def write_policy(self, root: Path, schema_version: str = "sq-fp8-policy-v0.1") -> Path:
        policy = {
            "schema_version": schema_version,
            "candidate_id": "sq-fp8-w8a16-r9700-v0",
            "policy_id": "kup6_gate5_down5",
            "status": "six_layer_strict_top1_regression_subset_not_full_sq_policy",
            "scale": {
                "granularity": "row_block",
                "block_cols": 32,
                "dtype": "f32",
            },
            "fp8_selection": {
                "include_regex": (
                    r"^model\.language_model\.layers\."
                    r"((3|7|11|15|19|23)\.(self_attn\.k_proj|mlp\.up_proj)"
                    r"|(3|7|11|15|19)\.mlp\.(gate_proj|down_proj))\.weight$"
                ),
                "expected_fp8_tensor_count": 22,
            },
            "fallback_policy": [
                {
                    "family": "self_attn.q_proj",
                    "layers": "all_tested",
                    "reason": "strict_top1_risk_under_row_block32",
                }
            ],
            "prompt_bundle_result": {
                "strict_top1_pass_count": 3,
                "case_count": 3,
                "promoted_full_sq_policy": False,
            },
        }
        path = root / "policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        return path

    def test_policy_fills_builder_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self.write_policy(Path(tmpdir))
            args = argparse.Namespace(
                policy_json=policy_path,
                candidate_id=None,
                include_regex=[],
                scale_granularity=None,
                scale_block_cols=None,
            )

            policy = self.builder.resolve_policy_args(args)

        self.assertIsNotNone(policy)
        self.assertEqual(args.candidate_id, "sq-fp8-w8a16-r9700-v0")
        self.assertEqual(args.scale_granularity, "row_block")
        self.assertEqual(args.scale_block_cols, 32)
        self.assertEqual(len(args.include_regex), 1)
        self.assertIn("kup6_gate5_down5", self.builder.policy_manifest_entry(policy_path, policy)["policy_id"])

    def test_policy_regex_selects_expected_tensor_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self.write_policy(Path(tmpdir))
            args = argparse.Namespace(
                policy_json=policy_path,
                candidate_id=None,
                include_regex=[],
                scale_granularity=None,
                scale_block_cols=None,
            )
            self.builder.resolve_policy_args(args)

        tensors = []
        source_file = Path("model-00001-of-00001.safetensors")
        for layer in [3, 7, 11, 15, 19, 23]:
            for suffix in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]:
                tensors.append(
                    self.builder.SourceTensor(
                        name=f"model.language_model.layers.{layer}.{suffix}.weight",
                        source_file=source_file,
                        dtype="BF16",
                        shape=[8, 8],
                    )
                )
        tensors.append(
            self.builder.SourceTensor(
                name="lm_head.weight",
                source_file=source_file,
                dtype="BF16",
                shape=[8, 8],
            )
        )

        patterns = self.builder.compile_patterns(args.include_regex, "include-regex")
        selected, passthrough = self.builder.selected_tensors(tensors, Path("artifact"), patterns, [], 0)
        selected_names = {item.source.name for item in selected}

        self.assertEqual(len(selected), 22)
        self.assertEqual(len(passthrough), len(tensors) - 22)
        self.assertIn("model.language_model.layers.23.self_attn.k_proj.weight", selected_names)
        self.assertIn("model.language_model.layers.19.mlp.down_proj.weight", selected_names)
        self.assertNotIn("model.language_model.layers.23.mlp.down_proj.weight", selected_names)
        self.assertNotIn("model.language_model.layers.3.self_attn.q_proj.weight", selected_names)
        self.assertNotIn("lm_head.weight", selected_names)

    def test_policy_scale_override_resolves_per_tensor_scale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy_path = self.write_policy(root)
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
            payload["scale"]["overrides"] = [
                {
                    "id": "k-rowblock16",
                    "include_regex": r"\.self_attn\.k_proj\.weight$",
                    "granularity": "row_block",
                    "block_cols": 16,
                }
            ]
            policy_path.write_text(json.dumps(payload), encoding="utf-8")
            args = argparse.Namespace(
                policy_json=policy_path,
                candidate_id=None,
                include_regex=[],
                scale_granularity=None,
                scale_block_cols=None,
            )
            self.builder.resolve_policy_args(args)

        tensors = [
            self.builder.SourceTensor(
                name="model.language_model.layers.3.self_attn.k_proj.weight",
                source_file=Path("model.safetensors"),
                dtype="BF16",
                shape=[8, 32],
            ),
            self.builder.SourceTensor(
                name="model.language_model.layers.3.mlp.up_proj.weight",
                source_file=Path("model.safetensors"),
                dtype="BF16",
                shape=[8, 32],
            ),
        ]
        patterns = self.builder.compile_patterns(args.include_regex, "include-regex")
        selected, _ = self.builder.selected_tensors(
            tensors,
            Path("artifact"),
            patterns,
            [],
            0,
            self.builder.ScaleConfig(args.scale_granularity, args.scale_block_cols),
            args.scale_overrides,
        )
        by_name = {item.source.name: item for item in selected}
        k_tensor = by_name["model.language_model.layers.3.self_attn.k_proj.weight"]
        up_tensor = by_name["model.language_model.layers.3.mlp.up_proj.weight"]

        self.assertEqual(k_tensor.scale.granularity, "row_block")
        self.assertEqual(k_tensor.scale.block_cols, 16)
        self.assertEqual(k_tensor.scale.override_id, "k-rowblock16")
        self.assertEqual(up_tensor.scale.granularity, "row_block")
        self.assertEqual(up_tensor.scale.block_cols, 32)
        self.assertIsNone(up_tensor.scale.override_id)

        k_entry = self.builder.tensor_manifest_entry(k_tensor, Path("artifact"), True)
        up_entry = self.builder.tensor_manifest_entry(up_tensor, Path("artifact"), True)
        self.assertEqual(k_entry["scale_elements"], 16)
        self.assertEqual(k_entry["scale_block_cols"], 16)
        self.assertEqual(k_entry["scale_override_id"], "k-rowblock16")
        self.assertEqual(up_entry["scale_elements"], 8)
        self.assertEqual(up_entry["scale_block_cols"], 32)
        self.assertNotIn("scale_override_id", up_entry)

    def test_policy_schema_is_checked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self.write_policy(Path(tmpdir), schema_version="wrong")
            args = argparse.Namespace(
                policy_json=policy_path,
                candidate_id=None,
                include_regex=[],
                scale_granularity=None,
                scale_block_cols=None,
            )

            with self.assertRaises(SystemExit):
                self.builder.resolve_policy_args(args)

    def test_policy_scale_override_block_cols_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self.write_policy(Path(tmpdir))
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
            payload["scale"]["overrides"] = [
                {
                    "include_regex": r"\.self_attn\.k_proj\.weight$",
                    "granularity": "row_block",
                    "block_cols": 0,
                }
            ]
            policy_path.write_text(json.dumps(payload), encoding="utf-8")
            args = argparse.Namespace(
                policy_json=policy_path,
                candidate_id=None,
                include_regex=[],
                scale_granularity=None,
                scale_block_cols=None,
            )

            with self.assertRaises(SystemExit):
                self.builder.resolve_policy_args(args)

    def test_policy_scale_block_cols_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = self.write_policy(Path(tmpdir))
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
            payload["scale"]["block_cols"] = 0
            policy_path.write_text(json.dumps(payload), encoding="utf-8")
            args = argparse.Namespace(
                policy_json=policy_path,
                candidate_id=None,
                include_regex=[],
                scale_granularity=None,
                scale_block_cols=None,
            )

            with self.assertRaises(SystemExit):
                self.builder.resolve_policy_args(args)


if __name__ == "__main__":
    unittest.main()
