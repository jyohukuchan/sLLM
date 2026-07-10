import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-product-promotion.py"
PRODUCT = Path("/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1")


def load_validator():
    spec = importlib.util.spec_from_file_location("sq8_product_promotion_validator", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sq8ProductPromotionValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_validator()

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            path.write_text('{"key":1,"key":2}\n', encoding="ascii")
            with self.assertRaises(self.validator.ValidationError):
                self.validator.read_json(path)

    def test_nonfinite_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            path.write_text('{"key":NaN}\n', encoding="ascii")
            with self.assertRaises(self.validator.ValidationError):
                self.validator.read_json(path)

    def test_unsafe_payload_paths_are_rejected(self) -> None:
        for value in ("", "/absolute", "../escape", "dir/../escape", "./relative"):
            with self.subTest(value=value):
                with self.assertRaises(self.validator.ValidationError):
                    self.validator.safe_relative_path(value, "fixture")

    def test_exact_key_drift_is_rejected(self) -> None:
        with self.assertRaises(self.validator.ValidationError):
            self.validator.exact_keys({"a": 1, "extra": 2}, {"a"}, "fixture")

    @unittest.skipUnless(PRODUCT.is_dir(), "local promoted SQ8 product is absent")
    def test_real_product_metadata_passes(self) -> None:
        promotion = self.validator.validate_promotion(PRODUCT)
        artifact = self.validator.validate_artifact(PRODUCT / "artifact", full_payloads=False)
        package = self.validator.validate_package(PRODUCT / "package", full_payloads=False)
        self.assertEqual(promotion["schema_version"], self.validator.SCHEMA_VERSION)
        self.assertEqual(artifact["selected_pair_count"], 280)
        self.assertEqual(package["payload_count"], 163)
        self.assertFalse(artifact["payloads_hashed"])
        self.assertFalse(package["payloads_hashed"])

    def test_expected_contract_is_json_serializable(self) -> None:
        encoded = json.dumps(
            {
                "artifact": self.validator.EXPECTED_ARTIFACT,
                "package": self.validator.EXPECTED_PACKAGE,
            },
            sort_keys=True,
        )
        self.assertIn("2243acf1", encoded)
        self.assertIn("c2133dfe", encoded)


if __name__ == "__main__":
    unittest.main()
