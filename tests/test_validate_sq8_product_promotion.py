from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "validate-sq8-product-promotion.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROMOTION = load_module("test_validate_sq8_product_promotion_tool", TOOL_PATH)


class ProductTreeTests(unittest.TestCase):
    def test_read_only_tree_accepts_a_real_nonwritable_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            root.mkdir(mode=0o755)
            payload = root / "payload"
            payload.write_bytes(b"payload")
            os.chmod(payload, 0o444)
            os.chmod(root, 0o555)

            PROMOTION.validate_read_only_tree(root, 1)

    def test_read_only_tree_rejects_root_write_bits_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "artifact"
            root.mkdir(mode=0o755)
            with self.assertRaisesRegex(
                PROMOTION.ValidationError, "root has write bits"
            ):
                PROMOTION.validate_read_only_tree(root, 0)

            os.chmod(root, 0o555)
            link = parent / "artifact-link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(
                PROMOTION.ValidationError, "root is not a real directory"
            ):
                PROMOTION.validate_read_only_tree(link, 0)


if __name__ == "__main__":
    unittest.main()
