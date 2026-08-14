import json
import math
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]


class Phase8Bf16NumericsTest(unittest.TestCase):
    def test_frozen_contract_matches_closed_schema(self) -> None:
        document = json.loads((ROOT / "ci/matrix/phase8-bf16-numerics-v1.json").read_text())
        schema = json.loads((ROOT / "ci/schema/phase8-bf16-numerics-v1.schema.json").read_text())
        jsonschema.Draft202012Validator(schema).validate(document)

    def test_gamma_bound_is_defined_for_all_registered_reductions(self) -> None:
        document = json.loads((ROOT / "ci/matrix/phase8-bf16-numerics-v1.json").read_text())
        unit_roundoff = document["matmul"]["fp32_unit_roundoff"]
        for shape in document["matmul"]["required_shapes"]:
            product = shape["k"] * unit_roundoff
            self.assertGreater(product, 0.0)
            self.assertLess(product, 1.0)
            self.assertTrue(math.isfinite(product / (1.0 - product)))

    def test_boundaries_cover_fast_path_edges(self) -> None:
        document = json.loads((ROOT / "ci/matrix/phase8-bf16-numerics-v1.json").read_text())
        shapes = {(row["m"], row["k"], row["n"]) for row in document["matmul"]["required_shapes"]}
        self.assertTrue({1, 3, 17, 37}.issubset({m for m, _, _ in shapes}))
        self.assertTrue(any(k % 16 for _, k, _ in shapes))
        self.assertTrue(any(n % 16 for _, _, n in shapes))


if __name__ == "__main__":
    unittest.main()
