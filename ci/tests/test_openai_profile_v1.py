from __future__ import annotations

import copy
import json
import unittest

from ci.tools.validate_openai_profile_v1 import FIXTURE, validate


class OpenAIProfileV1FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_canonical_fixture_passes(self) -> None:
        validate(self.fixture)

    def test_pin_terminal_and_negative_matrix_fail_closed(self) -> None:
        for mutate in (
            lambda value: value["official_openapi"].update(commit="main"),
            lambda value: value["positive"]["stream"].update(terminal="done"),
            lambda value: value["negative"].pop(),
        ):
            candidate = copy.deepcopy(self.fixture)
            mutate(candidate)
            with self.assertRaises(ValueError):
                validate(candidate)


if __name__ == "__main__":
    unittest.main()
