from __future__ import annotations

import copy
import unittest

from ci.tools.run_openai_profile_differential import validate_common


def snapshot() -> dict[str, object]:
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    return {
        "non_stream": {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        },
        "stream": [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "qwen",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "qwen",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            },
        ],
        "terminal": "[DONE]",
    }


class OpenAIProfileDifferentialTests(unittest.TestCase):
    def test_common_profile_shape_passes(self) -> None:
        self.assertEqual(validate_common("sllm", snapshot())["result"], "PASS")

    def test_unstable_id_missing_role_and_done_fail_closed(self) -> None:
        mutations = (
            lambda value: value["stream"][1].update(id="different"),
            lambda value: value["stream"][0]["choices"][0].update(delta={}),
            lambda value: value.update(terminal=None),
        )
        for mutate in mutations:
            value = copy.deepcopy(snapshot())
            mutate(value)
            with self.assertRaises(ValueError):
                validate_common("sllm", value)


if __name__ == "__main__":
    unittest.main()
