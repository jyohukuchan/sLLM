from __future__ import annotations

import unittest
from unittest.mock import patch

from ci.tools.run_openai_a6_gpu import metric_observation, parse_sse, process_count


class OpenAIA6EvidenceRunnerTests(unittest.TestCase):
    def test_raw_sse_parser_accepts_profile_sequence(self) -> None:
        body = (
            b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
            b"data: [DONE]\n\n"
        )
        self.assertEqual(len(parse_sse(body)), 3)

    def test_raw_sse_parser_fails_closed_on_id_and_terminal(self) -> None:
        invalid = (
            b'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            b'data: {"id":"y","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        with self.assertRaises(RuntimeError):
            parse_sse(invalid)

    def test_process_parser_counts_only_real_processes(self) -> None:
        empty = [{"gpu": 1, "process_list": [{"process_info": "No running processes detected"}]}]
        busy = [{"gpu": 1, "process_list": [{"process_info": "123 sllm-server"}]}]
        self.assertEqual(process_count(empty, 1), 0)
        self.assertEqual(process_count(busy, 1), 1)

    def test_only_gfx942_can_record_provider_blocked_metrics_as_unavailable(self) -> None:
        with patch(
            "ci.tools.run_openai_a6_gpu.optional_amd_smi", return_value={"state": "unavailable"}
        ) as optional, patch(
            "ci.tools.run_openai_a6_gpu.amd_smi", return_value={"temperature": 42}
        ) as required:
            self.assertEqual(metric_observation("gfx942", 0), {"state": "unavailable"})
            self.assertEqual(metric_observation("gfx1201", 2), {"temperature": 42})
        optional.assert_called_once_with("metric", 0)
        required.assert_called_once_with("metric", 2)


if __name__ == "__main__":
    unittest.main()
