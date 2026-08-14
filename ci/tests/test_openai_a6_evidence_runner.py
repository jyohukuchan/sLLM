from __future__ import annotations

import unittest

from ci.tools.run_openai_a6_gpu import parse_sse, process_count


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


if __name__ == "__main__":
    unittest.main()
