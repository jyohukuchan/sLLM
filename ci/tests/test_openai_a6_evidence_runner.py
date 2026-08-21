from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ci.tools.run_openai_a6_gpu import (
    build_server_command,
    metric_observation,
    parse_reasoning_sse,
    parse_sse,
    process_count,
    reasoning_payload,
    seeded_sampling_payload,
    validate_reasoning_response,
    validate_seeded_response,
)


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

    def test_server_command_uses_public_gguf_inputs_only(self) -> None:
        command = build_server_command(
            Path("/tmp/sllm-server"),
            Path("/models/qwen.gguf"),
            Path("/models/qwen.derived-lock.json"),
            0,
            "gfx942",
            18080,
        )
        self.assertEqual(
            command,
            [
                "/tmp/sllm-server",
                "--gguf",
                "/models/qwen.gguf",
                "--derived-lock",
                "/models/qwen.derived-lock.json",
                "--device-index",
                "0",
                "--target",
                "gfx942",
                "--listen",
                "127.0.0.1:18080",
                "--model",
                "qwen3.5-4b",
            ],
        )
        self.assertNotIn("--lock", command)
        self.assertNotIn("--cache", command)

    def test_reasoning_payload_and_response_require_separation(self) -> None:
        payload = reasoning_payload(17, stream=True)
        self.assertEqual(payload["sllm"], {"thinking": "enabled", "separate_reasoning": True})
        self.assertTrue(payload["stream"])
        response = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "because",
                        "content": "answer",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        self.assertEqual(validate_reasoning_response(response), ("because", "answer"))
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "because"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 2}},
        ]
        self.assertEqual(parse_reasoning_sse(chunks), ("because", "answer"))
        with self.assertRaises(RuntimeError):
            validate_reasoning_response(
                {
                    **response,
                    "choices": [
                        {
                            **response["choices"][0],
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "<think>leak",
                                "content": "answer",
                            },
                        }
                    ],
                }
            )

    def test_seeded_sampling_payload_is_explicit_and_replay_validation_is_strict(self) -> None:
        payload = seeded_sampling_payload(17, seed=1902)
        self.assertEqual(payload["seed"], 1902)
        self.assertEqual(payload["temperature"], 0.8)
        self.assertEqual(payload["top_p"], 0.9)
        value = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "sample"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self.assertEqual(validate_seeded_response(value), ("sample", value["usage"]))
        with self.assertRaises(RuntimeError):
            validate_seeded_response({**value, "usage": None})


if __name__ == "__main__":
    unittest.main()
