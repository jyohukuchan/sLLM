#!/usr/bin/env python3
"""Execute one pinned peer engine's real Pydantic response serializers."""

from __future__ import annotations

import argparse
import json


def vllm_snapshot() -> dict[str, object]:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
        ChatMessage,
    )
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage, UsageInfo

    usage = UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    response = ChatCompletionResponse(
        id="chatcmpl-diff",
        created=1,
        model="qwen-diff",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
    chunks = [
        ChatCompletionStreamResponse(
            id="chatcmpl-diff",
            created=1,
            model="qwen-diff",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-diff",
            created=1,
            model="qwen-diff",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content="hello"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-diff",
            created=1,
            model="qwen-diff",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ),
    ]
    return serialize(response, chunks)


def sglang_snapshot() -> dict[str, object]:
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
        ChatMessage,
        DeltaMessage,
        UsageInfo,
    )

    usage = UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    response = ChatCompletionResponse(
        id="chatcmpl-diff",
        created=1,
        model="qwen-diff",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
    chunks = [
        ChatCompletionStreamResponse(
            id="chatcmpl-diff",
            created=1,
            model="qwen-diff",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-diff",
            created=1,
            model="qwen-diff",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content="hello"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionStreamResponse(
            id="chatcmpl-diff",
            created=1,
            model="qwen-diff",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ),
    ]
    return serialize(response, chunks)


def serialize(response, chunks) -> dict[str, object]:
    return {
        "non_stream": json.loads(response.model_dump_json(exclude_none=True)),
        "stream": [
            json.loads(chunk.model_dump_json(exclude_none=True)) for chunk in chunks
        ],
        "terminal": "[DONE]",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    args = parser.parse_args()
    snapshot = vllm_snapshot() if args.engine == "vllm" else sglang_snapshot()
    print(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
