#!/usr/bin/env python3
"""Resolve the mtp-bench-v1 corpus and assemble one frozen prompt per condition.

Repository sources are read from a pinned commit with `git show`, never from the
working tree, so the fixture is reproducible from the recorded commit alone.
Authored sources live under ci/fixtures/mtp-bench-v1/corpus/.

Prompt length is frozen per condition, not equalised across conditions: every
metric in the benchmark is a within-condition paired comparison, so only
per-condition stability matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = REPO / "ci" / "fixtures" / "mtp-bench-v1"
CORPUS = FIXTURE / "corpus"
PROMPTS = FIXTURE / "prompts"

CODE_INSTRUCTIONS = {
    "implement": {
        "en": "You are working in this repository. Read the source below, then implement the function described in the TASK section. Return only the new code with a short comment block explaining the chosen approach.",
        "ja": "あなたはこのリポジトリで作業しています。以下のソースを読み、TASK節で指示された関数を実装してください。新しいコードと、選択した方針を説明する短いコメントだけを返してください。",
        "zh": "你正在这个代码库中工作。请阅读下面的源码，然后实现 TASK 节中描述的函数。只返回新代码，并附上一段简短的注释说明所选方案。",
    },
    "bugfix": {
        "en": "You are working in this repository. The source below fails the check described in the TASK section. Identify the defect and return a minimal corrected version of the affected region, followed by one sentence naming the root cause.",
        "ja": "あなたはこのリポジトリで作業しています。以下のソースはTASK節に記した検査に失敗します。欠陥を特定し、影響範囲の最小限の修正版を返したうえで、根本原因を一文で述べてください。",
        "zh": "你正在这个代码库中工作。下面的源码未能通过 TASK 节所述的检查。请找出缺陷，返回受影响区域的最小修正版本，并用一句话说明根本原因。",
    },
    "test": {
        "en": "You are working in this repository. Read the source below and write tests for the behaviour described in the TASK section. Cover non-aligned values and both sides of each relevant boundary, not only powers of two.",
        "ja": "あなたはこのリポジトリで作業しています。以下のソースを読み、TASK節の挙動に対するテストを書いてください。2の冪だけでなく、非整列値と各境界の両側を含めてください。",
        "zh": "你正在这个代码库中工作。请阅读下面的源码，为 TASK 节描述的行为编写测试。除了 2 的幂之外，还要覆盖非对齐取值以及每个相关边界的两侧。",
    },
    "refactor": {
        "en": "You are working in this repository. Read the source below and propose a refactoring for the region named in the TASK section. Behaviour must not change. Explain what improves and what stays the same.",
        "ja": "あなたはこのリポジトリで作業しています。以下のソースを読み、TASK節で指定した箇所のリファクタリングを提案してください。挙動は変えないこと。何が改善し何が変わらないかを説明してください。",
        "zh": "你正在这个代码库中工作。请阅读下面的源码，对 TASK 节指定的区域提出重构方案。行为不得改变。请说明哪些方面得到改善、哪些保持不变。",
    },
    "review": {
        "en": "You are reviewing a change to this repository. Read the source below and write review comments for the concern raised in the TASK section. State each finding, why it matters, and what you would change.",
        "ja": "あなたはこのリポジトリの変更をレビューしています。以下のソースを読み、TASK節で挙げた観点についてレビュー指摘を書いてください。指摘内容、なぜ問題か、どう変えるべきかを述べてください。",
        "zh": "你正在评审对这个代码库的改动。请阅读下面的源码，针对 TASK 节提出的关注点撰写评审意见。请说明每一处发现、为何重要，以及你会如何修改。",
    },
    "explain": {
        "en": "You are onboarding a new contributor to this repository. Read the source below and explain the part named in the TASK section in prose. Describe what it does, why it is written this way, and what a reader should be careful about.",
        "ja": "あなたはこのリポジトリに新しく参加した人へ説明しています。以下のソースを読み、TASK節で指定した部分を散文で説明してください。何をしているか、なぜこの書き方なのか、読む際の注意点を述べてください。",
        "zh": "你正在向新加入这个代码库的同事讲解。请阅读下面的源码，用散文说明 TASK 节指定的部分。请描述它做什么、为什么这样写，以及阅读时需要注意什么。",
    },
}

TRANS_INSTRUCTIONS = {
    ("en", "ja"): "Translate the following text into Japanese. Preserve paragraph structure, technical terms, and the register of the original. Output only the translation.",
    ("ja", "en"): "以下の文章を英語へ翻訳してください。段落構成、専門用語、原文の文体を保ってください。訳文だけを出力してください。",
    ("en", "zh"): "Translate the following text into Simplified Chinese. Preserve paragraph structure, technical terms, and the register of the original. Output only the translation.",
    ("zh", "en"): "请将下面的文本翻译成英文。请保持段落结构、专业术语和原文的语体。只输出译文。",
}

CREATIVE_INSTRUCTIONS = {
    "en": "Continue the following opening into a short scene. Keep the narrator, tense, and setting consistent.",
    "ja": "以下の書き出しに続けて短い場面を書いてください。語り手、時制、舞台設定は変えないでください。",
    "zh": "请接着下面的开头写一个短场景。保持叙述者、时态和场景设定一致。",
}

CREATIVE_OPENINGS = {
    "en": "The last tram had gone twenty minutes ago, and the conductor was still standing on the platform, counting something on his fingers.",
    "ja": "最終の路面電車は二十分前に出たはずなのに、車掌はまだホームに立って、指で何かを数えていた。",
    "zh": "末班有轨电车二十分钟前就开走了，可售票员还站在站台上，用手指数着什么。",
}

# code-heavy condition and prose condition share one source per language, so the
# task effect is isolated from the content effect.
LANGUAGES = [
    # key, display, source spec, target tokens, tasks (code-heavy, prose), nat langs
    ("c", "C", [("repo", "native/hip/src/header_c_compile.c"), ("repo", "native/hip/src/evidence_abi.h")], 4608, ("implement", "explain"), ("en", "ja")),
    ("html", "HTML", [("corpus", "html_benchmark_dashboard.html")], 2304, ("test", "review"), ("ja", "zh")),
    ("rust", "Rust", [("repo", "crates/sllm-core/src/mtp_quantized_sidecar.rs")], 8192, ("bugfix", "refactor"), ("zh", "en")),
    ("cuda", "C++ (CUDA)", [("corpus", "cuda_block_scaled_gemv.cu")], 2048, ("implement", "explain"), ("en", "ja")),
    ("python", "Python", [("repo", "ci/tools/common.py")], 8192, ("test", "review"), ("ja", "zh")),
    ("sycl", "C++ (SYCL)", [("corpus", "sycl_block_scaled_gemv.cpp")], 2048, ("bugfix", "refactor"), ("zh", "en")),
    ("go", "Go", [("corpus", "go_inference_gateway.go")], 2048, ("implement", "explain"), ("en", "ja")),
    ("typescript", "TypeScript", [("repo", "webui/lib/sllm-api.ts"), ("repo", "webui/app/page.tsx")], 8192, ("test", "review"), ("ja", "zh")),
    ("java", "Java", [("corpus", "java_bpe_tokenizer.java")], 2048, ("bugfix", "refactor"), ("zh", "en")),
    ("sql", "SQL", [("corpus", "sql_benchmark_schema.sql")], 2048, ("implement", "explain"), ("en", "ja")),
]

TASK_BODY = {
    ("c", "implement"): "Add `sllm_evidence_header_probe_v2` that validates the ABI layout fields above and returns a non-zero status naming the first mismatching field.",
    ("c", "explain"): "Explain the compile-time layout assertions and what breaks if one of them is removed.",
    ("html", "test"): "Write browser tests for `renderRow` and the baseline-drift tile, including the case where `results.json` returns a non-200 status.",
    ("html", "review"): "Review the dark-mode token definitions and the `load()` error path for accessibility and failure handling.",
    ("rust", "bugfix"): "`validate_record_rejects_nonaligned_k_dtype_scale_length_and_truncation` fails for a K that is a multiple of 32 but whose scale array is one element short. Locate the missing length check.",
    ("rust", "refactor"): "Refactor the manifest-encoding parsing so the roundtrip and packed recipes no longer duplicate their converter-name validation.",
    ("python", "test"): "Write tests for the manifest loading and digest helpers, covering an absent file, a truncated JSON body, and a digest string of the wrong length.",
    ("python", "review"): "Review the error handling and path resolution helpers for cases where a relative path escapes the repository root.",
    ("cuda", "implement"): "Add a variant of the kernel that hoists the block scale into a register once per 32-element block instead of reloading it for every element.",
    ("cuda", "explain"): "Explain why the activation stays in BF16 while the weights are E4M3, and what that costs and saves at M=1.",
    ("sycl", "bugfix"): "`decode_e4m3` returns the wrong magnitude for subnormal inputs when the mantissa is zero after normalisation. Identify and fix the defect.",
    ("sycl", "refactor"): "Refactor `submit_mxfp8_w8a16_gemv` so the two-column body is expressed once instead of duplicating the column-0 and column-1 paths.",
    ("go", "implement"): "Add `bytesReader` and a `Probe` method that marks a backend healthy again after a successful health check, with bounded concurrency.",
    ("go", "explain"): "Explain why `TokensPerSecond` derives the rate from the block decomposition instead of timing the request end to end.",
    ("java", "bugfix"): "`encode` drops the final symbol when the last two symbols merge on the very last iteration. Identify the linked-list update that is wrong.",
    ("java", "refactor"): "Refactor the symbol linked list so `Symbol` no longer carries an `alive` flag that every reader must remember to check.",
    ("typescript", "test"): "Write tests for the API client's error mapping and the metrics formatting helpers, including a response with a missing counter field.",
    ("typescript", "review"): "Review the client-side state handling for requests that are cancelled while a stream is still open.",
    ("sql", "implement"): "Add a `baseline_reference` table and the migration that backfills it from the first campaign recorded for each target.",
    ("sql", "explain"): "Explain why `tier_a_paired_delta` excludes tier B by joining on the condition table rather than filtering the result afterwards.",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_show(commit: str, path: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                         capture_output=True, check=True)
    return out.stdout.decode("utf-8")


def truncate_lines(text: str, tokenizer, budget: int) -> str:
    """Keep the longest line-aligned prefix that fits in `budget` tokens."""
    if len(tokenizer.encode(text, add_special_tokens=False).ids) <= budget:
        return text
    lines = text.split("\n")
    low, high = 0, len(lines)
    while low < high:
        mid = (low + high + 1) // 2
        chunk = "\n".join(lines[:mid])
        if len(tokenizer.encode(chunk, add_special_tokens=False).ids) <= budget:
            low = mid
        else:
            high = mid - 1
    return "\n".join(lines[:low]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="pinned repository commit for repo sources")
    parser.add_argument("--tokenizer", required=True, help="absolute path to tokenizer.json")
    parser.add_argument("--out", required=True, help="manifest output path")
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    PROMPTS.mkdir(parents=True, exist_ok=True)

    conditions = []

    for key, display, sources, budget, (code_task, prose_task), (code_nat, prose_nat) in LANGUAGES:
        parts, refs = [], []
        for kind, ref in sources:
            if kind == "repo":
                body = git_show(args.commit, ref)
                refs.append({"kind": "repo", "ref": f"{ref}@{args.commit}", "sha256": sha256(body)})
            else:
                body = (CORPUS / ref).read_text(encoding="utf-8")
                refs.append({"kind": "authored", "ref": f"ci/fixtures/mtp-bench-v1/corpus/{ref}",
                             "sha256": sha256(body)})
            parts.append(f"--- {ref} ---\n{body}")
        source = truncate_lines("\n".join(parts), tokenizer, budget)
        source_sha = sha256(source)

        for task, nat in ((code_task, code_nat), (prose_task, prose_nat)):
            prompt = (f"{CODE_INSTRUCTIONS[task][nat]}\n\n{source}\n"
                      f"TASK ({display}): {TASK_BODY[(key, task)]}\n")
            cond_id = f"code-{key}-{task}-{nat}"
            (PROMPTS / f"{cond_id}.txt").write_text(prompt, encoding="utf-8")
            conditions.append({
                "cond_id": cond_id, "tier": "A", "group": "coding_agent",
                "task": task, "code_language": display, "instruction_language": nat,
                "code_prose_axis": "code_heavy" if task in ("implement", "bugfix", "test") else "prose_side",
                "sources": refs, "source_sha256": source_sha,
                "prompt_file": f"prompts/{cond_id}.txt", "prompt_sha256": sha256(prompt),
                "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False).ids),
                "output_tokens": 256,
            })

    trans = [
        ("trans-en-ja-tech", "en", "ja", "tech", ("repo", "AGENTS.md")),
        ("trans-ja-en-tech", "ja", "en", "tech", ("repo", "docs/history/2026/09/11-20/mtp-language-task-acceptance.md")),
        ("trans-en-zh-tech", "en", "zh", "tech", ("repo", "AGENTS.md")),
        ("trans-zh-en-tech", "zh", "en", "tech", ("corpus", "trans_source_zh_tech.txt")),
        ("trans-en-ja-general", "en", "ja", "general", ("corpus", "trans_source_en_general.txt")),
        ("trans-ja-en-general", "ja", "en", "general", ("corpus", "trans_source_ja_general.txt")),
    ]
    for cond_id, src, dst, domain, (kind, ref) in trans:
        if kind == "repo":
            body = git_show(args.commit, ref)
            refs = [{"kind": "repo", "ref": f"{ref}@{args.commit}", "sha256": sha256(body)}]
        else:
            body = (CORPUS / ref).read_text(encoding="utf-8")
            refs = [{"kind": "authored", "ref": f"ci/fixtures/mtp-bench-v1/corpus/{ref}",
                     "sha256": sha256(body)}]
        source = truncate_lines(body, tokenizer, 1024)
        prompt = f"{TRANS_INSTRUCTIONS[(src, dst)]}\n\n{source}\n"
        (PROMPTS / f"{cond_id}.txt").write_text(prompt, encoding="utf-8")
        conditions.append({
            "cond_id": cond_id, "tier": "A", "group": "translation",
            "source_language": src, "target_language": dst, "domain": domain,
            "sources": refs, "source_sha256": sha256(source),
            "prompt_file": f"prompts/{cond_id}.txt", "prompt_sha256": sha256(prompt),
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False).ids),
            "output_tokens": 256,
        })

    for nat in ("en", "ja", "zh"):
        cond_id = f"creative-{nat}"
        prompt = f"{CREATIVE_INSTRUCTIONS[nat]}\n\n{CREATIVE_OPENINGS[nat]}\n"
        (PROMPTS / f"{cond_id}.txt").write_text(prompt, encoding="utf-8")
        conditions.append({
            "cond_id": cond_id, "tier": "B", "group": "creative",
            "instruction_language": nat, "sources": [{"kind": "authored", "ref": "inline", "sha256": sha256(prompt)}],
            "source_sha256": sha256(prompt),
            "prompt_file": f"prompts/{cond_id}.txt", "prompt_sha256": sha256(prompt),
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False).ids),
            "output_tokens": 256,
        })

    doc = json.loads(pathlib.Path(args.out).read_text(encoding="utf-8"))
    doc["corpus_commit"] = args.commit
    doc["tokenizer_sha256"] = hashlib.sha256(pathlib.Path(args.tokenizer).read_bytes()).hexdigest()
    doc["conditions"] = conditions
    doc["tier_a_count"] = sum(1 for c in conditions if c["tier"] == "A")
    doc["tier_b_count"] = sum(1 for c in conditions if c["tier"] == "B")
    pathlib.Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for c in conditions:
        print("%-34s %-12s %5d tok" % (c["cond_id"], c["group"], c["prompt_tokens"]))
    print("tierA=%d tierB=%d" % (doc["tier_a_count"], doc["tier_b_count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
