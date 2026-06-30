# ADR 0001: License and reference-code policy

## Status

Accepted for v0.1 planning.

## Context

uLLM is currently configured as Apache-2.0 on GitHub. The project will need to study llama.cpp, vLLM, SGLang, ATOM, TensorRT-LLM, and other inference runtimes. Some of these projects are Apache-2.0 compatible, some are permissive but not Apache-2.0, and some may have missing or unclear license metadata in the checked source tree.

Even when licenses are permissive, unconscious code copying is a practical risk. The project needs reference access without mixing external implementation code into uLLM.

## Decision

- uLLM remains Apache-2.0 unless a later ADR changes it.
- Reference source trees are kept under `reference-src/`, which is ignored by Git.
- `tools/fetch-reference-sources.sh` is the reproducible way to fetch reference repositories.
- Reference code may be read for behavior, architecture, benchmark design, API comparison, and compatibility research.
- Reference code must not be copied into uLLM implementation files.
- Implementation notes derived from reference reading must be written as design notes, specs, or benchmarks, not as translated code.
- Any intentional import of third-party code requires a new ADR before merging.
- That ADR must document source project, license, files copied, modifications, NOTICE requirements, and compatibility with Apache-2.0 distribution.
- License-unknown repositories are read-only references. Their code must not be reused until license status is resolved.

## Current Reference License Notes

- `llama.cpp`: MIT license in the checked source tree.
- `vllm`: Apache-2.0 license in the checked source tree.
- `sglang`: Apache-2.0 license in the checked source tree.
- `tensorrt-llm`: Apache-2.0 license in the checked source tree, with bundled third-party license notices.
- `atom` (`ROCm/ATOM`): MIT license in the checked source tree.

## Consequences

- The project can safely keep Apache-2.0 as the default while still studying other runtimes.
- Contributors must be careful to implement behavior from first principles or from published specifications, not by porting external code.
- Release checklists must include license and NOTICE review.
- If future performance work requires direct reuse of external kernels, it must be explicitly reviewed before adoption.
