# Third-party notices

## llama-cpp-phase9-mmvf-001

The Phase 9 BF16 M=1 matvec fast path adapts llama.cpp's paired-load and
wave-reduction MMVF structure to sLLM's fixed BF16 semantic ABI.

```yaml
schema_version: 1
id: llama-cpp-phase9-mmvf-001
component: sLLM HIP BF16 decode matvec v3
upstream:
  repository: https://github.com/ggml-org/llama.cpp
  commit: f5919bf458ef190468b5c329bb293f8a54a1e69c
  sources:
    - path: ggml/src/ggml-cuda/mmvf.cu
      git_blob: d7dbc8b992820c5da385575526a85a7524a6aaa2
      sha256: 23b580ce14a45e71cc9be31047301d502be74a832084c16662985f93f533ba1c
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/ggml/src/ggml-cuda/mmvf.cu
local:
  files:
    - path: native/hip/src/matmul_kernel.hip.cpp
      imported_sha256: 82f6d5952ce75753707fcc461351f2a768055ed076a5e396894faa467d494815
copyright:
  - Copyright (c) 2023-2026 The ggml authors
license:
  spdx: MIT
  file: docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt
  upstream_blob: e7dca554bcb802f98408383a864404e3aa4eacca
  sha256: 94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d
reuse:
  mode: adapted
  modifications:
    - Removed ggml tensor, fusion, ID routing, CUDA graph, and generic type dependencies.
    - Converted the input/output contract to BF16 with FP32 accumulation and checked scalar fallback for odd or unaligned reductions.
    - Fixed wave32 launch geometry for the canonical AMD targets and retained the independent sLLM dispatch registry.
import:
  commit: 6444555cc2dab919bd98994c1e2cfb3941969ed1
```

## llama-cpp-phase9-gdn-layout-001

The Phase 9 V620 recurrent GDN state layout adapts llama.cpp's wave-coalesced
state-shard access. The R9700 retains sLLM's prior contiguous-row layout because
the adapted layout regressed that target.

```yaml
schema_version: 1
id: llama-cpp-phase9-gdn-layout-001
component: sLLM Qwen3.5 linear-attention recurrent state layout
upstream:
  repository: https://github.com/ggml-org/llama.cpp
  commit: f5919bf458ef190468b5c329bb293f8a54a1e69c
  sources:
    - path: ggml/src/ggml-cuda/gated_delta_net.cu
      git_blob: 1b431a724d7237121dea29ca9c82bcd4817337a7
      sha256: fe5cfe4a35195fac999e8bd93d3ed18c68830096d4019bafa60e5da91d6ef4bf
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/ggml/src/ggml-cuda/gated_delta_net.cu
local:
  files:
    - path: native/hip/src/linear_attention_kernel.hip.cpp
      imported_sha256: 05faf73a9aa1e5854b5f8f81833b9cbfeaedbfe05396abe726798c19e1a22c7a
copyright:
  - Copyright (c) 2023-2026 The ggml authors
license:
  spdx: MIT
  file: docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt
  upstream_blob: e7dca554bcb802f98408383a864404e3aa4eacca
  sha256: 94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d
reuse:
  mode: adapted
  modifications:
    - Limited the adaptation to the private FP32 recurrent-state physical index mapping.
    - Preserved sLLM's BF16 input, transactional state publication, public ABI, and numerical operation order.
    - Enabled the transposed layout only for gfx1030 after real-GPU differential and performance testing.
import:
  commit: 6444555cc2dab919bd98994c1e2cfb3941969ed1
```

## llama-cpp-phase35-gdn-column-state-001

The Phase 35 long-prefill GDN provider adapts llama.cpp's column-owned
recurrent-state organization. sLLM retains its own Qwen graph, BF16 and FP32
contracts, transactional state publication, target dispatch, and host runtime.

```yaml
schema_version: 1
id: llama-cpp-phase35-gdn-column-state-001
component: sLLM Qwen3.5 long-prefill GDN column-state provider
upstream:
  repository: https://github.com/ggml-org/llama.cpp
  commit: f5919bf458ef190468b5c329bb293f8a54a1e69c
  sources:
    - path: ggml/src/ggml-cuda/gated_delta_net.cu
      git_blob: 1b431a724d7237121dea29ca9c82bcd4817337a7
      sha256: fe5cfe4a35195fac999e8bd93d3ed18c68830096d4019bafa60e5da91d6ef4bf
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/ggml/src/ggml-cuda/gated_delta_net.cu
local:
  files:
    - path: native/hip/src/linear_attention_kernel.hip.cpp
      imported_sha256: cf8e8aafa5e7e64c8fe5bc082912b5b8a328d0a9ed407965d6782cad72b3bc4a
copyright:
  - Copyright (c) 2023-2026 The ggml authors
license:
  spdx: MIT
  file: docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt
  upstream_blob: e7dca554bcb802f98408383a864404e3aa4eacca
  sha256: 94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d
reuse:
  mode: adapted
  modifications:
    - Adapted state-column ownership to a fixed wave32 x 4 HIP workgroup and sLLM's existing exact-target physical state mapping.
    - Split normalization, recurrent update, and output postprocessing into sLLM-owned kernels with BF16 RNE boundaries and FP32 state.
    - Preserved transactional next-state publication, existing short/decode routing, public ABI, cleanup contract, and gfx1030/gfx1201 common source path.
import:
  commit: bca482251bd21b144d950956af39a769c4211417
```

## llama-cpp-profile-v1-sampling-001

Portions of the profile-v1 sampler were ported from llama.cpp.

```yaml
schema_version: 1
id: llama-cpp-profile-v1-sampling-001
component: sLLM profile-v1 sampler
upstream:
  repository: https://github.com/ggml-org/llama.cpp
  commit: f5919bf458ef190468b5c329bb293f8a54a1e69c
  sources:
    - path: src/llama-sampler.cpp
      git_blob: a9cb6bee5fd78728e5c94d5d1d008c3022abf330
      sha256: ff421839a5fb33d781dff2125e28aef7503cd1e98220b0813e40d159839be93d
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/src/llama-sampler.cpp
local:
  files:
    - path: crates/sllm-core/src/sampling.rs
      imported_sha256: 0965ba54bc21bad846f050143b4f8034129b03c6180d950790500a104ecb8013
copyright:
  - Copyright (c) 2023-2026 The ggml authors
license:
  spdx: MIT
  file: docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt
  upstream_blob: e7dca554bcb802f98408383a864404e3aa4eacca
  sha256: 94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d
reuse:
  mode: ported
  modifications:
    - Restricted the implementation to temperature, top-p, presence penalty, and frequency penalty.
    - Replaced the C++ candidate array and mt19937 interfaces with checked Rust ownership and an injected random-source trait.
    - Added fail-closed handling for NaN, infinite logits, empty mass, overflow, and parameter ranges.
    - Preserved the temperature-zero device Argmax path without a host logits readback.
import:
  commit: b3fbfdccda87628b94d1440df1bf25707cd93c35
```

## llama-cpp-profile-v1-sampling-tests-001

Selected tiny-logit and boundary ideas were adapted from llama.cpp's sampling tests.

```yaml
schema_version: 1
id: llama-cpp-profile-v1-sampling-tests-001
component: sLLM profile-v1 sampler tests
upstream:
  repository: https://github.com/ggml-org/llama.cpp
  commit: f5919bf458ef190468b5c329bb293f8a54a1e69c
  sources:
    - path: tests/test-sampling.cpp
      git_blob: 2aecff90e7bb4b8c09e32ae3dab24d41ca2138f0
      sha256: f6ef72cf70e2ead3384893c68ff167bf5292f6590273bce14f2933d50f454d74
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/tests/test-sampling.cpp
local:
  files:
    - path: crates/sllm-core/src/sampling.rs
      imported_sha256: 0965ba54bc21bad846f050143b4f8034129b03c6180d950790500a104ecb8013
    - path: crates/sllm-core/tests/sampling_contract.rs
      imported_sha256: 431b4892ddd431c5933c1188ff446d58362a686e24535baf1b5b7d9b0f580079
copyright:
  - Copyright (c) 2023-2026 The ggml authors
license:
  spdx: MIT
  file: docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt
reuse:
  mode: adapted
  modifications:
    - Limited cases to the profile-v1 sampler and rewrote them as Rust unit tests.
    - Added non-aligned vocabulary, both parameter boundaries, NaN/Inf, deterministic random injection, and stable tie cases.
import:
  commit: b3fbfdccda87628b94d1440df1bf25707cd93c35
```

## llama-cpp-profile-v1-http-tests-001

Selected Chat Completions response, usage, streaming, disconnect, and
authentication test ideas were adapted from llama.cpp into sLLM's narrower
profile-v1 fixture and raw HTTP/SSE test. No llama-specific request field,
resume endpoint, server implementation, or harness structure was imported.

```yaml
schema_version: 1
id: llama-cpp-profile-v1-http-tests-001
component: sLLM OpenAI-compatible profile-v1 HTTP tests
upstream:
  repository: https://github.com/ggml-org/llama.cpp
  commit: f5919bf458ef190468b5c329bb293f8a54a1e69c
  sources:
    - path: tools/server/tests/unit/test_chat_completion.py
      git_blob: 0258b539ed870a7ed90ff4acc6bbd5ee233286aa
      sha256: 32199bdf961bf3227667b1763456f3b981401cc67024be94ee986a92573f9622
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/tools/server/tests/unit/test_chat_completion.py
    - path: tools/server/tests/unit/test_stream.py
      git_blob: a1ef55567bc7e1f830b46b8624b8f8a5a77b6d1e
      sha256: 8156d07e0e0886dbe790eb90b1aed8870c25b0b2e3ed90b769870c04c4617002
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/tools/server/tests/unit/test_stream.py
    - path: tools/server/tests/unit/test_security.py
      git_blob: ac0544575bd267a35b5551107529a9c1d9eb265c
      sha256: 3a3d1c1355a85ea025edafca93c5eda4d2a653000db99414d591209b152f4a34
      url: https://github.com/ggml-org/llama.cpp/blob/f5919bf458ef190468b5c329bb293f8a54a1e69c/tools/server/tests/unit/test_security.py
local:
  files:
    - path: crates/sllm-server/tests/http_contract.rs
      imported_sha256: 3906f29b9882749197379c6cc122046b01df7adf40a2447d6ca325478f78da5f
    - path: tests/fixtures/openai_chat_profile_v1.json
      imported_sha256: 9a2e19252de24ae37b9a866c62463999f45cdf281e37c28b4da5005fcccbda24
copyright:
  - Copyright (c) 2023-2026 The ggml authors
license:
  spdx: MIT
  file: docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt
  upstream_blob: e7dca554bcb802f98408383a864404e3aa4eacca
  sha256: 94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d
reuse:
  mode: adapted
  modifications:
    - Limited cases to the pinned sLLM profile-v1 request and response subset.
    - Rewrote the cases as Rust raw HTTP/SSE tests and one declarative JSON fixture.
    - Retained response object, one choice, assistant role, finish reason, usage, stable stream identity, final chunk, and exact DONE checks.
    - Adapted disconnect and bearer-authentication ideas to the bounded Rust scheduler without importing llama.cpp server or harness structure.
    - Added strict unsupported-field, type, size, queue, cancellation, and mid-stream terminal-error cases required by the sLLM profile.
import:
  commit: b3fbfdccda87628b94d1440df1bf25707cd93c35
```
