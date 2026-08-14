# Third-party notices

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
  commit: pending-until-A3-import-commit
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
  commit: pending-until-A3-import-commit
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
  commit: pending-until-A6-import-commit
```
