# Phase 6 A2 llama.cpp import計画

これはA2時点で作成した実装前のimport manifest案である。A3でsamplerと選択testの実importを行い、
現在のactual importはrepository rootの[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)を正とする。
upstreamは
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) commit
`f5919bf458ef190468b5c329bb293f8a54a1e69c`、MIT、license SHA-256
`94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d`、noticeは
`Copyright (c) 2023-2026 The ggml authors`である。

## 再利用unit

| upstream path | blob | reuse | 範囲 / planned local |
| --- | --- | --- | --- |
| `src/llama-sampler.cpp` | `a9cb6bee5fd78728e5c94d5d1d008c3022abf330` | ported | temperature、top-p、presence/frequency penaltyだけ / `crates/sllm-core/src/sampling.rs` |
| `src/llama-sampler.h` | `b9bfc20d251738289b6c8357c38a0a04178b8b8c` | ported | 上記sampler contractだけ / 同上 |
| `tests/test-sampling.cpp` | `2aecff90e7bb4b8c09e32ae3dab24d41ca2138f0` | adapted | tiny-logitと境界case / `crates/sllm-core/tests/sampling_contract.rs` |
| `tools/server/tests/unit/test_chat_completion.py` | `0258b539ed870a7ed90ff4acc6bbd5ee233286aa` | adapted | profile v1 request/response/usage/finish reason case |
| `tools/server/tests/unit/test_stream.py` | `a1ef55567bc7e1f830b46b8624b8f8a5a77b6d1e` | adapted | chunk順、終端、disconnect case |
| `tools/server/tests/unit/test_security.py` | `ac0544575bd267a35b5551107529a9c1d9eb265c` | adapted | 該当body/path/header negative case |
| `tools/server/tests/utils.py` | `ae56bc70a15aa90b5dfe2810c7cc93b9f810e9bc` | facts-only | harness責務だけ。実装・構造は持ち込まない |

各file SHA-256、scope、planned localは
[`ci/contracts/phase6-a2-v1.json`](../../ci/contracts/phase6-a2-v1.json)を正とする。sampler file全体や
profile外sampler/testはimportしない。C++ server architectureはfacts-onlyである。

## A3 actual import

A3で`src/llama-sampler.cpp`のprofile v1 subsetを`crates/sllm-core/src/sampling.rs`へportedし、
`tests/test-sampling.cpp`のtiny-logit/boundary ideaをRust unit/integration testへadaptedした。local source header、
upstream URL/commit/blob、imported SHA-256、copyright/license、reuse mode、変更内容を
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)へ記録し、upstream MIT licenseも保持した。
noticeのimport commitは実際にlocal bytesを導入した
`b3fbfdccda87628b94d1440df1bf25707cd93c35`へ確定した。

## A6 actual HTTP test adaptation

A6で`test_chat_completion.py`、`test_stream.py`、`test_security.py`からprofile v1に該当する
request/response/usage/finish、SSE順序/終端、body negative caseだけを
`tests/fixtures/openai_chat_profile_v1.json`と`crates/sllm-server/tests/http_contract.rs`へadaptした。
llama.cpp固有fieldとC++ server architectureは持ち込んでいない。exact blob、source/local SHA-256、変更内容、
MIT license、reuse modeはrepository rootの`THIRD_PARTY_NOTICES.md`に別noticeとして記録した。import commitは
実際にlocal bytesを導入した`b3fbfdccda87628b94d1440df1bf25707cd93c35`へ確定した。
