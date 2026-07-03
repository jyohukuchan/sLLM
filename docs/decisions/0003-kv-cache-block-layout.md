# ADR 0003: KV cache block layout

## Status

Proposed for v0.1 implementation.

## Context

uLLMの推論エンジン設計は、vLLM、SGLang、ATOM、TensorRT-LLMの既存設計を参照しつつ進める。
まずは実装リスクを抑えるため、初期段階では単純な paged KV cache を実装する方針が妥当だと考える。

これらの参照実装は、block 管理、scheduler 連携、backend 境界、disaggregation の扱いを比較するために使う。

## Decision

- block-size の初期デフォルトは `16` トークンとし、固定値で運用する。
- block id は `u32` とする。
- sequence の block table は `Vec<u32>` で保持し、layer 横断で参照できる形にする。
- allocator は block の連続配置を要求せず、非連続 block を許容する。
- per-layer の KV payload は backend 側が所有し、Rust の scheduler は block table と寿命（lifetime）の管理のみを行う。
- prefix reuse / radix reuse は当面の対象外とし、初期実装では行わない。

## Consequences

- allocator の断片化状況を可視化する telemetry（空き率、欠片数、再利用率など）を必須にする。
- 将来、block-size は backend の capability として取得する方向に拡張する。
- prefill/decode の disaggregation を前提に、block table を KV transfer metadata の一部として送受信できる形で保持する。
- hybrid KV、sliding window cache は後続実装で再検討する。

## References

- `docs/research/inference-engine-reference-notes-v0.1.md`
  - vLLM: `kv_cache_manager.py` / `single_type_kv_cache_manager.py` / `block_pool.py` / paged attention 設計
  - SGLang: `scheduler.py` / `unified_radix_cache.py` / prefill・decode 分離フロー
  - ATOM: `paged_attention.py` / `paged_prefill.py` / `paged_decode.py` / KV transfer 系
  - TensorRT-LLM: `paged-attention-ifb-scheduler.md` / `kvcache.md` / `kv-cache-connector`
