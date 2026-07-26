# Fixed build used for the valid redesign window

- source worktree: `/home/homelab1/coding-local/ultimateLLM/uLLM-bh-redesign-b65e63c3`
- source commit: `b65e63c3e36381091d5823c06d42c105f48de14b`
- source worktree status before build: clean
- target directory: `/home/homelab1/coding-local/ultimateLLM/uLLM-bh-redesign-target-b65e63c3`
- build parallelism: `CARGO_BUILD_JOBS=8`

Commands completed before the GPU window:

```text
CARGO_TARGET_DIR=…/uLLM-bh-redesign-target-b65e63c3 CARGO_BUILD_JOBS=8 \
  cargo build --release -p ullm-runtime-sys \
  --example sq8_0_paged_decode_attention_probe

CARGO_TARGET_DIR=…/uLLM-bh-redesign-target-b65e63c3 CARGO_BUILD_JOBS=8 \
  cargo build --release -p ullm-engine --features rocm-ck-gfx1201 \
  --example sq8_0_paged_decode_steady_bench
```

SHA-256:

```text
67ec4f3eca06e26ce22c8e24f46c3ccca52294569331bd25704a218d38c80415  sq8_0_paged_decode_attention_probe
d5c59454171162eafdb159386a73a74741fbb6609a3236453923aa18399d7792  sq8_0_paged_decode_steady_bench
```

The compiler emitted pre-existing anonymous-namespace subobject-linkage
warnings from `ullm-runtime-sys` and one unused-method warning from
`ullm-engine`; both builds exited successfully.
