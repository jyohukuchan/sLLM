# Step 2: full-attention split admission is not valid for Gemma4 E2B

The supplied premise describes a Gemma4 full-attention shape at or below the
generic split ABI's 256-element limit.  The checkpoint used by the resident
benchmark is `/home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B`, and
its `config.json` says otherwise.

`text_config` values read from that file:

| attention kind | Q heads | KV heads | head dim | value dim | scale | output gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sliding | 8 | 1 | 256 | 256 | 1.0 | no |
| full | 8 | 1 | 512 | 512 | 1.0 | no |

The full layers are indices `4,9,14,19,24,29,34`.  `global_head_dim=512` is
the full-attention dimension; `head_dim=256` applies only to sliding attention.
`num_global_key_value_heads` and `query_pre_attn_scalar` are absent/null.  The
resident descriptor in `model_config.rs` deliberately selects
`global_head_dim` for `FullAttention`, sets `value_dim=head_dim`,
`ResidentAttentionScale::One`, and `output_gate=false`; the executor passes
that scale as literal `1.0` to the direct attention call.

The existing F32 split ABI rejects this shape before launch:

```c++
if (head_dim > 256 || value_dim > 256) {
    set_error("f32 paged decode split attention head_dim/value_dim must not exceed 256");
    return ULLM_STATUS_INVALID_ARGUMENT;
}
```

The scalar split partial and merge symbols independently return when their
dimensions exceed 256.  Therefore a Rust admission descriptor for Gemma4 full
attention cannot call that existing body correctly.  The direct generic F32
kernel does support the 512-wide full heads and was the baseline's observed
launch.  Widening this capability requires a new 512-wide split partial and
merge implementation (new GPU math/finite-precision contract), not the
descriptor-only change requested here.

No runtime or kernel source was changed in this step; Qwen3.5's admission and
kernel source are consequently byte-identical.  `tools/check-runtime-tu-identical.sh`
passes at this commit.
