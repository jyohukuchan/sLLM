# Served-model execution contract verification

Commit `bfc76a72aeee31ca1558c5367cef941689e2047e` extends only
`ullm.served_model.v2` with an optional typed execution field:

```json
{
  "worker": {
    "execution": {
      "paged_decode_attention": {
        "kernel": "gqa_grouped_split",
        "split_tile": 20
      }
    }
  }
}
```

The parser is fail-closed: no unknown keys, no other kernel name, and no tile
outside `20`, `128`, `256`, or `512` are accepted.  An execution contract is
currently admitted only for `SQ8_0`, `gfx1201`,
`rdna4_w8a8_block_ck`, and a manifest that requires
`ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL`.  It cannot be smuggled through
`worker.required_environment`.

Manifest-mode gateway startup first removes all four relevant parent selectors
and then, for the typed candidate only, sets exactly tile, grouped, and
allow-multitile.  The rejected pipeline selector is never represented.  The
Rust worker independently requires this same exact selector set, including
the absence of pipeline; direct invocation with stale selectors fails rather
than silently selecting a different body.

`active-aq4-p3-validation.json` is the read-only regression check for the
current production manifest.  It validates with unchanged SHA-256
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` and
`worker.execution: null`, so the old `AQ4_0` P3 worker receives no new selector.

The promotion and rollback wrappers did not need a schema-specific rewrite:
they atomically exchange raw manifest bytes.  Their test covers a typed
execution field across promotion and rollback, proving they do not parse and
re-serialize it away.
