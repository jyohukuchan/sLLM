# Step 4: no valid grouped-GQA specialisation for the benchmarked full layers

Steps 1–3 were committed and measured before this review.  The requested
larger fast-body attempt was examined but no specialisation was added.

The current grouped split implementation is not a template whose constants can
be safely widened by dispatch.  Its specialised branch is a 256-thread,
wave-32 body and the generic split launch follows it only after rejecting
`head_dim > 256` or `value_dim > 256`.  Gemma4 E2B full attention is
`q_per_kv=8`, `head_dim=512`, `value_dim=512`.  A new branch for that shape
would need a different per-wave Q layout, a 512-element score reduction, and
at least a 1024-float K/V staging contract.  It is new GPU math and a new
finite-precision contract, not a multi-specialisation of the existing body.

The local Gemma4 layers are `8/1/256/256`, but their cache has sliding-window
semantics and they are not the requested full-attention decode geometry.
Routing only those layers would not make Gemma4 full attention reach the
optimised path and would produce a misleading result.  Therefore there is no
Step-4 performance number or new kernel symbol to report.

The next legitimate optimisation is a separately designed and validated
8Q/1KV/512/512 split (or direct grouped) kernel, followed by a multi-step,
full-model numerical comparison; it must not be presented as the descriptor-
only Step 2 change.
