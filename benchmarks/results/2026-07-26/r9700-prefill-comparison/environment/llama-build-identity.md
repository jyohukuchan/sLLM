# llama.cpp measurement-build identity

| item | value |
| --- | --- |
| source HEAD | `68a5592c10666d4d89b8480b5b9e8f8068b2f64c` |
| binary SHA-256 | `50a7e67db3fad77f18f9310e486c8547c2e53fe85b96af093b5910a3d07a8481` |
| ROCm target | local `build-rdna4` gfx1201 build |
| working-tree difference | [minimal f32 cache parser mapping](llama-f32-cache-parser.patch) |

The lone local source modification is the f32 parser mapping above, required
for `llama-bench -ctk f32 -ctv f32`.  It does not alter a kernel, graph,
model loading, cache layout, or benchmark timing.  It predates this prefill
run and is recorded because the F32-KV row otherwise cannot be reproduced
from pristine commit 68a5592 alone.  The F16-KV row uses the unmodified
existing parser path.
