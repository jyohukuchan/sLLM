# Phase 9 engine optimization reader

This bounded reader records the external implementation units inspected for
Phase 9. It is not a source mirror and does not authorize reuse from engines
other than llama.cpp.

## Fixed source identity

- Repository: `https://github.com/ggml-org/llama.cpp`
- Commit: `f5919bf458ef190468b5c329bb293f8a54a1e69c`
- License snapshot: `docs/provenance/licenses/llama.cpp-MIT-f5919bf4.txt`

| Unit | Upstream range | Git blob | SHA-256 | Phase 9 use |
| --- | --- | --- | --- | --- |
| HIP/CUDA Graph state | `ggml/src/ggml-cuda/ggml-cuda.cu:2509-2617` and `common.cuh` graph state near line 1230 | `561ab7ac599f9e285d2a0296caee0ab0a14ea5c8`, `33be16dc5cced190c62de4a392bd4892a3140b1f` | `f6a2f64eef7ebc3f05df4ca12ee960c40ebee35289e99d20586c839a537d6aa5`, `0e16f5badb87661a47e0499770aaa4ed83d8937fcbca69a036afc29475f4e21e` | Reader only. Informed the independent HIP Graph PoC and explicit segment-cut decision; no graph/runtime source was imported. |
| Floating MMVF | `ggml/src/ggml-cuda/mmvf.cu:7-369,412-505` | `d7dbc8b992820c5da385575526a85a7524a6aaa2` | `23b580ce14a45e71cc9be31047301d502be74a832084c16662985f93f533ba1c` | Adapted paired loads and wave reduction into the bounded BF16 M=1 v3 kernel. ggml tensor/runtime, generic dispatch, and fusion were not imported. |
| Qwen GDN | `ggml/src/ggml-cuda/gated_delta_net.cu:4-221` | `1b431a724d7237121dea29ca9c82bcd4817337a7` | `fe5cfe4a35195fac999e8bd93d3ed18c68830096d4019bafa60e5da91d6ef4bf` | Adapted only the wave-coalesced recurrent-state layout for gfx1030. The kernel body, public ABI, state transaction, and gfx1201 layout remain sLLM-specific. |

## Selection decisions

- HIP Graph capture works on both canonical targets for a one-node sLLM kernel
  and a two-node hipBLAS mixed segment. Production Phase 9 nevertheless uses
  explicit same-stream segments: KV append and terminal readback are required
  state/publication boundaries, while per-operation completions are retained and
  queried only after the boundary event is terminal. This avoids request-local
  graph instantiation and preserves the existing prepared semantic plan.
- The MMVF adaptation is used for `M=1` on both targets. `M>1` stays on the
  existing tiled16 kernel on gfx1030 and uses model-resident hipBLAS handles on
  gfx1201.
- The GDN transposed physical state layout is target-specific: it improved
  gfx1030 but regressed gfx1201 in full-model measurements.
- No source expression from vLLM, SGLang, AITER, CK, or other inference engines
  was copied or adapted during Phase 9.

The two adapted units have release records in `THIRD_PARTY_NOTICES.md`. Their
import commit and imported-file hash remain explicitly pending until the changes
are committed; they must be resolved before release or distribution.
