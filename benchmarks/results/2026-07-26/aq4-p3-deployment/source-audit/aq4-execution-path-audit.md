# AQ4_0 P3 deployment source audit

## Scope and selected base

The audit starts from active production source `0cd760568e197e1adb4c4df3d6149591a912f709`.
The supplied 255-commit scope was captured at shared-worktree snapshot
`0455b119f23f9971d185c219ecda1e534f5eab6`.  The selected source is the stable P3 endpoint
`c4c9a9b344fc10e9a77ab0ded3293469d21b2f72`, built in the detached worktree
`uLLM-aq4-p3-deployment-source-c4c9a9b3`; shared `HEAD` was not moved.

This selection follows the existing delta analysis: the continuous 47-commit P3 sequence contains
both prefill and decode work, while excluding subsequent AQ5/importance work and the experimental
SQ8 v2/runtime line.  Selecting only the 19 direct runtime-source files would create a new,
unvalidated reconstruction, so the contiguous endpoint is the smaller safe source selection.

## Commits that enter the AQ4_0 candidate

All 47 commits below are execution-path relevant and are included by the selected base.

### Prefill path (28)

| Commit | Subject |
|---|---|
| `de0cd86` | Add AQ4 group8 register BM8 GEMM path |
| `5044cdb` | Add AQ4 WMMA GEMM prototype |
| `5acb228` | Promote AQ4 WMMA GEMM for group16 M128 |
| `406ec02` | Clamp AQ4 differential test scale_index generation to valid range |
| `152df61` | Add AQ4 end-to-end prefill timing binary |
| `39386e6` | Keep linear attention recurrent state in registers |
| `b26568a` | Add AQ4 WMMA v2 pipeline experiment |
| `43a8133` | Extend AQ4 WMMA v2 validation coverage |
| `dd282a1` | Promote double-buffered AQ4 WMMA kernel |
| `b67560b` | Add paged causal GQA WMMA prototype |
| `5fab4c6` | Promote paged causal GQA WMMA QK kernel |
| `bbf0ebb` | Add AQ4 group8 WMMA GEMM prototype |
| `01a5da2` | Promote AQ4 group8 WMMA GEMM for M128 |
| `1c60d69` | Add linear attention recurrent shuffle prototype |
| `0171740` | Promote linear attention recurrent shuffle reduction |
| `00200ee` | Add AQ4 WMMA v3 two-Wide-K prototype |
| `eb09ee5` | Add AQ4 WMMA v4 direct-output occupancy prototype |
| `3ce0a0e` | Exercise AQ4 WMMA v4 prototype in GPU test harness |
| `ef62dc4` | Promote direct-output AQ4 WMMA kernel |
| `cac3666` | Add AQ4 session prefill chunk diagnostic |
| `95ac8eb` | Fix AQ4 prefill operation audit alternates |
| `9460a9c` | Widen paged GQA WMMA reader chunk range |
| `ce16588` | Add ragged-M AQ4 WMMA prototype |
| `67bd2a2` | Promote ragged-M AQ4 WMMA dispatch |
| `075c7f6` | Register AQ4 ragged-M WMMA startup guard |
| `a864ef5` | Add group8 ragged-M AQ4 WMMA prototype |
| `e6d8139` | Promote group8 ragged-M AQ4 WMMA dispatch |
| `cb5e74c` | Track group8 ragged-M WMMA probe checkpoint |

### Decode path (19)

| Commit | Subject |
|---|---|
| `b7b1e28` | Add AQ4 decode step profiling diagnostic |
| `457a480` | prototype AQ4 M1 wide-load matvec |
| `5be2525` | fix AQ4 wide-load packed-byte traversal |
| `0451b06` | complete AQ4 fused wide-load prototypes |
| `ea99359` | test: relax AQ4 fused wide-load differential tolerance |
| `f746627` | Promote AQ4 M1 wide-load matvec |
| `6fbf4dd` | Prototype AQ4 matvec add wide loads |
| `a85305e` | Promote AQ4 matvec add wide loads |
| `8e53b16` | Prototype AQ4 SiLU-mul shuffle reduction |
| `76cfa76` | Promote AQ4 SiLU-mul shuffle reduction |
| `e044391` | Prototype AQ4 QKV shuffle reduction |
| `c747f3f` | Promote AQ4 QKV shuffle reduction |
| `ac9b71a` | Prototype AQ4 triple/rmsnorm/qkv-prepare shuffle reduction |
| `6df3680` | Promote AQ4 triple/rmsnorm/qkv-prepare shuffle reduction |
| `815b9a4` | Prototype AQ4 segmented RMSNorm SiLU-mul shuffle reduction |
| `6c55f7b` | Promote AQ4 segmented RMSNorm SiLU-mul shuffle reduction |
| `3521d5c` | Prototype AQ4 matvec-add shuffle reduction |
| `27b246d` | Promote AQ4 matvec-add shuffle reduction |
| `c4c9a9b` | Promote AQ4 lm_head matvec_f32 shuffle reduction |

The candidate has the corresponding 36 required HIP guards: the active production manifest has
30 and the six new ones are recorded in `../performance/measurement-summary.md`.  The worker's
sorted guard contract and the manifest list were compared before staging.

## AQ4_0-reachable changes intentionally excluded

| Commit(s) | Finding | Candidate decision |
|---|---|---|
| `82d3658`, `7c888c6`, `90869be` | SQ8 v2 shared worker/reasoning runtime changes are reachable through common worker contracts; `90869be` also constrains reasoning delimiters.  They are not purely SQ8_0-only. | Excluded by the P3 base.  The active AQ4_0 profile's single-token delimiters are compatible, but this was not made a new deployment dependency. |
| `b21b2723` | Config-driven loader validation changes `model_config.rs` and the Qwen3.5/AQ4 load path. | Excluded deliberately to keep a P3-only binary.  Independent check of the served Qwen3.5-9B `config.json` found 32 layers and the expected repeating `linear_attention, linear_attention, linear_attention, full_attention` pattern. |
| `b3d78b42` | Captures Qwen3 config-trace evidence; it is a test/evidence follow-up to `b21b2723`, not a further AQ4_0 runtime implementation. | Excluded; BF's successful AQ4_0 config/layer-pattern execution is corroborating evidence, not a promotion gate. |
| `0a2a67d0` | AQ4 runtime source formatting only. | Excluded; no semantic/runtime contract change. |
| `473d987`, `850cbdb`, `719302f`, `5aeda9e`, `7cf4da9`, `f283f3d`, `9e7c1da`, `0455b119` | SQ8_0 paged-decode experiments, gates, and tile work after P3. | Excluded.  These do not enter the detached P3 source tree or staged AQ4_0 worker. |

Thus the premise that every SQ8_0 change is irrelevant is not literally true: the three v2 shared
runtime commits above can affect an AQ4_0 worker when adopting later shared `HEAD`.  They are
irrelevant to this release because the chosen `c4c9a9b` source predates and excludes them.

## Control-plane changes used but not compiled into the candidate

`ebf9a545` changes the promotion policy and `d33b6772`, `03324f3a`, `bcc70752`, and `0a601d0d`
provide the generic lightweight promotion and bounded start-limit recovery route.  They do not
alter the detached P3 worker source.  The current generic tools are used for promotion; no
candidate-specific promotion apparatus was added.

## Build and package binding

The candidate worker and two timing binaries were built from the selected detached worktree with
`CARGO_BUILD_JOBS=16`.  Worker SHA-256 is
`ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265`.
The candidate uses the same protected Qwen3.5-9B AQ4 package manifest as active production:
`a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`.
