# SQ8 P6 Full-Model M=8 Promotion

## Outcome

The Qwen3-14B-FP8 SQ8 full-model prefill path passed all frozen P6 promotion gates on the isolated Radeon AI PRO R9700. The promoted result is `full-model-m8-final.json` (SHA-256 `bcc7355ed9deed102ae620bf237466d2aebd78b897dd776fcb088377d4241647`) at source commit `27e3bd242f15334f7f8af6d61af68942cd24de84`.

The fixed workload is one M=8 prefill stack with token IDs `[1,2,3,4,5,6,7,8]` and positions `[0,1,2,3,4,5,6,7]`. The model revision is `9a283b4a5efbc09ce247e0ae5b02b744739e525a`.

## Frozen Gates And Numerical Results

All 40 optimized GPU layer boundaries were finite. Against the independent CPU SQ8 oracle, every layer passed relL2 `<= 0.10` and cosine `>= 0.995`; layer 39 also passed the tighter relL2 `<= 0.08` and cosine `>= 0.997` gates. The worst case was layer 39 at relL2 `0.030492298862366745` and cosine `0.9995673845541831`. As a diagnostic comparison to the source vLLM layer boundaries, the worst case was also layer 39 at relL2 `0.06655599993460926` and cosine `0.9985139105064265`.

The final head was finite and matched its CPU oracle on the same SQ8 hidden state:

| Comparison | relL2 | Cosine | Frozen gate |
| --- | ---: | ---: | --- |
| Device vs CPU final normalized hidden | `1.0960481842317075e-7` | `0.9999999999999909` | relL2 `<= 0.002`, cosine `>= 0.999999` |
| Device vs CPU logits | `2.060008209116002e-7` | `0.9999999999999016` | relL2 `<= 0.002`, cosine `>= 0.999999` |
| Device vs vLLM final normalized hidden | `0.04253753323548277` | `0.9991131536875918` | relL2 `<= 0.15`, cosine `>= 0.99` |
| Device vs vLLM logits | `0.041190031517998854` | `0.9992311957756344` | relL2 `<= 0.15`, cosine `>= 0.99` |

The final token top-1 was `353` for the device, CPU head oracle, and vLLM source oracle. The device top-10 IDs were `[353,3764,25010,220,5572,671,3014,374,368,262]`; the vLLM top-10 IDs were `[353,3764,25010,220,5572,671,3014,374,262,16]`. Their overlap was 9, above the frozen minimum of 5.

## Execution Contract

Each timed sample performed exactly one M=8 stack invocation with 40 layers, 280 projections, 160 activation quantizations, and 40 layer-to-layer D2D copies. It used one final stack synchronization, then one head D2D copy, one RMSNorm call, one BF16 matrix-vector call, two result readbacks, and one head synchronization. The timed path used neither fallback nor host staging.

The 280 measured CK dispatches were `mem_v1_default_tile_16x128x128` x160, `mem_v1_default_tile_16x128x256` x40, and `mem_v1_kpadding_tile_16x128x256` x80. The separate non-timed audit performed 40 host readbacks. Across audit, warmup, and measurement, the run recorded 14 fresh uploads, 14 input-ready checks, 14 output-ready checks, 14 validated stack reports, 14 validated head reports, and 13 timed-output hash stability checks.

## Timing

Timing used 3 warmups and 10 measured repetitions. Input upload and inter-stage host validation were excluded. Stack timing included its final synchronization; head timing included readback, decode, and validation.

| Region | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| Full stack and head | `34.1816625` | `34.235882600000004` |
| 40-layer stack | `29.987294999999996` | `30.03552355` |
| Final head | `4.187013` | `4.232364` |

## Resident VRAM Accounting

The device reported `34208743424` bytes of VRAM. The minimum accounted resident footprint was `14775996928` bytes: artifact weights and scales `13213670400`, layer norms `1679360`, shared stack workspace `3989504`, resident stack hidden state `163840`, and model head `1556493824`. The remaining unaccounted capacity was `19432746496` bytes. This accounting excludes allocator and backend overhead, and the result reported `fits_device=true`.

## Reproducibility And Identity

The initial fail-closed run rejected a runtime identity with an empty `gcn_arch_name`; its log SHA-256 is `a56ccb47e5a80f360d2ace9cc608fa164be7edac1b63fda7b1386e7a8e6a43cd`. After the R9700 identity checks were unified, two independent successful runs produced result files with SHA-256 `f5656dfd85fbc6fafa7cf9790e7d5d60bde7e326a1bf14e9f73b80fec712725b` and `bcc7355ed9deed102ae620bf237466d2aebd78b897dd776fcb088377d4241647`. After removing only `.timing` and `.cpu_oracle.elapsed_ms`, the two JSON documents were byte-identical in sorted form with SHA-256 `cb88838d4a698ca75ba1aeec2668c957491248a5048758b61edb5df326607210`.

The promotion log is `run-02-final.log` with SHA-256 `23b3947aca23a501a05ab76b96758b94ec74eab570e13c373e2f3029ca9f3547`. Full host, device, artifact, package, oracle, guard, and command provenance is recorded in `environment.json`.
