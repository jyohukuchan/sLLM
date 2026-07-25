# SQ8_1 Format Design Input v0.1

> Status: design input only. `SQ8_1` is not added to the public format-ID registry, an artifact,
> runtime dispatch, candidate, campaign, release, service, or activation path by this document.

## 前回の要点

- `SQ9_0` は V620/gfx1030 の M=1 実測で `SQ8_0` 比最大 +6.069% に留まり、必要な
  +7.29% に達しなかった。さらに実重みと gfx1030 ISA の比較で、signed int8
  block-scale が容量・W8A8・W8A16・重み誤差の各軸で優位だったため、`SQ9_0` は破棄済みである。
- `SQ8_0` は FP8 配布元の E4M3 payload と BF16 [128,128] scale をそのまま保持する。
  gfx1201 の native FP8 経路では、再量子化しない `SQ8_0` を維持する。
- `AQ4_0` は既存の compact format であり、この設計では変更しない。
- `SQ8_1` の W8A8 は活性量子化誤差が未確認だった。本書の CPU 実測でその条件を評価した。

## 今回の変更点

- 既存 `tools/collect-activation-stats.py` の model loader、corpus parser、Linear pre-hook
  命名規則を再利用した CPU-only 測定を追加し、実 Qwen3.5-9B の生 Linear 入力を hook 内で
  量子化した。GPU API・R9700・V620 は使用していない。
- K=32、signed symmetric int8、FP16 scale の上向き丸めで、81,788,928 activation 値の
  relative L2 は 0.00994415、true clipping は 0、sampled W8A8 linear-output relative L2 は
  0.00775109 だった。W8A8 を `SQ8_1` の条件付き primary execution path とする根拠が得られた。
- scale を通常の RNE で FP16 保存すると K=32 で 1.594585% の値が保存後 scale に対して
  clip されることを確認した。payload byte 数を増やさず clip を避けるため、scale は
  「最小の `>= raw_scale` の FP16 値」へ上向き丸めることを確定する。
- offline compiler recheck で gfx1030 と gfx942 は `v_dot4c_i32_i8` を出力した一方、gfx1201 は
  `__builtin_amdgcn_sdot4` を `dot1-insts` feature 不在として拒否した。gfx1201 は `SQ8_0` を
  選ぶ規則を明文化する。

## 次の行動

1. 本書の payload/scale/tail contract を満たす bounded-memory CPU reference quantizer を作る。
2. reader と CPU reference を先に検証し、その後に gfx1030/gfx942 向け W8A16、W8A8 kernel を
   段階的に追加する。
3. W8A8 の full weight-plus-activation model logits は未確認なので、artifact/runtime 採用前に
   held-out corpus と BF16 reference で gate を実施する。

## Goal

`SQ8_1` を、row-major signed int8 payload と contiguous K=32 FP16 block scale による、
dot4-capable GPU 向け high-fidelity format として設計確定する。W8A8 は dynamic per-token
activation quantization と int32 block dot を用いる。W8A16 は品質または kernel admission が
W8A8 を許さない場合の必須 fallback とする。

この Goal は implementation-ready semantics を確定することであり、production adoption を決める
ことではない。

## Success Criteria

1. `SQ8_1` の logical value、scale rounding、payload/scale layout、tail accounting、W8A8/W8A16
   equations が曖昧さなく定義される。
2. K=16/32/64/128、FP16/BF16 scale、outlier strata、weight-plus-activation linear output を
   実モデル活性で比較し、K=32 と FP16 を根拠付きで選ぶ。
3. W8A8 の成立判断を、生 activation、sampled linear output、activation-only logit smoke の
   結果と限界を分けて記録する。
4. gfx1030、gfx942、gfx1201 の dispatch rule と、`SQ8_0` / `AQ4_0` との位置関係を明記する。
5. quantizer、reader、GPU kernel、quality/performance validation の実装順と各 admission gate を
   Next Actions に残す。

## Non-Goals

- `SQ8_0`、`AQ4_0`、既存 release/candidate/campaign/authorization、`/opt/ullm`、service、
  active manifest を変更しない。
- `SQ8_1` の production quantizer、reader、GPU kernel、artifact、registry entry を実装しない。
- R9700/gfx1201 で実行しない。V620/gfx1030 でもこの task では実行しない。
- 現測定を full-model W8A8 quality、GPU throughput、thermal safety、または promotion の証明と
  扱わない。

## Confirmed Inputs And Evidence Boundary

### CPU activation measurement

[`benchmarks/results/2026-07-26/sq8_1-w8a8-activation-error/`](../../benchmarks/results/2026-07-26/sq8_1-w8a8-activation-error/)
contains the raw-result aggregates, exact command, model/corpus hashes, and tool hashes.

| item | value |
| --- | --- |
| model | local Qwen/Qwen3.5-9B BF16 source checkpoint |
| corpus | frozen importance-score `D_stats-shard-00.jsonl`, eight chat records |
| forward coverage | 8 forwards, 962 valid tokens, sequence cap 128 |
| hooked modules | 248 selected `torch.nn.Linear` inputs |
| raw activation handling | real pre-hook values are quantized in process then discarded; no raw activation file is retained |
| activation error sample | 8 deterministic rows per module call = 64 raw token rows/tensor; 81,788,928 values total |
| output sample | 16 evenly spaced output rows/tensor; 253,952 sampled matrix-output values total |
| device | CPU-only PyTorch; `CUDA_VISIBLE_DEVICES=''`, `HIP_VISIBLE_DEVICES=''`; no GPU execution |

This is a real-model measurement, but it is not a held-out full-model quality result. The corpus is
one frozen shard, activation rows are sampled, and the full W8A8 model was not constructed.

### Earlier weight and ISA evidence

The rejected `SQ9_0` document contains independent source-correct evidence for signed int8 K=32
weights: on a 990,904,320-weight Qwen3-14B FP8 reconstruction sample, K=32 FP16-scale int8 had
relative L2 0.00562448658, compared with 0.00696264838 for K=128. That is weight evidence from a
different model/source contract; it is not combined statistically with the Qwen3.5 activation run.

The new [offline ISA recheck](../../benchmarks/results/2026-07-26/sq8_1-w8a8-activation-error/static-isa-recheck.md)
uses `--offload-device-only` only. It shows 32 dot4 instructions for the fixed-K=128 probe on
gfx1030 and gfx942, and a compiler feature error on gfx1201. It is ISA eligibility evidence, not
GPU performance evidence.

## Working Hypotheses

1. **H1 — W8A8 is viable only if activation errors are bounded on real model tensors.** Confirmed
   only at the sampled linear-output and activation-only-logit level below; full-model W8A8 remains
   unconfirmed.
2. **H2 — K=32 is the best initial balance.** It costs 8.5 bpp, exposes eight dot4 operations per
   block, and substantially limits the outlier scope relative to K=64/128.
3. **H3 — FP16 is the right scale storage type for `SQ8_1`.** It provides more mantissa precision
   than BF16 in the measured scale range. A positive upward rounding rule is required to prevent
   storage-rounding clipping.
4. **H4 — symmetric codes are required for the primary dot4 path.** An affine zero point would add
   per-block correction work and metadata; its possible quality benefit is unmeasured, so it is not
   silently claimed away.
5. **H5 — `SQ8_0` remains the correct gfx1201 selection where a source FP8 artifact exists.** It
   avoids re-quantization and the compiler does not admit the proposed sdot4 path on gfx1201.

## W8A8 Activation Measurement Result

### Scale and block-size ablation

The following values use upward-rounded stored scales, which give zero true clipping. bpp is the
full-block weight density, before container/header and physical tail padding.

| K | FP16 relative L2 | BF16 relative L2 | FP16 true clipping | weight bpp | decision signal |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.00725145 | 0.00748306 | 0 | 9.000 | lower error, but doubles K=32 scale metadata/work |
| 32 | 0.00994415 | 0.01011369 | 0 | 8.500 | selected |
| 64 | 0.01358876 | 0.01372090 | 0 | 8.250 | 36.7% higher FP16 error than K=32 for 2.94% fewer bytes |
| 128 | 0.01832550 | 0.01843859 | 0 | 8.125 | 84.3% higher FP16 error than K=32 for 4.41% fewer bytes |

At K=32, ordinary FP16 RNE storage has relative L2 0.00993614 but a 0.01594585 true-clipping
rate; BF16 RNE has 0.01817686 true clipping. FP16-upward changes the K=32 FP16 result only to
0.00994415 while reducing that rate to zero. No positive scale underflowed to zero or became
non-finite in any measured encoding.

The result supports K=32 rather than K=16 because K=16 gains 0.00269270 absolute relative-L2 while
paying 0.5 bpp and twice the scale products. K=64 and K=128 save only 0.25/0.375 bpp from K=32 but
materially worsen both activation and prior independent weight error.

### Matrix-output and logit evidence

| scope | relative L2 | maximum absolute error | qualification |
| --- | ---: | ---: | --- |
| activation only, sampled linear outputs | 0.00646202 | 0.0828502 | BF16 weights, K=32 FP16-upward activations |
| W8A16, sampled linear outputs | 0.00426766 | 0.0540940 | K=32 FP16-upward int8 weights, BF16 activations |
| W8A8, sampled linear outputs | 0.00775109 | 0.102548 | int32 block dots, K=32 FP16-upward weights and activations |
| activation-only final 16 token logits, one prompt | 0.01401899 | 0.492188 | mean KL 0.000323955; top-1 16/16 matches |

The logit smoke quantizes all selected Linear inputs but retains BF16 weights. It is useful evidence
that activation quantization alone did not disrupt the sampled final-token ranking; it is **not** a
W8A8 end-to-end logit result. Combined weight-plus-activation full-model logits, generated-token
behavior, and held-out behavior are unconfirmed.

### Outlier impact

The [derived outlier analysis](../../benchmarks/results/2026-07-26/sq8_1-w8a8-activation-error/outlier-analysis.md)
reports the raw distribution. Across 2,555,904 observed K=32 blocks, 79.5194% are in
`max(abs)/RMS` [2,4), 17.1521% are in [4,8), and none reaches [8,∞). The median per-tensor
relative L2 rises from 0.00653891 in [2,4) to 0.01163607 in [4,8), so outliers have a visible local
effect but are confined to K=32 blocks. The maximum observed block ratio is 5.65683.

Transient channel spikes are visible especially in `linear_attn.out_proj`, `mlp.down_proj`, and
`self_attn.o_proj`. The 64-token channel statistic cannot establish a population tail, so no
per-channel/outlier escape path is justified by this result alone.

### W8A8 disposition

**Decision: `SQ8_1` W8A8 is accepted as a design and implementation candidate; it is not accepted
as a runtime/artifact/release candidate.** The measured K=32 error, zero clipping under the chosen
scale rule, sampled W8A8 output error, and activation-only logit smoke are sufficient to proceed to
a reference implementation. W8A16 remains mandatory until the full W8A8 model-level gate passes.

## Exact `SQ8_1` Value Definition

For a logical matrix `W` with shape `[rows, cols]`, define `G = ceil(cols / 32)`. The logical scale
shape is exactly `[rows, G]`.

For each real K=32 block `b` of a weight row:

```text
raw_s_w[r, b] = max(abs(W[r, 32*b : min(32*b + 32, cols)])) / 127
s_w[r, b]     = ceil_fp16(raw_s_w[r, b])
q_w[r, k]     = clamp(RNE(W[r, k] / s_w[r, floor(k/32)]), -127, 127)
W_hat[r, k]   = q_w[r, k] * s_w[r, floor(k/32)]
```

`ceil_fp16(x)` is the smallest finite positive IEEE binary16 value greater than or equal to `x`.
For an all-zero logical block, `s_w=1.0` and all codes are zero. If a nonzero raw scale cannot be
represented as a finite positive FP16 value, the quantizer must fail this tensor or explicitly
route it to a declared fallback; it must not emit an infinite or zero scale.

For one token activation vector `a`, use the same equation independently for each contiguous K=32
block to obtain `q_a` and `s_a`. Codes use RNE/ties-to-even and the signed range [-127, 127]; -128
is intentionally not emitted. Padding values exist only for physical tails and are zero.

For each output row and block:

```text
d[r, b] = sum_{j=0..31} int32(q_w[r, 32*b+j]) * int32(q_a[32*b+j])
y[r]    = sum_b float(d[r,b]) * s_w[r,b] * s_a[b]
```

The `s_w * s_a` product is applied to each K=32 partial, before combining blocks. A K=32 partial
is bounded by `32 * 127 * 127 = 516,128`, so its signed int32 dot is safe.

## 設計論点と決定

### Block granularity — choose K=32

K=32 is the initial `SQ8_1` block size. It is a multiple of dot4 packing, places eight dot4 MAC
instructions in a block, costs exactly 0.5 scale bpp, and bounds outlier influence to 32 channels.
The activation ablation and earlier independent real-weight ablation both reject K=64/128 as the
initial trade-off. K=16 is retained only as a diagnostic ablation, not a format variant.

### Scale type and rounding — choose FP16 upward

FP16 is selected, not BF16. Both cost 16 bits, but FP16’s 10-bit significand gives lower measured
activation error at every tested K. Its range was sufficient in the measured model: no stored
positive scale underflow/overflow occurred. The scale storage operation is explicitly upward, not
ordinary RNE, because the latter causes measurable clipping after a max-derived scale rounds down.

### Symmetric rather than affine quantization

The base format is symmetric and has no zero point. This is a performance/format decision, not a
claim that affine quantization cannot reduce error for an asymmetric activation tensor.

With symmetric weights but an activation zero point `z_a`, the dot must become:

```text
sum(q_w * (q_a - z_a)) = dot4(q_w, q_a) - z_a * sum(q_w)
```

Thus each weight K=32 block needs a retained `sum(q_w)` (up to +/-4,064, requiring at least an
int16) plus a correction multiply/subtract; that is another 2 bytes per weight block, or 0.5 bpp,
before the dynamic activation zero point. Affine weights add the `sum(q_a)` and `K*z_w*z_a` terms.
These terms preserve mathematical correctness but weaken the simple direct-dot loop and the 8.5 bpp
contract. No affine quality ablation was run; it remains a later, separately justified option only
if the required model-level W8A8 gate fails.

### Payload packing and 128-bit wide loads

The selected layout has separate aligned payload and scale planes. It avoids an unaligned 34-byte
interleaved record while retaining the exact logical scale shape.

```text
header:
  rows, cols, group_size=32, scale_dtype=FP16, payload_row_stride,
  scale_shape=[rows, ceil(cols/32)], little_endian

payload plane:
  signed int8 logical row-major q_w values
  payload base and every row start are 16-byte aligned
  payload_row_stride = round_up(cols, 16); row-end bytes are zero padding

scale plane:
  little-endian FP16 s_w[rows][ceil(cols/32)] in logical row-major order
```

For every full K=32 block, the payload offset is `row * payload_row_stride + block * 32`; it is
16-byte aligned. The kernel issues two aligned `uint4` (128-bit) loads for 32 int8 values, packs
eight 32-bit dot4 operands, and separately reads the 2-byte scale. For the normal target projection
widths that are multiples of 32, this is exactly 32 payload bytes + 2 scale bytes = 8.5 bpp.

| layout | full-block density | wide-load property | decision |
| --- | ---: | --- | --- |
| interleaved `[i8[32], f16]` 34-byte record | 8.5 bpp | the next record is not 16-byte aligned | reject |
| padded 48-byte record per K=32 block | 12 bpp | aligned but wastes 3.5 bpp | reject |
| selected separate payload/scale planes | 8.5 bpp | two aligned `uint4` payload loads/block | adopt |
| interleaved K=256 superblock, 256 bytes + 8 FP16 scales = 272 bytes | 8.5 bpp | 17 `uint4` loads/superblock and scale locality | valid future benchmark alternative; not v0.1 layout |

For a tail, only valid logical values determine the last scale. Full K=32 blocks retain the two
wide loads; the final partial block uses guarded byte/dword loads and zero fill. Exact physical bpp
is `8 * (payload_row_stride + 2*ceil(cols/32)) / cols`, and the artifact manifest must report that
actual value rather than always claiming 8.5 bpp. The CPU reference must test 1/15/16/17/31/32/33
columns and reject any nonzero padding read.

### W8A8 and W8A16 policy

`SQ8_1` uses W8A8 only where all of the following hold:

1. the architecture dispatch admits the dot4 kernel;
2. the artifact/reader has passed exact pack/unpack and tail validation;
3. the target model’s predeclared full-model W8A8 quality gate passes;
4. FP16 scale validation finds no zero/non-finite positive scale; and
5. the selected kernel supports the shape and tail.

Otherwise, a dot4-capable architecture uses `SQ8_1` W8A16: int8-to-float conversion, multiplication
by `s_w` per K=32 block/value as implemented by the reference, and BF16/FP16 activations. W8A16 is
not a quality-free substitute; it still includes weight quantization error, but it removes the
dynamic activation error observed here.

### Rounding and error compensation

- Codes: deterministic RNE/ties-to-even, saturating to [-127,127].
- Scales: positive FP16 upward rounding as defined above. Quantizer reports raw/stored min/max,
  positive-scale underflow, overflow, and post-storage clipping counts.
- Base payload: no learned or hidden compensation.
- Existing optional `row_scale_overrides` semantics from
  [`quantizer-row-compensation-plan-v0.1.md`](quantizer-row-compensation-plan-v0.1.md) remain
  format-external manifest metadata. If an approved override applies to a complete output row, a
  W8A8 kernel may multiply the completed row sum once; it must not pre-scale the stored codes or
  silently alter the 8.5 bpp payload definition.
- A SmoothQuant-style diagonal transform or sparse outlier side path is not in the base format. It
  requires a separately frozen calibration/held-out evaluation and becomes admissible only after
  demonstrating a full-model W8A8 failure that the base K=32 path cannot meet.

## Architecture Selection Rule

| architecture | evidence | `SQ8_1` rule | `SQ8_0` rule |
| --- | --- | --- | --- |
| gfx1030 (V620) | static probe emits 32 `v_dot4c_i32_i8`; prior V620 evidence identifies FP8 fallback conversion cost | primary W8A8 candidate after CPU and thermal-guarded GPU gates; W8A16 required fallback | retained only for existing source/artifact comparison; not preferred for new int8-dot path |
| gfx942 (CDNA3) | static probe emits 32 `v_dot4c_i32_i8_e32` | W8A8 candidate pending device-specific quality/performance validation; W8A16 fallback | no replacement claim; select only if its own native path is independently admitted |
| gfx1201 (RDNA4/R9700) | ROCm 7.2.1 device-only compilation rejects `__builtin_amdgcn_sdot4` for missing `dot1-insts`; native FP8 route exists | do not dispatch `SQ8_1` W8A8 in v0.1 | choose `SQ8_0` when the source FP8 artifact is available |
| other architecture | unconfirmed | no `SQ8_1` direct dispatch in v0.1 | existing per-format rules remain unchanged |

For prefill/batch, int8 dot4 still has a structural arithmetic advantage: one dot4 instruction
performs four int8 MACs, whereas the prior packed FP16 comparison used two MACs/instruction. This
is a reason to prioritize `SQ8_1` W8A8 batch measurement after correctness; it is not a throughput
claim until target-GPU measurement exists.

## Position Relative To Existing Formats

| format | persistent density | decode/compute form | quality position | expected hardware position |
| --- | ---: | --- | --- | --- |
| `AQ4_0` | 4-bit index payload; group/tensor metadata makes exact effective bpp policy-dependent | nibble unpack, codebook lookup, scale handling | calibrated existing compact path; not re-evaluated here | existing compact path; unchanged |
| `SQ8_0` | 8.0009765625 bpp for the documented FP8 payload + BF16 [128,128] source scale contract | native FP8 conversion/WMMA where implemented; fallback conversion elsewhere | no additional requantization relative to its FP8 distributed source | preferred gfx1201/RDNA4 source-FP8 route |
| `SQ8_1` | 8.5 bpp for full K=32 blocks; exact tail bpp manifest-reported | W8A8 int32 dot4 + two scales/block; W8A16 conversion fallback | K=32 activation and sampled linear output evidence above; full W8A8 model quality unconfirmed | gfx1030/gfx942 dot4 candidate; not gfx1201 v0.1 |

`SQ8_1` is a separate, wire-incompatible artifact family. It is neither a replacement for `SQ8_0`
on native FP8 paths nor a change to `AQ4_0`.

## Decision Tree

```text
artifact and target GPU selected
  |
  +-- gfx1201 and compatible source FP8 SQ8_0 artifact?
  |     +-- yes -> SQ8_0 native FP8 dispatch
  |     +-- no  -> no SQ8_1 W8A8 dispatch in v0.1; use existing declared path
  |
  +-- gfx1030 or gfx942 with validated SQ8_1 reader/kernel?
        +-- no  -> existing declared path; do not infer a fallback
        +-- yes -> predeclared W8A8 quality gate passed, finite FP16 scales, supported tail?
              +-- yes -> SQ8_1 W8A8
              +-- no  -> SQ8_1 W8A16 if its independent weight-quality gate passes
                           otherwise declared non-SQ8_1 fallback
```

## Risks

| risk | impact | handling |
| --- | --- | --- |
| 8-record activation corpus and 64 rows/tensor under-sample rare outliers | W8A8 quality appears better than deployment distribution | require larger, held-out, domain-stratified activation/logit gate before adoption |
| activation-only logit smoke leaves quantized-weight interaction unmeasured | W8A8 full-model quality is unknown | build CPU fake-quant reference and run full weight-plus-activation logits before any artifact admission |
| upward FP16 scale changes code assignment versus RNE | tiny extra quantization error | preserve the explicit `ceil_fp16` rule and test CPU/GPU bit agreement |
| physical row tails exceed nominal density | artifact size/performance surprise | report exact bytes; test tails; do not advertise 8.5 bpp for a tail-bearing tensor |
| affine quantization may help a later asymmetric tensor | symmetric choice may leave quality on table | keep it out of base format; only measure it after a documented quality failure because correction metadata/work changes the contract |
| gfx942 static dot4 eligibility is mistaken for performance | premature hardware claim | require target-GPU disassembly, numerical, and timing data before dispatch promotion |
| V620 passive cooling | hardware risk during later validation | use its existing PCI-BDF-to-own-hwmon thermal guard; abort at junction >=85 C |
| row compensation silently alters semantics | fidelity/debug ambiguity | retain only explicit optional manifest metadata, default absent, and validate separately |

## Next Actions

1. **CPU reference quantizer.** Implement streaming quantize/pack/unpack for `SQ8_1`, including
   FP16-upward scales, logical vs physical tail accounting, 1/15/16/17/31/32/33-column tests,
   deterministic RNE, zero/non-finite scale rejection, and byte-for-byte manifest accounting.
   Admission: pack/unpack equation and scale/clipping counters match the reference exactly.
2. **Artifact reader.** Add a new isolated schema/reader only after the reference is stable. It must
   reject a wrong group size, scale count, row stride, endian tag, payload length, checksum, or
   nonzero padding. Admission: old `SQ8_0` and `AQ4_0` artifacts remain untouched and all malformed
   `SQ8_1` fixtures fail closed.
3. **Numerical quality gate.** Build a CPU fake-quant full-model path and compare W8A16 and W8A8
   against BF16 on a larger held-out, stratified corpus. Record logits, KL, top-k, generated tokens,
   outlier-bin coverage, and the separate impact of weights versus activations. The pass threshold
   must be declared before that run; this document does not retroactively invent one.
4. **Kernel sequence.** First implement a W8A16 numerical reference kernel, then a K=32 W8A8
   int32-dot kernel with two aligned `uint4` payload loads/block and one FP16 scale product/block.
   Compile/disassemble each for gfx1030 and gfx942. Admission: CPU/GPU differential, tail guards,
   and compiler evidence of the intended dot instruction all pass.
5. **GPU validation.** Under separately authorized windows only, measure W8A8/W8A16/`SQ8_0` on
   matched workloads. For V620, select `0000:03:00.0` through `hipDeviceGetPCIBusId`, read its own
   junction `temp2_input`, sample during runs, and abort/cool down at >=85 C. Measure M=1 and
   prefill/batch separately; do not extrapolate either direction from the other.
6. **Dispatch admission.** Enable gfx1030/gfx942 selection only after numerical and performance
   gates. Keep gfx1201 on `SQ8_0` until a separately verified int8 route exists; do not change
   `SQ8_0` or `AQ4_0` release/campaign state in this work.
