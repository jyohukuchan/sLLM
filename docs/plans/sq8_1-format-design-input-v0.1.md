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
5. **GPU validation.** In separately scheduled shared-resource windows, measure W8A8/W8A16/`SQ8_0` on
   matched workloads. For V620, select `0000:03:00.0` through `hipDeviceGetPCIBusId`, read its own
   junction `temp2_input`, sample during runs, and abort/cool down at >=85 C. Measure M=1 and
   prefill/batch separately; do not extrapolate either direction from the other.
6. **Dispatch admission.** Enable gfx1030/gfx942 selection only after numerical and performance
   gates. Keep gfx1201 on `SQ8_0` until a separately verified int8 route exists; do not change
   `SQ8_0` or `AQ4_0` release/campaign state in this work.

## 実装・検証状況（2026-07-26）

設計本文を置き換えず、reference implementation の到達点だけを追記する。

- `tools/sq8_1_artifact.py` と `tools/build-sq8_1-artifact.py` は、検証済み canonical
  `SQ8_0` v0.2 を row-by-row F32 に再構成して、別 namespace の `SQ8_1` artifact を生成する。
  K=32、signed `[-127,127]`、zero-point なし、`ceil_fp16`、payload row stride
  `round_up(cols,16)`、separate F16 scale plane、zero tail を packer/reader とも検証する。
  row compensation は format-external のままであり、payload に書き込まない。
- Rust reader は strict `SQ8_1` manifest だけを受け付け、legacy `sq` / `sq-fp8` aliases を
  `SQ8_0` のまま維持する。runtime には W8A16 default C ABI と explicit-only W8A8 C ABI を
  分離して追加した。W8A8 の暗黙 fallback/auto dispatch はない。
- CPU tests は Python packer 5/5、Rust reader/reference 4/4、CPU runtime 2/2、existing
  canonical SQ8_0 reader 14/14、format-ID/SQ8 policy Python tests 13/13 を通過した。Python
  artifact を Rust reader が cross-check し、実 Qwen3-14B source の 1024x5120 K projection
  （5,242,880 values）では reconstructed-SQ8_0 source に対する weight relative L2
  `0.005592543546739809`、max abs `0.0017452239990234375`、post-storage clipping 0 を記録した。
  この single-tensor measurement は既存の BF16/full-model gate の代替ではない。
- V620 differential は HIP BDF `0000:03:00.0` → `card0` → own junction `temp2_input` を確認し、
  85 °C guard 下で W8A16/W8A8 各8 launch を実行した。junction は 41–42 °C、CPU reference
  に対する relative L2 は W8A16 `6.076546605e-08`、W8A8 `4.333164297e-08` だった。
- offline static audit は runtime の HIPRTC source そのものを runtime whitelist 全五 target
  （gfx1030/gfx1100/gfx1201/gfx942/gfx950）で compile した。W8A8 は gfx1030 で
  `v_dot4c_i32_i8`、gfx1100/gfx1201 で `v_dot4_i32_iu8`、gfx942/gfx950 で
  `v_dot4c_i32_i8_e32` を出した。gfx1201 instruction には `neg_lo:[1,1,0]` があり、同 target
  の reference kernel は VGPR 53 / SGPR 59 / LDS 1024 B / private and spill 0 だった。これは
  gfx1201 の VOP3P spelling が signed/signed dot semantics を実装できるという実装証拠である。

Evidence は `benchmarks/results/2026-07-26/sq8_1/` に保存した。full-model W8A8 logits gate は
未確認・未通過のままなので、W8A16 は default reference path、W8A8 は explicit-only のままとする。
この実装は candidate/release/campaign/authorization/active manifest を変更していない。

## V620 カーネル最適化と `SQ8_0` 比較（2026-07-26）

reference implementation を置き換えずに、`SQ8_1` の HIPRTC kernel を次のように最適化した。

- W8A16 は 256 threads を eight logical wave32 rows として使い、row ごとの LDS tree reduction
  を wave shuffle に置換した。K=32 の payload は従来どおり aligned `uint4` 二回（32 B）であり、
  完全 block あたり 1/16 本の 128-bit payload-load / element である。
- explicit W8A8 は K=32 activation plane を eight output rows で一回だけ quantize して dynamic
  LDS に置き、`v_dot4*` 経路で共有する。5,120 columns では code 5,120 B + scale 160 × F32 =
  5,760 B で、runtime は 48 KiB の conservative cap を超える形状に fallback kernel を選ぶ。
  必要な dot は依然 8 dot4 / K=32 = 0.25 dot4 / element であり、短縮したのは dot 数ではなく、
  activation の scale/divide/round を eight rows 間で amortize した点である。1 output row あたりの
  activation quantization は K=32 ごとに 32 values から amortized 4 values となる（8×削減）。
- static device-only audit は runtime HIPRTC source そのものを全 whitelist target で compile した。
  gfx1030 W8A16 は fixed LDS 1,024 B -> 0 B、`s_barrier` 2 -> 0、spill 0 のまま、W8A8 は
  fixed LDS 1,024 B -> 0 B（上記 dynamic LDS）、barrier 2 -> 1、VGPR/SGPR 53/59 -> 39/32、
  spill 0 である。gfx1030 の W8A8 は `v_dot4c_i32_i8` を emit し、RDNA3/RDNA4 は
  `v_dot4_i32_iu8` signed-control path を emit する。実効 occupancy の profiler measurement は
  **未測定**であり、resource table 以上の occupancy 数値は主張しない。

V620 は `HIP_VISIBLE_DEVICES=2` で隔離し、実行時の `hipDeviceGetPCIBusId` で
`0000:03:00.0` を確認した後、その BDF に属する DRM `card0` の own junction
`hwmon5/temp2_input` を読むことを必須にした。R9700 は実行していない。Qwen3-14B FP8
`self_attn.q_proj`（5120 × 5120、M=1）を 32 warmups + 31 timed launches、three independent
runs、<=42 C cooldown で測った。次の値は各 run median の median である。

| format / path | median ms | modeled GB/s | 512 GB/s 比 | `SQ8_0` 比 |
| --- | ---: | ---: | ---: | ---: |
| `SQ8_0` E4M3 + F32 block-scale V620 fallback | 0.639007 | 41.034 | 8.014% | baseline |
| `SQ8_1` W8A16 wave32 × 8 rows | 0.237362 | 117.343 | 22.919% | **2.692× faster** |
| `SQ8_1` explicit W8A8 tiled wave32 × 8 rows | 0.249762 | 111.517 | 21.781% | **2.558× faster** |

従ってこの限定した V620 M=1 問いへの答えは **はい** である。`SQ8_1` の resident weights は
27,852,800 B で `SQ8_0` fallback の 26,220,800 B より 6.224% 多いので、勝因を payload サイズの
縮小と取り違えてはならない。比較対象 `SQ8_0` は
`benchmarks/results/2026-07-26/sq9-v620-viability/raw/final-m1-r{1,2,3}-card0-v4.jsonl` の既存
同一形状・同一 thermal/protocol evidence であり、一つの process 内で co-dispatch した A/B trace
ではない。この制限は結論に付随する。

CPU は 1/15/16/17/31/32/33/65 columns、signed endpoints、zero row、tail padding を含む
`SQ8_1` 3/3 tests を通過した。GPU full-shape pre-timing gate は W8A16 relative L2 0、W8A8
`2.331406575e-07`、K=65 runtime tail differential はそれぞれ `6.076546605e-08` と
`4.333164297e-08` で全て pass した。`SQ8_0` CPU regression、`SQ8_0`/`SQ8_1` artifact separation、
`AQ4_0` offline oracle も pass した。温度は final suite 全体で 41–43 C、85 C guard / cooldown timeout
は 0 件だった。

M>1/prefill、old reference `SQ8_1` の elapsed-time baseline、full-model W8A8 logits quality、
profiler-derived occupancy/DRAM transactions は **未測定**である。W8A8 は従来どおり explicit-only、
W8A16 は default のままであり、candidate/release/campaign/authorization/active manifest は変更して
いない。完全な raw/thermal/ISA evidence は
`benchmarks/results/2026-07-26/sq8_1-v620-optimization/` に保存した。

## `SQ8_0` 同等最適化後の公平比較と M sweep（2026-07-26）

### 過去の 2.692x の解釈訂正

上の historical table と測定値は書き換えない。ただし W8A16 の **2.692x** と W8A8 の
**2.558x** は、最適化済み `SQ8_1` と未最適化 `SQ8_0` fallback の別 process 比較だったため、
フォーマット差だけではなく最適化量の差を含んでいた。従って、それらを format-only speedup と
引用してはならない。

今回 `SQ8_0` の gfx1030-only generic body に、同じ 256-thread / eight wave32 reduction、aligned
`uint4` payload load、32 B LDS wave-partial handoff を加えた。scale boundary、unaligned payload、
tail は scalar fallback を維持し、公開 symbol/ABI/dispatch を変更していない。exact HIPRTC static
audit は direct kernel で fixed LDS 1024 B -> 32 B、`s_barrier` 2 -> 1、`ds_write` 2 -> 1、
`global_load_dwordx4` 0 -> 1、spill 0 の維持を確認した。最終 direct metadata は 31 VGPR / 48 SGPR /
32 B LDS である。`__launch_bounds__(256,2)` の isolated static prototype は 30 VGPR / 48 SGPR /
32 B LDS のままで追加の occupancy constraint を支持しなかった。profiler-derived occupancy と
DRAM transaction は依然 **未測定**である。

`#if defined(__gfx1030__)` の外側の legacy bodies は source hash で byte-stable に gate し、runtime
HIPRTC source を gfx1201 へ device-only compile した normalized disassembly / metadata は baseline
と `cmp=0` だった。R9700/gfx1201 は実行していない。

### 同一 process M=1 比較

V620/card0 を `HIP_VISIBLE_DEVICES=2`、`hipDeviceGetPCIBusId`=`0000:03:00.0`、同 BDF の own
junction `hwmon5/temp2_input` で固定した。Qwen3-14B `self_attn.q_proj` と同じ 5120 x 5120 shape の
deterministic common synthetic E4M3+block-scale source を用い、`SQ8_1` はその source を再量子化した。
これは shape/kernel の公平比較であり、actual-model throughput/quality result ではない。

各 run は <=42 C cooldown 後、32 warmups + 31 timed launches を `SQ8_0` / W8A16 / W8A8 の rotating
order で一つの process 内に co-dispatch した。run 3 の全三経路は start temperature が同程度でも
absolute latency が約 2.5x 小さい。原因は **未確認**であるため、absolute median を混ぜず、対応する
run 内 ratio の median を公平な主値にする。

| run | optimized `SQ8_0` ms | W8A16 ms | W8A8 ms | `SQ8_0` / W8A16 | `SQ8_0` / W8A8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.627807021 | 0.237481996 | 0.248962998 | 2.644x | 2.522x |
| 2 | 0.628367007 | 0.238682002 | 0.249202996 | 2.633x | 2.522x |
| 3 | 0.251682013 | 0.100920998 | 0.112040997 | 2.494x | 2.246x |
| paired-ratio median | — | — | — | **2.633x** | **2.522x** |

従って V620/gfx1030 のこの shape では、`SQ8_1` の優位は公平比較後も残る。ただし format-only 根拠は
W8A16 **2.633x**、W8A8 **2.522x** であり、historical 2.692x/2.558x をそのまま再利用しない。優位は
W8A16 で約 2.2%、W8A8 で約 1.4% 縮小した。これは gfx1201 の主張ではない。

pre-timing numerical gate は optimized `SQ8_0` direct 8 rows で max abs
`2.384185791e-07` / relative L2 `8.552191029e-07`、W8A16 で
`1.132488251e-06` / `3.767329717e-06`、W8A8 で 0 / 0、exact `SQ8_0` batch symbol (2 x 8 rows)
で `2.384185791e-07` / `3.869252278e-07` を全て pass した。

### M={1,8,32,128}

現行 runtime の exact direct API には batch ABI がないため、まず M independent direct matvec launches
を一つの event に束ねた。これは deployed kernel semantics の測定であり、fused GEMM/new ABI の主張では
ない。three cooldown-normalized runs の median-of-runs は次のとおりで、W8A8 は全点で W8A16 に負けた。

| M | W8A16 ms | W8A8 ms | W8A16 / W8A8 paired-ratio median |
| ---: | ---: | ---: | ---: |
| 1 | 0.237403005 | 0.253161997 | 0.938x |
| 8 | 0.423444003 | 0.483125001 | 0.876x |
| 32 | 1.691699028 | 1.868780017 | 0.905x |
| 128 | 7.008595943 | 7.329880238 | 0.958x |

したがって **現行 direct path では M=128 まで W8A8 の逆転はない**。M を増やすだけでは、この API の
activation plane を CTA 外へ hoist しない。

活性量子化 reuse 自体を分離するため、runtime source/ABI/dispatch に触れず benchmark-only HIPRTC source
で「input row ごとに一度だけ exact K=32 prequant、その後 2-D output grid」を実装した。2 batch x 全 5120
output rows の CPU differential は W8A16 max abs `2.205371857e-06` / relative L2
`3.989831230e-06`、W8A8 `1.788139343e-07` / `2.827435228e-07` で pass した。

| M | W8A16 prototype ms | W8A8 prequant prototype ms | W8A16 / W8A8 paired-ratio median |
| ---: | ---: | ---: | ---: |
| 1 | 0.237802997 | 0.168281004 | 1.415x |
| 8 | 1.562656999 | 0.460604996 | 3.393x |
| 32 | 1.309733987 | 0.591805995 | 2.214x |
| 128 | 5.224856853 | 2.095663071 | 2.493x |

この isolated prototype では W8A8 はすでに **M=1** から速く、sampled M>1 crossover は存在しない。
重要なのは M の数そのものではなく、activation quantization を eight-output-row CTA ごとの再実行から
全 output tile に共有可能な prequant plane へ hoist したことである。この prototype は production
prefill/batch path の performance potential を示すだけで、runtime ABI、dispatch、artifact/release
selection の admission ではない。

final suite の junction は全体で 40–54 C（M=1 co-dispatch 40–43 C、current direct M sweep 41–54 C、
prequant prototype 41–50 C）で、85 C guard と cooldown timeout は 0 件だった。指定した
M={1,8,32,128} はすべて完走し、熱で測れなかった項目はない。full-model W8A8 quality、production
batch/prefill implementation、profiler occupancy/DRAM traffic は熱以外の理由で未確認である。完全な
raw/thermal/summary/static evidence は
`benchmarks/results/2026-07-26/sq8_0-sq8_1-fair-comparison/` に保存した。


## SQ8_1 W8A8 full-model FP32 quality gate（2026-07-26）

### 判定

**W8A8 は採用不可（No-Go）** とする。`SQ8_1` の runtime/artifact/release
採用、または W8A8 prequant API への実装投資は開始しない。W8A16 はこの gate
で事前定義した fallback L2 条件を通過したため、必須 fallback / default reference
path のまま維持する。W8A8 は explicit-only の研究候補に留め、candidate、release、
campaign、authorization、active manifest は変更しない。

これは性能結果で相殺できない品質判定である。W8A8 は aggregate relative L2 / KL /
top-10 / W8A16 比の条件は満たした一方、事前に凍結した logits max abs、final hidden
max abs、greedy top-1 の三条件に独立して失格した。

### 凍結した契約と測定範囲

詳細な契約は
`benchmarks/results/2026-07-26/sq8_1-w8a8-full-model-gate/gate-criteria.md`、
機械可読結果は同 directory の `summary.json`、prompt 別値は
`per-prompt.jsonl`、層別値は `layer-metrics.json`、mismatch margin は
`top1-mismatches.jsonl` に保存した。

- Qwen3.5-9B の local BF16 source weights を CPU 上で FP32 に読み、unmodified
  FP32 Hugging Face forward を参照とした。R9700、V620、その他 GPU、サービスは未使用で
  あり、GPU 温度履歴は **N/A (CPU-only)** である。
- Primary scope は既存 SQ8_1 collector pattern の transformer projection 248
  Linear。weight は一度だけ SQ8_1 K=32 signed symmetric int8（`[-127,127]`、
  zero-point なし、RNE、upward-rounded FP16 scale）へ、W8A8 はその Linear
  入力も同じ規約へ動的量子化した。codes/scale は FP32 に再構成して同じ FP32
  `F.linear` boundary を通し、HIP accumulation order / throughput は主張しない。
- `lm_head` は primary では unmodified FP32、249th Linear を加えた
  all-Linear stress は別集計とした。
- frozen `D_stats-shard-00.jsonl` を deterministic evenly spaced に 20 records
  選び、chat/code/general/multilingual_ja/reasoning_math 各 4 records、4,243 valid
  scored positions を測った。v0.1 の 256-token cap は 3,568 positions で既存
  4,000-position coverage 条件を満たさず、raw evidence を
  `attempt-1-coverage-incomplete/` に保存して非適格とした。threshold を緩めず、
  同じ IDs の cap を 384 に拡張してから v0.2 の qualifying run を凍結した。
- control は logits / final hidden の relative L2 と max abs が全て `0.0` で、
  `1e-5` / `2e-5` の harness 条件を通過した。weight / activation は finite、
  post-storage clipping 0、code range `[-127,127]` だった。

事前合格条件は、W8A16 aggregate / worst-prompt logits relative L2
`<=0.040` / `<=0.060`、W8A8 aggregate / worst-prompt logits relative L2
`<=0.060` / `<=0.080`、logits max abs `<=1.0`、mean / worst-prompt KL
`<=0.005` / `<=0.010`、W8A16 比 `<=1.60` かつ `+0.020`、max layer L2
`<=0.080`、final hidden L2 / max abs `<=0.060` / `<=1.0`、top-10
`>=0.950`、reference top-1 in candidate top-10 100%、top-1 `>=99.0%` /
Wilson lower 95% `>=98.5%`、かつ全 swap が FP32 top-2 への margin
`<=0.050` であることだった。詳細な全条項は artifact の契約を正とする。

### Full-model 結果

| scope / candidate | logits rel L2 | logits max abs | mean KL | top-1 agreement | top-10 overlap | final hidden rel L2 / max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W8A16 primary | 0.016971283 | 13.834223 | 0.000399046 | 4,218/4,243 (99.410794%) | 99.066698% | 0.024402447 / 14.862573 |
| W8A8 primary | 0.023506802 | **7.889154** | 0.000665853 | **4,189/4,243 (98.727316%)** | 98.378506% | 0.031263589 / **13.696337** |
| W8A8 `outlier_bypass_ge4` diagnostic | 0.015238139 | 8.991999 | 0.000264083 | 4,208/4,243 (99.175112%) | 98.920575% | 0.020038844 / 3.448392 |
| all-Linear W8A8 stress | 0.024505412 | 7.878294 | 0.000727507 | 4,197/4,243 (98.915861%) | 98.286590% | 0.031263589 / 13.696337 |

W8A16 は fallback に事前指定した two L2 gates を通過した（aggregate
`0.016971283 <= 0.040`、worst prompt `0.056227554 <= 0.060`）。
W8A16 の表中の max abs / top-1 は比較情報であり、W8A16 を exact-equivalent と
主張するものではない。

W8A8 は W8A16 に対する incremental logits penalty ratio `1.385093`、absolute
delta `0.006535519`、final-hidden ratio `1.281166`、delta `0.006861142` を通過した。
しかし logits max abs `7.889154 > 1.0`、final hidden max abs
`13.696337 > 1.0`、top-1 `98.727316% < 99.0%`、Wilson lower
`98.343243% < 98.5%` であるため No-Go となった。KL、aggregate / prompt L2、
top-10、reference top-1 retention は pass であり、この結論は一つの単発 metric
だけには依存しない。

W8A8 の 54 top-1 mismatch のうち 38 は既知 AQ4 の許容と同じ、FP32 top-2
への near-margin swap（margin `<=0.050`）だった。しかし 16 は事前規則を満たさず、
そのため「near-margin quantization noise」で全体を許容することはできない。全 mismatch
margin の min / median / p90 / max はそれぞれ `0.000068665` / `0.024856567` /
`0.071977615` / `0.115995407` だった。reference top-1 は全 4,243 positions で
W8A8 top-10 に残ったが、これは greedy agreement gate の代替ではない。

### Hidden error の層別伝播

relative L2 は前段から増え、W8A8 は layer 0 の `0.00796172` から layer 30 の
最大 `0.03357130`、final norm `0.03126359` に達した。W8A16 は対応して
`0.00381995`、layer 31 の最大 `0.02773978`、final norm `0.02440245` だった。

| location | W8A16 relative L2 | W8A8 relative L2 | W8A16 max abs | W8A8 max abs |
| --- | ---: | ---: | ---: | ---: |
| layer 0 | 0.00381995 | 0.00796172 | 0.039909 | 0.067305 |
| layer 8 | 0.00863527 | 0.01633916 | 0.279565 | 0.362812 |
| layer 16 | 0.01425341 | 0.02543664 | 0.962643 | 1.496008 |
| layer 24 | 0.01765943 | 0.03083318 | 7.105297 | 37.357048 |
| layer 30 | 0.02350839 | **0.03357130** | 46.444366 | 57.469147 |
| layer 31 | **0.02773978** | 0.03168793 | 65.010483 | 29.131115 |
| final norm | 0.02440245 | 0.03126359 | 14.862573 | **13.696337** |

これは late layers で relative L2 と rare max error の双方が増大するという観測であり、
個別の internal mechanism はこの測定だけでは**未確認**である。

### Outlier の寄与と救済見込み

Primary W8A8 の K=32 activation blocks は `[4,8)` の
`31,857,747 / 222,363,648 = 14.326868%`、`[8,inf)` は 0 だった。
base activation relative L2 は `0.009489628`、clipping は 0 である。

diagnostic `outlier_bypass_ge4` は `14.331775%` の blocks を source FP32 のまま通す
**非 deployable な上限**である。この場合、activation L2 は `0.004431349` になり、
W8A8-to-W8A16 aggregate-logit-L2 gap `0.006535519` は 0（100% removal）になった。
凍結済み rule（50% 以上の gap removal）では outlier side route は
**promising** である。

ただしこの diagnostic 自体も logits max abs `8.991999`、final hidden max abs
`3.448392`、disallowed top-1 mismatch 9 により numeric / overall gate を通過しない。
従って outlier bypass は W8A8 の relative-L2 excess を説明する有力な寄与だが、この上限
diagnostic 単独では全 gate を救えないことが確認された。per-channel scale / SmoothQuant が
max-error と greedy failure も同時に解消するかは**未確認**である。

### Next Actions

1. W8A8 prequant API、runtime/artifact/release admission は開始しない。W8A16 を required
   fallback / default reference path とし、W8A8 は explicit-only のままにする。
2. 次の W8A8 mitigation prototype は、`max(abs)/RMS >= 4` K=32 blocks 用の明示的 mask と
   compact FP16 side plane（cold blocks は existing W8 code + FP16 scale）にする。side
   payload、mask/index overhead、latency を実測し、outlier threshold を探索したうえで同じ
   20-record full-model gate を再実行する。上限 diagnostic の source FP32 bypass を
   deployable result と取り違えない。
3. 独立案として linear input-channel diagonal transform
   `x'_j=x_j/s_j, W'_{ij}=W_{ij}s_j` を用いる per-channel / SmoothQuant calibration を
   prototype 化する。calibration / held-out split、artifact semantics、weight re-quantization
   を明示し、同一 max-abs / top-1 条件で再 gate する。救済可否は未確認である。
4. 再 gate の admission 条件は v0.2 の全値を維持する。特に logits/final-hidden max abs
   `<=1.0`、top-1 rate `>=99.0%`、Wilson `>=98.5%`、zero disallowed mismatch を満たさない
   限り W8A8 を採用しない。
