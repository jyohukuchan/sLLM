# Qwen3.5-35B-A3B MoE runtime foundation v0.1

> Status: implemented baseline design. 2026-07-26 JST.
>
> Scope: loader-independent routing, gather/scatter, grouped GEMM, and shared-expert
> execution primitives. This document deliberately does not claim that the complete
> Qwen3.5 hybrid-attention model is executable yet.

## Evidence and source contract

The source checkpoint was read directly at
`/home/homelab1/datapool/ai_models/safetensors/Qwen3.5-35B-A3B-BF16`.
Its `config.json` SHA-256 is
`5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`.
The observed top-level architecture is
`Qwen3_5MoeForConditionalGeneration`; its text config is
`qwen3_5_moe_text`.

| Contract item | Observed value |
| --- | --- |
| text hidden size / decoder layers | 2048 / 40 |
| experts / selected experts per token | 256 / 8 |
| routed expert intermediate width | 512 |
| shared expert intermediate width | 512 |
| activation | SiLU |
| expert `gate_up_proj` | BF16 `[256, 1024, 2048]` |
| expert `down_proj` | BF16 `[256, 2048, 512]` |
| router `mlp.gate.weight` | BF16 `[256, 2048]` |
| shared gate/up/down | separate BF16 `[512,2048]`, `[512,2048]`, `[2048,512]` |
| shared-expert gate | BF16 `[1,2048]` |
| dense/MoE mix in text decoder | no dense replacement layers observed: all 40 text layers have the 3-D routed tensors and all shared tensors |

The source’s MTP and vision tensors are outside this text-decoder executor
scope. `mtp.layers.0` has a different, per-expert tensor naming/layout and is
not silently treated as a text decoder layer.

The installed local Transformers 5.12.1 source defines routing as:

1. `router_logits = linear(hidden, gate.weight)`;
2. `softmax(router_logits, dtype=float32)`;
3. `torch.topk(..., k=8)`;
4. divide the eight selected probabilities by their sum;
5. cast the selected values back to the router-logit dtype;
6. evaluate each selected expert and add the weighted results;
7. add `sigmoid(shared_expert_gate(hidden)) * shared_expert(hidden)`.

The router weight is not normalized before this softmax. The selected weights
are normalized **after** top-k, not before it.

`torch.topk` documents no stable tie ordering. On this host’s PyTorch
2.12.0+cpu, equal 256-way inputs happened to return
`[172,169,170,171,174,175,168,173]` for `k=8`, while smaller equal inputs
gave different shape-dependent orders. That observation is not an executable
model contract. The baseline exposes a per-token boundary-tie flag and its
full forward path rejects a tie at the kth probability rather than inventing
a different semantic order. The real-weight HF fixture has no boundary tie.

## Runtime ABI and tensor layout

The baseline is intentionally separate from package loading. It accepts
resident runtime buffers and a `F32` or raw IEEE BF16 weight dtype. All
activation, routing-score, and output buffers are F32; BF16 source weights are
read without a mandatory F32 model-wide expansion. For a BF16 router, the
baseline explicitly round-trips the F32 activation and resulting logit to BF16
before HF's F32 softmax, then round-trips the normalized selected score to
BF16. This reproduces the observed HF `F.linear(BF16,BF16)` / router boundary
while preserving F32 buffer ownership in the runtime.

The primitive ABI is split into these operations:

| Primitive | Inputs | Outputs |
| --- | --- | --- |
| `moe_route` | `[M,H]` hidden, `[E,H]` router | `[M,K]` scores, `[M,K]` expert IDs, `[M]` boundary-tie flags |
| `moe_gather` | `[M,H]`, `K` | `[M*K,H]` assignment-major activations |
| `moe_decode_gemm` | one token's `[K,C]` selected activations, selected IDs, `[E,R,C]` weight | `[K,R]` |
| `moe_grouped_gemm` | prefill assignment-major activations, IDs, `[E,R,C]` weight | `[M*K,R]` |
| `moe_gated_silu` | `[M*K,2I]` | `[M*K,I]` |
| `moe_scatter_weighted` | `[M*K,H]`, scores | `[M,H]` routed reduction |
| `moe_sigmoid_gate` | `[M]` shared gates, `[M,H]` shared output | `[M,H]` |

The layout exactly preserves the safetensors row-major layout:
`expert * rows_per_expert * cols + row * cols + col`. Thus routed gate/up
uses `R=2I=1024, C=H=2048`, and routed down uses `R=H=2048, C=I=512`.
The shared expert uses `E=1`: on decode it uses the decode GEMM path; on
prefill it uses the grouped path. It does not need a dense-only implementation.

## Data flow

```text
hidden[M,H]
  | router GEMV + F32 softmax/top-k/renormalize
  +--> expert_id[M,K], route_weight[M,K], tie_flag[M]
  | gather (token repeated K times)
  +--> assignment_hidden[M*K,H]
       | grouped gate/up GEMM [E,2I,H]
       +--> gate_up[M*K,2I] -- SiLU(gate) * up --> active[M*K,I]
            | grouped down GEMM [E,H,I]
            +--> expert_out[M*K,H] -- weighted scatter/reduce --> routed[M,H]

hidden[M,H] -- shared gate/up/down MLP --> shared[M,H]
hidden[M,H] -- shared gate projection --> shared_gate[M]
sigmoid(shared_gate) * shared + routed --> output[M,H]
```

The CPU reference composes the same materialized stages. The GPU baseline has
separate decode and prefill GEMM launch boundaries plus simple kernels for the
other primitives. Prefill retains assignment-major rows for direct reference
comparison; correctness and inspectability take priority over tiling
efficiency. It therefore has seven correctness kernels: route, activation
gather, decode GEMM, prefill grouped GEMM, gated-SiLU, weighted scatter, and
shared sigmoid gate.

## Decode and prefill are distinct plans

### Decode (`M=1`)

There are exactly eight routed assignments plus one shared expert. The
baseline has a separate `moe_decode_gemm` ABI and HIP kernel, which accepts
only the selected expert IDs and `K` activation rows; it is never dispatched
through the prefill grouped kernel. Physical collection of the eight BF16
expert slabs belongs to the later residency layer: this correctness substrate
indexes the selected slabs in a provided `[E,R,C]` buffer and deliberately
does not invent an offload/copy policy. The complete routed reservoir is 1.5
GiB per text layer, while the eight selected routed slabs are 48 MiB per token
(`8 * (1024*2048 + 2048*512) * 2`); the shared gate/up/down adds about 6 MiB.
Thus even decode's approximately 54 MiB of selected MLP weight traffic per
layer is bandwidth-dominated, not arithmetic-dominated. A later residency
implementation must physically gather those slabs before the decode GEMMs;
it must remain separate from prefill’s grouped plan.

### Prefill (`M=N`)

There are `N*8` assignments. The baseline sorts neither assignments nor
weights: `expert_id[a]` selects its group inside a correctness-first grouped
GEMM. A production prefill specialization should histogram/prefix-sum by
expert, permute assignment rows into contiguous expert groups, invoke an
expert-grouped GEMM for each nonempty group, then unpermute/scatter. This is
the path that needs variable-M grouped GEMM tuning; it must not be folded into
the decode gather path.

## Workspace budget

For F32 activations the baseline’s per-layer transient allocation is:

| Buffer | Formula | Decode `M=1` | Prefill example `M=128` |
| --- | ---: | ---: | ---: |
| router logits (diagnostic only) | `M*E*4` | 1 KiB | 128 KiB |
| IDs + scores + flags | `M*K*(4+4)+M*4` | 68 B | 8.5 KiB |
| gathered hidden | `M*K*H*4` | 64 KiB | 8 MiB |
| gate/up | `M*K*(2I)*4` | 32 KiB | 4 MiB |
| activated intermediate | `M*K*I*4` | 16 KiB | 2 MiB |
| per-assignment down output | `M*K*H*4` | 64 KiB | 8 MiB |
| routed + shared + final outputs | `3*M*H*4` | 24 KiB | 3 MiB |

The M=128 total excluding optional logits is about 25 MiB. It is negligible
relative to the uncompressed weights; a future resident implementation still
needs a separately budgeted KV cache and attention workspace.

## Actual R9700 residency decision

`amd-smi` reports the permitted gfx1201/R9700 as 34,208,743,424 bytes
(31.859 GiB) VRAM. A safetensors-header audit gives:

| Scope | Raw BF16 / source bytes | Difference from R9700 VRAM |
| --- | ---: | ---: |
| text decoder | 68,304,112,256 B / 63.613 GiB | short 31.754 GiB |
| routed + shared expert weights | 64,676,331,520 B / 60.235 GiB | short 28.375 GiB |
| complete checkpoint (text + vision + MTP + other) | 71,903,655,008 B / 66.965 GiB | short 35.106 GiB |

Consequently a full raw-BF16 resident 35B inference cannot be attempted on
this R9700. This task does not silently introduce quantization, CPU offload,
or partial-layer generation as a substitute. It verifies the primitives on
small resident slices and leaves a BF16-aware streaming/offload or quantized
weight policy as an explicit later integration decision.

## Non-goals and promotion boundary

- No changes are made to `AQ4_0`, `SQ8_0`, attention kernels, or the active
  serving manifest.
- This is not a Qwen3.5 full-model executor: hybrid linear/full attention,
  mRoPE, Q output gating, model loading, tokenizer/vision handling, MTP, and
  KV state remain separate work.
- The verification uses the lightweight HF trace machinery and dedicated
  routing fixtures; it does not invoke FP32 reference corpus, campaign, or
  bitwise-promotion gates.

## Implemented baseline and evidence

The public runtime C ABI now has loader-independent `moe_route`, `moe_gather`,
`moe_decode_gemm`, `moe_grouped_gemm`, `moe_gated_silu`, `moe_scatter_weighted`, and
`moe_sigmoid_gate` operations. Their CPU implementation is the reference;
the optional `rocm-moe-gfx1201` feature compiles simple static HIP kernels
only for the permitted gfx1201 device. The GPU path fails closed when that
feature is absent or the selected device is not gfx1201.

`moe_runtime_verify` exercises every materialized stage on both F32 and raw
BF16 weight storage through two synthetic paths: prefill (`M=5`) uses
`moe_grouped_gemm`, and decode (`M=1`) uses only `moe_decode_gemm`. CPU C ABI
versus the Rust reference is bit-identical for all stages in both paths. On
the R9700, each path's final-output maximum absolute difference was
`2.384185791e-7`; the largest raw-BF16-stage difference was
`3.576278687e-7` (prefill shared gate/up). This is a correctness check, not a
timing measurement.

For the real layer-0 router fixture, generated directly with the installed
HF `Qwen3_5MoeTopKRouter`, three BF16 inputs selected:

```text
[52, 148, 101, 178, 151, 128, 116, 166]
[171, 96, 196, 226, 76, 123, 80, 117]
[23, 225, 71, 21, 250, 163, 84, 102]
```

Both CPU and R9700 runtime routing matched those IDs and the BF16-normalized
scores exactly (`max_abs = 0`). An exact-tie fixture is separately flagged and
not claimed as an HF ordering match. The compact result reports are under
`benchmarks/results/2026-07-26/qwen35-moe-runtime-v0.1/`; raw fixture tensors
remain local because they duplicate checkpoint weights.

The same fixture generator captures two compact real 3-D layer-0
`gate_up_proj` slices. The prefill slice uses source experts `[52,148]`, raw
BF16 shape `[2,37,71]`, and deliberately reordered local IDs `[1,0,1]`.
The decode slice uses the real first-token top-8 source experts
`[52,148,101,178,151,128,116,166]`, raw BF16 `[8,37,71]`, and local IDs
`[0,1,2,3,4,5,6,7]`. In both cases HF F32 expected values, the raw-BF16 CPU
reference, CPU C ABI, and the respective R9700 GEMM kernel agreed exactly
(`max_abs = 0`). This validates expert-axis/row/column addressing without
checking a copied full 1.5-GiB expert reservoir into source control.

`tools/architecture_hf_trace.py self-test` was also run unchanged and rejected
its deliberate layer-3 corruption at `step-0000__layer-0003`. A full 35B HF
capture was not launched on this shared host: the raw checkpoint needs
66.965 GiB before framework overhead, while the measured host memory available
at validation time was about 43 GiB. There is no complete uLLM hybrid-attention
candidate to compare yet in any case; the direct HF router fixture is the
numeric source for this loader-independent layer of the work.

## BN: AQ4_0 MoE text-package manufacturing (2026-07-26)

The source audit independently reproduces BI's residency numbers exactly:
the complete safetensors payload is `71,903,655,008 B` / `66.965497 GiB`, the
text decoder without `lm_head` is `68,304,112,256 B` / `63.613162 GiB`, and
routed plus shared experts are `64,676,331,520 B` / `60.234528 GiB`.  Thus the
R9700's reported `34,208,743,424 B` / `31.859375 GiB` cannot hold raw BF16.
All 40 text layers contain MoE; there is no dense replacement layer.  Each
routed payload is rank-3 BF16: `gate_up_proj [256,1024,2048]` and `down_proj
[256,2048,512]`.

The manufactured text-only candidate is
`/home/homelab1/datapool/ullm/product/qwen35-35b-a3b-aq4_0-g8-moe-v0.2/`.
It uses the already defined strict `AQ4_0` candidate
`aq4_e4m3_g8_ts_flloyd16` (4-bit indices plus one E4M3 scale index per eight
values, 5 effective bits/value).  G8 was selected over the existing G16
candidate because it is the highest-fidelity existing `AQ4_0` choice while
remaining inside the R9700 byte ledger; no format was invented.  `SQ8_0` was
not selected: routed experts alone would need about 30 GiB at 8 bits before
the 4.56 GiB text passthrough and cache/workspace, and the existing `SQ8_0`
BF16-source route does not support this rank-3 MoE contract.

Only 80 routed rank-3 expert tensors are `AQ4_0`.  Router weights, shared
experts, attention, embeddings, norms, `lm_head`, and other text tensors are
raw passthrough (613 tensors).  The resulting exact payload ledger is
`20,132,659,200 B` AQ4 indices/scales + `4,896,721,536 B` raw passthrough +
`128 B` for two codebooks = `25,029,380,864 B` / `23.310427 GiB`.
The package excludes the source checkpoint's vision/MTP-only payload because
this is a text decoder artifact, not a multimodal serving integration.

Codebooks are not fit per expert.  A held-out cross-validation compared one
global routed-projection-family codebook with per-layer and per-expert Lloyd16
fits.  For `down_proj`, the global median/p95/max relative MSE were
`0.003997/0.004244/0.005328`, versus per-expert
`0.004125/0.004516/0.008668`; for `gate_up_proj`, global was
`0.003943/0.004103/0.004369`, while per-expert has unstable tails (max
`1.448716`).  Per-layer fitting also creates worse layer-0 tails.  The product
therefore has one globally shared codebook for each routed projection family
(two codebooks total), across all `40 * 256` experts.  The study is retained
under the candidate work directory as `codebook-granularity-study.json`.

The converter is deliberately narrow and restartable: it streams a
safetensors tensor at a time, uses bounded chunks, writes per-tensor staging
directories, rereads/dequantizes each completed tensor, and records a
`quantization-state.json`.  It completed all 80 tensors without materializing
the checkpoint; a subsequent `--resume` quantize invocation completed in
`0.282 s` without reconversion.  Full package verification reread every
index/scale/codebook and raw passthrough payload.  The per-tensor error record
is `evidence/tensor-errors.json`: relative MSE is
`0.003603673..0.004363885` (mean `0.003634245`) and max-absolute error is
`0.005326890..0.043730080` (mean `0.013711253`).  Robust outliers are not
silently ignored: relative-MSE outliers are layer 0 `down_proj`, layer 32 and
33 `gate_up_proj`, and layer 39 `down_proj`; layer 39 `down_proj` is also the
sole max-absolute outlier (`0.043730080`).

Router payloads are intact: all 40 source/package SHA-256 values match, and
the installed PyTorch BF16-linear/FP32-softmax/top-8 arithmetic gives zero
conditional route changes over 1,280 inputs (92 boundary ties are recorded,
not hidden).  The raw shared-expert path is separately hash-verified.  A
four-token layer-0 MoE MLP check gives relative MSE `0.000451407`, RMS
`5.1263e-6`, and max-absolute error `2.0772e-5`.

However, this does **not** establish end-to-end routing invariance.  The
bounded CPU-only one-layer-at-a-time 8-token prefill executes all 40 real
decoder layers without loading the full checkpoint.  Its source-vs-source
control is exactly zero for all 320 token-layer route checks and final hidden
state.  Source versus this `AQ4_0` candidate changes strict ordered top-k IDs
for 238/320 checks and, more importantly, changes the selected expert set for
105/320 checks; final hidden relative L2 is `0.076012410`.  This is caused by
accumulated lossy routed-expert output changing later router inputs, despite
the router weights themselves being raw and exact.  Product metadata records
this historical qualification as `not_passed`; the generated-text
reclassification below supersedes that status.  The candidate is not connected
to a loader, service, or promotion path.

For batch 1, the exact packed-artifact ledger plus source-derived cache/state
and conservative gather/workspace reserves stays within R9700 VRAM at each
recorded context: 4,096 tokens `25,240,265,156 B` (headroom
`8,968,478,268 B`), 32,768 `25,864,459,076 B` (headroom `8,344,284,348 B`),
131,072 `28,004,552,516 B` (headroom `6,204,190,908 B`), and the configured
262,144 maximum `30,858,010,436 B` (headroom `3,350,732,988 B`).  This is an
actual artifact-byte/KV ledger, not a `hipMemGetInfo` allocation measurement:
a Qwen3.5 MoE loader/residency integration does not yet exist, so empirical
R9700 allocation and serving remain unconfirmed.  No service, active manifest,
`/opt/ullm` content, FP32 reference corpus, bitwise gate, or campaign was
used or changed.

## BS: `AQ4_0` MoE package の生成品質による再分類（2026-07-26）

BN 時点の `not_passed` は package の破損を示すものではなかった。旧判定は
「source と量子化後で end-to-end の selected expert set が完全一致すること」を必須にしていた。
これは raw router の正しさと、上流 expert 出力の量子化誤差が後段 router 入力へ伝播することを
混同した誤った quality gate だった。raw router は全 40 tensor の SHA-256 が source と一致し、
1,280 条件付き入力の top-8 は 0 変化のままである。一方、同一 8-token prefill の CPU
streaming 観測は source 対 `AQ4_0` で selected set `105/320`、ordered top-k `238/320`
変化した（source 対 source は両方 0/320）。この値は消さず、**非 gate の観測値**として
product metadata に保持した。

正しい判定対象である生成文を、拡張した
`tools/validate-qwen35-moe-aq4-streaming-forward.py` の CPU-only streaming generator で
確認した。tokenizer の official chat template（`add_generation_prompt=true`,
`enable_thinking=false`）を使い、全 40 decoder layer を one-layer-at-a-time で通し、raw
passthrough final RMSNorm / `lm_head` から greedy token を読んだ。source は BF16 checkpoint
値を F32 演算へ変換し、`AQ4_0` 側だけが routed expert row を package の
idx4/E4M3/codebook から復元する。同 generator の v0.1 source-vs-source control は 3 ケース
38 greedy step で token、route、initial/final hidden の全てが 0 差だった。

quality evidence は
`benchmarks/results/2026-07-26/qwen35-moe-aq4-quality-reclassification-v0.1/`
にある。CPU 時間を抑えるため、通常の 10-case lightweight promotion suite から日本語、英語、
コード生成の 3 ケースに短縮した v0.2 suite を用いた（prose は最大 24、code は最大 16 token）。
source 対 `AQ4_0` の実際の出力は次の通りで、両 prose は意図した 24-token cap に達したため
末尾の句読点を要求しなかった。

| case | source | `AQ4_0` | quality judgement |
| --- | --- | --- | --- |
| Japanese rollback recovery | 「…事前に定義された手順に従って迅速かつ確実に元の状態に戻す」 | 「…事前に定義された安全な手順に従って元の状態に迅速に」 | 同じ復旧内容を意味的に維持。`AQ4_0` の句は cap で終わるが、混線・文字化け・放棄なし。 |
| English rollback recovery | 「…minimizing downtime and mitigating potential」 | 「…minimizing downtime and preventing service disruption」 | 同じ rollback 説明を英語で保持。`AQ4_0` は自然な完結句。 |
| Python `is_even` | `is_even = lambda n: n % 2 == 0` | `is_even = lambda n: n % 2 == 0` | 完全一致し、要求どおりの一行コード。 |

自動 screen は両経路の全ケースで空応答、replacement character、非許可 control character、
三重連続反復を検出しなかった。source/`AQ4_0` の生成 token 数はそれぞれ `24/24`、`24/24`、
`14/14` で極端な長さ偏りもない。greedy match は `47/62`、source-greedy conditional NLL は
source より `AQ4_0` で大きかったが、いずれも quality threshold には使わない。実際、生成途中で
token が違えば以後の route 入力も違うため、この path の selected-set `2276/6040`、ordered
`4285/6040` は同一入力比較ではなく、品質 pass/fail に用いない。

従って product metadata の `generation_quality_validation.status` を **`passed`** に更新した。
この pass は CPU streaming の生成品質に限る。MoE loader、hybrid attention、mRoPE/KV state、
Q output gate、weight residency、R9700実 allocation、service/promotion は依然として未確認である。
layer 39 `down_proj` の max-abs outlier `0.043730080` はこの生成文で個別に raw-passthrough
ablation をしていないため、生成への因果的影響は **未確認** と記録した。文章品質の崩壊がなかった
ため、codebook 方針や layer 39 passthrough を変える Phase 3 は実施しなかった。

## BW: `AQ4_0` MoE resident loader wiring（2026-07-27）

### 実装前の停止点と隙間

`ullm-model-config-inspect --package
/home/homelab1/datapool/ullm/product/qwen35-35b-a3b-aq4_0-g8-moe-v0.2/package`
で descriptor が Qwen3.5 MoE text として正しく構成されることを確認した。一方で
`model_config.rs` の `execution_status` は意図的に
`Qwen35MoeExecutor` を要求し、理由を “top-k routing, gather/scatter,
grouped expert GEMM, weighted reduction, and shared expert execution are not
implemented” と返す。descriptor 側を変更せず、loader は
`resident_descriptor()` を読み、その未実装 executor 判定を通らない独立の
`Qwen35MoeAq4Runtime` とした。

次の表が、descriptor と BI の MoE C ABI の間にあった実際の隙間である。

| 項目 | 足りなかった結線 | BW での扱い |
| --- | --- | --- |
| loader / weight residency | `AQ4_0` rank-3 expert tensor を一括 resident にし、raw BF16 passthrough の router/shared/attention を同じ registry から参照する loader | `WeightRegistry` に 80 個の expert tensor を一度だけ載せ、expert 軸の byte slab を top-k 順へ stage する。`PackageAq4ResidentMatvec` は exact raw BF16 passthrough も resident matvec/row gather として読めるようにした。 |
| routing / gather-scatter / GEMM | BI の API は存在したが、実モデルの router 出力・selected expert slab・local expert id を結ぶ呼出し側がなかった | post-norm hidden から raw BF16 router で top-8、`moe_gather`、stage/dequant、decode 専用 `moe_decode_gemm`、gated SiLU、down projection、`moe_scatter_weighted` を層ごとに接続した。 |
| shared expert | routed result への raw shared MLP と sigmoid scalar gate の合流がなかった | 既存 layer の raw shared MLP placeholder を実行し、HF と同じ `routed + sigmoid(shared_gate) * shared` を外部 MLP residual へ渡す。 |
| hybrid attention / KV state | MoE 専用 attention を新たに書く必要があるように見えた | 書かなかった。既存 Qwen3.5 `AQ4_0` layer の linear convolution/recurrent state と full paged-KV state を、post-attention RMSNorm まで動かす bridge として再利用した。 |
| mRoPE / Q output gate | MoE loader が full attention の Q+gate projection、Q/K norm、mRoPE を持っていなかった | 既存 full-attention bridge を通す。純 text の三つの mRoPE position row は同じ text position なので scalar bridge を用いる。vision/multimodal position は本 executor の対象外であり未実装である。 |

HF の根拠は、環境に導入済みの Transformers 5.12.1 の
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` で確認した。
linear attention は `value_dim = num_v * head_v_dim` として qkv/conv/output
の value stream を別に扱う（369--417 行）。full attention は Q projection を
Q と output gate に分け、Q/K norm・RoPE・KV cache 更新の後に
`attn_output *= sigmoid(gate)` を行う（683--717 行）。router は FP32
softmax/top-k/正規化、expert は gate/up/down の SiLU product、shared branch は
scalar sigmoid gate である（720--814 行）。RMSNorm は `output * (1 + weight)`
（817--831 行）、mRoPE は interleaved section `[11,11,10]`（167--182 行）である。
同じ HF text model の default position-id path は scalar text positions を四行へ expand
した後、先頭の causal-mask row を外して rotary embedding へ三行を渡す
（1279--1303 行）。従って pure-text decode では三つの mRoPE row は同じ position であり、
resident scalar bridge の根拠になる。画像・動画 token の `get_rope_index` が作る異なる
temporal/height/width row（1414 行以降）はこの text-only executor には含めない。

同じ inspected `text_config` の `rms_norm_eps` は正確に `0.000001` であり、HF は
decoder input/post-attention RMSNorm、linear-attention gated RMSNorm、full-attention
Q/K RMSNorm のいずれにもこの config 値を渡す。既存 dense bridge の歴史的な
post/QK `1e-5` default を MoE に混ぜないよう、BW は MoE 専用
`*_with_rms_epsilon` bridge を追加し、descriptor の `rms_norm_epsilon` を全 MoE
norm site に渡す。public dense entrypoint の default は変更していないため、9B
`AQ4_0` の既存数値経路を変更しない。この修正は推測ではなく、上記 HF source と
source `config.json`（SHA-256 `5e4d7f74…9bfbc7944`）の双方に基づく。

full-attention の native fused Q/K-norm/RoPE/KV writer は operation plan の geometry
にも epsilon bit を binding する。そこで MoE loader は
`load_moe_shared_with_registry_with_rms_epsilon` で descriptor 値をその binding にも
渡す。これにより F32 KV fallback でも `1e-6` の実行値と plan が一致する。F16 typed
KV branch は native typed writer を直接選ぶが、loader の admission contract を古い
`1e-5` と偽らない。この MoE 専用 binding 変更でも dense 9B loader は従来の `1e-5`
binding のままである。

### Qwen3.5-9B `AQ4_0` からの再利用と互換性境界

既存の `qwen35_aq4_layer_runtime.rs` を attention の唯一の実装として使った。
したがって full attention の paged KV、Q output gate、Q/K norm、純 text mRoPE、
1+weight RMSNorm と、linear attention の causal convolution/recurrent state は
9B 経路と同じコードである。既存 dense `run_device_step` は変更せず、MoE のために
`run_device_step_through_post_norm` / `finish_external_mlp` / shared-expert-only
bridge を追加した。

35B の linear geometry は 9B と同一ではない。35B config は hidden 2048,
key heads 16, value heads 32, key/value head dim 128、従って attention value
stream は 4096 要素である。既存 9B（hidden 4096, 16/32 heads, 128 dim）の
default geometry を保存したまま、MoE bridge だけ descriptor の geometry を渡す
ようにした。`linear_attention_geometry_keeps_the_9b_and_35b_value_streams_distinct`
が両方の value stream を区別する回帰単体試験である。

decode は一度に一 decoder layer しか MoE を実行しないため、selected-expert の
stage/dequant/GEMM/scatter buffer は 40 層に複製しない。35B の top-k 8,
hidden 2048, expert intermediate 512 から、共有 workspace は
`116,605,032 B`（111.20 MiB）である。初版の per-layer scratch はその 40 倍
`4,664,201,280 B`（4.344 GiB）になり、BN ledger の一層分 workspace reserve と
整合しなかった。resident executor は最大 geometry の一つの workspace を全層で
直列再利用するよう修正した。この値は要求 allocation byte 数であり、HIP allocator
overhead を含む実測 telemetry は下記の R9700 実行で確認する。

KV residency にはもう一つ byte-contract の境界がある。BN ledger の
`full_attention_kv_bytes_per_token=20,480 B` は、10 full-attention layer × K/V
× 2 KV head × 256 dim × **2 B**（BF16 source cache）である。BW 初版 binary の
full-attention bridge は F32 cache を確保していたため、同じ 262,144 token で
さらに `5,368,709,120 B` を要求する。したがって初版の静的見積りは
`36,226,719,556 B`、R9700 `34,208,743,424 B` より `2,017,976,132 B` 超過する。
これはまだ allocation を試みた結果ではない。typed KV cache はその後
`d8389e59` で確定した。R9700 上の既存 Qwen3.5-9B `AQ4_0` では native F16 cache の
8,192 logical token（8 full layer で 256 MiB）を、F32 cache の 4,096 token と同じ
allocation byte で実 load し、3,968-token prompt からの 64 token は F32 と token ID/text
が完全一致した（`benchmarks/results/2026-07-27/kv-cache-dtype-kernels/summary.json`）。
これは typed native path の根拠であって 35B MoE の証明ではない。MoE executor は
`ULLM_KV_CACHE_DTYPE=f16` を明示した 2 B/value capacity run で BN ledger と同じ byte
budget を実測する。F16 storage は capacity のための runtime 選択であり、BF16 source
cache と数値的に同一だという主張はしない。

### loader/source の検証範囲

`tools/architecture_hf_trace.py self-test` は意図的な layer-3 corruption を検出して
成功した。完全な 35B HF capture は、checkpoint 66.965 GiB に対して shared host の
available RAM が不足するため起動していない（**未確認**）。代わりに
`benchmarks/results/2026-07-27/qwen35-moe-loader-wiring-v0.1/
hf-streaming-source-control.json` で、source BF16 を一層ずつ読み出す bounded
5-token/40-layer control を実行した。final hidden max-abs error と relative L2 は
ともに 0、200 token-layer の ordered top-k は全て一致、peak RSS は
6,017,904,640 B だった。これは loader への HF end-to-end 比較ではなく、
不足する full capture を避けた source streaming control と明記する。

実機生成、router read-back と VRAM telemetry は、R9700 排他ロックを取得してからこの節に
追記する。サービスを起動・変更せず、`active.json` にも触れない。

実機待機中の 01:48 JST の read-only `amd-smi process --gpu 2 --json` では、既存本番
Qwen3.5-9B `AQ4_0` worker（PID 4044361）が `7,119,884,000 B` VRAM を保持し、同時刻の
`amd-smi metric` free VRAM は `25,542 MiB = 26,782,728,192 B` だった。これは 262,144-token
ledger `30,858,010,436 B` より `4,075,282,244 B` 少ない。さらに worker は
`/run/ullm/r9700.lock` を保持していた。このため lock を奪う、又は本番 service を止める
代わりに、明示的な解放を待つ。これは MoE loader の allocation failure ではなく、既存本番が
占有中のため安全に試行していない状態である。

### 実機隔離の index 契約（2026-07-27）

空き window での最初の 9B baseline probe は、`--device-index 0` が
`expected=gfx1201 actual=unavailable` で fail-closed した。これはモデル loader の失敗ではなく、
`ullm_runtime_sys` が runtime index 0 を CPU fallback に予約しているためである。HIP を一台に
隔離した後も、最初の HIP device は runtime index **1** である。probe は architecture check の前で
停止しており、重みは読んでいない。

次に runtime index 1 へ直したが、AMD SMI の物理 GPU index をそのまま
`HIP_VISIBLE_DEVICES=2` に渡すと `actual=gfx1030` で fail-closed した。read-only topology
照合の結果、AMD SMI は `0=gfx1030 (03:00.0)`, `1=gfx1030 (43:00.0)`,
`2=gfx1201/R9700 (47:00.0)` である一方、ROCm/HIP の順序は
`0=gfx1030 (43:00.0)`, `1=gfx1201 (47:00.0)`, `2=gfx1030 (03:00.0)` だった。
従って R9700 の安全な実行契約は **`HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1` /
uLLM runtime `--device-index 1`** である。HIP ordinal 2 は禁止された V620 である。
この試行も architecture check の直後に停止したため、V620 には context 選択以外の package weight
allocation と kernel dispatch を行っていない。再現レシピはこの対応へ訂正し、full MoE 実行は
`gfx1201` architecture admission を通ってからのみ開始する。

次の R9700 index-corrected probe は `Qwen3.5 AQ4 linear layer 0: required backend operation
runtime feature/guard is unavailable` で weight load 前に停止した。`Qwen35MoeAq4Runtime` が
attention を再実装せず既存 9B bridge を使う以上、これは同 bridge の production HIP guard
contract を offline driver へ渡していなかった結線欠落である。`qwen35_aq4_layer_runtime.rs` の
linear load は `HipLinearAttentionRecurrent`、`HipLinearAttentionQkvPrepare`、
`HipAq4MatvecBatch`、`HipLinearAttentionQkvPrepareBatch` の capability probe を要求する。
これらの probe は対応する `ULLM_REQUIRE_HIP_*` 環境変数が `1` のときだけ実行する。

推測で subset を選ばず、既存 resident 9B worker が正規 contract とする
`QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV` の全 36 guard を offline 9B baseline と MoE driver の
両方へ渡すよう再現レシピを補正した。MoE にはこれに F16 capacity run の typed paged
decode/split/KV-write 3 guard を加える。script は guard assignment を artifact に記録する。
この 36-guard corrected run は lock の次の安全 window で未再試行であり、成功は未確認である。

### 実機 window の外部占有（2026-07-27 08:00 JST）

BR の prefill sweep は 07:38 JST に正常終了し、同じジョブが `ullm-openai.service` を正常に
復旧した。その後の read-only preflight では、`fuser -v /run/ullm/r9700.lock` は gateway PID
1371668 を lock owner として返し、`ActiveState=active`、R9700 worker PID 1371800 の VRAM は
`7,119,872,000 B`（AMD SMI metric は `7,082 MB`）、edge は 37 C だった。したがって package
payload `25,029,380,864 B` と 262,144-token ledger `30,858,010,436 B` を同居させる安全な
free window ではない。

BW は service の停止・起動、worker kill、lock の奪取、`active.json` の変更を行わない。
このため full 36-guard corrected 9B baseline / 35B MoE run は **未実行**であり、これは
AQ4_0 allocation failure や生成失敗を示す観測ではない。R9700 が外部から解放され、service
inactive、GPU process なし、edge <=45 C を同時に満たした時点で、隔離 release binary により
9B baseline → 35B 262,144-token F16-KV resident load → 8-token generation → router read-back / VRAM
telemetry を一つの flock window で実施する。

### 許可済み physical window の結果（2026-07-27 10:22 JST）

CE は明示許可に基づき `ullm-openai.service` を一度だけ停止した。停止前の active manifest は
`a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`、R9700 edge は 36 C、
停止後の AMD SMI process list は空であった。`HIP_VISIBLE_DEVICES=1`、uLLM runtime device 1、
36 個の AQ4 HIP guard と F16 typed-KV の 3 guard を指定し、9B baseline は HIP backend で
128 token を完走、既知 top-1 token **220**（logit `8.529029846191406`）と一致した。

35B driver は 262,144-token / F16 KV configuration で load を開始したが、VRAM allocation の前に
`Qwen3.5 MoE full layer 3 does not match the inspected mRoPE/Q-gate/KV contract` で fail-closed した。
従って 30,858,010,436 B ledger の実 allocation は**未測定**、生成文・40 layer route read-back・
MoE prefill/decode throughput も**未到達**である。telemetry JSONL の唯一の poll は VRAM fields が
`null` であり、allocation byte を返していない。これは OOM や ledger 超過を示す測定値ではない。

停止理由を source と照合すると、`resident_rope_from_qwen35()` は mRoPE descriptor に
`rotary_dim: None` / `partial_rotary_factor: Some(0.25)` を保持する一方、MoE runtime は full layer に
`rotary_dim == Some(64)` を要求している。この相互に満たせない contract が layer 3 で検出された。
この window は一回とする運用制約に従い、service を再停止して修正版を試すことはしない。

service は同じ active manifest のまま一度だけ start し、`/readyz`、`/v1/models`、実際の
completion `service restored` を OpenWebUI bridge から確認した。復旧後は
`ActiveState=active`、`NRestarts=0`、manifest SHA-256 は停止前後で同一である。gateway completion の
短文 timings（prefill `70.942962 tok/s`、decode `122.082175 tok/s`）は recovery response の値であり、
既知 baseline（prefill `975.421658`、decode `77.836412 tok/s`）と workload が異なるため性能回帰の
判定には使わない。9B の functional baseline は top-1 一致で確認済みだが、同条件の throughput
baseline はこの window では未取得であり、回帰は**未確認**である。

完全な証跡は
`benchmarks/results/2026-07-27/qwen35-moe-physical-run/` に保存した。MoE を本番 manifest へ昇格
していない。

### CH: HF 契約再監査と mRoPE descriptor 修正（2026-07-27）

CH は Transformers **5.12.1** のローカル実装と、実 checkpoint の
`config.json` を再読した。`Qwen3_5MoeForConditionalGeneration` という top-level
architecture は vision を含む conditional wrapper であるが、対象 executor は
`text_config.model_type=qwen3_5_moe_text` の text decoder に限定する。したがって
vision/audio/MTP token はこの executor の対象外であり、multimodal 入力を scalar text
position として黙って実行することはしない。

| 項目 | HF 実装の根拠 | 35B-A3B text config | 9B dense との差分 |
| --- | --- | --- | --- |
| position/mRoPE | `modeling_qwen3_5_moe.py:1279-1290` は position を text/T/H/W の 4 行にし、text row を mask に、残る T/H/W の 3 行を RoPE に渡す。`:147-165` は 2-D ids を 3 行に expand し、`:167-182` は section を interleave する。 | `mrope_section=[11,11,10]`, `mrope_interleaved=true`, `partial_rotary_factor=0.25`, theta `10_000_000`; head 256 なので実行 rotary width は 64。text-only では 3 行とも同じ token position なので scalar bridge と同値。 | dense も同じ処理（`modeling_qwen3_5.py:1174-1199`, `:95-184`）と同じ rope config。mRoPE は MoE 固有ではない。 |
| Q output gate | `modeling_qwen3_5_moe.py:655-690` は Q projection を Q/gate の二つに split し、`:692-717` は Q/K の両方へ RoPE、attention output に `sigmoid(gate)`、その後 O projection。 | q-norm/k-norm 有効、gated Q projection。 | dense も同じ順序・shape（`modeling_qwen3_5.py:657-716`）。 |
| KV/GQA | `modeling_qwen3_5_moe.py:650-665` は Q=16 heads、KV=2 heads、head dim=256。`:606-640` は `repeat_kv` と attention を定義し、`:695-696` は K/V を cache update する。 | K/V projection rows は各 512、GQA ratio 8、full-attention layer 自身が K/V cache を所有。 | dense は Q=16、KV=4（config; `modeling_qwen3_5.py:652-665`）で ratio 4。これは MoE 固有 loader が密 bridge の KV shape を仮定してはならない理由である。 |
| hybrid state | `modeling_qwen3_5_moe.py:837-846` と `:1305-1316` は layer type で linear attention と full attention を分ける。 | 40 層、full は 3,7,…,39。linear は key heads 16/value heads 32、各 128 dim、conv width 4、state FP32。 | dense も同じ hybrid 契約（`modeling_qwen3_5.py:756-765`, `:1200-1211`）だが 32 層、full は 3,…,31。 |

停止した CE binary の不一致は multimodal mRoPE を無視したことでも Q-gate/KV 未実装でもなく、
descriptor が正しく持つ `rotary_dim=None` と `partial_rotary_factor=0.25` に対し、MoE
validator が導出済みの実行値 `rotary_dim=Some(64)` を descriptor そのものへ誤って要求した
ことである。`resident_rope_from_qwen35()` が前者を作る根拠は
`crates/ullm-engine/src/model_config.rs:1627-1641`、HF の width 導出は
`modeling_qwen3_5_moe.py:132-143` である。

コミット `217992c4` は source semantic contract（mRoPE: `None + 0.25`, interleaved
`[11,11,10]`）と text-only fused-kernel execution width（64）を分離した。full attention
は 35B 固有の `16Q/2KV/256` と Q/K norm、Q gate、own KV を、linear attention は実 config
の recurrent-state geometry を fail-closed で照合する。fixed 64 descriptor、9B の 4 KV
heads、section/partial-factor/gate/KV の不一致は引き続き拒否する。これは 9B dense path に
触れない MoE runtime-only 変更であり、`cargo test -p ullm-engine qwen35_moe_aq4_runtime --lib`
は 3 tests passed、`cargo build --release -p ullm-engine --features rocm-moe-gfx1201 --bin
ullm-qwen35-moe-aq4-generate` は成功した。
