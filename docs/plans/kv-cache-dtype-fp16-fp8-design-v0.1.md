# Paged KV cache FP16 / FP8 設計 v0.1

## 結論とスコープ

この文書は persistent paged K/V cache の F32 / FP16 / FP8 E4M3FN
（以後 FP8 と略記）の storage contract を固定する。`Q8_0` は実装・ABI
とも対象外である。

- F32 は既定値であり、既存の F32 writer / reader dispatch はそのまま残す。
- FP16 は payload を IEEE binary16 にするだけで、scale metadata を持たない。
- FP8 は **K, V とも OCP E4M3FN payload + 独立した FP16 scale** とする。
  scale の粒度は plane ごとの `(physical_token, kv_head)`、すなわち head
  dimension 256 値ごとである。
- K と V は runtime で別々に選べる。例えば K=FP16 / V=FP8 は有効である。
- `Q8_0` / `q8` を selector に渡すと明示的に拒否する。

v0.1 の実装は generic `PagedDecodeState` の CPU path、同一 ABI の HIP
staging fallback、確保・readback・direct decode・causal prefill fallback
まで到達している。native HIP typed kernel と AQ4 resident production path
への接続は未完了であり、速度・実生成の合否をまだ主張しない。

## 1. 現状 KV 経路の棚卸し

### 1.1 persistent payload の writer

| 層 | F32 現状 | dtype 変更で必要なこと | v0.1 |
|---|---|---|---|
| Engine generic state | `crates/ullm-engine/src/decoder.rs`: `PagedDecodeState::{write_token_at,write_sequence_from_device,decode_step_to_device,decode_step_from_device}` | allocation byte count、optional scale buffers、K/V dtype 伝播、typed writer 呼出し | 実装済み |
| Engine AQ4 resident | `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs`: `ResidentSelfAttentionRuntime` の `k_cache_buffer` / `v_cache_buffer`、`execute_paged_kv_write_f32`、`execute_fused_qk_norm_rope_paged_kv_write_f32`、chunk writer | layer-owned allocation、operation registry plan、fused Q/K norm/RoPE writer の dtype / scales、reader 全部を同時に切替 | **未変更**。AQ4_0 本番実装を変更しない指示に従う |
| Engine operation dispatch | `crates/ullm-engine/src/backend_operation_registry.rs`: F32 writer/reader plan と `execute_*_f32` | typed operation kind、buffer contract、fallback/error policy | AQ4 path 用は未変更 |
| Rust FFI | `crates/ullm-runtime-sys/src/lib_parts/part_00.rs` | enum 値、payload / scale size validation、null exactly-for-non-FP8 contract | `paged_kv_write_typed_f32` 実装済み |
| C ABI | `runtime/include/ullm_runtime.h` | dtype enum と typed writer/read APIs、scale layout の説明 | 実装済み |
| Runtime API | `runtime/src/ullm_runtime_api_attention.inc`: `ullm_runtime_paged_kv_write_f32` | dtype validation、payload byte count、scale buffer validation、CPU/HIP dispatch | `ullm_runtime_paged_kv_write_typed_f32` 実装済み |
| CPU reference | `runtime/src/ullm_runtime_parts/part_00.inc`: `paged_kv_write_f32_host`、`qwen35_qk_norm_rope_paged_kv_write_f32_host`、`paged_kv_write_chunk_f32_host` | F16 encode、FP8 max/scale/encode、physical page address、K/V別 scale | plain typed writer 実装済み。fused Qwen / chunk は未実装 |
| HIP launch/cache | `runtime/src/ullm_runtime_parts/part_01.inc`: `paged_kv_write_f32_hip_kernel`、`qwen35_qk_norm_rope_paged_kv_write_f32_hip_kernel`、chunk kernel launch | typed kernel cache、specialization key、arguments、HIPRTC compilation、fallback | 未変更（BR と競合） |
| HIPRTC source | `runtime/src/ullm_runtime_hiprtc_sources.inc`: `ullm_paged_kv_write_f32_kernel`、`ullm_qwen35_qk_norm_rope_paged_kv_write_f32_kernel`、chunk writer | F16 / FP8 writer kernel、per-head reduction と scale store、mixed K/V type specialization | 未変更（BR と競合） |

`qwen35_qk_norm_rope_paged_kv_write_f32` は Q/K norm と RoPE の後に K を
cache へ書く fused path である。ここを typed にしなければ AQ4 resident
decode は typed cache を生成できない。`paged_kv_write_chunk_f32` は prefill
batch の persistent writer であり、同様に typed 対応が必要である。

### 1.2 persistent payload の reader

| Reader | 現状 | dtype 変更で必要なこと | v0.1 |
|---|---|---|---|
| Decode | `ullm_runtime_paged_decode_attn_f32` → `paged_decode_attn_f32_host` / `ullm_paged_decode_attn_f32_kernel` | page-table address の維持、K/V payload decode、FP8 scale load、F32 accumulate、GQA head mapping の維持 | generic direct typed API と CPU reader 実装済み。HIP native は未実装 |
| Generic prefill | `cached_prefix_attn_f32` / `_flash2`、`PagedDecodeState::prefill_chunk_from_device` | typed cached-prefix reader、または各 query を typed direct reader へ因果長付きで発行 | F32 fused path は不変。typed は後者で正しく実装済み（性能暫定） |
| Existing FP8 cached-prefix family | `cached_prefix_attn_fp8_e4m3*` と HIPRTC `ullm_cached_prefix_attn_fp8_e4m3*` | 現行 API は global K/V scale を取るため、per-token/head FP16 scale plane を引く別 contract が必要 | 再利用しない。payload decode helper の参考のみ |
| AQ4 prefill | `paged_causal_gqa_chunk_f32` / `paged_causal_gqa_chunk_f32_impl` と `ullm_paged_causal_gqa_chunk_f32_kernel` | persistent K/V typed payload と scale plane に対応する reader、writer と同じ page contract | 未実装。AQ4_0 production file は変更禁止 |
| `causal_attn_*` | `causal_attn_f32`, `_flash2`, batch variants | これは temporary contiguous `[T, H, D]` K/V を読む attention で、persistent paged cache は読まない | KV storage dtype の変更対象ではない。source tensor dtype を変える別タスクなら対象 |

`ullm_causal_attn_*` と persistent paged K/V reader を混同しない。前者は
temporary sequence buffer の API であり、本仕様の cache allocation / page
stride / scale metadata を持たない。

### 1.3 設定、確保、ページ管理、diagnostic readback

`PagedDecodeShape` は block table と shape を保ち、logical token `t` を

```text
logical_block = t / block_size
physical_token = block_table[logical_block] * block_size + (t % block_size)
```

へ写像する。この u32 block table と block size は dtype に依存せず不変である。
v0.1 では `crates/ullm-engine/src/kv_cache_dtype.rs` に次を集約した。

- `KvCacheDtype::{F32,F16,Fp8E4M3Fn}`
- `KvCacheDtypes { key, value }`
- `KvCacheLayout`（payload / scale の exact byte accounting）
- `ULLM_KV_CACHE_DTYPE`（K/V 一括）と、優先する
  `ULLM_KV_CACHE_TYPE_K` / `ULLM_KV_CACHE_TYPE_V`

`PagedDecodeState::new` はこの selector を読み、未指定なら F32/F32 になる。
embedding caller は `new_with_kv_cache_dtypes` で環境を経由せず明示指定も
できる。`reset` と serving reset は payload だけでなく FP8 scale buffer も
zero にする。readback は physical payload を F32 に復元し、logical-prefix
readback は block table を再適用する。

AQ4 resident path は別々の `RuntimeBuffer` を直接所有する。これは実稼働
F32 allocation / page management のもう一つの経路だが、今回の変更禁止
範囲である。そのため v0.1 の selector は AQ4 runtime には未伝播である。

追加で確認した engine-side propagation / reporting path は以下である。

- `crates/ullm-engine/src/sq8_generation_runtime.rs` の
  `Qwen3Sq8GenerationRuntime::load` と
  `crates/ullm-engine/src/sq8_serving_runtime.rs` の serving load は layer ごとに
  `PagedDecodeState::new` を呼ぶ。したがって generic allocation 自体には
  selector が伝わる。
- ただし `sq8_serving_runtime.rs` の `Sq8ServingLoadReport` と
  `qwen3_14b_sq8_serving_kv_cache_bytes_per_layer` /
  `qwen3_14b_sq8_serving_total_kv_cache_bytes` は F32 byte count
  33,554,432 B/layer を frozen contract として検証している。typed serving を
  promoted path にする際は `KvCacheLayout` を report に伝え、K/V dtype と
  scale bytes を表示・検証する必要がある。このファイルには同時作業の未コミット
  変更があったため v0.1 では編集しなかった。
- `crates/ullm-engine/src/main_parts/part_03.rs` の benchmark/report JSON は
  `kv_cache_value_dtype: "f32"` と F32-only byte accounting を書く。typed
  full-model integration 時には K/V を別々に report し、FP8 scale bytes を
  `kv_cache_bytes` に含める必要がある。
- `main_parts/part_00.rs` の runtime cached-prefix probe は既存の global-scale
  FP8 experiment を扱う。これは dynamic per-token/head scale contract では
  ないので、v0.1 typed cache の benchmark/quality report に流用してはいけない。

## 2. storage layout と VRAM 収支

### 2.1 一般形

各 plane は token-major / KV-head-major / dim-major であり、payload index は

```text
payload[(physical_token * kv_heads + kv_head) * plane_dim + element]
```

である。F32/F16 はそのまま value を並べる。FP8 payload と scale は別
`RuntimeBuffer` であり、scale index は

```text
scale[physical_token * kv_heads + kv_head]
```

である。scale は正の IEEE FP16 で、writer は head の `max(abs(x)) / 448`
を FP16 に上向き丸めし、E4M3FN finite maximum 448 に収まるようにする。
all-zero head は exact 1.0 scale と zero payload を書く。reset/unwritten row
の zero scale + zero payload は diagnostic readback で zero と扱う。

F16 payload と FP8 scale は 2-byte alignment を必要とする。v0.1 は
`RuntimeBuffer` の独立 allocation（CPU `malloc`、HIP `hipMalloc`）だけを
使い、型付き sub-buffer / odd offset を作らない。native HIP implementation
は vectorized load を入れる前に base alignment と page-stride alignment を
明示的に検証する必要がある。FP8 scalar payload の alignment requirement は
1 byte だが、FP8 x2/x4 vector load を採る場合はその vector の alignment が
別途必要になる。

### 2.2 指定 geometry の exact bytes

以下は `block_size=16`, `cache_blocks=256`, `kv_heads=4`,
`head_dim=value_dim=256`、すなわち 4,096 physical tokens **per layer** の
計算である。F32 と同じ KV allocation budget に page 単位で収まる文脈長も
併記する。model layer 数を掛けても比率は不変である。

| storage | K page | V page | scale/page | K+V page | K+V per layer (4,096 tok) | effective byte/value | 同一 F32 KV budget の blocks | 同一 budget の context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F32 | 65,536 B | 65,536 B | 0 B | 131,072 B | 32 MiB | 4 | 256 | 4,096 |
| F16 | 32,768 B | 32,768 B | 0 B | 65,536 B | 16 MiB | 2 | 512 | 8,192 |
| FP8 E4M3FN + FP16 scale | 16,384 B | 16,384 B | 128 B K + 128 B V | 33,024 B | 8 MiB + 64 KiB = 8.0625 MiB | 1.0078125 per K, 1.0078125 per V | 1,016 | 16,256 |

FP8 scale overhead is `2 / 256 = 0.0078125 B/value` per plane, or 0.78125%
of its one-byte payload. The page rounding matters: the byte ratio is about
3.969x, but a fixed F32 256-page budget fits 1,016 whole FP8 pages, hence
16,256 rather than a fractional-token result. This is a capacity statement,
not a throughput prediction.

## 3. FP8 format selection

### 3.1 E4M3FN for both K and V

OCP E4M3FN is selected for both planes in v0.1. This is not a claim that
unscaled K and V have identical ranges. K is after RoPE and may have a wider
or otherwise different distribution; V may differ too. The design gives each
plane, token, and KV head its own scale, so a K outlier cannot reduce V
precision and a later token never requires rewriting an already stored row.

The selected local evidence is:

- `/opt/rocm/include/hip/amd_detail/amd_hip_ocp_fp.hpp` documents E4M3
  maximum finite magnitude 448 and E5M2 maximum 57,344. It also confirms
  E4M3 has one more mantissa bit than E5M2.
- `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc` already selects
  `__builtin_amdgcn_cvt_f32_fp8(..., 0)` on `__gfx1201__` for OCP E4M3FN.
- `runtime/src/sq8_ck_gfx1201.hip.cpp` contains a local E4M3 RNE encoder
  and E4M3 scale convention. This gives a project-local semantic reference.
- No existing gfx1201 E5M2 paged-attention conversion/kernel was found.
  The local OCP header's E5M2 route is a distinct BF8 conversion path; its
  native availability and generated code on gfx1201 are **unconfirmed**.

E5M2's wider exponent would be useful only if a coarse scale had to span
widely different rows. With a scale over exactly one K or V head row, the
range advantage is intentionally amortized away while E4M3 retains a more
useful mantissa. A long-context actual-K/V range capture has not yet been
made; therefore this is an implementable v0.1 choice, not a claim that
E5M2 can never improve a future quality experiment.

### 3.2 Why the scale is token × KV-head, not page/head

A page/head scale would need a new maximum observed when a later append has a
larger magnitude. Updating it would require rescaling and requantizing all
previous page rows, for which the original F32 inputs may no longer exist.
The same issue applies to a 16-token block scale. A per-token/head scale is
append-local: writer work is one reduction over 256 values, no old payload is
read or rewritten, and K/V can differ safely.

### 3.3 `SQ8_0` relationship

The existing `SQ8_0` artifact uses OCP E4M3FN payload with static BF16
`[128,128]` block-scale metadata for model weights. Its E4M3 representation
and conversion experience are reusable. Its layout and scale contract are
not: persistent KV is dynamically appended, has K/V plane dimensions,
physical page indirection, and requires a scale that can be written without
touching earlier tokens. `SQ8_0` is therefore not used as a KV format name or
as a binary layout.

### 3.4 Conversion cost: what is known and what is not

For FP8 direct decode each K or V payload byte must be converted to F32 and
multiplied by its FP16 scale before F32 score / value accumulation. The
writer must reduce 256 absolute values per head, form/store an FP16 scale,
and encode 256 values. Native gfx1201 E4M3 conversion is available in the
project's existing HIPRTC code; using it is the required native-kernel route.

No cycle-level conversion-cost claim is made here. The current HIP fallback
copies the full typed cache to host for correctness, so it is deliberately
not an eligible throughput path. Only a full-model run with native writer and
reader can answer whether reduced traffic beats conversion and scale loads
after the BH GQA-cooperative redesign.

## 4. ABI and implementation status

`runtime/include/ullm_runtime.h` adds:

```c
typedef enum ullm_kv_cache_dtype {
    ULLM_KV_CACHE_DTYPE_F32 = 0,
    ULLM_KV_CACHE_DTYPE_F16 = 1,
    ULLM_KV_CACHE_DTYPE_FP8_E4M3FN = 2,
} ullm_kv_cache_dtype;
```

and the F32-source / F32-query APIs `ullm_runtime_paged_kv_write_typed_f32`
and `ullm_runtime_paged_decode_attn_typed_f32`. For FP8, `k_scale_buffer` or
`v_scale_buffer` is non-null exactly for the corresponding FP8 plane and is
the raw F16 `[physical_token, kv_head]` plane above. Rust FFI repeats all
size checks before calling C.

The generic engine keeps the existing F32 symbols for exact F32/F32 selection.
Only non-F32 selection invokes the typed ABI. Source-tiled split attention is
F32-only, so a typed cache deliberately takes the direct typed reader instead
of silently treating FP16/FP8 bytes as F32.

The generic prefill path preserves existing F32 `cached_prefix_attn_*`. For a
typed cache it writes the chunk then issues direct typed attention separately
for each causal query length. This is correct because each read's `cache_len`
excludes future rows, but it is an interim performance path. A typed batched
cached-prefix / `paged_causal_gqa_chunk` native reader remains required.

## 5. Validation and promotion boundary

CPU evidence is saved in
`benchmarks/results/2026-07-26/kv-cache-dtype/`. The targeted tests cover:

- F32 default and rejection of `Q8_0`;
- exact F32/F16/FP8 allocation accounting;
- F16, FP8, and mixed K/V physical page write/readback/direct decode;
- typed causal prefill fallback;
- all existing `decoder::tests` F32 cases.

An accidental broad `ullm-runtime-sys --lib` invocation was discovered to
contain opportunistic HIP tests before the required R9700 lock check. It is
not accepted as a GPU measurement or validation artifact. No follow-up GPU
command is allowed until the prescribed lock/service preflight is clean.

Full-model result status at this commit:

| dtype | decode tok/s | prefill tok/s | long-context generated text | status |
|---|---:|---:|---|---|
| F32 | 27.378731 is the prior BH reference, not rerun here | not rerun | not rerun | reference only |
| F16 | unmeasured | unmeasured | unmeasured | no native AQ4 typed route |
| FP8 E4M3FN | unmeasured | unmeasured | unmeasured | no native AQ4 typed route |

The old 20.002232 tok/s estimate is intentionally not used. BH has already
reduced semantic K+V load from 42,434,560 B to 8,486,912 B, so cache-width
ratios cannot substitute for an end-to-end speed result. Capacity accounting
in section 2 remains independently valid.

## 6. Native-kernel handoff after BR

Before editing `runtime/src/ullm_runtime_parts/part_01.inc` or
`runtime/src/ullm_runtime_hiprtc_sources.inc`, rerun:

```bash
pgrep -af 'codex exec' | grep -c '依頼BR'
```

Only a zero result permits the following work:

1. Add direct typed paged writer and reader kernel caches/launchers. Compile
   separate F32/F16/FP8 K/V specializations or a verified equivalent; do not
   add a runtime inner-loop type branch to the hot path.
2. For FP8 reader, load the scale once per `(source token, kv head)` and use
   gfx1201 E4M3 conversion. Preserve GQA cooperative mapping and online
   softmax order from BR's redesign.
3. Add fused Qwen Q/K norm/RoPE typed writer and typed chunk writer/reader,
   then expose matching FFI APIs.
4. In a separately authorized AQ4_0 production change, propagate `KvCacheDtypes`
   through resident layer allocation, operation registry plans, decode,
   cached-prefix, and paged causal GQA paths. Do not repurpose F32 buffers or
   overwrite existing F32 behavior. Update serving/benchmark reports to use
   `KvCacheLayout` rather than frozen F32 byte constants.
5. With R9700 unlocked and the required preflight recorded, collect F32/F16/
   FP8 full-model decode and prefill at long contexts, plus side-by-side
   generated text. The promotion decision must be qualitative text review,
   not a single numerical threshold.
