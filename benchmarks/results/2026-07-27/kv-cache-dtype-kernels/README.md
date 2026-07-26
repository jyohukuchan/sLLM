# KV cache dtype kernel measurement — 2026-07-27

## 結論

AQ4_0 Qwen3.5-9B の persistent paged KV を F32 / FP16 / OCP FP8
E4M3FN (S1E4M3) で native HIP writer/reader に接続した。F32 の既定経路は
変更せず、非 F32 のみ typed native path を使う。FP8 は K/V ごとに独立した
`[physical token, KV head]` FP16 scale を持つ。

- 3,968-token prefix / 128-token decode では、F16 は **68.938 tok/s**
  (F32 65.748 の 1.0485x)、FP8 は **68.358 tok/s** (1.0397x) だった。
- canonical M=128 prefill は現行 typed causal reader が F32 WMMA reader より
  遅い。4,095 token では F16 464.278、FP8 440.562、F32 899.934 tok/s。
- 実モデル load により F32 4,096、F16 8,192、FP8 16,256 logical token の
  cache allocation を確認した。FP8 の 16,256 は達成した。
- 3,968-token natural-language input の 64 token 生成は F32/F16/FP8 で
  token ID と decoded text が完全一致した。64 token 上限のため、三者とも
  reasoning 冒頭で止まり、最終の短い回答までは到達していない。

## 計測条件

- GPU: R9700 (`gfx1201`, AMD SMI index 2) のみ。プロセスには
  `HIP_VISIBLE_DEVICES=1` を渡し、runtime は CPU=0 / HIP=1 と検証した。
- AQ4_0 package: Qwen3.5-9B、8 full-attention layers、16 Q heads、4 KV heads、
  head/value dim 256、page/block size 256、M=128 prefill。
- timing: 各 prefill/decode row は同一長の warm-up 1 回 + timed repeat 5 回。
  model load、warm-up、request construction、reset は時間外。decode は
  3,968-token prefix の後に 128 token を測った。
- non-F32 rows: `ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL=1`、
  `..._SPLIT_KERNEL=1`、`..._KV_WRITE_KERNEL=1` を指定した。host/staging
  fallback は利用できない。
- thermal gate: 各 condition の前に edge <= 45 C。26 conditions 全てが
  edge 37--45 C で通過した。gate 中の観測 edge 範囲は 37--73 C。

## Full-model AQ4_0 throughput

### Prefill (tok/s, five-repeat mean)

| prompt tokens | F32 | F16 | FP8 E4M3FN |
|---:|---:|---:|---:|
| 128 | 1019.612 | 979.171 | 976.188 |
| 512 | 1020.838 | 900.256 | 887.719 |
| 1024 | 1004.636 | 799.390 | 776.468 |
| 2048 | 966.915 | 648.768 | 631.446 |
| 4095 | 899.934 | 464.278 | 440.562 |

### Long-context decode (3,968 prefix + 128 generated tokens)

| dtype | tok/s | vs F32 |
|---|---:|---:|
| F32 | 65.748 | 1.0000x |
| F16 | 68.938 | 1.0485x |
| FP8 E4M3FN | 68.358 | 1.0397x |

The 128 generated token IDs are identical across all three rows. This is a
full-model timing, not a profiler-range time.

## Actual cache allocations

| dtype | requested logical context | physical KV tokens | all 8 self-attention layers | note |
|---|---:|---:|---:|---|
| F32 | 4,096 | 4,096 | 256 MiB | 16 pages/layer |
| F16 | 8,192 | 8,192 | 256 MiB | 32 pages/layer |
| FP8 E4M3FN + FP16 scales | 16,256 | 16,384 | 258 MiB | 64 256-token pages/layer |

The FP8 logical 16,256 configuration rounds allocation up to 16,384 physical
tokens because AQ4_0 has 256-token pages. Therefore it uses 258 MiB, 2 MiB
above the F32/F16 256 MiB rows; the model load itself succeeded. The 64.5 MiB
at 4,096 logical tokens remains the exact per-design FP8 total for eight
full-attention layers.

## F32 regression evidence

The current-source SQ8_0 F32 control reran BR's serial-GQA oracle at prompt
lengths 128, 512, 1024, 2048, and 4095. Every final hidden-state and logit
byte stream matched (`10/10`). The BH decode-control result is 27.576901 tok/s
(five repeats, 16 measured steps), above the historical 27.378731 tok/s
reference; it is an independent SQ8_0 control and must not be conflated with
the AQ4_0 throughput table.

## Files

- [`run-r9700-window.sh`](run-r9700-window.sh): one-window safety wrapper and
  exact command record.
- [`run-20260727T021656+0900`](run-20260727T021656+0900): raw AQ4, SQ8, thermal,
  service, binary-identity, quality, and comparison artifacts.
- [`summary.json`](summary.json): compact machine-readable index.
- [`conditions.md`](conditions.md) and [`accounting.md`](accounting.md): timing
  boundary and model-specific accounting scope.
- [`prepare-quality-prompt.py`](prepare-quality-prompt.py) and
  [`decode-quality-output.py`](decode-quality-output.py): deterministic
  long-context prompt and decoded side-by-side output tooling.

## Promotion decision

No served-model promotion was performed. This change does not add an
authorized fail-closed served-model selector for `ULLM_KV_CACHE_DTYPE`, so it
cannot safely switch a manifest by itself. The measured prefill regression also
makes an activation inappropriate before a native typed causal-prefill
optimization. This is not a judgement based on a numeric threshold: the
long-context text was retained as qualitative evidence and is identical in this
sample.
