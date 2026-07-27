# RDNA4 / CDNA3 共通ハードウェア・マイクロベンチ設計

## 契約

`tools/hw-microbench-rdna4-cdna3.hip.cpp` は推論ランタイムから独立した
HIP バイナリである。`tools/run-hw-microbench-rdna4-cdna3.sh` が 1 コマンドで
対象 arch 向けに build、ISA 監査、実行、telemetry 保存を行う。

すべての GEMM は dense `2*M*N*K` FLOPs、HIP event で囲んだ kernel launch
だけを分子/分母にする。5 warmup、11 サンプルの中央値、各サンプル 10 launch
を両 arch で固定する。プロファイラ範囲時間は使用しない。STREAM は float
read=4N、copy=8N、triad=12N bytes とし、各 vector 256 MiB（R9700 の 64 MiB
Infinity Cache と MI300X の 256 MiB LLC より大きい）を既定値とする。

GEMM shape は 256³、1024³、4096³、Qwen3-14B hidden projection
`M=256,N=5120,K=5120` である。最後の shape は peak-only の値と推論形状を
区別するために必ず保存する。

## ISA と FP8

- gfx1201: rocWMMA `float8_t` / `bfloat16_t`。静的監査は
  `v_wmma_f32_16x16x16_fp8_fp8` を必須にする。
- gfx942: rocWMMA `float8_fnuz_t` / `bfloat16_t`。静的監査は
  `v_mfma_f32_16x16x32_fp8_fp8` を必須にする。

FP8 host data は canonical OCP E4M3FN byte で、NaN (`0x7f`,`0xff`) と
negative zero (`0x80`) を生成しない。gfx942 では既存
`sq8_fnuz_prepack.rs` の規約通り、同一 raw byte の FNUZ 値が OCP の半分に
なる。A と B の両方を FNUZ として読むので accumulator fragment を正確に
`×4` してから store する。CPU oracle は OCP 値で GEMM を計算し、BF16 と
FP8 の双方が timing 前に pass しなければ停止する。従って二つの GPU が
計算する数値は同一である。

## peak の記録規約

run wrapper は peak 値を暗黙に埋めない。実行者は環境変数で明示する。

| GPU | memory | BF16 | FP8 | 出典 / 注記 |
|---|---:|---:|---:|---|
| R9700 | 640 GB/s | 未確認 | 383 TFLOPS | AMD product page は 640 GB/s、FP8 matrix 383、FP16 matrix 191 を掲載。BF16 単独 peak は同ページでは未掲載なので、BF16 比率に 191 を流用してはいけない。 |
| MI300X | 5300 GB/s | 1307.4 TFLOPS | 2614.9 TFLOPS | AMD MI300X product page。レンタル host の `rocminfo` と `amd-smi` も成果物に保存して単一 MI300X であることを確認する。 |

R9700 の BF16 peak を AMD が明示するまでは、`HW_MB_BF16_PEAK_TFLOPS=0`
で走らせて `%` を未確認扱いにする。測定値そのものを推測で補わない。

## rental 運用

`run-sq8-cdna3-mi300x-validation.sh --stage hw_microbench` は P0 ではない。
`preflight,cpu,hiprtc,build,isa,physical` の全 `.done` がなければ fail する。
従って P0 失敗を隠して benchmark は走らない。stage は idempotent で
`state/hw_microbench.done` を使う。lease が厳しい場合の切捨て対象であり、
P0 を省略して得る時間に置き換えてはならない。

各 run は `telemetry-before.json` / `telemetry-after.json`、`rocminfo`、
`amd-smi` の実クロック、power、temperature、throttle 表示を同じ results
directory に残す。`THROTTLED` 一語ではなく実クロックを判定材料にする。
開始は edge <=45 C を待つ（40 C 固定は禁止）。

## 比較表

`benchmarks/results/2026-07-27/hw-microbench/comparison.md` を正本の器にし、
各 GPU の JSONL から値を転記する。未測定欄を 0 で埋めない。
