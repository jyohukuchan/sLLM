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

## GEMM の解釈（選択 A）

このベンチは現時点では **(A) 素朴な WMMA / MFMA 実装を両 arch で比較する**。
各 CTA は一つの `16x16` 出力 tile を担当し、K loop の operand fragment を
global memory から直接 `load_matrix_sync` する。LDS タイリングは無い。静的
resource audit の GEMM `LDS=0` はこの実装選択と整合する（private memory と
spill が無いことは別の健全性確認であって、shared-memory tile の証拠ではない）。

従って表の TFLOPS は行列演算器のピーク値ではない。global-memory traffic、
fragment load、CTA 単位の小tile、store を含む naive WMMA/MFMA の実効値である。
この選択は同一ソース・同一tileの RDNA4/CDNA3 比較と、`256x5120x5120` を
uLLM の projection 形状に対応付けるためには有用だが、「MI300X matrix-core
peak」を答える値としては使わない。peak を問う後続作業は LDS タイル化を別の
benchmark として実装・明記する。

## peak の記録規約

run wrapper は peak 値を暗黙に埋めない。実行者は環境変数で明示する。

| GPU | memory | BF16 | FP8 | 出典 / 注記 |
|---|---:|---:|---:|---|
| R9700 | 640 GB/s | 191 TFLOPS | 383 TFLOPS | AMD product page は 640 GB/s、FP8 matrix 383、FP16 matrix 191 を掲載。BF16 単独 peak は同ページでは未掲載である。BF16 191 は 2026-07-27 のユーザー提供値で、RDNA4 の BF16=FP16 throughput 根拠と `383 ≈ 2×191` で整合するが、AMD の BF16 単独記載として扱わない。 |
| MI300X | 5300 GB/s | 1307.4 TFLOPS | 2614.9 TFLOPS | AMD MI300X product page。レンタル host の `rocminfo` と `amd-smi` も成果物に保存して単一 MI300X であることを確認する。 |

R9700 は上記ユーザー提供値を明示して `HW_MB_BF16_PEAK_TFLOPS=191` で走らせ、
比率欄には同値に対する割合を記録する。AMD が BF16 単独 peak を明示するまでは、
出典欄でその区別を維持する。測定値そのものを推測で補わない。

## rental 運用

`run-sq8-cdna3-mi300x-validation.sh --stage hw_microbench` は P0 ではない。
`preflight,cpu,hiprtc,build,isa,physical` の全 `.done` がなければ fail する。
従って P0 失敗を隠して benchmark は走らない。stage は idempotent で
`state/hw_microbench.done` を使う。lease が厳しい場合の切捨て対象であり、
P0 を省略して得る時間に置き換えてはならない。

各 measurement group は `telemetry-<group>.jsonl` に 250 ms 間隔で
`amd-smi metric --gpu INDEX --temperature --clock --power --violation --json`
の実サンプルを残し、最後の sample は `-latest.jsonl` にも残す。タイミング前は
12 秒以上の GPU clock warmup を行い、最後の連続3 GFX sample が 1 GHz 以上かつ
最大差 5% 以下でなければ fail-closed にする。`clock-steady.json` は判定に用いた
全 sample と最後の3 sample を残す。これにより `THROTTLED` 一語や窓の前後値を
測定クロックの代用にしない。開始は edge <=45 C を待つ（40 C 固定は禁止）。

## 比較表

`benchmarks/results/2026-07-27/hw-microbench/comparison.md` を正本の器にし、
各 GPU の JSONL から値を転記する。未測定欄を 0 で埋めない。
