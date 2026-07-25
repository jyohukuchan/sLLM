# Independent SQ8_0 CDNA3 (gfx942) port Phase 0

Date: 2026-07-26

## 前回の要点

- 独立 `SQ8_0` の現在の本番高速経路は RDNA4 (gfx1201 / R9700) 向けであり、CDNA3 (gfx942) の実機はこのホストに無い。
- 本タスクの範囲は Qwen3-14B-FP8 の独立 `SQ8_0` のみである。Qwen3.5 `AQ4_0` の旧48 QKV/Z tensor overlay とは別系統として扱う。
- gfx942 は ROCm 7.2.1 でオフラインコンパイル対象として利用できる一方、differential、timing、実測 occupancy は実機待ちである。

## 今回の変更点

- ソース棚卸しの結果、最適化された独立 `SQ8_0` projection は手書き WMMA カーネルではなく、`runtime/src/sq8_ck_gfx1201.hip.cpp` の gfx1201 固定 CK `DeviceGemmXdlUniversal` だった。`validate_device()`、Cargo feature `rocm-ck-gfx1201`、R9700 identity gate、worker protocol/profile はすべて CDNA3 にそのまま流用できない。
- 一方、`runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc` の4つの generic matvec は scalar OCP E4M3FN-to-F32 decode、F32 FMA、`float partial[256]` の LDS tree であり、WMMA/MFMA/shuffle を使わない。256-thread CTA は gfx942 では four wave64 となる。
- 独立作業ディレクトリで、実行時と同じ HIPRTC option `--offload-arch=gfx942 --std=c++17 -O3` により既存 matvec source をコンパイルした。23,440-byte HSACO が生成され、4 kernel はすべて wave64、LDS 1024 B、spill 0、compiler annotation `Occupancy: 8` だった。

| kernel | VGPR | SGPR | LDS |
|---|---:|---:|---:|
| `ullm_sq_fp8_matvec_f32_kernel` | 20 | 54 | 1024 B |
| `ullm_sq_fp8_matvec_batch_f32_kernel` | 20 | 59 | 1024 B |
| `ullm_sq_fp8_matvec_pair_f32_kernel` | 19 | 52 | 1024 B |
| `ullm_sq_fp8_matvec_triple_f32_kernel` | 19 | 56 | 1024 B |

- この compile-pass は native FP8 route の証拠ではない。ISA は software OCP decoder、scalar `v_fmac`、LDS read/write であり、`v_mfma`、`v_wmma`、native FP8 conversion は無かった。従って「静的に通るが、実行意味・数値・性能は未確認」の分類である。
- gfx1201 WMMA intrinsic を gfx942 へ直接向けた probe は `gfx12-insts,wavefrontsize32` を要求して失敗した。対照的に gfx942 FNUZ FP8 probe は `v_mfma_f32_16x16x32_fp8_fp8` と `v_mfma_f32_32x32x16_fp8_fp8` を実際に出力した。最小 16x16x32 probe は wave64、VGPR 6、SGPR 14、LDS 0、spill 0、compiler annotation `Occupancy: 8` だった。
- ROCm header と256 byte enumerationを確認した。独立 `SQ8_0` は OCP E4M3FN raw payload + `[128,128]` scale であるのに対し、gfx942 MFMA は FNUZ input である。有限 OCP 254 code は、OCP negative-zero `0x80` を FNUZ `0x00` へ正規化し、scaleを2倍すれば `OCP(raw) = 2 * FNUZ(mapped_raw)` を満たす。payload/activation両方を変換する native MFMA route では scale product が4倍になる。OCP NaN、`0x80` の実artifact出現、scale範囲、fragment/lane layout、実機数値は未確認であり、raw payload の直接 MFMA 投入は不可と判定した。
- `runtime/src/sq8_ck_gfx1201.hip.cpp` を隔離コピーで gfx942 device compile すると補助 quantizer/BF16 conversion は compile できたが、runtime の gfx1201 gate と build feature が非gfx1201を拒否し、CK GEMM instance も standalone compile では materializeされない。この結果は「compile-pass only / runtime semantic invalid」であり、移植可能性の証拠ではない。
- CDNA3 decode 帯域指標は SKU/partition 固有に再定義する。現在の F32 KV cache の論理 read は `327,680 * context_length B/token`、write は `327,680 B/token`（4096 context では read 1.25 GiB/token）。実機では payload+scale、KV、activation/output/page/workspaceを固定式で計上し、HBM peak は実SKUとXCD/NPS partitionから選ぶ。TCC counter由来の実 HBM bytes は logical metricと別に報告する。
- 本番 build/release tree、`/opt/ullm`、`/etc/ullm/served-models/active.json`、service lifecycle/configurationには変更を加えていない。今回の repository changes は計画とこの journal のみである。

## 次の行動

- 正確な `gcnArchName` による gfx942 selector、OCP-to-FNUZ byte/scaling oracle、artifact scan を作り、既存 gfx1201 body/ABI/dispatchを保持したまま Phase 1 を進める。
- FNUZ prepack + native MFMA と dequant-to-FP16/BF16 control を隔離 prototype として作り、gfx942 HSACO の ISA/resource metadata を継続監査する。
- 実機到着後にのみ、kernel/end-to-end differential、actual occupancy/residency、HBM/L2/XCD profiling、partition別 timing を実施する。それらの gate を通るまで native path の本番選択・final activation は行わない。
