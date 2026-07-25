# `SQ8_0` R9700 handwritten-kernel optimization Phase 0

Date: 2026-07-26

## 前回の要点

- 独立 `SQ8_0` の R9700 最適化 projection は、手書き WMMA 本体ではなく `runtime/src/sq8_ck_gfx1201.hip.cpp` の gfx1201 固定 CK `DeviceGemmMultipleD_ABScale` / `DeviceGemmXdlUniversal` であることが既知だった。
- generic `SQ8_0` matvec 4本は scalar OCP E4M3FN decode、F32 FMA、`float partial[256]` の LDS tree を使う。従って、source の見た目だけで generic 経路を最初の最適化対象にはできなかった。
- ユーザー方針は、CK の内部を直接いじるのではなく、将来の CDNA3 案 A（手書き MFMA）へ引き継げる手書き経路を維持することである。外部 ABI / dispatch と本番 symbol の変更は本タスク外とした。

## 今回の変更点

- R9700 のみ（AMD SMI GPU 2、`gfx1201`、PCI `0000:47:00.0`）で、ROCTx selected region を使った `rocprofv3` を取得した。decode は 1024-token seed、4 warm-up を除外し、cache `1028 -> 1044` の16 M=1 stepだけを scope に入れた。generic `ullm_sq_fp8_matvec_{f32,batch,pair,triple}_kernel` は decode/prefill のどちらにも出現しなかった。
- decode の summed kernel-time 上位は `ullm_paged_decode_attn_f32_kernel` 50.9968%、CK projection 合計 40.1305%、LM-head BF16 matvec 4.0754%、`ullm_segmented_rmsnorm_f32_kernel` 1.9049%、`bf16_to_f32` 0.7336%、`quantize_activation_block128` 0.7022% だった。CK dispatch は `16 * 40 layers * 7 projections = 4,480` 回で、実際の projection 境界を確認した。
- M=128 prefill では `ullm_cached_prefix_attn_f32_flash2_kernel` が 74.8409%、CK projection 合計が 11.3154%、KV write が 3.4418%、activation quantizer が 1.5623% だった。prefill のプロファイル外 tok/s は未確認であり、プロファイラの range 時間を throughput として扱わなかった。
- selected handwritten bodiesを静的監査した。activation quantizer は 128-element LDS max tree、RMSNorm は `partial[256]` LDS tree、Flash2 は QK/max/sum の反復 `reduce[256]` LDS tree を使う。paged decode は通常 source では wave-shuffle reduction を使うが、共有メモリ fallback の実測時有効性は環境記録不足のため未確認である。
- generic reference は `global_load_u8` を含む narrow payload load だが未選択だった。一方、選択 CK code object は `buffer_load_b128` を含むため、現在の serving projection に「128-bit load が無い」という証拠はない。wide-load を最初に generic へ適用する優先度は下げた。
- offline metadata では、選択 CK 128x256 forms が LDS 36,864 B / VGPR 242 または 175、256x128 form が LDS 34,816 B / VGPR 154、128x128 form が LDS 18,432 B / VGPR 100 だった。64 KiB LDS/CU の LDS-only 計算では前二者は1 CTA/8 wave32（32-wave reference の25%）であり、手書き projection の資源設計を測る根拠になった。
- KV-inclusive logical stream metric を固定した。manifest 集計の 280 projection payload `13,212,057,600 B`、BF16 scales `1,612,800 B`、BF16 LM head `1,555,824,640 B`、F32 KV read/write を含め、scope midpoint `C=1036` で `15,109,299,200 B/token`。unprofiled decode 5 repeat の平均 `15.294955751 tok/s` は 640 GB/s reference に対し `eta_logical=36.1088%`（`231.096063 GB/s` logical rate）だった。物理 HBM counter 効率は未確認である。
- raw CSV、telemetry、offline metadata、集計、service record を `benchmarks/results/2026-07-26/sq8-r9700-handwritten-kernel-phase0-v0.1/` に保存し、`docs/plans/sq8-r9700-handwritten-kernel-optimization-plan-v0.1.md` を追加した。
- `ullm-openai.service` は隔離計測ごとに停止後復旧した。prefill window の自動復旧は一度 `start-limit-hit` になったため、isolated profiler process が残っていないことを確認して `reset-failed` と同じ service の start を実施した。最終確認（04:04:33 JST）は `ullm-openai.service=active/running/enabled`、`llama-qwen35-udq4.service=inactive/dead/disabled`。後者は起動していない。

## 次の行動

1. isolated `quantize_activation_block128` wave-shuffle prototype を最初に作り、metadata・byte/scale differential・同一 scoped profile を取得する。期待効果の kernel-time 上限は decode 0.7022%、prefill 1.5623%。
2. 次に `ullm_segmented_rmsnorm_f32_kernel` の LDS tree を wave-shuffle に置換した isolated prototype を同じ gate で比較する。kernel-time 上限は decode 1.9049%、prefill 0.5434%。
3. Flash2 は prefill の最大対象だが、softmax reduction の数値 gate を分離してから扱う。kernel-time の Amdahl 上限 74.8409% は達成効果ではない。
4. 低リスク候補の証跡形式が固まった後、CK を比較対照にした gfx1201 手書き projection を private symbol で研究する。共通化するのは canonical `SQ8_0` payload/scale semantics・validation・differential harness のみとし、wave32/R9700 body と CDNA3 案 A のwave64/MFMA bodyは分離する。
