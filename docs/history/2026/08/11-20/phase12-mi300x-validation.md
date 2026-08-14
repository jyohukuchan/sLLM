# Phase 12 MI300X validation history

## 2026-08-14: Hot Aisle実機計画の作成

- Hot Aisle Small VMのMI300X x1をPhase 12に採用する計画を作成した。192 GB HBM3、8/13 CPU core、
  224 GB RAM、12 TB NVMeはsingle GPU/batch 1の4B/9B BF16・FP8と限定27B FP8 spotに十分と判断した。
- multi-GPU、Infinity Fabric、RCCL/RDMA、bare-metal固有挙動、別CDNA3 SKUはこの一台の証拠範囲外とした。
- 標準予定を10〜12 GPU時間、現実的な上限を16時間、必要な場合だけ追加4時間とし、2〜3時間のpreflightと
  6〜8時間のintegration/performanceを別sessionに分割する。
- VMMなしを想定し、first-hourでexact tuple、FNUZ、contiguous-resident KV、profilerをstop/go判定する。
- 詳細は[active plan](../../../../plans/active/2026/08/11-20/phase12-mi300x-validation.md)を正とする。
