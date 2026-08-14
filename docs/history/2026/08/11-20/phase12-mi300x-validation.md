# Phase 12 MI300X validation history

## 2026-08-15: VM取得延期とlocal先行queue

- ユーザーが十数時間以上MI300X cloudを継続管理できないため、本Phaseを`ready`のまま保持し、VMを起動しないことを
  固定した。
- 待機中はPhase 13以降をlocal forward queueで先行する。再開時はlatest mainからexact `gfx942` candidateを再buildする。
- Phase 12 matrixはQwen3.5 4B/9B BF16/FP8、contiguous-resident KV、service、性能比較のまま維持し、先行した
  Gemma/NVFP4/MoEを自動追加しない。

## 2026-08-14: Hot Aisle実機計画の作成

- Hot Aisle Small VMのMI300X x1をPhase 12に採用する計画を作成した。192 GB HBM3、8/13 CPU core、
  224 GB RAM、12 TB NVMeはsingle GPU/batch 1の4B/9B BF16・FP8と限定27B FP8 spotに十分と判断した。
- multi-GPU、Infinity Fabric、RCCL/RDMA、bare-metal固有挙動、別CDNA3 SKUはこの一台の証拠範囲外とした。
- 標準予定を10〜12 GPU時間、現実的な上限を16時間、必要な場合だけ追加4時間とし、2〜3時間のpreflightと
  6〜8時間のintegration/performanceを別sessionに分割する。
- VMMなしを想定し、first-hourでexact tuple、FNUZ、contiguous-resident KV、profilerをstop/go判定する。
- 詳細は[active plan](../../../../plans/active/2026/08/11-20/phase12-mi300x-validation.md)を正とする。
