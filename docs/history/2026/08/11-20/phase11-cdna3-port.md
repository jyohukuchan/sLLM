# Phase 11 CDNA3 port history

## 2026-08-14: 詳細計画の作成

- Phase 11をexact `gfx942`、wave64、BF16、FNUZ FP8の実装・compile/oracle phaseとして具体化した。
- model storageのOCP E4M3FNをload時にE4M3FNUZへ数値変換し、テスト専用FNUZ modelを作らない既存方針を維持した。
- AMDの公開MI300X例でVMMなしが示されるため、opaque KV契約を維持した`contiguous-resident` providerを
  Phase 11へ追加する。Paged Attentionへの選定変更や実行時silent fallbackではない。
- 実機evidenceはPhase 12へ分離し、未所有GPUをPhase 11実装完了のblockerにしない。
- 詳細は[active plan](../../../../plans/active/2026/08/11-20/phase11-cdna3-port.md)を正とする。
