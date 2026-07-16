# AQ4 Phase 3c-prep GPU window runbook v0.1

## 前回の要点

- fused kernelの静的レビューでは、有効AQ4 payloadに対する07/14規模の高確信度の通常算術バグは未発見だった。無効scale-indexのsilent skipとRPB compile/cache不整合は別の条件付き問題として記録した。
- Phase 3c stage toolingはCPU-onlyでbuild/test済みであり、GPU production M=1のlayer 0 bufferを10 stageでD2Hし、CPU streamと比較できる。GPU、service、systemd、active manifest、P3 harnessは実行・変更していない。

## 今回の変更点

- `docs/plans/aq4-phase3c-gpu-window-runbook-v0.1.md` に、承認後一回だけ実行する正確なcommandを記録した。tooling source identityは`5a0fb4c50476d5153ced22bd6847c2729bfdb975`に固定した。
- 07/14/Phase 1と同じ3 context、CPU hybrid input、package、global device index 1 / `HIP_VISIBLE_DEVICES=1` / `gfx1201`、R9700 lock、7つのlayer 0 fusion guard、RPB値を明文化した。
- active serviceを操作しないfail-closed lock、P3 pathと別のoutput root、single-use/no-retry、incomplete evidence保存、comparison schema/checksumの判定を定義した。
- relative L2 `1e-5`、`1e-3`、`1e-2`を、それぞれ丸めと両立・有意差・強い候補/contract failureの調査帯域として定義した。これはfixの承認基準ではない。

## 次の行動

1. GPU windowの明示承認を待つ。承認前にrunbookのcommandを実行しない。
2. 承認後はrunbookを一回だけ実行し、最初に有意となるstageまたは測定無効理由をjournalへ保存する。
3. evidenceが揃ってからのみ、Phase 4の修正候補を別途提示する。今回の範囲ではkernel/source fixを行わない。
