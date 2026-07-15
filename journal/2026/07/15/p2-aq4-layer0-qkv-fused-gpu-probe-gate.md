# P2 AQ4 layer-0 QKV/Z/gate/beta fused GPU probe gate

## 前回の要点

fused probe report schema v2とCPU診断は既存実装で完了していた。HIP GPU、service停止、holdout、数値Go/No-Go、promotionは未実施である。

## 今回の変更点

- clean detached worktree `6082df4966190ae4977b699460a5ecb93fee8e34`から作成したprobe artifact（binary SHA-256 `42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840`、receipt SHA-256 `90e9ef6d383f7ef25e9526659f035e40291ba1a5efa7f8ba36340c8b245d9504`）は前回のまま固定した。
- 今回はgateの観測と復旧を強化した。lock取得後、物理card 2かつgfx1201を検証するamd-smi JSON observerを別プロセスで開始し、初回sample/failure markerを待ってからprobeを実行する。終了時はobserver停止、failure marker不在、sample存在を検証する。observer子プロセスへ親の終了シグナルtrapを継承させない。
- 失敗trapはobserver停止、lock cleanup、service startを独立した戻り値で保持し、lock cleanupが失敗してもservice startを必ず試行する。start後にhealth/GPU/worker操作は実行しない。
- 出力契約はHIP standalone QKV reference（operation `standalone_aq4_matvec_f32`、raw RPB unset、effective 32、source `architecture_default:gfx1201`）、Q/K/V row segments、全sidecarのshape/bytes/rows/case offset/SHA/finite、output layout、report/sidecarのnlink=1までfail closedで検証する。promotionは常にfalseである。

## 検証

- 実行: `bash -n`、binary/receipt SHA256SUMS検証、`PREFLIGHT_ONLY=1`、`MOCK_PREFLIGHT=1`、5 tests（observer mock、read-only mock、validatorのwrong-reference/layout negativeを含む）。
- 未実施: GPU probe、service stop/start、ROCm SMI/KFD実測、health実測、holdout、数値閾値判定、promotion。

## 次の行動

この限定commitを親worktreeへcherry-pickする。君の明示的なGPU/service実行許可までは、preflight/mockと静的契約検証だけを行う。
