# P2 AQ4 layer-0 QKV/Z/gate/beta fused GPU probe gate

## 前回の要点

fused probe report schema v2とCPU診断は既存実装で完了していた。HIP GPU、service停止、holdout、数値Go/No-Go、promotionは未実施である。

## 今回の変更点

- clean detached worktree `6082df4966190ae4977b699460a5ecb93fee8e34`から作成したprobe artifact（binary SHA-256 `42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840`、receipt SHA-256 `90e9ef6d383f7ef25e9526659f035e40291ba1a5efa7f8ba36340c8b245d9504`）は前回のまま固定した。
- 今回はgateの観測と復旧を強化した。lock取得後、物理card 2かつgfx1201を検証するamd-smi JSON observerを別プロセスで開始し、初回sample/failure markerを待ってからprobeを実行する。終了時はobserver停止、failure marker不在、sample存在を検証する。observer子プロセスへ親の終了シグナルtrapを継承させない。
- 失敗trapはobserver停止、lock cleanup、service startを独立した戻り値で保持し、lock cleanupが失敗してもservice startを必ず試行する。start後にhealth/GPU/worker操作は実行しない。
- 出力契約はHIP standalone QKV reference（operation `standalone_aq4_matvec_f32`、raw RPB unset、effective 32、source `architecture_default:gfx1201`）、Q/K/V row segments、全sidecarのshape/bytes/rows/case offset/SHA/finite、output layout、report/sidecarのnlink=1までfail closedで検証する。promotionは常にfalseである。
- execute準備ではfresh BASEの`attempts` parentを作成した後、attemptディレクトリをcreate-newで確保する。source checkoutのmodeを信頼せず、probeをattempt内へinstall/chmod `0555`し、runtime copyのregular/nlink=1/SHA-256を検証して、runtime binaryのstat/hashを`runtime-probe-stat.json`へarchiveする。実行対象はこのruntime copyだけである。

## 検証

- 実行: `bash -n`、binary/receipt SHA256SUMS検証、fresh BASEの`MOCK_ARCHIVE_SETUP=1`（source checkout mode `0775`からruntime mode `0555`へ変換するケースを含む）、`PREFLIGHT_ONLY=1`、`MOCK_PREFLIGHT=1`、6 tests（observer mock、runtime copy archive、read-only mock、validatorのwrong-reference/layout negativeを含む）。
- 未実施: GPU kernel probe、fused report/output、health実測、holdout、数値閾値判定、promotion。service stop/startは下記の実行で復旧確認済みである。

## 実行結果（2026-07-15）

- 通常EXECUTEは2回試行した。attempt1はsudo非対話認証不足でservice確認前に終了し、`attempts/attempt1/`へruntime copyと`runtime-probe-stat.json`（runtime SHA-256 `42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840`、stat SHA-256 `9d1855b3f5f8b1375708b25e50a9fd3f90471a883046b06dedcaa96ffc97e9c6`）だけを保存した。
- PTY `sudo -v`後のfresh `attempt2/attempts/attempt1/`はservice stop→stable2停止→lock acquireまで進んだが、observer初回sampleで停止した。実行対象のprobe kernelは未起動で、`run.log`は`observer failed before first sample`、observer failure markerが残った。`amd-smi --showmeminfo vram --showuse --showpower --json`は、実機AMD-SMI 26.2.2/ROCm 7.2.1のsubcommand式CLIと不一致で`AmdSmiInvalidSubcommandException`になった。
- 失敗trapの復旧は完了した。serviceはactive/running、旧MainPID `1834403`から新MainPID `2381944`、workerは`1834494`から`2382329`、`NRestarts=0`である。`/run/ullm/r9700.lock`はuid/gid `1000:1000`、mode `0600`、nlink `1`の新inode `770421`をservice workerが保持している。GPU kernel、numeric threshold、promotionは実施していない（promotion=false/unclassified固定）。
- attempt2の証跡はruntime stat SHA-256 `aeae99d8b4c06d16570f3b119d836a7995fe6f27e7e7c18c291c6244d77ab9a9`、prestate SHA-256 `fab7cebdcbb8304528fa53fec9727fc6612bfe7e8409d77e353be3b44ed88fb5`、run log SHA-256 `eb00c14a402386228ca5dbf656ffaf0271e78e47dcde2e0f16bfa4edf21b04a8`である。

## 次の行動

observer CLIを現行AMD-SMIのsubcommand形式へ別途適合させ、再実行前にpreflightで確認する。今回の失敗証跡を限定commitとして保存し、GPU kernel実行・閾値判定・promotionは行わない。
