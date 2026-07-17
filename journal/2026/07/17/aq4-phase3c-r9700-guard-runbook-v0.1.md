# AQ4 Phase 3c: R9700-only guard and H9 telemetry runbook update v0.1

## 前回の要点

- Phase 1/2/2b/3bとPhase 3c-prepにより、H8（単純な深さ方向の蓄積）とH6（M=1/cold診断の構成差）は既知のlayer 0差分の説明として棄却方向となり、H5（HIP fused kernel固有）が有力だが未確証のままである。
- 既存Phase 3c runbookは`HIP_VISIBLE_DEVICES=1`、global runtime device `1`、`gfx1201`を固定していたが、その物理device mappingは過去の記録への依存であり、実行時に対象architecture/nameを機械的にassertしていなかった。
- ユーザーは今回のGPU windowをR9700限定で承認し、H9（ECC、温度、clock/power、driver/firmwareを含むハードウェア固有要因）も同時に記録することを条件にした。service/systemd/active manifestおよび07/16に停止したP3 harnessは対象外である。

## 今回の変更点

- `tools/query-hip-device-identity.cpp`を追加した。これはhost-only `g++` buildで`libamdhip64`へlinkし、`HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1`下のfiltered HIP ordinal 0だけに対して、可視GPU数、architecture、name、PCI BDF、HIP versionを読み取り専用でJSONへ記録する。device memoryの確保、stream作成、kernel launchは行わない。可視GPU数が1以外、architectureが`gfx1201`以外、name/BDFが欠ける場合はnon-zeroで終了する。
- runbookはHIP guardのBDFを唯一の`amd-smi --gpu`対象にし、ASIC recordで同一BDF、`gfx1201`、R9700のPCI device ID `0x7551`、non-empty market nameをassertするようにした。対象指定なしの`amd-smi`、`amd-smi list`、V620へのqueryは手順に含めていない。
- H9 telemetryは同じBDFに限定して、ECC/error block、bad page、clock、power、temperature、DPM performance level、driver/IFWI、firmwareをtrace前後に保存する。metric非対応時もstderr/exit codeをevidenceとして残し、設定変更、daemon導入、再試行は行わない。
- lock/guard failure時はtraceを起動しない。traceが失敗した場合もpost health telemetryとexit statusを残してから終了するが、checksum/比較へは進まない。4段階のrelative L2区分は`<=1e-5`、`>1e-5 && <=1e-3`、`>1e-3 && <=1e-2`、`>1e-2`へ明確化した。
- CPU-only検証として、guard sourceを`g++ -D__HIP_PLATFORM_AMD__ -lamdhip64`でcompileし、引数拒否JSONを確認した。runbookのshell blockは`bash -n`、3個の埋め込みPython blockは`compile()`で構文確認した。GPU query、GPU kernel、service/systemd/active manifest、P3 harnessにはこの準備段階で触れていない。

## 次の行動

- このguard/runbook更新を独立commitとして固定する。
- 更新済みrunbookのblockを一回だけ実行する。HIP/ASIC guardが失敗した場合はGPU traceを起動せず、evidenceとexit codeだけを保存する。成功時はR9700 health telemetry、CPU/GPU 10 stage比較、H5/H9の限定的な判定をjournalへ追記する。
