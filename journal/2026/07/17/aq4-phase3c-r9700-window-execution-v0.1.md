# AQ4 Phase 3c: R9700 GPU window execution v0.1

## 前回の要点

- Phase 3c-prepはCPU-onlyで完了し、H8/H6は既知のM=1/cold layer 0差分の説明として棄却方向、H5（GPU fused kernel固有）は有力だが未確証だった。
- ユーザー承認の条件に従い、commit `53e12823878f143e8fe67fbf7a5f4391262eb84e`でrunbookへR9700-only architecture guardとH9 health telemetryを追加した。guardはfiltered HIP ordinal 0をread-onlyで確認し、同じPCI BDFだけを`amd-smi`へ渡す。
- active production service、systemd unit、active manifest、07/16に停止したP3 harnessは今回も変更しない。lock取得失敗時は待機・再試行しない契約である。

## 今回の変更点

### 実行結果

- 更新済みrunbook blockを**一回だけ**実行した。read-only preflightは、既存regular fileの`/run/ullm/r9700.lock`、未作成の専用OUT root、trace tooling sourceのfrozen diff、guard sourceのHEAD clean、必要なfixture/package/compiler/`amd-smi`の存在をすべて確認した。
- R9700 architecture guardは`status=valid`だった。`HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1`で可視HIP GPU数は1、filtered ordinal 0は`gfx1201`、nameは`AMD Radeon Graphics`、PCI BDFは`0000:47:00.0`だった。続く同一BDFのtargeted `amd-smi` recordも`gfx1201`、PCI device ID `0x7551`、market name `AMD Radeon Graphics`だった。V620を対象にしたquery、`amd-smi list`、対象指定なしのGPU queryは行っていない。
- CPU input identityは`status=valid`で、固定3 contextのembedding residualがpackage BF16 passthroughとbit-exactだった。CPU stage stream（24,692,172 bytes）とCPU report（`status=valid`）までは作成された。
- しかしtrace開始時に`flock -n`が失敗した。read-only `lslocks`確認では`/run/ullm/r9700.lock`をactive gatewayの`ullm-openai-gat` PID `1218698`が`FLOCK WRITE`で保持していた。runbook規約に従い、lock待機、service操作、lock作成/修復、GPU traceの再試行を行わなかった。
- `gpu-trace.exit-status.txt`は開始・終了とも`2026-07-17T02:22:53+00:00`、exit code `1`である。`gpu-trace/`、`kernel-stages.f32le`、`cpu-gpu-stage-compare/`は存在しないため、GPU kernelは起動しておらず、30 record比較も実行されていない。preflight file作成からlock failure receiptまでのartifact時刻差は約45.62秒だった。

### H9 GPU health telemetry（R9700 BDF `0000:47:00.0`のみ）

実行前・後の`gpu-health-*-summary.json`はともに`status=complete`で、metrics/bad-pages/static/firmwareの4 commandはすべてexit code `0`、JSON parse成功だった。bad page、static、firmwareは前後でbyte-identicalだった。

| 観測 | 実行前 | 実行後 | 評価 |
| --- | --- | --- | --- |
| ECC total correctable / uncorrectable / deferred | `0 / 0 / 0` | `0 / 0 / 0` | 異常なし |
| UMC ECC correctable / uncorrectable / deferred | `0 / 0 / 0` | `0 / 0 / 0` | 異常なし |
| bad page (pending / retired / unreservable) | すべて`No bad pages found.` | 同一 | 異常なし |
| socket power / throttle | `12 W` / `UNTHROTTLED` | `12 W` / `UNTHROTTLED` | throttle観測なし |
| edge / hotspot / memory temperature | `36 / 36 / 34 C` | `36 / 37 / 34 C` | 高温ではない |
| GFX clock / perf level | `2824 MHz` / `AUTO` | `1362 MHz` / `AUTO` | trace未起動かつactive service下のidle状態差。異常とは扱わない |
| driver / IFWI | `amdgpu 6.16.13`; `SAPPHIRE RADEON AI 32GB`, IFWI `00158746`, build `2025/07/25 02:52` | 同一 | 記録済み |

firmwareはCP_PFP `2950`、CP_ME `2880`、CP_MEC1 `3200`、RLC `12484000`、SDMA0/1 `7966358`、VCN `09.10.B0.01`、PSP_SOSDRV `00.3A.10.14`、ASD `553648388`、TA_RAS `1B.3A.00.01`、PM `00.104.75.00`を記録した。raw evidenceは`benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/`に保存した。

### 段階比較と仮説判定

| stage | relative L2 | cosine | max abs | 判定 |
| --- | ---: | ---: | ---: | --- |
| `qkv_dequant_row_scale` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `z_dequant_row_scale` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `recurrent_gate` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `recurrent_beta` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `recurrent_state_after` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `recurrent_output` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `attention_residual` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `post_norm` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `mlp_activation` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |
| `layer_output` | 未測定 | 未測定 | 未測定 | lock failureにより測定無効 |

- 最初の有意乖離stageは**特定不能**である。GPU streamが存在しないため、`<=1e-5`、`1e-5〜1e-3`、`1e-3〜1e-2`、`>1e-2`のいずれにも分類しない。
- H5は**判定不能**である。GPU kernelを起動していないため、支持・否定のどちらにも使えない。
- H9について、対象R9700のread-only snapshotにはECC、bad page、temperature、throttleの明白な異常は見つからなかった。ただし実負荷trace、決定性比較、他GPU比較は行っていないため、ハードウェア固有要因を否定したものではない。

### 実行した検証

| command / evidence | 結果 |
| --- | --- |
| `g++ -D__HIP_PLATFORM_AMD__ ... tools/query-hip-device-identity.cpp ... -lamdhip64` | host-only compile成功。引数拒否JSONも確認済み。 |
| runbook shell blockの`bash -n` | 成功。 |
| runbookのR9700 BDF guard、ASIC cross-check、health summaryのembedded Python `compile()` | 成功。 |
| 更新済みrunbook block | guard・health・CPU input/referenceまで成功、`flock -n`がexit `1`で終了。再試行なし。 |
| read-only evidence check | guard/trace exit receiptの存在、GPU trace/comparison不在、health before/afterのcomplete statusを確認。 |

service/systemd/active manifest/P3 harnessは変更していない。fix実装（Phase 4以降）にも進んでいない。

## 次の行動

- 今回のwindowはlock failureで消費済みとし、同一window内の再試行は行わない。
- 数値比較、最初の乖離stage、H5の判定を得るには、R9700 lockが利用可能な別windowについてユーザーの明示承認が必要である。active gatewayを停止・変更する判断はこの作業から提案・実行しない。
- H9をさらに切り分ける決定性/thermal/他GPU比較も、今回の結果を踏まえた別途承認事項として扱う。
