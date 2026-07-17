# SQ8 authorized runtime materialization 362cfa95

## 前回の要点

source `ba3f02ee3773919259450099db6ca6a58ba52065`からexact12 lineage `dd926fad8ef0b1686e57d9d13877be466a6e4414c146e7baea25404aef63be1e`とunauthorized runtime `/tmp/ullm-sq8-overlay-gpu-promotion-ba3f02ee-dd926fad-unauthorized-v2`をcreate-new生成した。formal runtime GOは`/tmp/ullm-sq8-overlay-independent-audit-v2-ba3f02ee-dd926fad/audit-receipt.json`、SHA `362cfa9587b0419703b99ba9436acfbfef82f6464da681d1bb77fc54bc0efcbf`である。

## 今回の変更点

- formal builderでauthorized runtime `/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-362cfa9587b04197`をcreate-new生成した。Gate SHAは`c580e07927711760c6564da9a20958e6372a385df223175fb2e603f24ca07ec9`、served-model SHAは`67f88042f85a0efdda74676bfde0d4db86742a04989b5ae5c32859783d0d7851`、prepared receipt SHAは`2dc293b6109409888af12b70fae1f042badefbfb0cc010543554ae5cf69c552b`、SHA256SUMS SHAは`6035e828e37262470df39b4b2225086c8c81e10270bf643c5962c3934cefde2f`である。
- Gateは`authorized_pending_execution`、`actual_run_allowed=true`、`max_attempts=1`、`maximum_actual_runs=1`である。fixed request `sq8-promotion-8aded6069819cf2ceb8bff166b63596daab892bf7a2e95b3fc0b3b6ca91c0654`はunauthorized runtime、formal audit、authorized Gate/build/receiptで一致した。
- formal audit `362cfa95…`は4箇所、exact12 lineage `dd926fad…`とcurrent GO `46a536aa287bde7def56f9c0a1141a256873fac996e94cab60ab9770ce37f1e3`は5箇所へ一致して伝播した。authorized workerはSHA `4dcf1bd3164d0a83aec4ded51c199876d407e22a325fff9d7015df7648c9e050`、0555、nlink 1で、source/immutable pathはいずれもauthorized runtime自身である。
- exact8メンバー、regular 0444/nlink 1、worker 0555/nlink 1、SUMS 7 entries、historical runtime reference 0、source commit/tree/archive、live artifact/package manifest、readinessの一致を機械監査した。
- release rlibをcleanして`CARGO_BUILD_JOBS=1`、`--jobs 1`で再構築した。rlib SHA `f092100f641e2ddd7571d743769a22509878c9795e761e002382dc5c6f4213c2`へリンクしたCPU validator SHA `6a83df01de986d6604665f6f16e4526c0a3061a82ca00aa2564a2024c82b4f58`がauthorized served-modelを受理した。
- wrapper dry-runはcandidate SHA `280dd8cb6991bb5245db00416b53b39787bbb6f4091616023cb1cc8ddc18a094`で成功した。actual output `/tmp/ullm-sq8-overlay-gpu-promotion-actual-362cfa9587b04197`、`.incomplete` staging、`.attempted` sentinelは前後とも存在しない。GPU、service、sudo、actual executionは行っていない。source worktreeはHEAD `ba3f02ee3773919259450099db6ca6a58ba52065`のclean状態を維持した。

## 次の行動

authorized runtimeをfresh pre-execution auditへ渡す。actual executionは、その監査と明示的なone-shot実行指示が揃うまで開始しない。
