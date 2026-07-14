# AQ4 P2 resident profile diagnostic ready v0.1

## 前回の要点

通常のone-case maintenance chainはサービス停止、zero-owner gate、launcher実行、外側`finally`復旧を保証していたが、ROCTx markerとrocprofv3 captureを信頼鎖へ束縛していなかった。

## 今回の変更点

runner `e93a2c162eb059cb2db883953d331f7a158d3a16`、validator `82635456825503c535ce0b662e72a7a233d18c40`、B `7e59baee0c1ac93a350da58a4292a84fbfde9f1c`、launcher `bdb06083ca3646c8f934fea10dac691a6efd4626`、harness `c5f6f2ac0130642f3d5c31204e84e15eecaf1e29`の順に再固定した。

profile-ready artifactは`resident-one-case-smoke-profile-ready-v1`である。`ready-binding.json` SHA-256は`6cd1a7862ba2b12494c44dc1727c11613537c970c3fe1460b0ab863c552679a9`、`harness-trust.json`は`b22b0a8d66afc743b56a9513cdcdebe9c1cb31a3fd3d67352b05ed6311b5eb78`、`qa-attestation.json`は`72225d27f773d1e8f2b7f51d1466c19104a505a4bf8612f9bcd6a68a482106ef`、`SHA256SUMS`は`5816be78c4d1ba4e0946e2c45e07a5c69526a285f2c0ea18add7b900b6074d1b`である。

artifactは`execution_mode=profile_diagnostic`、actual one-case最大1回、measurement/promotion不可、全output no-reuseである。profile以外の直接実行を拒否し、rocprof環境、CSV output、profiler ctor、mapped tool/ROCTx libraryをサービス停止前に検査する。

outer commandはcapture tool commit `489183abba581332544d0d004338a2cee08a0d89`、SHA-256 `be62caa7eee810cd6b33033eab15418b803ac1cee6559153e0cf7af446fa21f7`から`rocprofv3`をexactly once起動し、そのcommandとしてmaintenance harnessを包む。profilerは`/opt/rocm-7.2.1/bin/rocprofv3`、SHA-256 `13060810d6b80653631b14f0f5e33ea160c2b79a6a3a4c6850142010b48b8ec8`である。

launcherはrunnerへ`--profile-roctx-ranges --roctx-library /opt/rocm/lib/libroctx64.so.4 --roctx-library-sha256 22bbc6946fdf5d7d8b1755cbd738c42a63f3795d18ac3ed1285b09cc772dee17`を`--driver-command`より前に透過する。resolved libraryは`/opt/rocm-7.2.1/lib/libroctx64.so.4.1.70201`へ固定する。launcherは12個のbalanced range、同一PID/thread、audit SHA、run ID、resident session ID、case ID/SHA、library invocation/resolved path/SHAを検査する。

capture outputは`p3/aq4-p3-diagnostic-rocprof-capture-v1`、resident summary/rawはprofile専用runner outputへ固定した。capture toolはrawのsession IDとmarkerのsession ID、run ID、case ID/SHAを照合し、warmup 0–1を除外してmeasured 2–11を分割する。

canonical dry-runは`resident-one-case-smoke-profile-ready-dry-run-v1`である。evidence SHA-256は`5568fdbd9e7a5e8debe3be8bb2d502e8266f87634ee970841cfc87758bf5abd3`、`SHA256SUMS`は`a3b21b76c4e632b70c815f3315ffcf844d514f1fa339307d2dcd289ab12c28e2`である。sudo、stop/start、launcher、rocprof、capture toolのprocess countsは全て0で、service/GPU/modelは未操作である。

回帰は主要セット155 tests、marker chain 55 tests、diagnostic capture 8 testsが通過した。独立marker QAの手動境界15件も通過している。

## 次の行動

profile diagnostic actual runには別の明示承認と、artifact内のexact capture commandを使う。同一PTY sudo cache、rocprof wrapper、fresh output、pre-stop/live gateのいずれかが不成立なら実行しない。取得結果は性能昇格に使わない。
