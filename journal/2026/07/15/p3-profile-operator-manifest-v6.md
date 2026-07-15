# P3 profile actual operator manifest v6

## 前回の要点

- operator manifest v5はcommit `7b2a6a29046a4a4f0abdbeea73fdefbf8d74857c`、tree `b930015ba06d882cd4062e1a90e6e5f8e0a2661b`で固定されていた。
- quiet window v11はcommit `5a79b3fe476e5a67ffd4dca2b8db09bd58c6b6b3`、evidence SHA-256 `218358d35c24b4ba14778ebcbbbf7afd67623f0a7d8f97e0510450ca81f651a1`でGOになった。

## 今回の変更点

- operator manifest v5を`unused_historical_immutable`としてreadbackし、profile diagnostic v6のoperator manifestをfresh作成した。
- artifact/source authorityはv5から変更せず、quiet authorityだけをv11へ更新した。
- exact argvは10要素、ready artifactは`profile-ready-v5`、maintenance evidenceはfreshな`profile-maintenance-evidence-v5`、cwdはrepo root、`shell=false`、maximum invocationは1で固定した。
- runtime output 7件はv5 authorityのまま維持し、post-execution operator resultとactual auditだけをv6 pathへ更新した。quiet v11が監視した旧v5 operator pathとの差は`quiet_transition`へ明記した。
- 9件のexecution outputはすべてabsentであり、static target command manifestは存在しない。
- この作業ではactual実行、GPU command、model load、sudo、service操作を行っていない。

## Quiet boundary

- quiet v11 final streakは27 samples、`172.270818469`秒、reset 0である。
- final HEAD/treeは`7b2a6a29046a4a4f0abdbeea73fdefbf8d74857c` / `b930015ba06d882cd4062e1a90e6e5f8e0a2661b`である。
- relevant setはsealed root 6個、regular file 43個であり、byte aggregateは`9160cbb9003cea1bb8589cb286ea91a0e3244078ddfdbd49b2339e6a826a8ceb`、identity aggregateは`6f91f077e08aed8b23abaed6ea043eb9d35c13f42c8fd96f69fdd63c08880294`である。
- formal health identityはstart/end/confirmationとも`51b37eda8771d36e154f6cd52a22be0bf2d33f2eafa0d90958afe57d14a7b82f`、blocking identityは`f3628d4e3949557fcd0d6173d0f6f7319eb583434af2deb259088280b18ecd0b`である。
- external processは0、AMD/KFD ownerはworker PID `29805`だけであり、service main PIDは`29720`である。

## Verification

- dry/fake/readback selected tests: `5 passed in 0.37s`
- Git `commit:path` readback: `10/10`
- input SHA-256 readback: `19/19`
- fresh execution outputs: `9/9 absent`
- historical v3/v4 failures: `5/5 SHA256SUMS passed`
- historical operator manifest v5: `1/1 SHA256SUMS passed`、mode/nlink/readback PASS
- manifest semantic self-hash、command/input/fresh set hash、secret scan、static target absence: PASS
- command SHA-256: `001b98a9b2567f9d3f1d0ba1a92cc3319e23bf7cc56a013965ed4a0b06ab7d25`
- input set SHA-256: `b6f26b1fd20ea12195a93c94c82e96159393b3dee47360030f317383807f07ea`
- fresh output set SHA-256: `ed44871a2a06ba70f11f90835cdad76a643e0e8e108f567312fe3dc9a1e1e14b`
- manifest self SHA-256: `9f8f83c334696813e928eea01e5e8fb6711a7fc6997a962ece8e75eecdb8f269`
- manifest file SHA-256: `4c3b7ba685ecc465e77169d9ae186c93bdc548357ef883c396bbd06a18d685ce`
- `SHA256SUMS` file SHA-256: `349a20579488d08ef7c215c430d195ce33c2d3f195382b973418156ef124eda4`
- manifest/SUMSはmode `0444`、directoryはmode `0555`、各memberはnlink 1である。

## 次の行動

- actual実行はこのmanifest作成とは別の明示承認段階として扱う。
- 実行直前には9 fresh outputs、quiet/current authority、service/GPU ownershipを再検証し、manifestのargvをexactに最大1回だけ使用する。
