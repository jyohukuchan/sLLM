# P3 profile actual operator manifest v5

## 前回の要点

- downstream cascade v5により、official artifactはcommit `3a330dce18285e6b64a5a3c5060409ad0d4857f6`、maintenanceは`bf2ef47a`、launcherは`0ad6d0e6`、captureは`0e8bf9f4`、runnerは`076c3662`、bindingは`80d9b3a0`で確定した。
- quiet window v10はcommit `7c77535e525cadf9a73fc21ed5c70daf932e531b`、evidence SHA-256 `30e5cb3c1e443ce8f0eb49d76ce68d69a2519fba5c26b0381c37ffc07d57af84`でGOになった。

## 今回の変更点

- profile diagnostic v5の単一actual実行に使用するoperator manifestだけをfresh作成した。
- exact argvは10要素、cwdはrepo root、`shell=false`、maximum invocationは1で固定した。
- ready artifactは`resident-one-case-smoke-profile-ready-v5/ready-binding.json`、maintenance evidenceはfreshな`resident-one-case-smoke-profile-maintenance-evidence-v5`へ固定した。
- profile runner、launcher evidence、maintenance evidence、capture root、implicit artifact/stdout/stderr、operator result、actual auditの9件をすべてabsentとして固定した。
- static target command manifestは含めず、launcherがlive preflight後に実行ごとに生成する契約を維持した。
- この作業ではactual実行、GPU command、model load、sudo、service操作を行っていない。

## Quiet boundary

- quiet v10 final streakは27 samples、`172.226505035`秒、reset 0である。
- final HEAD/treeは`5518d9133000be3c08e32676eadf8df38c492c5e` / `a5cff162672876f41459a8ca5f9401442a275ab3`である。
- relevant setはsealed root 6個、regular file 43個であり、byte aggregateは`9160cbb9003cea1bb8589cb286ea91a0e3244078ddfdbd49b2339e6a826a8ceb`、identity aggregateは`6f91f077e08aed8b23abaed6ea043eb9d35c13f42c8fd96f69fdd63c08880294`である。
- formal health identityはstart/endとも`51b37eda8771d36e154f6cd52a22be0bf2d33f2eafa0d90958afe57d14a7b82f`である。
- external processは0、AMD/KFD ownerはworker PID `4058378`だけである。

## Verification

- dry/fake/readback selected tests: `5 passed`
- Git `commit:path` readback: `10/10`
- input SHA-256 readback: `19/19`
- fresh execution outputs: `9/9 absent`
- historical v3 runner/capture failureとv4 launcher/maintenance/capture failure: `5/5 SHA256SUMS passed`
- manifest semantic self-hash、command/input/fresh set hash、secret scan、static target absence: PASS
- command SHA-256: `001b98a9b2567f9d3f1d0ba1a92cc3319e23bf7cc56a013965ed4a0b06ab7d25`
- input set SHA-256: `0052419753d57dc829ecdfade00023eea80dc8e496bf3ad2d957d0f8d88130a3`
- fresh output set SHA-256: `84512ce15e599deb0a87418fc91fd7873c8e5a17ed2c89417f86d120b5467777`
- manifest self SHA-256: `34ef0bb4ee20f9a69d790527579c8c1a81a00fad8b89393dd11d3e4c536eba76`
- manifest file SHA-256: `537afd215296ba6f7475dc0fbf64ef113228dee060fc422a4d51060b3b763def`
- `SHA256SUMS` file SHA-256: `84d9787bcd990881310cf38414f14a1af224c1c5019f385bce8d1601d8127e35`
- manifest/SUMSはmode `0444`、directoryはmode `0555`、各memberはnlink 1である。

## 次の行動

- actual実行はこのmanifest作成とは別の明示承認段階として扱う。
- 実行直前には9 fresh outputs、quiet/current authority、service/GPU ownershipを再検証し、manifestのargvをexactに最大1回だけ使用する。
