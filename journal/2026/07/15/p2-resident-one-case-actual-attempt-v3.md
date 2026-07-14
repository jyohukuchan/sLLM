# P2 resident one-case actual attempt v3

## 前回の要点

- v2はstop直後のAMD process sentinelを旧owner recordとして解釈できず、poll attempt 0でterminal failureとなった。
- v3はAMD process parserをzero sentinel対応にし、stable 2後だけlauncherへ進む契約だった。

## 今回の変更点

- commit `5fc462ef5f716744d36094752ea6d7064087b08a`、manifest SHA-256 `188027f0796af0deee98fb24eedea2e5857aed3eb38fcb63d900964d67e2d708`を唯一のargv源とした。HEAD/input/SHA256SUMS/permissions、3 fresh output ABSENT、formal health `9/6/6`、service/worker/`NRestarts=0`/GPU/KFD/lock、RAM約79.6GB、disk約2.58TBのpreflightはPASSした。
- 同一PTYでsudoをprimeし、manifestのcwdと9要素argvを`shell=false`で1回だけ実行した。開始`1784060797718311033`、終了`1784060849332856187`、elapsed `51,614,545,154 ns`、return code `1`。再試行とprofile実行はしていない。
- poll attempt 0は両service inactive、旧worker absent、AMD process return code 0/stdout SHA `c623fc11440b2bf81199ddefe42cadc330fa31ecde1cd268ff0ab930889e09ca`のzero sentinel、AMD static VRAM return code 0まで通過した。その直後に`error_type=LauncherError`でterminal failureとなりstable `0/2`、launcher/model load/warmup/measuredは`0/0`だった。
- deadline checkpointsは`stopped-poll-kfd:after`まで成功し、`stopped-poll-lock:before`直後に`after-error`となっている。KFDは今回の原因ではなく、failureは`_poll_lock_observation()`に限定される。error typeが`LauncherError`なので、同関数冒頭の`LAUNCHER.reject_symlink_components()`がmissing/symlink componentを検出した経路である。
- systemd unitは`RuntimeDirectory=ullm`を宣言しており、service stop時に`/run/ullm`自体を削除する。したがって`/run/ullm/r9700.lock`の親component missingがexact operational causeで、pollが要求する既存regular file/identity/free lock契約はservice停止中には成立しない。post start後はsystemdがdirectoryを再作成し、lockはmode `0600`のregular fileとして新workerが保持している。
- safe classificationでは、systemd管理の`/run/ullm`とlock leafのENOENTはservice inactiveかつold service/worker absentを同時確認したattemptに限り「lock path absent and therefore not reusable yet」としてpending扱いにできる。ただしそのままstable/freeとみなすとrunnerが親directory不在でopenできない。launcher前にsudoでRuntimeDirectoryと同一owner/modeの専用永続lock directoryを用意するか、lockをservice lifecycle外の永続pathへ移す必要がある。symlink、inode replacement、EACCES、EIO、malformed `/proc/locks`、unknown holderはfailを維持する。
- outer restoreはattempted/passed。post service main PID `18825`、worker PID `18897`、`NRestarts=0`、formal health `9/6/6`、GPU/KFD owner、lock、production hashesは正常。actual関連child、launcher output、runner raw/summaryは存在しない。
- immutable evidence `resident-one-case-smoke-maintenance-evidence-v3/`は`0555`/`0444`でSHA256SUMS PASS。SHA-256はlauncher evidence `5413cb361fd8e113d938a32c786a26785ca0f851ea75d7ac3dbedec8c459f707`、poll `6105775c04ebfd2a15e48f0a6c9e3839ac62bc0ae0ec4ce2191f7a04d4d4f64e`、marker `8b526e24741f3aa55d36d0639a2d271a43d6bb1664f877ee7696e5a9dfdd6fea`、SUMS `447c4b060f0b02be8212739bd86a6cfeed848de3367a33387879cb9e078b5c11`。

## 次の行動

- v3 single-use outputは再利用せず、actualも再試行しない。
- lock pathをsystemd `RuntimeDirectory`のservice lifecycleから分離するか、stop後に安全なowner/modeで再作成してからstable 2を評価する。単にENOENTをfree扱いしてlauncherへ進めない。
- poll evidenceへlock exception message、errno、missing component、pre/post device/inode、`/proc/locks` holderをsecret-freeで保存する。
