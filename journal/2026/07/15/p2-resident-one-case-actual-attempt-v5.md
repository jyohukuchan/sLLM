# P2 resident one-case actual attempt v5

## 前回の要点

- v4 preflightはKFD `gpuid`の末尾改行なしをmalformedとして拒否し、actualを開始しなかった。
- v5は末尾改行なしの正のdecimal `gpuid`を受理しつつ、service stop後に専用lock substrateを作成し、stable 2回の成立後だけlauncherへ進む契約だった。

## 今回の変更点

- commit `65102c73f1a9a7138cbbcb0cbbd371c1352b2317`、operator manifest SHA-256 `5b5ee6302b60e5976137cdeff7fc59475ef4a15728d07681102d7570224805a3`を唯一のargv源とした。manifestのworking directoryは`/home/homelab1/coding-local/ultimateLLM/uLLM-project`、9要素argvは次のとおりである。

```text
/usr/bin/python3.12
/home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-p2-resident-smoke-maintenance.py
--mode
execute
--ready-artifact
/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-ready-v1/ready-binding.json
--evidence-output
/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-maintenance-evidence-v5
--confirm-one-case
```

- HEAD/input/SHA256SUMS/permissions、3 fresh output ABSENT、formal health `9/6/6`、全endpoint 200、service main PID `18825`、worker PID `18897`、`NRestarts=0`、AMD/KFD/lock/production hashes、RAM約82.1GB、disk約2.58TBのpreflightはPASSした。
- 同一PTYでsudoをprimeし、manifestのcwdとargvを`subprocess.run(..., shell=False)`へそのまま渡して1回だけ実行した。開始`1784064373943026138` unix ns、終了`1784064444092336303` unix ns、elapsed `70,149,310,165 ns`、return code `1`。再試行とprofile実行はしていない。
- service stop後のlock substrate作成はPASSした。directory `/run/ullm`はdevice `26`/inode `747100`/owner `1000:1000`/mode `0750`、lock `/run/ullm/r9700.lock`はdevice `26`/inode `747101`/owner `1000:1000`/mode `0600`だった。
- stopped gateは2 pollともstableだった。poll SHA-256は`0d9cbaf6ca5d9d91d1dcf1849ca4cc218526d3e9f802b7768a165275994a7657`と`b05504626b8d0633643ae38ac4ccb7e7defbb37c2103817579bb90bcd2d3a33b`。両pollでAMD owner `[]`、KFD owner `[]`、VRAM used `0`、lock device `26`/inode `747101` freeを確認した。AMD zero sentinel raw SHA-256は`c623fc11440b2bf81199ddefe42cadc330fa31ecde1cd268ff0ab930889e09ca`。KFD `gpuid`は末尾改行なしの正のdecimalとして受理された。
- immutable launcherのvalidatorはexit code `0`。runnerはlive preflight SHA-256 `0028092c9fad19b2c79529f7c22008124bf02ae36eaad3a5dbabec114e563320`を検証し、同じlock device `26`/inode `747101`を取得した。partial runner outputの`resident-batch.lock-owner.json`はPID `241780`、SHA-256 `09e71de58473c5b14edc12885ecd960df4529913d50372382cab870278f2955d`である。
- launcherが保存したrunner argvは次のexact arrayである。

```text
["/usr/bin/python3.12", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v4/trusted-runner.py", "--expanded", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/case-binding.json", "--fixture-index", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/fixture-index.json", "--identity", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/identity.json", "--preflight", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/preflight.json", "--policy", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/policy.json", "--bundle-root", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1", "--trusted-validator", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/prepare-aq4-p2-resident-smoke-bundle.py", "--trusted-validator-sha256", "b6b7b249d39a8a6c7312535f220a008d058b9b979a9ee8efffb0ddef127bbc28", "--output-dir", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-execute-v1", "--run-id", "p2-r9700-resident-one-case-smoke-execute-v1", "--baseline-kind", "active-production", "--lock-path", "/run/ullm/r9700.lock", "--one-case-smoke", "--live-preflight", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-execute-evidence-v1/live-preflight.json", "--driver-command", "/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/resident-driver", "--served-model-manifest", "/etc/ullm/served-models/active.json", "--device-index", "1", "--build-git-commit", "319d6187b29e877536aa5dfe80c02bde0c77ed7a"]
```

- runnerが`shell=False`で起動したresident driverのexact argvは次のとおりである。

```text
["/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1/resident-driver", "--served-model-manifest", "/etc/ullm/served-models/active.json", "--device-index", "1", "--build-git-commit", "319d6187b29e877536aa5dfe80c02bde0c77ed7a"]
```

- resident processのspawnは成功した。その直後、最初のready handshakeを待つ`_recv()`がdriver stdoutのEOFを読み、exact error `resident driver exited before response`となった。runner exit codeは`1`、launcher failureはstage `runner`、reason `execute runner subprocess failed`、`runner_started=true`。runner stderrはexact `AQ4 P2 resident batch failed: resident driver exited before response\n`、SHA-256 `d4e032c691321da93e28cbcaac192e4f6b0da28d5efc10bd445b9543fe51a01e`、runner stdoutは空、SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`だった。
- trusted runnerはdriver stderrを`PIPE`へ接続するが読み取らず、`finally`でwaitした後のdriver return codeもartifactへ保存しない。したがってdriverのreturn code、driver stderr、ready前のmodel load実行有無、GPU command実行有無、EOFの内側の根本原因は、保存済み証拠ではすべて`unknown`である。これらは推測しない。
- ready handshakeは完了せず、`case_begin`とrun loopへ進んでいない。warmup completed `0/2`、measured completed `0/10`、raw case resultと`resident-batch.summary.json`は存在しない。runner outputはlock owner 1ファイルだけのpartial treeで、tree SHA-256は`5c6d2d73ce876bb9f0005a466f70eb795d349f2b73a98c8374e40dcaa1f6e9d1`。
- cleanupはPASSした。launcher `children_remaining=[]`、maintenance `runner_finished=true`/`runner_children=[]`で、lockをunlinkし、`/run/ullm`をrmdirした後に両pathのabsenceを確認した。追加のread-only確認でもtrusted runner、maintenance harness、resident driverの残存processはなかった。
- outer restoreはattempted/passed。保存証拠のpost service main PIDは`242454`、worker PIDは`242547`、`NRestarts=0`、formal healthは`9/6/6`、全endpoint 200、production hashes、AMD/KFD owner、busy lockは正常だった。追加のread-only確認でもserviceはactive/running、main PID `242454`、worker PID `242547`、`NRestarts=0`。systemdが再作成した現lockはdevice `26`/inode `747123`、directoryはdevice `26`/inode `747116`で、actual substrateとは別のservice epochである。
- immutable maintenance evidenceのSHA-256は`launcher-evidence.json` `4164df343b02f369b3e2fdd8f3f3b24d086c57ed8d86efbe47f8f75d71d89dd2`、`maintenance-marker.json` `480ca223855487c2b1af1412b5e956202a611f6a33a3e73943e9575c5de72cec`、poll 0 `0d9cbaf6ca5d9d91d1dcf1849ca4cc218526d3e9f802b7768a165275994a7657`、poll 1 `b05504626b8d0633643ae38ac4ccb7e7defbb37c2103817579bb90bcd2d3a33b`、`SHA256SUMS` `612234237c566fdae3729da9e48f8215a9f579a6f691d628e84a418297a727a4`。SHA256SUMS verificationはPASSした。
- immutable launcher evidenceのSHA-256は`launcher-evidence.json` `750cc4c49606aaea8d4b157fd55737e6ea6436d76b0accef845453520e2e1547`、`live-preflight.json` `0028092c9fad19b2c79529f7c22008124bf02ae36eaad3a5dbabec114e563320`、runner stderr `d4e032c691321da93e28cbcaac192e4f6b0da28d5efc10bd445b9543fe51a01e`、runner stdout `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`、validator stderr `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`、validator stdout `7c463b16bab152c3554ee355938e1731b1ba65e3ea059adf22e0ccf329635c2a`、`SHA256SUMS` `734b57dd3f856c19280eb4b8359f10f1678301ede3910da21e2d14c925144c69`。SHA256SUMS verificationはPASSした。

## 次の行動

- v5のsingle-use outputは再利用せず、このversionは再試行しない。
- 次版を作る場合は、trusted runnerがready前にdriver stdout EOFを検出したとき、driver stderrをbyte/time bound付きでsecret-safeに回収し、driver return codeとともにimmutable evidenceへ保存する。driver stderr本文を無条件に露出せず、少なくともbytes、SHA-256、truncation、redaction status、return codeを残す。
- 追加証拠なしにmodel loadやGPU commandの実行有無、driver内部の失敗原因を断定しない。
