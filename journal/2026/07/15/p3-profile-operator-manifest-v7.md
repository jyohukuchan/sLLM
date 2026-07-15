# P3 profile operator manifest v7

## 前回の要点

- profile operator v6のactual failureはimmutable evidenceとして保持し、再試行や上書きをしない。
- authorityはresident binding v6へcascadeし、fresh profile ready-v6の確定を待っていた。

## 今回の変更点

- profile ready-v6をcommit `ff15b75ceed5e7b7eabe376e27859106694c285f`、`ready-binding.json` SHA-256 `0fca1b6d5b561b582bd0e59c33d88e558be785faa026a058a1a8d3e9d3b4e54e`、`SHA256SUMS` SHA-256 `49c5535e617db3598029e0968e253a8771ad3489bc683cf541d11c54d13a1ccc`へ固定した。
- 再現可能なcollector/generatorを`tools/prepare-aq4-p3-profile-operator.py`、critical-path testsを`tests/test_prepare_aq4_p3_profile_operator.py`として追加した。targeted testは6件すべてpassed。
- read-only current auditはservice main PID `193179`、worker PID `193294`、`active/running`、`NRestarts=0`、lock busy、AMD/KFD owners `[193294]`、9/9 fresh outputs absent、targeted processなしだった。
- quiet-window v12は27/27連続clean sample、span `311.064241828`秒、reset 0、最終confirmation passedでGOになった。actual、GPU command、service操作はいずれも実行していない。
- operator-command v7はexact 10 argv、`shell=false`、maximum invocation 1、9/9 fresh、retry forbidden、outer-finally restore 120秒を固定した。manifest semantic SHA-256は`f4e7678f5ac4108701699e818596fc956fc06f92776e5bfb3f781abd070efa13`。
- 旧operator-command-v6、profile-actual-audit-v6、profile-operator-result-v6は`git diff --exit-code`で不変を確認した。

## 監査コマンド

current service epochとquiet前提のread-only監査:

```bash
python3 tools/prepare-aq4-p3-profile-operator.py audit-current
```

sealed成果物の独立readback:

```bash
python3 tools/prepare-aq4-p3-profile-operator.py validate-quiet
python3 tools/prepare-aq4-p3-profile-operator.py validate-operator
```

明示的GO後に許可されるexact-one actual argv。現時点では実行禁止:

```text
/usr/bin/python3.12 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-p2-resident-smoke-maintenance.py --mode execute --profile-diagnostic --ready-artifact /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-v6/ready-binding.json --evidence-output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-maintenance-evidence-v6 --confirm-one-case
```

## failure時のimmutable capture/restore

1. actualは1回で打ち切り、同じoutputへ再試行しない。
2. operator stdout/stderr、maintenance、launcher、capture、READY auditをそのまま保持し、failure reportより先にimmutable captureを完了する。
3. outer `finally`のservice restoreを必ず実行し、固定absolute monotonic deadline 120秒を超えて延長しない。
4. restore後は新service epochの`active/running`、`NRestarts=0`、worker、lock、AMD/KFD owner、formal health、worker/package/served hashesを再確認する。
5. 子孫processが0であることを確認し、failure evidenceをsealed化する。検証に失敗した場合も再実行せず、その状態を報告する。

## 次の行動

- 独立GOを受領し、GO直前の`audit-current`と9/9 fresh absenceがcleanであることを確認した。
- sealed manifestのargvを`os.execv`でshellを介さず、同一PTYのsudo cacheを使って1回だけ実行した。returncodeは`1`で、retryは`0`。
- 最上位reason codeは`ready_candidate_marker_absent`。capture failureは`expected exactly one marker trace, got 0`で、runnerは完了しtrace CSVを4件生成したが、capture contractがfail-closedした。
- 今回のprofile evidenceはfailure evidenceであり、measurementおよびpromotionには使用しない。

## actual failureの保存とrestore結果

- operator result status: `failed`
- actual audit status: `failed_immutable_evidence_preserved_restore_passed`
- operator result JSON SHA-256: `15a0f971350b403106c4d7f37c5148d88b4d590b396fe9b1d8dcf056a0fcbb55`
- actual audit JSON SHA-256: `fb72e65c2219aae049ac6133cddbe35c84329ef320cf0f4b8eb499e931a0e8be`
- capture failure JSON SHA-256: `de9518084982fe49e39e9a3939eed4cecade57f91bd8289af6e3c6d94be0061e`
- outer-finally restoreは`15.094784399`秒で成功し、120秒のabsolute deadline内だった。
- serviceはmain PID `193179`から`466848`、workerは`193294`から`467004`の新epochへ移行した。`active/running`、`NRestarts=0`、lock busy、AMD/KFD owners `[467004]`を確認した。
- worker/package-manifest/served-manifest/formal-health hashはpreflightと一致した。package full content hashはexactly 1回でpassedし、最終metadata identityも一致した。
- capture children、launcher children、lock holders、targeted residual processはいずれも0。secret pattern hitも0だった。
- maintenance-v6、execute-evidence-v6、runtime-v6、capture-v6、operator-result-v7、actual-audit-v7はすべて`SHA256SUMS`付きでroot `0555`、member `0444`へ封印した。

## 次の行動（actual後）

- retryは禁止されたままであり、同じprofile-v6 outputを再利用しない。
- marker trace欠落の原因調査は、今回のimmutable evidenceを入力にした別作業として行う。
