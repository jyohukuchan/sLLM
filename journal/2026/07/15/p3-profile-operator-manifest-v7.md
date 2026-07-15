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

- 親エージェントから明示的GOが来るまではactualを実行しない。
- GO直前に`audit-current`と9/9 fresh absenceを再確認する。quietまたはservice epochが変化していればactualを止め、fresh quietを再採取する。
