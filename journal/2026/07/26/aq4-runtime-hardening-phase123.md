# AQ4_0 runtime hardening Phase 1–3

## 前回の要点

`docs/plans/aq4-runtime-hardening-promotion-plan-v0.1.md` の Phase 1〜3 を、SQ8 の `/opt/ullm/releases` と分離して実行する段階だった。live AQ4_0 は user-owned な closure を参照しており、worker の再ビルド、P3 flag 混入、GPU 使用、evidence/activation は禁止されていた。

## 今回の変更点

- `/opt/ullm/aq4-runtime-hardening-v0.1/` を root-owned/non-group-world-writable な専用 namespace として作成した。`releases`、`products`、`tokenizers`、`sources`、`control-source` と将来 Phase 用の親だけを配置し、SQ8 namespace は変更していない。
- live worker と legacy engine を `cp --reflink=never` で独立コピーし、root:root `0555` に seal した。worker は source/destination の `cmp`、SHA-256、size が一致し、inode は異なる。worker SHA-256 は `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b` のままである。
- product は `package/` だけを `rsync --recursive --times --no-perms --no-owner --no-group --no-links --no-specials --no-devices` でコピーした。`artifact -> package`、historical `artifacts/`、sidecar は持ち込んでいない。1,045 files / 5 dirs / 7,700,872,459 logical bytes、manifest SHA-256 `a790…e5ad`、source-relative 全 member SHA-256 の照合を通過した。
- tokenizer は `merges.txt`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`chat_template.jinja` の5ファイルだけを独立コピーし、全ての `cmp`/SHA-256 を通過した。
- promotion source を detached `0cd760568…`、manifest-freezer control source を detached `f71bb2e…` で `git clone --no-hardlinks --no-checkout` から作成した。両方とも direct `.git` directory、alternates なし、clean、no symlink/hardlink/ACL/capability、`git fsck --no-dangling` PASS である。
- すべての final closure member（24,634 entries）の owner/mode/nlink/size/device/inode/SHA-256 を `benchmarks/results/2026-07-26/aq4-runtime-hardening-phase123/closure-members.tsv` に記録した。Phase 1〜3 closure と ancestors の runtime-seal precheck は PASS、full activation readiness は Phase 4+ 未実施のため意図どおり NOT_READY である。
- `/etc/ullm/served-models/active.json`、unit、gateway environment の hashes は計画値のまま。live manifest の required worker flags は30件、P3-only 6 key は0件だった。GPU、evidence、receipt、profile、candidate manifest、activation、campaign は実行していない。

計画側へ申し送り: source の regular file を `0444` に seal すると、tracked executable files の mode だけで `git status` が dirty になる。final standalone clone では clone-local `core.filemode=false` を明示して、required immutable modes と clean-status guard を両立した。計画の Phase 3 手順にはこの扱いを明記する必要がある。

初回 promotion source staging clone は `core.filemode=true` のまま seal して mode-only dirty になったため、publish せず `/opt/ullm/aq4-runtime-hardening-v0.1/.staging/sources/aq4-promotion-0cd760568e197.20260725T193039Z.03ovtM` に quarantine として残した。新しい staging clone を作り直して final publish した。なお task 開始時の `ullm-openai.service` は inactive、最終 read-only snapshot は active だったが、この task は service command を一度も発行しておらず、変化の原因は未確認である。

## 次の行動

1. GPU window が許可された別 task でのみ、live manifest から機械的に30 flag profile を作り、fresh protected-path evidence と receipt を収集する。P3 profile は入力にしない。
2. sealed promotion source と freezer control source を用いて Phase 4 candidate manifest の生成・freeze を行う。ただし `/etc/ullm/served-models/active.json` は変更しない。
3. Phase 5 の reviewed AQ4-to-AQ4 activation control source を別途実装・seal し、non-mutating preflight が完了するまで activation は行わない。
