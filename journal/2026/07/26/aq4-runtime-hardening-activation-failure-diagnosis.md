# AQ4_0 runtime hardening activation failure diagnosis

## 前回の要点

- AQ4_0 hardening の protected worker、minimal product/tokenizer closure、fresh receipt、frozen candidate manifest、immutable rollback copy、sealed activation control source は準備済みだった。
- candidate manifest SHA-256 は `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`、元の live manifest SHA-256 は `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`、worker SHA-256 は双方 `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b` だった。
- activation は candidate live proof で失敗して元 bytes を restore したが、rollback live proof を完遂できず、healthy rollback の宣言はしていない。

## 今回の変更点

- candidate/live worker を service と同じ `homelab1`、gateway working directory、offline/cache/HIP environment、30 個の `ULLM_REQUIRE_HIP_*` guard で直接対照起動した。stdout/stderr を完全に分離捕捉し、candidate は約 4.8 秒で ready、live は約 6.1 秒で ready となった。両者の ready stdout と 192 行・223,428 bytes の stderr は byte-identical であり、candidate stderr に error/fatal/panic/permission/OOM/segfault はない。120 秒後の exit 124 はテスト timeout が送った終了であって自発的な crash ではない。
- activation journal を再構成した。candidate service は 10:59:50.004 JST に systemd start、10:59:50.286 JST に stop で、282 ms しか走っていない。server process/startup complete より前なのでモデルロード中の worker death ではない。11:00:27 の worker stdout EOF は rollback 済み live service が systemd stop request を受けた後の shutdown event だった。
- sealed route source を読み、`reconcile()` が `active/running` だけで return し、続く `observe()` が gateway/OpenWebUI の5 endpointを retry なしで一回だけ probe することを確認した。candidate live proof は readiness race により失敗し、route が candidate を止めて rollback した。
- product closure は source package と candidate package が 1,045 files で完全一致し、package manifest の 1,044 referenced members に欠落はない。122 omitted files は artifacts/history sidecar。`artifact -> package` symlink は `product.artifact: null` の AQ4 contract では不要であり、worker は artifact directory を拒否する。5-file tokenizer closure も実 contract load に成功した。closure、manifest 記述、read-only ownership は原因ではない。
- outcome/audit と source を突合した。outcome の `failure_stage` は `candidate_live_proof`、recovery audit の `error` は `ActivationError` のみ。control route が child stderr/message/cause を immutable audit に残さない設計なので、過去の五つの endpoint のうち最初の失敗 endpoint は復元不能である。
- 診断の raw stdout/stderr、environment、closure diff、journal、activation source extract、safety verification を `benchmarks/results/2026-07-26/aq4-activation-failure-diagnosis/` に保存した。`/etc/ullm/served-models/active.json` は read-only hash/stat のみ、`ullm-openai.service` に stop/restart/reload は行っていない。最終確認は `active/running`、`Result=success`、direct test 開始以後の unit journal entry なし。

## 次の行動

1. consumed activation plan を再試行しない。candidate closure を rebuild せず、今回の failure outcome/audit を incident evidence として保持する。
2. 別途承認された実装 task で、sealed activation-control source を更新する。systemd active 後に bounded readiness retry/backoff を行い、stable PID/manifest と5 endpoint/model ID の coherent success を待つようにする。
3. 将来の failed live proof は sanitized return code/stderr/cause、時刻、endpoint state を credential 非含有の immutable audit に保存し、stage を `failed` と明示する。rollback/recovery にも同じ wait contract を適用する。
4. 新 control source/new immutable plan/read-only preflight/明示的人間承認が揃うまで activation は実行しない。
