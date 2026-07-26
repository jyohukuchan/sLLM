# AQ4 activation failure diagnosis

Date: 2026-07-26 JST

## 結論

candidate worker はモデルロード中に異常終了していない。activation control route が systemd の `active/running` を gateway の readiness と取り違え、candidate の server process が開始する前に one-shot live proof を実行した。その probe が失敗したため、route 自身が candidate service を停止して rollback を開始した。

candidate の service start は 10:59:50.004 JST、stop は 10:59:50.286 JST であり、稼働時間は 282 ms である。candidate に対する `Started server process`、`Application startup complete`、または `worker_fatal` は journald にない。対照となる restored live gateway は systemd start から 464 ms 後に server process を起動し、約 2.8 秒後に application startup を完了している。この時間関係だけでも candidate を正常 ready まで待たなかったことが分かる。

11:00:27 JST の `{"event":"worker_fatal","reason":"unexpected worker stdout EOF"}` は candidate ではない。systemd の stop request (11:00:27.613) の後、rollback 済み live gateway PID 2691844 が stdout close を受けて記録した shutdown event である。10:59:49 の同種 EOF も、activation 開始のため停止された旧 live gateway の event である。kernel journal に OOM、amdgpu reset、segfault、または kill の記録はない。

## 直接 worker 対照実験

candidate と live の worker を、service と同じ `homelab1`、gateway working directory、offline/cache/HIP variables、および 30 個すべての `ULLM_REQUIRE_HIP_*` guard を用いて直接起動した。stdin は protocol 待機を維持し、stdout/stderr は別ファイルへ全量捕捉した。テスト用に `ULLM_SERVED_MODEL_MANIFEST` だけをそれぞれ candidate/live manifest に設定した。

| 項目 | candidate | live control |
| --- | --- | --- |
| stdout | ready record 1 行 | 同一の ready record 1 行 |
| stderr | 192 行、223,428 bytes | candidate と byte-identical |
| stderr SHA-256 | `1e4eb40769bf54c774669a986690189fd7571eba67b3b4c309a77c8b0d6a49cd` | 同一 |
| stdout SHA-256 | `12179085c067ef6d5645251c0052e447603161e0c35841ef2005d6d1f10bafd1` | 同一 |
| 自発的な失敗 | なし | なし |
| 終了 | 120 秒の意図的 timeout が TERM/KILL を送ったため outer exit 124 | 同左 |

candidate の ready record は `model: "ullm-qwen3.5-9b-aq4"`、`device: "gfx1201"`、`execution_profile: "rdna4_aq4_resident"`、`package_manifest_sha256: "a790a033…e5ad"` を返した。candidate stderr の実際の全内容は `candidate.stderr` にあり、192 行すべてが `schema_version: "ullm.backend_operation.load.v1"` の backend load trace である。`error`、`fatal`、`panic`、`permission denied`、`no such file`、OOM、segfault は含まれない。これは live control の stderr と完全に同一である。

candidate は開始後約 4.8 秒で ready を stdout に書き、57 秒時点でも生存していた。live control の ready は約 6.1 秒後だった。従って 282 ms の activation window で candidate gateway の endpoint が ready でないことは期待どおりである。

完全な標準出力・標準エラーと実行結果は次に保存した。

- `candidate.stdout`, `candidate.stderr`, `candidate.result.txt`, `candidate.live-observation.txt`
- `live.stdout`, `live.stderr`, `live.result.txt`
- `live-worker-environment.txt`, `service-configuration-and-environment.txt`, `direct-worker-comparison.txt`

## manifest / closure の対照

worker binary は candidate/live とも SHA-256 `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b` の同一 bytes である。manifest の意味上の差は `product.root`、`tokenizer.root`、`worker.binary`、`promotion.receipt`、`promotion.receipt_sha256` の path-bound 5 項目だけである（`manifest-semantic-diff.patch`）。

- source product の 1,167 regular files に対して candidate は 1,045 files / 5 directories。source `package/` と candidate `package/` の 1,045 files は同じで、package manifest が参照する 1,044 payload members の欠落は 0。
- 落とした 122 files は `artifacts/sq8-...` と promotion/history sidecar であり、AQ4 package runtime member ではない。
- source の `artifact -> package` symlink を candidate に再作成する必要はない。両 manifest は `product.artifact: null` を宣言し、AQ4 worker は artifact directory を受け取ることを拒否する。
- candidate tokenizer は契約上の 5 files だけだが、実 gateway tokenizer contract を通じて `Qwen2Tokenizer` の load に成功した。削除した model shard / model config / README 等は tokenizer load に不要である。
- candidate product/tokenizer は root-owned read-only のままで、`homelab1` の直接 worker はモデルをロードした。読み取り・traverse・一時書き込みの権限仮説は否定された。

よって minimal closure、tokenizer closure、symlink 解消、絶対 path 長、mount 境界、root-owned read-only directory は今回の失敗原因ではない。closure の作り直しも manifest 記述の修正も不要である。

## ActivationError の解読

`activation/outcome.json` は次を記録する。

- `status: "failed_restore"`
- `failure_stage: "candidate_live_proof"`
- candidate SHA-256 は `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`
- observed active SHA-256 は restored original `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`
- restoration は attempted だが rollback live proof は未完了

locked recovery audit の具体的な `error` field は `ActivationError` だけである。sealed control source の `_record_attempt()` は `type(error).__name__` だけを JSON に書き、message/cause を落とす。operation source も例外を catch して stderr に `AQ4 hardening operation failed` としか書かない。activation wrapper は child operation stderr/return detail を outcome/audit に保持しない。そのため過去の immutable audit だけから「五つの endpoint のどれが最初に失敗したか」というさらに細かい文字列を復元することは不可能である。

ただし source と時系列から失敗内容は十分に確定できる。`reconcile()` は `systemctl restart` の後、`active/running` になった時点で return する。直後の `observe()` は gateway `/healthz`、`/readyz`、`/v1/models` と OpenWebUI `/health`、`/api/models` を一度だけ probe し、retry/backoff を持たない。candidate は server process 前に stop されたため、candidate live proof は readiness 前の endpoint observation failure である。`stages.candidate_live_proof: "not_run"` は別原因を表さない。例外処理がまだ `pending` の stage を最後に `not_run` へ変換するためであり、`failure_stage` の方が実際に試行を開始した stage を示す。

`activation-evidence.txt`、`activation-source-extract.txt`、`runtime-contract-source-extract.txt` に immutable evidence と該当 source extract を保存した。

## 修正方針（実装しない）

minimal closure は作り直さない。candidate frozen manifest も書き換えない。代わりに、consumed activation plan を再実行せず、以下を含む新しい sealed control-source と新しい immutable plan を別途レビューする。

1. `active/running` の直後に live proof を一回だけ実行せず、bounded deadline と backoff を持つ readiness loop を設ける。
2. 同一 PID/starttime と active manifest の安定を確認しつつ、5 endpoint すべての 200、model ID 一致、gateway/OpenWebUI の coherent snapshot が揃った時だけ candidate または rollback proof を publish する。
3. deadline failure のときは、credential を含めずに operation return code、sanitized stderr/cause、timestamp、各 endpoint の結果を root-only immutable audit に保存する。stage state は `failed` とし、`pending` を `not_run` に潰さない。
4. 同じ wait/proof contract を rollback/recovery にも適用する。direct worker ready time（約 5–6 秒）より十分長い、レビュー済みの deadline を使う。
5. 新 plan の read-only preflight と明示的人間承認が揃うまでは activation を呼ばない。candidate closure は新 plan の immutable input として revalidate すればよい。

## 非介入の確認

この診断では activation、rollback、recovery を一切呼んでいない。`/etc/ullm/served-models/active.json` は read-only に hash/stat しただけで、SHA-256 は original `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` のままである。`ullm-openai.service` に stop/restart/reload 操作は行っていない。直接 worker は service 外の別 process として実行し、両方を timeout 後に終了させた。最終 read-only check では service が `ActiveState=active`、`SubState=running`、`Result=success` であり、最初の直接 test 開始以後の unit journal entry はない。詳細は `safety-verification.txt` にある。
