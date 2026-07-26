# AQ4_0 activation readiness v0.2

## 前回の要点

- consumed plan `72140ff475b29e28f4ab6685459a344939bc54fcd12aa4f0b7c44cd7a8753194` の activation は `candidate_live_proof` で readiness race を起こした。candidate closure、manifest、5-file tokenizer は診断で正常と確定しており、作り直し対象ではない。
- candidate worker の direct control は約 4.8 秒で ready、gateway は約 3 秒で startup complete になる一方、旧 route は systemd `active/running` の直後に retry なしの one-shot probe を実施していた。

## 今回の変更点

- activation control を commit `05014a8c22baf050299056a40dd24e44367bc0ef` で bounded readiness/audit/isolated-preflight 対応にし、commit `389a58f534a10afee8c5b1f4f7aa61c4a00aaa39` で gateway MainPID ではなく manifest-bound worker child の実効環境を読むよう修正した。さらに commit `af7298bad50cfc7b8166c5505aaaffe0e9ad465f` で deadline 到達時にも既成功 endpoint state を audit から失わないようにした。MainPID に HIP guards/served-manifest binding がないことは credential 値を出さない key-presence check で確認済みであり、worker child には candidate の 30 guard と R9700 HIP index 1 binding があることを確認した。
- readiness contract は timeout 120 秒、最大 15 attempts、0.5/1/2/4 秒後は 8 秒 cap の exponential backoff、同一 worker PID/starttime を含む coherent pass 2 回連続とした。candidate/rollback/recovery の reconcile と live proof で同じ contract を使う。15 attempts の idle delay 合計は 87.5 秒で、deadline 内に endpoint probe 時間を残す。
- coherent success は active manifest の pass 前後一致、worker command/environment の manifest binding、stable worker process、5 endpoint の HTTP/model-ID success をすべて要求する。timeout/incoherent は success とせず rollback/recovery path へ渡す。
- failed live proof は immutable audit に `stage_status: failed`、時刻、sanitized return code/stderr/cause、stderr/stdout hash、全 endpoint state を保存する。stdout content は保存せず、stderr は 16 KiB 上限かつ bearer/API-key/token/JWT/session/secret/password pattern を redact する。
- tests は CPU/private copy/mock 25 件 PASS。遅延 retry、deadline failure と既成功 endpoint state 保全、部分 endpoint/model mismatch、unstable PID、candidate/rollback reconcile の同一 wait contract、swap 後 SIGKILL、rename 後 fault、one-shot replay 拒否、failed live-proof audit と credential/JWT 非漏洩を含む。

## 新しい sealed plan と preflight

- `af7298ba…` の detached clean standalone control source を `/opt/ullm/aq4-runtime-hardening-v0.1/control-source/aq4-hardening-activation-af7298bad50c/` に root seal した。tree は `d6738b269d30605f1f36edb8fc06b4d698085f88`、Git alternates はない。operation launcher は payload SHA-256 `563560125f08d52cf7fc674d951d8a5eb1120c0a414de6a1f2ab69f408c66d07` を bind する。
- final immutable plan は `/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/activation-plan.json`、SHA-256 `0e12fe09ad4d00578ee74f1bcc730a6b401e63a6fc91bb1d237346251e8f81f8`。既存 candidate SHA-256 `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` と既存 rollback SHA-256 `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` をそのまま bind し、closure/products/releases/tokenizers/sources は再作成していない。
- isolated candidate worker preflight は service user/working directory と worker child から whitelist した offline/cache/HIP/lock environment で candidate だけを起動し、`gfx1201`、`rdna4_aq4_resident`、3,195 ms ready を確認して SIGTERM (`returncode: -15`) で cleanup した。service restart、active-manifest swap、credential output はない。
- final normal read-only preflight は `ready: true`、`blockers: []`、`production_activation_performed: false`。initial preflight の不足は isolated receipt のみで、receipt 後は全 checks PASS。証跡は `benchmarks/results/2026-07-26/aq4-runtime-hardening-activation-v0.2/` に保存した。

## 安全状態

- `/etc/ullm/served-models/active.json` は read-only hash comparison のみで、最終 SHA-256 は rollback target と同じ `5d015a…d1c8a`。変更していない。
- `ullm-openai.service` の stop/restart/reload は 0 回。isolated preflight 前後とも MainPID `2694276`、`active/running`、`NRestarts=0` を確認した。
- V620 は使用していない。isolated candidate worker は service worker と同じ R9700 HIP index 1 (`gfx1201`) だけを使用した。
- 先行する preliminary plan `a1da456f3b4a07a7342c8031da8f8748ccad6a87d13766f37ab6e6cd86dd6184` と r2 plan `0ba5746bb7dbe60bcd79c3b7c2085d816ab37dee7d8a2cb1ddcab45ae5850e06` は immutable evidence として保持した。前者は worker start 前に effective-environment binding の問題を露出し、後者は activation 前に endpoint deadline state 保存を強化するため supersede した。いずれも activation は実行していない。

## 次の行動

1. plan `0e12fe09…` に対しても、明示的人間承認と service-maintenance window なしに `--execute` を実行しない。
2. 将来の承認直前に final read-only preflight を再実行し、active/unit/environment/credential seals と isolated receipt が drift していないことを確認する。drift があれば plan を上書きせず、新しい reviewed plan を作成する。
