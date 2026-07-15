# P3 READY candidate maintenance consumer

## 前回の要点

- profile actual v4のcapture failureはschema v1で、READY候補の監査情報を持たないimmutable historical evidenceだった。
- runner/capture側はfailure schema v2と`ready_candidate_audit` envelopeを追加したが、maintenance consumerはv1のroot keysだけを検証していた。
- B4では、launcherから返された未検証のprofile lifecycle情報をlock substrate cleanupへ渡さない契約が必要だった。

## 今回の変更点

- maintenanceの新規capture failure契約を`ullm.aq4_p3_diagnostic_rocprof_failure.v2`へ更新し、capture hardening commit `0e8bf9f47583d10cf4daf1092aef5a0e388aa496`、source SHA-256 `b66ef14ebaaa9b2828dbe17e93aeed13595284e361776e7c67dc197e318f01af`へpinした。
- v2 rootと`ready_candidate_audit` envelopeのexact keys、schema、self-hash、型、サイズ、stream SHA-256、marker count/SHA-256を独立検証する。validは`rocprof.stderr`内のmarker payloadが保存auditのcanonical bytesと完全一致する場合だけ受理する。
- auditはraw byte count/hash、safe key/type要約、safe scalar、session要約、nested identity/binding hash、7 predicatesを検証し、要約からpredicateを再計算する。最初のfalse predicateとreason code、または全predicate通過後のdownstream failure reasonを厳密に対応させる。
- secret名・secret値、raw path、`/proc/self/fd`、extra field、未知status/reason、self-hash不一致、stream/audit不一致を拒否する。absentはmarker count 0だけを受理し、invalidはmalformed、oversize、multiple、termination差のbounded診断だけを受理する。
- READY envelopeのvalid/absent/invalidは診断情報に限定した。process group cleanup、children state、restore判断は従来の独立したlifecycle fieldsからだけ決定し、invalid envelopeから安全判定を作らない。B4の未検証raw lifecycleによるcleanup禁止テストも維持した。
- schema v1は新規captureでは拒否し、明示的historical readback経路だけでversion-dispatchする。actual v4 failure SHA-256 `58619cb05c13cac5fed392d587c7d9878a53bba6ed02ace15e1c37d5969e99c5`をimmutable fixtureとして読み戻し、READY auditなし・cleanup complete・children known emptyを確認した。
- 既存ready artifactsは新consumer/capture pinより古いため変更しなかった。通常readbackとdry-runはsemantic binding mismatchでprocess起動前にfail closedすることへ期待を更新した。
- actual、GPU、service、HTTP、rocprof実機実行は行っていない。`pytest -q tests/test_aq4_p2_resident_smoke_maintenance.py`は155 passed、`py_compile`と`git diff --check`もPASSした。

## 次の行動

- 別laneでcurrent maintenance source、QA attestation、profile ready/bindingを再生成し、新しいcapture pinとfailure v2必須契約へ結び直す。
- 次のactual authorization前にfake profile failureでvalid/absent/invalidのevidence retentionと、invalid時にもunvalidated cleanupが禁止されることを再確認する。
