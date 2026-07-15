# P3 READY candidate failure observability

## 前回の要点

- profile actual v4はresident runnerが`resident driver did not prove one model load`で終了したが、partial runner rootとdriver process v2が残らず、READY候補のどのpredicateが不一致だったかを特定できなかった。
- runnerの`_recv`はvalid JSON objectまで受信していたため、timeout、EOF、invalid JSONではなかった。従来の一括predicateはfield set、event、schema、`model_loads`、session IDを同じエラーへ畳み込んでいた。

## 今回の変更点

- runnerはREADY JSON objectをvalidation前に、raw byte count/SHA-256、sanitized sorted keysとJSON type、bounded safe scalar、session IDのpresence/type/length/hash、nested identity/bindingのcanonical SHA-256とkeys/typesへ変換する。session ID、logical path、descriptor、nested raw valueは保存しない。
- 7個のREADY envelope predicateとstable reason codeを追加した。identity、binary SHA、served-model bindingの後段失敗もcoarse stable reason codeへ分離し、成功時は`ready_candidate_valid`をdriver process v2へself-hash付きで保存する。
- READY validation failureは従来の75-byte generic stderr行を先頭に維持し、16 KiB上限の`ULLM_AQ4_READY_CANDIDATE_AUDIT_V1` compact JSON markerを続ける。process evidence writeが後から失敗してもsafe in-memory auditからmarkerを出す。
- capture failure schemaをv2へ更新した。`rocprof.stderr`をbounded line scanし、marker exact-one、schema、exact fields、JSON types、audit self-hashを検証する。valid markerはstream SHA-256、marker SHA-256、audit SHA-256、capture binding self-hashとともに`ready_candidate_audit`へ保存する。absent、malformed、oversize、multiple markerは成功扱いにせず、absent/invalid reason codeでfail-closedに記録する。
- actual v4と同じgeneric stderr prefixをfixtureに使い、runner rootが不存在でもcapture failure evidenceへauditが残る回帰テストを追加した。Rust driver sourceのREADY field goldenをPython validatorへ渡すcross-language test、success process v2、secret/path omission、fallback marker、malformed/oversize/multiple testsも追加した。
- actual、GPU、service、HTTP、rocprof実機実行は行っていない。`pytest -q tests/test_run_aq4_p2_resident_batch.py tests/test_capture_aq4_p3_diagnostic_profile.py`は85 passed / 1 skipped、`py_compile`と`git diff --check`もPASSした。

## 次の行動

- capture failure v2の新fieldを読むmaintenance validatorと、そのsource SHA/pin、profile ready/binding/generated artifactは別laneで再生成する。今回のlaneではRust、prepare、maintenance source/artifact、generated evidenceを変更しない。
- 次のsingle-use profile authorizationではfresh output versionを使い、failureならcapture rootの`ready_candidate_audit`だけでREADY predicateを判定できることをactual前のfake integrationで確認する。
