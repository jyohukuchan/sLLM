# P3 READY candidate capture validator hardening

## 前回の要点

- commit `076c3662aad6a3c8c74b3875882df4b41c026de7`はREADY candidate auditをrunner stderr markerとcapture failure evidenceへ保存した。
- capture validatorはschema、field、型、self-hashを確認したが、self-hashを再計算した攻撃者がsafe scalarへpathやsecret-like文字列を入れること、predicateとreason/statusを矛盾させること、nested key/type summaryを偽装することを十分に拒否していなかった。

## 今回の変更点

- captureのsafe scalar検証をrunner producer形式へ合わせた。JSON型をbool/intの混同なしで判定し、記録済みstringは128文字以下、`SAFE_AUDIT_KEY_RE`の許可文字だけ、path separatorなし、authorization/bearer/API key/token/secret/password/credential-likeでないことを要求する。記録済みscalarはcanonical SHA-256を再計算し、empty withheld stringのhashも固定した。
- top-levelとnestedのkey/type summaryはsorted unique、safe keyまたは`sha256:<digest>`、最大1個の`omitted-sha256:<digest>`、omitted/typeの対応、object以外のempty summary、presence/type、top-level key countをexact検証する。secret-like raw keyは拒否する。
- 7 predicateはsafe scalar、session summary、top-level summaryから再導出する。最初のfalse predicateとstable failure reason/statusの対応を固定し、all trueは`passed/ready_candidate_valid`またはproducerが出す3つのdownstream READY failure reasonだけを受理する。
- self-hashを再計算したsecret/path/unsafe charset/overlong string/bool-int confusion/unsafe nested key/secret key/omitted type/unhashable type/wrong reason/wrong status/predicate contradictionをnegative testへ追加した。malformed valueは例外を外へ漏らさずinvalid capture bindingへ閉じる。
- runner producer、maintenance、generated artifact、actual、GPU、serviceは変更・実行していない。

## 次の行動

- maintenance側のcapture failure v2 validatorは、このcapture validatorと同じaudit envelope semanticsを信頼境界として利用し、source SHAとgenerated pinを別laneで更新する。
- fresh fake profile integrationでvalid failure markerとsemantic-invalid markerの双方をmaintenance境界まで通し、前者だけがstructured auditとして保持されることを確認する。
