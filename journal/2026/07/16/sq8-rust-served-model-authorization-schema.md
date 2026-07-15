# SQ8 Rust served-model認可schema修正

## 前回の要点

formal GO auditに基づくone-shot actualは1回だけ実行され、workerが`served-model.json.promotion`を`invalid_manifest`として拒否した。Python gatewayは`authorization_audit`、`authorization_lineage`、`readiness`を認識していたが、Rust worker parserは旧3 fieldだけを許可していた。

## 今回の変更点

- Rustのpromotion contractへ3 fieldの専用型を追加し、全nested objectでunknown fieldを拒否する。
- authorization auditはcanonical absolute path、0444、single-link、live SHAに加え、formal receiptのschema、source、Gate state、topology、execution boundaryを検証する。
- authorization lineageはexact reference、2つのimmutable manifest、live SHA、source、exact 6 entriesとcanonical entries digestを検証する。
- readinessはcontainer、network、bridge interface、endpoint、expected body SHAをPython gatewayと同じ固定identityで検証する。
- 旧3-field manifestは後方互換として許可するが、non-null auditを持つ認可経路ではaudit、lineage、readinessの3 identityを必須にする。
- 実際に失敗したSHA `a4d541a8c44edd73e505f223b15cf92933b4e0bf2a257e8e9d08dbad94192542`をCPU-onlyでloadする回帰testと、unknown、duplicate、missing、type、path、hash、schema、verdict、readiness tamperのnegative testを追加する。

## 次の行動

新source commitからfresh worker、external lineage、unauthorized runtime、formal audit、authorized runtimeをすべてcreate-newで再構築する。消費済みrequestとactual failure artifactは再利用せず、次のactual executionには新しいformal chainと明示的なone-shot認可を必要とする。
