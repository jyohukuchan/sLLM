# AQ4 P3 Candidate A production activation v0.1

## 1. 目的

Candidate Aのdirect sequence-output routeは、環境変数だけでは有効化しない。diagnostic captureとproduction activationを別経路にし、productionではP2 fidelity GOとP3 selectionを再検証したactivation artifactを必須にする。

## 2. activation生成

`tools/activate-aq4-p3-candidate-a.py build`へselection artifactと、そのselectionに使ったpromotion rawをすべて渡す。toolはrawを再検証してselectorを再実行し、既存selection artifactとexact一致することを確認する。

次をすべて満たす場合だけ`ullm.aq4_p3_candidate_a.production_activation.v1`をcreate-newで発行する。

- selection statusが`selected`で、selected candidateが`sequence-output-direct-v1`である。
- Candidate Aのcandidate resultがeligibleである。
- 全rawがpromotion eligibleで、同じbuild identityと同じ`qualified_go` upstream P2 qualificationを持つ。
- selection file、raw file、qualification fileのpathとSHA-256が一致する。
- activation self-hash、candidate/build/profile/selection/qualificationのexact fieldsが一致する。

実際のP2 `rejected_no_go` artifactからactivationを生成してはならない。

## 3. worker gate

production workerは次のCLIを使う。

```text
--served-model-manifest PATH
--p3-production-activation PATH
--p3-production-activation-sha256 SHA256
```

`ULLM_AQ4_PREFILL_DIRECT_SEQUENCE_OUTPUT=1`だけを指定した場合はworker startupを拒否する。production activation、またはdiagnostic gateとbinding sidecarのどちらかexactly oneがdirect requestと組にならなければならない。productionとdiagnosticの併用も拒否する。

workerはactivation、selection、qualification、rawをsingle-link regular fileとして再読込し、file SHA-256、activation/qualification self-hash、raw semantic hash、Candidate A selected/eligible、P2 qualified statusを再確認する。成功後だけprocess内の不可逆activation flagを立てる。runtime routeはdirect requestに加えて、このflagまたはdiagnostic gateがなければ常にOFFである。

## 4. fail-closed条件

missing/unknown field、bool-as-int、candidate swap、selection/raw/qualificationのhash swap、symlink、hardlink、読み取り中のfile mutation、既存outputへの上書きはすべて拒否する。
