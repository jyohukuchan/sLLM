# AQ4 production prefill/decode P2 preparation（2026-07-14）

## 前回の要点

- P0/P1の計画・検証契約は、active identityをhashで固定し、component・full_model・production_serverを混同せずにbaselineを取得する方針である。
- P2は、現active Qwen3.5-9B AQ4_0について、CPU referenceとR9700/RDNA4を同一ケース定義で比較できるbaseline準備を担当する。

## 今回の変更点

- `benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json`を追加した。prompt 1/8/32/128/512/1011/1024/1339/2048/3584/4096、prefill M 1/8/16/32/64/128、all_m1/cold_batched/cached_prefix_chunked、decode context 16/128/512/1024/1339/2048/3584・request_count=1・生成64、3 scope、CPU/R9700、AQ4_0/SQ8_0/reference controlを定義した。
- smoke→representative→fullの順序、ケース数、context edge、安全なストリーミング条件、R9700排他queue、実binary/package・実M・resolved M・fallback・state resetを記録するcommand-adapter要件を定義した。全stageはP1 gate確認前に実行不可である。
- `benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json`を追加した。model/manifest/worker/package/oracle/baseline/power/policyのhash binding、R9700電力・VRAM捕捉、prefill 1011/2048/1024、decode 1339/short-context、OOM/fallback、SQ8_0/reference control依存を未束縛テンプレートとして定義した。
- `benchmarks/workloads/validate-aq4-production-opt-p2-manifest.py`を追加した。CPU上で重複JSONキー・非有限値・軸・stage順序・ケース数・control依存・R9700排他・安全条件・policy閾値を構造検証する。worker、GPU、network、live requestは実行しない。
- 追加レビューを反映し、stageごとの軸・device/control対応・重複を完全一致で検証するようにした。`1e999`のようなJSON数値の非有限化も再帰的に拒否し、preflight項目、排他queue、atomic publication、adapter要件を省略できない契約にした。
- `cached_prefix_chunked`は非空cached prefix（128 token）と、prefix + 新規tokenが4096以下となるprompt集合を明示した。all-M=1 path oracleのcase link/hashを必須化した。
- decodeではprefill Mとrequest countを分離し、`decode_request_count=1`として記録する。case IDにもdecode request countをMとして解釈しない規則を追加した。
- Qwen3 dense full-model controlとV620 capability decisionをP2実行行から分離した上位validation dependencyとして追加した。外部artifactが未bindingの間はpromotion不可である。
- policy templateにbound status、SHA-256・power・correctness thresholdの必須binding項目とcanonical self-hash規則を追加した。unbound templateはcase実行前に拒否するadapter契約である。

## 検証

- 実行済み: `python3 benchmarks/workloads/validate-aq4-production-opt-p2-manifest.py benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json --policy benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json`
- 結果: `valid=true`。smoke 84件、representative 1,705件、full 3,075件の計画ケース数を再計算した。
- 実行済み: 標準JSONパーサーによる両JSONの読み取り、およびvalidatorのCPU実行。
- 実行済み: validatorの変異試験でstage軸欠落・重複、decode request count逸脱、cached-prefix edge超過、非有限値をすべて拒否することを確認した。
- 未実施: GPU実行、R9700電力取得、resident worker/live request、P1 runner変更、production trace/result生成。親エージェントがP1 Gateを確認するまで実施しない。

## 次の行動

- P1 Gate後に、親エージェントが同一active identity・policy hash・power captureを束縛してsmokeをCPU→R9700の順に実行する。
- P2 runner接続前にmanifestをflattenし、manifest SHA、bound policy、実binary/package、構造化executor record（requested/resolved M、actual token/request width、fallback、preflight/peak、prepare/commit/discard/reset）を同じrun rootへbindingする。既存P1 runnerのelapsed-only raw行をperformance evidenceへ昇格させない。
- representativeへ進む前に、SQ8_0はcross-format control/reference用途に限定し、AQ4_0と同一case identityのcontrol evidenceが揃わない行をineligibleにする。
- full matrixはR9700を常に1ケースずつ実行し、OOM・fallback・reset未完了を成功へ置換しない。Qwen3 dense control artifactとV620 capability status（supported/unsupported/skipped＋理由）をP6前にhash-bindingする。
