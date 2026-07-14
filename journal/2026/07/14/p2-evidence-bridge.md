# AQ4 P2 証跡ブリッジ QA 修正

## 前回の要点

初版 `c8ef5ee` ではケース展開、identity binder、runner、prefill result builder、独立validatorを追加した。しかし、cached-prefixをcold stateのall-M1へ結び付けていたこと、bound policyが未完成だったこと、runnerが任意commandを許したこと、正式result schemaと完全matrix検証が不足していたため、QAはFAILだった。

## 今回の変更点

- cached-prefixは非空prefix軸だけを展開し、同じphase、prefix、prompt、requested M、device、controlのall-M1へ結び付けた。`4096 + 128` は生成しない。完全件数はsmoke 84、representative 2,245、full 3,885、合計6,214件である。
- binderはexpanded fileと各case self-hashを結び、任意サイズのpackage fileをstreaming SHA-256で処理する。`effective_at`、power 4値、correctness 5値を必須にし、最終bound policy形に対してself-hashを計算する。
- runnerは宣言済みworker/package/policy/identity/caseだけを受け付け、任意`--command`を削除した。argv値を証跡へ保存せず、canonical R9700 lock、GPU process snapshot、VRAM headroom、streaming output境界を検証する。
- builderは正式な `ullm.prefill_validation.v1` を生成する。raw statusは上書き不可で、source oracle validator、same-state path oracle result、独立validation artifact、trace、2 warmup + 10 measured、p50/p95、TTFT、ITL、state/reset/fallback/memory、baseline regressionを実検証する。CPU syntheticは常にpromotion不可である。
- validatorは6,214件の完全matrixを要求し、partial、duplicate、extra、全case identity、run-root path、case/raw/oracle/policy/trace/state/measurement/regressionのhashと内容を再構築する。component/full-model/control/CPU syntheticのpromotionを許さない。
- negative testsを12件へ拡張した。65MiB sparse package、1/6,214 partial matrix、dummy trace、cached state mismatch、lock競合、foreign GPU process、32/33-byte出力境界などを含み、全件成功した。P1 trace tests 7件も成功した。
- 再QAで、trace JSON内の64桁文字列だけを独立検証として扱える欠陥が見つかった。builderとfinal validatorの両方からP1 strict validatorを再実行し、trace manifest、executor record、binding、detached report、aggregation source tracesの実体・run-root・SHA-256・scope・status・promotionを照合するよう修正した。`report_sha256="x"` と64個の`0`を使うbuilder/final validator negativeを追加し、専用testは14件になった。
- 次のQAで、strict-validなP1 trace bundleを別P2 case/rawへ流用できる欠陥が見つかった。各caseの`fixture_id = case_id`を固定し、rawへcase contractとtrace path/hash/trace_idを保存する。builderとfinal validatorはrequest summary、phase kind/mode、token/context/generated count、requested/resolved/actual width、device、model/served manifest/worker/package/artifact identity、sampling/control、per-sample timing aggregation、terminal auditを完全一致で再検証する。full bundle fixtureの正例と、別decode case・別prefill raw linkへの差し替え負例を追加し、専用testは15件、P1との合同testは22件になった。
- その後のQAで、実際の6,214件にはassociationが要求する実装、正規化device、sampling/control、request_countが不足し、production pathが常時拒否される欠陥が見つかった。workload/expanderで全caseへこれらをhash-bound fieldとして展開し、prefillの単一requestとdecode request countを分離した。手作りcaseの正例を廃止し、実CLIのexpander→binder case、P1 strict-valid bundle、associationを通す統合testへ置き換えた。件数は6,214件のままで、合同test 22件とplanning validatorが成功した。
- 標準AQ4 targetのimplementation IDがP0 snapshot/full-model contractと異なっていたため、`qwen35_aq4_rdna4_v1`へ統一した。expanderはplanning identity bindingとtarget control、binderはhash-bound target casesとmodel identityおよびserved-model formatの一致を必須にした。標準manifest由来caseと実モデルcontractの正例、model implementation差し替え負例を追加し、6,214件と合同test 22件を維持した。

## 次の行動

GPU/liveは実行していない。実R9700 power capture、実worker/package、production-server trace、完全6,214件matrix、P1親ゲートの承認が揃うまでpromotionは常にfalseとする。commit後に独立reviewを再実行する。
