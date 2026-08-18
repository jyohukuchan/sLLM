# Phase 24 prefill terminal-row projection optimization history

## 2026-08-18: Phase 23結果を受けた詳細計画

- ユーザーの明示指示により、Phase 23 shortlist最上位`P23-O1`をPhase 24へ割り当てた。
- Phase 23ではQwen3.5-4B BF16の256-token terminal LM-head-shaped workがdevice timeのV620 13.48%、R9700
  46.92%を占め、production E2E Amdahl上限を13.06%/37.92%、現実的期待改善を8〜13%/20〜38%と評価した。
- 現行frontendがprefill outputの最終tokenと最終行logitsだけを消費する一方、Qwen/Gemma graphが全`M`行へLM headと
  Argmaxを実行するsource contractを再確認した。
- normal prefillをlast-row mode、speculative block等の明示all-logitsをall-row modeとして分離し、final RMSNorm、MTP hidden、
  multimodal、KV/state、sampling、public APIを維持する計画とした。
- primaryはQwen3.5-4B BF16とcanonical V620/R9700である。Gemma 4はR9700 M>1 baselineが5% E2Eまたは10% device shareの
  事前gateを満たす場合だけ同じPhaseへ含め、Qwen primaryの完了をGemmaへ依存させない。
- adoption thresholdは256-token prefill E2Eを両GPUで5%以上改善、short prefill/long decodeの悪化を2%以内、
  last-row/all-row correctness、sampling/MTP/multimodal state、VRAM high-waterを必須とした。
- 5%/2%、Gemma 5%/10%、case/evidence範囲はPhase 23結果に基づくAI提案であり、今回の計画作成指示だけではhard gateにしない。
  ユーザーがPhase 24開始またはplanを承認した場合にP24-A0で凍結し、同一candidate中の緩和を禁止する。
- Phase 24へprojection-family fusion、continuous batching、cold loader、provider tuning、TurboQuant、DeepSeek V4を含めない。
  candidateが基準を満たさない場合はproduction defaultへ残さず、否定結果でPhaseを完了する。
- この時点ではactive planとmain plan/historyの同期だけを行った。production source変更、schema/runner実装、baseline再取得、
  GPU実行、Gemma scope gate、candidate実装はまだ開始していない。

## 2026-08-18: bounded実装、dual-GPU採否、candidate棄却

- ユーザーのPhase 24開始指示により、256-token / 2-output E2EをV620/R9700の両方で5%以上短縮し、short prefillと
  long decodeの回帰を2%以内にするcriteria、およびprovider tuning/fusionを混ぜないscopeを凍結した。
- final RMSNormの最終行aliasだけをterminal projectionへ結び、通常prefillのlogits/Argmax descriptorを一行へするprivate
  candidateを作成した。明示all-logits pathは全行のままとし、`M=1,2,3,17,255,256,257,2047,2049`のchecked row viewと
  MTP all-rowのhost testを含むfocused test 17件をcandidate上でPASSさせた。
- Phase 23 binaryをbaseline、candidateをtarget別release binaryとして、同一model lock、256 input token、greedy、thinking disabled、
  2 output token、3 warmup + 10 measuredで再計測した。全採用runはHIP-only、fallbackなし、cleanup 0、生成token
  `[9419,0]`、stop `max_new_tokens`で一致した。
- baseline→candidate→baselineのbracketを取得した。V620はbaseline E2E平均2,338,783,930.75 nsに対して
  candidate 2,041,489,764 nsで12.71%、prefillは13.17%短縮し、target単独thresholdを通過した。
- R9700はbaseline E2E平均394,173,254.5 nsに対してcandidate 393,788,256 nsで0.10%、prefillは0.24%の短縮だけで、
  5% gateを失敗した。
- native providerを確認すると、R9700の全行`M>1`はhipBLAS GEMM、一行`M=1`は既存decode reduction kernelだった。
  このprovider遷移によりPhase 23 profiler shareから見込んだ削減がproduction wallへ転化しなかった。
- candidateは既存どおりfull `[M,vocab]` request bufferを確保していたため、workspace high-waterはbaseline/candidateとも
  1,149,766,656 bytesで、physical allocation縮小criterionも未達だった。
- R9700のrocprof observer runは`generation service failed: execution published no device Argmax token`でFAILした。
  profiler wallを採用値には使わないが、candidateの追加correctness riskとして記録した。
- primary P2が一方のtargetで失敗した時点で、凍結planに従いtarget固有provider tuning、fusion、2K救済、Gemma extensionを
  追加しなかった。P0/P1/P3/D0を追加しても共有candidateの不採用判定を変更できないため、その後のadoption/regression laneも
  実行しなかった。
- candidate production sourceとcandidate専用host testを除去し、元のall-row behaviorへ戻した。復帰後の
  `cargo test -p sllm-core qwen_execution --no-fail-fast`は15/15 PASSした。
- 結論はPhase 24 candidate棄却である。再検討する場合はR9700のterminal projectionとprovider選択を一体で扱う別scopeとし、
  今回のshared candidateを暗黙に再採用しない。

## 2026-08-18: 採用基準改訂によるPhase 24再開

- ユーザーは旧「両GPU各5%以上」を、「全対象patternで悪化がなく、任意のpatternで5%以上改善」へ明示的に変更した。
- 問題がない限りgfx1030/gfx1201は共通経路を使い、correctness defectまたは再現する性能悪化がtarget固有の場合だけ分岐する。
- 旧基準で棄却したshared last-row candidateを復元し、未完だったphysical allocation、GPU numerical/sampling/all-row、
  P0/P1/P2/P3/D0のdual-GPU確認を完了して新基準で再採否する。
- 旧結果は探索履歴として保持するが、Phase 24の最終状態ではない。

## 2026-08-18: shared terminal-row candidate採用・完了

- `TerminalOutputRows::{Last, All}`をQwen executionのprivate contractとして復元し、gfx1030/gfx1201で共通のloweringを採用した。
  通常requestの255 token以上だけをlast-rowへ切り替え、255未満、明示all-logits、MTP target、MTP draftはall-rowを維持した。
- 初回profilerで発生したdevice Argmax未公開は、MTP target frontendが行別Argmaxと全hidden rowを消費するのにterminal
  outputだけを一行へ縮小していたことが原因だった。MTP targetをall-rowへ戻し、normal all-row実行順も旧pathのまま分離した。
- `cargo test -p sllm-core qwen_execution --no-fail-fast`は19/19 PASSした。distinctive-row GPU oracleは両targetの
  `M=2,3,17,255,256,257`でlogical `M-1`を選び、projection/Argmaxの最大絶対誤差は0、fallbackなし、cleanup 0だった。
- full-model correctnessはgreedy token `[9419,0]`、3 sampling profileのtoken列/stop/usage、MTP幅2のoutput、audit、cleanupが
  baseline/candidateおよび両targetで一致した。profileでは通常P2のterminal projection/Argmaxが各一行であることを確認した。
- 改訂後の固定matrixでは全10 target/patternが非悪化だった。gfx1030 P0/P1/P2/P3/D0は
  0.14%/13.14%/12.08%/12.73%/0.17%、gfx1201は0.32%/0.40%/0.49%/0.35%/0.09%改善した。
  任意pattern 5%以上を満たすためcandidateを採用し、target固有分岐は追加しなかった。
- P2 workspace high-waterは1,149,766,656 bytesから1,023,122,436 bytesへ126,644,220 bytes減り、
  model-resident 8,411,592,192 bytesは不変だった。physical terminal allocation縮小も完了した。
- Gemma extension、projection-family fusion、provider tuning、continuous batchingは追加せず、Phase 24を完了した。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase24-prefill-terminal-row-projection-optimization.md)
[bounded summary](../../../../../ci/matrix/phase24-terminal-row-summary-v1.json)
