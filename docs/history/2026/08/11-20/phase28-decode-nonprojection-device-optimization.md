# Phase 28 decode projection外device最適化 history

## 2026-08-18: 詳細計画

- ユーザーの明示指示により、Phase 28をdecodeのprojection外device処理の短縮へ限定した。projection、host residual、prefill、
  batching、quantization、DeepSeek V4、TurboQuantは実装対象に含めない。
- Phase 27のV620 5.379 ms、R9700 5.257 msという値は、全kernel aggregateからprefill projectionだけを引いてnominal decode stepで
  割ったcoarse residualだった。prefill中のGDN/attention/norm等とR9700 MTP内部workが残るためdecode-onlyではなく、
  llama.cppに対する3.80倍/3.54倍claimを撤回した。
- Phase 28 A0/A1ではexecution transactionのcommitted output stepを正規境界にし、prefill、target decode、MTP draft/verify/replayを分離する。
  evidence/profile modeだけでdispatchをmodel componentとop familyへ写像し、production defaultへper-op timing overheadを追加しない。
- projection外familyをlinear/GDN recurrent、attention preprocess、causal attention/KV、RMSNorm、elementwise、Argmax、device copy/fillへ分け、
  device ns/output tokenとproduction TPOT shareから最大fixable contributionを持つ一work unitを固定する。
- 最初の仮説は、複数state passとserial reductionを持つlinear/GDN recurrent、およびhead-dim 256を1 threadで処理するattention preprocessである。
  順位はfresh両GPU profileで決め、source形状だけでcandidateを採用しない。
- 当初の採用規則は全固定target/patternで非悪化、任意patternでfull-model decode 5%以上としていた。
- 本更新は計画とPhase 27測定限界の訂正だけである。production source、kernel/default、GPU evidenceは変更していない。

## 2026-08-18: pattern限定採用規則への改訂

- ユーザーの明示指示により、全pattern非悪化をshared path採用の条件へ限定した。
- 任意patternでfull-model指標を5%以上改善すれば、他patternに悪化があっても改善patternをstable runtime keyへ限定してcandidateを採用し、
  その他はbaseline providerへrouteする。
- keyはexact target、dtype、semantic op、shape/layout/alignment、request mode、mechanism上意味のあるcontext境界等から実行前に決定する。
  benchmark case名、prompt内容、実測結果、個別token列を使うoverfit分岐は認めない。
- shared semantic/model graphを維持し、pattern差は共通registryのprovider selectionへ閉じ込める。shared pathを管理上優先するが、
  他patternの悪化だけを理由に5%以上改善する安全な限定pathを棄却しない。

## 2026-08-18: adoption scopeの形式化

- ユーザーの明示指示により、単一benchmark patternではなく、stable dispatch keyで同じproviderへrouteされるproduction入力集合
  `adoption scope S`を採用単位にした。
- `S`の代表full-model caseの一つ以上が5%以上改善し、`S`内の全validation caseが非悪化ならcandidateを`S`へ採用する。
  `S`外はbaseline providerへrouteし、provider identityとselection overhead込みの非悪化を確認する。
- contextや数値範囲のkeyには`B-1/B/B+1`とscope内の複数代表値を要求し、単一benchmark点、prompt、token列、実測後の結果を
  routing keyにしない。
- `S`が固定matrix全体ならshared adoption、真部分集合ならscoped adoptionとし、final performance run前にkey、代表case、境界、
  baseline complementをmanifestへfreezeする。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)

## 2026-08-18: 実装・採用

- target-only benchmarkの224 Argmaxを14 request × 16 tokenへ分割し、prefill terminalを除く210 committed decode stepを集計した。
  projection外device時間はV620 3.584 ms、R9700 3.328 ms/tokenで、最大familyはGDN recurrentの1.763/1.415 msだった。
- transactional stateの初回copy、decay、previous projectionを一passへ統合した。FP32 state、BF16 RNE、演算順、double-buffer publication、
  semantic graphは維持し、target分岐を追加していない。
- GDN device時間はV620で23.18%、R9700で56.23%短縮した。同一source/inputの3 warmup + 10 measured full-modelでは
  32.7703→33.2765 tok/s（+1.54%）、37.2055→38.3093 tok/s（+2.97%）だった。token IDs一致、all-HIP、fallbackなし、cleanup正常を確認した。
- 通常の5%採用基準には未達だが、ユーザーの明示指示により規則を変更せず本candidateだけを例外採用した。例外はGDN shared kernelに限定し、
  今後のcandidateへ前例として自動適用しない。正本は
  [Phase 28 bounded summary](../../../../../ci/matrix/phase28-nonprojection-summary-v1.json)とする。
