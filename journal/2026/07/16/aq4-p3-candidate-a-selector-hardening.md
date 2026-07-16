# AQ4 P3候補A selector/producer監査修正

## 前回の要点

- 候補Aのdirect sequence-output証拠契約は、代表7件×10 measured runとfull-model pairをhash-bound traceから生成する。
- selector監査では、pairだけを含むraw inputのcapabilityが集約から漏れる問題と、整数sampleのmedianを`int()`で切り捨てる問題がNO-GOだった。
- D2D・resource・fallback・safety・trace bindingの否定系matrixも不足していた。

## 今回の変更点

- selectorは候補のmeasurementまたはfull-model pairを含む全raw sourceでcapabilityをAND集約する。pair専用sourceのfalse/missingも候補を不適格にする。
- D2D bytes、workspace bytes、peak VRAMの整数sample medianは整数またはexactな半整数として保持する。`.5`を切り捨てず、exactに表現できない範囲と他の分数はfail-closedにする。
- producerのcount medianも浮動小数点変換を使わず整数演算で計算し、半整数になるcountは従来どおり拒否する。
- selectorの否定系へ、measurement/pairのD2D copy・launch・workspace・peak・fallback回帰、fallback reason subset、安全性3項目、component/full-model p50/p95逆転、pair-only capability falseを追加した。
- producerの否定系へ、trace missing/unknown/duplicate/non-finite/reuse、root/schema/file/identity/case/run/pair tamper、trace内および10-run間のfidelity不一致を追加した。
- 非一様10-run fixtureでcomponent/full-model p50/p95と、D2D/workspace/peakの`.5` medianを確認した。
- selector/producer仕様書へsource集約、median表現、否定系matrixを追記した。

## 検証

- `PYTHONPATH=. pytest -q tests/test_select_aq4_p3_candidate.py tests/test_build_aq4_p3_selection_raw.py tests/test_capture_aq4_p3_diagnostic_profile.py tests/test_profile_aq4_p2_family_exclusive.py`
  - 130 passed
- `git diff --check`
  - passed
- GPU、R9700、worker、production service、実raw生成は実行していない。

## 次の行動

- 修正差分のidentity、親commit、runtime非変更を再監査する。
- jobs=1の対象testとPython構文検査を最終実行する。
- 監査修正を新規commitとして保存し、commit/tree/archive digestを報告する。
