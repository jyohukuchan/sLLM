# AQ4 P3 selection raw producer

## 前回の要点

- selectorは7代表promptのE/N、4/7、M=128+別M、paired full-model 95% CIを再計算する。
- 現行family-exclusive profileは診断専用で、D2H・stream/device同期の一次行を持たず、promotion rawを作れない。
- selector QAでnumeric type、nested unknown、non-finite、派生overflowのfail-openが見つかり、`d7c6c5e`と`14826e2`で段階的にstrict化した。

## 今回の変更点

- `tools/build-aq4-p3-selection-raw.py`を追加した。
  - P2 identity、resident summary/raw、rocprof kernel/HIP API CSVをhash-bound入力として読む。
  - kernel family exclusive時間をsweep-lineで再計算する。
  - D2Hとstream/device同期の回数、重複を除いたunion時間を一次API rowから再計算する。
  - unknown/ambiguous API、empty API trace、trace incompletenessを0と推定せず拒否する。
  - 7 prompt、M幅、measured indices 2..11、trace非再利用、同一identity/workload/run indexのfull-model pairを検証する。
  - promotion modeとone-case diagnostic modeを分離する。
- selector raw v1 measurementへ`d2h_time_ms`と`stream_sync_time_ms`を追加した。paged KVは回数と時間の両方を必須とする。
- `tests/test_build_aq4_p3_selection_raw.py`へsynthetic rocprof/resident fixtureを追加した。
  - D2H/sync分類、重複interval union、unknown/empty API、unknown kernel、hash swap、missing prompt/M、pairing、順序不変、profile eligibility/completeness、diagnostic非promotionを検査する。
- 独立QAで、HIP launch/H2DだけのAPI traceがD2H=0・sync=0として通る問題と、`reset.attempted=true`が整数1として通る問題を確認した。
- `ullm.aq4_p3_rocprof_capture_capabilities.v1`を追加した。
  - file SHAとself-hashでcapabilityを固定する。
  - kernel/HIP API、D2H、stream/device syncの全domainと`api_filter=all_functions`を要求する。
  - 0件はcomplete domain captureを証明した非空traceでだけ受理する。
- resident rawのnested境界をdriver実schemaに合わせてstrict化した。
  - baseline identity、resident、device lock、workload、linksをexact検証する。
  - audit、state、lifecycle、reset、resource、terminalをexact検証し、bool/int/float代用を拒否する。
- capability missing/hash swap/incomplete/unknown、HIP launch+H2D zero、reset bool代用の回帰試験を追加した。
- 追加QAで、`timing.prefill_ms=100`や`resource.samples[].monotonic_ms=1`の整数がfloat fieldを通る問題を確認した。
- `finite()`を`finite_float()`へ分離し、resident rawの全float fieldで`type(value) is float`を必須化した。5 fieldの型matrix回帰試験を追加した。
- `docs/specs/aq4-p3-selection-raw-producer-v0.1.md`へ入力、一次trace、hash、統計、pairing、fail-closed契約を固定した。

## 検証

- selector strict fix `d7c6c5e`: selector 23 passed、profiler 27 passed
- selector type fix `14826e2`: selector 26 passed、profiler 27 passed
- `python3 -m pytest -q tests/test_build_aq4_p3_selection_raw.py tests/test_select_aq4_p3_candidate.py tests/test_profile_aq4_p2_family_exclusive.py`
  - capability/strict-type修正前: 64 passed
- `python3 -m pytest -q tests/test_build_aq4_p3_selection_raw.py`
  - capability/strict-type修正後: 16 passed
- `python3 -m pytest -q tests/test_build_aq4_p3_selection_raw.py tests/test_select_aq4_p3_candidate.py tests/test_profile_aq4_p2_family_exclusive.py`
  - capability/strict-type修正後: 69 passed
- `python3 -m py_compile tools/build-aq4-p3-selection-raw.py tests/test_build_aq4_p3_selection_raw.py`
  - passed
- `git diff --check`（producer、test、spec、journal）
  - passed
- float/int相互代用修正後のproducer + selector + profiler: 74 passed
- `python3 -m py_compile tools/build-aq4-p3-selection-raw.py tools/select-aq4-p3-candidate.py tests/test_build_aq4_p3_selection_raw.py tests/test_select_aq4_p3_candidate.py`
  - passed
- GPU、R9700、worker、serviceは実行していない。

## 残課題

- 実rocprof captureはまだproducer manifestを生成していない。
- 現行one-case artifactはdiagnostic modeだけで利用でき、promotion rawには7×10のkernel/API traceとpaired resident rawが必要である。
- `hipMemcpyAsync`のdirectionを引数から証明するparserは未実装で、現状は明示的DtoH APIだけを受理する。

## 次の行動

- 実P2 capture laneが各measured runのkernel/API traceを別fileへ保存し、producer manifestを作る。
- producer rawをselectorへ渡し、selected candidateが決まるまでruntime候補を実装しない。
- one-caseは`one_case_diagnostic`のまま保持し、promotion evidenceへ混ぜない。
