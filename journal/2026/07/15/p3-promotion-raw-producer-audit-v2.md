# P3 promotion raw producer audit v2

## 前回の要点

- `build-aq4-p3-selection-raw.py` は、P2 resident raw/summary と identity、rocprof kernel/HIP API一次CSVを入力にして、selector互換の raw evidence を生成する。
- one-case diagnostic は `measurement_eligible=false`、`smoke_only=true`、`promotion_eligible=false` のまま保持し、promotion raw へ流用しない。

## 今回の変更点

- 現行 main (`ec257544`) から専用 clean worktree `/tmp/ullm-p3-promotion-raw` と通常 branch `p3-promotion-raw` を作成した。
- selector/raw producer仕様、candidate selection仕様、one-case rocprof仕様、および `4c89c602` 以降の profile/evidence変更を照合した。
- 既存契約を重複実装せず、producerのidentity入力境界だけを厳密化した。
  - identity root、resident driver、runtime device、hash binding の exact fields を検証する。
  - build commit、runtime device、source/build/package/worker/served-model の SHA-256 を必須化する。
  - resident driver と hash binding の package/worker/served-model hash の一致を検証する。
  - full-model pair が同じ case・measured run sample を二重利用しないことを検証する。
- selector互換の出力schema、既存の7代表case×10 measured、M=128+別M、D2H/sync一次trace再計算、trace hash非再利用、統計再計算、atomic create-new出力は変更していない。

## 契約監査結果

| 契約 | 現状 |
|---|---|
| 7代表case | promotion modeでexactly 7件、prompt/case SHA重複を拒否 |
| 10 measured | 各caseのresident run index 2..11をexactly 10件要求 |
| M幅 | `resolved_m=128` と `resolved_m!=128` の両方を要求 |
| full-model paired | promotionで2件以上、baseline/candidate別run・同一case/SHA/workloadを検証。今回、同一case/run index再利用も拒否 |
| D2H/HIP API/sync | 明示API名のみ分類し、一次CSV row数と同種interval union時間を再計算。方向不明/未知/空traceはfail-closed |
| identity/hash | identity self-hash、file SHA、resident/source/build/runtime/package/worker/served-model identityを検証 |
| raw eligibility | promotion と one-case diagnostic のschema/flag/statusを分離。selectorはdiagnostic rawを拒否 |
| output publication | 既存outputを拒否し、同一parentのtemporary fileをfsyncしてatomic publish |
| one-case reuse | diagnosticは1 case、non-promotion、profile binding `measurement_eligible=false` のまま |

## 検証

- `python3 -m pytest -q tests/test_build_aq4_p3_selection_raw.py` — **24 passed**。
- fixtureで次の fail-closed を確認した。
  - identity runtime device、package hash binding、worker hash の欠落
  - full-model pair の measured run sample 再利用
  - 既存の欠落case、M幅欠落、trace欠落/再利用、hash差替え、duplicate、unknown API/kernel、空trace、non-finite、bool/int/float型代用、既存output
- `python3 -m py_compile tools/build-aq4-p3-selection-raw.py tools/select-aq4-p3-candidate.py tests/test_build_aq4_p3_selection_raw.py tests/test_select_aq4_p3_candidate.py` — passed。
- `git diff --check` — passed。
- GPU、rocprof実capture、service、systemctl は実行していない。
- capture/launcher全体の既存テストには、mainで並行更新されたbundle root identityとの不一致による4件の失敗がある。これは本laneのraw producer変更ではなく、capture実測を行わずに親agentへ引き渡す。

## 次の行動

1. 親agentが `p3-promotion-raw` の通常commitをmainへ統合する。
2. 実GPU/profileを行う前に、7 case × 10 measured の各runへ unique kernel/HIP API trace、capture capability、resident identity、full-model pair を割り当てる。
3. producer rawをselectorへ渡し、`selected` 以外ではP3 runtime候補を昇格させない。
