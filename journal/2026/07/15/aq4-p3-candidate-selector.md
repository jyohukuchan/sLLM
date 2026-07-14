# AQ4 P3候補選定ツール

## 前回の要点

- P3候補1のpaged KV block-table検証削減は、P2で実D2H・stream同期・family wall timeが支配的と確認できた場合だけ実装する。
- 候補選抜条件は、回収可能時間比率`E`がノイズ下限`N=max(5%, 3×baseline CV, 2×CI halfwidth/p50)`を上回り、代表7件中4件以上、M=128と別M、full-model paired 95%信頼区間下限が0超であることだった。
- 現行family-exclusive profilerは診断専用で、`measurement_eligible=false`、D2H/stream同期回数なしのため、単独では候補を選べない。

## 今回の変更点

- `tools/select-aq4-p3-candidate.py`を追加した。
  - 現行diagnostic profileと、新規hash-bound raw evidence schemaを区別して読む。
  - rawの7代表promptから`E_i`と`N_i`を再計算し、median E/N、4/7、M幅を判定する。
  - full-model paired sampleからStudent tの95%信頼区間を再計算する。
  - paged KVにはD2Hとstream同期のcapability・実数を追加要求する。
  - eligible候補を`E-N`、支持数、CI下限、candidate IDの順で決定論的に選ぶ。
  - schema、unknown/missing、duplicate、non-finite、hash/identity swap、smoke/promotion/measurement不可をfail-closedにした。
- `tests/test_select_aq4_p3_candidate.py`へsynthetic fixtureを追加した。
  - eligible、5%/3CV/2CIの各noise項、`E=N`境界、exactly 4/7、M幅、CI下限0、array順序不変を検査する。
  - missing/unknown/non-finite、hash swap、入力eligibility flag、現行profileの証拠不足、D2H/同期欠測、候補順位を検査する。
- `docs/specs/aq4-p3-candidate-selection-v0.1.md`へ入力・hash・統計・出力・失敗条件を固定した。

## 検証

- `python3 -m pytest -q tests/test_select_aq4_p3_candidate.py`
  - 19 passed
- `python3 -m pytest -q tests/test_profile_aq4_p2_family_exclusive.py`
  - 27 passed
- `python3 -m py_compile tools/select-aq4-p3-candidate.py tests/test_select_aq4_p3_candidate.py`
  - passed
- `git diff --check -- tools/select-aq4-p3-candidate.py tests/test_select_aq4_p3_candidate.py docs/specs/aq4-p3-candidate-selection-v0.1.md journal/2026/07/15/aq4-p3-candidate-selector.md`
  - passed
- GPU、R9700、worker、production serviceは実行していない。

## 残課題

- 実P2 producerはまだ`ullm.aq4_p2_candidate_selection_raw.v1`を生成しない。現行profileだけをselectorへ渡すと、意図通り`no_eligible_candidate`になる。
- paged KV candidate用raw producerでは、family exclusive時間とbaseline p50を同じ測定契約へ揃え、D2H・stream同期をraw eventから再計算する必要がある。
- full-model paired sampleは同一case/identity、同一warmup/repeat、baseline/candidateの対応をproducer側で固定する必要がある。

## 次の行動

- P2 evidence producerが新raw schemaを出力するlaneを、P3 runtime実装とは別所有で追加する。
- 実測rawをselectorへ通し、`selected_candidate_id`が確定するまでP3 runtimeの候補コードを変更しない。
- paged KVが証拠不足または統計不合格なら、AQ4 register BM8、chunk実行、fusion候補を同じpolicyで比較する。
