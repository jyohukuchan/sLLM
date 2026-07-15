# P3 profile runner boundary

## 前回の要点

profile diagnosticは`outer maintenance -> capture -> rocprofv3 -> launcher -> runner`だったため、rocprof診断がvalidator stderrへ混入してroot/B拒否になった。またrunnerは固定`EXECUTE_ENV`で起動されるため、rocprof環境がrunnerへ届かなかった。

## 今回の変更点

- launcherにprofile専用runner executorを追加し、validatorとlive gateの完了後だけ選択する。
- live preflight後に、exact runner argv、固定base environment、入力file SHA、runtime path identity、fresh runner output、1回限定authorizationを束縛した自己ハッシュ付きtarget manifestを生成する。
- capture toolはtarget manifestをspawn前後に再検証し、manifestのbase environmentだけをrocprofへ渡す。
- rocprofv3へ`--log-level error`を固定し、warning診断をstderr成功契約へ混入させない。
- profile executor未接続、validator warning再現、environment変更、runtime path置換、target self-hash不一致をfail-closedにした。
- 通常executeではprofile executorを禁止し、既存の非profile runner経路を維持した。
- actual service停止、GPU command、model load、rocprof captureは実行していない。

## 検証

- `tests/test_capture_aq4_p3_diagnostic_profile.py`と`tests/test_aq4_p2_resident_roctx_ranges.py`: 21 passed。
- launcher、capture、ROCTxの通常/profile unit回帰（canonical launcher trust artifact再固定テストを除外）: 86 passed, 1 deselected。
- canonical execute bindingはlauncher自己SHA変更により再固定が必要であり、このレーンではartifactを変更していない。

## 次の行動

maintenanceのprofile分岐をlauncher先行へ変更し、`profile_runner_executor`へcapture toolを接続する。capture tool、launcher、ready artifact、execute bindingのSHA/commitを再固定した後、fake統合テストを通す。actual実行は別の明示的な実行段階まで行わない。
