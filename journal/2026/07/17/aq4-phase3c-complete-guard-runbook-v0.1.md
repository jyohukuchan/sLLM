# AQ4 Phase 3c complete guard runbook v0.1

## 前回の要点

- `aq4-phase3c-complete-guard-audit-v0.1.md` でsource literalを機械抽出し、layer 0 production M=1 linear pathは9 guard、固定full-model traceは16 guardと確定した。
- `811d4271a9ef92f3df4699f0ba8a1862525e2661` は、必要16件の欠落をGPU runtime/context/stream/kernelより前にJSONで全列挙する `--print-phase3c-trace-guard-requirements` と、layer loadの4 RuntimeFeature/9 guard定数を追加した。
- V620、停止中P3 harnessのlock/root/artifact/environmentにはアクセス・変更していない。

## 今回の変更点

- `tools/run-aq4-phase3c-service-window.sh` をtrace tooling commit `811d4271…` に固定した。
- trace childは16 required guardだけを`=1`にし、sourceから抽出した残り34 `ULLM_REQUIRE_HIP_*` 名と、dispatch/fallbackを変える9 selectorを `env -u` で明示的に消す。これによりworkerの30-guard profileを継承しない。
- staged binaryを使うCPU-only診断をservice read/stopより前に追加した。診断JSONがschema、`status=valid`、required 16件、linear-stage guard map 16件を満たさなければexit 39で終了し、serviceを操作しない。
- runbookの固定commit、output leafを新規 `service-stop-window-v0.6-complete-guard-set`、CPU environment、trace command、manifest assertを同じ16 guardへ更新した。既存evidenceは上書きしない。direct runbook側のlock openも `9<>` ではなく既存file read-only open `9<` にした。
- 静的driver testを拡張し、診断がstop前にあり、16 guardすべてを渡すことを検査した。`bash -n tools/run-aq4-phase3c-service-window.sh` と `pytest -q tests/test_aq4_phase3c_service_window_driver.py` は成功した。

## 次の行動

1. source commit `811d4271…` を埋め込んだrelease trace binaryをCPU-onlyでbuildし、新規v0.6 leafへcreate-new stagingする。
2. service稼働中に、staged binaryの16-guard自己診断と、R9700だけに限定したHIP+targeted amd-smi guard chainをリハーサルする。service/systemd/manifestは変更しない。
3. その全てが成功した場合のみ、更新済みdriverでservice-stop windowを一度だけ実行する。traceの成否にかかわらずrestoreを確認し、同一windowでは再試行しない。
