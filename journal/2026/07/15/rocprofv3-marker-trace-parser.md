# rocprofv3 marker trace parser

## 前回の要点

actual profile v7ではkernel、HIP API、memory copy traceが生成された一方、`*_marker_api_trace.csv`は0件だった。runnerは旧`libroctx64.so.4`をpinned FDから`RTLD_LOCAL`で直接loadしており、rocprofv3がpreloadするSDK ROCTxのsymbol overrideを通らなかった。

CPU対照ではrocprofv3 1.1.0のmarker出力名が`*_marker_api_trace.csv`であることを確認した。CSV schemaは`Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp`であり、balanced push/pop rangeは`Domain=MARKER_CORE_RANGE_API`、marker文字列は`Function`へ出力される。

## 今回の変更点

`tools/capture-aq4-p3-diagnostic-profile.py`は、`Function`をmarker名として読む場合に全行の`Domain`がexact `MARKER_CORE_RANGE_API`であることを必須にした。未知domain、大小文字違い、空domain、混在domain、`Function`欠落、legacy名列との併存をfail-closeする。従来の`Name`、`Marker_Name`、`name` schemaは、`Domain`を持たない既存形式として維持した。

fixtureを実rocprofv3 1.1.0のheaderへ更新し、legacy schema互換、domain制約、missing Function、duplicate header、無効timestamp、12 range不足を検証した。sealed actual v6 trace inventoryについて、marker風CSVがなく`discover()`が`expected exactly one marker trace, got 0`を返すことも固定した。これにより、今回の失敗がdiscover filename mismatchではなくproducer欠落であることを回帰試験で区別できる。

検証結果:

- `python3 -m pytest -q tests/test_capture_aq4_p3_diagnostic_profile.py`: 55 passed, 1 skipped
- `python3 -m py_compile tools/capture-aq4-p3-diagnostic-profile.py tests/test_capture_aq4_p3_diagnostic_profile.py`: passed
- `git diff --check`: passed

## 次の行動

launcher側でbound ROCTx libraryをSDK `librocprofiler-sdk-roctx.so.1.1.0`へ切り替え、pinned FD + `ctypes`経路のCPU marker goldenを通す。その後にだけ新しいno-reuse output versionへtrust/bindingを再生成する。
