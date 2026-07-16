# P3 direct instrumentation evidence hardening

## 前回の要点

候補Aのdirect trace producerはruntime countersと外部profiler observationを結合していたが、request単位の発行経路、error terminal、計測lane、profiler rawの由来、候補Aのenriched bindingが十分に強制されていなかった。

## 今回の変更点

- runtimeにrequest単位のdefault-off collectorを追加し、binding検証、reset、全error terminal、GPU同期後のcounter確定、exactly-once serializationを実装した。
- runtime由来のcounter/safetyを`instrumented_diagnostic`かつ測定不適格、profiler由来のlatency/peak/fidelityを`profiler_off_measurement`かつ測定適格としてevent単位で固定した。
- profiler executable/version/SHA、raw path/SHA/inode/link count、command/exit/timestamps、case/identity/request、parser SHAを固定するproducerを追加し、raw sampleから値を再計算するようにした。
- assemblerとselection raw builderで、raw tamper、TOCTOU、hard link、oversize、lane laundering、旧形式の候補A bindingをfail-closedで拒否した。既存のE/N契約は変更していない。
- Rustが実際にserializeしたfixtureをPython assemblerへ渡す互換テストと、error/reset/cancel/overflow/改ざんの否定系テストを追加した。

## 次の行動

実R9700/HIPの保守窓で、instrumentationを無効にしたprofiler-off rawを正式commandから採取し、7 promptとfull-model pairsを揃えてselection rawを再生成する。それまでは実GPU計測、ゼロオーバーヘッド、promotionを未証明として維持する。
