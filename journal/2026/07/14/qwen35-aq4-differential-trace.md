# Qwen3.5 AQ4 differential trace

## 前回の要点

- 保存済みAQ4 path payloadはfinal hidden/logitのbounded sampleだけであり、source/path差分がembedding・decoder層・final norm・LM headのどこで生じたかを判定できなかった。
- GPU再実行とactive service変更は行わず、既存source checkpointと保存済みpayloadだけを対象にする。

## 今回の変更点

- `tools/trace-qwen35-aq4-differential.py` を追加した。CPU source modeはforward hookでembedding、32 decoder層、final norm、LM headを同じprompt/context hashと固定座標で逐次sampleし、全hidden/logitを保存しない。
- path endpoint adapterは既存path payloadを同じtrace schemaへ変換するが、final norm/LM head以外が無い場合は診断を `inconclusive_missing_intermediate_aq4_trace` に固定し、原因を過剰断定しない。
- source CPU traceは3 rowsを14.84秒、最大RSS約15.3GiBで生成し、既存source oracleのfinal hidden/logit/greedyと一致した。endpoint比較ではfinal norm差（max-abs 0.7083/0.9728/1.0846）が確認できるが、中間trace欠落のため分類は保留した。
- 専用fixture testsはdecoder layer mismatchの局所化と中間trace欠落のfail-closedを検証する。

## 次の行動

- 共有runtime/session/modelを変更せず、専用候補ビルドでAQ4 path traceを取得できる最小instrumentationを提案する。既定OFF、専用binary、固定3 rows、各stage 5座標、上限約32KiB/row、active service非変更とする。
- AQ4 intermediate trace取得後に同じanalyzerを再実行し、最初の不一致を量子化差、tensor mapping、runtime演算、LM headのいずれかへ限定する。GPU窓はこのtraceが必要になった時点で親へ報告する。

候補instrumentationは共有serviceではなく専用候補binaryだけに置く。対象は
`crates/ullm-engine/src/qwen35_aq4_model_runtime.rs` のdiagnostic-only visitor
（embedding bufferと各`Qwen35Aq4ResidentLayer::output_buffer()`を既存streamで
read-back）と専用`crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs`、および
その専用testだけとする。既定はOFF（専用binaryの明示flagが無い限りvisitorを呼ばない）、
active worker/serviceのargv・unit・manifestは変更しない。各rowは既存source replayの
case/context hashに束縛し、embedding + 32層 + final norm + LM headの各stageでhidden
5座標、logit 32座標、max-abs/L2だけを出す。host scratchはhidden 4096 f32（16KiB）を
再利用し、JSON payloadは1 row 32KiB以下、全traceは3 rowsだけに制限する。候補実行は
`CARGO_BUILD_JOBS=1 cargo build -p ullm-engine --bin ullm-aq4-differential-trace`後、
R9700排他lock・明示`HIP_VISIBLE_DEVICES=1`・active service停止済みの専用windowでのみ
行う。GPU窓まではこの候補をbuild/testするだけで、モデル実行はしない。

## 検証

- `python3 -m py_compile tools/trace-qwen35-aq4-differential.py`
- `pytest -q tests/test_qwen35_aq4_differential.py`（2 passed）
- CPU source trace生成（GPU/service変更なし）と既存source oracle endpoint一致を確認。
