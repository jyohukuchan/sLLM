# AQ4 capture ROCm 7.2 memory parser

## 前回の要点

- actual-v12 は full-model の warmup 2回と measured 10回を完走したが、capture assembly が方向不明の `hipMemcpyAsync` を拒否して failure evidence だけを封印した。
- source trace の全行監査では、kernel 12,263行は全familyを分類できた。memory traceはROCm 7.2の `Kind` + `Direction` schemaで10,845行あり、H2D 6,438行、D2H 4,407行だった。
- generic HIP APIをdirectional APIへ変更した次のFAIL境界は、既存memory parserが `Kind` と `Direction` を競合するoperation aliasとして拒否する箇所だった。

## 今回の変更点

- legacy schemaは `Name`、`Kind`、`Direction`、`Operation`、`name` のいずれか単一operation列だけを従来どおり受理する。
- ROCm 7.2 schemaはoperation aliasがexactly `Kind` + `Direction` の組である場合だけ専用分岐へ入れる。
- 専用分岐では全行の `Kind` が空白を含まないexact `MEMORY_COPY` であることを要求し、operationは `Direction` だけから分類する。
- `MEMORY_COPY_` prefixを除いたH2D、D2H、D2D、H2H、peer表現だけを既存allowlistで受理する。
- wrong/empty/unknown Kind/Direction、duplicate correlation、3 operation aliasesをfail-closedで拒否するテストを追加した。
- actual-v12 memory traceのexact header、10,845行、H2D 6,438行、D2H 4,407行を読み戻す回帰テストを追加した。
- capture artifact schema `ullm.aq4_p3_diagnostic_rocprof_capture.v1` とfailure schema v2は変更していない。変更は入力trace schemaの厳格な追加対応だけである。
- `python3 -m py_compile tools/capture-aq4-p3-diagnostic-profile.py`、capture tests 79 passed / 1 environment-dependent skipped、`git diff --check` が成功した。
- runtime、bundle、launcher、GPU、service、actualは変更または実行していない。

## 次の行動

- runtime側でgeneric `hipMemcpyAsync` をdirectional HtoD/DtoH/DtoD APIへ分け、worker/package/bundle authorityを更新する。
- capture source hash、maintenance QA manifest、ready/operator authorityをrecascadeする。
- sealed v9 traceは再利用せず、fresh runtime-v10、execute-evidence-v10、capture-v10と、それらに対応するfresh maintenance/operator result/audit namespaceで一度だけ再採取する。
