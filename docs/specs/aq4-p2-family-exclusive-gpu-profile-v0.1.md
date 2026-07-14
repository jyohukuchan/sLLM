# AQ4 P2 family-exclusive GPU profile v0.1

## 前回の要点

P2 resident one-case smokeは、R9700、requested/resolved M=128、2 warmup + 10 measured transactionsをhash-bound inputとして準備できる。一方、kernel別時間の単純合算は、複数streamや同時実行kernelの重複時間を二重計上するため、GPU総時間またはfamily attributionとして使用できない。

## 今回の変更点

`profile-aq4-p2-family-exclusive.py`は、resident one-case command全体を`rocprofv3`の1 subprocessで1回だけprofileする。2026-07-14のWRX80には`/opt/rocm/bin/rocprofv3` 1.1.0、ROCm 7.2.1があり、`--kernel-trace --output-format csv --output-directory DIR --output-file NAME -- COMMAND`を使用できる。parserはrocprofv3の`Dispatch_Id`、`Kernel_Name`、`Start_Timestamp`、`End_Timestamp`と、legacy互換の`Index`、`KernelName`、`BeginNs`、`EndNs`を受理する。timestampの単位はnanosecondsである。

artifactは次をhashで束縛する。

- one-case bindingとcase self-hash、case ID、requested/resolved M=128
- bound v2 identity、model/revision、worker、served manifest、guard set
- exact R9700/gfx1201 runtime device index 1
- detached resident binary、package manifest/content、threshold policy
- profiler path/version/ROCm version/version出力hash、実行command、kernel trace
- kernel-family mapping schema/hashとunknown kernel一覧

kernel familyは次の保守的な名前規則で分類する。複数familyへ一致する名前は拒否し、一致しない名前はunknownとする。

- `paged_validation`: paged KV write、Q/K norm+RoPE、Q split、paged cache/block validation
- `aq4_projection`: AQ4 matvec、GEMM、projection、register BM8
- `attention`: paged decode attention、paged causal GQA、attention read/split
- `recurrent`: linear attention、gated delta、recurrent、QKV prepare
- `normalization`: RMSNorm、SiLU/sigmoid multiply、standalone RoPE/add
- `head`: LM head、top1、argmax

GPU総時間は全kernel intervalのunionである。inclusive kernel durationの単純合算は診断値だけに残し、GPU総時間には使わない。sweep-line集計では、unknownがactiveな区間を`unclassified`、二つ以上の既知familyがactiveな区間を`cross_family_overlap`、一つだけの既知familyがactiveな区間をそのfamilyの`exclusive`へ一度だけ帰属する。同一family内で複数kernelが重なる区間は`exclusive`へ一度だけ帰属し、単一kernelだけの部分を`non_overlap`として別記する。これらのpartitionはGPU total unionとexactに一致しなければならない。

traceにexactな`Phase=prefill|decode`がある場合はphase別にも同じ集計を行う。通常のrocprofv3 kernel CSVだけではphase境界を証明できないため、phase列がないtraceは`unclassified_phase`へ分離し、将来のROCTxまたは同等のclock-bound phase markerが揃うまでmeasurement対象にしない。

unknownは設定したGPU union比率を超えるとfail-closedとする。default thresholdは0で、新規kernel名を推測で既存familyへ入れない。

artifactは常に`measurement_eligible=false`、`promotion=false`である。family profileは2 warmup + 10 measured performance aggregationとは分離し、inclusive kernel sum、profile overheadを含むrun、phase未証明traceをthroughput/latency値へ転用しない。

## 次の行動

GPU実行前にsynthetic intervalでunion、exclusive、same-family/cross-family overlap、unclassified、prefill/decode分離を検証する。sanctioned runではresident one-case commandを1回だけrocprofv3で包み、unknown kernelを確認してmapping specを明示更新する。phase markerが得られないrunはdiagnosticのまま保持する。
