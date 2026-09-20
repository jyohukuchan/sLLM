# Phase87 WU-D2: NVFP4遅延の持続範囲とR9700対照

2026-09-20完了。V620では、GQA型のKV読み出しを1回だけ行った後のNVFP4を連続32回まで増やしても、
遅延は解消しなかった。活性値量子化、軽いRMSNorm、1 MiB copyでもほぼ回復しない。
R9700では同じprobeによる遅延を再現しなかった。物理的なMALL／DRAM内訳は未特定である。

## 条件と測定

[WU-D1](phase87-wu-d1.md)のpayload、分離KV plane、context8256、4-copy weight pool、production NVFP4 launcherを
[新規probe](../../../../../native/hip/tests/phase87_wud2_persistence_probe.hip.cpp)で再利用した。実モデルの重み・activationではなく合成値である。
対象はV620 `gfx1030`／R9700 `gfx1201`、M=1、K5120/N17408（wide）とK17408/N5120（down）。

- predecessor: isolated、staged32（query-headごと6回）、GQA128（KV-headごと1回）。
- 持続範囲: predecessor1回の後にNVFP4をn=1,2,4,8,16,32回。同じstreamへ列をenqueueし、各dispatchと系列合計を測る。
- 介在処理: none、production BF16→NVFP4 activation quantizer、production RMSNorm（幅256、1行）、production copy（1 MiB）。
  介在条件はn=1で、isolated／staged／GQAすべてに同じ処理を挟むmatched controlを設ける。
- 各conditionで300 ms継続warmup、同一processの3 round×AB/BA×9 samples。
  decayは各GPU1944系列、介在処理は各GPU1296系列。
- eventは先行kernelをenqueueする前に作成する。weight slotは系列間も`sample*n`から連続巡回する。
- exact gfxをruntime照合。NVFP4は各shape37出力の独立FP32 oracleが0 ULP、全系列後の全出力finite／bitwise一致。
  KV reader checksumと3介在処理の独立oracleも成功した。

初回のdecayでは`sample`だけを開始slotに使ったため、n=2で前系列の最後の重みを直後に再利用していた。
この初回は`exploratory-overlap/`へ隔離し、最終decayには修正後の再計測だけを使用する。
介在処理はn=1なのでslot列が変わらず、元binaryのidentityを別に保存してその測定を維持した。

## 持続範囲

次は各AB/BA cell中央値を集約した**系列の最後のNVFP4**（µs/call）。系列内の全位置は結果JSONに保存した。

| n | V620 wide staged | V620 wide GQA | V620 down staged | V620 down GQA |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 124.162 | 135.462 | 123.722 | 134.802 |
| 2 | 124.002 | 135.442 | 123.442 | 134.963 |
| 32 | 124.122 | 135.623 | 122.983 | 135.562 |

n=4/8/16を含む全条件でも、V620は系列末尾で遅いままだった。**少なくとも32 dispatch持続**するという下限であり、
無限に続くことや、実モデルの168 dispatch全部をこれだけで説明できることは主張しない。
集計器は「その位置以後のmatched isolated比がすべて±2%以内」を回復の表示基準にしたが、これは採否gateではない。

R9700のn=32末尾は、wide staged102.881／GQA102.500、down staged103.881／GQA103.621 µs。
全nでGQAはほぼ同水準で、遅延非再現だった。さらにD1の5条件（isolated、staged/GQA、split32/128）を
そのままR9700でも測り、wide約103.5〜103.9、down約104.3〜104.5 µsで一致した。
既存の実モデルprofileで観測したR9700の約6%差を、この合成readerで説明できたことにはしない。

## 介在処理

V620のGQA／matched isolated比（n=1）。

| 介在処理 | wide | down |
| --- | ---: | ---: |
| なし | 1.1013 | 1.1010 |
| NVFP4 activation量子化 | 1.1003 | 1.1009 |
| RMSNorm、幅256 | 1.0970 | 1.0993 |
| 1 MiB copy | 1.1005 | 1.0860 |

いずれも回復しない。copyのdown側は小さくなるが、約8.6%が残る。処理時間を追加する緩和策として採用していない。
R9700では全介在条件が概ね0.997〜1.003であり、ここでも遅延を再現しない。

## gfx1201のcounterと限界

`rocprof-compute 3.7.0`はgfx1201を解析対象に持たず、ROCm 7.14のgfx1201 catalogにもMALL／DRAM destination／credit-stallの
直接counterはない。gfx115x向けのDRAM counterを代用していない。

`rocprofv3`でtrace／fetch／cacheの3 passを各160 NVFP4 dispatchずつ取得した。
traceではwide isolated96.343／staged96.428／GQA95.973、down97.240／97.135／96.858 µsで差を再現しなかった。
一方FETCH_SIZE、L2CacheHit、MemUnitBusyは全条件で0を返した。非zeroの重み読み出しと整合しないため、
**counterは使用不能として扱い、転送量やhit率が0だったとは解釈しない**。MALL／DRAMの内訳は未取得のままである。

## 完了判定と成果物

持続範囲、介在処理、gfx1201の再現有無を数値で記録し、WU-D2を完了する。
この結果からWU-D3では、軽い処理を追加するのではなく、V620の元のKV読み出し方式を変える3候補を調べる。
D2中のproduction source変更はなく、採用済み最適化とGPU性能設定を変更していない。

- [結果・identity・全位置の曲線](phase87-wu-d2-results.json)
- [集計器](../../../../../ci/tools/phase87_wud2_summarize.py)は今回の両shape／全predecessor matrix用。
- rawログ、各版binary、build script、counter、失敗・修正前記録: `.local-artifacts/phase87/wu-d2/`。
- GPU実行は全成功、Python static／compile、diff whitespace、1回の統合レビューで測定境界と限界を確認。
  WU-D3で使う一時production変更とは別の作業単位として記録した。

計画: [Phase87 WU-D2](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md) ／ 後続: [WU-D3](phase87-wu-d3.md)
