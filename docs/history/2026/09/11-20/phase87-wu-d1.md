# Phase87 WU-D1: NVFP4 decodeの直前kernel依存

2026-09-20完了。**KV読み出しだけの先行kernelでも、GQA型の後にNVFP4が約9〜11%遅くなる現象を再現した。**
同じ単体probe内では、NVFP4単独と旧staged型の後はほぼ同じ速度だった。旧stagedが単独時より大幅に速くしているのではなく、
GQA型の先行処理が次のNVFP4を遅くする方向である。ただし、MALL／DRAM page・bank等の物理機構までは特定できていない。
採用済みWU1／WU1.1／WU2とproduction sourceは変更していない。

## 条件と実装

[WU-D1計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu-d1-nvfp4-m1-decodeの近傍依存の原因調査段階1のnvfp4最適化より前)の3手順を実行した。
基準は[WU2後の8192/128・MTPなしprofile](phase87-idle-recheck.md)。排除済み原因の再検証は行っていない。

- V620 `gfx1030`、UUID `GPU-76a08c022586fed6`、ROCm 7.14.0。Qwen補助サービス停止、GPU単独使用、performance levelは変更なし。
- [新規probe](../../../../../native/hip/tests/phase87_wud1_nvfp4_neighbor_probe.hip.cpp)から、WU2時とhash一致する既存lowp archiveの
  production `launch_nvfp4_w4a4`／ID84を呼び出す。kernel本体・selector・起動設定を複製・変更しない。
- M=1、K5120/N17408とK17408/N5120。packed E2M1、別planeのblock16 E4M3 scale、FP32 tensor scaleという本番配置。
  重みと活性値の内容は合成データ。各shapeのweight＋scaleは50,135,040 byte、4組を巡回（200,540,160 byte）。
- 先行kernelは分離したK／V／scale planeを読む。4 KV head、6 query head/KV head、head dim256、context8256
  （基準decode区間中の長さ、split/tile末尾の非整列を含む）、KV実payload17,436,672 byte。
- staged型はquery-head単位のwave32読み出しを6回、GQA型はKV-head単位・192 thread・8-token tileの共有読み出しを1回。
  両方にsplit32/128を設けた。読み出しを消されないようchecksumだけを出力し、attention計算やKVへの書き戻しはしない。
- 全conditionで300 ms継続warmup（32 dispatchずつenqueue）。同一processで3 roundのAB／BA、各9 sample。
  計540 sample。先行処理後のHIP event間でNVFP4だけを測る。
- 2 shapeそれぞれ37出力を独立FP32復号・累積oracleで照合し、最大0 ULP。
  全出力finite、全conditionでisolated出力とbitwise一致、4 readerのchecksumも各shapeで一致した。

## 単体時間

各AB／BA cell中央値をさらに集約した値（µs/call）。通常のevent測定値であり、実モデルのprofile絶対値と混ぜない。

| 先行処理 | K5120/N17408 | K17408/N5120 |
| --- | ---: | ---: |
| なし | 123.501 | 123.481 |
| staged型・split32 | 124.501 | 123.462 |
| staged型・split128 | 123.921 | 123.921 |
| GQA型・split32 | 136.122 | 134.602 |
| GQA型・split128 | 136.262 | 134.962 |

staged32を対照にしたpaired比較は、GQA32がwide **+9.42%**／down **+8.99%**、
GQA128がwide **+9.57%**／down **+9.47%**。各比較とも3 round×AB/BAの6組すべてで遅い。
split数だけでは分かれず、staged/GQAという読み出し実行方式で分かれる。

別のkernel-trace計測でも、wideは単独113.111／staged32 113.216／GQA128 124.771 µs、
downは112.506／112.601／124.804 µsとなった（各16 dispatchの平均）。
HIP eventとの絶対値差には計測境界・profilerの観測影響があり、同一時間系内の方向と相対差を証拠とする。

## カウンタと特定限界

`rocprofv3 1.3.2`のselected regionでwarmupを除外し、production NVFP4 symbolだけを選択した。
trace、fetch、cache、instruction、wait、EAの6成功passで各160 NVFP4 dispatchを確認。
前後のconditionはstdoutの実行順とtimestamp順のdispatchを対応付け、shapeのgridも照合した。

以下はstaged32 → GQA128、各16 dispatchの平均。異なるcounter pass間の絶対値を足し合わせない。

| 指標 | wide | down |
| --- | --- | --- |
| GL2C/EA fetch（KiB） | 58,808.80 → 58,968.49（+0.27%） | 57,401.91 → 57,879.28（+0.83%） |
| L2 hit率 | 15.329% → 15.125% | 15.540% → 14.924% |
| GPU active cycle | 301,628.75 → 329,793.19（+9.34%） | 302,005.56 → 334,952.94（+10.91%） |
| EA busy cycle | 267,711 → 297,519（+11.13%） | 267,823.63 → 297,691.44（+11.15%） |
| EA busy / GRBM_COUNT | 89.047% → 90.035% | 89.211% → 90.191% |
| WAVE_DEP_WAIT | 70.098% → 68.494% | 88.929% → 88.098% |
| SQ_WAVES | 4,352 → 4,352 | 1,280 → 1,280 |

- 読み出しbyte数の増加は時間増加より小さく、L2 hit率も近い。EAがbusyなcycleは時間とともに増えている。
  これはEA側の活動が長引く観測であって、EA latencyやDRAM stall時間の直接測定ではない。
- WAVE_DEP_WAITは広いwave待ちの割合であり、今回その割合は増えていない。DRAM待ちだけが増えたと断定しない。
- SQ_INSTS_VALU／SQ_INSTS_SMEM／SQ_INST_CYCLES_VMEMは、固定のNVFP4処理にもかかわらずcondition間で大きく変動した。
  先行処理・counter window等の影響を排除できず、命令数差やVMEM stallの原因証拠には使用しない。raw値は保存した。
- `rocprof-compute 3.7.0`はgfx1030を解析対象に持たない。SDK catalogにはgfx1030のMALL、DRAM destination、
  DRAM credit-stall counterがなく、**MALL／DRAMの内訳は未取得**。別GPUのTCC counterを流用しない。
- EAの最初のpassは3 GRBM counter同時指定でhardware capacity error38／SIGABRTとなった。
  その試行は失敗として保存し、EA_BUSY＋GRBM_COUNTの2 counterへ分割した再実行だけを成功扱いにした。

この実験は「attentionの数値計算がなくても、先行するKV読み出し実行方式が十分条件になる」ことまで示す。
読み出し回数・順序・block/workgroup配置・実行時間はfamily間で同時に変わるため、それらの単一要因への分解や、
page/bank状態とMALL内容のどちらなのかという物理的帰属は残る。

## 実モデルtraceの追加局在

保存済みの旧staged32／現行traceをtimestamp順へ直し、最後の127 token×1171 dispatchを境界・grid付きで照合した。
各tokenのNVFP4は168 dispatch（56層×3行列）。遅延はfull-attention層だけでなく他の層にも及ぶ。
層を4つのmodulo群へ分けても、gate/up/downの全12群で現行側が遅く、差は6.80〜19.59 µs/callだった。
これは局在の観測であり、独立した因果介入ではない。単体probeの結果と合わせ、直後の1 kernelだけに限定しない。

## 完了判定と後続

- 手順1: attentionなしの両shapeを計測。probe内ではisolatedとstaged型が同程度で、GQA型が遅い。
- 手順2: KV readerのみで差を再現。split32/128・両shape・全AB/BAにおいて方向一致。
- 手順3: 利用可能なcounterを取得し、取得不能なEA/MALL/DRAM stall内訳と、解釈不能なSQ raw値を区別した。
- 受入条件の「特定できない場合はどこまで絞れたかを証拠付きで記録」に従い、この原因調査を完了する。
  性能修正は行っていない。後続のNVFP4最適化ではisolated値だけでなく、実モデルのGQA先行条件を対照として扱う。
  新たな性能修正の作業単位や必須gateは追加せず、次の作業はユーザー判断へ残す。

## 成果物・検証

- [集約結果・identity](phase87-wu-d1-results.json)。rawログ、build、counter CSVは`.local-artifacts/phase87/wu-d1/`。
- [単体集計](../../../../../ci/tools/phase87_wud1_summarize.py)、[profile収集・集計](../../../../../ci/tools/phase87_wud1_profile.py)、
  [既存trace局在](../../../../../ci/tools/phase87_wud1_existing_trace.py)。build commandはraw保存先の`build.sh`、完全hashは集約JSON。
- gfx1030の最終build・GPU実行が成功。Python static、Markdown local link、diff whitespaceを確認。
  integration reviewを1回行い、読出しfamilyの複数要因と測定境界の限界を反映した。
- production source全ファイルの開始前／終了後hash一致、終了後GPU使用率0・VRAMは開始時水準へ復帰。
  外部コードの新規copy/adapt/port、commit、pushは行っていない。

計画: [Phase87 WU-D1](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu-d1-nvfp4-m1-decodeの近傍依存の原因調査段階1のnvfp4最適化より前)
