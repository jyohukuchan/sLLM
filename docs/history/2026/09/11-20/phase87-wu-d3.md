# Phase87 WU-D3: KV読み出し方式によるNVFP4罰則の緩和

2026-09-20完了。3候補を比較したが、正味の採用基準に届く候補はなく、**すべて不採用**とした。
本番のWU1／WU1.1／WU2は維持し、試験用に入れたblock配置変更も復元した。

## 候補と条件

[WU-D2](phase87-wu-d2.md)でV620の遅延が32 dispatchまで持続し、軽い介在処理で解消しないことを確認したため、
元のKV読み出し方式を変える次の3候補をV620で評価した。

- C1: GQA stage1のdecoded K/V tileを8→32 tokenへ拡大（LDS 64 KiB）。tile64は128 KiBとなるため事前に除外。
- C2: blockの割当順を`query,kv_head,split`から`query,split,kv_head`へ変更。各block内の計算とworkspaceの論理indexは維持。
- C3: tile8を二重buffer化し、次tileを先読み。keyの処理順・softmax更新順は維持。

[probe](../../../../../native/hip/tests/phase87_wud3_attention_neighbor_probe.hip.cpp)は実attention stage1／stage2と
production NVFP4を同じstreamで接続し、各時間と合計を分離する。
KVはMXFP8 E4、context8256、M=1、wide/downのNVFP4をそれぞれ16回、300 ms継続warmup、同一processで3 round×AB/BA×9 samples。
weightは4組を巡回する合成値であり、attention出力をNVFP4 activationとして渡す実モデルのdata flowは再現していない。

**この実attention harnessのcontrol後NVFP4は、D1/D2の読み出しだけのGQA probeの約135µsを再現しない。**
従ってcandidate後が約122µsであることを罰則解消の証拠にせず、実attention controlとの候補比較と実モデル測定を別々に扱う。

## 最終単体結果

単位ms。NV列は16 dispatchの合計。wide/downごとに対応するcontrolと比較する。

| 候補 | shape | control attention | 候補attention | control NV列 | 候補NV列 |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 tile32 | wide | 0.311286 | 0.739173 | 1.958155 | 1.962196 |
| C1 tile32 | down | 0.311166 | 0.736974 | 1.914315 | 1.918275 |
| C2 block配置 | wide | 0.311326 | 0.312445 | 1.959756 | 1.959676 |
| C2 block配置 | down | 0.312005 | 0.312566 | 2.050958 | 2.058158 |
| C3 先読み | wide | 0.311486 | 0.493889 | 1.958276 | 1.959315 |
| C3 先読み | down | 0.311525 | 0.493289 | 1.913195 | 1.916035 |

C1はattentionが約2.4倍、C3は約1.6倍へ退行し、NV側の利益がない。C2はほぼ中立だった。
attention16回、wide112回、down56回/tokenで単純換算した正味短縮の**推定**はC1 −6.872、C2 −0.038、C3 −2.931 ms/token。
この換算は実モデルの実測ではなく、harnessの局所時間差に基づく目安である。
C1/C3はattentionの追加費用だけでNVFP4罰則の想定上限1.9 ms/tokenを超え、打ち切り線0.95 ms/tokenに達しない。
C2も正味TPOT 1%（今回のV620 baselineでは0.631 ms/token）という採用基準に届かない。

## 実モデル対照

上のharnessの限界があるため、最も中立だったC2をproduction GQA stage1のshort/long両方へ一時適用し、
V620で通常8192/128、1 warmup＋3 measured、MTPなし／ありを実行した。stage2、累積順、KV配置、samplingは変更していない。
別のfrozen binaryへ保存した後にsourceを戻し、そのbinaryだけで測った。

| GPU／構成 | baseline decode tok/s | C2 decode tok/s | 変化 | token列 |
| --- | ---: | ---: | ---: | --- |
| V620・MTPなし | 15.83907 | 15.89356 | +0.344% | 一致 |
| V620・MTPあり | 28.96129 | 28.93425 | −0.093% | 一致 |
| R9700・MTPなし | 21.37186 | — | provider変更なし | baseline取得 |
| R9700・MTPあり | 34.05221 | — | provider変更なし | baseline取得 |

R9700の本番はregular-wave attentionで、今回のV620 GQA候補を適用していない。
probeでも`control-fallback`と明記して現行controlを確認し、R9700でcandidateを実装した／改善したとは呼ばない。
両GPUのbaselineとV620のcandidateはすべて正常終了、HIP実行、finite、cleanup zero。

baselineはWU2の固定binary、candidateは現行sourceへC2だけを入れて作った別binaryである。
両者のidentityとsource mappingを保存し、完全に同じbuildと表記しない。
モデル出力は両MTP条件で一致したが、性能差は小さく、単体の正味改善基準も満たさないため採用しない。

## 数値・境界・後片付け

- 3候補・両NV shapeの最終単体runは、attentionの独立FP32 oracle、NVFP4の37出力oracle、finite、repeat、
  output/workspace guard、candidate/controlのbitwise比較を成功した。数値はこの検査範囲でN0。
- C2の境界は両GPUで各22条件: context1024/1025/8191/8192/8193/8256 × M1/2/3、
  さらに1025/8193・M3のcausal-tail／強いQ/K pattern。両NV shape、全出力bitwise一致とguardを確認した。
  1024未満は既存の別providerの範囲であり、このprobeの対象に含めない。
- R9700のcontrol-only performanceはAB/BAの両armがcontrolとなるため108 samplesだった。
  集計器の54 samples固定という初回検査を修正し、同じraw結果を再解釈した。GPU再実行で置き換えていない。
- 本番sourceを作業開始時とbyte一致へ戻し、通常build cacheもcontrol sourceで再buildした。
  candidate binaryと試験用probeは再現用のGit管理外artifact／testにだけ保持する。採用による数値台帳の変更はない。
- Python static／compile、Markdown link、diff whitespace、1回の統合レビューを実施。
  新しい環境変数、production fallback、必須gate、後続最適化の自動開始は追加していない。

## 完了判定

WU-D2を受けて3候補を実装・数値検査し、attention＋NVFP4の正味費用と両GPUの通常モデル条件を記録した。
どれも打ち切り線・採用基準に届かないため、計画の規則どおり探索を終了する。
D1/D2で分かった現象自体を修正済みとは主張しない。productionは現行のまま維持する。

- [結果・identity](phase87-wu-d3-results.json)
- [候補実装](../../../../../native/hip/tests/phase87_wud3_candidates.hpp)、[実行・検査](../../../../../ci/tools/run_phase87_wud3.py)
- rawログ、各版probe/model binary、一時patch、復元前source: `.local-artifacts/phase87/wu-d3/`。

計画: [Phase87 WU-D3](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md) ／ 前提: [WU-D2](phase87-wu-d2.md)
