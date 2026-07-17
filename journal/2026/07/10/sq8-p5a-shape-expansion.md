# SQ8 P5-A shape expansion

日付: 2026-07-10

## 前回の要点

P4ではlayer 0のq projection `[5120,5120]`について、source-correctな動的activation量子化、CK ABScale、warm/256 MiB退避測定、native FP8 WMMAを確認し、P5 entry gateをgreenとした。

## 今回の変更点

- layer 0のo/k/v/gate/up/downについて、source checkpointからbyte-exactなone-tensor canonical artifactを作成した。
- q/o/k/v/gate/up/downの7 tensorそれぞれでM=`1,2,4,8,16,32,128`のfixture、source-correct HIP reference、CK warm、CK cache退避ありを逐次実行した。
- 合計49 fixture、49 reference、98 optimized resultを検証した。全optimized resultで38候補中6候補が対応し、6候補すべてが数値gateに合格した。activation FP8 byteとscale bitは完全一致し、NaN/Infとfallbackは0だった。
- 全98 optimized resultの最大relative L2は0.00169415、最小cosineは0.999998566だった。
- M=8/M=2の集約スループット比は全tensorで3.851倍から3.936倍、256 MiB退避ありでも3.884倍から3.952倍だった。
- M=8のwarm optimizedはreferenceより25.0倍から35.6倍速かった。全Mを含む最小speedupも3.25倍で、退避ありoptimizedとwarm referenceの保守的比較でも最小2.66倍だった。
- k/v、gate/up、downの三つの新geometryをrocprofv3で記録し、cache eviction、activation quantization、CK OCP FP8 GEMMの順序を確認した。
- `tools/summarize-sq8-shape-expansion.py`を追加し、196個のfixture/result JSONをfail closedで再検証して、完全なCK `GetTypeString()`を持つ`dispatch-table.json`を生成した。未測定のshapeまたはMはoptimized dispatchを拒否する契約とした。

正式結果は`uLLM-project/benchmarks/results/2026-07-10/sq8-shape-expansion-v0.1/`に保存した。P5-A shape gateはgreenである。

関連commitは、validatorが`a9b586c`、正式shape evidenceが`f2b9f47`である。

P5全体はまだ完了していない。現在のCK経路はstandalone componentであり、production runtime API、shared activation quantization、BF16からF32への接続、一層runnerは未実装である。

## 次の行動

P5-Bとして、明示的なROCm/gfx1201 featureの背後にCK projection primitiveを実装する。activation量子化とGEMMを分離し、QKVは量子化を一回、gate/upも一回だけ実行する。`dispatch-table.json`に存在しないshape/MやCK instanceの不一致はfallbackせず失敗させ、選択implementation IDをtyped resultで返す。続くP5-Cでは、既存CLIのresident layer順序を再利用した狭い一層runnerを作り、P5-Dで独立oracleとの中間・最終tensor比較を行う。
