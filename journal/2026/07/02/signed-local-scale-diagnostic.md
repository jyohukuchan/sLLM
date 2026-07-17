# signed local-scale診断

## 前回の要点

- 直前のlocal-scale形式比較では、aq4 g16でE4M3の平均BF16-errorが`0.005071093`、UE5M3が`0.005092034`だった。
- UE5M3は符号無しE5M3として扱っており、数学的にはE4M3より広い正値scaleを表現できるため、単純にはE4M3以上になることを期待していた。
- ただしcodebookは+nと-nが1対1になるように制約していないため、signed local-scaleが使える場合はcodebook非対称性と結びついてBF16-errorへ影響する可能性がある。

## 今回の変更点

同じ36 tensor sampleで、E4M3とUE5M3についてunsigned local-scaleとsigned local-scaleを直接比較した。

- 出力: `benchmarks/results/2026-07-02/aq/2026-07-02-aq-signed-local-scale-diagnostic-e4m3-ue5m3-ud36.json`
- 条件: aq4、block-size 16、8 iteration、global-scale FP16、最大262144要素/tensor
- signed local-scaleはblockごとにlocal-scaleの正負を探索し、codebookとglobal-scaleも再最適化した。

主要結果:

| local-scale形式 | unsigned平均BF16-error | signed平均BF16-error | signed/unsigned | 負local-scale block率 |
| --- | ---: | ---: | ---: | ---: |
| E4M3 | `0.005072593` | `0.004469881` | `0.881182` | `0.499225` |
| UE5M3 | `0.005086941` | `0.004492363` | `0.883120` | `0.498506` |

E4M3とUE5M3の比較:

- unsignedではUE5M3/E4M3が`1.002835`。
- signedではUE5M3/E4M3が`1.005071`。
- signed条件でもUE5M3がE4M3を平均では上回らなかった。
- signed条件でUE5M3が良かったtensorは36件中10件、E4M3が良かったtensorは26件だった。

codebook対称性:

- 初期codebookの対称性誤差平均はE4M3で`0.001019`、UE5M3で`0.000970`だった。
- unsigned最適化後はE4M3で`0.001616`、UE5M3で`0.001760`だった。
- signed最適化後はE4M3で`0.016589`、UE5M3で`0.016050`まで増えた。
- signed local-scaleを許すと、optimizerはcodebookをより非対称にし、block単位の符号反転で補う方向へ進む。

## 解釈

今回の結果から、signed local-scaleはBF16-errorを約12%下げる余地を持つ。ただし、これはscale formatの優劣というより、blockごとの符号bitを追加して表現力を増やした効果である。

一方で、前回のUE5M3<E4M3という結果は「E4M3だけがscaleの符号を使えていた」ためではない。現行実装のE4M3/UE5M3/E5M2/E8M0 local-scaleはすべて正値のみであり、signed条件にしてもUE5M3/E4M3の相対関係は改善しなかった。

負local-scale block率が約50%になることと、signed最適化後のcodebook非対称性が増えることから、signed local-scaleはcodebook非対称性と強く結びつく。これは精度面では有利だが、仕様として採用するとlocal-scaleの符号bitが追加の表現チャネルになり、raw-value/codebook-indexの意味が不安定になる。

## 次の行動

- local-scale、tensor-scale、family-scaleは原則としてunsignedに固定する。
- signed local-scaleは本仕様ではなく、codebook非対称性の影響を見る診断条件として扱う。
- signedで得られた改善分は、scaleへ符号bitを入れるのではなく、codebookの対称性制約、family別codebook、またはblock統計に応じたcodebook決定アルゴリズムで回収できるか調べる。
- UE5M3がE4M3を必ず上回らなかった点は、E4M3 warm-start、best-so-far保持、scale候補探索幅の調整で再確認する。
