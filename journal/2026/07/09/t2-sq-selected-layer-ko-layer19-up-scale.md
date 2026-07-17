# T2 SQ selected-layer k/o layer19 up scale

## 前回の要点

- 20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` がcurrent passing branchだった。
- 次はlayer19 `up_proj` row-block32、失敗時row-block16 recoveryだった。

## 今回の変更点

- layer19 `up_proj` row-block32/16 policyとartifactを作成した。
- R9700 six-layer token-id model-loop prompt bundleを2候補で実行した。
- どちらもlen4でSQ top1が `102446` になり、AQ4 top1 `110784` はSQ top8の2位だった。

## 次の行動

- layer19 `up_proj` row-block32/16はfailure guardとして残す。
- current passing branchは20 tensor版のまま維持する。
- 次はT1 full-package real request-batch throughput runnerへ戻るか、T2をselected layer外へ広げる。
