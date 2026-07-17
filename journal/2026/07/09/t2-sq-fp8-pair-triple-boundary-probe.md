# T2 SQ FP8 pair/triple boundary probe

## 前回の要点

- SQ FP8 direct pathは単発、batch、pair、triple matvec API境界まで実装済み。
- full mixed strict-top1保守候補はlayer3 `k_proj` 1 tensorだけで、pair/triple境界は実ベンチで未確認だった。

## 今回の変更点

- `q/k` layer3 policyと `q/k/v` layer3 policyを追加した。
- `q/k` はtriple無効化 + pair required envでB=1/4/8を測った。
- `q/k/v` はtriple required envでB=1/4/8を測った。
- 両候補ともAQ4 final top1と一致した。

## 結果

| boundary | B=1 e2e tok/s | B=4 e2e tok/s | B=8 e2e tok/s | top1 |
| --- | ---: | ---: | ---: | --- |
| q/k pair | 9.324354 | 15.277466 | 34.187742 | all match |
| q/k/v triple | 8.414481 | 25.402500 | 35.980289 | all match |

## 次の行動

- `q/k/v` layer3を最小triple boundary候補として固定し、prompt bundleまたは長めのprefill gridへ広げる。
- stdout/JSONLにpair/triple境界の実行modeを明示する。
