# T1 full package logical batch grid

## 前回の要点

- T1はpackage-backed component real-batch rowとflattened component batch gridまで進んでいた。
- full packageのprefill/decode/end-to-end total throughput rowはまだ不足していた。
- SQ評価ではlogical batchとreal batchを混ぜない必要がある。

## 今回の変更点

- AQ4 full-package logical batch small grid workloadを追加した。
- R9700で `batch=1/4/8`、`prompt_tokens=4`、`generated_tokens=2` を実測した。
- 3行すべて `status=ok`、`verified_all=true` だった。
- JSONLにはprefill/decode/end-to-end total throughput、KV cache bytes、VRAM consumedが保存された。
- 全行 `batching.mode=logical` であり、real request-batch性能ではない。

## 結果

| batch | prefill total tok/s | decode generated tok/s | end-to-end tok/s | KV cache bytes | VRAM consumed bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.574674811 | 68.797769301 | 2.172326744 | 393216 | 4279500800 |
| 4 | 58.744142319 | 69.070493674 | 2.433498426 | 1572864 | 4206096384 |
| 8 | 67.008573726 | 69.071288010 | 2.533759132 | 3145728 | 4279500800 |

## 次の行動

1. logical full-package gridはschema/control-plane guardとして扱う。
2. 次はfull package pathへreal request-batch prefill/decode executorを接続する。
3. 同じ `batch=1/4/8` gridで `batching.mode=real` の行を保存する。
