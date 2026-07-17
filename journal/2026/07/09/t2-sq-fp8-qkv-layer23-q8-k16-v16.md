# T2 SQ FP8 QKV layer23 q8/k16/v16

## 前回の要点

- layer19 `q8/k16/v16` はprompt bundleとB=1/4/8 short guardでstrict top1を維持した。
- 次の境界として、同じQKV patternをlayer23へ広げられるか確認する段階だった。

## 今回の変更点

- layer23 full QKV候補 `sq-fp8-w8a16-r9700-v0-qkv-layers3-7-11-15-19-23-q8-k16-v16` を作成し、R9700 prompt bundleへ流した。
- full QKVは`case_a`でAQ4 top1 `4105` からSQ top1 `5582` に反転し、strict top1 `2 / 3` で失敗した。
- split probeでは、layer23 `q8` と `v16` が単体でも`case_a`を崩し、layer23 `k16` だけがstrict top1 `3 / 3` を維持した。
- layer23 `k16` はB=1/4/8 short guardでもtop1一致を維持した。

## 結果

| row | strict top1 | prefill tok/s | decode tok/s | final top1 |
| --- | ---: | ---: | ---: | --- |
| layer23 full `q8/k16/v16` | 2 / 3 | 60.821302 | 72.084189 | `24218,5582,329` |
| layer23 `q8` split | 2 / 3 | 59.189428 | 72.379743 | `24218,5582,329` |
| layer23 `k16` split | 3 / 3 | 59.818278 | 72.952135 | `24218,4105,329` |
| layer23 `v16` split | 2 / 3 | 59.042268 | 72.719037 | `24218,5582,329` |

Short batch guard for layer23 `k16`:

| B | SQ prefill tok/s | SQ decode tok/s | SQ end-to-end tok/s | top1 match |
| ---: | ---: | ---: | ---: | --- |
| 1 | 17.649348 | 70.880232 | 8.779914 | true |
| 4 | 48.840512 | 73.523424 | 24.964220 | true |
| 8 | 60.051932 | 73.442133 | 34.338277 | true |

## 判断

- layer23 full QKVはpromoteしない。
- layer23 `q_proj` と `v_proj` は現strict top1 policyではfallbackに残す。
- layer23 `k16` はcurrent diagnostic extensionとして保持するが、`case_a` marginが薄いためfull SQ policyではない。
- 次はlayer27 k-only extension、layer23 q/vのscale強化、または広いprompt/text guardのどれを優先するか選ぶ。

## 次の行動

1. layer23 `k16` policyとshort guard結果を保存してcommitする。
2. layer23 `q8` / `v16` はfailure guardとして残す。
3. 次のT2候補は、品質重視ならtext-level guard、coverage重視ならlayer27 k-only、format探索重視ならlayer23 q/v scale/layout再探索に分岐する。
