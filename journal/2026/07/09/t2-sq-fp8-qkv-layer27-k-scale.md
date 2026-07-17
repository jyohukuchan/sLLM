# T2 SQ FP8 QKV layer27 k scale

## 前回の要点

- layer23 `k16` はprompt bundleとB=1/4/8 short batch guardでstrict top1を維持した。
- layer23 `q8` と `v16` は単体でも`case_a`を崩すためfallback扱いにした。

## 今回の変更点

- layer23 `k16` 通過boundaryにlayer27 `k16` を追加してprompt bundleを測った。
- layer27 `k16` が`case_a`で失敗したため、layer27だけrow-block8にした `k8` recoveryも測った。
- `k16` と `k8` の両方で、final top1は `24218,5582,329` になり、AQ4 baseline `24218,4105,329` と一致しなかった。

## 結果

| row | strict top1 | prefill tok/s | decode tok/s | final top1 |
| --- | ---: | ---: | ---: | --- |
| layer27 `k16` | 2 / 3 | 59.984624 | 72.659910 | `24218,5582,329` |
| layer27 `k8` | 2 / 3 | 60.600922 | 72.557192 | `24218,5582,329` |

## 判断

- layer27 `k_proj` はrow-block16でもrow-block8でも`case_a`を崩す。
- row-block8化だけでは回復しないため、現branchのcoverage拡大はlayer23 `k16` までで止める。
- 次はtext-level guard、layer23 q/vとlayer27 kの別format/layout、またはSQ評価基盤側へ戻る。

## 次の行動

1. layer27 `k16` / `k8` をfailure guardとして保存する。
2. layer31 k-onlyへ単純に進む前に、quality guardまたはformat方針を見直す。
3. commit対象はuLLM-project内の結果・policy・計画書に限定し、journalは作業記録として残す。
