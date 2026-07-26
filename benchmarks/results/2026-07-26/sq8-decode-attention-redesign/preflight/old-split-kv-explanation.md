# 旧 tile-128 の 1.2365x を説明する既存計測の再計算

この文書は新カーネルの性能主張ではない。新しい隔離窓の前に、依頼 BH が
解くよう求めた「9 分割なのに full-model が 1.24x に留まった」矛盾について、
既存の生証跡だけから再計算した記録である。

## 確認できたこと

- `sq8-decode-attention-rootcause/phase0-pmc-root/.../564735_kernel_trace.csv`
  の split partial dispatch は `Grid_Size_X=92160`、`Workgroup_Size_X=256`。
  したがって tile 128 / C=1036 では **360 WG**（40 Q head × 9 tile）であり、
  split が tile を一つの WG 内で逐次走査していた、という仮説は否定される。
- `1.236512x` の旧 full-model 測定そのものは C=513--519 で行われた。tile 128
  なら partial は 5 tile、すなわち **200 WG**（40 Q head × 5 tile）である。
  その値を C=1036 / 9 tile / 360 WG の trace と直接同一視することはできない。
  以下では、9 tile が実際に launch されることと、短い C の full-model の Amdahl
  効果を別々に確認する。
- 同じ trace の最初の attention dispatch は direct `0.926690 ms`、split partial
  `0.202283 ms`、split merge `0.018560 ms` である。profiler 時間は throughput
  ではなく、ここでは merge が partial を支配していなかったことと実際の launch
  形状を確認するためだけに使う。profiler が数値結果へ影響した記録もあるため、
  この時間から速度を報告しない。
- profiler を使わない旧 API probe
  `sq8_0-attention-optimization-u-v0.1/decode/split-bench.json` は C=1036 で
  direct `0.643241770 ms`、tile 128 `0.228016370 ms`（attention 呼出しで
  `2.82103x`）を記録している。

## Amdahl の再計算

旧 full-model 測定は C=513--519 で direct `53.519086 ms/token`、tile 128
`43.282296 ms/token`（`1.236512x`）だった。C=1036 の direct attention
`30.773224 ms/token` を C=516 に線形換算すると `15.327204 ms/token` で、
direct full-model の残差は `38.191882 ms/token` になる。これは cache length
の線形換算を含むため推定であり、物理 bandwidth の測定ではない。

この attention 部分だけを旧 API probe の `2.82103x` で短縮すると、

```text
38.191882 + 15.327204 / 2.82103 = 43.625 ms/token
53.519086 / 43.625 = 1.227x
```

となり、観測済み `1.236512x` に近い。つまり、WG 供給の増加は attention
呼出しを速くしたが、旧 full-model の短い C≈516 区間では attention が全体の
およそ 29% に過ぎず、残りの model layers は短縮されなかった。この Amdahl
効果が、10 倍近い partial-WG 数が 1.24x の end-to-end 倍率に見えた主因である。

split partial 内には token ごとの既存の 2 CTA barrier が残るため、WG 数の増加
がそのまま 10 倍の attention 短縮にもならない。L2 局所性、物理 HBM byte、
達成 occupancy、起動オーバーヘッドの個別寄与は、gfx1201 の有効な PMC がない
ため **未確認** である。

新設計の隔離測定では、generic split、GQA grouped、GQA grouped+pipelined を
同一 C≈1036 と同一 full-model runner で比較し、この説明を再検証する。
