# R9700 sLLM / llama.cpp 10,001/2 E2E比較

## 結果

2026-08-21にcanonical R9700、exact `gfx1201`、ROCm 7.14.0で、Qwen3.5-4B BF16、FP16 KV、
token ID `23066`を10,001個入力してgreedy 2 tokenを生成するdirect E2Eを、sLLMとfixed llama.cpp
`b10453`で各3 warmup＋10 measured実行した。両engineは全試行で`[23066,23066]`を生成した。

| metric（10回中央値） | sLLM | llama.cpp | sLLM / llama.cpp |
| --- | ---: | ---: | ---: |
| TTFT | 3.865440494 s | 2.040909736 s | 1.89398 |
| prefill | 3.858745634 s | 2.039533770 s | 1.89197 |
| prefill throughput | 2,591.775 tok/s | 4,903.576 tok/s | 0.52855 |
| TPOT（1 decode token） | 53.015937 ms | 21.509063 ms | 2.46482 |
| decode throughput | 18.862 tok/s | 46.492 tok/s | 0.40571 |
| E2E | 3.936429665 s | 2.063845785 s | 1.90733 |

sLLM E2E MADは10.414 ms、llama.cppは5.529 msだった。sLLMは全13 performance sampleで
HIP-only、fallbackなし、kernel dispatch `13,416`、submission `12,168`、model load 1回・resident reuse、
cleanup 0を報告した。llama.cppは33/33 layerを同じ一台へoffloadし、model/contextを1回だけ作成して再利用し、
cleanup 0だった。開始前・終了後ともR9700 processは0で、終了時VRAMは0%、throttle statusは`UNTHROTTLED`だった。

## Identityと比較境界

sLLMはbase commit `faf39339d42c837c1ff899f90b03632ac5fe57af`のcaptured dirty worktreeから新規にexact
gfx1201 release buildした。binary SHA-256は`9ac318611bb8e95bd69769d8d98a9b0b517c56d81ce7dde2739c4de22a15164f`、
build-input manifest SHA-256は`cc05d6de988f7084453c705bb068346402dca9dffb9e8cd0f2430880f24b90ee`、
worktree patch SHA-256は`fc08847677cd2370ff3c32e870deff6e1dee12e9a6c9471c4935211fa585d0e9`である。

llama.cppはcleanなcommit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`、tree
`f9a9f82f92eb23b6dbc05494e542ddb1f907a0c4`、tag `b10453`から、`GGML_NATIVE=OFF`、
HIP architecture `gfx1201`、HIP Graph on、HIP VMM offで新規buildした。wrapper source SHA-256は
`482db0addf9e2321dcd37869dc7401bc301c48bd671a264489135dc0f92a134e`、wrapper binary SHA-256は
`3b9571d33456d8b4af4af050a3f307b54dc32c1108f91985c69d07e29467c353`で、いずれも外部raw artifact storeへ保持した。

両modelは同じ`Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`に由来するが、
sLLM GGUFはMTP tensorを含む`c571c54e...`、llama.cpp GGUFは`--no-mtp`変換の`636158bd...`であり、
byte、tensor set、converter lineageが異なる。MTPは両engineで無効にした。同一token IDs、同一生成数、同一weight/KV dtype、
同一GPU上のsystem-equivalent比較ではあるが、strict artifact identityではないため分類は`E1_SYSTEM_EQUIVALENT`とする。

Phase 35のR9700 `65.214`秒はmessages inputと生成`[2064,5686]`の別protocolであり、このdirect比較の比率には使用しない。
また、Phase 36 Session DのMI300X `26.4975x`との差は、当該gfx942 sourceが最適化Full Attention/GDN provider対象外だった
既存診断と整合するが、このR9700測定自体はPhase 36を再開せず、完了後の独立比較記録として扱う。

全10 sample、GPU health、binary/source/build identityはtracked summaryにdigest付きで固定し、rawはmodel/binaryと同様に
repository外の`/home/homelab1/.local/share/sllm-evidence/r9700-sllm-llama-e2e/2026-08-21-r1`へ保持する。
このsummaryは測定時のbase commitとworktree patchを固定したhistorical evidenceであり、その後のcommitを現在の測定candidateとして
遡及適用しない。

[tracked summary](../../../../../ci/matrix/r9700-sllm-llama-e2e-v1.json) /
[main plan](../../../../plans/main-plan.md)
