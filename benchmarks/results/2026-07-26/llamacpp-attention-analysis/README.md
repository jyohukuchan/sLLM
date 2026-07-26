# llama.cpp R9700 decode-attention analysis

## 結論

**仮説は確認できた。** llama.cpp は、この Qwen3-14B の単一 token decode で
KV を workgroup 間に 10 分割する flash-decoding / split-KV 型の vector FATTN を
実行している。各 partial は online-softmax state を作り、別 kernel が
max/sum/weighted-V を統合する。これは、uLLM が multi-tile で数値 gate により
既定化しなかった手法と同型である。

ただし、llama.cpp の 86.5% は engine 全体の logical GB/s 指標であり、今回確認した
attention 機構だけで物理 HBM 86.5% を証明するものではない。特に raw profile の
kernel-duration sum では llama.cpp attention は **2.7628%**、uLLM phase1 の
direct paged attention は **51.05%** である。split は uLLM の大きな attention
ボトルネックを説明する強い構造証拠だが、weight format、KV precision/layout、
kernel群も異なるため、絶対時間だけを end-to-end の公正比較とは扱わない。

## 実機プロファイル

| item | value |
| --- | --- |
| device | R9700 / gfx1201、Agent 2、64 CU、wave32、32 waves/CU |
| llama.cpp | `68a5592c10666d4d89b8480b5b9e8f8068b2f64c`、既存 `build-rdna4/`、再ビルドなし |
| model/path | Qwen3-14B Q8_0 GGUF、F16 K/V、flash attention on |
| measured work | depth=1028 の prefill 後、single stream、16 M=1 generation |
| command isolation | `env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1` により llama-bench は ROCm0=gfx1201 のみを列挙 |
| CPU restraint | profiler child は `nice -n 19`、`ionice -c3`、`-t 8`、`-r 1` |
| profiler | rocprofv3 1.1.0 / ROCm 7.2.1、`--runtime-trace --stats` |

実行コマンドは [profile-command.txt](profile-command.txt)、R9700-only の device
列挙は [r9700-only-device-list.txt](r9700-only-device-list.txt)、raw CSV は
[rocprof](rocprof) に保存した。`/opt/rocm/bin/rocprofv3` は
`/opt/rocm-7.2.1/bin/rocprofv3` に解決される。

`--no-warmup` は benchmark warmup を除外するが、depth prefill 自体は raw trace に
残る。そこで `llama-bench` の `test_prompt()` / `test_gen()` が各 phase 後に
`llama_synchronize()` を呼ぶ source 順を使い、HIP API trace の 25 本の
1 ms 以上の `hipStreamSynchronize` のうち、depth completion の次にある 16 本を
decode completion fence とした。各 interval は vector main 40 回 + combine 40 回を
検証している。抽出手順と raw input SHA-256 は
[analyze_rocprof_decode.py](analyze_rocprof_decode.py) と
[profile-summary.json](profile-summary.json) に固定した。

この profile の 14.10 tok/s は `-t 8` と profiler を含むため、unprofiled baseline
34.885 tok/s の代替ではない。

## decode attention の実測

profiled 16 step 合計は 1,280 attention dispatch、12.219677 ms である。
kernel-duration sum での値を以下に示す。grid は rocprof の global work-item 数で、
WG 数は `ceil(grid_x/block_x) × ceil(grid_y/block_y) × ceil(grid_z/block_z)` で算出した。

| family | launches / decode step | raw grid / block | WG / launch | wave32 / launch | kernel ms / step | wave supply proxy |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| vector FATTN main | 40 | `(32,40,40)` / `(32,4,1)` | 400 | 1,600 | 0.663241 | 78.125% |
| FATTN combine | 40 | `(128,40,1)` / `(128,1,1)` | 40 | 160 | 0.100489 | 7.8125% |
| attention total | 80 | main + combine | 17,600 WG / step | 70,400 waves / step | 0.763730 | launchごとに上記 |

main の `Grid_Y / Block_Y = 40 / 4 = P=10` である。Grid Z=40 は Q head、40 main
launches/token は 40 transformer layer を示す。内部 KV length は 256 整列に pad
された 1280 で、128-token tile が 10 個になる。

R9700 の machine ceiling は `64 CU × 32 waves/CU = 2,048 wave32`。したがって
78.125% / 7.8125% は各 launch が queue に供給する wave 数を 2,048 で割った
**静的 supply proxy** であり、同時 residency、achieved occupancy、HBM bandwidth の
実測値ではない。rocprofv3 trace だけではそれらは **未確認**である。

選択 region の全 kernel-duration sum は 442.300522 ms（27.643783 ms/step）。
attention の 12.219677 ms は **2.762754%** である。step ごとの再現可能な値は
[decode-step-summary.csv](decode-step-summary.csv)、family ごとは
[attention-kernel-summary.csv](attention-kernel-summary.csv) にある。

## uLLM との直接比較

uLLM 側は既存の R9700 phase1 selected 16 M=1 trace
`benchmarks/results/2026-07-26/sq8-r9700-attention-phase1-v0.1/`
の default direct route を用いた。比較の machine-readable version は
[comparison-summary.json](comparison-summary.json) にある。

| per decode step | llama.cpp F16 profile | uLLM phase1 direct F32 paged | ratio / note |
| --- | ---: | ---: | --- |
| attention launches | 80 (main 40 + merge 40) | 40 | llama.cpp 2.0× |
| main attention WG / layer dispatch | 400 | 40 | llama.cpp 10× |
| attention WG / step | 17,600 | 1,600 | llama.cpp 11.0× |
| main launch wave supply | 78.125% | 15.625% | main geometry の比較 |
| attention kernel time / step | 0.763730 ms | 30.773224 ms | 40.29×、format/layout/profile scope の差あり |
| attention kernel-time share | 2.7628% | 51.05% | 独立 capture の sum-of-dispatch-duration share |
| all kernel dispatches / step | 851 | 1,364 | raw trace count |
| all kernel WG / step | 1,573,062 | 195,821 | 異種 kernel の幾何和であり occupancy ではない |

uLLM direct は 256-thread WG 一つを Q head ごとに launch するため、40 WG/layer、
8 wave32/WG、320 waves/layer dispatch、`320 / 2048 = 15.625%` である。llama.cpp
main は 128-thread WG を 40 Q heads × 10 partials に置くため、400 WG/layer、
4 wave32/WG、1,600 waves/layer dispatchになる。combine は別に 40 WG/layer を要する。

依頼文の「uLLM 約 288 launch/decode-step」は、利用可能な phase1 raw trace では
裏付けられなかった。同 trace は 21,824 `KERNEL_DISPATCH` 行 / 16 = **1,364**
all-kernel launches/step を記録する。ここではその raw trace と同じ定義で比較し、
288 の元となった別 scope/version は **未確認**とする。

## 構造上の意味

llama.cpp は連続 KV tensor を使い、vector body の `blockIdx.y` を source partial に
割り当てる。partial は online softmax の `(max, sum, weighted-V)` を出し、combine
が global max で rescale/merge する。uLLM direct は paged `block_table` をたどり、
各 Q head の単一 256-thread WG で token 順の online state を更新する。

このため、llama.cpp の並列度の源泉は「KV split + merge」であり、単に連続 cache
だからではない。連続 layout は page table 解決を不要にする別の差だが、40→400
main WG/layer への増幅そのものは P=10 split が作っている。

GQA も両者とも 5:1 を head index から処理している。llama.cpp は
`head / gqa_ratio`、uLLM は `q_head / q_per_kv` で 40 Q head を 8 KV head に
写像する。よって split は KV head を増やすのではなく、同じ GQA read を Q head と
source partial の積へ展開する。

詳細な source 根拠、F32 KV→F16 cast、current uLLM source と phase1 source snapshot
の SHA 不一致は [source-analysis.md](source-analysis.md) に記録した。

## 数値的性質と uLLM への適用

llama.cpp は P=10 partial merge を通常経路で実行しているが、P を指定する public
knob はなく、fixed input で P を変えた出力 A/B は実測できなかった。従って、
**llama.cpp の実際の出力差の大きさ・許容基準は未確認**である。一方 source の
reduction tree は direct と同一順序ではないため、bitwise invariance を保証しない
ことは確認できる。

uLLM ではまさに同型の multi-tile partial/merge が direct と異なる有限精度の
online-softmax association を作り、SQ8_0 の逐次 activation quantization がその差を
feedback で増幅した。既存 full-model gate は tile128/tile256 を NO-GO としており、
現在の `decoder.rs` は multi-tile を direct fallback にしている。

したがって次の方針になる。

1. 現行の bitwise gate を維持する限り、llama.cpp 型 `P>1` split/merge の移植は
   不可である。llama.cpp の merge は exact-state merge の実装ではない。
2. 数値契約を壊さずに検討できるのは、direct の token 順 online state を保持した
   page-address計算、load/coalescing、GQA reuse、launch overhead の改善である。ただし
   これらは 40→400 WG/layer という supply 増を生まないので、llama.cpp と同じ主効果は
   期待できない。
3. frozen v0.2（artifact-FP32 relative quality、JSON SHA-256
   `64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`）の下では、
   split は **明示的に非-bitwise候補**として再評価する余地がある。control と同等以上の
   FP32-reference 近さを、全 layer/hidden/logits/top-k と feedback decode で先に
   実証する必要がある。これは pass を予測するものではなく、可能性が **未確認**である。

本調査では uLLM production kernel、active manifest、`/opt/ullm` は変更していない。

## サービス・隔離記録

`ullm-openai.service` は実機 profile のため一度だけ stop → isolate → restore した。
時刻は [service-window-events.tsv](service-window-events.tsv) にある。

| time (JST) | event |
| --- | --- |
| 15:04:17 | window begin / stop begin |
| 15:04:26 | stop 完了、R9700 renderD129 / kfd holder check、profile begin |
| 15:04:37 | profiler exit=0、restore begin/完了 |

前後とも `ullm-openai.service=active/running/enabled, NRestarts=0`。
`llama-qwen35-udq4.service` は前後とも `inactive/dead/disabled, NRestarts=0` で、
起動していない。systemd unit、`/etc/ullm/served-models/active.json`、activation、
campaign、`/opt/ullm` は変更していない。

holder check の `lsof` は Docker overlay を stat できない warning を出したが、R9700
holder 行は出なかった。このため「その check が列挙した範囲で holder なし」は確認済み、
Docker namespace を含む絶対的な無 holder は **未確認**である。実際の model/profile
subprocess は visibility mask 後に gfx1201 一台だけを確認して実行した。

補足として、準備中に一度だけ visibility mask 前の `llama-bench --help` が backend
初期化で全 adapter を列挙した。model load、kernel dispatch、profile、service 操作は
なく、V620 上の GPU workload は実行していない。その後の llama-bench model/profile
実行はすべて上記 R9700-only mask を通した。
