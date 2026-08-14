# KV memory方式とattention kernelの初期決定

## 決定

2026-08-13時点のPhase 6初期方式は、canonical AMD Radeon Pro V620 `gfx1030`と
Radeon AI PRO R9700 `gfx1201`に限定して、HIP VMMによるvirtual-contiguous KV memory
（vAttention型）とする。KV storageはFP16のtoken-major `[capacity, kv_heads, head_dim]`であり、
最大logical capacityのvirtual addressをcreate時に予約し、K/V planeのphysical pageをappend時に
必要量だけcommitする。

vAttentionとFlashAttentionは排他的な方式ではない。vAttentionはKVのmemory management方式で、
attention kernelには連続したvirtual addressを通常のK/V pointerとして見せる。従って、対象backendで
利用可能なcontiguous-KV FlashAttention kernelはblock-table対応へ書き換えずに利用できる。
今回AMDで実測したkernelはFlashAttention-2そのものではなく、同じtiled online-softmax contractを持つ
`FA2-style proxy`である。production attention kernelをCK/CK Tileまたは別のFlashAttention系kernelへ
置き換える際も、KV memory contractは変更しない。

Paged Attentionは比較用model-free proxyだけを実装し、production backendとしては採用しない。
prefix block sharing、copy-on-write、RadixAttention、continuous batchingはこの決定の範囲外である。

## 比較のidentityと範囲

- local tuple: Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0。
- exact target: V620 `gfx1030`、R9700 `gfx1201`。他のRDNA2/RDNA4 SKUやROCm versionへ一般化しない。
- shape: Q heads 16、KV heads 4、head dimension 256、BF16 Q/output、FP16 token-major K/V。
- cases: query length 1/37、KV length 255/256/257/1023/1024/1025、warmup 3、measured 9。
- numerical contract: NumPy float64 softmax/matmul oracle、output absolute tolerance 0.016、
  mode間absolute tolerance 0.004。最大mode間誤差はV620 0.000732421875、R9700 0.000946044921875。
- comparison source: `ci/tools/vattention_a1_compare.hip.cpp`、SHA-256
  `9dbd91d2bf3c30bad505506ace62f95b324618c7afa9b02f2dec586e00c8bd9e`。
- production probe: `ci/tools/vattention_a1_production_probe.cpp`、SHA-256
  `3b973081b9d27acf1d6baccc0c2073af4846c814732257effb8aec3363505862`。
- local aggregate SHA-256:
  `453756b16f55ef81ff28dcb48cdebe69b9bdd83381b3a04202f94855af236021`。
- upstream facts-only pins: FlashAttention `145b1010051dbfd4bdc41a0ae55d495b08d7a458`
  （release v2.8.3 `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`）、Microsoft vAttention
  `ef3fff25dbe4e10f5897da8648718c53df6a20ea`、ROCm AITER
  `ef7dd32ca159e86b24f51447dbc9868d0aad7d1b`（release v0.1.13
  `cdcfa833bdf554ca75594c90dde4316ea9b50199`）、local vLLM
  `568afb3a13806beb53bb2e6bd518269357b237c0`。いずれもno-copyである。

local `amdrocm-ck7.14` 7.14.0-3にはcontiguous FMHA headerとpaged-KV FMHA headerがあるが、
このexact shape/targetを選択するprebuilt/generated instanceと、そのまま呼べる安定したdispatch経路は
確認できなかった。header SHA-256はそれぞれ
`6c116c1c9666387a528bbcef9845fdd3f86d8a5af4390d918eb0c676fc76af88`と
`4785bea24c9833d21190493a0a798f11618c8fc3a5f4f35db9af276290d67ecf`である。
このため数値・性能のAMD実測は、accessor以外を同じにした独立HIP proxyで行った。下表の値を
CKまたはupstream FlashAttentionの性能値として扱ってはならない。

## AMD実測

代表caseのkernel p50/p95をμsで示す。`contiguous`と`vAttention`は同じkernelとpointer arithmeticで、
違いは通常allocationかVMM mappingかだけである。`paged`は同じ演算contractのblock-table accessorである。

| target | Q/KV | contiguous | vAttention | paged |
| --- | ---: | ---: | ---: | ---: |
| V620 `gfx1030` | 1/255 | 689.164 / 695.524 | 147.521 / 157.641 | 191.281 / 193.961 |
| V620 `gfx1030` | 1/1025 | 552.365 / 557.326 | 561.644 / 599.924 | 770.005 / 772.525 |
| V620 `gfx1030` | 37/255 | 1949.575 / 1967.295 | 1937.972 / 1973.253 | 2290.414 / 2383.334 |
| V620 `gfx1030` | 37/1025 | 7700.648 / 7761.488 | 7697.567 / 7765.328 | 9271.974 / 9529.896 |
| R9700 `gfx1201` | 1/255 | 83.080 / 83.560 | 72.800 / 74.000 | 88.880 / 89.120 |
| R9700 `gfx1201` | 1/1025 | 296.801 / 298.361 | 272.961 / 274.001 | 344.922 / 346.962 |
| R9700 `gfx1201` | 37/255 | 412.962 / 433.442 | 416.082 / 426.001 | 472.122 / 486.003 |
| R9700 `gfx1201` | 37/1025 | 1330.646 / 1344.446 | 1303.727 / 1336.647 | 1897.409 / 1972.049 |

短いcaseにはallocation順序やclock状態の影響が見えるため、vAttentionが通常allocationより高速だとは
判定しない。両者の長いcaseは概ね同等であり、目的どおり同一contiguous kernelを維持できた。
Q=37/KV=1025のvAttention p50はpaged proxyよりV620で約17.0%、R9700で約31.3%短かった。

logical 16 MiBに対し、vAttentionのK/V合計commitはKV 255〜1024で4 MiB、1025で8 MiBだった。
page growはcaseごとに約74〜221 μsで、1024 token境界単位へ償却できる。Paged proxyの論理block bytesは
KV 255で1 MiB、1025で5 MiBだが、device allocationの観測VRAMはallocation granularityにより
それぞれ2 MiB、10 MiBだった。これは異なるallocatorの厳密なpeak-memory比較ではない。

## FlashAttention世代との関係

| kernel family | contiguous KV + vAttention | paged KV | V620/R9700 evidence |
| --- | --- | --- | --- |
| FA2 / FA2-style | contiguous pointerのまま利用可能。今回同一proxyで実測 | upstreamにはblock-table interfaceがあるがkernel側metadata処理が必要。今回proxyで実測 | proxy numerical/performance PASS。upstream FA2/CK性能は未実測 |
| FA3 | virtual-contiguousというmemory contract自体は再利用可能 | 対応には対象実装のpaged interfaceが必要 | upstream実装はHopper向けのためdesign comparisonのみ |
| FA4 | virtual-contiguousというmemory contract自体は再利用可能 | 対応には対象実装のpaged interfaceが必要 | upstream実装はHopper/Blackwell向けのためdesign comparisonのみ |

この表はalgorithm/interfaceの比較であり、FA3/4をAMDで動かした証拠ではない。将来AMD向けの
FA3/4相当kernelが現れた場合も、contiguous pointerを受ける限りvAttention上で利用できる。

## production契約

- public C ABIのKV create/view versionは2で、memory kind
  `SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS`とlayout
  `SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR`を明示する。
- native `KvState`がVA reservation、physical handles、mapping、event lifetimeを所有する。
  createはVAだけをreserveし、appendはlaunch前にK/Vを同じpublished capacityまでgrowする。
- cancelはpublicationを行わない。既にcommitしたpageはstate lifetime中は保持し、release時に
  unmap、handle release、VA freeを行う。cancel/release cleanupはidempotentである。
- viewはlogical capacity、mapped token capacity、physical page bytes、K/Vのcommitted bytesを返すが、
  scheduler、generation service、HTTP層へdevice pointer、VMM handle、page tableを公開しない。
- private evidence readbackもpublishedかつmappedな範囲だけを許可し、未map領域を成功扱いにしない。
- actual public runtimeのfocused probeは両targetで1023/1024/1025 token、FP16全要素oracle、
  2/2/4 MiB per-plane commitment、未map readback拒否、fallbackなし、cleanupをPASSした。

## 再検討条件

次のいずれかが具体的な要件または実測結果になった時点でPaged Attentionを再比較する。

- prefix KVのblock sharing、copy-on-write、eviction/reuse、continuous batchingを実装する。
- 新しいsupported targetでHIP VMMが利用不能、またはreserve/map/accessの正しさを証明できない。
- 小刻みなpage activationがrequest latencyを支配し、pageの事前growや償却でも解消できない。
- VA reservation量、fragmentation、mapping数またはdriver制約が実用上の上限になる。
- 採用予定のAMD attention backendがpaged-KVだけを提供し、同一数値contractで明確な総合優位を示す。

比較用proxyはproduction Paged Attentionの完成を意味しない。再検討時は同じ数値oracle、exact target、
health/cleanup、非整列値とblock境界を維持して測り直す。

[対応するPhase 6計画](../plans/archive/2026/08/11-20/openai-chat-completions-v1.md)
[対応する履歴](../history/2026/08/11-20/openai-chat-completions-v1.md)
