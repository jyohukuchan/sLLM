# Phase 53 KV FP8 block16／標準MXFP8実装・target別判定履歴

> 日付: 2026-08-27
> 状態: 完了（descriptor v2 target-separated evidence、default mappingなし）
> safety default: `fp16`

## superseded recipeで実装したcontract

- `kv-fp8-e4-block16`／`kv-fp8-e5-block16`のdescriptor v1は同一token、K/V plane、KV headのhead-dimension方向16値ごとに
  E8M0 scaleを持った。physical valueはE4M3 FNUZ／OCP E4M3FN／software E5M2をexact target descriptorで分離する。
- 標準OCP MXFP8はcanonical public name `kv-mxfp8-e4`／`kv-mxfp8-e5`、block 32、E8M0 scale、OCP E4M3／E5M2、
  token内head-dimension方向である。initial runtimeではexplicit-onlyで、default候補ではない。
- 両形式の末尾value planeはblock境界へpaddingし、valid laneだけからscaleを計算してtailをcanonical zeroにする。K/V valueと
  scale planeは独立ownerで、append publication、grow/COW rollback、export/import、checkpoint、prefix cache、Qwen/Gemma state
  identityへencoding、block size、physical variantを結合した。
- CLI/server/model manifestはrequested／resolved encoding、selection source、reason、physical variant、descriptor ID、policy versionを
  reportする。省略時とunknown／scope外はFP16 safety defaultを維持し、unsupported explicit requestはsilent fallbackしない。
- `gfx1201`はMXFP8 E4 OCP、`gfx1030`はMXFP8 E5 OCPを明示選択できる。`gfx942:sramecc+:xnack-`はnative FNUZ byte列を
  standard OCP E4M3と再解釈できないため標準MXFP8を明示unsupportedとした。

この節と以下のv3／v4 evidenceでblock16が使ったdescriptorは`kv-fp8-e4-block16-v1`／`kv-fp8-e5-block16-v1`、
scale recipeは有限値を飽和させない最小E8M0 scaleである。
2026-08-27のユーザー決定により、このrecipeはsupersededとなった。

## descriptor v2 follow-up contractと結果

- `kv-fp8-e4-block16`／`kv-fp8-e5-block16`はhead-dimension方向のblock size 16を維持し、descriptorを
  `kv-fp8-e4-block16-v2`／`kv-fp8-e5-block16-v2`へ上げる。32へ変更しない。
- scale決定はE4／E5とも`StandardMxFloorPowerV1`へ統一し、有限amaxの最大2冪をE4では256、E5では32768で割って
  E8M0範囲へ収める。element encodeはRNE＋SATとする。
- 下記のdescriptor v1 correctness／quality evidence、summary、mappingは監査履歴としてdigestと値を変更せず保持するが、
  新recipeのcorrectness、quality、default採否を証明しない。
- gfx1201／gfx1030のfresh correctnessとfull-model品質を再取得し、両targetともcorrectness PASS、品質threshold未達で
  `retain-fp16`とした。default mappingは空である。MI300X `gfx942:sramecc+:xnack-`は将来の一括検証へdeferしたままである。

### descriptor v2 correctness evidence

| target / encoding | raw evidence | SHA-256 | result |
| --- | --- | --- | --- |
| `gfx1201` / `kv-fp8-e4-block16-v2` | `external:phase53/gfx1201/block16-standardmx-v2.json` | `98bdd7046454146186408ce425160cd26ee5de2b886cc4d2b6f722324606318f` | host 15/16/17/255/256/257、GPU append 6、direct attention 1、scale／value byte exact、fallback 0、cleanup 0、PASS |
| `gfx1030` / `kv-fp8-e5-block16-v2` | `external:phase53/gfx1030/block16-standardmx-v2.json` | `3b8e11015266ff1a401d7e2c76048542d3575f54f16f074922eee1f7f7f2210f` | host 15/16/17/255/256/257、GPU append 6、direct attention 1、scale／value byte exact、fallback 0、cleanup 0、PASS |

最初のgfx1030実行ではV2 E5 wrapperがencoding ID 9を渡す一方、共有quantizerのE5判定が旧ID 5だけを見てE4 codecを選ぶ欠陥を
検出した。V2 IDも判定するよう修正し、上記fresh binaryでhost／GPU byte exactとattention numerical matchを再確認した。

### descriptor v2 full-model qualityと判定

各targetでFP16→block16→MXFP8を完全直列に3 repeatし、residentを次の方式の前に解放した。全repeatはHIP-only、fallback 0、
terminal cleanup 0で同値だった。

| target | raw quality report / SHA-256 | block16 v2 metrics | reference-only MXFP8 metrics | decision |
| --- | --- | --- | --- | --- |
| `gfx1201` | `external:phase53/gfx1201/quality-block16-standardmx-v2.json` / `311b1d4a8c6981cc7252a68e77bb99eac50ec7439e30a07fd9efcedae29719ea` | perplexity relative delta `-0.024857492069955328`、KLD p99 `0.006562189165612111`、top-1 `0.9`、task loss `0`、long-context loss `0.16666666666666663` | KLD p99 `0.004945428206833837`、top-1 `0.85`、long-context loss `0.16666666666666663` | top-1とlong-context threshold未達のため`retain-fp16` |
| `gfx1030` | `external:phase53/gfx1030/quality-block16-standardmx-v2.json` / `0f2a04bba11552758fe8e8dcea45e2a53447ea59d9719b7196c7998daac2ffa4` | perplexity relative delta `-0.04207210262266109`、KLD p99 `0.03659844555378746`、top-1 `0.9`、task loss `0`、long-context loss `0.08333333333333337` | KLD p99 `0.03218873133110086`、top-1 `0.8`、long-context loss `0.16666666666666663` | top-1とlong-context threshold未達のため`retain-fp16` |

E5 block16 KLDは旧recipeの`0.04331390780013198`から`0.03659844555378746`へ改善し、旧scale ruleが劣化要因の一部だったことを
支持する。ただし通常MXFP8の`0.03218873133110086`とは一致しない。E4もv2 block16 `0.006562189165612111`とMXFP8
`0.004945428206833837`に差がある。両形式は同じscale ruleでもblock sizeが16／32で、block境界ごとのamax、scale exponent、
丸め値が変わるためであり、同じscale ruleは同じ量子化結果を意味しない。KLDの大小はモデルの非線形伝播を含むので小blockが常に優位とも限らない。

v2 summary `external:phase53/phase53-kv-default-summary-standardmx-v2.json`のSHA-256は
`c259e81bc76fb341e9dbba8cdcc0c132456a0762585b471556267cdc59165e10`、空mapping
`external:phase53/phase53-runtime-mapping-standardmx-v2.json`は
`283911d387c67d7ba25546ce702fd89ee6a25db482d9e62bf0b61854d5613e77`、policyは
`3f0ae660804d4f2782cadfcf54533e88db214ab1af91643301ec984f4e03415d`である。品質FAIL後のfrozen early-stopによりperformance/resourceは
実行せず、memory／performanceをPASSとは扱わない。

## E5M2 scale selector診断

2026-08-27にexact gfx1030／Qwen3.5-4Bだけで、`LocalMinMse`（`e16`／`e16-1`／`e16+1`）と
`Parent32GuardedMinMse`（左記＋parent MXFP8の`e32`）を一回ずつ比較した。FP16→candidate block16→MXFP8を完全直列に実行し、
同時常駐、fallback、cleanup残留はなかった。両候補ともhost／GPU scale・value byte exactとdirect attention numerical oracleをPASSした。

| candidate | KLD p99 | top-1 | long-context loss | 比較結果 |
| --- | ---: | ---: | ---: | --- |
| `LocalMinMse` | `0.04063529273873547` | `0.8` | `0.16666666666666663` | 旧v1 `0.04331390780013198`より改善、production v2 `0.03659844555378746`とMXFP8 `0.03218873133110086`には未達 |
| `Parent32GuardedMinMse` | `0.04063529273873547` | `0.8` | `0.16666666666666663` | `LocalMinMse`と同一。parent `e32`追加による改善なし |

値の再構成SSE最小化は最終logit KLDを最小化せず、現在のamax由来standard MX ruleより悪化した。両候補を棄却し、production source、
descriptor v2、空mapping、FP16 defaultを変更しない。診断summary
`external:phase53/gfx1030/e5-scale-selector-diagnostic-summary-v1.json`のSHA-256は
`17622a66dee6d7312028a8699683f9e76a7919764bf6a1a856a994f48b2aebf6`である。raw quality runnerは既存v2 report型を再利用したため、
selector identityはsummaryの個別binary SHAへ結合し、v2 aggregate／runtime mappingから除外する。

[対応する保存済み診断計画](../../../../plans/archive/2026/08/21-31/phase53-e5-block16-scale-selector-experiment.md)

## superseded descriptor v1 correctness evidence

| target / encoding | raw evidence | SHA-256 | result |
| --- | --- | --- | --- |
| `gfx1201` / `kv-fp8-e4-block16` | `external:phase53/gfx1201/block16-correctness-v4.json` | `4acadb97d815edc730d3918bf2ba51a3fb4a2361f1959c8f18bfa97910760a62` | host 15/16/17/255/256/257、GPU append 6、direct attention 1、fallback 0、cleanup 0、PASS |
| `gfx1201` / `kv-mxfp8-e4` | `external:phase53/gfx1201/mxfp8-correctness-v3.json` | `5a5ab3fcc4eac635aaadb59e7549c7deedded8811d20c924bd126c89fba2af98` | host 31/32/33/255/256/257、GPU append 6、direct attention 1、fallback 0、cleanup 0、PASS |
| `gfx1030` / `kv-fp8-e5-block16` | `external:phase53/gfx1030/block16-correctness-v3.json` | `294eb0363ea73f0bfff73ffd2fea4f493ece9123c37c7a740f659d639f9a8a17` | host 15/16/17/255/256/257、GPU append 6、direct attention 1、fallback 0、cleanup 0、PASS |
| `gfx1030` / `kv-mxfp8-e5` | `external:phase53/gfx1030/mxfp8-correctness-v3.json` | `f262458ccd460922db5238a22ff572c3de37ea9c1df3c070d987dda6517d4967` | host 31/32/33/255/256/257、GPU append 6、direct attention 1、fallback 0、cleanup 0、PASS |

全reportはpolicy SHA-256 `3e8b1696ebfd485606762d9b3c07fd2694f6157abf43745eabbbc2913240cb1d`へ結合する。
gfx1201 correctness binaryは`40518586be413d2b370a12e2dba05ff2ba3661392853d64a224b48e38a294e3d`、gfx1030は
`4e09989f6d2f3b38eeaa1b6aca70e4b81c5c57c08331e0c5fd4bb670faf04c66`である。

## superseded descriptor v1 full-model qualityと判定

固定Qwen3.5-4B model lock fingerprintは
`sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、datasetは
`sha256:a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d`である。各repeatはFP16 residentを
解放後block16、block16解放後MXFP8を作り、MXFP8解放後に次repeatへ進む完全直列順で3回実行した。各repeatは20 selected、
perplexity 10、KLD/top-1 20、task 10、long-context 12 sampleで、全値はrepeat間で同一、HIP-only、fallback 0、terminal cleanup 0だった。

| target | raw quality report / SHA-256 | block16 metrics | reference-only MXFP8 metrics | decision |
| --- | --- | --- | --- | --- |
| `gfx1201` | `external:phase53/gfx1201/quality-block16-mxfp8-v3.json` / `2bff500c4edf504d566fa854a2939f2debf4b5ef385919979b1a41f976b2ccc4` | perplexity relative delta `0.005612459472037752`、KLD p99 `0.0038687249522990803`、top-1 `0.85`、task loss `0`、long-context loss `0.08333333333333337` | KLD p99 `0.004945428206833837`、top-1 `0.85`、long-context loss `0.16666666666666663` | block16のtop-1とlong-contextがfreeze済みthreshold未達のため`retain-fp16` |
| `gfx1030` | `external:phase53/gfx1030/quality-block16-mxfp8-v3.json` / `d66a6375fe5de4ec7f9f0bfd9d85ac3fe57abcc850bc07c59e765b31a18f40a1` | perplexity relative delta `-0.0314909276120897`、KLD p99 `0.04331390780013198`、top-1 `0.8`、task loss `0`、long-context loss `0.16666666666666663` | KLD p99 `0.03218873133110086`、top-1 `0.8`、long-context loss `0.16666666666666663` | block16のtop-1とlong-contextがfreeze済みthreshold未達のため`retain-fp16` |

旧recipeのgfx1030でblock16のKLDが標準MXFP8より大きかった主因候補はscale recipeの差である。E5M2の最大有限値は
`57344 = 1.75 * 2^15`で、block16は有限最大値を飽和させない最小scaleを選ぶ。このためblock内amaxを`m * 2^e`
（`1 <= m < 2`）とすると、`m > 1.75`でscaleが`2^(e-14)`へ一段上がる。標準MXFP8はblock 32でも
`2^(e-15)`を維持し、最大側をSATし得る代わりに残りの値へ2倍細かい刻みを使う。仮数2 bitのE5M2ではこの一段差が
小さいK/V値へ強く効き、block16の局所性を上回ったと推定する。これはaggregate品質値とcodec式からの説明であり、
scale exponent差、SAT数、K/V・layer別誤差を直接countした因果証拠ではない。gfx1201 E4では逆にblock16 KLDが小さく、
block 16または32の普遍的優劣を示す結果ではない。

freeze済みgateはtop-1 agreement `>=0.99`、long-context score delta `<=0.02`である。correctness／qualityの明確なFAIL後は
7行performance/resourceを要求しないearly-stop ruleに従い、両targetの通常5行＋長時間2行は実行しなかった。これは性能PASSでも
memory FAILでもなく`insufficient-evidence`として保存し、品質FAILを隠さない。

## superseded descriptor v1 summary、rollback、未完了範囲

- `external:phase53/phase53-kv-default-summary-v3.json` SHA-256:
  `2440fd7726fca24919731abdcbd2b0f74fdd9d663ecca850b369b5ae3e69dd2b`。
- `external:phase53/phase53-runtime-mapping-v3.json` SHA-256:
  `ecc05a91899c9275a7dc1234f418555ad860b18680c2ad15ea7eb745b7127dff`。mappingは空で、statusは
  `candidate-not-runtime-policy`、safety defaultは`fp16`である。
- `gfx942:sramecc+:xnack-`はfresh correctness、quality、performance/resourceが全て未取得なので`insufficient-evidence`。
  過去PhaseのMI300X証拠やcompile-onlyをPhase 53 PASSへ流用しない。
- rollbackはruntime mappingを必要とせず、現状の省略時FP16を維持する。新形式のexplicit selection、descriptor ID、checkpoint tag、
  raw evidenceは保持する。correctness defectが見つかった場合は該当explicit providerをunsupportedへ狭める。

2026-08-27のユーザー指示により、MI300X実機検証は追加の検証項目がまとまった将来の一括実行へ延期した。Hot Aisleの
固定IPへ到達できることをVM存在の証拠にせず、gfx942の`insufficient-evidence`とFP16 safety defaultを維持する。
gfx1201／gfx1030の旧recipe非採用判定、空mapping、explicit形式の実装完了をもって元のPhase 53作業単位をtarget分離で
archiveした。その後のdescriptor v2／`StandardMxFloorPowerV1`決定により、旧非採用判定を新recipeへ流用せず、fresh correctness／品質で
再判定した。gfx1201／gfx1030はともに`retain-fp16`、MI300Xはdeferを維持し、Phase 53を完了した。

[対応する保存済み計画](../../../../plans/archive/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)
