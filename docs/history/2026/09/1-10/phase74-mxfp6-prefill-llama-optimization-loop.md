# Phase 74: MXFP6 prefill llama.cpp比較最適化ループ履歴

状態: `完了（2026-09-03）`

## 結論

Qwen3.5 denseのOCP MXFP6 E3M2 W6A6 prefillだけを対象に、llama.cpp Q6_Kとの比較から
`比較→単一候補実装→operator検証→full-model benchmark→残差profile`を3回実施した。

- Loop 1はexact `gfx1030`へID47 half2 dot2 32x32を採用した。4B 3 warmup＋10 measuredの
  512／2,048入力は旧ID25比`1.636倍／1.601倍`になった。
- Loop 2はexact `gfx1201`へID48 packed E3M2x4→E4M3x4 SWAR ingressを採用した。同条件で
  旧ID45比`1.315倍／1.192倍`になった。
- Loop 3は両target共通のactivation quantizer packed-store候補を試したが、4B 1 warmup＋3 measuredで
  全行が約0.35〜0.84%退行した。候補は棄却し、製品sourceから除去した。

最終既定経路は4B 512／2,048で`gfx1030=411.82／393.60 tok/s`、
`gfx1201=2,843.97／2,959.30 tok/s`を再現した。27B 512も`57.22／475.51 tok/s`で両target PASSした。
llama.cpp Q6_Kとの差は残るが、異なる量子化形式間のsystem比較であり、Phase 74の完了阻害条件にはしていない。

## 固定identityと測定境界

- 開始Git HEAD: `586c27b60d976781963b4a3e0901f9be3cb2c9e2`。Phase成果は未コミットのdirty sourceであり、
  Git commitを最終semantic identityとはしていない。
- ROCm `7.14.0`、AMD clang `23`、Code Object V6、wave32。
- V620: exact `gfx1030`、UUID `GPU-76a08c022586fed6`、BDF `0000:03:00.0`。
- R9700: exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`。
- 4B: `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、lock fingerprint
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`。
- 27B: `Qwen/Qwen3.5-27B` revision `fc05daec18b0a78c049392ed2e771dde82bdf654`、lock fingerprint
  `sha256:a4a0a6192babfdb7b1fc3ac75cc340e96df87fe2b0e629cc1510085bfeced97f`。
- sLLMはdirect pretokenized `[23066, ...]`、512／2,048 token、single chunk、FP16 KV、greedy、
  `max_new_tokens=2`、ignore EOS。採否値はrequest内のprefill submit→completeだけを計測し、model loadとdecodeを含めない。
- fixed llama.cppはcommit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`、Q6_K 4B、
  `n_batch=2048`、`n_ubatch=512`、FP16 KV、10 repetitionのprompt processing値を使った。Q6_KとMXFP6は
  数値形式、activation形式、scale契約が異なるため、同一算術kernelのoracleではなくsystem性能の参照値である。

最終CLI SHA-256は`gfx1030=a3ed26a5aa01c7f3651fb735383fdc5311c65324fb5484b84e2222765a5e1df5`、
`gfx1201=aef19ee6de27099e5425eae57e9f7bf17c717ada2088bedf0148d5e7daff4294`。operator runnerは
`d1b7d7dcdb87afea17f58a8aca69b0a35de440d07efbeb629eb86f34aeb37c73／`
`6d4b055e77cf82cf14266589701eaa8427da9b78684fd13fca42679968ae78dc`である。

## llama.cpp比較

固定llama.cpp sourceでは、Q6_KのRDNA2／RDNA4 shape表を
`ggml/src/ggml-cuda/mmq-config-rdna2.cuh`と`mmq-config-rdna4.cuh`、packed tile loadを
`mmq-load-tiles.cuh`、Q6_K×Q8_1 dotを`mmq-vec-dot.cuh`と`vecdotq.cuh`で確認した。

sLLMへ転用可能だった考え方は、target別shape、packed load、1 workgroup内の複数出力、K tileを介した再利用、
hardware dot／matrix primitive、明示rollbackである。一方、Q6_Kのinteger code／複数scaleとQ8_1 activationを前提にした
`dp4a`の式は、E3M2実数とE8M0 block32 scaleを持つMXFP6へそのまま適用できない。ID47は全64 E3M2 codeがFP16へ
exactに写ることを利用し、ID48はE3M2からE4M3FNへexactに写して既存WMMAへ渡した。llama.cppコードの直接流用はなく、
比較したtile／load／dot構成を形式固有契約から分離して独自実装したため、新しいprovenance importは発生していない。

固定baselineは次のとおり。llama.cpp値はQ6_K、sLLM値はMXFP6なのでformat差を含む。

| target | input | Phase開始時sLLM | llama.cpp Q6_K | 開始時の差 |
| --- | ---: | ---: | ---: | ---: |
| `gfx1030` | 512 | 256.01 tok/s | 2,077.47 tok/s | 8.11倍 |
| `gfx1030` | 2,048 | 249.84 tok/s | 2,061.67 tok/s | 8.25倍 |
| `gfx1201` | 512 | 2,176.82 tok/s | 3,850.23 tok/s | 1.77倍 |
| `gfx1201` | 2,048 | 2,467.01 tok/s | 3,828.11 tok/s | 1.55倍 |

## Loop 1: gfx1030 ID47 half2 dot2

fresh profileでは旧ID25 tiled16がkernel durationの`94.78%`を占めた。ID25は16x16 output、scalar E3M2展開、
1 output/thread、FP32 FMAだった。比較結果から、32x32 output tile、256 thread、4 output/thread、K32ごとのFP16 LDS、
exact E3M2→FP16、`v_dot2_f32_f16`を使うID47を実装した。

- logical: `matmul.mxfp6.w6a6.gfx1030.half2.32x32.v1`
- device: `sllm_mxfp6_w6a6_gfx1030_half2_32x32_v1`
- resource: LDS `4,352` byte、SGPR `33`、VGPR `65`、private／spill `0`、wave32、workgroup `256`、
  static `v_dot2_f32_f16=64`。code object SHA-256は
  `fc07ec222c7ea31453a8fc140a83a29a30a08f091b8777fb6abacf80ba0657dd`。
- 数値classはN1。K32ごとのblock scale適用とreduction treeがID25と異なるため、BF16 output digestが変わるshapeを許容し、
  独立FP32 oracleで判定した。

全64 E3M2 codeのFP16 bits、tail、selector境界、signed zero、E8M0 edge／NaN、repeatをPASSした。
operatorの最大relative errorは`0.0038758`、nonfinite mismatchは0。`M=512/2048,K=2560,N=9216`の
kernel medianは`11.723/46.923 ms`から`7.754/30.885 ms`へ短縮した。

4B 3 warmup＋10 measuredは次のとおり。

| input | ID25 median±MAD | ID47 median±MAD | speedup | prefill time |
| --- | ---: | ---: | ---: | ---: |
| 512 | 249.37±1.07 tok/s | 407.97±0.17 tok/s | 1.636倍 | 2.053→1.255 s |
| 2,048 | 244.08±2.08 tok/s | 390.69±0.30 tok/s | 1.601倍 | 8.391→5.242 s |

exact `gfx1030`、`M>=128`、`K>=2048`かつ32整列、`1024<=N<=32768`だけをID47の既定scopeにした。
`SLLM_MXFP6_PREFILL_FORCE_TILED16=1`をrollbackとして維持した。

## Loop 2: gfx1201 ID48 packed conversion SWAR

開始時profileではID45 matrixが`66.68%`、activation quantizerが`2.44%`だった。ID45のWMMA tile、LDS、scale、
gridは維持し、packed E3M2x4からE4M3FN x4への変換だけを32-bit byte-lane SWARへ置換したID48を実装した。

- logical: `matmul.mxfp6.w6a6.gfx1201.wmma128x64.pack4-swar.v1`
- device: `sllm_mxfp6_w6a6_gfx1201_wmma128x64_pack4_swar_v1`
- resource: LDS `6,912` byte、SGPR `36`、VGPR `114`、private／spill `0`、wave32、workgroup `256`、
  static FP8 WMMA `8`。code object SHA-256は
  `fa5b961db8f06d97cb70ab259904910b12c67981ad56f623719f9df914b46519`。
- ID45と同じ数値treeを保持するN0候補で、全operator output digestが一致した。

全64 E3M2 codeを4 laneへ混在させたSWAR／scalar比較、tail、selector境界、repeat、独立FP32 oracleをPASSした。
最大relative errorは`0.0038758`、nonfinite mismatchは0。`M=512/2048,K=2560,N=9216`のkernel medianは
`1.219/4.478 ms`から`0.853/3.135 ms`へ短縮し、いずれもID45比約0.70だった。

4B 3 warmup＋10 measuredは次のとおり。

| input | ID45 median±MAD | ID48 median±MAD | speedup | prefill time |
| --- | ---: | ---: | ---: | ---: |
| 512 | 2,126.69±3.94 tok/s | 2,796.37±2.73 tok/s | 1.315倍 | 240.75→183.09 ms |
| 2,048 | 2,437.99±7.62 tok/s | 2,906.63±8.36 tok/s | 1.192倍 | 840.04→704.60 ms |

exact `gfx1201`、`M>=17`、`K>=2048`かつ32整列、`1024<=N<=32768`という既存ID45 scopeだけを
ID48へ置換した。`SLLM_MXFP6_PREFILL_FORCE_PHASE70=gfx1201-n64-pack4`でID45へrollbackできる。

## Loop 3: activation quantizer packed-store候補

Loop 1／2後もactivation quantizerは最終profileの`gfx1030=0.545%`、`gfx1201=3.092%`を占めた。
E3M2変換とscale treeは変えず、4個の24-bit groupを隣接pair化し、block当たり24回のbyte storeを
4回のaligned 32-bit＋4回のaligned 16-bit storeへ減らす候補を実装した。

両targetの10 operator shape、各3 repeatは、既定quantizerと全output digest一致、最大relative error
`0.0038758`、nonfinite mismatch 0、deterministicでPASSした。しかし4B screeningは全行で退行した。

| target | input | control | packed-store候補 | 変化 |
| --- | ---: | ---: | ---: | ---: |
| `gfx1030` | 512 | 412.53 tok/s | 410.99 tok/s | -0.37% |
| `gfx1030` | 2,048 | 394.36 tok/s | 392.99 tok/s | -0.35% |
| `gfx1201` | 512 | 2,832.64 tok/s | 2,809.00 tok/s | -0.83% |
| `gfx1201` | 2,048 | 2,960.17 tok/s | 2,941.88 tok/s | -0.62% |

追加shuffleと広幅storeの費用がscalar store削減を上回り、削減可能上限も小さいと判断した。候補は両targetで棄却し、
force selectorとkernelを最終sourceから除去した。証拠JSONは棄却判断の再現用に保持した。

## 最終4B、27B、残差

最終sourceを両targetで再buildし、force変数なしのproduction defaultを1 warmup＋3 measuredで確認した。

| target | model | input | median prefill | resident | peak | state |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `gfx1030` | 4B | 512 | 411.82 tok/s | 4,061,763,072 | 4,400,326,144 | PASS |
| `gfx1030` | 4B | 2,048 | 393.60 tok/s | 4,061,763,072 | 5,261,284,864 | PASS |
| `gfx1201` | 4B | 512 | 2,843.97 tok/s | 4,061,763,072 | 4,400,326,144 | PASS |
| `gfx1201` | 4B | 2,048 | 2,959.30 tok/s | 4,061,763,072 | 5,261,284,864 | PASS |
| `gfx1030` | 27B | 512 | 57.22 tok/s | 24,115,002,880 | 24,776,887,808 | PASS |
| `gfx1201` | 27B | 512 | 475.51 tok/s | 24,115,002,880 | 24,776,887,808 | PASS |

全行でgenerated tokenは`[23066,23066]`、全dispatch HIP、fallbackなし、allocator cleanup正常、
retryable cleanup／durable quarantineは0だった。27B selectorにmodel名や`N=17408`の特例はなく、同じshape predicateで適用された。

最終512 profileのkernel duration内訳は次のとおり。profileは短いdecodeを1 token含むため、decode matrixはPrefill採否値に
含めず残差として分離した。

| target | MXFP6 prefill matrix | activation quantizer | GDN | prefill attention | decode matrix |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gfx1030` | 92.62% | 0.545% | 2.19% | 1.01% | 2.61% |
| `gfx1201` | 60.56% | 3.09% | 9.74% | 4.02% | 13.07% |

llama.cpp Q6_Kのcanonical平均値との最終差は次のとおり。format差を含むため、未達倍率は次の候補順位付けにだけ使う。

| target | input | sLLM MXFP6 final | llama.cpp Q6_K | 残差 |
| --- | ---: | ---: | ---: | ---: |
| `gfx1030` | 512 | 411.82 tok/s | 2,077.47 tok/s | 5.04倍 |
| `gfx1030` | 2,048 | 393.60 tok/s | 2,061.67 tok/s | 5.24倍 |
| `gfx1201` | 512 | 2,843.97 tok/s | 3,850.23 tok/s | 1.35倍 |
| `gfx1201` | 2,048 | 2,959.30 tok/s | 3,828.11 tok/s | 1.29倍 |

`gfx1030`は引き続きmatrix本体が支配的で、より大きいM/N/K tile、packed ingress、LDS bank配置、dot再利用が次の候補になる。
`gfx1201`はmatrix残差が縮小し、GDN、attention、activation quantizerの合計寄与が相対的に増えた。ただしPhase 74では
Prefill MXFP6 matrix／quantizer以外へscopeを拡張していない。

## 検証と証拠

- `cargo fmt --all -- --check`: PASS。
- `cargo check -p sllm-hip --bin sllm-mxfp-wa-evidence`: PASS。
- `cargo test -p sllm-hip --bin sllm-mxfp-wa-evidence`: 14/14 PASS。
- `sllm_public_runtime_host_test`を含むhost CTest: 5/5 PASS。
- exact `gfx1030`／`gfx1201` codec GPU test: 両方PASS、全64 E3M2→E4M3、全64 E3M2→FP16、fallbackなし。
- operator: ID47／ID48、Loop 3 control／candidateとも独立FP32 oracle、tail、境界、repeat、nonfinite位置をPASS。
- full-model: 4B final、27B transferともHIP-only、fallbackなし、token、VRAM、cleanupをPASS。
- `git diff --check`: PASS。

主要なGit-excluded evidence rootは`/home/homelab1/.local/share/sllm-evidence/phase74`。
`loop1-operator`、`loop1-final`、`loop2-operator`、`loop2-final`、`loop3-operator`、`loop3-full-model`、
`final-default`、`final-27b`、`final-profile-gfx1030`、`final-profile-gfx1201`を保持した。

[全体計画](../../../../plans/main-plan.md) /
[保存済み計画](../../../../plans/archive/2026/09/1-10/phase74-mxfp6-prefill-llama-optimization-loop.md) /
[追跡要約](../../../../../ci/matrix/phase74-mxfp6-prefill-llama-optimization-loop-v1.json)
