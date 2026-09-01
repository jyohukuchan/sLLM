# Phase 62 再利用可能low-precision block codecとMXFP最適化 履歴

## 2026-08-31: 次Phaseへ計画

- ユーザー指示により、Phase 61の次をPhase 62とし、MXFP/NVFPの数値primitiveをmatmul固有実装から分離して
  KV append、attention、将来のMoE等でも再利用できる最適化基盤へ移す計画を作成した。
- 現行sourceではE4M3FN/E8M0等のscalar変換が`matmul_kernel.hip.cpp`、`kv_state_kernel.hip.cpp`、
  `causal_attention_kernel.hip.cpp`に重複し、block quantize、scale、packed I/Oもconsumer内へ埋め込まれていることを確認した。
- Phase 62はscalar codec、block policy/packed I/O、target trait/provider、typed `BlockScaledView`相当の層に分ける。
  runtime format分岐はdispatch境界へ置き、hot loopはcompile-time specializationとdevice inlineを使う。
- MXFP8/MXFP6 matmul、MXFP8 KV append/attentionを主要consumerとし、NVFP4は既存のE2M1 value、E4M3 block scale、
  FP32 outer scaleを維持した互換consumerとして共通primitiveを利用する。NVFP4の品質recipeや既定値変更は含めない。
- Phase開始時の共通化はbit exactを要求し、数値変更候補と性能だけの変更を混在させない。W/Aの実測は明示FP16 KV、
  KVの実測はBF16 weight＋standard OCP MXFP8 E4 KVで分離する。
- Phase 61のMXFP W/A品質残差はbit-exact最適化では改善しないため、Phase 62でproduction defaultへ昇格させない。
  reviewed scopeの省略時KV MXFP8 E4、明示FP16 rollback、block16経路廃止を維持する。
- persistentなFP32 attention/KV planeを追加せず、量子化済みactivationのmaterializeは複数consumerで再利用利益がある範囲、
  単一consumerはinline/fusion候補として個別採否する。
- この時点では計画文書だけを変更し、production source、GPU実行、モデル成果物、既定値、外部serviceは変更していない。

## 2026-08-31: 共通codec実装と両RDNA採用

- `native/hip/src/low_precision_block_codec.hpp`へscalar codec、MX/NV block policy、typed immutable/mutable view、
  wave amax/flag reduction、scale selection、packed load/storeを集約した。matmul、KV append、causal attentionの重複実装を削除し、
  NVFP4はE2M1 value＋E4M3 block scale＋FP32 outer scaleを維持して共通view/loadを使う。
- attentionはgenericだけでなくdecode wave、GQA shared、qtile、scaled long-prefillもencodingごとのkernel templateへ移し、
  format switchを起動境界へ限定した。gfx1201 E4はAMD native builtin、gfx1030はbit construction/software codecを選ぶ。
- 新しい直接GPU testを両targetで実行し、E4M3FN/FNUZ/E5M2/E8M0全256、E3M2全64、E2M1全16の計1,104 decode code、
  encodeのzero/subnormal/tie/max/Inf/NaN境界、MX `31/32/33/256`、NV `15/16/17/256`を独立host oracleへ照合してPASSした。
- W/A evidenceはMXFP8/MXFP6のM=`1/3/17`全6 hashがbeforeおよび両target間で一致した。KV evidenceはhead dim
  `31/32/33/255/256/257`のvalue/scale byte、tail、K/V独立scale、dim 256 direct attentionをPASSした。
  full-attention 29-caseも両targetでPASSし、代表M=1/17/64およびKV=8,193のhashはbeforeとbit exactだった。
- 固定Qwen3.5-4B、FP16 KV、17 input／4 outputのprefill before→afterはgfx1030 MXFP8
  `47.31→48.48`、MXFP6 `98.23→99.23 tok/s`、gfx1201 `36.67→72.87`、`32.72→115.30 tok/s`だった。
  3 input短形状ではgfx1030 `20.60→21.44`／`28.03→28.00`、gfx1201 `30.55→34.97`／`28.73→37.00 tok/s`だった。
  生成token列、HIP-only、fallback false、request/session cleanup 0を維持した。
- MXFP8 KV attention medianはgfx1030のM=1/17/64/KV8193が`29.80/156.92/615.85 us/5.248 ms`から
  `27.28/107.00/463.45 us/2.462 ms`、gfx1201が`12.12/36.00/133.36 us/3.515 ms`から
  `11.64/33.72/105.20 us/1.569 ms`へ短縮した。
- BF16 weightでKV形式だけを分離した17／4 caseは、FP16→MXFP8 E4 KVでgfx1030
  `255.77/44.45→261.99/44.42 tok/s`、gfx1201 `399.25/44.60→398.12/44.81 tok/s`だった。
- rocprofv3で量子化kernelはMXFP8 operator時間のgfx1030 `21.95%`、gfx1201 `34.05%`、MXFP6の`4.70%`／`5.65%`を占めた。
  量子化結果のcross-plan reuseはbuffer generation/liveness契約なしではstale readを防げず、単純fusionは出力N tileごとの再量子化を招く。
  追加のmaterialized cache/fusionは棄却し、bounded plan workspaceを維持した。
- workspace/HBMとdispatchは増加なし。release CLI code sizeはgfx1030 +1.54%、gfx1201 +1.79%で、起動境界templateの
  限定コストとしてattention短縮とともに採用した。
- 数値分類はN0で、W/A recipe/default、reviewed scopeのMXFP8 E4 KV default、明示FP16 rollback、block16廃止、public ABIを変更していない。
- `cargo fmt --all --check`、変更行のclang-format、`cargo test --workspace --all-targets`、両target release build、
  直接codec/W/A/KV/full-attention GPU evidenceをPASSし、Phase 62を完了して計画をarchiveへ移した。

## 2026-08-31: llama.cpp MXFP4 MMQ構造follow-up

- 固定llama.cpp `b10453`／`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`の`mmq.cuh`、
  `mmq-load-tiles.cuh`、RDNA2／RDNA4 configを追加参照に固定した。AMD経路のQ8_1 activation、E2M1→int8展開、
  DP4A／integer WMMA、Blackwell native FP4は対象外とし、multi-M×multi-N×K tileとformat/load/dot分離だけを参考にした。
- sLLMのpacked MX値、E8M0 scale、FP32 accumulator、row-8 K走査／wave reductionを保ったまま、activationを4／8 N列へ
  再利用するMXFP8／MXFP6 `mmq-col4/col8-v4`を実装した。直接のsource expressionは流用していない。
- M=`17`, K=`2560`, N=`9216`の5回中央値で、col8はMXFP8をgfx1030 `8.924→2.988 ms`、gfx1201
  `1.608→0.601 ms`へ短縮した。MXFP6は`3.525→3.628 ms`／`1.096→0.876 ms`だった。N=`32`では逆に
  MXFP8が悪化し、MXFP6が短縮した。全output hashは現行defaultと一致し、CPU oracle、dispatch、fallbackなし、cleanup 0をPASSした。
- 固定Qwen3.5-4B、FP16 KV、17 input／4 outputでは、col8のprefill中央値がMXFP8でgfx1030
  `48.09→114.99 tok/s`、gfx1201 `73.06→157.85 tok/s`、MXFP6で`100.19→109.88 tok/s`／
  `116.53→131.91 tok/s`となった。decode差は0.7%以内で、生成token列とcleanup contractは一致した。
- 構造は有効だがformat／Nで勝敗が逆転するため、col4/col8は`SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS=4|8`の
  explicit benchmark-onlyとした。既定provider、W/A recipe、public ABI、KV defaultは変更していない。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/archive/2026/08/21-31/phase62-reusable-low-precision-block-optimization.md) /
[Phase 37以降のロードマップ](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
