# Phase 62: 再利用可能low-precision block codecとMXFP最適化

## 状態

`完了・共通採用／MMQ follow-up評価済み`。2026-08-31に実装、両local RDNA実機検証、実モデル比較、文書同期まで完了した。
同日のllama.cpp MXFP4 MMQ follow-upではmulti-column候補を実装・評価し、形状別の改善と逆転を確認したため
explicit benchmark-onlyとした。数値recipe、public ABI、MXFP W/Aの既定値、MXFP8 E4 KVの既定値、明示FP16 rollbackは変更していない。

## 目的

Phase 61で成立したOCP MXFP8 E4M3 W8A8／MXFP6 E3M2 W6A6と、既存のMXFP8／NVFP4 KV経路に重複する
scalar encode/decode、block scale選択、packed I/Oを、modelとconsumer opに依存しない再利用可能なHIP primitiveへ分離する。
同じ数値形式の最適化をmatmul、KV append、attention、将来のMoE等へ一度の実装で横展開できる構造へ移し、
exact `gfx1030`／`gfx1201`のtarget別providerをその共通境界の内側で最適化する。

このPhaseの主目的は性能と保守性である。Phase 61のMXFP W/A品質を変更せず、reviewed Qwen3.5-4B BF16 dense textの
省略時KVであるstandard OCP MXFP8 E4と、明示FP16 rollbackを維持する。

## 固定acceptance

### 共通数値primitive

- E4M3FN、E4M3FNUZ、E5M2、E3M2、E2M1、E8M0のscalar codecを、memory layoutやconsumer opから独立した
  `__device__ __forceinline__`の単一正本へ抽出する。NaN、signed zero、subnormal、roundTiesToEven、最大有限値への
  saturation等の既存format semanticを変えない。
- OCP MX block 32とNVFP block 16について、amax、scale選択、quantize/dequantize、packed load/store、scale broadcastを
  format policyとして表現する。MXFP8 E4、MXFP8 E5、MXFP6 E3、NVFP4のblock size、scale形式、packing、tensor scale有無を
  曖昧な共通enumへ潰さず、compile-time specializationで区別する。
- hot loop内で要素ごとのruntime format分岐を行わない。runtime descriptorはdispatch境界でcompile-time specializationへ
  解決し、gfx固有命令またはsoftware変換はtarget trait/providerの内側へ閉じ込める。
- value plane、block-scale plane、任意のouter/tensor scale、logical shape、stride、physical variantを表す内部の
  `BlockScaledView`相当を設ける。既存allocationのoffset viewで表現できる場合は不要な再配置や複製を行わない。

### consumer統合

- Phase 61のMXFP8／MXFP6 matmul、MXFP8 KV append、causal attentionのMXFP8読出しを共通primitiveのconsumerへ移す。
- NVFP4はE2M1 value、E4M3 block scale、FP32 outer scaleという形式差を保持したまま、意味が一致するscalar codecと
  packed block I/Oを共通primitiveから利用する。Phase 62はNVFP4の品質recipeや既定値を変更するPhaseではない。
- standaloneの`QuantizeBlockScaled`相当は、Q/K/V、gate/up等で同じ量子化済みactivationを複数consumerが再利用し、
  追加write/readと生存期間を含めても利益がある範囲に限ってmaterializeする。単一consumerではdevice helperのinlineまたは
  quantize+consumer fusionを候補とし、汎用kernel起動を無条件に追加しない。
- matmulのtile/MFMA/reduction、attentionのsoftmax、KVのtoken-major配置等、演算固有部分は各providerに残す。
  共通化したことを理由に一つの万能kernelへ統合しない。

### 数値・互換性

- 共通化だけを行う段階では、既存sourceと同じ入力に対するvalue/scale byte、matmul BF16 output、KV append byte、
  attention BF16 outputをbit exactに維持する。差が出た場合は最適化として処理せず、原因を解消するまで採用しない。
- scalar decodeはE4M3FN/E4M3FNUZ/E5M2/E8M0の全256 code、E3M2の全64 code、E2M1の全16 codeをhost oracleへ
  照合する。encodeは±0、subnormal境界、tie、最大有限値前後、Inf、NaNを含める。
- block oracleはMX block境界`31/32/33`、NV block境界`15/16/17`、production head dimension 256、
  matmulのM=`1/3/17`を含める。Phase 61契約どおりmatmul K非32倍、scale欠落、未対応targetはfallbackせず拒否する。
- block16製品経路を復活させず、予約ABIと履歴を再利用しない。public encoding名、GGUF recipe、既存KV physical layoutを
  数値最適化の都合で黙って変更しない。

### 性能・資源・採用

- Phase開始時に、Phase 61と同じ固定Qwen3.5-4B BF16 sourceから作成したMXFP8／MXFP6 artifact、明示FP16 KV、
  exact `gfx1030`／`gfx1201`をbaseline identityへ固定する。KV側はBF16 weight＋standard OCP MXFP8 E4 KVで別に測り、
  W/A改善とKV改善を混同しない。
- model-freeではscalar/block quantizer、M=`1/3/17` matmul、KV append、短いKVと長いKVのattentionをbefore/afterで測る。
  full modelでは短いprefill/decodeと、量子化・packed I/Oの比率が現れる代表prefillを各targetで測る。
- 固定の改善率を完了条件にしない。共通採用、exact-target限定採用、棄却をcandidateごとに分け、演算子とmodel全体の絶対時間、
  一貫性、保守費用、workspace、dispatch、将来再利用性をmain planの`adoption scope S`規則で判断する。
- persistentなFP32 attention/KV planeは追加しない。materialized activationはrequest arenaでboundedに管理し、
  同時liveでないconsumer間で再利用する。追加dispatch、workspace/HBM peak、fallback、終了後の資源復帰を採否に含める。
- Phase 61のMXFP8／MXFP6 W/Aはbit exactな性能最適化だけでは品質残差が変わらないため、このPhaseではproduction defaultへ
  昇格させない。数値recipeを変更する候補が必要になった場合はN1/N2/N3へ分類し、bit-exact性能laneとは別に採否する。

### llama.cpp MXFP4 MMQ follow-up

- 追加の実装参照を、固定llama.cpp tag `b10453`、commit
  `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`の`ggml/src/ggml-cuda/mmq.cuh`、
  `mmq-load-tiles.cuh`、`mmq-config-rdna2.cuh`、`mmq-config-rdna4.cuh`へ固定する。調査したblobは順に
  `2eb15fdfad93be4d842f9e52eac56f2587c7b2ce`、`8ed704c281a4df00184d311148660e26ca396bcf`、
  `8324d9e1a830e26a594f439c2867442a41669c07`、`9293d9d558856041b4a82285493d6ef191e9f7b8`である。
- llama.cppのAMD MXFP4 MMQは、FP32 activationをQ8_1へ量子化し、E2M1 weightをtile-local int8へ展開して
  DP4A／integer WMMAへ送るW4A8経路である。Q8_1 activation、E2M1→int8 lookup、integer dot/MMA、Blackwell限定の
  native FP4 MMAはsLLMのMXFP8 W8A8／MXFP6 W6A6へ流用しない。一般的なllama.cpp INT4/INT8+scale形式を
  製品対応へ追加する決定でもない。
- 参照する技術要点は、format固有tile load・dot・write-backの分離、複数M×複数Nのoutput tile、K tileごとの
  packed value／block scale共有、edge fallback、RDNA2／RDNA4別のcompile-time tile設定である。sLLM側は既存の
  `ScalarCodec`／`BlockCodec`、FP32 accumulator、BF16 RNE output、OCP value/scale planeを維持する。
- 最初の候補は既存row-8の各wave/laneによるK走査と32-lane reduction順を変えず、一つのworkgroupが4または8個の
  N列を同時に計算するmulti-column tileとする。同じactivation value／scaleを複数N列へ再利用し、weight value／scaleは
  8 M行で共有する。追加persistent workspace、activationの再materialize、INT8 intermediateは作らない。
- model-freeではM=`3/17`、production projection K/Nを含むshapeで現行row-8／tiled-16とmulti-column候補を同じ
  target別release artifactから比較する。境界NとM tailを数値oracleへ照合し、候補ごとにexact `gfx1030`／`gfx1201`の
  median、output hash、dispatch、workspaceを記録する。安定したdispatch key上で改善しない候補は既定化せず削除または
  explicit benchmark-onlyに留める。

## 完了結果

- `low_precision_block_codec.hpp`を単一正本とし、E4M3FN/FNUZ、E5M2、E3M2、E2M1、E8M0、MX block 32、
  NV block 16、typed immutable/mutable block-scaled viewをdevice-inline primitiveへ分離した。matmul、KV append、
  causal attention、NVFP4の意味が一致するread/writeを移行し、consumer固有tile、softmax、token-major layoutは各kernelへ残した。
- runtime encodingはcausal attentionのgeneric、decode wave、GQA shared、qtile、scaled long-prefillでkernel起動境界の
  compile-time specializationへ解決する。exact gfx1201はnative AMD E4変換、gfx1030はbit構築／software変換を同じcodec境界内で使う。
- 直接GPU codec testは両targetでdecode 1,104 code、encode境界5形式、MX `31/32/33/256`、NV `15/16/17/256`を
  host oracleへ照合してPASSした。W/AのM=`1/3/17`全6 output hash、KV append byte、attention output hashはbeforeとbit exactで、
  fallbackなし、cleanup 0だった。これは数値変更なしのN0である。
- 固定Qwen3.5-4B、17 input／4 output、明示FP16 KVのbefore→after prefill中央値は、gfx1030でMXFP8
  `47.31→48.48`、MXFP6 `98.23→99.23 tok/s`、gfx1201で`36.67→72.87`、`32.72→115.30 tok/s`だった。
  3 input短形状はgfx1030 MXFP8 `20.60→21.44`、MXFP6 `28.03→28.00`、gfx1201 `30.55→34.97`、
  `28.73→37.00 tok/s`である。全試行の生成token列、HIP-only、fallback、cleanup contractを維持した。
- MXFP8 KV attentionのbefore→after medianはgfx1030でM=1 `29.80→27.28 us`、M=17 `156.92→107.00 us`、
  M=64 `615.85→463.45 us`、KV=8,193 decode `5.248→2.462 ms`、gfx1201で`12.12→11.64 us`、
  `36.00→33.72 us`、`133.36→105.20 us`、`3.515→1.569 ms`だった。4形状のoutput hashは両targetでbeforeと一致した。
- KV比較をBF16 weightで分離した17／4短caseでは、FP16→MXFP8 E4 KVのprefill／decodeがgfx1030
  `255.77/44.45→261.99/44.42 tok/s`、gfx1201 `399.25/44.60→398.12/44.81 tok/s`だった。
  21-token時のcommitted page byteはVMM page粒度でMXFP8のscale planeが別pageを持つため論理圧縮率を表さず、採否根拠にしない。
- profiler上のactivation quantizer比率はMXFP8でgfx1030 `21.95%`、gfx1201 `34.05%`、MXFP6で`4.70%`／`5.65%`だった。
  ただし単純fusionはN方向のworkgroupごとに量子化を再計算し、cross-plan reuseはinput bufferのgeneration/liveness identityなしでは
  stale readを防げない。従って追加workspace cacheと単純fusionは棄却し、現行のplan-local bounded workspaceを維持した。
- plan workspace/HBM allocationとdispatch数は増やしていない。compile-time specializationによりrelease CLIはgfx1030で
  `20,920,232→21,241,712 byte`（+1.54%）、gfx1201で`20,974,072→21,349,032 byte`（+1.79%）となり、
  このcode-size増を両targetのattention短縮とformat switch除去に対する限定的な保守費用として採用した。
- 固定llama.cpp MXFP4 MMQからQ8_1／int8 dotを移さず、multi-M×multi-N×K tileという構造だけを参考にした。既存row-8の
  K走査、FP32 accumulator、32-lane reductionを維持し、activationを4／8 N列へ再利用する`mmq-col4/col8-v4`を追加した。
  直接のsource expressionは流用しておらず、固定commit／blobから得た構造上の事実を参照したconcept-only実装である。
- M=`17`, K=`2560`, N=`9216`の5回中央値では、col8がMXFP8をgfx1030 `8.924→2.988 ms`、gfx1201
  `1.608→0.601 ms`へ短縮した。MXFP6はgfx1030 `3.525→3.628 ms`、gfx1201 `1.096→0.876 ms`だった。一方N=`32`では
  MXFP8がgfx1030 `0.122→0.199 ms`、gfx1201 `0.031→0.044 ms`へ悪化し、MXFP6は`0.534→0.190 ms`／
  `0.237→0.062 ms`へ短縮した。全output hashはdefaultと一致し、相対誤差上限、fallbackなし、cleanup 0を維持した。
- 固定Qwen3.5-4B、FP16 KV、17 input／4 outputの1 warmup＋3 measured中央値は、col8によってMXFP8 prefillが
  gfx1030 `48.09→114.99 tok/s`、gfx1201 `73.06→157.85 tok/s`、MXFP6が`100.19→109.88 tok/s`／
  `116.53→131.91 tok/s`となった。decode差は0.7%以内で、全sampleの生成token列、HIP-only、fallbackなし、cleanup 0が一致した。
  ただしformat／Nでoperatorの勝敗が逆転し、固定一model短caseだけでは安全なshape selectorを固定できないため、col4/col8は
  `SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS=4|8`のexplicit benchmark-onlyに留め、既定providerは変更しない。

## 実装構造

候補となる構造は次のとおり。正確なfile分割は実装時に調整できるが、依存方向とcompile-time specializationは維持する。

```text
scalar codec
  E4M3FN / E4M3FNUZ / E5M2 / E3M2 / E2M1 / E8M0
        ↓
block policy + packed I/O + target traits
  MX block32 / NV block16 / gfx1030 / gfx1201
        ↓
BlockScaledView + optional QuantizeBlockScaled semantic op
        ↓
matmul     KV append     attention     future MoE/other consumers
```

想定するdevice境界は次の形で、format enumを内側のloadごとに判定しない。

```cpp
template<class Format, class Target>
struct ScalarCodec;

template<class BlockFormat, class Target>
struct BlockCodec {
  __device__ static EncodedBlock quantize(...);
  __device__ static float load(const BlockScaledView&, uint64_t row,
                               uint32_t element);
};
```

## 作業単位

1. **baseline固定**: 現行codec重複、consumer、dispatch、Phase 61 artifact、operator/full-model時間、workspaceを記録する。
2. **scalar codec抽出**: host oracleと全code/boundary testを先に固定し、matmul、KV、attentionの重複実装をbit exactに置換する。
3. **block codecとtyped view**: MX/NVのlayout policy、scale、packed I/O、offset viewを分離し、既存ABI/layoutを維持してconsumerを移行する。
4. **target別最適化**: gfx1201 native E4 encode/decode・vector I/Oと、gfx1030 bit構築・wave scale共有・packed I/Oを共通trait内で比較する。
5. **量子化再利用/fusion**: Q/K/V、gate/up等の共有候補と、単一consumerのfusion候補をprofileし、liveness/arena込みで個別採否する。
6. **統合採否**: exact gfx1030/gfx1201のmodel-freeと固定Qwen3.5-4Bを取得し、shared/scoped/rejectedを記録する。
7. **closeout**: 数値台帳、runtime/compatibility文書、main plan、matching historyを同期し、完了時にこのplanをarchiveへ移す。

## 対象外

- 新しい数値形式、GGUF type番号、block16製品経路、TurboQuant、MXFP4 KVの追加。
- MXFP8／MXFP6の量子化recipeや品質改善、imatrix再設計、production default化。
- full attention algorithmそのもののFlashAttention化、matmul全shapeの新規MFMA設計、MoE grouped GEMM全体。
- exact `gfx942`実機採否、新hardware/model family、multi-GPU、release packaging。

## 完了時の記録

- 共通化したsemanticと、演算固有のまま残した部分。
- candidateごとのtarget、shape/context範囲、before/after、採否、fallback範囲、workspace/HBM、既知制約。
- bit-exact確認または数値変更分類、GPU target、build/artifact/model identity、cleanup結果。
- 次に低精度形式を追加するときに実装すべき最小interfaceと、未共通化部分の理由。

[全体計画](../../../../main-plan.md) /
[Phase 37以降のロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[対応する履歴](../../../../../history/2026/08/21-31/phase62-reusable-low-precision-block-optimization.md)
