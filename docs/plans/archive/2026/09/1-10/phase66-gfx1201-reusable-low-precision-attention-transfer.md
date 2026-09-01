# Phase 66: gfx1201 reusable low-precision providerとattention移植

## 状態

`完了・ID37限定採用／attention棄却`。2026-09-01のユーザー指示により、Phase 65後のMXFP8残差を共通providerとして改善し、
MXFP8での実証後にMXFP6、NVFP4／MXFP4、weight型非依存attentionへ実候補を移植した。

## 目的

exact `gfx1201`の大規模prefillで、モデル名に依存しないlow-precision matmul provider familyとcausal attention providerを
確立する。最初にOCP MXFP8 E4M3 W8A8で数値・性能・資源を実証し、その実装を単一形式専用kernelで閉じず、
architecture policy、format codec、activation packer、tile policy、format固有inner productへ分離する。

続いて同じprovider境界へMXFP6 E3M2 W6A6とNVFP4を実際に接続し、MXFP4は既存の対応済みexecution範囲で候補を作る。
causal attentionはweight形式から独立した共通経路としてMXFP8だけでなくBF16と、実行可能なNVFP4／MXFP4 modelで確認する。
移植候補は必ずoperatorで実行・計測して採否を決めるが、全形式で同じtileが勝つことやproduction採用を完了条件にはしない。

## 固定baselineと原因分解

- primaryはAMD Radeon AI PRO R9700、exact `gfx1201`、ROCm 7.14.0、Code Object V6、wave32、単一GPU、単一request、
  MTPなし、明示FP16 KV、2,048-token one-chunk prefillとする。
- MXFP8 baselineはPhase 65最終CLI
  `sha256:d4472be3d5faff90af4a68256f165a690e9b302e97fee581ab8f554c03e3dffe`、Qwen3.5-4B artifact
  `sha256:f253d9f47603d84718b4fdb898b434e493d732b52838ba9abfdfafe73a5d076f`とする。通常測定の2,048-token中央値は
  4B `3,053.502 tok/s`、9B `1,761.989 tok/s`である。
- 固定llama.cppは`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`、Q8_0 weight／runtime Q8_1 activation、FP16 KV、
  Flash Attention有効とする。2,048-token通常測定の中央値は`6,516.930 tok/s`で、MXFP8とQ8は数値形式が異なるため
  system-equivalentな診断値でありstrict同形式比較ではない。
- pure prefillのrocprofv3 kernel-duration sumを1評価へ正規化すると、sLLM MXFP8／llama.cpp Q8は全kernel
  `662.345 / 296.591 ms`、low-precision linear `438.802 / 149.560 ms`、causal attention本体
  `107.968 / 26.225 ms`、GDN recurrent本体`65.903 / 56.472 ms`だった。両engineは1評価あたり248回の
  activation量子化とlinearを実行し、linear差は全kernel絶対時間差の79.08%を占める。GDN周辺一式は概ね同等であり、
  Phase 66ではmodel固有GDN微調整より共通linearとattentionを優先する。
- llama.cppのQ8演算、Q8_1 layout、式、source表現をMXFPへ転記しない。比較から得る入力は、target／format／shape別provider、
  consumer向けactivation配置、複数tile familyという抽象的な設計点に限定する。直接reuseを行う場合は別途MIT provenanceを
  file単位で記録するが、本PhaseのMXFP実装はOCP形式とsLLM独立oracleを根拠にできる。

## 固定acceptance

### 1. 共通provider契約

- provider selectorはexact targetとcapability、weight／activation encoding、block policy、layout／alignment、M/N/K、
  accumulation／output dtypeだけを受け取る。model名、layer番号、prompt、token ID、benchmark case名、測定後の性能値をkeyにしない。
- 共通境界を`architecture policy`、`format codec`、`activation packer`、`tile policy`、`format-specific inner product`へ分ける。
  prepared planでproviderとworkspaceを固定し、hot loopの動的format判定やmodel固有dispatchを追加しない。
- Phase 62のscalar/block codec、typed block-scaled view、既存prepared selectorを拡張し、同じsemanticを持つ機構を別の
  MXFP8専用APIとして重複させない。一方、block 32のMXFP、block 16とtensor scaleを持つNVFP4、packed nibbleのMXFP4の
  形式差はtrait／specialization内に保持する。
- whole-layer／whole-modelのBF16・FP32 weight展開、persistent FP32 attention／KV plane、request間の無検証activation cacheを
  追加しない。追加workspaceはrequest arena内でboundedにし、終了後に解放する。

### 2. MXFP8での実証

- Phase 65の`128x64x32` direct-bothをcontrolとし、N方向の再利用を増やすwide tile、K方向の複数block処理、
  providerが直接消費するactivation layout、small-N／非整列fallbackを独立候補として比較する。
- 候補値はsLLMのM/N/K sweep、ISA/resource、同期済みoperator timingから決める。固定llama.cppのQ8 tile表をそのまま
  MXFP8の初期値やselector表へ移さない。
- main aligned family、N=`64/128/256/512/1024`、wide／down projection、K/N tail、LM headを分離し、単一tileへ強制しない。
  adoption scopeは静的dispatch keyでshared／scoped／benchmark-only／rejectedのいずれかに分類する。
- OCP E4M3 value、E8M0 scale、block 32、FP32 accumulation、BF16 RNE output、NaN scale伝播、Inf saturation、
  K非32倍のfail-closeを維持する。演算treeや丸めstageが変わる場合は数値変更台帳のN0〜N3へ分類し、N2はユーザー判断前に
  production採用せず、N3は採用しない。

### 3. 共通causal attention

- Phase 65 pure-prefillで1評価`107.968 ms`だったQwen GQA causal attention本体を、weight形式から独立したsemantic opとして扱う。
  Q/K/V／KV encoding、head dim、GQA ratio、query tile、context境界から選ぶproviderとし、model名をselectorへ使わない。
- 既存FP16 KVとstandard OCP MXFP8 E4 KVの数値・layout差を維持し、attention高速化のためにFP32 attention／KVを常駐保存しない。
- MXFP8 Qwen3.5-4B／9Bで候補を絞った後、同じeligible shapeのBF16 weight経路で実dispatchを確認する。
  NVFP4／MXFP4 modelはshapeがeligibleならcandidateを実行し、非eligibleなら共通selectorによる既存provider非選択を記録する。

### 4. 他形式への実移植

- **MXFP6**: block 32／E8M0を共有する最初の移植先とし、E3M2 codecとinner productだけを形式固有に保つ。
  MXFP8で採用または有望と判断したactivation pack／tile policyを少なくとも一つ実kernelへ接続し、operator oracleと性能を取得する。
- **NVFP4**: block 16、packed 4-bit value、tensor scaleを維持した別specializationとして、既存gfx1201 execution routeへ
  共通provider候補を接続する。tensor scaleの欠落やMX block 32への誤解釈をfail-closeし、既存Gemma NVFP4 routeで
  model-levelのdispatch、速度、resident、cleanupを確認する。
- **MXFP4**: 対応済みのprovided-model／operator範囲で同じprovider境界へ候補を接続する。Phase開始時点でproduction model routeが
  対象shapeを持たない場合も、model-free operator candidate、selector非選択、未対応理由までを記録し、NVFP4と同一semanticだと
  仮定しない。
- 各移植はsource上の型追加やcompileだけで完了扱いにせず、exact gfx1201で少なくとも一つのnontrivial M>1 operatorを
  数値oracleへ照合し、baseline/candidateを同期計測して採否を残す。候補が遅ければ既存providerを維持して完了できる。

### 5. correctness・性能・資源

- MXFP8の主な境界はM=`1/3/17/127/128/129/255/256/257/511/512/513/2047/2048/2049`、
  format境界K=`31/32/33`、selectorが定めたN/K境界の`B-1/B/B+1`から必要な代表集合をmanifestへ固定する。
  zero、subnormal、tie、最大有限値、Inf、NaN、非整列値を含める。
- model-free operatorは独立dequantized oracle、最大absolute／relative error、非有限位置、repeat digest、HIP-only、fallback、cleanupを記録する。
  compile-only、CPU fallback、timeout、crash、0 caseをGPU PASSにしない。
- MXFP8 full-modelはQwen3.5-4B／9B、input 512／1,024／2,048／4,096、明示FP16 KVを対象とする。
  BF16 controlと固定llama.cpp Q8は性能位置の診断として併記するが、形式差を隠してstrict比較とは呼ばない。
- 移植先はMXFP6 Qwen3.5とNVFP4の既存reviewed modelを代表に、同じsource identity内でbaseline/candidateを比較する。
  format間でartifact、model、演算量が異なる値から横断的な倍率を主張しない。
- profilerはactivation量子化、main aligned matrix、small-N／fallback matrix、causal attention、GDN、elementwise、host/launch gapへ分ける。
  MXFP8のlinear、attention、全kernel絶対時間とwall prefillの両方を採否へ用いる。
- draftは1 warmup＋3 measured、最終採用候補は3 warmup＋10 measuredを基本とし、全反復値、median、MAD、artifact／binary、
  model resident、workspace／peak、dispatch count、fallback、終了後のHBM/GTT復帰を記録する。固定改善率やllama.cpp同等をhard gateにしない。

## 作業単位

1. **P66-A0 baseline固定**: Phase 65 MXFP8、同一BF16、固定llama.cpp Q8の2,048 pure-prefillを再現し、
   248 linearのshape列、kernel分類、binary／artifact／software tupleを追跡要約へ固定する。
2. **P66-A1 provider contract**: 共通selector、format trait、activation pack、tile policy、workspace／liveness契約を既存
   low-precision block codecとprepared executionへ接続し、model非依存host testを追加する。
3. **P66-A2 MXFP8 matrix候補**: wide-N、deep-K、consumer layout、small-N／tailを独立候補として実装し、
   operator oracle、ISA/resource、shape sweepで候補を絞る。
4. **P66-A3 common attention候補**: gfx1201 GQA prefillのtile／load／reduction候補をoperatorで比較し、FP16 KVと
   standard MXFP8 E4 KVのselector／数値契約を確認する。
5. **P66-A4 MXFP8実証**: 4B／9B full-model、品質、profile、VRAM、cleanupを取得し、matrixとattentionのadoption scopeを固定する。
6. **P66-A5 MXFP6移植**: 共通activation pack／tile policyをE3M2 W6A6へ実kernelとして移し、Qwen operator／full-modelで採否する。
7. **P66-A6 NVFP4／MXFP4・BF16 attention移植**: NVFP4と対応済みMXFP4範囲へformat specializationを接続し、
   BF16を含むweight型非依存attention dispatchを実モデルまたは明示非選択で検証する。
8. **P66-A7 integration／closeout**: focused host/GPU test、ABI、numerical ledger、runtime／compatibility文書、main plan、
   追跡済み要約とmatching historyを同期し、完了後に本計画をarchiveへ移す。

## 完了条件

- exact gfx1201のMXFP8で、共通provider境界を使うmatrix候補とattention候補を実装し、operator、4B／9B full-model、
  profile、数値、資源の採否を完了する。
- 同じprovider境界からMXFP6とNVFP4へ少なくとも一つずつ実kernel candidateを移植し、exact gfx1201の数値oracleと同期性能測定を
  完了する。MXFP4は対応済み範囲でcandidate実行または明示的な非選択理由を固定する。
- attention候補をMXFP8だけへ結合せず、BF16 weight経路で同じsemantic selectorによる実dispatchまたはshape非選択を確認する。
- production採用したselectorはmodel名を使わず、範囲外format／shape／targetを既存providerへ戻す。候補不採用でも、実移植、測定、
  原因、再検討条件が記録されていればPhaseを完了できる。
- persistent BF16/FP32 weight展開、persistent FP32 attention/KV、無検証cross-request cache、fallbackによる見かけのPASSを導入しない。

## 対象外

- exact `gfx1030`／`gfx942`向けの新しいnative matrix／attention kernel。共通source変更によるcompile／selector回帰は行うが、
  Phase 66のprimary実機最適化はexact `gfx1201`に限定する。
- M=1 decodeの全面再最適化、MoE grouped GEMM、continuous batching、複数GPU、HIP Graph全面導入。
- MXFP8／MXFP6／NVFP4／MXFP4の量子化recipe、GGUF encoding、model default、standard MXFP8 E4 KV default、
  FP16 rollback、block16廃止方針の変更。
- Q8_0／Q4_K等の一般的なllama.cpp量子化形式をsLLMの製品入力として追加すること。
- llama.cpp Q8の絶対速度を、異形式のMXFP8に対する一律必達値または達成可能上限として扱うこと。

## 完了時に残す記録

- 共通provider／selector／workspace／activation layout契約と、形式固有specializationの境界。
- MXFP8、MXFP6、NVFP4、対応済みMXFP4、BF16 attentionのimplemented／adopted／scoped／rejected一覧。
- operator/full-modelの全反復、median/MAD、kernel分類、ISA/resource、数値分類、VRAM、fallback、cleanup。
- model共通であることを示す4B／9Bまたは異modelの実dispatchと、shape非eligible時の明示rollback。
- llama.cpp比較から採用した抽象所見、移植しなかったQ8固有要素、provenance境界、残存ボトルネックと次の再検討条件。

## 完了結果

- prepare時にformat/block、activation pack、tile、inner productと具体kernel variantをfreezeするmodel非依存providerを
  MXFP8／MXFP6／NVFP4／MXFP4へ接続した。request workspace arena high-waterは`1,080,836,096` byte、allocator auditは
  process drop後0。persistent BF16/FP32 weight展開、FP32 attention/KV plane、cross-request cacheは追加していない。
- exact `gfx1201`のMXFP8 ID37 N128 direct-bothをPhase 65 familyかつN%128=0へ限定採用した。final operatorのwide/downは
  ID36→37で`181,641→155,402 ns`（-14.45%）／`398,403→373,404 ns`（-6.27%）。special-value E4M3/E8M0、
  signed zero、Inf/NaNもnonfinite `4/4`、mismatch 0でPASSした。
- FP16／MXFP8 E4 KVのtyped q4k4／q4k8／q8k8 attentionはbit一致したが、primary同期行が4.3〜27.3%遅く棄却した。
  selectorは`sliding_window`と明示`score_scale`もfail-close keyに含める。reviewed Gemma q16/kv8とQwen3.5 MoE MXFP4
  q16/kv2は明示的typed candidate非選択、BF16 weightのeligible shapeは実dispatchをPASSした。
- final 3+10 prefill中央値は4B 512/1,024/2,048/4,096が
  `3,840.804836/3,806.640973/3,767.237995/3,249.069405 tok/s`、9Bが
  `1,988.722356/2,231.573186/2,261.647647/2,069.842794 tok/s`。全runはHIP-only、fallback false、cleanup 0だった。
- MXFP6はfull-shapeに必要なtiled16を維持しsmall-N selectorを後続候補、NVFP4／MXFP4 W4A4は既存device kernelへの
  frozen routing採用とした。MXFP4 full MoE productionはscope外、第三者code reuseは0である。詳細値とimmutable identityは
  [Phase 66履歴](../../../../../history/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md)と
  [追跡要約](../../../../../../ci/matrix/phase66-gfx1201-low-precision-provider-summary-v1.json)を正本とする。

[全体計画](../../../../main-plan.md) /
[Phase 37以降のロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[matching履歴](../../../../../history/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md)
