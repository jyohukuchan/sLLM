# Phase 10: model本体FP8 W8A8

> 状態: planned
> 作成日: 2026-08-14

## 目的

Phase 9で固定したdtype非依存のprepared semantic plan、completion segment、provider registryを再利用し、
Qwen3.5 model本体のFP8 W8A8を実装する。RDNA4 `gfx1201`ではOCP E4M3のnative FP8 fast path、
RDNA2 `gfx1030`ではnative FP8 matrix命令がないことを明示したsoftware conversion/emulation pathを扱う。
両者を同じ「FP8対応」とだけ表記せず、model storage、activation quantization、execution provider、
accumulation/output dtypeをdispatch metadataへ分けて残す。

Phase 10はlocal R9700/V620で完結させる。CDNA3のFNUZ変換と`gfx942` kernel/providerはPhase 11、
MI300X実行証拠はPhase 12で扱う。

## 固定する数値・model契約

- 開発・回帰の正本は既存のverified `Qwen/Qwen3.5-4B` BF16 lockから作る派生FP8 lockとする。
- 派生物はblock size 128のfine-grained E4M3FN weightとscaleを基本候補とし、実装開始時にtensor別の
  axis、scale dtype、zero-pointなし、saturation、特殊値規則をfixtureとともに固定する。
- 変換元lock fingerprint、変換toolのrepository/commit、環境、引数、tensor別出力hashをmodel lockへ記録する。
  第三者公開の4B FP8 checkpointを正しさの基準にはしない。
- activationはlinear op入力で動的にscaleを求めE4M3へ量子化し、weight/activationはFP8、accumulationは
  FP32、linear出力はBF16を基本契約とする。RMSNorm、softmax、RoPE、GDN state、KV cache、samplingは
  Phase 10では既存BF16/FP16/FP32契約を維持する。
- 公式Qwen3.5 FP8 modelは現在27B以上が中心である。27B FP8はPhase 12の追加interop spotとし、4Bの
  実装・受入をblockしない。

## target別production path

| target | weight/activation | provider | 表記 |
| --- | --- | --- | --- |
| RDNA4 exact `gfx1201` | OCP E4M3 / OCP E4M3 | hipBLASLtまたは検証済みcustom kernel | native FP8 W8A8 |
| RDNA2 exact `gfx1030` | OCP E4M3 / OCP E4M3 | byte decodeを含むcustom kernel | FP8 W8A8 emulation |
| RDNA2 unsupported shape | FP8 modelをload時BF16へ変換 / BF16 | 既存BF16 provider | explicit BF16 conversion path |

RDNA2のemulationをnative FP8と呼ばない。emulationが同じshapeのBF16 pathより遅い場合も正しさの対応は残すが、
defaultに昇格しない。load時BF16変換は黙ったfallbackではなく、model load/prepare時に選び、resident dtypeと
追加VRAMを診断へ出す。

## スコープ外

- CDNA3 FNUZ、MI300X実行、MI300A/MI325X。
- KV cache FP8/NVFP4、Weight NVFP4、load時のBF16→FP8自動量子化。
- FP8 FlashAttention 4-like、vision、MTP、MoE、multi-request、multi-GPU。
- 公式27B FP8の完全なproduct support。Phase 12ではload/限定generation spotだけを候補とする。
- FP8を理由にPhase 9の未着手graph/MLP fusionを同時実装すること。

## 受入条件

1. dtypeとencodingを分離し、OCP E4M3FN、将来のFNUZ、scale granularity、resident representationを
   public/internal descriptorで区別できる。
2. converterとloaderは非整列tensor、block境界127/128/129、正負zero、subnormal、finite最大値、NaN/Inf、
   saturationを独立NumPy oracleで検査する。raw byte reinterpretでCDNA3へ渡せる設計にしない。
3. RDNA4 providerはexact `gfx1201`、runtime capability、hipBLASLt solution support、shape/alignment/workspaceを
   すべて満たした場合だけ選ぶ。実行時失敗をBF16成功へ読み替えない。
4. RDNA2はW8A8 emulationとBF16 conversionを別providerとして監査できる。どちらもnative FP8と表示しない。
5. linear opのslice differential、4B logits/top-1/KLD、fixed/Unicode/stop generationをBF16基準と比較する。
   toleranceはtensor/opと入力範囲ごとにA0で固定し、単一の緩い全体閾値を置かない。
6. model-resident packed weight/scale/workspaceをrequest間で再利用し、requestごとの全weight変換やrepackを行わない。
7. exact target、selected provider、FP8 encoding、scale contract、fallbackなし、ECC/health、process/VRAM cleanupを
   reportへ残す。
8. performanceは4B short-odd、32/32、prefill-long、decode-longの影響caseを測り、BF16 Phase 9と固定llama.cppへ
   TTFT、prefill/decode token/s、TPOT、E2E、resident/peak/workspace VRAMを比較する。性能倍率はhard gateにしない。
9. 通常iterationはoperator/slice/4B shortだけに限定し、2B/9B、long、service、全model評価はencoding・graph意味が
   変わった時またはA6だけ実行する。
10. affected host/compile/GPU test、1回のintegration review、指摘箇所だけのfocused re-review、model lock、
    provenance、compatibility/main plan/history同期を完了する。

## 実装順序

### P10-A0: acceptance・format・oracle固定

- 4B BF16 lockから派生するFP8 fixture/converter contractと、block 128、scale、rounding、saturationを固定する。
- BF16 slice/logitsからtop-1、KLD、op別誤差のbaselineを作る。
- fixed llama.cpp commitとPhase 9 BF16 reportを性能比較の基準として固定する。
- 対象linearをembedding/lm-head、QKV/O、gate/up/down、GDN projectionへ分類し、FP8化しないopを明示する。

### P10-A1: dtype・model lock・loader

- tensor logical dtype、quant encoding、scale metadata、resident dtypeを分離する。
- safetensors index/metadataと派生model lockを検査し、欠落scale、shape不一致、unsupported axisをload時に拒否する。
- packed FP8 weightとscaleをmodel-residentに所有し、provider prepare keyへencoding/layoutを含める。

### P10-A2: RDNA4 native FP8 operator PoC

- exact `gfx1201`でhipBLASLtのE4M3 input/output/compute contractとsolution queryを実shapeで検査する。
- M=1とM>1を分け、hipBLASLtと小さいcustom kernel候補を独立oracle、workspace、packing cost込みで比較する。
- 127/128/129、255/256/257、Qwen実shapeと非整列shapeを含む。選べないshapeはprepare時に理由を返す。

### P10-A3: RDNA4 production W8A8

- dynamic activation scale/quantizationとlinear providerをprepared planへ接続する。
- per-opのquantize→GEMM/matvec→BF16出力をsegmentへ統合し、host同期とrequest-local allocationを増やさない。
- 4B slice、generation、O1性能を確認し、改善したproviderだけdefaultへ昇格する。

### P10-A4: RDNA2 explicit emulation/conversion

- exact `gfx1030`でFP8 byte decodeを行うW8A8 emulationを実装し、まずoperator correctnessを成立させる。
- BF16 conversion providerもmodel load/prepare時の明示pathとして用意し、resident VRAMと変換時間を記録する。
- M=1/M>1ごとにBF16 Phase 9と比較し、速い場合だけemulationをproduction defaultにする。

### P10-A5: model統合と精度評価

- 4Bの対象linearを段階的にFP8へ切り替え、最初に壊れるlayer/opを特定できるauditを残す。
- slice error、top-1、KLD、固定generationをBF16へ比較する。2B/9Bはformat/architecture差のspotに限定する。
- OpenAI non-stream/SSEはtransport再検証ではなく、FP8 model aliasとgenerationが既存serviceへ通るsmokeとする。

### P10-A6: 性能・統合・Phase 11 handoff

- O2相当の影響caseを両local targetで一度だけ測り、native/emulation/conversionを混ぜずに集計する。
- `gfx942` compile-onlyを追加し、FNUZ conversion、wave64、KV VMM capability、hipBLASLt FNUZ providerを
  Phase 11の未解決項目として具体的に渡す。
- 完了時にplanをarchiveへ移し、history、main plan、compatibility、model lock/provenanceを同期する。

## 中断・再計画条件

- RDNA4 hipBLASLtがQwen実shapeで一つもsupport solutionを返さない場合は、custom kernel範囲を再見積りする。
- 4B精度がconverter/oracleを二度修正しても合意した閾値を満たさない場合は、block/scale contractを再固定する。
- 検証・文書が作業の30%を超える、または見積りが1.5倍を超える場合は広いmodel matrixを止めて再計画する。

## 終了時更新先

- [メイン計画](../../../../main-plan.md)
- [AMD GPU互換性](../../../../../compatibility/amd-gpu.md)
- [software互換性](../../../../../compatibility/software.md)
- [model lock](../../../../../models/model-lock.md)
- [Phase 10 history](../../../../../history/2026/08/11-20/phase10-fp8-w8a8.md)
