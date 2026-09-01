# Phase 65 inference-engine comparison boundary

## 目的

Phase 64後のexact `gfx1201` MXFP8 prefill残差について、他推論engineは性能構造と評価方法を抽出する比較対象として使い、
非MIT softwareのsource expressionをsLLMへcopy、adapt、portしない境界を固定する。

## 固定した参照identityとlicense

| mirror | revision | repository license | Phase 65で許可する利用 |
|---|---|---|---|
| llama.cpp | `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` | MIT | 技術参照。直接reuseする場合は別途import recordを作る |
| SGLang | `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` | Apache-2.0 | no-copy比較だけ |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` | Apache-2.0 | no-copy比較だけ |
| LMDeploy | `f4b8140ba19cd823c541241cbb113cc32f854e6a` | Apache-2.0 | no-copy比較だけ |
| KTransformers | `924754a00bd8e5c6a2ad97929065c113f35782cf` | Apache-2.0 | no-copy比較だけ |
| TensorRT-LLM | `376f7e1bd8ed543f75014309e3fd4b237e9b0e73` | Apache-2.0ほかfile別notice | no-copy比較だけ |

licenseは各mirror rootのlicense fileで確認した。package全体の表記を個別fileの再利用許可とはみなさない。
今回の実装ではllama.cppを含め、第三者sourceの直接reuseを行わない。

## 比較から抽出した技術事実

- llama.cppの量子化matrix providerはGPU family、行方向の問題規模、output tileなどに応じて複数configurationを選ぶ。
  RDNA4 MXFP4はsLLMのMXFP8 W8A8とはformatと内積が異なるため、kernel本体やconfiguration tableを移植せず、
  「単一tileを全shapeへ一般化しない」というprovider設計だけを比較点とする。
- sibling SGLangのR9700 Qwen3.5-4B BF16記録は、512 input中央値`6,760.856 tok/s`、4,096 input中央値
  `9,856.357 tok/s`だった。ROCm 7.2、BF16 weight、AITER tuned GEMM、graph、入力／出力条件がsLLM MXFP8測定と異なるため、
  速度比はapples-to-apples結果ではない。shape-tuned GEMM、fused前後処理、graphが独立した性能軸であることだけを使う。
- SGLang／vLLMのROCm経路はAITER等の外部tuned providerをshapeやdtypeで選択する。sLLMはそのsource、dispatch table、
  tuning値、symbolを参照実装へ持ち込まず、公開HIP／rocWMMAと自前oracleからproviderを作る。
- Phase 64後のsLLM 2,048-token profileでは、ID31が35.13%、ID34 direct-weightが26.54%、causal attentionが15.25%、
  linear recurrentが6.07%だった。matrix合計が引き続き最大で、最初の対象は残存ID31 shapeである。

## clean-room実装入力

現在のID31はactivationとweightをLDSへ置く。M128 workgroupではactivation rowはwave固有で再利用されず、weight tileは
8 waveから再利用される。したがって次の独立候補をsLLM既存bodyだけから導く。

1. activation valueだけをglobalからwave fragmentへ直接loadし、weight valueはLDS共有を維持する。
2. wide shapeではPhase 64 direct-weightとの組合せも別候補として測る。
3. 非整列Mの末尾はzero-padded LDS fallbackを使い、out-of-bounds global loadを許さない。
4. 採否はmodel名ではなくexact target、dtype/encoding、M/K/N、layout、測定済みshape境界で決める。

この候補は第三者のsource text、control flow、疑似コード、tile値、symbolから作成したものではない。MIT llama.cppからの
copy/adapt/portも行わないため、Phase 65時点で`THIRD_PARTY_NOTICES.md`へ追加するimportはない。

## 実装結果

sLLM既存bodyから独立にID35 activation-directとID36 activation／weight directを実装し、ID31／34を含む4経路比較で
ID36を測定済みshapeへ採用した。外部engine由来のcode、tile table、dispatch定数、symbolは実装差分に含まれず、
Phase 65で第三者import recordを追加しない判断は完了時点でも変わらない。

## Phase 66 follow-up

Phase 66も同じ比較境界を維持した。外部engine比較から使ったのは、target／format／shape別provider、consumer向けactivation
layout、複数tile familyを独立に評価するという抽象所見だけである。llama.cppのQ8_0／Q8_1式、packed layout、tile table、
kernel body、configuration定数、symbolをMXFP実装へ移植せず、SGLang／vLLM等のsource、pseudocode、dispatch tableも
実装入力にしていない。

sLLM既存ID36をoutput列方向だけN128へ広げたID37、typed prepared-provider契約、FP16／MXFP8 KV attention候補、
MXFP6／NVFP4／MXFP4 routingは、sLLMの既存arithmetic、OCP format契約、独立oracleから実装した。NVFP4／MXFP4 W4A4は
既存sLLM device kernelをfrozen providerへ接続した変更で、第三者由来kernelも同期比較用の別kernelも追加していない。

したがってPhase 66でも第三者code reuseは0で、`THIRD_PARTY_NOTICES.md`へ追加するimportはない。MIT llama.cppからの
直接reuseを将来行う場合は、従来どおりfile単位のimport recordを別途作成する。

[provenance policy](README.md) /
[Phase 65保存済み計画](../plans/archive/2026/09/1-10/phase65-gfx1201-mxfp8-asymmetric-staging.md) /
[Phase 66履歴](../history/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md)
