# Phase 15: Weight NVFP4

> 状態: complete
> 作成日: 2026-08-15

## 目的

Qwen3.5とGemma 4 Denseのmodel weightをNVFP4 encodingへ変換・保存し、BF16 activationから実行できる
weight-only NVFP4 pathを追加する。NVFP4 value、block scale、tensor scaleをtensor/quantization descriptor、
derived artifact、loader、kernel/provider、auditの全境界で保持し、tensor scaleの欠落や暗黙BF16 fallbackを許さない。

AMD targetで利用できる命令/library capabilityはexact targetで実測してproviderを分ける。NVFP4 storageを読めること、
FP4 native arithmeticを使うこと、実用的に高速であることを別々に判定する。

## 開始条件

- Phase 13のmodel-neutral prepared execution制御が完了している。
- Phase 14 Gemma 4 Denseが少なくともQwenと異なるadapterとしてproduction executorへ接続されている。
- Phase 14後のfresh profileで、NVFP4前に行う共通RDNA最適化候補の採否が記録されている。
- NVFP4の公式format/value/scale/tensor-scale規則と、参照実装source revisionを固定できる。

共通RDNA性能bridgeは2026-08-15に候補二つを採用した。現行BF16開始baselineはR9700でGemma
`3/17` `14.221 tok/s`、`32/32` `13.949 tok/s`、Qwen3.5-2B short-odd `66.490 tok/s`、V620で
Gemma `11.768/11.398 tok/s`、Qwen short-odd `54.942 tok/s`とする。詳細は
[bridge履歴](../../../../../history/2026/08/11-20/cross-model-rdna-performance-bridge.md)を正とする。

Phase 12 MI300X実機PASSは開始条件にしない。CDNA3でnative NVFP4を推測または主張しない。

## 固定する製品契約

- 本Phaseは**weight-only NVFP4**であり、activationは既存BF16を基本とする。W4A4、KV NVFP4、FP4 attentionを含めない。
- physical FP4 value、block/group scale、tensor scale、logical shape、packing order、padding、original dtypeを
  `DType`一つへ詰め込まずversioned quantization encodingで表す。
- tensor scaleはoptional metadataにせず、NVFP4 tensorのcapability/prepare/manifest/kernel引数へ必須として渡す。
- converterはverified BF16 model lockからreproducible sidecarを生成し、runtimeは起動時量子化を既定にしない。
- unsupported target/providerはloadまたはprepareで失敗する。runtime kernel failure後にBF16、FP8、別scale規則へfallbackしない。

## target/provider lane

| target | lane | claim |
| --- | --- | --- |
| R9700 `gfx1201` | native/library capabilityを実測し、なければpacked dequant+BF16 candidateを明示 | production候補、採用は性能・精度次第 |
| V620 `gfx1030` | byte/packed decode oracle、emulation、必要ならload-time BF16 conversion | correctnessまたはexplicit converted path |
| MI300X `gfx942` | compile/descriptor準備だけ。実機を起動しない | local-only、native/performance claimなし |

provider名は`native`、`packed-dequant`、`emulation`、`converted-bf16`等の実態を表し、NVFP4 sidecarをloadしただけで
native FP4と表記しない。

## スコープ

- official NVFP4 numeric/packing/scale contractと独立oracle。
- reproducible BF16-to-NVFP4 converter、derived sidecar manifest、hash/range verification。
- tensor/weight plan、quantization encoding、loader、model alias、resident owner。
- Matmul/linear provider、workspace、prepared plan cache key、dispatch/audit。
- Qwen3.5とGemma 4の代表linear tensor、model slice、full-model accuracy/generation。
- R9700/V620のresident/peak VRAM、load time、prefill/decode/E2E性能。
- CLI/OpenAI alias、no-silent-fallback、cleanup。

次は含めない。

- KV cache FP8/NVFP4、activation NVFP4、W4A4、FP4 attention。
- calibration/imatrix、mixed-bit自動探索、quality-aware layer exclusionの一般optimizer。
- multi-GPU、CDNA3実機、RDNA4 FA3-like、MoE expert-specific quantization。
- Qwen/Gemma graphのNVFP4専用複製、requestごとのweight unpack、untracked runtime conversion cache。

## 受入条件

1. official fixed sourceからvalue、special value、rounding/saturation、packing、block scale、tensor scale、padding規則を固定する。
2. 全code point、境界前後、非整列group/tensor、zero/subnormal/finite max、NaN/Inf入力を独立oracleで検査する。
3. converter outputがsource model fingerprint、tool identity、arguments/config、environment、全tensor range/hash、scale range/hash、
   artifact hashをmanifestへ保存し、同じ入力からbyte-identicalに再生成できる。
4. loaderはmissing/extra tensor、shape、packing、block/tensor scale、hash、source lock、provider capability不一致を拒否する。
5. prepared cache keyがencoding、scale layout、provider、target、weight identityを含み、BF16/FP8/NVFP4間で誤再利用しない。
6. R9700/V620 operatorが独立higher-precision oracleへ一致し、selected provider、native/emulated/converted、fallbackをauditする。
7. QwenとGemmaのmodel sliceがBF16 referenceに対するerror、top-1、KLDを記録し、単一promptの文章だけで採否を決めない。
8. 少なくとも収容可能な一modelでfull generation、fixed/Unicode/stop、連続request、CLI/OpenAI smokeを通す。
9. resident/peak VRAM削減量、sidecar size、load time、TTFT、prefill/decode tok/s、TPOT、E2EをBF16および適用可能なFP8へ比較する。
10. VRAMを削減しても遅いproviderはdefaultへ自動昇格せず、opt-in/correctness/convertedを明示する。
11. affected checks、integration review、provenance、model lock/runtime/compatibility/main plan/history同期を完了する。

## 実装順序

### P15-A0: format/source lockとaccuracy budget

- official NVFP4 specification、reference conversion、packing/scale定義の完全revisionとlicenseを固定する。
- Weight NVFP4として採用するvalue/scale contractを文書化し、似たFP4 formatやvendor別variantを混同しない。
- Qwen/Gemmaからrepresentative linear tensorと小graphを選び、BF16 reference、top-1/KLD、tensor error指標を固定する。
- Q2 profileからlinear wall time、memory bandwidth、launch costを取り込み、期待するVRAM/帯域効果とdequant overheadを分ける。

### P15-A1: encoding descriptorとderived artifact schema

- quantization descriptorへvalue format、packing order、block/group size、scale dtype/layout、tensor scale、logical/padded shapeを追加する。
- weight planとtensor viewがpacked bytesとscale bufferを別owned resourceとしてcompletionまで保持する。
- sidecar manifest schemaをFP8 sidecarと共通化できるidentity/range部分とNVFP4固有metadataへ分ける。
- unknown version、scale欠落、overflow、noncanonical padding、alias overlapをhost contractで拒否する。

### P15-A2: converterと独立oracle

- verified BF16 tensorをstreaming chunkで読み、定義済みrounding/saturationでvalueとscaleを生成する。
- tensor scaleを計算、保存、適用する順序を固定し、block scaleだけで近似しない。
- 全code point、random/structured tensor、group境界B-1/B/B+1、末尾padding、極端rangeでdecode/reconstructionを比較する。
- converterを二回実行してmanifest/artifact byte identityを確認し、large sidecarをrepository外cacheへ置く。

### P15-A3: loader、resident layout、provider selection

- model aliasをsource lockとNVFP4 sidecar fingerprintの双方へ結び、部分一致や別converter outputを拒否する。
- packed weight、block scale、tensor scaleをmodel load時に一度uploadし、request間でresidentに再利用する。
- exact target/capability、shape/alignment、activation dtype、scale layoutからproviderをprepare時に選ぶ。
- provider metadata、resident bytes、workspace、conversion/unpack有無をauditする。

### P15-A4: RDNA operator/provider

- R9700で利用可能なnative/library instruction pathをcompile/query/microでfail-closedに確認する。
- native pathがない、または実shapeで遅い場合はpacked weightを保持するfused dequant+BF16 matvec/GEMM candidateを実装する。
- V620はbyte/packed decode emulationを数値証拠に使い、実用pathは必要に応じてexplicit load-time BF16 conversionへ分ける。
- M=1とM>1、K/N非整列、group境界、tail、tensor-scale極値、alias、unsupported targetを独立oracleで検査する。

### P15-A5: Qwen/Gemma graphとaccuracy

- model graphを複製せず、linear weight encoding/providerだけをprepared plan bindingへ差し替える。
- embedding、normalization、attention state、KV、samplingは既存dtypeを維持し、量子化対象tensor集合をmanifestと一致させる。
- Qwen最小modelとGemma representative sliceでlayer output、logits、top-1、KLDをBF16へ比較する。
- qualityが悪いtensor/layerを場当たり的にBF16へ戻さず、必要なら明示mixed-encoding policyとして再計画する。

### P15-A6: full model、VRAM、性能

- 収容可能なQwen/Gemma modelを一度loadし、fixed/Unicode/stop generationと短いaccuracy setを実行する。
- resident/peak/workspace、sidecar bytes、upload/load timeをBF16/FP8と同じ定義で測る。
- short-odd、32/32、prefill-longまたはdecode-longのうち影響する代表caseをO1/O2で測る。
- R9700/V620のproviderを別々に判定し、一方の改善を別targetへ一般化しない。

### P15-A7: service、採用判断、closeout

- CLI/OpenAI model alias、non-stream/SSE、disconnect、連続request、unload/process cleanupを確認する。
- `default / opt-in production / correctness-only / converted`をtarget/providerごとに決め、native claimを監査する。
- provenance、derived artifact、runtime/model lock、GPU/software互換性、main plan、historyを同期する。
- Phase 16 KV cache FP8/NVFP4へvalue/scale/layout oracleとprovider経験を渡し、本planをarchiveする。

## 計測lane

| lane | 内容 | 通常の使用 |
| --- | --- | --- |
| N4-H | format/schema/manifest/loader/cache host contract | 各work unit |
| N4-O | 全code point、group/tail、operator micro/oracle | provider変更時 |
| N4-S | Qwen/Gemma tensor・small graph slice | encoding/model変更時 |
| N4-G | 収容可能modelのshort generation | integration単位 |
| N4-P | short-odd/32-32のO1、最終O2 | 採用判断時 |
| N4-I | CLI/non-stream/SSE/disconnect/cleanup | closeoutで一回 |

## 再計画条件

- official NVFP4 contractまたはtensor scale規則を固定できない場合は、推測formatを実装しない。
- R9700にnative/library pathがないことはPhase失敗ではないが、providerをnativeと呼ばずpacked-dequant/convertedの価値を測る。
- accuracyが悪い場合はthresholdを緩めず、scale、packing、accumulation、対象tensorを切り分ける。
- VRAM削減を上回る性能退行がある場合はopt-in/correctness laneに留め、default化のための無制限tuningを行わない。
- 同じwork unitの2回reject、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超では追加runを止めて記録する。

[対応する履歴](../../../../../history/2026/08/11-20/phase15-weight-nvfp4.md)

## 完了結果

- P15-A0〜A4: official format lock、独立codec/oracle、deterministic converter、fail-closed sidecar loader、
  exact `gfx1030`/`gfx1201` packed-dequant providerを完了した。`gfx942`はcompile/descriptorだけである。
- P15-A5/A6: Qwen3.5-2B full sidecarと両GPU full accuracy、Gemma 4-12B real-weight slice、R9700 CLI/service、
  V620 resident/high-waterを取得した。follow-upで両GPUのBF16/NVFP4 short-odd・32/32を各3 warmup + 10 measuredで
  比較し、resident 52.43%削減と、NVFP4 decode約20〜22%低下、R9700 prefill/TTFT大幅退行を確認した。
  Qwen最大KLD `0.2637523`とGemma top-1 2/3によりdefault化しない。
- P15-A7: 両targetをcorrectness-only opt-inとし、native FP4 claimを行わない。互換性、runtime、model lock、
  provenance、main plan、historyを同期し、本planをarchiveする。
