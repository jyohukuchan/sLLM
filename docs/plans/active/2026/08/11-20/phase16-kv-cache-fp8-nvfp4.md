# Phase 16: KV cache FP8/NVFP4

> 状態: planned
> 作成日: 2026-08-16

## 目的

現行のopaqueなKV state、HIP VMM `virtual-contiguous`、VMM非対応target向け`contiguous-resident`、
token-major `[capacity, kv_heads, head_dim]`という上位契約を維持したまま、FP8とNVFP4のKV storageを追加する。
append時に新しいK/Vだけを一度量子化し、attentionはscaleを含む量子化cacheを直接消費する。requestごとの
全cache FP16展開、CPU fallback、別encodingへの暗黙fallbackは行わない。

本PhaseはFP8 KVを先に完成させ、その同じversioned KV encoding/layout境界へNVFP4を追加する。Phase 16Fで
`unsloth/gemma-4-12b-it-NVFP4`の公開mixed recipeを忠実に実行するため、FP8 KVを先行依存として提供する。
model weight/activationのfirst-class FP4 loaderやW4A4 linear自体はPhase 16F、TurboQuantと将来MXFP4 KVは後続とする。

## 依存関係と維持する契約

- Phase 6のKV memory decision、Phase 11の`contiguous-resident`、Phase 13のmodel-neutral transactionを再利用する。
- scheduler、frontend、generation service、HTTP層はopaque KV resource、logical length、encoding IDだけを扱い、
  value/scale pointer、VMM page size、block tableを所有しない。
- K/V planeは同じpublished logical lengthまでtransactionally appendし、片側だけ成功したstateを公開しない。
- cancel、timeout、kernel error、partial appendではlogical lengthとstate generationを進めない。commit済みphysical pageの
  lifetime規則は現行FP16と同じにする。
- `DType`とquantization encodingを分離する。FP8/NVFP4のscale、block、packingをscalar dtypeへ押し込まない。
- exact `gfx1030`、`gfx1201`をcanonical runtime targetとする。`gfx942`は既存Phase 12の証拠を変更せず、利用可能な
  実機がない間はhost contract/HIP compile-onlyに留め、RDNA結果をCDNA3 PASSへ一般化しない。

## 受入条件

### correctness/security blocker

1. `kv-fp8-v1`と`kv-nvfp4-v1`について、value code、scaleの意味、rounding、saturation、NaN/Inf、zero block、
   axis、block size、K/V plane配置、token/head stride、tail paddingをversioned contractへ固定する。
2. artifact由来のstatic scaleとruntime dynamic scaleを同じfield名で推測しない。encoding descriptorが要求するscaleを
   missing/extra、shape、dtype、range、alignmentまでload/create時に検証する。
3. appendは入力BF16から新規token範囲だけを量子化し、既存prefixを再量子化しない。失敗時はK/Vとも未公開にする。
4. attentionは量子化valueとscaleから直接計算し、request全体のFP16/BF16 KV mirrorを作らない。kernel内のregister/LDS
   展開は許容するが、resident full-cache展開はrejectする。
5. NumPy独立oracleとHIPを、block境界`15/16/17`、head dimension境界`255/256/257`、token/page境界
   `255/256/257`と`1023/1024/1025`、odd query `1/3/7/37`、padded stride、非法alignmentで比較する。
6. full-modelでは同じmodel lock、prompt token、sampling seed、provider以外を固定し、FP16 KVとのteacher-forced logits、
   KLD、top-1、greedy divergence、非finiteを記録する。silent quality regressionを自動選択へ昇格させない。
7. fixed/Unicode/stop、連続request、OpenAI non-stream/SSE、disconnect/cancel/recoveryでstate publication、usage、cleanupを
   回帰する。timeout、crash、CPU fallback、zero test selectionをPASSにしない。
8. model、raw KV、raw logits、profile、large sliceをGitへ追加せず、source/build/model/encoding identityとbounded summaryだけを残す。

### 性能・memory採用条件

- FP8/NVFP4は同じlogical capacityのFP16 KVに対するvalue、scale、padding、allocator granularity込みのcommitted/resident/
  peak byteを記録し、理論値だけで削減を主張しない。
- append、decode attention、prefill attentionを分け、kernel timeとE2E TTFT/TPOT/token/sを測る。3 warmup + 10 measuredを
  初期反復とし、clock/health、p50/p95、exact target、fallback、encodingを保持する。
- 一律の必達倍率は置かない。性能差がnoise envelope内なら共通性、memory削減、複雑性で採否を決め、明確に遅いproviderを
  target自動選択の優先経路にはしない。正しいencoding support自体は性能providerの順位と分離する。
- artifact metadataがKV encodingを明示する場合はそのencodingを使う。通常起動に低bit用の許可flag、確認、警告を追加せず、
  未対応encoding/targetはerrorにする。開発benchmarkだけがprovider overrideを使える。

## 実装・検証順序

### P16-A0: source、encoding、layoutの固定

- OCP/提供元config、現行Unsloth compressed-tensors metadata、既存FP8/NVFP4 encodingをfacts-onlyで比較し、KV専用の
  scale lifetimeとaxisを決める。weight用sidecar v1をKVへ流用しない。
- FP8を先に固定する。E4M3FN/FNUZの区別、per-token/per-headまたは明示static scale、K/V独立scale、scale storage dtypeを
  contractへ記録する。CDNA3 load/runtime conversionが必要ならRDNA encodingと別providerとして表す。
- NVFP4はpacked E2M1、block scale、outer scaleの意味をKV append単位に固定する。block-16がhead dimensionを割り切らない
  caseをpaddingとlogical lengthで明示し、weight tensor scaleのlifetimeをそのまま転用しない。
- `KvStateDescriptor`、C ABI、view/auditにencoding ID、value/scale byte、selected providerを追加するadditive/versioned方式を選ぶ。

### P16-A1: host contractと独立oracle

- Rust側にmodel-neutralな`KvEncoding`/scale layout descriptor、checked byte計算、capability query、serialization/auditを追加する。
- Python+NumPyで全FP8 code、E2M1 code、round-to-nearest-even、saturation、zero、非finite、block tailを独立実装する。
- fake backendではpayloadをmaterializeせず、descriptor、checked range、K/V atomic publication、stale generation、cancel/dropを検証する。
- malformed encoding、unknown version、scale shape/range mismatch、overflow、capacity 0/超過、unaligned viewをfail-closedにする。

### P16-A2: FP8 append、memory、readback

- `virtual-contiguous`と`contiguous-resident`のvalue/scale planeを同じlogical state ownerへ追加する。VMMはvalueとscaleの
  必要pageをappend前にgrowし、片側のmapping failureでpublicationしない。
- BF16 K/V入力から新規範囲だけをFP8へ変換するHIP append kernelを追加し、scale reductionとvalue writeの完了を一つの
  state-publication boundaryで管理する。
- evidence readbackはpublished rangeだけを許可し、value bytes、scale bytes、dequantized bounded viewを区別する。
- G1で小shape、非整列、page grow、cancel、release、recovery、fallbackなし、resource zeroを両canonical targetで確認する。

### P16-A3: FP8 attention consumption

- causal attention descriptorへKV encodingを渡し、FP8 value/scaleをtile単位にregister/LDSへ読み、FP32 accumulateと既存
  output dtype/toleranceを維持するbaseline providerを実装する。
- query length 1のdecodeとM>1 prefillを別dispatch/providerとして計測する。全KVの事前dequant kernelやhost round-tripを作らない。
- GQA、RoPE済みK、causal mask、sliding/full attention、Qwen/Gemmaの異なるKV head/layoutをmodel adapter側の既存semanticへ結ぶ。
- NumPy attention oracle、FP16 provider、FP8 providerの三者で境界matrix、actual dispatch ID、fallback falseを照合する。

### P16-A4: NVFP4 appendとattention

- A2/A3のencoding-neutral ownerとattention descriptorを再利用し、NVFP4 value/block/outer scale planeだけを追加する。
- RDNA2/RDNA4ではnative FP4と表記せず、packed-dequant attention providerとして実装する。全cache mirrorなしをauditする。
- FP8と同じcaseにblock-16 tail、scale極値、zero block、odd head/tokenを追加し、operator error分布とnonfiniteを記録する。
- FP8/NVFP4の混在K/Vをdescriptorでは表現可能にするが、Phase完了matrixは同一encoding K/Vをrequiredとし、全組合せを
  無条件に増やさない。model recipeが混在を要求した時だけ該当caseを追加する。

### P16-A5: full model、quality、capacity

- Qwen3.5-4B BF16 modelをprimaryに、FP16/FP8/NVFP4 KVでfixed short/odd/long promptとteacher-forced位置を比較する。
- Gemma 4はR9700で代表full/sliceを使い、共通KV pathがQwen固有になっていないことを確認する。V620は収容可能なQwen caseへ限定する。
- context長は少なくとも`1/7/255/256/257/1023/1024/1025`と、実memory差が現れる長いcaseを含める。
- committed/resident/peak VRAM、maximum admitted context、append/attention/E2E性能、KLD median/p90/max、top-1、greedy divergenceをまとめる。

### P16-A6: service、文書、closeout

- normal generate/serverはmodel metadata/configからKV encodingを解決し、低bit専用modeを追加しない。doctor/benchmark auditだけが
  encoding、scale、provider、bytesを表示する。
- fixed/Unicode/stop、sampling、連続request、SSE、disconnect/cancel/recoveryを最終candidateで実行する。
- runtime、KV memory、model lock、GPU/software compatibility、main plan、historyを同期する。1回のintegration reviewと、
  findingを直した箇所だけのfocused re-reviewを行い、本planをarchiveしてPhase 16Fへ進む。

## 計測matrix

| level | encoding | 主case | 指標 |
| --- | --- | --- | --- |
| format | FP8/NVFP4 | 全code、scale極値、zero、15/16/17 tail | byte、dequant値、rounding |
| state | FP16/FP8/NVFP4 | append、page grow、cancel、stale generation | published length、bytes、cleanup |
| attention | FP16/FP8/NVFP4 | Q=1/3/37、KV=255/256/257/1023/1024/1025 | output error、kernel、fallback |
| model | Qwen4B、bounded Gemma | short/odd/long、teacher-forced/greedy | KLD、top-1、divergence、VRAM |
| service | adopted encoding | fixed/Unicode/stop/SSE/cancel | text、usage、state、cleanup |
| performance | adopted provider | decode/prefill/E2E | p50/p95、TTFT、TPOT、token/s、peak |

## 非対象

- model weight/activationのfirst-class FP4 artifact loader、W4A4、MXFP4/MXFP8 model execution。
- TurboQuant、K4V4、将来K3/K2.5/K3.5、MXFP4/MXFP8 KV。
- Paged Attention production化、prefix sharing、RadixAttention、continuous batching。
- KV/conversation永続化、multi-GPU、Infinity Fabric、RCCL/RDMA。
- FP8/NVFP4品質のために既存model-weight KLD thresholdを変更すること。

## 停止・再計画条件

- 公開artifact metadataと選んだKV encodingのscale意味が一致しない場合はkernel実装を止め、encoding/versionを分離する。
- quantized attentionがfull-cache mirrorなしでは数値contractを満たせない場合、mirrorをproduction採用せずalgorithm/layoutを再計画する。
- VMM value/scale planeをatomicにgrow/publicationできない場合、上位opaque契約を壊さずstate owner設計へ戻る。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  acceptance変更時は追加探索を止めて同じwork unitを再計画する。

[対応する履歴](../../../../../history/2026/08/11-20/phase16-kv-cache-fp8-nvfp4.md)
