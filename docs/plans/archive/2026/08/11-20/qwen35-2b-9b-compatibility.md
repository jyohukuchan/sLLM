# Phase 4: Qwen3.5-2B・9B互換性確認計画

Status: 2026-08-11完了。

## 目的

Phase 4として、Phase 3で完成したQwen3.5-4B BF16、単一GPU、batch 1、text-onlyの実装を、
モデル固有の複製経路を作らずQwen3.5-2BとQwen3.5-9Bへ適用する。モデルlock、
shape-driven graph、weight plan、request-local state、CLI generation、G2/G3 evidenceを
同じ実装系列で成立させ、Qwen3.5 dense text family内の再利用境界を確定する。

この計画はモデルfamily互換性の確認であり、vision、MTP、量子化、複数GPU、性能最適化、
新しいGPU targetの対応昇格は含めない。

## 前提と依存関係

- [Phase 3 Qwen3.5-4B BF16計画](../../../../archive/2026/08/1-10/phase3-qwen35-4b-bf16.md)は完了済み。
- 現行4B model lockは
  [`qwen3.5-4b-bf16.json`](../../../../../models/locks/qwen3.5-4b-bf16.json)を正とする。
- model取得、cache検証、lock fingerprintは
  [model lock正本](../../../../../models/model-lock.md)に従う。
- GPU evidenceはcanonical V620 `gfx1030`とR9700 `gfx1201`を対象とし、CPU fallback、
  timeout、crash、0件収集をGPU PASSにしない。
- 本計画のintegration完了後に
  [Phase 5エンジン性能baseline計画](engine-performance-baseline.md)へ進む。

## 固定候補と構成差分

2026-08-11にHugging Face Hubの公式repositoryから次の完全SHAを観測した。実装開始時は
branchを再解決して追従せず、このSHAのmetadata・license・全runtime/evidence fileを検証し、
完全model lockへ固定する。

| model | repo_id | resolved revision候補 |
| --- | --- | --- |
| 2B | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 9B | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

公式`config.json`で確認したtext構成差分は次のとおり。

| 項目 | 2B | 4B（現行） | 9B |
| --- | ---: | ---: | ---: |
| hidden size | 2048 | 2560 | 4096 |
| layer count | 24 | 32 | 32 |
| attention heads | 8 | 16 | 16 |
| KV heads | 2 | 4 | 4 |
| MLP intermediate | 6144 | 9216 | 12288 |
| head dim | 256 | 256 | 256 |
| vocab size | 248320 | 248320 | 248320 |
| linear value heads | 16 | 32 | 32 |
| tied LM head | yes | yes | no |

`full_attention_interval=4`、linear key head dim 128、linear value head dim 128、
RoPE theta `10000000`、partial rotary factor `0.25`は共通である。共通値もhard-codeせず
lockされたtyped configから検証する。9Bのuntied LM headは4Bのtied aliasを再利用できないため、
独立weight descriptor、upload、buffer lifetime、final projection bindingを必要とする。

## 対象

- 2B/9Bの完全model lock、外部cache、aliasとfingerprint。
- config、safetensors catalog、tensor shape、required/known-unconsumed分類。
- layer/head/hidden/intermediate/state shapeをtyped configから導くgraphとweight plan。
- tied/untied output projectionを一つの型付き分岐で表すload/execution path。
- tokenizer、chat template、stop policyのasset同一性またはモデル別差分のfail-closed検証。
- BF16 weight/activation、FP16 full-attention KV、既存linear-attention stateによるCLI生成。
- 2B/9Bそれぞれのreal-weight G2とfixed-case G3。

## 非対象

- vision tensorとMTP tensorの消費。
- Qwen3.5 Base、MoE、派生fine-tune、非公式変換model。
- FP8/NVFP4、load-time quantization、複数GPU、request batching。
- kernel最適化、performance threshold、llama.cpp性能比較。
- 4Bの既存golden token sequenceを2B/9Bへ流用すること。

## 作業単位

### M1: model lockとtyped config差分

1. 2B/9Bのlicense、model card、base model、全runtime/evidence file、index、shard、
   tokenizer、chat templateを完全SHAへ固定する。
2. 既存lock schemaが2Bの24 layer、8/2 attention heads、9Bのuntied LM headを表現できるか
   host fixtureで確認し、不足するschema fieldだけを追加する。
3. config由来値とsafetensors tensor shape/countを独立に照合し、vision/MTPを
   `known-unconsumed`として明示する。
4. model cacheはcheckout外へ置き、tracked treeにはlock、hash、summaryだけを保存する。

受入条件:

- 2 lockが全入力byteのsize/SHA-256、Hub blob/LFS identity、完全revisionを持つ。
- missing shard、wrong shape、unexpected tensor、tied/untied矛盾、mutable revision、symlink、
  cache差し替えをhost negative testが拒否する。
- 4B lockとruntime behaviorを変更しないschema migrationまたは後方互換readerになる。

### M2: shape-driven load planとexecution generalization

1. [`crates/sllm-core/src/model.rs`](../../../../../../crates/sllm-core/src/model.rs)、
   [`weights.rs`](../../../../../../crates/sllm-core/src/weights.rs)、
   [`qwen_graph.rs`](../../../../../../crates/sllm-core/src/qwen_graph.rs)、
   [`qwen_execution.rs`](../../../../../../crates/sllm-core/src/qwen_execution.rs)に残る4B固定値を
   「model invariant」「config-driven」「kernel capability上限」に分類する。
2. layer list、Q/K/V packing、linear state、KV state、MLP、RMSNorm、embedding、final outputの
   shapeをtyped configからchecked arithmeticで導く。
3. `EmbeddingAndTiedOutput`を、tied aliasと独立LM headの双方を表せるweight/output planへ
   拡張する。9Bでは独立LM headのsource range、upload、non-aliasを必須とする。
4. allocation前にmodel-resident bytes、request-state bytes、workspace bytesを算出し、
   deviceの利用可能memory不足をGPU allocation途中ではなくpreflightで拒否する。
5. 1、3、17、255、256、257 tokenと各modelの実hidden/layer境界をhost contractで確認する。

受入条件:

- 2B/4B/9Bが同じgraph/load/execution型を使用し、model別source複製を作らない。
- checked arithmetic、tensor range、alias、lifetime、state publicationの既存契約を維持する。
- 9B untied projectionと2B/4B tied projectionのpositive/negative testがある。
- 既存4B host suiteとPhase 3 fixed casesが回帰しない。

### M3: model-specific G2/G3 integration

1. 各modelからRMSNorm、embedding、final outputのreal-weight slice identityを固定し、
   raw sliceをGit管理せずrecipeとhashだけを残す。9Bでは独立LM headを必ず含める。
2. draftでは変更箇所に対応するfocused host testと、利用可能な一方のcanonical GPUで
   modelごとの最小smokeを行う。
3. integrationでは2B/9Bそれぞれをcanonical `gfx1030`と`gfx1201`で実行する。
   fixed prompt、Unicode chat、入力長255/256/257、stop/max-token caseを含め、同一model内の
   cross-target token列、stop reason、dispatch auditを比較する。
4. model load後のVRAM、request完了後のrequest-local state解放、process/VRAM cleanupを記録する。

受入条件:

- 2 models × 2 targetsのG2/G3がexact model lock、exact target、HIP dispatch、fallbackなしでPASSする。
- 各modelでcross-target token列とstop reasonが一致し、1 token以上を生成する。
- 2Bと9Bの出力同士は一致を要求しない。
- 既存4BのG3全件再実行は、共通execution semanticsまたはkernel sourceが変わった場合だけ
  integration範囲へ含める。
- 実行後health、process、VRAM、request-state cleanupに残留がない。

## 検証lane

- Draft: M1/M2のfocused host testとmodelごとの最小GPU smoke。dirty treeを許容する。
- Integration: model family全体のaffected host suite、2B/9B dual-target G2/G3、1回のintegration review。
- Release/push: clean final identity、関連H0〜H3とGPU evidence、累積reviewを固定する。
- Docs-only: link、model SHA、config表、main-planとの整合だけを確認し、GPU再実行を要求しない。

## Rollbackと停止条件

- M1はlock/schema/fixtureだけ、M2はhost generalization、M3はevidence/toolingとして独立して
  rollback可能にする。
- 既存4B pathを壊す変更は受け入れず、最後の4B検証済みidentityを維持する。
- 9Bがcanonical GPUのmemoryへ収まらない場合はCPU fallbackや部分loadで成功扱いにせず、
  required/available/peak bytesを記録してmodel placementの別計画へ切り出す。
- 同じ作業単位が2回reject、機能進捗が1時間以上停止、見積り1.5倍超過、または受入条件が
  途中で変わった場合は追加review/testを止めてreplanする。

## 完了後

2B/9Bのmodel lock fingerprint、integration identity、G2/G3 summaryをmain-planとhistoryへ記録し、
本計画をarchiveする。その後、
[Phase 5エンジン性能baseline計画](engine-performance-baseline.md)へ進む。

## 完了結果

- M1: 2B/9Bの完全lockと外部cacheを固定し、全byte、catalog、shape、tied/untied contractをPASSした。
- M2: 2B/4B/9Bの単一shape-driven graph/load/execution path、untied LM head、空きVRAM preflightを
  実装し、host/workspace/native両target buildをPASSした。
- M3: 2B/9Bのreal-weight RMSNorm G2とfixed/Unicode/255/256/257/max/stop G3をcanonical両GPUで
  PASSし、共通source変更に対する4B G3 12/12回帰もPASSした。全行HIP、fallbackなし、cleanup 0、
  cross-target一致である。
- integration identity、binary hash、dispatch、VRAM、slice recipe/hashの詳細は対応historyを正とする。
- local integrationまでを完了した。release/pushは実施していない。

[対応する履歴](../../../../../history/2026/08/11-20/qwen35-2b-9b-compatibility.md)
