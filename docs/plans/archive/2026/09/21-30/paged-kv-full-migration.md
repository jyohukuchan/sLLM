# Paged KV／Attention本移行（Phase 87 段階10の詳細計画）

2026-09-24作成。状態: **2026-09-25完了**。exact `gfx1030`／`gfx1201`のproduction KVをPagedへ統一し、旧VMM／resident providerを退役した。公開GPU数値、fork／COW／cancel／rewind／graph、Paged V2 image・checkpoint、指定hot shapeのkernel 10%条件、短いQwen3.8モデル速度・物理VRAMを確認。`gfx942`はユーザー決定により現行runtimeで明示未対応。実測範囲とartifact不足の制約は[段階10履歴](../../../../../history/2026/09/21-30/phase87-stage10-paged-kv.md)を正とする。Phase 87全体は段階11が残り、ユーザー確認まで完了扱いにしない。
[Phase 87 WU-P1](../../../../../history/2026/09/21-30/phase87-wu-p1-paged-attention.md)では、
exact V620 `gfx1030`／R9700 `gfx1201`のMXFP8 E4でPaged Attention試作を比較し、
decode／prefillのkernel単体増加が全roundで10%未満、数値bitwise一致を確認した。
この結果を受け、[ユーザー決定](../../../../../history/2026/09/21-30/kv-paged-migration-decision.md)どおり
vAttentionを廃止してPaged Attentionへ完全移行する。2026-09-24のユーザー指示で、本計画を[Phase 87](../../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)の段階10として統合した。性能改善は段階11で行い、本計画では計算の構造を変えない。
Phase 87の段階12（MTP幅3・4）は2026-09-25に完了した。ここから段階10のproduction移行を進め、
共有するgraph/runtime fileの変更時に順序を調整する。
段階1のtest-only行列積評価と、段階10のhost側pool設計は独立に進める。

## 範囲と固定方針

- 128-token blockの物理KV poolを使い、key/valueのMXFP8 E4 value planeとE8M0 scale planeを同じblock tableで参照する。
  logical token順・causal mask・数値形式は変えない。FP16 KV等の現行で公開済み形式も対応表を作って移す。
- 2026-09-25の着手時設計では、物理poolはsegmented slab、device上のblock descriptor tableとrequestのlogical
  block tableは固定pointerとする。slabはcapture外で必要量だけ確保し、entryを次replay前に更新する。
  現行のfull-capacity preflightは初期移行で維持し、全logical容量の物理VRAM前確保はしない
  （[棚卸し・判断](../../../../../history/2026/09/21-30/phase87-stage10-paged-kv.md)）。
- requestのlogical block table、pool上の物理block、参照数、公開済みlogical lengthを一つのKV state契約で管理する。
  prefix共有とrequest forkではblockを共有し、末尾の部分blockは書込み時にCOWする。
  cancel／失敗時に親・兄弟のKVを変えない。
- graph capture済みdecodeが同じdevice上のblock tableを読み続けられるよう、append／fork／grow後のtable内容と
  logical lengthを次replay前に更新する。host同期・graph再capture・CPU KV計算を暗黙に増やさない。
- VMM provider、page共有・末尾COW、grow transaction、V620の65,536-token境界、R9700全量resident例外を
  paged poolの契約へ置き換えて削除する。poolのblock確保／scale plane／table更新は一transactionとし、
  途中失敗時は参照数・free list・公開lengthをappend前へ戻す。復元不能なstateは再利用しない。
- [KV memory方式の決定](../../../../../architecture/kv-memory.md)、公開C ABIのKV memory kind／view／version、
  Rust HIP adapter、capability選択、model memory preflightを実装と同期する。
  古いVMM／resident指定の公開値は、互換性と拒否方法を決めてから退役させる。
- 現行連続KV経路は、本移行内の最終A/B比較までだけ保持する。採用後は一つのpaged既定経路にして
  連続KV providerと一時比較コードを削除する。実行時の方式切替は作らない。

最終A/B後の退役契約: 旧`virtual-contiguous`／`contiguous-resident`／
`capability-selected` memory kindの公開番号はABI予約値として保持し、
旧KV create／view／fork／image入口では`UNSUPPORTED`を返す。Pagedは専用create／view／image ABIを使う。
既存V1 checkpoint fileはwire parserを残して形式を識別し、productionでは
「Paged V2で作り直す必要がある」と明示拒否する。V1の生KV planeを
Paged topologyとして暗黙に読み替えない。non-KV linear stateのV1 plane契約は
V2 payloadでも使うため維持する。

対象は最初にQwen3.8 NVFP4／MXFP8 E4 KVの単一要求とexact gfx1030/gfx1201。
他の現行KV形式・model・targetは「使える既存経路を黙って消さない」範囲で互換表を作り、
実装対象と未対応の明示拒否を決める。8〜16並列のscheduler／batch attentionと複数GPU実行はPhase 88以降の範囲。
2026-09-25のユーザー決定で、Paged実機証拠がない`gfx942`は段階10の現行runtimeで
明示`unsupported`とする。過去のMI300X実機証拠は履歴として残し、起動時の
target拒否をRust session／公開C contextでGPU allocation前に実装した。
旧ABI値／selector／CI記述は旧resident経路の撤去と同時に揃える。
1M以上のcontextは容量・整数境界・preflightの設計対象であり、この32GB GPUでQwen3.8の1M KVを
常駐させられるという主張ではない。

## 実装の順序

1. 現行のKV state、公開ABI、Rust adapter、attention provider、HIP Graphのpointer生存期間を一覧化する。
   形式別のtoken/block bytes、pool上限、tableの型と更新順、shared prefixの所有権を決める。
2. paged poolとblock tableを実装し、append／fork／share／COW／cancel／releaseの状態遷移を揃える。
   page未使用部分、非整列長、block 127／128／129、65535／65536／65537の境界、容量不足とrollbackを扱う。
3. WU-P1のprobeから数値・addressingの知見を取り込み、production decode M=1〜5とprefillをpaged poolへ接続する。
   M=4／5は先行して完了した段階12の幅3／4によるverify経路であり、移行で失わない。
   既存のrounding、split、exact shape selector、MXFP8 E4のscale計算を保ち、他shapeを含むprovider表を更新する。
4. graph replay前にblock tableとlogical lengthを更新し、prefix forkと途中cancelを含む実要求のGPU経路を確認する。
   KV appendの公開点とtable更新の順序を同じstream/event契約で保証する。
5. 現行連続KVとの最終比較後、vAttention/VMM固有コード、resident特例、古いmemory kind選択を撤去する。
   ABI、memory決定、runtime／互換性文書、CI入口とモデル側の契約を同じ変更で揃える。

### Rust adapterの段階的接続（実施時の順序）

段階10の初期Rust adapterは`create_paged`／`query_paged`を明示的に呼ぶopt-in経路から始め、
最終比較後に`create_v2`／legacy queryの本番呼出しを削除した。paged poolの`max_physical_blocks`は、
一つの親と一つの子が同じprefixを共有した後に、それぞれ全logical capacityまで分岐できるよう
`2 * logical_table_capacity`をchecked arithmeticで予約し、nativeの`UINT32_MAX` ID上限を超える場合は拒否する。
production defaultへ切り替える前に、native append／attention／fork／graph replayをこの経路へ接続する。

Qwenの`memory_audit_snapshot`、CLI、server、benchmarkはtag付き`Vmm`／`Paged`
metadataを経て、現行HIP productionではPagedだけを生成する。Pagedのblock／table／plane byte値をVMM page fieldへ流用しない。

## 受入と証拠

- correctnessは数値oracleを伴うexact GPU実行で確認する。非identity block table、非整列長、block境界の両側、
  prefix共有・fork・末尾COW・pool不足・rollback・cleanup・graph replayを含める。
  CPU fallback、空選択、timeout、crashはGPU PASSにしない。
- WU-P1の同一process AB/BAをdecode／prefillのkernel単体対照として再利用し、最終production sourceで
  10%条件を再確認する。超えた場合は表参照・境界・prefill tileを切り分け、移行方針をユーザーへ戻す。
  - WU-P1で10%に最も近かったのは、V620のprefill KV65536 M128（`gqa6_qtile8_w16`）の+7.45〜+8.19%で、
    同じproviderのKV8192（約0%）よりKVが長いほど増えている。本移行の最終比較では、この経路をKV131072以上でも測り、
    block tableが順番どおりの場合と逆順の場合を分けて記録する（2026-09-24レビューでの追加。数M contextで増え続けるかを確認するため）。
  - R9700の大きな短縮（decode約−26〜−32%、prefill約−11〜−40%）は、pagingそのものではなく、
    paged候補に入れたwave-localなK/V読み出しと8-byte scale rowの読み方によるものである。
    モデル全体の速度差をpaging方式の効果として記録しない。
- 通常モデルのTPOT、TTFT、prefill、peak VRAM、KV append／graph費用を連続KV最終候補とpaged候補で記録する。
  WU-P1のkernel数値からモデル速度や大容量の安定性を推定して採用判定へ代用しない。
- affected host、HIP compile-only、両GPU correctness/performanceを既存の検証入口で実行する。
  公開ABIの互換性、model lock、CI参照、docsを同期し、統合時に一度レビューする。

本計画の着手時に現行のPhase 87採用済みsourceを基準として固定し、詳細な形式別互換表と測定identityを履歴へ残した。
新しい数値差やハードウェア対象の拡張を、WU-P1の合格から自動的に承認済みとは扱わない。

## 完了結果

- exact `gfx1030`／`gfx1201`のPaged公開GPU correctnessを両方PASS。127／128／129境界、M=1〜5、形式別、fork／COW、cancel／rewind、graph、V2 imageを独立oracleとcleanup 0で確認した。
- 最終連続KV対照とのkernel単体AB/BAは指定hot shapeで全round 10%以内。最大比はV620 `1.01062`、R9700 `1.04757`。逆順logical tableも独立oracleとbitwise対照をPASSした。
- 旧連続KV provider／selector／実行時切替を撤去し、旧公開KV ABIは予約番号を保って`UNSUPPORTED`。V1 checkpointは形式を識別して再生成を求める。non-KV linear stateのV1 planeはV2 payloadの構成要素として維持した。
- Qwen3.8実モデルの退役後単回smokeは両GPUで同じ5 token、MTP受理2/4、HIP-only、fallbackなし、cleanup 0。これは速度採否の追加runではない。
- 追跡済みStage5／Phase83の旧contiguous専用test sourceは履歴資料で、現行CMake／CIから非登録。Stage10の現行公開testはPagedと独立oracleへ移した。
- 一部modelの準拠GGUF／lockがなくHTTP実モデル証拠を追加できない範囲と、`gfx942`未対応を[履歴](../../../../../history/2026/09/21-30/phase87-stage10-paged-kv.md)に限定して記録した。

根拠: [Phase 87 WU-P1履歴](../../../../../history/2026/09/21-30/phase87-wu-p1-paged-attention.md)。

段階10履歴: [Paged KV本移行](../../../../../history/2026/09/21-30/phase87-stage10-paged-kv.md)。
