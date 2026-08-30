# Phase 53: KV FP8 block16実装とtarget別default採用

> 状態: 完了（2026-08-27。descriptor v2／`StandardMxFloorPowerV1`のgfx1201／gfx1030 correctness・品質を再取得し、両targetとも`retain-fp16`。gfx942実機は今後の一括検証へ延期）
> default候補: `kv-fp8-e4-block16`、`kv-fp8-e5-block16`
> explicit比較形式: `kv-mxfp8-e4`、`kv-mxfp8-e5`
> 順序: format／operator／state作業は独立開始可。default判定だけがPhase 46のfreeze済みKV品質policyを必要とする。

## 目的

16値ごとに独立scaleを持つMXFP8-likeなKV cacheをadditiveに実装し、exact GPU target、検証済みmodel／shape、品質・性能・
資源証拠に基づいて既定形式を選ぶ。CDNA3／RDNA4はE4M3系、RDNA2はE5M2を用い、scale統計は同一token内だけに限定する。

加えて標準OCP MXFP8を32値blockのexplicit-only形式として実装する。同じFP16 logit baselineに対するblock16と標準MXFP8のKLD等を
直列計測し、将来native MXFP8演算器を持つtargetを追加できるdescriptor／operator境界を先に固定する。標準MXFP8は本Phaseの
default候補へ自動追加せず、block16のtarget別採否を変更しない。

既存の`fp16`、`fp8`、`fp8-static`、`nvfp4`の公開意味とencoding IDは変更しない。新形式を既存IDへ詰め替えず、明示選択と
default選択のどちらでもresolved encoding／descriptorをstate identityとreportへ残す。

## ユーザー決定とformat名

- canonical public nameは正確に`kv-fp8-e4-block16`と`kv-fp8-e5-block16`とする。
- 追加suffixやaliasは公開しない。format versionはpublic nameへ連結せず、descriptor／schemaの
  独立version fieldで管理する。
- 1 scaleは同一token、同一KまたはV plane、同一KV headのhead-dimension方向に連続する16値だけを共有する。
- tokenを跨ぐper-channel scale、channel calibration、per-channel selectorは実装しない。
- CDNA3／RDNA4のvalueはE4M3系、RDNA2のvalueはE5M2とする。物理FP8 variantはtarget descriptorに固定し、OCP byte列と
  FNUZ byte列を再解釈しない。
- 2026-08-27のユーザー決定により、block16のscale決定はE4／E5ともstandard MX ruleへ統一する。block sizeは16のままで、
  standard MXFP8の32へ変更しない。descriptorは`kv-fp8-e4-block16-v2`／`kv-fp8-e5-block16-v2`、recipe identityは
  `StandardMxFloorPowerV1`とする。旧descriptor v1の「有限値を飽和させない最小scale」recipeはsupersededである。
- 標準形式のcanonical public nameは`kv-mxfp8-e4`と`kv-mxfp8-e5`とする。block sizeはOCP MX v1.0どおり32、elementはそれぞれ
  OCP E4M3とOCP E5M2、scaleはE8M0とし、initial runtimeではexplicit-onlyにする。

## format contract

1. logical shapeは既存KVと同じtoken-major `[capacity, kv_heads, head_dim]`、K/Vは独立planeとする。
2. block indexは`floor(head_index / 16)`、block数は`ceil(head_dim / 16)`とする。末尾partial blockのscaleはvalid laneだけから求め、
   storage paddingはcanonical zeroにする。paddingをattentionへ入力しない。
3. scaleは1 byteのE8M0 power-of-twoとし、`StandardMxFloorPowerV1`をE4／E5へ共通適用する。有限amaxの最大2冪をE4では256、
   E5では32768で割ってE8M0範囲へ収める。all-zero blockはunit scale＋zero payloadをcanonical表現とする。
4. value encodeはRNE＋SAT、NaNはvariantごとのcanonical NaN、±Infは符号付き最大有限値へ写す。OCP E4M3FN／E5M2はsigned zeroを
   保持し、FNUZは負zeroをcanonical正zeroへ写す。subnormal、underflow、scaleのexponent端、OCP／FNUZ差を独立oracleへ固定する。
5. append入力とattention出力は既存どおりBF16、score／softmax／weighted-value accumulatorはFP32を維持する。attentionはpacked
   value＋scaleを直接読み、request全体のFP16/BF16 mirrorまたは全KV dequantize dispatchを作らない。
6. appendはK value、K scale、V value、V scaleの全plane完了後だけlogical lengthをatomicに公開する。grow/COW途中失敗は
   Phase 52のtransactional rollback契約に従い、partial token／planeを公開しない。

Qwen3.5のhead dim 256では、KまたはVの1 token／head当たりFP16は512 byte、新形式はvalue 256 byte＋scale 16 byteの
272 byteであり、logical storageはFP16比46.875%減る。現行dynamic FP8の260 byteより12 byte、約4.615%増える。
実際のHBM削減はplane alignment、VMM page、resident capacityを含めて測定し、このlogical計算で代用しない。

### 標準MXFP8比較contract

1. scale axis、K/V独立、token-major shape、direct append／attention、atomic publication、standard MX scale ruleはblock16と同じで、
   block sizeだけを32にする。
2. 変換scaleはOCP MX v1.0 section 6.3の最低限のalgorithmを使う。有限amaxの最大2冪をE4M3では256、E5M2では32768で割り、
   E8M0範囲へ収める。elementはRNE＋SATとする。
3. 末尾partial blockはvalid laneだけでscaleを求め、storage paddingをcanonical zeroにする。31／32／33と255／256／257を
   host／GPU oracleへ含める。
4. initial explicit targetは`gfx1201`の`kv-mxfp8-e4`と`gfx1030`の`kv-mxfp8-e5`に限定する。`gfx942`はOCP element byte列と
   FNUZを同一視せず、標準OCP recipeの実装・証拠がない限りunsupportedとする。
5. quality repeatはFP16、block16、標準MXFP8の各residentを完全解放してから次を作り、同時常駐させない。同一sampleのKLD、
   top-1、perplexity、task、long-contextを比較するが、標準MXFP8の結果はblock16 default gateへ流用しない。

## target別default候補

| exact target scope | candidate default | descriptor | value physical variant | promotion scope |
| --- | --- | --- | --- | --- |
| `gfx942:sramecc+:xnack-` | `kv-fp8-e4-block16` | `kv-fp8-e4-block16-v2` | E4M3 FNUZ | 検証したHot Aisle MI300X tuple、model lock、shapeだけ |
| `gfx1201` | `kv-fp8-e4-block16` | `kv-fp8-e4-block16-v2` | OCP E4M3FN | 検証したR9700 tuple、model lock、shapeだけ |
| `gfx1030` | `kv-fp8-e5-block16` | `kv-fp8-e5-block16-v2` | software E5M2 | 検証したV620 tuple、model lock、shapeだけ |
| その他／suffix不一致／未検証 | `fp16`を維持 | 選択しない | 選択しない | 明示的な後続promotionまで変更しない |

- 最初のdefault promotion対象はQwen3.5-4B BF16、single GPU、検証済みtext/full-attention shapeとする。MoE、MTP、vision、
  別head dim、別model lockは個別証拠が揃うまでFP16を維持する。
- model load／session create時にexact target、model capability、shape、policy versionから一度だけ解決する。request長、現在の空きHBM、
  過去のOOM、prompt内容をrouting keyにせず、実行途中でencodingを変更しない。
- 選択優先順位はmodel entryの明示指定、process-wideの明示指定、verified target-default policy、FP16 safety defaultの順とする。
  serverは異なるtargetを持つmodel entryへ一つのglobal defaultを先に解決せず、各entryのexact target確定後に個別解決する。
- 新形式の明示指定がtarget／model／shapeでunsupportedならfail closedにし、FP16や別低bit形式へsilent fallbackしない。
- 三targetは独立に`adopt`、`retain-fp16`、`insufficient-evidence`を決定できる。一targetの未達を別targetの開始gateにしない。
- target-default表とmodel／shape capabilityはversioned runtime policyとして管理し、変換由来の`derived-gguf-lock-v1`へ暗黙追加しない。
  resolved encoding／descriptorはmodel/session/state identityへ入り、policy digestとselection sourceは監査reportへ入る。

## 依存関係

- format、ABI、append、attention、state lifecycleは既存Phase 16／31／41／52契約を使ってPhase 46と並行開始できる。
- default品質判定は[Phase 46保存済み計画](../../../../archive/2026/08/21-31/phase46-conversion-quantization-benchmark-quality-tools.md)が作った
  freeze済みbaseline／thresholdを変更せず、descriptor v2だけへ進めた`kv-cache-default-v2` policyを使用する。
  Phase 46のconverter、LoRA、split/merge等の完了は待たない。
- KV providerと物理byte契約は[KV memory decision](../../../../../architecture/kv-memory.md)を正とし、format追加とdefault採用時に
  runtime architectureと同時に更新する。
- exact GPU実行前に[GPU compatibility](../../../../../compatibility/gpu.md)、
  [AMD GPU compatibility](../../../../../compatibility/amd-gpu.md)、
  [software compatibility](../../../../../compatibility/software.md)を再確認し、実機tupleとtoolchain変更を同期する。

## 作業単位

### A. encoding descriptor・public selection・ABI

1. 新しいencoding enum／ID、canonical name、value variant、block size 16／32、scale dtype E8M0、axis、tail、rounding、non-finite、
   plane layoutをadditive descriptorへ追加する。既存ID、CLI value、serialized stateを再利用しない。
2. public CLI／config／server model settingはcanonical nameを受理し、resolved name、physical variant、descriptor digest、selection reasonを
   machine-readable reportへ返す。canonical name以外の派生名はunknown encodingとして拒否する。
3. target capabilityはlogical E4／E5とphysical OCP／FNUZ／software codecを分離する。exact `gfx942` suffix、`gfx1201`、`gfx1030`の
   code object／runtime target不一致をload前に拒否する。
4. C ABIを拡張する場合はversion／size guard、Rust binding、layout probe、old-client testを同じ変更単位で更新する。
5. Rust core enum／resident-byte accounting、HIP encoding／dtype／plane ID、checked-in binding、CLI/server/chat、Qwen／Gemma exhaustive match、
   prefix/session/state image tag、evidence schemaのclosed enumをinventory化し、追加漏れをcompile/testで検出する。

### B. KV owner・memory provider・append

1. opaque KV ownerへK/V value planeとblock-scale planeを追加し、virtual-contiguous／contiguous-residentのbyte計算、alignment、
   reserve、commit、COW、snapshot、releaseをchecked arithmeticで実装する。
2. appendはBF16入力からblock absmax、format別E8M0 scale、E4M3またはE5M2 valueを一度だけ生成する。K/Vおよびblockごとのscaleを共有せず、
   scale plane orderをdescriptorとhost oracleへ固定する。
3. head dim 15／16／17、31／32／33、255／256／257、token/page/capacity境界、partial block、all zero、tiny、最大有限、NaN、±Inf、signed zeroを
   host codecとexact GPU append oracleで検証する。
4. plane間failure injectionでcreate、grow、COW、append、checkpoint exportの途中失敗を検証し、logical length、mapping、handle、
   accounting、ownerがappend前へ戻ることを確認する。

### C. fused causal attention

1. decodeとprefillの既存semantic op内へblock scale decodeを追加し、K/Vをregisterへ直接展開してFP32 accumulatorへ渡す。
   full mirror、CPU decode、unsupported時のruntime fallbackを追加しない。
2. scalar／independent CPU oracleはquantize→dequantize後のK/Vを用いてcausal mask、GQA mapping、online softmax、BF16 outputを比較する。
   encode oracleとattention oracleを分け、同じbugが相殺されないようにする。
3. query `1/2/31/32/63/64/127/128`、KV `1/15/16/17/255/256/257/1023/1024/1025`、非整列capacityと
   current selector境界の両側をbounded matrixで含める。
4. target固有native conversionを使う場合もsoftware oracleとbyte分類を一致させ、actual launch symbol、physical FP8 variant、wave、
   nonzero dispatch、fallback falseをevidenceへ保存する。

### D. state、reuse、checkpoint、report identity

1. prefix cache、session checkpoint、fork/COW、context shift、speculation、offload/importへcanonical encoding name、descriptor/layout digest、
   physical variant、target semantics、capacity、plane metadata/checksumを結合する。
2. 異なるdescriptor version、scale recipe、block size、scale dtype、E4/E5、OCP/FNUZ、target、capacity、layoutのstateを
   hit／importせず、特にdescriptor v1とv2を明示missまたはrejectにする。
3. all-plane checksumを検証してからfresh ownerへpublishし、旧readerが新packed stateをFP16または現行FP8として読まないよう
   unsupported encodingでfail closedにする。
4. API/CLI debug、benchmark、quality reportへrequested／resolved encoding、default policy digest、selection reason、logical／physical bytes、
   provider、state reuse dispositionを追加する。pointer、raw KV、raw promptはcompact summaryへ含めない。
5. MoE、MTP、vision、Gemmaの既存compatibility分岐を明示分類し、Qwen dense textのdefaultを継承して誤って有効化したり、
   現行の対応経路を無関係に無効化したりしない。

### E. 品質・数値変更判定

1. 量子化recipeを変えて誤差が増えうるため、default変更は[数値変更台帳](../../../../../compatibility/numerical-output-changes.md)の
   **N2**として扱う。scope、memory／performance効果、KLD、top-1、perplexity、task、long-context、最初のtoken/logit分岐、rollbackを
   targetごとに提示する。
2. Phase 46でcandidateを見る前にfreezeしたpolicyを変更せずFP16 baselineと比較する。candidate実行後にthreshold、dataset、sample、
   集約法を緩めた場合は新policy versionとしてbaselineから取り直す。
3. short prompt、long continuation／retrieval、page/capacity境界を含め、同一provider repeatの決定性、finite output、stop、usage、
   state publicationを確認する。token完全一致は記録するが単独gateにしない。
4. targetごとの採否をユーザー決定として台帳へ記録する。本ユーザー指示は形式とtarget方向を承認するが、測定未達やN3を
   PASSへ読み替えるものではない。

### F. target別GPU evidenceとdefault selector

1. 各targetでformat append／attentionの独立数値oracle、full-model品質matrix、resource、性能を同一immutable candidateへ結合する。
   CPU emulation、compile-only、timeout、crash、0 caseをGPU PASSにしない。
2. defaultへ昇格するtargetは、既存の通常5行＋長時間2行をそのtargetのfresh candidateで再取得し、生成、HIP-only、fallback 0、
   cleanup 0、HBM/GTT settled、logical／physical KV bytesを記録する。全7行のllama.cpp速度同等は自動hard gateにせず差を開示する。
3. explicit `fp16`を必ず残し、explicit新形式、auto default、unsupported negative、他target非選択、model／shape scope外をselector testへ含める。
4. 品質policy PASSかつcorrectness/resource条件を満たしたtargetだけdefault mappingを有効化する。performance未達は効果と費用を示して
   target別に採否し、memory削減だけで自動昇格しない。

### G. closeout・rollback

1. target別summaryはsource/tree、binary、model/derived lock、input/dataset/policy、GPU/software tuple、encoding descriptor、全run、
   quality、performance、memory、fallback、cleanup、decisionを持つ。raw evidenceはrepository外へ置きdigest参照する。
2. runtime、KV memory、model lock、GPU／AMD／software compatibility、numerical-output ledger、CLI/API docsを採用mappingと同時に更新する。
3. rollbackはdefault mappingをtargetごとにFP16へ戻せるようにし、新形式の明示選択、state ID、過去summaryを破壊しない。
   correctness/security defect時はmappingだけでなく該当providerを明示unsupportedにする。

## 完了条件

- 両canonical nameのformat、codec、descriptor、state ownership、append、direct attention、explicit selectionが実装され、既存encodingの
  public／serialized意味を変更しない。
- block/tail/non-finite、token/page/capacity、K/V plane、failure rollback、checkpoint/reuse identityのhost・GPU oracleをPASSする。
- `gfx1201`、`gfx1030`についてfresh exact実機証拠を取得し、Phase 46のfreeze済みpolicyでtarget別判定する。
  `gfx942:sramecc+:xnack-`は実機を確保した後の一括検証へ延期でき、未取得をPASSへ読み替えず`insufficient-evidence`のまま閉じる。
- 判定が`adopt`のtargetだけ表のcandidateをdefaultにし、`retain-fp16`／`insufficient-evidence`、未知target、未検証model／shapeは
  FP16のままにする。部分採用でPhaseを完了でき、未採用targetの理由を隠さない。
- default採用はN2のtarget別decision、policy/summary digest、rollbackを数値変更台帳へ記録し、一回のintegration reviewと
  affected final gatesをPASSする。

## 2026-08-27 standard MX scale rule follow-up

ユーザー決定により、`kv-fp8-e4-block16`／`kv-fp8-e5-block16`はblock size 16を維持したまま、descriptor v2／
`StandardMxFloorPowerV1`へ変更する。下記のcorrectness、品質、summary、空mappingはdescriptor v1の
「有限値を飽和させない最小scale」recipeへ
結合した履歴であり、新recipeのcorrectness、品質、default採否へ流用しない。旧metricsとdigestは監査履歴として変更せず保持する。

Phase 53 implementation follow-upは完了した。gfx1201／gfx1030について新recipeのhost／GPU correctnessと同じfreeze済みpolicyによる
full-model品質をfresh binary／report identityで再取得した。両targetともcorrectnessはPASSしたが品質thresholdに未達だったため、
performance/resourceはfrozen early-stopにより実行せず`retain-fp16`とした。default mappingは空のままである。
MI300X `gfx942:sramecc+:xnack-`は将来の一括実機検証へdeferし、今回のfollow-upをblockしない。

### descriptor v2／standard MX scale ruleのtarget別判定

| exact target | format correctness | 3-repeat quality | frozen-policy decision | performance/resource |
| --- | --- | --- | --- | --- |
| `gfx942:sramecc+:xnack-` | fresh Phase 53 reportなし | 未取得 | `insufficient-evidence` | 未取得 |
| `gfx1201` | E4 block16 v2でhost 15/16/17/255/256/257、GPU append 6、direct attention 1、fallback 0、cleanup 0、PASS | 完全直列3 repeat。block16 KLD p99 `0.006562189165612111`、top-1 `0.9`、long-context loss `0.16666666666666663`。MXFP8 KLD p99 `0.004945428206833837` | top-1 `0.99`以上とlong-context loss `0.02`以下を満たさず`retain-fp16` | frozen early-stopにより7行を実行しない |
| `gfx1030` | E5 block16 v2でhost 15/16/17/255/256/257、GPU append 6、direct attention 1、fallback 0、cleanup 0、PASS | 完全直列3 repeat。block16 KLD p99 `0.03659844555378746`、top-1 `0.9`、long-context loss `0.08333333333333337`。MXFP8 KLD p99 `0.03218873133110086` | top-1とlong-context thresholdを満たさず`retain-fp16` | frozen early-stopにより7行を実行しない |

raw evidenceとSHA-256は[対応する履歴](../../../../../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)を正とする。
v2 summaryは`external:phase53/phase53-kv-default-summary-standardmx-v2.json`、空mappingは
`external:phase53/phase53-runtime-mapping-standardmx-v2.json`である。block16と通常MXFP8は同じscale ruleを使うがblock sizeが16／32で
異なるため、scale exponentと量子化値はblock境界ごとに異なり得る。scale rule統一後もKLDが完全一致しないことは実装不一致を示さない。

## 2026-08-27 superseded recipeのtarget別判定履歴

この節はscale rule変更前のimmutable evidenceを記録する。標準OCP MXFP8はpublic name `kv-mxfp8-e4`／`kv-mxfp8-e5`、head-dimension方向block 32、E8M0 scaleの
explicit-only形式として実装した。block16と標準MXFP8はpadded value plane、独立K/V scale plane、append、direct attention、
checkpoint／prefix／state identityへ別descriptorとして追加し、既存encoding IDと省略時FP16を変更していない。

| exact target | format correctness | 3-repeat quality | frozen-policy decision | performance/resource |
| --- | --- | --- | --- | --- |
| `gfx942:sramecc+:xnack-` | fresh Phase 53 reportなし。標準OCP MXFP8はFNUZ element byte列と同一視できないためunsupported | 未取得 | `insufficient-evidence` | 未取得 |
| `gfx1201` | block16 E4M3 OCP、MXFP8 E4M3 OCPともGPU append／direct attention、fallback 0、cleanup 0でPASS | FP16→block16→MXFP8を完全直列に3 repeat。block16 KLD p99 `0.0038687249522990803`、top-1 `0.85`、long-context loss `0.08333333333333337` | top-1 `0.99`以上とlong-context loss `0.02`以下を満たさず`retain-fp16` | frozen early-stopにより7行を実行しない |
| `gfx1030` | block16 E5M2 software、MXFP8 E5M2 OCPともGPU append／direct attention、fallback 0、cleanup 0でPASS | FP16→block16→MXFP8を完全直列に3 repeat。block16 KLD p99 `0.04331390780013198`、top-1 `0.8`、long-context loss `0.16666666666666663` | top-1とlong-context thresholdを満たさず`retain-fp16` | frozen early-stopにより7行を実行しない |

MXFP8はreference-onlyでありblock16 default gateへ流用していない。参考KLD p99はgfx1201 E4が
`0.004945428206833837`、gfx1030 E5が`0.03218873133110086`で、各targetの全3 repeatは同値だった。
runtime mapping候補は空であり、全target、unknown target、scope外model／shapeのsafety defaultはFP16のままである。

追跡対象外raw evidenceは`external:phase53/`に置く。final summary
`external:phase53/phase53-kv-default-summary-v3.json`のSHA-256は
`2440fd7726fca24919731abdcbd2b0f74fdd9d663ecca850b369b5ae3e69dd2b`、空mapping候補
`external:phase53/phase53-runtime-mapping-v3.json`は
`ecc05a91899c9275a7dc1234f418555ad860b18680c2ad15ea7eb745b7127dff`、freeze済みpolicyは
`3e8b1696ebfd485606762d9b3c07fd2694f6157abf43745eabbbc2913240cb1d`である。個別report digestは
[対応する履歴](../../../../../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)を正とする。

2026-08-27のユーザー指示により、MI300X実機検証は追加の検証項目がまとまった将来の一括実行へ延期した。Hot Aisle endpointの
IP疎通はVM存在またはGPU availabilityの証拠に使わない。gfx942は`insufficient-evidence`、default mappingは空のままにして、
compile-onlyまたは過去Phaseの別scopeをPASSへ読み替えない。gfx1201／gfx1030の`retain-fp16`は旧recipeについて確定した
歴史的判定であり、standard MX scale rule版の採否を決めない。品質FAIL後の旧recipe performance未実行は再要求しないが、
新recipeは上記fresh correctness／品質で再判定済みである。旧target別dispositionは監査履歴として保持し、descriptor v2の判定を現行結果とする。

## 対象外

- tokenを跨ぐper-channel scaling、per-channel calibration、要求途中のencoding変更。
- TurboQuant、INT8+scaling、MXFP4、weight／activation量子化、Paged Attention、continuous batching、multi-GPU。
- `gfx1200`、他RDNA2、他CDNA3 SKUやsuffix、別model／shapeへの証拠なしの一般化。
- 現行`fp8`／`fp8-static`／`nvfp4`の削除、名称変更、既存checkpointのin-place migration。

## 停止・再計画条件

- codec差、state corruption、race、non-determinism、原因不明のlogit差、N3が残るtargetはdefault採用せずFP16を維持する。
- scale planeのmemory accessが支配的で性能改善が見込めない場合もformatのcorrectness作業とdefault判定を分離し、測定を隠して
  per-channelや別block sizeへscopeを変えない。
- 実機を確保できないtargetはcompile／host準備を保存して`insufficient-evidence`とし、過去Phaseの別candidate証拠を流用しない。

## Phase closeout

元の作業単位とstandard MX scale rule follow-upをtarget別に閉じた。fresh correctness／品質と新しいtarget別summaryへ
非採用、exact identity、品質、early-stop、known limitationを記録した。Phase 46の他toolやPhase 47／48を
完了条件へ追加せず、defer済みMI300Xをlocal RDNA follow-upのgateにしない。

[全体計画](../../../../main-plan.md) /
[Phase 46保存済み計画](../../../../archive/2026/08/21-31/phase46-conversion-quantization-benchmark-quality-tools.md) /
[対応する履歴](../../../../../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)
