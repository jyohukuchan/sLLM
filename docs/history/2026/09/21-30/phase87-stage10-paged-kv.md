# Phase 87 段階10: Paged KV／Attention本移行

2026-09-25、段階1のGPU候補とは独立した読み取り棚卸しを開始した。
[詳細計画](../../../../plans/archive/2026/09/21-30/paged-kv-full-migration.md)の実装順1について、
現行KV state、形式別plane、公開ABI、graph pointerの差を確認した。production移行は未着手。

## 現行契約と128-token物理blockのサイズ

Qwen3.8のKVは4 head、head dimension 256。keyとvalueの2組を同じphysical block IDで管理する。
量子化groupとtoken blockは別の概念で、現在のABI `block_size`は前者を表す。

| KV形式 | KまたはVのbyte/token | K+Vのbyte/token | K+Vのbyte/128-token block | plane |
| --- | ---: | ---: | ---: | --- |
| FP16 | 2,048 | 4,096 | 524,288 | value |
| MXFP8 E4 | 1,056 | 2,112 | 270,336 | value＋E8M0 scale |
| NVFP4 | 592 | 1,184 | 151,552 | packed value＋block scale＋outer scale |

一般式は`H=kv_heads`、`D=head_dim`、`P32=ceil(D/32)*32`、`S32=ceil(D/32)`として、
KまたはVの1 tokenをFP16なら`2HD`、MXFP8なら`H(P32+S32)`、NVFP4なら
`H(ceil(D/2)+ceil(D/16)+4)` byteとする。物理blockは各値の128倍。
NVFP4の公開C ABIとRust HIP adapterで受理範囲が異なるため、対応表を確定するまでは
「ABIが受理する形式」を「Qwen productionが対応する形式」と同義にしない。

現行create契約とtarget selectorを照合した移行対象の一覧は次のとおり。128-token blockは
量子化group幅を変更しない。表の「既存」はPaged実装のPASSを意味しない。

| encoding | 公開v2 create | Rust HIP adapter | 現行target／制約 |
| --- | --- | --- | --- |
| FP16 | 受理 | 使用 | 既存full-attention経路 |
| dynamic FP8 E4 | 受理 | 使用 | F32 token scaleを保持 |
| static FP8 E4 | 受理 | 使用 | unit static scaleのsliding windowも別契約 |
| NVFP4 | 受理 | 使用 | block16 E4M3 scaleとF32 outer scaleを保持 |
| MXFP8 E4 | 受理 | 使用 | exact gfx1030／gfx1201／gfx942 |
| MXFP8 E5 | 受理 | 使用 | 現行native recipeはgfx1030 |
| 旧FP8 block16 v2 | 現行validatorは拒否 | 退役済み | Pagedへ復活させない |

Qwen3.8 NVFP4の公開CLIはgfx1030／gfx1201、KVはFP16またはMXFP8 E4に限定される。
Rust adapterの現行memory selectorはgfx1201／gfx942を全量resident、gfx1030を
65,536 token未満でcapability-selected、以上でresidentとする。sliding windowはVMM固定。
この差をPagedへ移す際は、既存の対応format／targetを暗黙に消さず、段階ごとに
Paged対応または明示的な未対応判定を実装と照合する。

## 先に固定する状態遷移

- logical blockは`token/128`、block内位置は`token%128`。論理tableの各entryは`u32`のphysical block ID、
  未割当はinvalid sentinelとする。K/Vと全scale planeは同じtableを使う。
- prefix forkは公開済み長さまでのblock IDを共有し、参照数を増やす。部分tailも共有するが、
  親か子が追記するときはCOWする。失敗時は増やした参照数と子tableを元に戻す。
- appendは`prepare→write→publish→retire`とし、block確保・COW・全plane書込み・device table更新を
  一つのtransactionとして扱う。公開lengthとgenerationは全planeとtableの更新完了後に確定する。
  table更新後の失敗は同じstreamで旧entryを復元し、復元不能ならcontextをpoisonする。
- graph captureが保持するdevice table pointerはstate寿命中固定する。block IDとlogical lengthの値を
  replay前に更新し、capture内でpool割当・fork・COWをしない。

物理poolは**segmented slab＋固定device descriptor table**を採る。物理block IDをdescriptorへ引き、
descriptorにK/Vと各scale planeのdevice pointerを持たせる。logical block table pointerとdescriptor
table pointerはcapture前にcapacity分を確保して固定し、slabの実device memoryは必要blockが増えた時だけ
capture外で確保する。次replay前にtable／descriptor entryを同じstream上で更新する。
WU-P1の`base + physical_id * stride`はdescriptor参照へ変わるが、block内のtoken順、softmax、丸めは変えない。
1 physical blockごとに6 planeを別々に`hipMalloc`する形はallocation数が過大となるため、slab内でplaneを
まとめる。全logical capacity分の連続領域を前確保する案はpointerが単純だが、未使用contextにもVRAMを
占有するため既定にしない。slab粒度とdescriptor layoutは、WU-P1 kernelへ接続した際の資源・速度で固定する。

現行のfull-capacity preflightを初期移行では維持し、実allocationだけlazyにする。これにより実行中の
予期しないpool不足を抑えつつ、model＋workspace＋最大要求state＋safety reserveがVRAMを超える設定は
従来どおり明示拒否する。数M contextをこの32 GB GPUで常駐可能と主張しない。

## 公開ABIの差分

現行`kv_state_create_v2`の`block_size`はNVFP4=16／MXFP8=32の量子化groupであり、
128-token blockへ読み替えられない。`SLLM_HIP_KV_MEMORY_KIND_PAGED`とadditiveなcreate情報版に、
token block size、pool上限、table capacity、物理layout versionを別fieldで持たせる必要がある。
`create_v2`はstruct sizeを厳密検査するため、Paged専用create入口とstructを追加する。
現行viewの`physical_page_bytes`／`tokens_per_page`／`mapped_token_capacity`はVMMの意味を持つので、
paged view用の新struct／query入口を使い、旧fieldの意味を上書きしない。fork infoの`page_bytes`と
state image/importも同様にversionで区別する。新しいimage契約ができるまではpaged stateを
旧image形式へ暗黙にexport/importしない。内部block table pointerは公開ABIへ出さない。

現行Rust selectorはR9700の全量residentとV620の65,536-token境界を残し、nativeはVMM pageと
residentのforkを別実装にしている。これらは最終A/B比較後にまとめて撤去する対象である。
段階10の受入には、計算順・丸め・causal mask、非identity table、block境界127/128/129と
65535/65536/65537、prefix fork、tail COW、cancel、pool不足、rollback、graph replayが含まれる。

## 実施状態

棚卸しと状態遷移の初稿に続き、6 planeの固定device descriptor layoutを
`native/hip/src/paged_kv_device_layout.hpp`へ追加した。48 Bの標準layoutでC++17 host compileはPASS。
host所有権metadataの`paged_kv_pool_state.hpp`では128-token blockのrefcount、
append予約／commit／rollback、fork共有、部分tail COW、releaseを実装した。
`phase87_stage10_paged_pool_host_test.cpp`で127／128／129、
65535／65536／65537、pool不足、rollback、forkとCOWを確認し、C++17
`-Wall -Wextra -Wpedantic -Werror`でPASSした。device slab allocationへは未接続。
公開されるappend metadataの改変に備え、prepare時の予約計画をState内部に保持して
commit前に完全一致を検査するようにした。Change削除、範囲内のstart/end改変、
無関係Change挿入、未予約physical IDへの差替え、コピー済みhandleでの二重commitを
所有権変更前に拒否する。rollbackは内部計画から予約を回収し、private tailのin-place追記は維持した。
host警告ありbuildとASAN／UBSANはPASSした。
既存Rust fork契約が子のlogical capacity拡大を許すため、host `State::fork_into`にも
子capacityの指定を追加した。公開済み長さ未満は所有権変更前に拒否し、
129-token親から300-token子へのfork、共有末尾COW、次block割当、親不変、全refcount解放を
host警告ありbuildとASAN／UBSANで確認した。
quiescentな末尾巻戻しもhost `State::rewind_last`へ追加し、257から129 tokenへ戻した際に
不要blockだけを解放し、fork siblingの内容と所有権を保ち、後続の部分tail追記でCOWすることを
同じ警告設定とASAN／UBSANで確認した。
続いて`paged_kv_slab_plan.hpp`で8 physical blockごとのsegmented slabと6 planeのbyte offsetを
checked arithmeticで計算するhost planを追加した。FP16／MXFP8 E4／NVFP4のblock stride、
partial slab、logical境界、overflow、physical IDのinvalid sentinelを
`phase87_stage10_slab_plan_host_test.cpp`で確認し、同じC++17警告設定でPASSした。
このhost設計時点では公開ABI、production attention、graph replayへの接続は未着手だった。

### production移行の独立部品（2026-09-25）

Paged専用のadditive create/view/fork C ABIとRust checked-in bindingsを追加し、
既存v1/v2のmemory kind／`block_size`／VMM page fieldの意味を維持した。
新createにはtoken block幅128、量子化group幅、logical table容量、physical block上限、
layout version、固定FP8 scale、sliding windowを別fieldとして持たせる。
新viewは6 planeのcommitted bytesと合計を返す契約とした。C/Rust ABI layout一致、
host header compile、`cargo check -p sllm-hip-sys --lib`はPASS。
`kv_state_api.cpp`のPaged create validatorはFP16／dynamic FP8／static FP8／NVFP4／
MXFP8 E4/E5のrecipe、128-token blockと量子化groupの区別、128／129境界、
容量とreserved fieldを確認する。従来v2へPaged memory kindを渡す操作は拒否する。
focused host testを警告ありbuildとASAN／UBSANでPASSした。

segmented 8-block物理slab、固定descriptor table、stateごとの固定logical tableを持つ
`paged_kv_device_pool.hpp`も追加した。両exact GPUのdescriptor/table更新・rollback・
cleanupの単体probeはPASSした。FP16／MXFP8 E4のPaged append kernelは既存連続
quantization算術を再利用し、非identity tableと127／128／129境界、固定位置と
device-control動的位置、無効table拒否を両exact GPUの単体probeでPASSした。
この時点ではproduction requestのappend、attention、fork、graph replayは未接続である。

Rust adapterのopt-in createは、一つの親と一つの子が同じprefixからそれぞれ全logical
capacityへ分岐できるよう、`max_physical_blocks = 2 * logical_table_capacity`をchecked
arithmeticで設定し、`UINT32_MAX`以上を拒否する。既定の`create_v2`／legacy queryは維持する。
Qwenの`memory_audit_snapshot`は現状もlegacy `physical_memory()`を読むため、pagedの
`physical_metadata()`／`paged_physical_memory()`へ移行するまでproduction defaultへ切り替えない。

whole decode graphは次のreplayを結果のreadbackより先にqueueへ積み、nativeの
`published_length`は停止時まで更新しない。この契約に合わせ、host `State`へ
`prepare_graph_through`／`commit_graph_reservation`／`finish_graph`を追加した。
graphで将来使うblock mappingだけを段階的に予約し、公開lengthとgenerationは進めず、
停止後に実際の長さより先の予約blockを解放する。共有partial tailのCOW、
2回の増分予約、途中rollback、親不変、最終trimをhost警告ありbuildとASAN／UBSANでPASSした。
device table更新と実graph replayへの接続は未完了であり、このhost PASSをGPU PASSとは扱わない。
graph replayのprivate ABIに`prepare_paged_kv(conservative_end)`を追加し、Rustの
one-ahead controllerから各submit前に「観測済みmodel位置＋2 replay×最大9行」を渡す
接続を加えた。legacy graphではno-op、Paged graphはdevice table予約が実装されるまで
明示拒否する。coreのreplay順序テストとHIP adapterの`cargo check`をPASSした。

Qwenの`memory_audit_snapshot`もtag付き`Vmm`／`Paged` metadataを受け取り、
KV committed bytesをそれぞれ「K/V planeの合計」と「6 planeの合計」で集計するようにした。
CLI、server、benchmarkはPagedに専用block／table／plane byte fieldを出し、
VMM page fieldへPaged値を流用しない。core、server、CLIのaffected host checksと
benchmarkの`cargo check`はPASS。production KV既定はnative request経路の完了まで維持する。

### Qwen3.8 production接続と数値対照（2026-09-25、進行中）

Paged専用の公開append／attention／fork／COW／cancel／rewind、graph事前予約と
capture/replayをnative runtimeへ接続した。Rust adapterはPaged専用providerの
ID・symbol・grid・exact target・fallback禁止を検証する。Qwen3.8 NVFP4の
MXFP8 E4選択時だけproduction sessionがPagedを既定で要求する。FP16 rollbackと
他modelの既存KV経路は、実モデルのPaged確認が終わるまで維持する。
初回state使用のdescriptor／logical table初期化はdevice eventでqueueへ順序付け、
MTP prefix primingの巻戻しはhost pool・device table・公開長を同時に更新する。
Paged GQA6 decodeのworkspaceはgraph capture前に確保し、graph寿命中に固定する。

両exact GPUの公開GPU oracleで127／128／129境界、decode M=1〜5、prefill、
prefix forkとtail COW、cancel、graph replay、Deferred append/fence、
49→48および129→128の巻戻しを確認した。FP16 GQA6/GQA4とdynamic／static FP8 E4、
NVFP4、MXFP8 E5（gfx1030限定）の公開経路も数値対照を通した。
`fallback=0`、cleanup=0を各テストで確認した。

R9700では初期Paged prefillのqtile8経路が旧packed wave providerと数値差を生み、
高エントロピーM47入力のBF16 4要素（最初のindex 185135）でbitwise不一致を確認した。
gfx1201専用Paged wave providerへ演算・還元順を合わせた後、M47/M64 prefillと
prefix47 decode M1/M2がbitwise一致した。M128・短prefixではqtile4、
長prefixではqtile8を旧経路と同じ条件で選び、高エントロピーを含む計20ケースを通した。
Qwen3.8の同一source
旧KV対照とPaged既定のseed固定64-token要求は、R9700で本文一致・MTP draft受理36、
V620でも本文一致・受理35となった。両GPUとも監査は`kv_memory_kind=paged`、
HIPのみ、fallbackなし、正常終了時のrequest/workspace残量とquarantineは0だった。
この短い要求は機能・数値の確認であり、モデル全体の最終性能採否を代替しない。

最終production sourceのM128 prefillをKV65536／131072で同一process AB/BAとし、
両GPUともPaged／連続KVの全測定順で1.10未満を確認した。KV131072では
sequential／reverse logical tableを各GPUで直接構成し、bitwise数値と10%条件を
ともに通した（V620の比は約1.001〜1.002、R9700は約1.007〜1.023）。

Paged imageの追加C/Rust ABI、host validator、native query/export/import、
Rust V2 topology・session API、Qwen V2 state imageまで追加した。
両GPUの公開HIP image roundtripでは129-token MXFP8 E4と1153-token sliding static FP8を
export/importし、欠落section拒否、retained start/ring tag、再append後のbitwise
attentionとcleanupを確認した。V1 checkpoint wireの意味は維持する。
Rust/core→HIPのPaged V2 image往復も両exact GPUで129-token境界、欠落plane拒否、
import後の追記とattention結果一致を確認した。`SLLMCKP2`の独立wireには
descriptor、6 plane、logical table／ring topologyとdigestのfail-closed検証を追加し、
core 702テストがPASSした。Qwen V2 checkpointのcore capture/restore APIは
Qwen tests 103件とclippyを通した。続いてV2 wireへLinear/GDNの
Conv2・Recurrent2・Scratchを含む5 planeを追加し、Qwen3.5 BF16の
24 Linear/GDN層も画像として保存できるようにした。
V620の実モデルsaveでは113,178,226-byteの`SLLMCKP2`を生成し、
Paged監査・save成功・cleanup0を確認した。初回loadはfresh native stateの
Scratch未割当（persistent 4 plane）と保存画像の5 plane差でbridgeが
fail-closed拒否した。保存Scratchを検証して永続4 planeへ復元する契約に修正後、
別起動のV2 load＋suffix生成がHTTP 200、`prompt_tokens=21`／
`completion_tokens=1`、HIPのみ、`kv_memory_kind=paged`、Linear24層、
checkpoint load成功、cleanup/quarantine0でPASSした。
R9700でも同じQwen3.5 BF16 GGUF・V2 save/load＋suffixを実行し、
113,178,226-byteの`SLLMCKP2`、HTTP 200、`prompt_tokens=21`／
`completion_tokens=1`、Linear24層、Paged、HIPのみ、cleanup0を確認した。
V1 wireは変更しない。
FP16 Paged graphはV620でGQA4/GQA6、M1〜5のbitwise一致を確認した。
Ministral形状の公開FP16 Paged probeは両GPUで境界127/128/129、M1〜5、
fork/COW/cancel、独立oracle、fallbackなし、cleanup0を通した。
Gemma sliding static FP8のprovider metadataを追加し、一般形状の公開C APIと
Gemma exact形状のRust admissionを区別した。metadata追加時に一般形状を拒否した
退行は既存の両GPU image roundtripで検出して修正済み。
Gemma4 MoEの初回Paged実モデル試験は、1024-token sliding stateに対し
Rust adapterがlogical tableを8枠と計算した一方、native ringが9枠を要求して
state作成で拒否された。GPU演算は始まっておらずPASSに含めない。
sliding ringの容量を最低9枠、parent-child physical上限をその2倍に修正した。
V620の同じ実モデル17-token prefill＋17 decodeを再実行し、
cancel/replay、finite出力、fallbackなし、cleanup0でPASSした。
R9700も同じartifact・17＋17・cancel/replay・finite出力・fallbackなし・
cleanup0でPASSした。V620/R9700のprefill／decode観測は
2178/9949 ms、2875/8295 msで、測定は機能smokeの1回だけである。
Gemma4 MoE production監査の固定`static-fp8-sliding`表示は、全30層の
authoritative physical metadataから求める方式へ修正した。全層Pagedなら
`kv_memory_kind=paged`とし、sliding/full層でtable容量・plane byteが
異なる値は誤って合算・同一視せず`None`とする。host testとcargo checkを通した。
Gemma4 MoE production sessionはcanonical両target・checkpointなしで
Pagedを選ぶよう接続した。後続のV2 checkpoint実装でcheckpointありも
Paged V2を選択し、既存V1 fileだけは明示したlegacy fallbackを一時維持する。
serverが要求する同一artifactのderived GGUF/lockは手元に無いため、
HTTP実モデルsmokeは未実施。上記の両GPU実モデル証拠はdirect core経路である。
CLIのGemma4 MoE generation／benchmark sessionもcanonical両targetでは
Pagedを選び、他targetは明示拒否するよう接続した。CLI host tests84件はPASS。
CLI用の準拠GGUF/lockがないためCLI GPU実モデル実行は未実施であり、
direct core証拠をCLI証拠へ読み替えない。
Gemma4 MoEのcore V2 state image/checkpointは30層のfull logical tableと
sliding ring、static FP8 descriptor/scaleをfail-closedに扱うよう追加した。
Gemma MoE host tests19件（V2 oracle2件を含む）とcore check/clippyはPASS。
serverのV2 save/load、既存V1 fileのlegacy fallback、terminal argmax marker復元、
Paged session選択まで接続し、server host tests135件とclippyをPASSした。
同一artifactの準拠derived GGUF/lockがないためHTTP GPU smokeは未実施で、
既存safetensors actual modelのdirect V2 export→`SLLMCKP2` encode/decode→
fresh restore＋suffixを両GPUで確認した。30層full/sliding、finite出力、
cancel/replay、fallbackなし、cleanup0。resident bytesは各17,636,771,900、
prefill17＋decode17、state capacity1024。これはdirect core GPU証拠であり、
HTTP serverの同一artifact実機証拠ではない。
Gemma4 denseのcoreには、full KV logical tableとsliding KV ringを含む
V2 state imageとcheckpoint export/importを追加した。static FP8 scale、descriptor、
長さ、layer/ring topologyをfail-closed検証し、core 729テスト・server check/clippyが
PASSした。R9700の初回保存要求はlayer5 Paged attentionが
「非unit static FP8 KV scale＋明示score scale」を拒否し、
checkpoint書込み前にHTTP 500となった。旧経路のimplicit scale契約へ揃えた後、
R9700の保存要求はHTTP 200、4 output、cleanup0で、22,037,623-byteの
`SLLMCKP2`を生成した。別起動load＋suffixはnative image importが
`paged image import table sections are incomplete`で拒否した。
sliding画像のlogical table section要件とRust exportのring専用sectionを
照合し、ring IDからlogical table mirrorを再構成しnative側でringとの一致を
検証するよう修正した。現sourceのfresh save→別起動load＋suffixを
R9700/V620の両GPUで実行し、それぞれHTTP 200、`SLLMCKP2` 22,037,623 byte、
Paged V2、HIPのみ、request/workspace残量0、shutdown cleanup0を確認した。
V620のsave時に別モデルサーバーが一時重複したため、loadは競合終了後に単独で確認した。
Qwen3.5 MoE FP16のtracked公開Paged oracleは両GPUでQ16/KV2/D256、境界、M1〜5、
fork/COW、cancel、fallbackなし、cleanup0を通し、exact target・checkpointなしの
production sessionへ接続した。同一Phase20 GGUF/lockの短いserver smokeは
GPU allocation前のGGUF検証でvisual rank5 tensorを現parserが拒否した。
artifactのsize/hash/semantic IDは一致するが、ggmlの[GGUF v3仕様](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)では現行tensor
次元上限は4であり、このartifactのrank5は非準拠と確認した。
parser上限は緩めず、準拠変換物が得られるまでPaged実要求PASSには含めない。
parser/writerはrank0・5・6をfail-closed拒否する現契約へ復元し、
`gguf_contract` host 12件がPASSした。再生成時はvisual tensorを要素数・byte長を
保つrank4 shapeへpackし、GGUFとderived lockを作り直す必要がある。
Qwen3.5 dense MXFP8 GGUFのcheckpointなし経路はV620 `gfx1030`の短い
17-input／1-output実要求でHTTP 200、出力`Okay`、`kv_memory_kind=paged`、
HIPのみ、fallbackなし、shutdown時request/workspaceとquarantine 0を確認した。
この時点の候補binaryの証拠であり、後続のQwen vision数値差で共通GGUF sessionの
Paged選択を一時差し戻した。その後GQA4 Paged prefillの数値修正後、
同じPhase62 MXFP8 W8A8 GGUF＋MXFP8 E4 KVを両GPUでlegacyと
固定seed短要求へ照合し、ともに本文`Hello`、HTTP 200、HIPのみ、
fallbackなし、cleanup0、`kv_memory_kind=paged`を確認した。
Qwen3.5-4B fingerprint・exact両target・checkpointなし・adapterなしの
quantized dense経路だけPaged既定へ再採用し、quantized visionはFP16 KV必須条件で拒否する。
Qwen3.5のOCP FP8 E4M3FN outer-F32重みGGUF（`gguf-native`）は
R9700だけが既存FP8 provider対象で、明示FP16 KV・checkpointなしの
同一seed短要求をlegacy/Pagedとも本文`Hello`、HTTP 200、HIPのみ、
fallbackなし、cleanup0で確認した。このexactレシピだけPaged既定へ加えた。
V620は既存`select_gguf_fp8_provider`が起動時拒否するため、対応を主張しない。
Qwen3.5 BF16 visionは両GPUでPagedのHTTP要求・HIPのみ・cleanup0を確認したが、
同一artifact・画像・seedのlegacy対照とtoken列が異なった。R9700は
Paged `[3212,294,662,13]`／legacy `[51623]`、V620は
Paged `[1414]`／legacy `[3212,294,662]`で、V620別seedでも差が再現した。
数値一致をFAILとしてvisionのPaged production選択を差し戻し、
実vision graphのattention形状・providerと高エントロピー入力を切り分けた。
M>=64のlegacyは`gqa4_shared.v6`、Pagedはgeneric FP16だったため、
同じwave還元・online softmax・BF16 RNE順のPaged GQA4 shared providerを追加した。
両GPUで高エントロピーM65/85/265、逆順table、境界127/128/129の
bitwise旧経路対照と独立oracleをPASSした。occupancyは両targetとも
16 waves/SIMDでspillなし。混在GPUでの実モデルrunはdurable quarantineが出て
証拠から除外した。単独GPUでtext-only M265とvision M85を再照合し、
V620はそれぞれ`[27775,383]`／`[11173]`、R9700は`[27775,383]`／
`[56127]`が同一targetのlegacy/Paged間で一致した。全case HTTP 200、
HIPのみ、fallbackなし、cleanup0。exact target・BF16 dense・FP16 KV・
checkpointなし・vision manifestありに限定してPaged既定を再有効化した。
Qwen3.8 FP16 KVもV620の固定seed・1-output実要求でlegacyとPagedの
出力`Okay`が一致し、HIPのみ、fallbackなし、cleanup0を確認した。
この短い条件の経過時間はlegacy約2.56秒、Paged約2.77秒で、
性能採否の代わりにはしない。R9700でも同一seed・1-outputの
legacy/Paged出力`Hello`が一致し、HIPのみ、fallbackなし、cleanup0を確認した。
これを受け、Qwen3.8のFP16 KVもPaged既定条件へ加えた。
長い出力とモデル全体の性能確認は残る。
Ministral3 FP16のPaged公開probeは両GPUでPASSしたが、R9700の実モデル
同一source A/Bでも同一seed・543-token promptでPaged `ok`、legacy
`Understood—here`と出力が異なった。M543を加えた独立FP16 attention probeは
bitwise一致したため、実モデルのprefill／KV統合を切り分ける。
correctness blockerとしてMinistral productionのPaged既定は差し戻した。
同一processの実モデルlogits比較ではprefill最終行の131,072語彙中
118,892要素にBF16差（最大絶対0.0625）があり、最初のdecode tokenは両方1115だった。
高エントロピーM543の独立attention入力でも最初の差をrow10/head30/dim7で再現した。
低エントロピー対照だけでは見えない差であり、append後K/V planeと
attentionの還元順を切り分けた。legacy raw FP16とPaged imageのappend後K/V planeは
全bit一致。gfx1201 legacy M543はwave還元、Paged genericはworkgroup還元を
選んでいた。Paged FP16 genericのgfx1201 `M=1`または`M>=32`を同じwave還元へ
合わせた後、高エントロピーM543のbitwise・独立oracle・fallbackなし・cleanup0を
確認した。実モデルの同一process再照合でもprefill最終logitsの全語彙hashと
decode logits hashがlegacy/Pagedで一致し、mismatch0、最初のtoken1115、
HIPのみ、fallbackなし、cleanup0だった。修正後のPaged production serverへ
同一seed・543-token固定要求を送ると、legacy対照と本文
`Understood—here`、4 output、finish=`length`が一致し、HTTP 200、
backend memory cleanup0だった。これを受けcanonical両targetに限って
MinistralのPaged既定を再有効化した。R9700の実モデル証拠であり、
V620の同一GGUF実モデル要求は未実施。
### Qwen3.8の短いモデル速度対照（2026-09-25）

同一artifact／固定fixture `129 input / 最大17 output`、MXFP8 E4 KV、
MTP幅2、固定sampling seed、各1 warmup＋2 measuredで通常benchmarkを実行した。
fixtureが早期EOSになり全runの生成は5 token、MTP受理は2/4、
token列は両targetでPagedと旧経路が一致した。TPOTは4 decode transitionの
限定値で、長い生成の代表値とはしない。誤って8192/128を選んだrunと
古いStage12 binaryのrunは受入証拠から除外した。

| exact GPU | KV経路 | prefill ms | TTFT ms | TPOT ms | E2E ms | KV committed bytes（全16層） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V620 `gfx1030` | Stage12最終legacy VMM | 682.808 | 711.568 | 62.643 | 962.224 | 402,653,184 |
| V620 `gfx1030` | Stage10 Paged | 679.862 | 708.190 | 60.477 | 950.166 | 34,603,008 |
| R9700 `gfx1201` | Stage12最終legacy resident | 401.839 | 431.117 | 47.260 | 620.225 | 346,030,080 |
| R9700 `gfx1201` | Stage10 Paged | 400.977 | 429.064 | 47.040 | 617.296 | 34,603,008 |

Paged／legacyのbinary SHA256はそれぞれV620
`a270e49fe24068497e80fd8b12136b7891dad1dc1b78b069720fba96f4a0ec84`／
`06c0fb9651a8822d7a1e5b93082b6127668d774e919b39542b8313c5229648fd`、
R9700 `ab55f55fefacbca3b2c66c984e652cc1b9e4c48df0f556b28ac42d4bb10588ff`／
`c091dbf28c25e1d685f44e7b640b051c08c6d5d1f27b59ab1a74eb473737165e`。
legacyはStage12最終候補、PagedはStage10 sourceであり、速度差をpaging単独に
帰属しない。両経路ともHIPのみ、fallbackなし、cleanup/quarantine0。
execution-session allocator high-waterは両経路とも24,848,429,088 byteだった。
別の単回`129/17` runにROCm物理VRAM監視を併走させ、V620では
Paged 26,160,230,400 byte（432標本）、legacy 26,528,915,456 byte
（373標本）のpeakを観測した。差は368,685,056 byte（約1.39%減）。
監視付きrunの時間を上表の速度中央値へ混ぜない。
KV append／graph個別時間は残件。
R9700も同じ別runでPaged 26,005,495,808 byte（389標本）、
legacy 26,628,718,592 byte（435標本）の物理peakを観測した。
差は623,222,784 byte（約2.34%減）。両GPUの監視付きrunも同一token列、
HIPのみ、fallbackなし、cleanup0であり、終了後のVRAM使用量は起動前へ戻った。

### Paged KV append／graph費用と最終hot shape

公開GPU probeで`M=1〜5`のKV append completion HIP event timingと、
graph span executeからfence waitまでのhost wall時間を別々に記録した。
初回`M=1`はpool初回確保を含むため、後続の`M=2〜5`ではV620の
appendが約97〜103 µs、graph wallが約301〜310 µs、R9700では
append約40〜60 µs、graph wall約150〜230 µsだった。
graph completionのGPU event timingは公開ABIで未対応であり、wall値を
kernel単体費用や通常モデルTPOTと混同しない。全caseは数値oracle、
fallbackなし、cleanup0でPASS。

最終sourceの`M=128`／KV65,536・131,072のAB/BAと、KV131,072の
sequential／reverse block tableを両GPUで再実行した。
Paged／legacy kernel時間比の最大はV620 `1.01062`、R9700 `1.04757`で、
10%線を下回った。両mappingともbitwise一致、独立oracle、fallbackなし、
cleanup0を確認した。`reverse_logical_table`は公開ABIから直接変更できないため、
専用native probeで構成した。

この測定時点では他modelの実要求と旧VMM経路の撤去が残っていた。

### `gfx942`の現行target扱い（2026-09-25ユーザー決定）

MI300X `gfx942`には旧resident経路のPhase 12/36実機証拠があるが、
Paged KVの実機証拠はない。ユーザーは段階10の完全移行時に
`gfx942`を明示`unsupported`へ変更すると決定した。過去の
`project-verified`記録は当時のsource/tupleの履歴として保持し、
現行runtimeの起動時fail-closed拒否、ABI/selector、CI、互換性文書を
旧resident経路の撤去と揃える。Rust sessionはnative context前、公開C contextは
HIP device設定・allocation前に`gfx942`とfeature付き名を明示拒否するようにした。
Rust focused testとstrict native hostの`--gfx942-context-only` probeはPASS。
strict host全体では旧NVFP4 baselineとVMM fork testにgfx942 contextを
作成できる古い期待値が残り、これをunsupportedへ更新した後、
`production public runtime host fault test: PASS`となった。
この時点では旧residentコード撤去が未完了だった。

### 旧経路退役と段階10完了（2026-09-25）

最終AB/BA後、nativeの旧VMM／resident state作成、grow transaction、
token-major append／contiguous attention providerとselectorを撤去した。
旧C ABIの番号は予約値として残し、create／view／fork／raw-plane image入口は
`UNSUPPORTED`を返す。Rust HIP adapterの通常KVはPaged create／query／fork／V2 imageへ固定し、
`with_paged_kv`の実行時切替を削除した。server／CLIのGemma4 MoEと
Qwen persistent chatもPaged V2 checkpointへ揃え、旧V1 checkpoint fileは
形式を識別した上で再生成を求める。non-KV linear planeのV1形式はV2内で維持した。

公開GPUテストは旧連続KV対照を取り除き、Paged＋独立oracleへ移した。
exact `gfx1030`／`gfx1201`の両方で、Paged state、requestの127／128／129境界と
M=1〜5、FP16・各KV形式、graph／cancel／deferred append、rewind、
V2 image import後の追記がPASSし、fallback 0／cleanup 0だった。
strict native host全件、Rust HIP lib 172件、server production 41件、
CLI 84件と該当check／clippyをPASSした。`gfx942`とfeature付き名の
context作成はGPU allocation前に`UNSUPPORTED`となることをfocused hostで確認した。

退役後sourceでのQwen3.8短い`129/17`単回smokeは、V620／R9700とも
生成ID `[1754,18169,15060,13,248046]`、MTP受理2/4、HIPのみ、
fallbackなし、cleanup0だった。report SHA-256はV620
`46224b97cd93383e59611d5ad4ca1ff8e5d6d7cfa41305092dde83975d2fbf5d`、
R9700 `eddabb27c4d945cb802865ae123221a6fcc57a6627a687c71fa7c38681c97a3a`。
binary SHA-256はそれぞれ
`214bf8cdb9f57169e17508ecb90d99a3684137b7f7e95a1a1ff31a0ec6fc1d5e`、
`a6617ab4a010da55c72be36f33472c756e1d7f75c65581647fe16654c6650c48`。
この単回runの時間値はcold pathを含み、先の性能対照や段階11基準へ混ぜない。

統合レビューの旧kernel呼出し指摘は、Stage5／Phase83の追跡済み
履歴test sourceが現行CMake／CIに非登録であることを確認した。
未追跡の古いStage10 append対照sourceは削除し、現行公開testはPaged-onlyで
両GPU実行済み。CPU-only stubの`HIP_UNAVAILABLE`はHIP runtimeを含まない
buildの契約であり、HIP buildの旧ABI `UNSUPPORTED`と区別した。
準拠GGUF／lockがないQwen3.5 MoEとGemma4 MoEのHTTP／CLI実モデル実行、
MinistralのV620同一GGUF実要求は追加証拠なしと明示し、
対象kernel・direct modelの実機証拠範囲を超えて一般化しない。
段階10は完了し、Phase 87全体は段階11とユーザー確認まで継続する。

対応する計画: [Paged KV本移行](../../../../plans/archive/2026/09/21-30/paged-kv-full-migration.md)。
