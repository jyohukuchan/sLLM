# Phase87 段階0: NVFP4／FP8候補の棄却・保留棚卸し

## 目的と判定方法

Phase87の段階0で、Phase78〜85に試したNVFP4 W4A4、FP8 W8A8 decode、prefill内の量子化・一時展開候補を再提案しないための台帳である。ここでいう「棄却」は、数値検査に失敗したという意味に限らず、候補の対象scopeで性能上の採用根拠がなくなったことも含む。「保留」は、数値・資源・production測定のいずれかが不足し、採否を閉じていないことを示す。

Phase87の正本は[Phase87計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)である。Phase82の候補別台帳は[削除したmatmul候補と試行結果](../1-10/phase82-retired-matmul-candidates.md)、既定化しなかった経路は[Phase82の既定採用範囲](../1-10/phase82-default-adoption-scope.md)を正とする。以下の「再検討時に確認すると有用な事項」は追加探索や新しい完了条件を作るものではなく、Phase87で同じ仮説を無条件に繰り返さないための参考情報である。Phase87の測定回数・受入条件はPhase87計画を優先し、過去履歴の4-copy／3+10条件を新しいhard gateへ昇格しない。

変更したkernelについて、Phase87計画が求める独立FP32 oracle、非整列・境界ケース、finite／fallback／cleanupなどの正しさ確認は引き続き必要である。ここで任意扱いにしているのは、過去の性能測定のcopy数・反復数・交互順序であり、正しさ確認そのものではない。

## 現在の基準として維持し、棄却候補に数えない経路

段階0では、次の経路を棄却案として再計測しない。いずれも現行selectorまたは限定されたrollback／比較経路であり、Phase87の新しいW4A4／W8A8 decode測定のcontrolを決める材料である。

| 経路 | 現在の扱い | Phase87での扱い |
| --- | --- | --- |
| NVFP4 activation quantizer wave8 | BF16→E2M1、block16 scale、出力契約を維持するN0。gfx1030／gfx1201の26 fixtureでNumPy oracle、canary、cleanupをPASSし、既定化済み | W4A4 activation quantizationのcontrolとして、M=1〜3と非整列K/Nを再測定する。wave8採用自体を再審査しない |
| NVFP4 decode ID67 wave4/col32 | gfx1030／gfx1201、M=1、Kが16の倍数、N>=1024の限定既定。integer dot4と固定reduction treeを使うN1 | W4A4 decodeの現行control。ID65やprobe-local multi-column候補と混同しない |
| NVFP4 decode ID84 scale LUT | exact `(K,N)=(5120,17408)/(17408,5120)`、両targetの限定既定。E4M3 scaleの入力項・累積順を維持するN0 | exact tupleのcontrol／rollback。generic shapeへ拡張する根拠にはしない |
| FP8 outer decode ID82 LDS LUT | gfx1030の4 tuple限定既定。E4M3FN→FP16、FP32 accumulation、BF16 RNEを維持するN0 | FP8 decodeのgfx1030 control。ID82の既存LUTを「未検証候補」として再実装しない |

根拠は[Phase82既定採用範囲](../1-10/phase82-default-adoption-scope.md)と[Phase82 evidence ledger](../1-10/phase82-optimization-evidence.json)である。wave8 quantizerの全model性能寄与、ID67のscope外数値同等性、ID84／ID82のgeneric shape性能は、そこから主張しない。

## 明示的に棄却された候補

| 候補・主なsource | 仮説 | 観測結果・棄却理由 | 再検討時に確認すると有用な事項（提案） |
| --- | --- | --- | --- |
| ID65 `matmul.nvfp4.w4a4.decode.columns128.v1`（[Phase78 NVFP4 decode probe](../../../../../native/hip/tests/phase78_nvfp4_gfx1030_decode_probe.hip.cpp)） | 128 thread／128 columnのserial-Kと大きなLDSで、複数列を同時に処理すればID58より速くなる | 21-case oracleはPASSしたが、QwenのV620 wide/downはID58の0.779/0.786 msに対し0.949/1.388 ms、R9700は0.538/0.520 msに対し2.789/1.035 ms。Gemmaでも採用ID67を上回らず、Phase82で削除 | ID67のwave4/col32 body、target、M=1、K/N形状を超えて、active blocksとcolumn reuseが実モデルで同時に改善する根拠 |
| ID69 `nvfp4.w4a4.prefill.gfx1201.wmma_f16scale128x64` | block scaleをLDS ingressでFP16 WMMAへ先に適用すれば、ID64のscale hot loopを減らせる | 初回ID64比4.8〜7.2倍遅く、loop展開修正後も約2.1倍遅い。oracle／repeatは成功したが速度で棄却 | scale ingressのresource使用量とWMMA／VALU依存を変える構造変更、かつID64比のproduction shape測定 |
| ID70 `fp8.outer.prefill.gfx1030.f16_staging` | E4M3をFP16へ展開してrocBLAS FP32 GEMMへ渡せば、FP8 software tileより速い | steady比較でID71より約39%遅い。数値不具合ではなく、展開とconsumerの合計コストが理由 | 変換を常駐化・融合する新しいconsumer設計と、ID71との同一production条件比較 |
| ID80 `nvfp4.w4a4.prefill.dp4a64x64_k128` | K128 stagingでDP4Aのload／scale再利用が増える | standalone加重値は約1.056倍だったが、512-token cold full-modelは2,132.2→2,129.9 msで実質差なし。Phase82で未採用 | 実modelで再現する利益、またはK128 stagingの同期・cache挙動を変える新設計 |
| ID81 `nvfp4.w4a4.prefill.gfx1201.wmma128x32` | N32向け小tileでoccupancyまたはtailを改善できる | 512/32、1 warmup＋3 measuredでID64 872.680 ms→895.151 ms（2.575%退行）。token hashは一致 | N32のresource／occupancy条件が変わったことを示す新証拠 |
| ID83 `nvfp4.w4a4.prefill.gfx1201.fp8_staging` | NVFP4をFP8 planeへ展開しhipBLASLtへ渡せばWMMAより速い | stage oracle／finite／repeatは通ったが、`6×448=2688`がFP8最大有限値448へ飽和し、ID64と非等価。追加丸めを含むN3としてselectorから隔離 | 再符号化による丸め・飽和を除去し、ID64と同じ実数式・誤差boundを示した上でproduction比較 |
| ID85 `fp8.outer.prefill.gfx1030.lds_lut.64x64` | FP8 ingressをLDS LUT化すれば変換分岐を減らせる | tiny oracle／BF16一致、probe加重1.299548倍でも、v3-r2 production複合比較でV620 512 prefill 1777.504→1845.994 msへ退行。v3-r3で無効化 | 現行ID71 controlに対するproduction単独比較で、旧micro結果を再現しない一貫した利益 |
| ID86 `fp8.outer.prefill.gfx1030.f16_tile` | FP16 tile stagingでFP8 consumerの変換をまとめる | stage 36.395 ms、consumer 777.696 msで、stageを除いてもID71 consumer 711.526 msより遅い。512 prefill約5.36%退行 | stagingなしのconsumer自体を改善する新しいtile／resource設計 |
| V620 NVFP4 transient int8 staging（`phase78_nvfp4_gfx1030_i8_staging_probe.hip.cpp`） | resident NVFP4をvalue*2のint8へ一時展開し、同じ64x64/K32 DP4A bodyを消費すればdecode／prefillを速くできる | 数値一致したが、stage込み総時間が全caseで退行し、非採用 | staging kernelを追加しない融合、またはstage費用を上回る実model単体証拠 |
| r24 NVFP4 exact-tuple constantization | `K5120/N17408`と`K17408/N5120`を定数化してloopの境界判定・address計算を除く | 114/114の数値・repeat・guard・finite・cleanupはPASSしたが、4-copy中央値平均でwide 127.8405→128.11025 us、down 131.3305→131.3705 us。active blocksは変わらず、追加探索停止 | runtime引数削減がactive blocksまたはload／reductionを変える新しい構造根拠 |
| r26 NVFP4 signedpack | 6 permを2 perm＋整数演算へ置換し、dot／scale／FMA／reduction順を保つ | host全65,536符号組合せ、両target oracle、全出力、repeat、cleanupはPASS。しかしV620 wide/down 128.6185→131.03875／132.4085→134.17875 us、R9700 106.009→106.569／106.389→106.839 usで退行 | perm削減に伴う整数演算増を消すISA／load設計 |
| gfx1201 FP8 hipBLASLt rank7→rank8 | 別rankを選べばM=1 FP8 native decodeが速くなる | 4 copyの交互比較でrank7 0.124422 ms、rank8 0.124042 msの約0.3%差に留まり、過去の約2.2倍差を再現しなかった。cleanup戻り未確認のため採用G1にも使わず、rank7維持 | 同一query、cold copy、cleanupを含む再現可能な優位と、algorithm変更のN1誤差bound。4-copyは履歴で採用した測定設計の例であり、Phase87の1+3条件を置き換えない |
| FP8 full-family FP16 resident cache | FP8 weightsまたは展開済みtileを常駐させれば、decode／prefillの変換を省ける | ID86 on/offでstageをゼロと仮定してもconsumer 777.696 msがID71 711.526 msより遅い。容量は約3.020 GBの追加余地があるが性能根拠なし | resident cacheの変換・consumer両方を改善する設計と、VRAM／GTT／実modelの再測定 |

数値検査が通った候補も、性能または意味保存の条件を満たさなければ再提案しない。特にID83は速度比較より先にN3で閉じており、FP8再符号化をNVFP4 W4A4 decodeへ流用しない。

## `native/hip/tests/phase78_*` probeの対応が未確定な候補

Phase82は、probe-local候補名を製品IDへ推測で割り当てなかった。したがって次のsourceは「各枝が個別に棄却済み」とは扱わず、**probeに候補仮説が書かれているが、現行台帳でbranch別のproduction性能判定が不足している**ものとして記録する。この区別を崩すと、Phase87で同じ候補を再試行するか、逆に未確認の候補を棄却済みと誤認する。

| probe | probe内の仮説 | 現在の結果の読み方 | 再検討時に確認すると有用な事項（提案） |
| --- | --- | --- | --- |
| [FP8 activation shared](../../../../../native/hip/tests/phase78_fp8_gfx1030_decode_activation_shared_probe.hip.cpp) | activation rowを一度FP16へ展開してdynamic LDSで8 waveが共有 | 4-column／8-column候補を含むstandalone probe。ID82／ID68へのbranch別identityとproduction A/Bはこの台帳で確認できない | exact Qwen FP8 233 tensorのoccurrence加重、両GPU対象shape、activation quantizer込みのID82対照 |
| [FP8 direct wave](../../../../../native/hip/tests/phase78_fp8_gfx1030_decode_direct_wave8_probe.hip.cpp) | LDSを使わずwave8 sequential／pair2、またはwave4 pair2でactivationを直読 | oracleを保ったprobe-only候補。VGPRとoccupancyを含むproduction結果が未固定 | M=1 decodeの4 cold copy交互測定、quantizer込みwall、全FP8 shapeへのdispatch証拠 |
| [FP8 decode LUT](../../../../../native/hip/tests/phase78_fp8_gfx1030_decode_lut_probe.hip.cpp) | 256-entry FP16 LUTをconstantまたはLDSへ置く | 旧probe自体はprobe-only。後続のproduction ID82は別identityとして限定採用されたため、probe枝をID82へ遡及しない | ID82 controlとのsource／binary identity対応と、未対応shapeを含めない明示scope |
| [FP8 software pipeline](../../../../../native/hip/tests/phase78_fp8_gfx1030_decode_pipeline_probe.hip.cpp) | K=2／4の独立chunkを先読みしてdecode／dotと重ねる | probe-localのdirect wave4 K-pipeline。現行台帳にproduction selector接続の性能判定なし | load／decode／dot順を固定した全FP8 decode shapeのAB、resource、quantizer込みモデル結果 |
| [FP8 gfx1201 rank sweep](../../../../../native/hip/tests/phase78_fp8_gfx1201_rank_sweep_probe.hip.cpp) | hipBLASLt M=1 outer-vector rankを選ぶ | rank7が現行基準、rank8再測定は上記のとおり非採用。probeの他rankは未確認候補として扱う | cold copy、cleanup、algorithm誤差boundを伴うrank別比較。4-copyは過去の再現性向上案であり、Phase87の必須条件ではない |
| [NVFP4 gfx1030 decode](../../../../../native/hip/tests/phase78_nvfp4_gfx1030_decode_probe.hip.cpp) | Packed64、Wave8Col64、ActivationSharedでactivation／weight reuseを増やす | Phase82がID65への明示identityを確認できずprobeを保持。ID65の棄却を3枝へ個別適用しない | branchごとのsource identity、M=1実形状、全出力oracle、両targetまたはtarget限定scope |
| [NVFP4 gfx1201 decode shared](../../../../../native/hip/tests/phase78_nvfp4_gfx1201_decode_shared_probe.hip.cpp) | activation rowとblock scaleをworkgroup LDSへ一度置く | ID67／ID84の現行selectorへ直接対応するbranch別判定は未固定 | activation quantization＋scale load込みのID67対照、VGPR/LDS、Qwen wide/down |
| [NVFP4 decode dot](../../../../../native/hip/tests/phase78_nvfp4_decode_dot_probe.hip.cpp) | signed-byte DP4A、gfx12 sudot4、scalar FP8 dot4を比較 | 現行ID67は固定DP4A/reduction契約。sudot4／scalar枝の採否証拠はこの台帳にない | 同じE2M1 block16、scale、reduction順の独立FP32 oracleとM=1実model A/B |
| [NVFP4 decode prefetch](../../../../../native/hip/tests/phase78_nvfp4_decode_prefetch_probe.hip.cpp) / [gfx1030 prefetch](../../../../../native/hip/tests/phase78_nvfp4_gfx1030_decode_prefetch_probe.hip.cpp) | ID67／ID84のP2／P4 weight・scale先読みでload latencyを隠す | 後続r21ではP2 bodyを保持したが、weight-only追加案はDCE確認後に追加測定なし。probeの各P枝を全面採用・棄却とはしない | 同一target／PCI、cold copy、交互測定、activation quantizer込み、cache／clock条件の固定。過去の4 cold copy／3+10交互は再現性向上の設計例であり、Phase87の必須条件ではない |
| [NVFP4 scale LUT](../../../../../native/hip/tests/phase78_nvfp4_gfx1030_decode_scale_lut_probe.hip.cpp) / [kernel verification](../../../../../native/hip/tests/phase78_nvfp4_decode_scale_lut_kernel_probe.hip.cpp) | E4M3 block scaleをdirect／constant FP16／LDS FP16／LDS FP32 LUTへ置換 | ID84は限定exact tupleで別のproduction identityとして採用済み。probe枝全体をgeneric decodeへ広げない | exact tuple外のshape、activation quantizer込み、ID67対照でのscope限定測定 |

この節のprobeは、ファイルが存在することだけではPhase87のGPU PASSにも棄却判定にもならない。Phase87段階0で計画されているのは、probeをそのまま再実行することではなく、現行W4A4／W8A8 decodeの1 token内訳、activation quantizer時間、対象shape、読出しbyte数、到達可能帯域との対応を取ることである。過去probeの4-copy／3+10条件は、必要に応じて再現性を高めるための任意の証拠設計案であり、段階0の完了条件ではない。

## Phase83.5以降の関連再試行

Phase83.5では、W4A4 small-Mの4-row shared-LDS候補が両tuple・M2〜4でoracle／repeat一致したが、全ケースで遅く、control/candidate速度比0.191〜0.522で棄却された。原因候補は追加stage同期であり、register共有案へ切り替えた。この候補はMTP verifyのM2〜3に関係するため、Phase87段階1でM=2〜3を測る際に同じ4-row LDS案を再提案しない。[Phase83.5履歴](../1-10/phase83-5-llama-guided-performance.md)

一方、NVFP4 small-M register row reuse、ID94、FP8 ID92等はscope限定で採用・統合された候補であり、棄却一覧へ混ぜない。Phase87のM=1〜3測定では、これらを無条件に現行W4A4 decodeへ流用せず、対象形状と実装identityを確認する。

## 段階0での再検討条件の要約

- W4A4 decode: 現行ID67／ID84の同一算術契約をcontrolに固定し、M=1〜3、非整列K/N、activation quantizer込みの独立FP32 oracleと実効byte数を取る。ID65、Packed64、Wave8Col64、4-row LDS、signedpackを名前だけで再利用しない。
- FP8 decode: gfx1030はID82／ID68、gfx1201は現行hipBLASLt rank7をcontrolに固定し、tokenごとのactivation quantizerをmatmulから分離する。rank変更、direct wave、pipeline、resident FP16 cacheは、古いmicro値だけで採用しない。4-copy交互測定は過去に再現性を高めるため使った任意の設計例であり、Phase87の1 warmup＋3 measuredを置き換えない。
- 一時展開・再符号化: NVFP4→int8／FP16、NVFP4→FP8は、stage込みのwallと意味保存を同時に満たす必要がある。ID83の追加飽和をW4A4の正しい代替とみなさない。
- 再検討時に確認すると有用な項目は、source／binary identity、target、shape／alignment、量子化を含む計時範囲、独立oracle、finite、repeat、cleanupである。これは履歴に基づくnonblockingな証拠設計の提案であり、Phase87に新しい開始gateや追加の測定回数を導入しない。単体速度だけをPhase87の採用根拠へ読み替えない。

参照: [Phase87計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md) · [Phase76〜78ロードマップ](../../../../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md) · [Phase82削除履歴](../1-10/phase82-optimization-cleanup-default-adoption.md) · [Phase82候補台帳](../1-10/phase82-retired-matmul-candidates.md)
