# Phase82: 既定採用スコープと保留項目

更新: 2026-09-08

この文書は、Phase82終了時点の現行selectorを基準に、未設定時の既定経路、数値分類、rollback、保留条件をまとめる。24件のmatmul候補の削除理由は[候補別履歴](phase82-retired-matmul-candidates.md)に分けて記録する。Phase79の一覧は調査起点であり、削除済み・既定採用済みの最終判定には現行sourceを優先する。

## 判定規則

`FORCE_*` が存在するだけでは既定採用と数えない。selectorは、明示的な `=1` のforceを対象shapeで優先し、`=0` または未知値をfail-closedでbaseline／fallbackへ戻す。既定経路は、対応target・shape・encodingで環境変数が未設定の場合だけ選ぶ。`FORCE_BASELINE=1` は既定より優先するrollbackである。

N0は実数式、演算順、丸めstage、state更新を保った変更、または同じ値を生成する表現・実行制御の変更である。N1は式と入力項を保ったままreduction依存深さを減らすなど、誤差boundが非増加と説明できる変更である。実測token一致だけでN0/N1へ昇格せず、対象外shapeと未測定範囲を残す。

## 未設定時の既定採用

| 経路 | 現行sourceの条件と優先順位 | 分類とbaselineとの差 | 検証・rollback |
| --- | --- | --- | --- |
| NVFP4 activation quantizer wave8 | `native/hip/src/matmul_kernel.hip.cpp:6203-6241`。compile targetが`gfx1030`または`gfx1201`で、`SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8`が未設定ならwave8。値`1`でも選択し、`0`／未知値はbaseline quantizerへ戻る。`SLLM_NVFP4_FORCE_BASELINE=1`または`SLLM_NVFP4_W4A4_FORCE_BASELINE=1`が最優先。 | **N0**。同じBF16→E2M1量子化、scale算出、block16の出力契約を並列なwave8配置で実行する。 | gfx1030／gfx1201の26 fixture、5モード（flag0、wave8、default、invalid、force-baseline）を独立NumPy oracle、canary、native status、`hipFree` cleanup付きでPASS。[Phase82 evidence ledger](phase82-optimization-evidence.json)。個別の全model性能寄与は主張しない。rollbackは同sourceのscalar baseline。 |
| NVFP4 decode ID67 | `native/hip/src/matmul_kernel_internal.hpp:1253-1322`。`M=1`、`1024<=K<=17408`、`K%16=0`、`N>=1024`、exact `gfx1030`／`gfx1201`。先行するscale-LUT／activation-shared branchが選ばれず、wave4 controlが未設定のときに既定wave4を選ぶ。scale-LUT／activation-sharedの明示`0`／未知値はその先行branchを選ばず、wave4 shapeならID67へ到達し得る。 | **N1**。codec、scale、BF16 RNE、入力項を保ち、block内のinteger dot4と固定treeへ移すためreduction順は変わる。Phase79のbound根拠をgfx1201へ同じshape契約で拡張した範囲だけを採用する。 | 旧ID67境界oracleとPhase82 native selector／projection検査を再利用する。`SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4=0`、`SLLM_NVFP4_W4A4_FORCE_BASELINE=1`でbaselineへ戻す。K/N外、他target、異なるencodingは従来経路。 |
| NVFP4 decode ID84 scale LUT | `native/hip/src/matmul_kernel_internal.hpp:1268-1322`。exact `M=1,(K,N)=(5120,17408)`または`(17408,5120)`、gfx1030／gfx1201で、wave4・activation-shared・LUT controlが未設定ならID84。 | **N0（この2 tuple内）**。E4M3 scaleの同じ値をLUTで供給し、decode、scale、FP32 accumulation、BF16 storeの入力項と順序を保つ。 | 両targetのprojection-pack oracleはbitwise、repeat、`max_bf16_ulp=0`、fallback 0、cleanup 0。[Phase82 evidence ledger](phase82-optimization-evidence.json)。`SLLM_NVFP4_W4A4_DECODE_FORCE_LDS_F32_LUT=0`等で既定を抑制すると、他のwave4／activation-shared controlが未設定ならgfx1030はexact tupleでID73 activation-shared、gfx1201はID67 wave4へ戻る。別のcontrolが設定されている場合は、そのsource上の明示優先順位に従い、generic shapeをID84既定へ広げない。 |
| FP8 outer decode ID82 | `native/hip/src/matmul_kernel_internal.hpp:2005-2063`。exact `gfx1030`、`M=1`、`(K,N)`が`(5120,17408)`,`(5120,10240)`,`(5120,6144)`,`(6144,5120)`のいずれかで、direct decode controlsが未設定ならLDS-LUT。 | **N0（4 tuple内）**。E4M3FN→FP16 LUTは既存decodeと同じ有限値変換で、FP32 accumulationとBF16 RNEを変更しない。 | ID68とのtuple oracle、境界、repeat、token/fixed-logits検査をこのscopeへ限定する。[Phase82 evidence ledger](phase82-optimization-evidence.json)。`SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT=1`は対応generic shapeの明示選択。`=0`／未知値はID82の既定条件を抑制し、他のdirect controlが未設定なら同じshapeで採用済みID68へ戻る。`SLLM_FP8_OUTER_DECODE_FORCE_BASELINE=1`だけがscalar baselineへ戻す。K/N外、FNUZ、他targetは既存経路。 |
| gfx1030 GDN row32 LDS | `native/hip/src/linear_attention_runtime.inc:34-50`。exact gfx1030、`token_count=1,qk_heads=16,value_heads=48,head_dim=128`。`SLLM_LINEAR_ATTENTION_GFX1030_ROW32_LDS`未設定または`1`で有効。 | **N0**。recurrent式、FP32 state、state index、BF16出力を保ち、LDS layoutとlaunch routeだけを変更する。 | Phase82 production probeはoracle、repeat、state/output diff 0、finite、cleanupをPASS。`=0`／未知値または`SLLM_GDN_FORCE_BASELINE=1`でgeneric providerへ戻る。他target／shapeは既定化しない。詳細は[Phase82 cleanup history](phase82-optimization-cleanup-default-adoption.md)。 |
| Qwen3.8 projection sharing / deferred / Graph / FP16 chain | `crates/sllm-core/src/qwen_execution.rs:944-1098,5330-5347,5387-5402,7068-7150`。HIPのgfx1030／gfx1201、verified Qwen3.8 artifactとFP8 sidecar、text/non-MTP、adapter/control空、対象Graphだけ。NVFP4／FP8 GDN projection packは個別rollbackを持つ。deferredとKV append/attention chainはtarget別env未設定を既定ON、Graphはdeferred選択時のstateless M1 decodeだけ。chainは全KV stateがFP16のときだけ。 | **N0**。projection共有は同じmemberとactivationを一回のprepared executionへまとめ、deferred／Graph／chainはcompletion、ownership、submission順を変える制御で、matmul／attention式・KV値・samplingを変更しない。 | `0`／未知値はfail-closed rollback。Graphは最初の対象decodeをeager warmup/captureし、terminal samplerとstateful attention/KVをcapture外に置く。core host 64件、prepared execution 15件、projection-pack両target oracleで検査済みだが、全model／全KV形式の性能・品質を主張しない。[Phase82 evidence ledger](phase82-optimization-evidence.json)。 |

NV67だけはreduction順の変更を含むためN1として扱い、ID84、ID82、GDN row32、Qwenの実行制御はN0の限定範囲として扱う。採用範囲外へ一般化するには別のsource・oracle・性能記録が必要である。

ID73はforce専用経路だけではない。ID84のLUT controlを`0`／未知値で抑制し、他のwave4／activation-shared controlが未設定のgfx1030 exact tupleでは、現行selectorのfallbackとしてID73が条件付き既定になる。このfallbackはID67との比較で、decode式・入力項・出力丸めを保つactivation-sharing／launch変更としてN0に分類する。

## 既定化しない保留

| 経路 | 現在の扱い | 既定採用に不足する条件 |
| --- | --- | --- |
| GQA6 decode P64 | `SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P64=1`だけの明示opt-in。query=1、Q/KV heads=24/4、head=256、FP16 KV、gfx1030 KV>=8192またはgfx1201 KV>=4096。P128が明示された場合はP128優先。 | P64 partition/mergeはN2の別providerで、P64 LDS表現だけがN0である。target別の採用根拠・品質分類・ユーザー判断が不足しているため既定ONにしない。 |
| GQA6 decode P128 | `SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128=1`、exact gfx1030、FP16 KV、KV>=8192。 | FP64 oracleで最大1 BF16 ULP差が観測されたN2。worst-case bound、品質、長文実modelの採用判断が不足している。 |
| GQA6 rocBLAS F32 prefill | `SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1030_ROCBLAS_F32=1`またはgfx1201相当。 | QK/PV providerとreduction順を変えるN2。限定probeはPASSでも誤差非増加の根拠と既定採用判断を満たさない。workspace、品質、性能をtarget別に確認し、明示判断が必要。 |
| NVFP4 ID62 prefill DP4A | `SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A=1`の候補。 | corrected ID59と同じ実数式でも、ID62はblock16 dotを逐次加算するため長いKの誤差上限が増える既知のN2。K5120ではID62が319回のblock加算対ID59の約28段、K17408では1087対約76段で、9-shape host stress sampleは最大絶対差0.75を観測した。したがって自動N1既定化せずHOLDとする。これはGPU PASSでも品質不合格判定でもなく、追加の全model gateを要求しない。[Phase82 cleanup history](phase82-optimization-cleanup-default-adoption.md)／[evidence ledger](phase82-optimization-evidence.json)。 |
| NVFP4 ID64 gfx1201 WMMA | `SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA=1`を含む既存WMMA baseline/rollback。 | ID69/72/81/83の比較基準として維持するが、一般shapeの既定採用へ拡張するtarget別数値・resource・model evidenceが不足。 |
| NVFP4 ID72 gfx1201 FP16 staging | `SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING=1`。 | N2。FP16 ingressは正確でもfull-K FP32 reduction順が変わり、既存ID64との差とtoken差が記録されている。ユーザー判断、誤差根拠、no-regression性能が揃うまで保留。 |

これらは削除候補ではない。共有baseline、rollback、ABI、履歴probeを維持し、保留理由を採用理由へ読み替えない。

R9700（gfx1201）のGemma同一入力比較では、NV67の新しい既定経路はbaselineとgreedy／host-fixed／device-fixedの生成token hashが一致しなかった。一方、両方ともHIP実行、finite、fallbackなし、cleanup 0をPASSしており、これはN1のtarget拡張を全model出力同値と扱わないための観測であり、token hashを自動hard gateや品質不合格へ読み替えない。V620では同じ3 modeのhashが一致し、device-fixed decodeは14.6764から15.6017 tok/s、R9700は11.3384から18.1205 tok/sだった。R9700 runnerは生成列のhashだけを記録し、最初の相違token indexは取得していないため、その値を主張しない。最終r7 source identityがcurrent source anchorであることは[Phase82 evidence ledger](phase82-optimization-evidence.json)の`final_build_reuse_mapping`に記録され、ここでの範囲判定もcurrent sourceを優先する。

## 残る明示force／rollback群

Phase82で既定化しなかった候補固有controlは、既存のbaseline比較と再検討に使うため残す。主な群は、NVFP4のID61/62/64/73とdecodeのwave4・activation-shared・LUT controls、ID82外のFP8 decode/prefill controls、GQA4/GQA6のP32/P64/P128・blocksoftmax・qtile・rocBLAS、gfx942 GDN wave64 column state、MXFPのtarget/shape force controlsである。削除済み24 matmul候補の専用enum・symbol・launch・probeはこの一覧に含めない。

## 証拠の境界

Phase82のGPU証拠は記載したtarget、shape、encoding、fixtureに限る。`max_bf16_ulp=0`やtoken一致はそのscopeの観測であり、全model・全shape・全KV形式の品質証明ではない。未設定時に既定へ到達すること、明示0／未知値でbaselineまたはfallbackへ戻ること、fallback・非finite・cleanupを引き続き検査する。詳細な数値分類は[数値・出力影響変更台帳](../../../../compatibility/numerical-output-changes.md)、候補削除の試行と理由は[Phase82 cleanup history](phase82-optimization-cleanup-default-adoption.md)、固定fixtureとsource/build identityは[Phase82 evidence ledger](phase82-optimization-evidence.json)を参照する。
