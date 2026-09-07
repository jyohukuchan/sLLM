# Phase79 selector inventory

> 2026-09-07 source inventory. This records the selector contract and the
> available evidence; it is not a claim that every candidate is generally
> adopted.

## 判定語と観測範囲

`supported` は target、shape、入力 encoding の契約を満たすこと、`enabled`
は現在の flag または既定条件が候補を要求すること、`adopted` は selector
がその範囲を既定経路にしたことを表す。matmul の prepare は
`variant/supported/enabled/adopted/reason` を決定として保存する。したがって
execute 中の環境再評価や token ごとのログは必要ない。baseline に戻る場合も
「候補未採用」「flag が off」「要求された候補が対象 shape 非対応」「入力形式が
baseline 必須」を区別できる。

主な実装ソースは
[matmul selector](../../../../../native/hip/src/matmul_kernel_internal.hpp)、
[causal-attention selector](../../../../../native/hip/src/causal_attention_runtime.inc)、
[Qwen projection pack](../../../../../native/hip/src/qwen38_projection_pack_runtime.inc)、
[graph-span gate](../../../../../native/hip/src/graph_span_runtime.inc) である。
projection/execution の共通契約の詳細は
[runtime architecture](../../../../architecture/runtime.md) に置く。

## NVFP4 / FP8 decode

decode の共通 NVFP4 形状は `M=1, K>0, K%16=0, K<=17408, N>0`、FP8
gfx1030 の half2/dword8 形状は `M=1, K%64=0, 64<=K<=17408, N>0` である。
候補の support はこの意味上の契約で判定し、モデル名や layer 番号を条件にしない。

| ID / variant | supported と encoding | 現在の選択・override | adopted / GPU evidence | 未採用・未確認の理由 |
| --- | --- | --- | --- | --- |
| 58 NVFP4 packed decode | gfx1030/gfx1201 の NVFP4 baseline shape | 候補範囲外、明示 baseline、または候補の `0` / 非対応時 | baseline fallback | 比較の基準経路。candidate の不採用理由は prepare decision に残る。 |
| 65 NVFP4 columns128 | gfx1030/gfx1201、上記 NV shape | `SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_COLUMNS=1` の opt-in | adopted ではない。過去探索では ID58 より遅く、V620/R9700 とも既定化しなかった | 一般 shape の速度・数値分類が未完了。 |
| 67 NVFP4 wave4col32 | gfx1030/gfx1201、上記 NV shape | `...DECODE_FORCE_DP4A_WAVE4=1` は opt-in。gfx1030 では `K>=1024,N>=1024` の範囲を既定化。flag が存在すれば値 `0` も既定を抑制 | gfx1030 の adopted 範囲。`phase79-operator-evidence.json` のproduction ABI oracle/repeatで境界・中間shapeを確認し、model evidenceは候補比較12/12 top1を記録する | operator evidenceは有限encoded入力のBF16出力と性能範囲を証明する。model evidenceの `quality_verdict` は `not_evaluated` であり、全モデル・全shapeの品質や性能を主張しない。gfx1201はsupported/forceのみ。 |
| 73 NVFP4 activation-shared | gfx1030、`(K,N)=(5120,17408)` または `(17408,5120)` のみ | `SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_ACTIVATION_SHARED=1` | opt-in shape-specific candidate。Phase79 の一般 adoption evidence なし | exact tuple の資源・数値分類が未確認。 |
| 84 NVFP4 scale-LUT | gfx1030 は ID73 と同じ exact tuple、gfx1201 は上記 generic NV shape。gfx1201 は exact tuple だけ activation-shared code object、それ以外は generic wave4 scale-LUT body | `SLLM_NVFP4_W4A4_DECODE_FORCE_LDS_F32_LUT=1` | opt-in。gfx1201 の generic shape support と exact body の区別を維持 | LUT の shape-specific 数値・性能採否が未確認。 |
| 66 FP8 half2 wave4col32 | gfx1030、FP8 outer、`M=1,K%64=0,64<=K<=17408,N>0` | `SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_HALF2=1` の opt-in | adopted ではない。V620 exploratory 6.0617 tok/s | dword8 との数値差、一般 shape の品質分類が未完了。 |
| 68 FP8 dword8 wave4col32 | ID66 と同じ FP8 shape | `SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8=1`。gfx1030 の `K>=128,N>=64` は既定化。flag `0` は既定を抑制 | gfx1030 の adopted 範囲。`phase79-operator-evidence.json` のproduction ABI oracle/repeatで境界・Gemma consumer shapeを確認し、V620 exploratory値は6.8076 tok/s | 固定 logitsの初回比較は8/12 top1、max KLD 5.834113。これは候補差を記録したもので、model evidenceの `quality_verdict` は `not_evaluated`。operator N1 scope以外の品質採否は主張しない。 |
| 75/76 FP8 activation-shared | gfx1030 の exact Qwen tuple 群のみ（wave4 / wave8） | `SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_ACTIVATION_SHARED=1` の opt-in | adopted ではない | tuple ごとの register/LDS と数値・性能の確認が未完了。 |
| 82 FP8 LDS LUT | ID68 の generic FP8 shape。Qwen の exact tuple は専用 code object、その他は generic fallback | `SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT=1` の opt-in | adopted ではない | LUT 経路の一般 shape の数値・性能採否が未確認。 |

`SLLM_*FORCE_BASELINE=1` は候補より優先される。force 値が `1` でない
明示値は、候補を要求した場合の `enabled=false` / baseline reason として扱われる。

## NVFP4 / FP8 prefill

| 既定または ID | shape / encoding | 現在の扱い | 理由・証拠の境界 |
| --- | --- | --- | --- |
| 59 NVFP4 row8 tiled256 | `M>1,K%16=0,N>0` の NVFP4 prefill。known target | NVFP4 の既定 prefill。`SLLM_NVFP4_W4A4_PREFILL_FORCE_ROW8=1` は同経路、`...FORCE_BASELINE=1` は packed baseline | Phase79 history の ID11/59 は同じ K 分割・加算順へ修正後、Gemma 3/17/65 の 12 logits が一致した。これは prefill 数値整合の証拠であり、全モデル性能証明ではない。 |
| 71 FP8 half2 64x64 | gfx1030、FP8 outer、非 FNUZ、`M>1,K>0,N>0` | gfx1030 の既定 prefill。`SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2_64X64=1` で明示選択 | 既存既定。Phase79 の decode 候補測定とは分けて扱う | prefill の候補別最終速度・品質の追加採否は未確認。 |
| 61 / 62 / 80 NVFP4 columns / DP4A / DP4A-K128 | NVFP4 prefill。K は 16 の倍数、ID80 は gfx1030 | `...FORCE_COL8`、`...FORCE_DP4A`、`...FORCE_DP4A_K128` の opt-in | adopted ではない | 明示 force 用の候補。現在の history は decode 探索と ID59 整合性を中心とし、候補別の最終採否を主張しない。 |
| 64 / 69 / 72 / 81 / 83 NVFP4 gfx1201 WMMA/staging | gfx1201 shape helpers。WMMA は `M>1,K%16=0,N>0`、F16 staging は `M>=128` 等の境界、ID83 は exact staging tuple | 各 `...FORCE_GFX1201_*` の opt-in。ID72 の M tail は ID64へ戻す | adopted ではない | WMMA/F16/FP8 staging の numerical classification と GPU evidence が不足。 |
| 63 / 70 / 85 / 86 FP8 gfx1030 candidates | 非 FNUZ FP8 prefill。half2、F16 staging、LDS LUT、F16 tile staging ごとに M/K/N 境界あり | 各 `...FORCE_GFX1030_*` の opt-in | adopted ではない | shape-specific workspace/LUT/staging の数値・資源・性能確認が未完了。FNUZ は ID60 tiled16 など既存形式へ残る。 |

## Causal attention と FP16 専用 GQA

`causal_attention_runtime.inc` の provider は sliding-window または explicit
score scale がある場合は候補を選ばず、`SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`
でも prefill candidate を止める。次は主な Phase78 由来候補の契約である。

| family | supported shape / KV encoding | enabled / adopted の扱い |
| --- | --- | --- |
| GQA6 decode split P32/P64/P128 | decode `query_count=1`、`q_heads=24, kv_heads=4, head_dim=256`、**FP16_V1 のみ**。expected KV は P32 `>=4096`、P64 は gfx1030 `>=8192` / gfx1201 `>=4096`、P128 は gfx1030 `>=8192` | それぞれの `...GQA6_DECODE_SPLIT_*` が `1` の opt-in。P128 > P64 > P32 の優先順。既定 adopted ではない。 |
| GQA4 decode split / P32 | decode `query_count=1`、`q_heads=16, kv_heads=4, head_dim=256`、expected KV `>=4096`、**FP16_V1 のみ** | gfx1030 の split は明示 opt-in、P32 は target 別 flag が未設定または `1` でも有効。existing provider の範囲であり Phase79 matmul ID67/68 の adoption とは別。 |
| FP16 pair / wave split | GQA4 の decode は `q_heads=16,kv_heads=4,head_dim=256`、短い wave split は KV `32..1023`、長い wave split は KV `>=1024`。FP16 pair と GQA split は **FP16_V1 のみ** | 一部 flag 未設定を enabled とする既存 provider。candidate-specific performance adoption は本 inventory では主張しない。 |
| GQA6 blocksoftmax / qtile4 | prefill `query_count>=128,q_heads=24,kv_heads=4,head_dim=256`、GQA6 blocksoftmax は **FP16_V1 のみ**。qtile4 K4/K8/K16/K32 も **FP16_V1 のみ**で、複数 flag は K4 > K8 > K16 > K32 | 各 flag の opt-in。rocBLAS GQA6 が有効なら blocksoftmax は抑制される。 |
| Phase66 tiled prefill / GQA4 control | gfx1201、GQA4 `q_heads=16,kv_heads=4,head_dim=256`、query count は Phase66 が `>=64`、通常制御が `>=128`。FP16_V1 または MXFP8_E4_V1 を受け、context により Q4K4/Q4K8/Q8K8 を選ぶ | `SLLM_CAUSAL_ATTENTION_PHASE66_TILED_PREFILL=1` または既存 GQA4 control。MXFP8 が許されるのはこの typed control 等であり、FP16 専用 GQA kernel の代替証拠ではない。 |
| GQA6 rocBLAS prefill | `query_count>1`、tail が `start_position+query_count==expected_kv_length`、`q_heads=24,kv_heads=4,head_dim=256`、**FP16_V1 のみ**。gfx1201 F16 tail は start>0 かつ追加 flag | gfx1030/gfx1201 の rocBLAS flags の opt-in。既定 adoption ではない。 |
| gfx1030 scaled / long prefill | GQA4、`query_count>=1024,q_heads=16,kv_heads=4,head_dim=256`。scaled は FP16 が未設定/`1`、MXFP8 は明示 `1`、long V2 は **FP16_V1 のみ** | 各 flag の opt-in。境界・数値・性能の最終分類は未確認。 |

ここで「FP16 専用」とした候補は KV encoding の条件自体が
`SLLM_HIP_KV_ENCODING_FP16_V1` である。MXFP8 を受ける別の GQA4/Phase66
control が存在しても、FP16 専用候補の supported 範囲を広げるものではない。

## Projection pack / graph gate の適用一覧

Qwen projection pack の public role 名は維持し、NVFP4 MLP gate/up role だけを
semantic pair として扱う。両 member が BF16 activation の `M=1`、同じ `K`
（16 の倍数）、非ゼロ `N`、NVFP4 W4A4 の scale/offset 契約、同じ activation
shape を満たせば、Gemma の `K=3840,N=15360` のような形も gate に入る。
workspace は packed K/2 と K/16 scale、quantizer grid と両 member の compute
grid を検査する。これは projection の supported gate であり、各 matmul 候補の
数値・速度 adoption を意味しない。

FP8 GDN/QKV/Z role は引き続き `M=1,K=5120,N=10240/6144`、E4M3FN とする。
graph の gfx1201 native FP8 は prepared hipBLASLt plan が両 member に必要で、
gfx1030 software FP8 は両 member が同一 safe variant でなければならない。
NVFP4 graph は両 member の shape、workspace、variant、target を再検査し、ID65/67
は generic decode shape、ID73 は gfx1030 exact tuple、ID84 は gfx1030 exact
tuple または gfx1201 generic wave4 shapeだけを通す。

## GPU evidence と未確認事項

本表の探索性能値は [Phase79 exploratory history](phase79-common-optimization.md)
の同一 binary、V620 gfx1030、1 warmup + 3 measured の値だけを引用した。
そこに記録された decode 値は既定 1.6275、NV wave4 1.8882、NV columns 1.7109、
FP8 half2 6.0617、FP8 dword8 6.8076 tok/s である。固定 logits は wave4
12/12 top1、dword8 8/12 top1 など候補間の差を示すが、差の大小だけで品質 PASS
や全モデル品質を決めていない。採用範囲の最終operator証拠は
[operator evidence](phase79-operator-evidence.json) にあり、gfx1030 production ABIの
有限encoded入力、独立oracle、repeat、境界shape、性能を記録する。5つのoperator runは
`cleanup: "not_reported"` であり、そこに記録されたPASSはoracle・比較・性能の結果だけを
示す。collector/model evidenceのcleanupは別に記録している。

[最終model evidence](phase79-model-evidence.json)では、Gemmaの共有packについて
V620/R9700の固定12位置のlogits一致とcleanup0を確認した。Qwen27BのGraph ON/OFF、
Qwen4BのFP16/MXFP8 KVにおける共通deferred経路も、記録した条件でtoken一致とcleanup0を確認した。
V620 Gemmaの最終同一binary比較では17/65入力の既定decodeが15.5622/15.3531 tok/s、
共通deferred追加が15.3575/15.3419 tok/sであり、deferredの追加利益は確認できずopt-inを維持した。
これらは記録したmodel/shape/encodingの証拠であり、全組合せの品質評価を代替しない。

したがって、ID67/68の既定化は[adoption scope](phase79-adoption-scope.json)に記載した
gfx1030 shape範囲に限る。model evidenceの `quality_verdict` は `not_evaluated` であり、
全モデル・全shapeの品質同等性は主張しない。現行selectorの未採用候補、Qwen/Gemma共通
packの一般性能、causal attention providerの一般採用はこの表の対象外である。

この表はselector契約と証拠範囲を整理するもので、一般的な品質証明や全候補の採用宣言ではない。
