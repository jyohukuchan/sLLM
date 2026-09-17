# `FORCE_BASELINE` 診断用参照経路の約束範囲

この文書は、[FORCE_BASELINE 計画](../plans/archive/2026/09/11-20/force-baseline-reference-oracle.md)の
Stage 1 で定める T2（GPU 上の帰属用参照経路）の範囲を記録する。T2 は本番のロールバック経路でも、
独立した真値でもない。T2 の結果は、差し替えた subsystem に差の帰属先があるかを調べるためにだけ使う。

## Stage 0 の状態

2026-09-17時点で、11 flag × 2 target の初期 Stage 0 long 棚卸しは両GPUで完了している。
集約 ledger の `missing_expected_longflags` と `running_expected_longflags` はともに空である。
初期投入では gfx1030 の FP8 decode long/short と、両GPUの NVFP4 W4A4 long が失敗したが、
FP8 decode Graph と NVFP4 baseline launcher の修復は完了した。以下の表には現在の selector と
launcher のソースから導いた到達性と期待値を記し、初期 Stage 0 の実測結果と修復後の状態を区別する。
初期棚卸しの完了は、修復後の実モデル確認や修復後 evidence を遡って示すものではない。
最終的な約束範囲は、exact target、UUID、実際の dispatch、kernel trace、非 zero dispatch、fallback なし、
cleanup 0 を備えた Stage 0 の証拠で確定する。

gfx1201はdefaultと11フラグのlong/short計24 runを終了した。longで失敗したのはNVFP4 W4A4だけで、
同じフラグのshortは参照kernelを840回実dispatchして完走した。無効果の5フラグ
（MX prefill、FP8 outer prefill、FP8 outer decode、MX M1、未実装のFP8 outer）は、longのkernel symbol件数と
hidden/logit hashがdefaultと一致した。GDN、BF16 matmul、attention、NVFP4 quantizer、FP8 quantizerは
kernel切替を確認した。NVFP4/FP8 quantizerだけの変更ではhidden/logit hashはdefaultと一致した。
R9700 serviceはunit/run.sh/binary hash不変、health/ready 200、performance level一致で復帰した。
raw記録は`.local-artifacts/force-baseline-oracle/stage0-gfx1201/`を参照する。

gfx1030では初期 Stage 0 の FP8 decode フラグが long/short とも target decode row 0 の stateless Graph
admission で拒否された。修正は下記の既存 M=1 pack 条件内への variant 追加に限定した。
実GPUの Graph／数値 probe は完了している。修復後は V620 (`model-r4`) の default と NVFP4 W4A4 long が
exit 0 で完了し、FP8 decode long も exit 0 で完了した。R9700 (`model-r1`) の default／NVFP4 W4A4 long
も exit 0 で完了している。`verify-completion-evidence.py` は両モデルの HIP-only、nonfinite 0、fallback false、
cleanup 0、参照 symbol を確認して PASS した。この結果は必要な修復後 full-model run の完了を示す。
T1 数値照合は下記の raw functional evidence と区別し、全形状の数値同等性へ一般化しない。

上記の初期棚卸しは [`force-baseline-oracle-v1.json`](../../ci/matrix/force-baseline-oracle-v1.json) と
`.local-artifacts/force-baseline-oracle/stage0-summary.json` に固定し、修復後の model-r4 再実行は
`.local-artifacts/force-baseline-oracle/model-r4-gfx1030/` に分離している。

棚卸しの固定入力は、既存の
`.local-artifacts/mtp-bench/wmma-check/prefix-one.json`（`code-rust-bugfix-zh`、8,284 prompt token）と
chunk 2,048 の Qwen3.8-27B NVFP4 グラフである。対象 target は exact `gfx1030`（V620）と exact
`gfx1201`（R9700）だけとする。prefill 到達性はこの固定 prefix で確認し、decode だけの確認には計画で指定する
短い prefill fixture を使う。prefill chunk を変更して baseline を通す回避は、この約束範囲に含めない。

## T1 と T2 の境界

T1 は既存の host 独立 FP32／NumPy oracle であり、正しい出力の判定を担当する。T1 の実装や契約はこの文書で
変更しない。T2 は GPU の `FORCE_BASELINE` 経路であり、特殊化経路と codec、scale 算出、量子化処理などを
共有する場合があるため、T1 の代用にはしない。

1 回の帰属実験では、対象の subsystem に対応するフラグを 1 つだけ変更し、それ以外の演算、chunk、graph
構造、KV encoding、MTP 設定を固定する。出力が T1 と一致したことと、対象 subsystem の変更で差が消えたことは
別々に記録する。

## Qwen3.8 固定グラフ上の inventory

検証対象の Qwen3.8 グラフは 64 層で、4 層ごとに FullAttention（16 層）、残り 48 層が GDN／LinearAttention
である。先頭 56 層の MLP projection は NVFP4 W4A4、残り 8 層の MLP projection、GDNのqkv/z/out、
FullAttention projection および `lm_head` は FP8 outer である。GDNの`in_proj_a`／`in_proj_b`はBF16のまま
残る。MTP は BF16 companion を使い、fusion、FullAttention の
Q/K/V/O、MLP gate/up/down の 8 BF16 matmul を持つ。MXFP8 KV は KV state の encoding であり、MX weight
selector を有効にしない。

| フラグ | Qwen3.8 固定グラフでの対象 | prefill / decode のソース上の効果 | T2 identity（source期待／Stage 0 状態） |
| --- | --- | --- | --- |
| `SLLM_MX_WA_PREFILL_FORCE_BASELINE` | MXFP8/MXFP6 weight のみ | BF16 MTP と NVFP4 本体には無効果。MX companion を明示した場合だけ M>1 prefill に効き、M=1 には効かない | MXFP8/MXFP6 の elementwise baseline。固定Qwenグラフでは **未到達（BF16 MTP）**。期待値は `sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_v1` または `sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_v1` |
| `SLLM_GDN_FORCE_BASELINE` | 48 GDN 層の recurrent provider | 両 target、prefill/decode で column、decode-pair、row32-LDS を抑止。conv は共有 | logical `linear_attention.gdn.v1`、recurrent `sllm_linear_attention_recurrent_gated_norm_v1`、2 dispatch。**Stage 0 実測（両target、hidden/logit hash差；kernel identityは同一）**。FP8 GDN qkv/z の projection-pack は別 subsystem として残り得る |
| `SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE` | FP8 outer weight の M>1 matmul | gfx1030 prefill だけ elementwise FP8 emulation へ切替。gfx1201 は `Fp8Native` の先行分岐で無効果。M=1 decode には効かない | `matmul.fp8.outer.byte_decode.v1` / `sllm_matmul_fp8_outer_emulation_v1`。**Stage 0 実測（gfx1030切替、gfx1201未到達）** |
| `SLLM_NVFP4_W4A4_FORCE_BASELINE` | NVFP4 W4A4 MLP gate/up/down | 両 target、prefill/decode。W4A4 compute と activation quantizer の両方を変更する | `matmul.nvfp4.w4a4.block16.packed.v1` / `sllm_matmul_nvfp4_w4a4_block16_packed_v1`。quantizer は `sllm_matmul_bf16_to_nvfp4_block16_v1`。**初期Stage 0は両target long未到達、修復後は両target full-model run exit 0・参照 symbol確認済み** |
| `SLLM_MATMUL_FORCE_BASELINE` | 本体GDNのBF16 a/b投影とBF16 MTP の 8 matmul | 両 target、prefill/decode。FP8／NVFP4のselectorには効かないが、本体のBF16補助投影には効く。gfx1201実測でもtarget hiddenが変化した | `matmul.bf16_fp32.v1` / `sllm_matmul_bf16_fp32_v1`。**Stage 0 実測（両target、hidden/logit hash差；kernel identityは同一）** |
| `SLLM_CAUSAL_ATTENTION_FORCE_BASELINE` | 16 FullAttention 層と MTP FullAttention | prefill 候補を抑止する。gfx1201実測ではdecodeのstaged 2段（各4,352回）が1段wave split（4,352回）へ替わった。一部の専用wave経路は残るため、全kernelをgenericへ戻す指定ではない | 完全な baseline identity は Stage 0 trace で確定する。完全に generic なら MXFP8 KV は `sllm_causal_attention_online_softmax_gqa_packed_kv_v3`。**Stage 0 実測（両target、hidden/logit hash差；identityは専用経路を含む）** |
| `SLLM_NVFP4_FORCE_BASELINE` | NVFP4 activation quantizer | Qwen W4A4 では両 target、prefill/decode の quantizer だけを scalar baseline に変更。W4A4 compute kernel は変えない | `sllm_matmul_bf16_to_nvfp4_block16_v1`。W4A16 provider では別途 compute baseline を選ぶが固定 Qwen graph にはない。**Stage 0 実測（両target、quantizer切替；hashはdefault同一）** |
| `SLLM_FP8_OUTER_DECODE_FORCE_BASELINE` | FP8 outer weight の M=1 matmul | gfx1030 decode だけ emulation へ切替。gfx1201 は Native の先行分岐で無効果。M>1 prefill には効かない | `matmul.fp8.outer.byte_decode.v1` / `sllm_matmul_fp8_outer_emulation_v1`。**初期Stage 0はgfx1030未到達（Graph admission fail）、gfx1201無効果。修復後V620 longはexit 0、emulation symbol確認済み** |
| `SLLM_MX_WA_M1_FORCE_BASELINE` | MXFP8/MXFP6 weight の M=1 decode | BF16 MTP と NVFP4 本体には無効果。MX companion を明示した場合、M1 Col2 を標準 decode へ戻す | MXFP8 は ID18 `sllm_matmul_mxfp8_w8a8_e4m3_block32_decode_v1`、MXFP6 は ID20 `sllm_matmul_mxfp6_w6a6_e3m2_block32_decode_v1`。固定Qwenグラフでは **未到達（BF16 MTP）** |
| `SLLM_FP8_QUANT_FORCE_BASELINE` | Qwen の FP8 outer projection と FP8 output head | 両 target、prefill/decode。compute variant は変えず、動的 activation quantizer だけを v1 に変更。FP8 GDN projection-pack も同じ quantizer を通る | `sllm_matmul_bf16_to_fp8_outer_v1`。**Stage 0 実測（両target、quantizer切替；hashはdefault同一）** |
| `SLLM_FP8_OUTER_FORCE_BASELINE` | production selector に対象なし | 無効果。現行 native selector／launcher はこの環境変数を読まず、旧 Gemma evidence の allowlist にだけ残る | **未到達／無効果（production identityなし）** |

この表の根拠は、matmul selector／identity／grid の
[`matmul_kernel_internal.hpp`](../../native/hip/src/matmul_kernel_internal.hpp)、matmul launcher の
[`matmul_kernel.hip.cpp`](../../native/hip/src/matmul_kernel.hip.cpp)、GDN selector の
[`linear_attention_runtime.inc`](../../native/hip/src/linear_attention_runtime.inc)、および causal attention
selector の [`causal_attention_runtime.inc`](../../native/hip/src/causal_attention_runtime.inc) である。

## 修復対象のT2約束範囲

計画に従い、Stage 0以降の修復・追加の範囲検査・T1再照合は、Stage 0で失敗したフラグに限る。
他フラグの表は今回の固定グラフでの観測範囲であり、別モデルの既存admissionを変更するものではない。

Stage 0で観測済みの失敗を修復したNVFP4 W4A4 T2のadmission範囲を次の通り定める。修復実装、両GPUの
operator／数値 probe、および必要な修復後 full-model run は完了している。以下の範囲全体を総当たりで検証済みとは扱わない。

- format: NVFP4 W4A4、block size 16、既存の E4M3FN block scale と tensor/input scale 配置。
- target: exact `gfx1030` または exact `gfx1201`。
- shape: `1 <= M <= 4096`、`1 <= K <= 17408`、`1 <= N <= 65536`。
  K上限は既存の公開matmul契約を維持する（草案の32,768は既存ABI上限を超えるため訂正）。
- 既存 descriptor/layout 契約を同時に満たすこと。metadata の byte range／scale offset／packed weight
  layout が正しく、入力・weight・output が非空であることを要求する。単独matmulは非整列Kのtailも扱う。
  projection-packには追加のK16整列など、既存のpack契約が適用される。
- mode: M=1 decode と、M>1 prefill。Qwen3.8 の M=2,048、K=5,120／17,408、N=5,120／17,408 と、
  projection-pack の対象 M を含む。

この範囲は、baseline の 1 output element あたり 256-thread block という launcher 契約に対し、1回の
launchで扱う output element 数を `16,777,215` 以下に制限する admission 上限である。初期 launcher は
Qwen の `M=2048,N=17408`（35,651,584 elements）を単一 launch に渡して `invalid configuration argument`
を発生させた。修復後 launcher は graph chunk を変更せず、`nvfp4_w4a4_baseline_launch_count` に従って
output range を複数 launch へ分割する。修復後 V620／R9700 の必要な full-model run と、両GPUの operator 境界
probe はこの分割経路を確認済みである。

上記の数値範囲はadmissionの契約であり、全 M×K×N の直積を検証済みとするものではない。T1 照合の証拠集合は、
実際に Stage 4 で測った形状だけとし、少なくとも M=1 と M=2,048、prefill 境界の両側、非整列 N、K=16 の
両側、および descriptor の上限近傍を含める。Stage 0 の trace または Stage 2 の実装結果がこの範囲を支持しない
場合は、範囲を狭めてこの文書と集約 ledger を同時に更新する。

`SLLM_NVFP4_W4A4_FORCE_BASELINE` は、W4A4 compute だけを WMMA の対照に置くフラグではない。現在の
`launch_nvfp4_quantize` もこのフラグを見て wave8 quantizer から scalar quantizer へ戻すため、T2 の subsystem
単位は「NVFP4 W4A4 の activation quantizer と matmul」である。この依存を固定したまま T1 と照合し、quantizer
だけ、compute だけという結論に分割しない。

## 約束外の失敗

修復対象のNVFP4 W4A4参照経路では、約束範囲外の target、shape、scale layout を、HIP の generic `invalid configuration
argument` へ到達させず、prepare 時に「baseline 参照経路の対象外」と明示して拒否する。Stage 3 の host contract
test では、M/K/N の内側・外側、M=1／2、M=4,096／4,097、K=15／16／17／17,408／17,409、N=1／0／65,536／65,537
を含む境界を検査する。GPU 成功、性能、モデル品質を約束範囲外へ一般化しない。

## FP8 decode参照経路のGraph範囲

`SLLM_FP8_OUTER_DECODE_FORCE_BASELINE=1`で選ばれるgfx1030の`Fp8Emulation`（ID6）を、
既存のFP8 GDN pack用stateless Graph許可リストへ追加した。許可するpackは以下をすべて満たすものに限る。

- exact `gfx1030`、M=1、K=5,120、QKV/ZのN=10,240／6,144。
- OCP E4M3FN＋FP32 outer scale、両memberが同じvariant、plan-owned workspaceが5,124 bytes。
- 既存のqueue、plan pin、in-flight/release、buffer lifetime検査に成功すること。

Graphのshape、node順序、chunk、workspace、kernelの算術・選択を変えず、既存の参照variantを受理するだけである。
範囲外のGraphは従来どおり、exact target／M1 stateless contractの対象外と明示して拒否する。
FP8単独matmulの既存契約を拡張したものではなく、任意のFP8 shapeのGraph化は保証しない。
gfx1201ではこのフラグはNative ID5のままで無効果である。

両GPUのM=1実GDN pairで、独立FP32解析式と全BF16出力の一致、異なる入力への更新、
3-node Graph capture、2回replay、fence/finalize、cleanup 0を確認した。
gfx1030ではID6と実kernel symbolを確認した。gfx1201は従来のNative経路の確認であり、
FP8参照kernelをgfx1201へ追加した証拠とは扱わない。

## 参照 source

- [計画: FORCE_BASELINE を T2 oracle に限定](../plans/archive/2026/09/11-20/force-baseline-reference-oracle.md)
- [Qwen3.8 mixed inventory](../../crates/sllm-core/src/quantized_model.rs)
- [Qwen3.8 layer schedule and MTP graph](../../crates/sllm-core/src/qwen_graph.rs)
- [既知の NVFP4 baseline 欠陥](../../ci/matrix/nvfp4-force-baseline-defect-v1.json)
