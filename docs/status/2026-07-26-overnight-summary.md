# 2026-07-26 overnight status summary

対象は 2026-07-26 JST の `git log` で確認した 49 commits と、それらが参照する
結果・journal・plan である。これは現状の事実を一箇所に集めた記録であり、activation、
campaign、authorization、または GPU 実行の許可ではない。

## 要約

| track | 現在の結論 | 本番への反映状態 |
| --- | --- | --- |
| `AQ4_0` runtime hardening | Phase 1--4 と activation control/bundle v1 publication の実装・seal が揃い、read-only preflight は `ready: true` / `blockers: []`。 | activation は未実行。Phase 6 の人間承認が必要。 |
| `SQ8_0` R9700 | decode/prefill の hot path は特定済みだが、数値 gate を通った置換候補はない。 | normal Flash2、legacy direct paged decode、CK projection のまま。 |
| `SQ8_0` CDNA3 | format/ISA と A′/B の offline 準備は完了。 | MI300X/gfx942 実機での correctness・occupancy・timing は未実施。 |
| `SQ8_1` | 実装・検証・V620最適化は完了し、W8A16 を default とする。 | W8A8 は full-model quality No-Go で explicit-only のまま。 |
| `SQ9_0` | V100 または exact RDNA1 向けの将来 option として保留。 | reader、kernel、selector、artifact、manifest は未実装・非選択。 |

## 1. `AQ4_0` runtime hardening promotion

Phase 1--3 の protected closure は完了している。worker は live worker と `cmp` を含めて
bit-identical（SHA-256 `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b`）で、
independent inode を持つ。live manifest の guard flag は順序を含めて 30 件、P3-only key は 0 件である。
Phase 1--3 の closure inventory、product/tokenizer/source-clone verification は
`benchmarks/results/2026-07-26/aq4-runtime-hardening-phase123/` にある。

その後、AQ4-to-AQ4 locked activation control route と bundle v1 の owner-bound immutable
publication/validator が実装・seal された。Phase 4 は live manifest から candidate profile を
機械的に導出し、fresh evidence、receipt、frozen candidate、immutable rollback copy、reviewed
operations、credential seal set、activation plan を作成した。plan SHA-256 は
`72140ff475b29e28f4ab6685459a344939bc54fcd12aa4f0b7c44cd7a8753194` である。

plan-bound default preflight は全 10 check が PASS で、
`ready: true`、`blockers: []`、`production_activation_performed: false` を記録した。
candidate の 30 guard flags は live と同じ順序で、`/etc/ullm/served-models/active.json` は
SHA-256 `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a` のまま未変更である。
主証跡は `benchmarks/results/2026-07-26/aq4-runtime-hardening-phase4/` と
`docs/plans/aq4-runtime-hardening-promotion-plan-v0.1.md` にある。

この hardening activation の admission に残るのは、表示された exact activation-plan SHA、
candidate/rollback SHA、service window を人間が確認した上での明示承認と required literal
confirmation だけである。これは execute 権限が自動的に発生した、という意味ではない。execute、
rollback、recovery は plan SHA と literal confirmation を要求し、activation はまだ一度も実行して
いない。activation 後の Phase 7 fresh campaign/browser evidence と complete bundle v1 の実行も
未実施の follow-on である。

## 2. `SQ8_0` R9700 optimization

R9700/gfx1201 の selected-region evidence では、M=1 decode の paged decode attention が
51.05%、CK projection が 40.13% を占め、M=128 prefill の Flash2 attention が 75.63% を占めた。
decode の既定 body はすでに wave-shuffle であり、forced shared-LDS fallback より unprofiled
decode throughput が 4.724077% 高い。したがって fallback を除くことによる現行経路の追加改善は 0% である。

次表の「数値不合格」は、それぞれで適用された事前固定の numerical gate を指す。Flash2 の
QK-only/QK+max は standalone adversarial pre-gate までであり、full-model gate を通過したという
意味ではない。full QK+max+sum は full-model vector gate まで実施した。いずれも置換候補としては
不合格である。

| candidate | 数値 gate の不合格理由 | 現在の default 状態 |
| --- | --- | --- |
| Flash2 QK-only staged wave32 | adversarial standalone で max abs `2.622604e-5`。固定された `2e-5` を超過。 | normal Flash2 body を維持。 |
| Flash2 QK+max staged wave32 | adversarial standalone で max abs `2.622604e-5`。固定された `2e-5` を超過。 | normal Flash2 body を維持。 |
| Flash2 QK+max+sum full staged wave32 | standalone max abs `2.646446e-5` に加え、full-model final hidden は max abs `0.7760314941` / relative L2 `0.0145683599`、logits は `0.2401080132` / `0.0084836396`。gate (`2e-5`, `1e-5`, cosine `0.999999`) を失敗。 | normal Flash2 body を維持。 |
| paged decode source tile 128 | token は exact、finite でも full-model hidden/logit 24 pair 中 4 pass / 20 fail。worst max abs `2.317678451538086`、relative L2 `0.08369554694605848`、cosine `0.9965189313620728`。 | legacy direct paged decode が default。selector は調査専用 opt-in。 |
| paged decode source tile 256 | token は exact、finite でも 12 pass / 12 fail。worst max abs `1.9435234069824219`、relative L2 `0.03318822738718883`、cosine `0.9996737107487421`。 | legacy direct paged decode が default。selector は調査専用 opt-in。 |

tile 128/256 の multi-tile split は、tile ごとの online-softmax `(max, denominator, weighted value)` を
rescale/merge するため、source 全体を一つの online state で処理する direct path と F32 association が
異なる。one-tile は exact でも multi-tile で差が発生し、40 layer の 160 activation quantization を
経ると full-model divergence になる。containment は multi-tile を direct body へ fallback させるもので、
その fallback を含む再 gate は 24/24、max abs 0 で通過した。しかしこれは safe な containment であり、
multi-tile performance implementation の pass や default promotion ではない。

private gfx1201 handwritten WMMA projection は上の 5 attention candidates とは別の projection track だが、
同じく numerical No-Go である。component の 4 actual M=1 shape は通った一方、full-model feedback
gate は全 step で hidden/logit が CK と不一致だったため、candidate timing は意図的に実行していない。

**性能改善は本番経路に一切入っていない。** 現在の選択は normal Flash2、legacy direct paged decode、
CK projection のままであり、`ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE` は不在なら direct を選ぶ
調査用 selector である。active manifest、production dispatch、campaign、authorization は変更されていない。

## 3. 数値契約から得た知見

1. split-merge online softmax は逐次量子化下で single-pass online softmax と等価ではない。
   multi-tile の standalone 差が `1e-8`--`1e-7` 程度でも、feedback decode と反復 activation
   quantization を通ると大きな hidden/logit 差になる。token の一致だけでは gate として不十分である。
2. token equality が品質の代替にならないことは、tile 128/256 の exact greedy token と vector gate
   failure、ならびに handwritten projection の `[66, 198, 197, 197]` 一致と full hidden/logit
   mismatch の両方で確認された。
3. handwritten projection の first observed divergence は layer 3 `down_projected` で、
   2 / 5,120 要素、first index 1,954、max abs `6.1035156e-5` だった。actual-artifact replay は
   同じ差を再現した。K128 block 1 (`K=128--255`) で既に差があり、block 内では K16 prefix 1--7 は
   exact、8 番目の K16（offset 112--127）を足した時点で初めて差が出る。final K16 operand/fragment
   mapping、WMMA reduction/issue association、または両方のどれが根因かは未確認である。

## 4. `SQ8_0` CDNA3/gfx942 port

CPU-only OCP E4M3FN-to-FNUZ prepack format gate は通過した。256 code のうち finite 254 code を検証し、
OCP negative zero `0x80` は FNUZ `0x00` に正規化、`0x7f`/`0xff` は reject、operand ごとの scale は x2、
両 operand の積は x4 とする。canonical artifact scan では 280 tensor、13,212,057,600 payload byte、
806,400 scale、`0x80` は 207,515、`0x7f`/`0xff` は 0、scale gate violation は 0 だった。

A′ は existing CK gfx942 XDL ABI を opaque に再利用する isolated prototype であり、derived FNUZ buffer
だけを受ける。offline gfx942 code object の Default `16x128x128` main-K-loop では
`v_mfma_f32_16x16x32_fp8_fp8` を 24 本確認した。B は OCP を BF16 に dequant して hipBLAS F32 GEMM を
行う correctness control である。これは format/ISA の証跡であり、fragment correctness、device
occupancy、residency、timing、end-to-end performance の証明ではない。

MI300X checklist は、one-wave fragment/lane diagnostic の後に 5 実形状で A′/B/CPU を比較し、
actual module/function の occupancy を取得する順序を固定している。入口 preflight と第一段だけの
最短 go/no-go は約 10--20 分である。**実機検証は未実施**であり、A′、B、handwritten MFMA A のいずれも
production selection ではない。

## 5. `SQ8_1`

`SQ8_1` の K=32 signed symmetric I8 + separated FP16 scale plane の設計、packer/reader、runtime ABI、
CPU/GPU differential、V620/gfx1030 kernel optimization は完了している。`SQ8_1` namespace の default は
W8A16 であり、これは active manifest への deployment を意味しない。

V620/card0 の Qwen3-14B `5120 x 5120` M=1 fair rotating co-dispatch comparison では、equally optimized
`SQ8_0` に対する paired-ratio median が W8A16 で **2.633x**、W8A8 で 2.522x だった。先行した 2.692x は
unoptimized `SQ8_0` fallback との比較を含むため、format-only conclusion には使わない。W8A16 が W8A8 より
速い current exact direct API の M={1,8,32,128} 結果も維持されている。

W8A8 は CPU FP32 reference を使う 248-projection full-model gate で **No-Go** になった。20 record / 4,243
scored position で、aggregate logits relative L2 `0.023506802` など一部の gate は通るが、logits max abs
`7.889154 > 1.0`、final-hidden max abs `13.696337 > 1.0`、top-1 agreement
`4,189/4,243 = 98.727316% < 99.0%`、Wilson lower `98.343243% < 98.5%`、disallowed mismatch 16 件である。
したがって W8A8 は explicit-only のままで、runtime/artifact/release selection には採用しない。

## 6. `SQ9_0` の保留

`SQ9_0` は current target の runtime/artifact format ではない。`gfx1030`、`gfx1100`、`gfx1201`、`gfx942`、
`gfx950` は current INT8-capable scope であり、`SQ9_0` reader、quantizer、generic dequant、selector、
manifest support はすべて未実装・非選択である。

保留を解除できるのは、V100 または exact RDNA1 GFX を必要とする product/serving requirement があり、
その exact target について useful FP8 route と practical INT8 matrix/dot route の双方が不足することを
target-specific toolchain/ISA/hardware evidence で確認し、`AQ4_0`/`SQ8_0`/`SQ8_1` との matched comparison、
review 済み implementation plan、別途のユーザー承認が揃った場合だけである。V100 に `dp4a` があること、
RDNA1 が `gfx1010` と `gfx1011`/`gfx1012` で同じではないことから、世代名だけで「INT8 dot がない」とは
結論しない。

## 7. ISA 訂正と format selection rule

RDNA4/gfx1201 に INT8 capability がない、という先行推論は誤りだった。gfx1100/gfx1201 は
VOP2 cumulative `v_dot4c_i32_i8` ではなく VOP3P `v_dot4_i32_iu8` を受理し、gfx1201 はさらに
`v_wmma_i32_16x16x16_iu8` と FP8/BF8 WMMA を持つ。`__builtin_amdgcn_sdot4` が bare target で
diagnostic/scalarization になっても命令不存在の証拠ではない。target HSACO/assembly で期待する
`v_dot4_*`、`v_wmma_*`、または `v_mfma_*` を確認し、RDNA3/RDNA4 では `sudot4` の
`v_dot4_i32_iu8` signed-control codegen を見る必要がある。

以下は選択ポリシーであり、未実装の reader/kernel/artifact/deployment を承認するものではない。
常に manifest の exact format を優先し、黙った再量子化や置換はしない。

| architecture | 現在の推奨最適化 format | 条件・非選択扱い |
| --- | --- | --- |
| gfx1030 | `SQ8_1` W8A16 | portable/legacy INT8 dot を使う。W8A8 は quality No-Go のため default にしない。`SQ9_0` は非選択。 |
| gfx1100 | `SQ8_1`（target gate 後） | VOP3P dot と INT8 WMMA がある。FP8 WMMA はこの ROCm 7.2.1 probe では確認されていない。 |
| gfx1201 | `SQ8_0` | source-preserving FP8 WMMA route がある。`SQ8_1` は valid alternative でも `SQ8_0` を自動置換しない。 |
| gfx942 | `SQ8_0`（OCP-to-FNUZ/native-MFMA gate 後） | INT8/FP8 MFMA を使えるが、A′ physical validation は未実施。 |
| gfx950 | `SQ8_0`（format/native-MFMA gate 後） | wider INT8/FP8 selectable MFMA があるが、payload compatibility は個別に要証明。 |

## 8. 未解決事項

1. PMC: `SQ_INSTS_VALU` と GL2C `32B/64B/128B` request は purpose-built load+FMA probe でも 0 で、
   `SQ_WAVES` だけが nonzero だった。selected Flash2 でも `FETCH_SIZE=0` と `VALUInsts=0` である。
   counter definition の typo ではないが、driver/firmware/permission/ROCm counter programming のどれかは
   未分離で、physical HBM efficiency と memory-bound/compute-bound の最終判定は未確認である。
2. `THROTTLED`: R9700 windows の raw state は残っているが、per-reason field が unsupported、raw field が
   atomic pair ではなく、1 秒 sample では瞬間 peak を見逃し得る。sampled temperature/power だけから
   持続的な物理 throttle 原因は確定できない。将来の timing は atomic metrics capture、cool-down/all-clear、
   reason-bit 時の discard/repeat を要する。
3. formatting: `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 cargo fmt --all -- --check` は現在も失敗する。
   distinct tracked file は 12 件（`sq8_ck_serving.rs`、7 AQ4 bin、`lib.rs`、`sq8_layer_runtime.rs`、
   `sq8_model_head_runtime.rs`、`sq8_0_paged_decode_split_bench.rs`）である。範囲外の既存状態なので、
   review 済み formatting-only change なしに `cargo fmt --all` は実行しない。

## 9. 人間の判断が必要な論点

| 論点 | 判断に必要な事実 | 要求する決定 |
| --- | --- | --- |
| `AQ4_0` hardening activation | current plan は `72140ff475b29e28f4ab6685459a344939bc54fcd12aa4f0b7c44cd7a8753194`、read-only preflight は `ready: true` / `blockers: []`、active manifest は未変更。 | exact plan/candidate/rollback SHA と service window を確認し、literal confirmation を伴う activation を承認するか。 |
| 数値 gate の基準 | 現行 projection gate は CK との multi-step complete/bitwise equality を要求する。`SQ8_0` 自体が損失のある量子化であるため、これは CK を正解の定義にし、CK と異なる association を持つ最適化を原理的に止める。 | **提案であり決定ではない**: optimized path が FP32 reference に対して CK と同等以上に近いことを、事前固定の full-model metrics で要求する基準へ変えるかを検討する。これは既存 gate を自動的に緩めず、別の review/validation plan を要する。 |
| W8A8 rescue | `outlier_bypass_ge4` は 14.331775% の activation block を FP32 bypass し、W8A8-to-W8A16 aggregate-logit-L2 gap を 100% 除去した。しかし max-abs/top-1 gate はなお不合格。 | compact FP16 side plane/mask または calibration を別 task で追うか、それとも W8A16 default を維持して停止するか。 |
| P3 performance deployment | 推奨 source cut は `c4c9a9b344fc10e9a77ab0ded3293469d21b2f72`。47 P3 commits を含み、HEAD の余分な SQ8 v2 shared-runtime surface を含まない。P3 candidate は 36 guards、新 worker/manifest/receipt、identity-bound fidelity/integration evidence、**29 以上**の既定 R9700 measurement/service window を必要とする。 | hardening と分離した P3 candidate/deployment programme を開始するか。開始しても active-manifest replacement は別の人間承認まで行わない。 |
| CDNA3 physical smoke | A′/B と MI300X checklist は準備済みで、最短 fragment go/no-go は 10--20 分。 | isolated MI300X/gfx942 smoke を別途承認するか。実機結果が出るまで performance/deployment conclusion は出さない。 |

