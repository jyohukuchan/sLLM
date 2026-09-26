# Phase 87 WU-3P: NVFP4 MTP prefix prefill改善

> 状態: 完了（2026-09-25）
> 対応履歴: [WU-3P記録](../../../../../history/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)

## 目的と固定範囲

2026-09-25のユーザー指示。段階3で既定化したQwen3.8 NVFP4 MTP companionのprefix準備を速くする。
[backlog P12](../../../../backlog.md)のM=1024付近のfc（K10240/N5120）、q（K5120/N12288）、
k/v（K5120/N1024）を主対象とする。target本体の量子化recipe、decode経路、MTPのp/q受理則は変えない。
対象GPUはexact V620 `gfx1030`とR9700 `gfx1201`。片方だけ改善する場合はexact targetで範囲を分ける。

現行8192/128のprefill中央値はBF16 companionがV620 37.95秒／R9700 14.88秒、
NVFP4が44.54／20.84秒。MTP prefix準備はBF16 1.56／0.11秒、NVFP4 7.76／6.04秒で、
NVFP4の汎用`row8_tiled256` 32起動が7.47／5.95秒を占めた（[段階3記録](../../../../../history/2026/09/21-30/phase87-stage3.md#nvfp4のprefix準備が遅い原因)）。

## 着手前に固定する確認事項

1. 分解した現行kernelを時間・数値の対照にし、既存の64×64 DP4A／WMMA等を上の実shapeで測る。
   新variantが必要ならshared device helperで演算を共有し、最初に代表variantのVGPR、LDS、occupancyを記録する。
2. BF16出力の現行経路とのbitwise比較に加え、独立した量子化値の数値oracle、非整列Mと
   1024の両側境界、両GPUのfinite/repeat/cleanupを確認する。丸めstageを変える候補は別分類で記録する。
3. 改善した実shapeだけproduction selectorへ接続し、runtime切替を新設しない。BF16とNVFP4の
   target本体・MTP decode経路は対照として残す。
4. 同一条件のoperator比較と通常8192入力／128出力のprefill・TTFT・MTP prefix wallを記録する。
   既存の[共通採否ルール](../../../../main-plan.md#変更の採否ルール2026-09-24ユーザー決定)を今回の
   prefill最適化には適用する。2026-09-24のNVFP4既定化だけに与えられた速度例外を転用しない。
   HIP-only、fallbackなし、数値・状態の正しさ、request/session cleanupも確認する。

## 終了条件

両GPUまたは改善を確認したexact targetで、上の数値条件を満たし、通常prefill／TTFTと
prefix準備の実測を現行NVFP4既定と比較して改善した実装を採用する。採用しない候補も測定と理由を履歴へ残す。
Phase 87全体の他段階はこの作業では閉じない。

## 完了結果

`gfx1030`では既存ID62 DP4A 64×64、`gfx1201`では既存ID64 WMMA 128×64を、
M32〜1024かつ上の3組のK/Nだけへ既定選択した。新しいdevice演算やruntime切替はない。
MTP prefix準備の中央値はV620 **7.527→0.287秒**、R9700 **6.018→0.201秒**。
通常8192/128のprefillはV620 **43.515→36.198秒**（16.82%短縮）、
R9700 **20.746→14.916秒**（28.10%短縮）で、TTFTも同方向に短縮した。
同条件BF16 companionのprefillはV620 37.628秒、R9700 14.832秒だった。

既存kernelとの全BF16出力差は両GPUの62ケース・各約1.51億値で最大1 ULP、
独立long-double oracle 744点は両GPUで最大0 ULP。通常実行はHIP-only、fallbackなし、
finite、固定K20 p/q、cleanup zeroでPASSし、MTPなしのtoken SHAも両GPUで従来と一致した。
full-modelの出力token列はMTP draftの丸め差で分岐したため、数値台帳へN2候補として記録する。
詳細なkernel候補、不採用と数値・binary identityは[履歴](../../../../../history/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)を参照。

履歴: [WU-3Pの実装と測定](../../../../../history/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)。
