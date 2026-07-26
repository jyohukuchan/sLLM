# 射影 / format の却下候補台帳（2026-07-26）

この台帳は、新しい lightweight promotion policy に合わせて再確認した
`SQ8_0` projection と format 候補だけを扱う。attention 候補は依頼BHの範囲なので
意図的に含めない。

## 実行可能な候補

| 候補 | 現状 | 今回の扱い | 根拠 |
| --- | --- | --- | --- |
| `SQ8_0` private handwritten gfx1201 WMMA projection (`ullm_sq8_handwritten_gfx1201_m1_wmma_kernel`) | Qwen3-14B-FP8 の private full-model selector で実行可能。現行 revision の layer 3 `down_projected` には 2 / 5,120、最大 abs `6.1035156e-5` の差が残る。 | R9700 isolated full-model decode の速度を先に測る。速ければ固定10 promptで CK control と生成文を比較する。 | [projection contract](../sq8_0-projection-contract/attempt-3/diagnostic/report.json)、[contract journal](../../../../journal/2026/07/26/sq8-r9700-handwritten-projection-contract.md) |
| `SQ8_1` W8A8 (K=32 I8 + FP16 scale) | CPU fake-quant full-model replay は実行可能だが、gfx1201/R9700 の W8A8 dispatch は v0.1 で禁止され、served selector / manifest / worker はない。 | CPU の実生成文は source-reference と並置して再評価する。R9700 full-model decode tok/s は測定不能であり、historical V620 kernel time を代替値にしない。 | [architecture rule](../../../../docs/plans/sq8_1-format-design-input-v0.1.md)、[quality gate journal](../../../../journal/2026/07/26/sq8_1-w8a8-full-model-quality-gate.md) |

## 別 revision / 将来 format

| 項目 | 判定 | 今回測定しない理由 |
| --- | --- | --- |
| 旧 `SQ8_0` handwritten attempt-2 | superseded | 現在の contract-traced revision とは別実装で、旧 full-model は広範に発散した。current candidate の速度を問う本タスクで旧 revision を再測定すると、候補を混同する。 |
| `SQ9_0` | deferred future option | packer、reader、validator、CPU oracle、generic dequant、runtime selector、manifest handling が現行 scope 外であり、gfx1201 で選択できない。V620 M=1 の historical `SQ8_0` 比 +6.069% は V620 使用禁止と R9700 full-model 非該当の双方から再利用しない。 |

## 横断時に見つかったが R9700 candidate ではないもの

| 項目 | 今回測定しない理由 |
| --- | --- |
| gfx1030-only `SQ8_0` direct/batch fallback specialization | V620/card0 専用の同一 process benchmark であり、gfx1201 legacy body は source-hash / static audit で不変と確認されている。R9700 execution は意図的に行われていないため、R9700 candidate ではない。V620 は本タスクで使用禁止。 |
| `SQ8_0` OCP E4M3FN → FNUZ prepack / CDNA3 A′ | gfx942 FP8 MFMA / MI300X 向けの portability route。CPU format oracle や gfx942 physical validation は R9700/gfx1201 served path の selector・worker・full-model candidate を作らない。R9700 以外を使えない本タスクの対象外。 |

前者は `journal/2026/07/26/sq8_0-gfx1030-fair-comparison-and-m-sweep.md`、後者は
`journal/2026/07/26/sq8-ocp-fnuz-prepack-oracle.md` と
`journal/2026/07/26/mi300x-rental-results-and-sq8-aprime-validation.md` を確認した。

`SQ9_0` の現行 status は [SQ9 format plan](../../../../docs/plans/sq9-format-design-input-v0.1.md)
および [deferred journal](../../../../journal/2026/07/26/sq9_0-deferred-v100-rdna1-scope.md) に記録されている。

## 登録済み served candidates の監査

`/etc/ullm/served-models/candidates/` は 5 manifest で、static validator は 4 pass / 1 fail
だった。全件 `AQ4_0` であり、この射影 / format 再評価の候補ではないため、内容を変更・
計測・昇格していない。個別 SHA-256 と validator 結果は
[registered-candidate-validation.md](registered-candidate-validation.md) に保存した。
