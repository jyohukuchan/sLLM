# V620でのbranchless MXFP MTP検証

2026-09-13のユーザー指示「V620で診断用改善経路を使ってMTPが改善するか検証して」に対応する。
[直前の診断](phase85-mxfp-m1-bottleneck-diagnosis.md)で確認した復号の改善を、通常のMTP実行で比較した。
本番採用と公開は今回の範囲に含めない。

## 変更と測定条件

現在の未公開Columns2 Cを基準に、source snapshotでMXFP8 ID99とMXFP6 ID100のgfx1030復号だけを
branchless helperへ置換した。各kernelのactivation・weight2列の計6 load箇所が対象。
FP32積和／reduction順、量子化recipe、selector C、MXFP6 Oの旧ID20、NVFP4本体、KVとsamplingは維持した。
production input照合では既存変更ファイルはmatmul_kernel.hip.cppだけで、helperを追加した。
本番checkoutのsourceは変更していない。

- canonical V620 exact gfx1030、UUID `GPU-76a08c022586fed6`。
- Qwen3.8-27B-NVFP4本体、既存BF16／MXFP8／MXFP6 companion、MXFP8 E4 KV。
- coding8192 fixture、8192入力／128出力、chunk2048、state capacity8320、MTP幅2。
- GPU固定sampling seed123、各構成1 warmup＋3 measured、stock auto clock。
- 今回新規測定したCのBF16／MXFP8／MXFP6を基準とし、MX形式のcandidateを比較する。
  実行順はBF16基準→MXFP8基準→演算子照合→MXFP8候補→MXFP6候補→MXFP6基準。
- 演算子は実MTP6形状×2形式×前後、3 warmup＋32 measured。
- candidateの実MTP profileは形式ごとに0 warmup＋1 measuredで、到達確認に限定する。
  profile実行の時間は性能代表値に使わない。

## 結果

[追跡用集約結果](../../../../../ci/matrix/phase85-v620-branchless-mtp-results-v1.json)にmedian/MAD、
入力identity、同形式の全repeat token列・MTP count一致、draft時間とbinary hashを保存した。
速度比較の5構成はすべてPASS。BF16のdecode中央値は25.4097 tok/s、MADは0.0043 tok/sだった。

| MTP形式 | decode基準→候補 (tok/s) | 改善率 | MAD基準→候補 (tok/s) | BF16比不足率 基準→候補 |
| --- | ---: | ---: | ---: | ---: |
| MXFP8 | 21.4158 → 22.6685 | +5.85% | 0.0209 → 0.0366 | 15.72% → 10.79% |
| MXFP6 | 23.1281 → 23.4621 | +1.44% | 0.0061 → 0.0004 | 8.98% → 7.66% |

| MTP形式 | decode中draft wall基準→候補 | draft wall短縮 | decode全体wall短縮 |
| --- | ---: | ---: | ---: |
| MXFP8 | 1074.33 → 763.05 ms | 28.97% | 5.53% |
| MXFP6 | 819.83 → 747.41 ms | 8.83% | 1.42% |

MXFP8のdecode全体は5930.19→5602.50 msで約327.70 ms短縮し、draftの約311.28 ms短縮が
大部分を説明する。単体の大行列が約2倍になっても、draftは基準decode時間の約18%で、
本体verificationや共有FP8 headなどが残るためMTP全体は同倍率にならない。
MXFP6は単体改善幅が小さく、O投影の旧ID20も維持している。

長いprefillはほぼ不変で、8192入力／128出力のE2E中央値短縮はMXFP8約0.47%、MXFP6約0.37%だった。
これは今回の観測値であり、1 warmup＋3 measured・1 fixtureの小さなE2E差を広い条件での改善保証とはしない。

同形式の基準と候補は、全3反復で128出力tokenの実配列、採用・棄却数、proposal block数を含む
MTP count項目が一致した。量子化形式間の品質同等性は判定していない。

| MTP形式 | proposal blocks | 提案token | 採用token | 採用率 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 53 | 106 | 74 | 69.81% |
| MXFP8（前後一致） | 58 | 115 | 70 | 60.87% |
| MXFP6（前後一致） | 56 | 111 | 71 | 63.96% |

改善後もBF16には届かない。残差にはdraftの処理時間に加え、形式ごとの採用差とproposal block数の差がある。
今回の復号改善で採用率が変わったという結果ではない。

## 実装照合と追加検証

gfx1030の113 kernelをCPU側で照合した。ID99は2084→1656 bytes、静的SALU157→99、
ID100は1732→1688 bytes、静的SALU119→106、VGPR21→17だった。BF16 ID3、量子化器、旧ID18/20の
命令とresource metadataは同じ。後続12 kernelのraw差分はcode位置の0x200-byte変化に対応する
PC相対literal offsetで、正規化後の命令列は一致した。これを命令数からのFLOPs推定には使わない。

実MTPの12比較行は独立FP32 oracle、全出力digest・32反復digest一致、非zero HIP dispatch、cleanup 0を確認した。
演算子のeventにはactivation quantizerとmatmulが含まれ、kernel単体値とは区別する。
MXFP8 gate/upは約720→368 us、downは約725→366 us。MXFP6 gate/upは約448→382 us、
旧ID20を維持したOは約221→221 usだった。

候補profileは両形式ともPASS。MXFP8 ID99は1040 dispatch、MXFP6 ID100は889 dispatch、
旧ID20は111 dispatchで、MXFP6 Oだけを旧経路へ戻す選択を確認した。各形式で6形状すべてに到達し、
直前のMX quantizerと対応した。profile実行も通常測定とtoken列・採用数が一致した。

境界ではK2016/2048/2080、N1023/1025の限定8行を候補で照合し、同条件の基準と全出力digestが一致した。
既存manifestに含まれた基準側の境界26行もすべて数値oracleをPASSした。
全測定はHIP-only、fallbackなし、cleanup 0。各controller終了時のperformance levelは元のautoへ復帰した。
Qwenローカルserviceは開始時から停止中で、R9700 serviceは操作していない。

検証はこれで完了。本番へのbranchless採用、追加のMTP形状・入力長・別fixture・別GPUの測定、
commit／pushは行っていない。

## 証拠

raw artifactは `.local-artifacts/phase85-branchless-mtp/` に保存した。
`prep/`は正確なcommand／環境、`build-gfx1030/identity.json`はbuild入力とbinary hash、
`source-comparison.json`はCとの差分、`operators/`は演算子、`baseline-first/`・`candidate/`・
`baseline-last/`は実MTP、`profiles/`は到達確認、`analysis/`は集計を含む。
モデルslice、binary、raw profileはGitへ追加しない。外部コードの新規importはない。

[計画](../../../../plans/archive/2026/09/11-20/phase85-v620-branchless-mtp.md) /
[メイン計画](../../../../plans/main-plan.md)
