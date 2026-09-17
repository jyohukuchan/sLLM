# R9700のMXFP MTP改善実験

2026-09-13のユーザー指示「同様の統計情報は取得できないかもしれないが、R9700についても改善を試みて。」に対応する。
[診断](phase85-mxfp-m1-bottleneck-diagnosis.md)で確認した改善経路を、R9700の通常MTPで試したローカル実験。

## 変更と条件

現在の未公開Columns2 Cを基準に、分離source snapshotでgfx1201だけを変更した。
MXFP8 ID99はnative E4M3変換を維持し、E8M0復号をbranchless化、scaleのlane0 load＋shuffleを
全laneの同一address直接loadへ置換した。MXFP6 ID100は既存2-load packed readerを維持して復号をbranchless化した。
FP32積和／reduction順、selector C、本体NVFP4、KV、sampling、gfx1030経路は維持した。

- exact gfx1201、UUID `GPU-a8e9ddefa2d60f55`。
- Qwen3.8-27B-NVFP4、本体共通のMXFP8 E4 KV、BF16／MXFP8／MXFP6 MTP companion。
- coding8192、8192入力／128出力、chunk2048、state8320、MTP幅2、固定sampling seed123。
- 1 warmup＋3 measured、stock auto clock。BF16と現行MX形式を今回新規測定した。
- 実行順はBF16基準→MXFP8基準→演算子／境界対照→MXFP8候補→MXFP6候補→MXFP6基準。
- 実MTP6形状×2形式の12演算子比較、K2016/2048/2080・N1024/1025の限定8境界比較。
  演算子は3 warmup＋32 measured、独立FP32 oracleと全出力digestを確認する。
- 候補の各形式profileは0+1で、到達確認に限定する。profile時間は性能値に含めない。
- 前の診断で無効だった主要PMCは再取得しない。GPU event、kernel trace、数値結果、生成ISAを根拠にする。

## 生成ISAの照合

gfx1201の共通113 kernelを比較し、raw text差分17件のうちlabel／PC相対位置を正規化した後の
実質的な命令列変更はID99/100だけだった。ID99は1952→1788 bytes、静的SALU178→160、
VGPR23→20、SGPR24→22。native `v_cvt_f32_fp8_e32` は3→3で維持し、`ds_bpermute_b32` は23→20で
scale broadcastの3経路が除かれた。ID100は2236→2140 bytes、静的SALU192→169、VGPR20で不変。
両候補ともscratch spillは0。他kernelのnamed resource metadataは維持した。
これは静的命令列の比較であり、PMCでの命令発行率・演算器飽和・実DRAM帯域の測定ではない。

## 結果

[追跡用集約結果](../../../../../ci/matrix/phase85-r9700-mtp-improvement-results-v1.json)に入力identity、
median/MAD、同形式の全反復token列・MTP count一致、draft時間、binary hashを保存した。
通常MTPの5構成はすべてPASS。BF16 decode基準は34.8683 tok/s（MAD 0.0058）。

| MTP形式 | decode基準→候補 (tok/s) | 改善率 | MAD基準→候補 (tok/s) | BF16比不足率 基準→候補 |
| --- | ---: | ---: | ---: | ---: |
| MXFP8 | 32.9703 → 33.3046 | +1.01% | 0.1176 → 0.0308 | 5.44% → 4.48% |
| MXFP6 | 30.6944 → 30.8218 | +0.41% | 0.0239 → 0.0276 | 11.97% → 11.61% |

| MTP形式 | decode中draft wall基準→候補 | draft短縮 | decode全体wall短縮 |
| --- | ---: | ---: | ---: |
| MXFP8 | 650.70 → 619.32 ms | 4.82% | 1.00% |
| MXFP6 | 645.98 → 618.17 ms | 4.31% | 0.41% |

MXFP8のdecode全体は3851.96→3813.29 msで約38.66 ms短縮し、draftは約31.38 ms短縮した。
MXFP6のdecode全体短縮は約17.10 msで、draft短縮約27.81 msの全量が同じまま全体差になるわけではない。
各値は独立した中央値で、区間の中央値を足してE2E中央値へ分解しない。
M1以外の本体検証、共有FP8 head、その他のdraft処理が残るため、行列単体の倍率は全体の倍率にならない。

8192入力／128出力のE2E中央値はMXFP8 19187.84→19041.71 ms（-0.76%）、
MXFP6 19497.19→19430.13 ms（-0.34%）だった。prefillも試行間で変動しており、このE2E差のすべてを
M1改善へ帰属させない。1 fixture・1 warmup＋3 measuredの小幅な観測改善で、別入力長や広い条件の保証ではない。
MXFP8基準のdecode 3反復は33.0878／31.7305／32.9703 tok/s、候補は33.2068／33.3354／33.3046 tok/sであり、
基準側に遅い反復が1件あった。全3反復を保持し、事前に定めた中央値で比較した。

同形式の基準と候補は、全3反復で128出力tokenの実配列、proposal blocks、提案・採用・棄却数を含む
MTP count項目が一致した。BF16相対の量子化品質同等性を認定する検証ではない。

| MTP形式 | proposal blocks | 提案token | 採用token | 採用率 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 50 | 100 | 78 | 78.00% |
| MXFP8（前後一致） | 51 | 102 | 77 | 75.49% |
| MXFP6（前後一致） | 56 | 112 | 71 | 63.39% |

両候補ともBF16未達。MXFP6にはBF16より6回多いproposal blocksと低い採用率があり、
今回の復号だけの変更ではこの差を解消しない。V620で観測した大きな復号制御の改善をR9700へ一般化しない。

## 演算子・到達確認

実MTP12行と限定境界8行の前後比較はすべて独立FP32 oracle・全出力digest一致、非zero HIP dispatch、
fallbackなし、cleanup 0を確認した。演算子eventはactivation quantizer＋matmulを含む。
MXFP8 gate/upは約608.2→528.7 us、downは642.7→556.8 us。MXFP6 gate/upは474.0→444.1 us、
downは476.6→429.2 usだった。これらは通常auto clockでの今回のevent中央値で、過去のstock peak固定probeとは別測定。

候補profileも両形式PASS。MXFP8 ID99は960 dispatch、MXFP6 ID100は1024 dispatchで、
各6形状すべてが直前のMX quantizerと対応した。legacy decodeは0で、期待どおり改善経路に到達した。
profile実行と通常実行のtoken列・採用数も一致し、HIP-only、fallbackなし、cleanup 0を確認した。

実験controllerは全49 jobを正常終了した（5性能行、40演算子行、2 profile行、build待ち・oracle集約）。
R9700のperformance levelは元のauto、既存serviceは元のunit／binary／run.sh hash一致、
healthz／readyz HTTP 200へ復帰した。検証完了時に本番production inputが基準Cと一致することも再確認した。
本番kernel採用、commit／pushは行っていない。

## 証拠と運用

raw artifactは `.local-artifacts/phase85-r9700-mtp/` に保存する。`prep/experiment.json` が正確なcommand／環境、
`source-comparison.json` がCからのsource差分、`build-gfx1201/identity.json` がbuildとbinary identity、
`experiment/` が各stdoutとtelemetry、`profiles/` がkernel trace、`analysis/` が集計である。
本番checkoutへのkernel採用、commit／pushは行わない。外部コードの新規importはない。
既存R9700 serviceはactive接続なしを確認して一時停止し、終了時に元のunit／binary／run.shとhealthへ復帰した。

[計画](../../../../plans/archive/2026/09/11-20/phase85-r9700-mtp-improvement.md) /
[メイン計画](../../../../plans/main-plan.md)
