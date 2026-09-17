# Phase85 follow-up: 実MTPのMXFP M=1高速化

> 状態: 実験完了（ローカル変更・未commit/push）
> 起点: `e25bcc077e301f3157b9d7e984e7663f9f3a45c2`
> 2026-09-13ユーザー指示: 「M=1の高速化を試してみて」

Phase85のsmall-M改善は実MTP draftのM=1へ届かなかった。本作業では実際のcompanionの
6種類のK/N（8行列）を優先し、MXFP8/MXFP6のM=1をcanonical V620 gfx1030、R9700 gfx1201で比較する。
Phase86の全精度最適化や本体NVFP4 verify、samplingの変更へは広げない。

## 比較条件と採否

- [実形状manifest](../../../../../../ci/matrix/phase85-mxfp-m1-mtp-v1.json)を候補実装前に固定。
  fusion 10240/5120、down 17408/5120、gate/up 5120/17408、q 5120/12288、k/v 5120/1024、o 6144/5120。
- 現行r10 binaryのproduction source hashを起点と照合し、現在のGPUで3 warmup＋32 measuredの基準を取得する。
  独立FP32 oracle、非finite、出力digest反復、HIP dispatch、cleanupの既存契約を使う。
- 候補はまずFP32加算順と量子化recipeを維持し、同形式の前後digest一致を確認する。
  適用scopeの非整列N/K32と境界は既存Phase85 manifestから必要な行を追加する。
- Phase84の多列M1候補、Phase85のwave共有／MX6全consumer 2-loadの棄却理由を確認し、
  同じ候補を根拠なく再実行しない。新しいschedule・loadの仮説を記録して比較する。
- 演算子に効果がある候補を通常MTPで比較する。8192/128、MTP幅2、MXFP8 E4 KV、固定sampling、
  同一model/sidecar/fixture、1 warmup＋3 measuredでBF16・MXFP8・MXFP6のdraft wallとdecode tok/sを分ける。
  前後token列・採用数・cleanupも確認し、kernel単体の倍率を全体の改善へ読み替えない。
- 速度下限やBF16超えを必達条件にせず、退行する候補はscopeを限定または撤去する。
  候補追加・再測定は観測した問題に絞り、関係ない全matrix・言語suiteを繰り返さない。

## 最初の候補

Columns2の共有bodyを試す。1出力あたりの8 wave／laneごとのK stride256とFP32加算・reduction順を
維持し、2出力列でactivation読出し・復号を共有する。Phase84で棄却した加算順の異なる多列候補とは区別する。
MXFP6のtarget別reader、MXFP8 gfx1201のwave scale共有は維持する。ID99/100、grid=ceil(N/2)、
workgroup256の別identityで、M=1・K>=2048かつ32倍数・N>=1024を初期比較scopeとする。
比較用の旧M1指定はprepare-timeの`SLLM_MX_WA_M1_FORCE_BASELINE=1`。モデル名による選択は追加しない。

## 候補Bへのscope修正

候補Aは両targetの実形状12件と境界26件で前後digestを維持した。gfx1030 MXFP6 K6144/N5120だけは
32反復の中央値が旧約243 usに対して約678 usとなり、対照を挟む再測定でも再現した。
候補内の後半反復では約184 usに下がったが、この部分だけを採用値へ使わない。
同じ候補の追加探索を止め、この演算形状は旧ID20へ戻すBへ作業を絞る。他の11形状・MXFP8・gfx1201の
GPU bodyと選択は維持する。開始済みのA MTP測定は履歴・全体への影響の判断に使い、Bで影響するV620 MXFP6を再比較する。

## 最終scope C

Bの近傍K6112/6176、N5119/5121でもgfx1030 MXFP6の退行を観測したため、単一点の除外をやめ、
`5120<K<10240`を旧ID20へ戻す保守的な範囲へ修正する。gfx1201 MXFP6の`N=1024 && K<5120`も
A境界測定で約0.75倍だったため旧ID20を維持する。GPU bodyはA/Bから変えない。
実MTPの6形状でCが選ぶIDはBと同じであり、BのV620 MXFP6測定と、未変更のAの他形式/targetを最終比較へ使う。
全matmul device fatbinとsource/selectorの対応を記録し、影響しない全モデル測定は繰り返さない。

## 実行と記録

保存済みMTP traceからM=1の回数・寄与と実形状を対応づける。native Lunaがkernel候補とprofile集計を分担し、
mainが基準・候補の実機実行と統合を担当する。これは実験の作業順であり、新しい独立reviewや承認gateではない。
GPU使用前にQwen停止・UUIDを確認し、R9700既存serviceを測定時だけ停止、元のunit/binary/configとhealthへ復帰する。
raw測定・build・profileは`.local-artifacts/phase85-m1/`へ置く。外部コードの新規取込みがあれば同じ作業中に来歴を記録する。

[メイン計画](../../../../main-plan.md) /
[Phase85履歴](../../../../../history/2026/09/11-20/phase85-mxfp8-mxfp6-common-kernels.md)

実験結果: 実MTPのdecodeは同形式の基準比で約1.9〜2.9%改善し、生成token列と採用数を維持した。BF16を上回らないため既定は維持する。最終scope C、失敗を含む経緯と実行identityは[履歴](../../../../../history/2026/09/11-20/phase85-mxfp-m1-mtp-followup.md)へ記録した。
