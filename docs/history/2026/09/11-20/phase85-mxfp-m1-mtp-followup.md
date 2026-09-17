# Phase85 follow-up: 実MTPのMXFP M=1高速化

2026-09-13のユーザー指示「M=1の高速化を試してみて」に対応したローカル実験。
起点は `e25bcc077e301f3157b9d7e984e7663f9f3a45c2`。Phase86全体の実施や常駐serviceの新版への更新は含めない。

## 実形状と実装

保存済みMTP traceをquantizerとdecodeの連続dispatchから対応づけ、companionの6種類のM=1形状を確認した。
K/Nは10240/5120、17408/5120、5120/17408、5120/12288、5120/1024、6144/5120。
gate/up、k/vの各2行列を含めて計8行列で、負担が大きいのはgate/up、down、qである。
共有FP8 headは別の処理であり、今回のMXFP body高速化の倍率へ混ぜない。

候補Aは1 workgroupで隣接2出力列を計算し、activationの読出し・復号を共有した。
256 threads／8 wave、laneごとのK順序、FP32積和とreduction順を維持し、奇数Nの最後の列も処理する。
MXFP8 gfx1201のwave scale共有、MXFP6のtarget別reader、量子化recipe、KV、samplingは維持した。
ID99 `sllm_mxfp8_w8a8_m1_col2_v1`、ID100 `sllm_mxfp6_w6a6_m1_col2_v1`を別のlaunch identityとして登録した。
新しい外部コード取込みはなく、既存sLLMのcodecとreductionを再利用した。

## 比較と範囲修正

実形状は3 warmup＋32 measuredで同じbinaryの旧kernel／候補を形状ごとに交互実行した。
両targetの12実形状と26境界ケースは、独立FP32 oracle、反復digest、HIP dispatch、cleanupを成功し、
同形式の前後出力digestも一致した。境界準備で指定したK17440は既存のMAX_K17408を超えて実行前に拒否された。
その失敗は保持し、数値比較を上限内のK17376へ訂正した。成功済みの行を失敗へ読み替えたり、上限を広げたりしていない。

V620 MXFP6 K6144/N5120では候補Aの中央値が約678 us、旧経路が約243 usとなり、対照を挟む再測定でも再現した。
反復後半の候補が約184 usへ下がったが、この部分だけを改善値として使わない。
一度は当該1形状を旧ID20へ戻すBを作ったが、K6112/6176とN5119/5121の近傍でも約0.36倍だった。
R9700 MXFP6ではN1024・K2048/2080が約0.75倍であり、N1025や実MTPのK5120/N1024とは分けた。

最終Cは追加kernel探索を止め、次の保守的な選択へ絞った。これらは未対応入力の追加ではなく、既存GPU kernelの選択である。

| 条件 | 最終選択 |
| --- | --- |
| MXFP8、exact gfx1030/gfx1201、M1、K>=2048かつ32倍数、N>=1024 | ID99 Columns2 |
| MXFP6、同基本条件、gfx1030で5120<K<10240 | 旧ID20 |
| MXFP6、同基本条件、gfx1201でN=1024かつK<5120 | 旧ID20 |
| MXFP6、同基本条件の残り | ID100 Columns2 |
| 上記外 | 既存selector |

既存の公開dimension上限は維持する。比較用の`SLLM_MX_WA_M1_FORCE_BASELINE=1`はprepare時に旧M1を選ぶ。
実MTPの6形状では、V620 MXFP6のo投影だけ旧ID20、残りとR9700はColumns2となる。

## 実MTP比較

Qwen3.8-27B NVFP4 target、既存MXFP8 E4 KV、MTP幅2、8192入力／128出力、固定sampling・seed123を揃えた。
各形式は1 warmup＋3 measuredで、profile runの時間を代表性能値には使わない。
基準も今回取り直し、以前のPhase85の測定値をそのまま分母にしない。
[集約結果](../../../../../ci/matrix/phase85-mxfp-m1-results-v1.json)に中央値/MAD、draft時間、BF16差、入力・artifact identityを記録する。
A/B/Cの同一GPU codeと実形状selectorの対応により、変更の届かない形式/targetの測定を再利用し、実際に実行したbinaryを区別する。
同形式の生成token列と採用数の維持を確認するものであり、BF16比の量子化品質同等性を認定するものではない。

## 最終結果

| GPU | MTP形式 | decode前→後 (tok/s) | 速度改善 | BF16比の不足率 前→後 |
| --- | --- | ---: | ---: | ---: |
| V620 | MXFP6 | 22.434 → 23.094 | 2.94% | 9.95% → 7.63% |
| V620 | MXFP8 | 20.904 → 21.342 | 2.10% | 16.09% → 14.64% |
| R9700 | MXFP6 | 30.146 → 30.711 | 1.88% | 13.53% → 12.33% |
| R9700 | MXFP8 | 31.831 → 32.541 | 2.23% | 8.70% → 7.10% |

BF16の同期間の中央値はV620 24.914→25.003、R9700 34.863→35.030 tok/s。量子化MTPは改善したがBF16を上回らず、BF16既定を維持する。
全6比較行で同形式の生成token列・採用数は一致した。実形状演算子の形式別幾何平均は約1.20〜1.28倍で、推論全体への効果と区別する。
Cの局所検査はV620 9形状、R9700 5形状で前後digest一致と選択IDを確認した。
A/B/Cのmatmul device fatbinは両targetで完全一致し、命令列と資源情報を含む。実MTPの最終測定binaryはV620 MXFP6がB、他がAであり、Cの実6形状における選択との対応を記録した。
C build後に変更したsourceは、release binaryに入らないnative host testの期待値だけである。初回のhost期待値failureは保持し、修正後のhost test成功を確認した。
5本の実MTP profileで到達を確認した。V620 MXFP6のB profileは新ID100が889 dispatch、旧decodeが111 dispatchで、o投影だけ旧経路である。

本作業はローカル変更として完了し、commit/pushと常駐serviceへの新binary適用は行っていない。

## 検証・運用

native hostのselector／metadata／force検査、Rust operator harnessの22 test、Clippy、C++書式・静的検査、
JSON/manifest・Markdownリンクを確認した。GPU成功はexact target、独立oracle、非zero dispatch、fallbackなし、cleanupを伴う。
profileでは実MTPの6形状でColumns2への到達を照合し、V620 MXFP6のo投影だけ旧decodeへ戻ることも区別する。
本体・KV・MTPの全面的な再検証、公開CI、commit/pushは本ローカル実験の実施内容に含めない。
R9700の既存serviceは測定時だけ停止し、元のunit/binary/configへ復帰し、health/readyを確認した。Qwenローカルsubagentは停止したままである。
raw report、profile、buildは`.local-artifacts/phase85-m1/`へ保存し、Gitへ追加しない。

[計画](../../../../plans/archive/2026/09/11-20/phase85-mxfp-m1-mtp-followup.md) /
[メイン計画](../../../../plans/main-plan.md)
