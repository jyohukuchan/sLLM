# Phase 81: 固定サンプリングの共通GPU経路とAPI性能

> 状態: 実装・ローカル検証完了。公開CIと完了処理は未完了。
> 開始日: 2026-09-08

## 方針と順序

ユーザーが採用したtemperature=1.0／top_p=0.95と無効sampler stageを公開APIへ適用し、
既存greedyに対するprefill／decode等の追加負担をほぼなくす作業をPhase81へ挿入した。
従来のstatic FP8 KV／MTP／文章生成をPhase82、他精度を83、NVFP4 batchingを84へ移した。
任意sampling設定への対応を狭める判断はユーザー決定であり、全モデルの最適品質を保証する変更ではない。

Qwen3.5 coding／Qwen3.8 thinkingのtop_kは20、Gemma4は64。
Ministral3のlocked metadataにはtop_kがなく、追加制限を入れないK0を明示採用した。
metadataの欠落を一般的なK0の意味や公式推奨とは扱わない。

## 実装

- GPU上でmask、top-k、top-p、抽選を実行する共通TokenSelect契約を追加した。
  K20／64は並列候補選択、K0はBF16値のhistogramとprefixによる正確なnucleus選択を使う。
  モデル名やNVFP4重みによるkernel分岐ではなく、語彙数、dtype、mask、K、target能力で選ぶ。
- request所有のworkspaceと制約bufferを再利用し、補正・maskなしの全語彙配列作成を省いた。
  通常samplingの全語彙logits D2HとCPU候補処理を除き、結果は16 bytesだけを読み戻す。
  grammar等の内部maskは維持し、変更時のCPU生成とH2Dは残る。無制約textの速度を制約付き生成へ一般化しない。
- Qwen共通経路、Gemma Dense／MoE、Ministralへ接続した。QwenのGraph replayとKV append/attention chainを
  sampling時も維持する。画像経路はembeddingとmRoPEを保ち、最終prefill行へ共通selectorを接続した。
- 公開APIは省略時と明示同値を同じ固定profileとして扱う。seed、出力上限、stop、対応済みtool／JSON／reasoningを保持する。
  未対応設定を黙って書き換えたり、CPU samplingへ送ったりしない。API仕様・fixtures・WebUIも同期した。
- GPU乱数契約は版を明示し、同一seed／counterの再生を検証する。旧CPU乱数列との一致は約束しない。
  公開C ABI全体の版と117関数は保持し、descriptorの拡張を版で区別する。CIのsource／symbol manifestも更新した。

## 検出して修正した不具合

K0のRust descriptorがlegacy版を選ぶ誤り、K0の並列nucleus境界で複数blockが共有値を更新する競合を修正した。
単純な同率fixtureだけでは後者を検出できなかったため、非同率248,320語彙、mask有無、複数seed／counterを追加した。
Qwen画像executorのselector転送漏れ、Ministralの明示logprobs=false拒否、Gemma MoEのproduction Argmax-only拒否も修正した。
取消済み要求は再利用を禁止し、公開済みKVを巻き戻さず要求破棄で解放する。

## 性能

以下は1回warmup・3回測定の中央値。Aは既存最速greedy、Bは同じ固定profileのCPU sampler、Cは共通GPU sampler。
同一モデル・target・KV・入力履歴・出力数を使う内部replayで生成内容の違いを除いた。
Qwen3.8ではA/C双方のGraph／chain選択、HIPのみ、fallbackなし、cleanup成功を確認した。

| V620 gfx1030・条件 | 指標 | A | B | C | C対A |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8 NVFP4、17入力／17出力 | prefill ms | 207.199 | 219.634 | 203.662 | −1.71% |
| 同上 | TPOT ms | 60.383 | 87.403 | 60.304 | −0.13% |
| Qwen3.8 NVFP4、9,435入力／128出力 | prefill ms | 29,670.015 | 29,670.706 | 29,844.548 | ＋0.59% |
| 同上 | TPOT ms | 64.368 | 90.649 | 64.115 | −0.39% |
| Gemma4 Dense NVFP4、17入力／17出力 | prefill ms | 1,538.350 | 1,543.269 | 1,545.895 | ＋0.49% |
| 同上 | TPOT ms | 60.510 | 85.256 | 60.921 | ＋0.68% |
| Ministral3 BF16、17入力／18出力、K0 | prefill ms | 213.902 | 224.927 | 216.475 | ＋1.20% |
| 同上 | TPOT ms | 48.842 | 83.912 | 46.693 | −4.40% |

Qwen3.8のGPU固定はCPU固定よりTPOTが約29〜31%短縮した。長文decodeのCPU時間中央値も11,880 msから8,400 msへ減少した。
これらの代表条件ではユーザー目標の「速度にほぼ影響しない」を支持する。5%は非拘束のAI提案であり、承認済み必達gateへ変更していない。
小差をsampling自体の負のコストとは解釈せず、測定ばらつきと併せて評価する。

R9700 gfx1201のQwen3.8実API codingでは44入力／64出力を揃えた要求全体の中央値が
A 3.734秒、B 5.350秒、C 3.520秒だった。CはBより34.21%短い。
自由生成ではtoken列が異なるため、sampler単体の改善量や厳密な同一演算比較とは扱わない。

## 正しさ・API・証拠の範囲

- native samplerはgfx1030／gfx1201の実機でK0／20／64、非整列語彙、境界、同率、非有限値、mask、seedを検証した。
  独立NumPy fixtureとC++ oracleを使い、GPU以外のfallbackを認めていない。
- Qwen3.5 MXFP8 KVとMinistral K0は両targetで実モデル接続が成功した。Gemma MoEの最終core検証はgfx1030で成功した。
  新規GPU、全形式の直積、全モデルの品質を検証したとは主張しない。
- Qwen3.5 BF16／FP16 KVのAPIは15項目成功。画像付きJSON、tool引数、reasoning、stop、生成token受信後の
  SSE切断と回復を含む。終了時allocation／workspace／quarantineは0。
- Gemma MoEのAPIは固定K64のseed再生、JSON制約、生成開始後の取消と回復が成功した。
  他profileのK指定がHTTP500になる不整合は修正済みで、最終binaryのChat APIでHTTP400を確認した。
  Completions／Responsesは`sllm` sampler拡張自体を公開しておらず、その指定をparserで400にする。
  初回の追加probeは誤ったparam名を期待して失敗し、既存契約に沿った2件の再確認が成功した。
  Gemma MoEの最終終了auditもallocation／workspace／quarantine 0、完了4要求すべてHIP・fallbackなしだった。
- host core 566件、frontend 96件、server 127件の統合時成功に加え、後続変更のfocused testを実施した。
  既存ignoreをGPU成功件数へ加えていない。公開HIP H3 compile-onlyは両targetで5段階成功した。
  workspace全target／全featureのClippy、Rust fmt、clang-format18、source inventoryも成功した。
  最終公開HEADのCIは別途確認する。

[集約値・artifact digest](phase81-fixed-gpu-sampling-evidence.json)に証拠との対応を保存する。
生ログ、binary、モデル、全raw sampleはGitへ保存しない。r3採取後のsource差分と再利用の根拠をfile hashで記録し、
nativeはarchived sourceへ同じclang-format18を適用した結果と現sourceのbyte一致を検証した。
coreの切上げ除算とbenchmark/testのliteral・castの同値修正を別記し、修正後serverの実機証拠へ対応付けた。

## 後続

Phase82でstatic FP8 KV／MTPを共通固定samplerへ接続する。確率的MTP、batchingの要求別RNG、
既存Qwen3.8専用APIで未対応のtool等の新規bring-upは本Phaseの成功へ含めない。
未対応の組合せをgreedyやCPU fallbackへ変換して動作したことにはしない。

[計画](../../../../plans/active/2026/09/1-10/phase81-fixed-gpu-sampling.md) /
[メイン計画](../../../../plans/main-plan.md)
