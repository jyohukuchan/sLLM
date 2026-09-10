# Phase84: MTP重みの量子化

> 状態: 計画。2026-09-10ユーザー指示による次Phase。Phase83.5とモデル方向の共通化が完了した後に着手する。

## 目的と境界

BF16 companionの重み転送・演算費用を減らし、採用率を含めたMTP全体の実効速度を改善する。
Qwen3.8 27B NVFP4 target、標準MXFP8 E4 KV、固定T=1/P=.95/K20、single GPU/batch1を出発点とする。
Phase83.5の旧速度目標を自動的に必達条件として引き継がない。量子化形式・対象tensorと新しい数値目標は未決定。

## 実装前の確認

1. Phase83.5後の[モデル方向の共通化](../../../../archive/2026/09/1-10/phase83-common-speculation.md)を含む最終candidateのBF16 companionを比較基準として固定する。Phase83.5旧candidateは履歴比較に残し、共通化と量子化の差を混在させない。8192入力/128出力、同じGPU・seed・promptと計時境界を維持する。
2. 取得済みprofileからdraft、target検証、sampling、state管理の費用を分ける。kernel時間の合計をwall時間と混同しない。
3. companion専用tensorとtarget共有embedding/headを棚卸しする。共有tensorの変更でtarget側まで暗黙に量子化しない。draft用の別表現を保持する場合は追加VRAMも比較に含める。
4. 既存量子化providerとllama.cppのMTP/量子化実装を確認し、利用できるencoding、scale、shape、target能力で形式を選ぶ。4bitだから速い、8bitだから採用率を維持できるとは仮定しない。

## 実装と検証

- 変換・artifact identity・loader・graph・GPU provider・capability条件を揃え、通常CLI/APIから選べるよう統合する。設定名だけを追加して未到達の経路を残さない。
- MTPは投機的デコーディングの提案方式として、共通の採否・確定・sampling制御を再利用する。量子化tensor、head/hidden接続、KV/GDN状態はモデルadapterの責務とし、Qwen専用の生成制御を新設しない。
- 固定GPU samplingとp/q検証を維持する。量子化後のdraft分布を実際の提案・採否判定に用い、旧BF16分布を混在させない。
- 数値誤差、非整列shapeと境界、有限値・repeat・資源解放を既存の検査入口で確認する。
- BF16 companionとのdraft時間、採用率、検証block数、prefill/decode/TTFT/E2E、VRAMを両GPUで比較する。速いdraftと速い推論全体を区別する。
- token完全一致を品質条件にしない。評価可能な参照・入力・指標・許容差を実装前に明記し、sampling差と量子化差を区別する。未証明のBF16 full-model品質同等性を主張しない。
- 公開API、SSE、MTP進行後cancel/recovery、要求再利用、unloadと32GB級VRAM収容を確認する。
- 不採用候補は形式・対象・結果・理由を記録する。完了時にcommit/pushし、CIを確認・必要な修正を行う。

## 後続

従来Phase84の他精度単一要求最適化はPhase85、従来Phase85のbatchingはPhase86へ繰り下げる。

正本: [main-plan](../../../../main-plan.md)、[ロードマップ](phase76-qwen38-27b-nvfp4-priority-roadmap.md)。
前Phaseの変更・試行: [Phase83.5履歴](../../../../../history/2026/09/1-10/phase83-5-llama-guided-performance.md)。
