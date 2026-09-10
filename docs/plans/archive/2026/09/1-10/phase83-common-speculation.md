# Phase83・83.5: モデル方向の共通化と投機的デコーディング整理

> 状態: 実装・host／実機検証完了。2026-09-10のユーザー指示によるPhase84前の追加作業。公開commitのCIを確認する。

## 目的・受入条件

1. Phase83・83.5の採用済み変更を棚卸しし、既に共通、共通化する、モデル固有のまま保持するものと、その条件・理由・検証範囲を記録する。不採用実験を復活させない。
2. モデル名・artifact・layer数などによる最適化の限定を、問題がない範囲で演算の意味、数値契約、shape/layout、接続、状態能力に置き換え、通常の実行入口へ接続する。成果物そのもののidentity検証は維持する。GPUの既存適用条件やkernelの数値式は、この目的だけで拡大・変更しない。
3. MTPを投機的デコーディングの提案方式として整理し、共通の提案・検証・採否・確定制御と、モデル固有のhead/hidden/KV/GDN実装を分離する。既存の共通契約を利用し、名前変更だけで終えない。固定T=1/P=.95/K20のGPU確率処理と、部分採用・棄却・停止・cancel時の状態整合性を維持する。
4. 共通化した条件について異なるモデルidentity・グラフ/演算構成の選択と非適用条件を検査する。影響する既存host・GPU数値/公開経路検査で、実装した範囲を確認する。CPU検査を実GPU・全モデル品質の証拠にしない。
5. main-planとruntimeの説明、変更一覧と検証結果を同期する。統合reviewと関係するCI確認・必要な修正を行い、完了時にcommit/pushする。

## 境界

- 目標はモデル方向の再利用であり、新GPU対応・新しい速度目標・MTP量子化は追加しない。量子化は既存の次Phase84に残す。
- MTP専用の重み形式・hiddenの結合方法・状態遷移はモデルadapterの責務とする。MTP headのないモデルへのMTP追加や、未実装の外部draft実行エンジンの新規実装は要求しない。
- 現行の`sLLM.md`にあるモデル機能一覧のMTP表記より、今回の明示指示による「投機的デコーディングの手法」という整理を優先する。要件文書自体は無断編集しない。
- Phase83.5の旧速度目標緩和とBF16 full-model品質同等性未証明の記録は維持する。

## 作業

- native/HIPのselector・projection pack・MXFP8 attention・状態処理の適用条件を調査する。
- Residual/RMSNorm融合などのgraph最適化を、演算契約とconsumer関係に基づく共通処理へ移す。
- 投機的デコーディングのmethodとmodel adapterを分け、既存MTP実行を共通処理へ接続する。
- 変更に対応する検証を実行し、残る制限を変更一覧へ記録する。

## 結果

- Add/RMSNormを共通semantic passへ移し、QwenとMinistralの通常入口で既定採用した。Ministralは状態公開境界を保った51組を融合し、CLIの出力一致とdispatch削減を確認した。
- NVFP4 packの演算条件をdynamic shapeへ合わせ、Gemma decodeで48 packを実確認した。Gemma prefillは既存native採用shape条件外のため通常Matmulへ戻る。全14位置の全logitは共有無効時と完全一致した。
- MTPを提案方式として分類し、Qwen/Gemmaの採否制御、固定GPUのqueue・確定数・RNG計算、GPU p/q decisionの処理を共用した。head/hidden/KV/GDNとartifact metadataはmodel adapterに残した。
- core608成功・22 ignored、frontend101成功・1 ignored、HIP shape2成功。affected Clippyとformat、Qwen3.8 MTPのCLI/API・8192/128・SSE・cancel/recovery・cleanupを確認した。
- 必要なshape・state・artifact制約、初回失敗と修正、binary identityと検証範囲は対応履歴に記録した。次Phase84は共通化後candidateを量子化比較の基準にする。

正本: [main-plan](../../../../main-plan.md)、[runtime](../../../../../architecture/runtime.md)。
変更・検証: [履歴](../../../../../history/2026/09/1-10/phase83-common-speculation.md)。
