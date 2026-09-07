# Phase 79: 既存最適化の共通化・条件付き既定採用

> 状態: 完了（2026-09-07開始・完了）
> 作成日: 2026-09-07
> 根拠: ユーザー指示による開発方針の明記と旧Phase 79以降の繰り下げ。

## 目的と範囲

既存の高速経路を、モデル名や手動環境変数への依存を減らして、対応する演算・GPU・数値形式で自動選択できる状態にする。
恒久方針は[メイン計画](../../../../main-plan.md#最適化の共通化と既定採用の方針)を正本とする。
Phase 78の完了・目標変更は維持し、旧速度gateを再開しない。

- 対象: 既存NVFP4/FP8 decode候補、projectionのactivation量子化共有、prepared execution／completion／Graph制御、適用状況の観測。
- GPU: V620 gfx1030、R9700 gfx1201。GPU別の採用範囲を許容する。
- KV: 既存FP16とOCP MXFP8経路の実行条件を整理する。新しいstatic tensor FP8 KVのmaterialization、専用高速Attention追加はPhase 80へ残す。
- 対象外: MTP、新しい精度形式、汎用FP8 artifact互換、batching、vision、multi-GPU、モデル固有の追加速度探索、汎用グラフコンパイラ・自動調整DBの新設。
- 稼働サービスへの配置変更や公開は、この計画の実装完了と分離する。計画作成後のユーザーgoal「Phase79を完了する」により実装・検証を開始した。配置・公開の指示は含まない。

## 2026-09-07着手記録

上記の範囲と以下の完了条件を実装前の基準とする。数値変更の採否はmain-planのN1/N2/N3方針に従う。
既存selector整理、共通execution/projection共有、数値oracleの準備を分担し、主担当が統合とGPU検証を行う。
作業開始時点でfrontend/serverとKV evidenceの未コミット変更があるため保持する。
V620 x2用ローカルQwenは停止済み。R9700では既存sLLMサービスが稼働しており、まず空いているV620で検証する。
既存Phase78 CLIによる候補別Gemma測定は探索比較とし、最終Phase79 source/buildの証拠とは区別する。

## 着手時に固定する現状

既存の未コミット変更を保持し、対象source/buildと測定artifactの対応を確認する。古いbinaryとの全体比較を個別最適化の寄与へ読み替えない。

- `crates/sllm-core/src/qwen_execution.rs`: モデル固定のprojection共有・deferred completion・Graph有効化条件、FP16 KV chain。
- `native/hip/src/matmul_kernel_internal.hpp`: 共通NVFP4/FP8 selector、decode候補と形状限定候補。
- `native/hip/src/causal_attention_runtime.inc`: FP16専用GQA候補とKV形式別選択。
- 既存共通execution層、HIP prepared API、CLI/serverの実行監査: 抽出済みの機構を調べ、重複基盤を作らない。

開始時の一覧は各候補について、意味上の対応条件、現行既定、opt-in、実測済みGPU/モデル/KV、未確認事項を記録する。
モデル指紋はartifact検証として維持し、最適化のモデル限定判定と区別する。

## 作業順と成果物

### 1. 選択条件と適用状況の可視化

- 既存selector・有効化判定を整理し、正しく実行できる条件と性能上の採用条件を分離する。
- GPU、semantic op、shape/layout/alignment、重みencoding、KV encoding、Graph条件を選択キーとする。
- 未対応、明示無効、性能上の非採用を区別し、選択されたvariantと理由を既存のprepare/起動監査または測定出力へ追加する。
- tokenごとのログ出力や判定を増やさず、prepare時に確定できる条件を再利用する。
- 強制選択は比較・切戻し用に維持する。未対応条件では既存の対応GPU経路を選ぶか明確に拒否し、CPUへの暗黙fallbackを追加しない。

成果物: 条件一覧、共通判定、選択理由を確認できる出力、境界条件のhostテスト。

### 2. 共通decode候補の数値確認と条件付き既定化

- 既にモデル非依存なNVFP4 DP4A decode、FP8 decode候補を優先する。候補を一つずつON/OFFし、他条件を固定する。
- Gemmaで既定NVFP4変更に伴って観測した生成差と短文prefill退行は、影響する演算出力/logitsを確認し、誤差の許容性とselector範囲を判断する。token差だけで不具合とも品質同等とも判定しない。
- 既存の基準演算と独立数値oracleを使い、累積順序の差と量子化自体の差を分ける。
- 確認したGPU・形状範囲だけ既定採用する。効果がない候補はopt-in維持または不採用理由を記録し、追加のモデル固有探索へ移らない。

成果物: 候補別の数値・性能比較、採用範囲を反映したselector、切戻し方法。

### 3. 非同期・Graph制御の共通化

- prepared cache、same-stream segment owner、completion集約、capture/replay制御の再利用可能部分を共通execution層へ抽出する。
- Qwen固有graph、attention preprocess、GDN、model stateはadapterに残す。
- KV encoding、scale・bufferの寿命、context growth時の更新、Graph互換性を明示する。Graph非対応区間では通常GPU実行を使用する。
- FP16専用KV append/Attention chainの条件を単に削除しない。既存MXFP8で再利用できる制御を接続し、専用kernelが必要な部分は非対応理由と後続範囲を記録する。
- cancel、要求間再利用、context変更、unload時にも所有権とcompletionを維持する。

成果物: 共通制御とadapter境界、FP16/MXFP8別の適用表、再利用・終了処理の確認。

### 4. projection共有の演算条件化

- 同じactivationを参照し、量子化方式・scale recipe・layoutが共有可能なprojectionを実行計画から判定する。
- 固定モデル条件をその意味条件へ置き換える。単にfingerprint検査や形状制約を外さない。
- 読み取り専用activationの寿命と共有workspaceの所有権を明示し、通常の個別projection経路を維持する。
- 適合する別モデルで経路選択と数値・性能を確認する。形状固有kernelの追加は今回の共通化へ混ぜない。

成果物: 演算条件による共有計画、非適合時の通常経路、別モデルへの適用確認。

## 検証方法

- 既存sllm-validationの入口を変更範囲に応じて選ぶ。GPU作業前に互換性文書とGPU利用状況を確認する。
- host: 条件の両側、未対応encoding/layout、明示override、所有権・終了処理。必要なHIP compile-onlyを実施する。
- 演算: 端数・非整列値、tile境界の前後、実モデルの代表形状を用い、対象GPU上の結果を数値oracleと比較する。誤差基準は既存契約を確認して実装前に固定し、測定結果に合わせて緩めない。
- モデル: 互換のある最小Qwenをアーキテクチャ共通変更の代表とし、異種モデル共通変更は既存Gemma artifactで確認する。Qwen3.8 27Bは小型モデルで覆えない混合recipe・固有形状・統合箇所に絞る。
- 性能: 同じsource/build・モデル・GPU・prompt・KV・sampling条件で候補のみを変更する。短文と長めの代表条件でwarmup後の反復値を取り、TTFT、TPOT、peak VRAMとばらつきを記録する。全候補×全モデル×全KVの直積を必須にしない。
- KV差と最適化差を同時に変更した比較は総合比較として分離する。prefillの改善倍率をdecodeへ流用しない。
- 数値的に使えることとモデル品質・速度を分ける。必要な範囲で固定入力logits、top1/KLD等を確認し、短い生成列一致だけで品質PASSを主張しない。
- GPU証拠は対象target、実dispatch、数値oracle、fallbackなし、cleanupを確認する。CPU試験やcompile-onlyをGPU PASSにしない。
- 影響箇所をまとめたintegration reviewを1回行い、以降は変わった指摘だけ再確認する。Phase 78の長時間試験・別GPUの全件再測定を一律に追加しない。

## 完了条件と後続への引継ぎ

- 対象候補の適用条件・選択理由が確認でき、対応外の条件は通常経路または明確な拒否になる。
- 共通decode候補の採否と根拠が記録され、採用候補は確認済み範囲で手動opt-inなしに選択される。不採用候補が残ること自体は未完了にしない。
- 共通execution制御とprojection共有の責務が整理され、適合する別モデルで実際の適用と影響を確認する。
- FP16/MXFP8で共有できる制御と形式別kernelの未対応部分を分離し、数値・資源・終了処理の確認を終える。
- 全モデル対応、一律改善倍率、全KV形式での全Graph化は要求しない。Phase 80のstatic FP8 KV・MTPを先取りしない。
- 実装後の契約は`docs/architecture/runtime.md`、詳細結果は対応する`docs/history/2026/09/1-10/`の履歴へ記録する。完了時に本計画をarchiveへ移し、計画・履歴末尾を相互リンクする。

既存の再計画条件を適用し、作業中に新しいレビュー段階や測定gateを追加しない。
共通化がモデル固有の追加速度探索へ変わる場合は、その候補を止めて範囲を見直す。

## Phaseの対応

| 現行Phase | 内容 | 旧番号 |
| --- | --- | --- |
| 79 | 本計画: 共通化・条件付き既定採用 | 新設 |
| 80 | static FP8 KV、MTP、文章生成の実用closeout | 79 |
| 81 | 他精度の単一要求最適化 | 80 |
| 82 | NVFP4 batching | 81 |

[メイン計画](../../../../main-plan.md) · [後続ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)

## 完了確認

上記4作業と完了条件を実装・実機確認した。ID67/68は条件付き既定採用、Gemma適合projection共有は既定採用。
共通deferredと既存モデル固有opt-inは採否理由を記録して維持する。ID59は誤差boundを低減し基準加算順へ復元した。
core 561 PASS、public-runtime host、両target build、独立operator oracle、両GPU Gemma固定logits、
V620 Gemma性能、Qwen GraphとFP16/MXFP8 KVの共通completion、cleanupを確認した。
統合reviewの3指摘を修正しfocused再確認済み。各完了条件の証拠・限界は以下の履歴を正本とする。

[実装・検証履歴](../../../../../history/2026/09/1-10/phase79-common-optimization.md)
