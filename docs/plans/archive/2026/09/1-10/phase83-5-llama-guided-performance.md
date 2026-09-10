# Phase 83.5: llama.cppを参考にしたMXFP8・固定sampling・MTP最適化

> 状態: 実装・検証完了。2026-09-10ユーザー承認により旧速度条件を緩和した。公開commitのCI確認を最終公開手順で行う。

## 目的と受入条件

[main-plan](../../../../main-plan.md)と[ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)のPhase83.5を実行する。
2026-09-10のユーザー指示により速度目標を緩和し、以下の旧目標は参考値として保存する。
追加の速度追求は終了し、正しさ・公開CLI/API・資源管理・最終比較記録・CI・公開の残件を閉じて完了する。
MTP量子化は次のPhase84へ分離し、現Phaseの追加実装には含めない。

| GPU | prefill | decode |
| --- | --- | --- |
| V620 gfx1030 | 200 tok/s以上 | 20 tok/s以上 |
| R9700 gfx1201 | 500 tok/s以上 | 25 tok/s以上 |

- Qwen3.8 27B NVFP4、standard OCP MXFP8 E4 KV、T=1／P=.95／K=20、MTP有効、single GPU／batch=1。
- template後の実8,192入力／128確定・公開出力。prefix cache再利用なし。1 warmup＋3 measuredの中央値を最終比較記録に用いる。速度の閾値は完了条件にしない。
  prefillにMTP prefix準備、decodeにdraft／verify／sampling／棄却・replayを含め、棄却tokenを速度に加えない。
  TTFT、end-to-end、ばらつき、採用率、VRAM／GTTとcleanupも記録する。
- CLI/API、短い対話、SSE、cancel/recovery、要求再利用、unload、32 GB級VRAMへの収容を維持する。
  CPU fallback、model/KVのGTT spill、常駐FP16 mirrorを高速化の代替にしない。
- MTP有無の出力完全一致は必須ではない。同一BF16参照比の精度劣化が同程度なら差を許容するユーザー方針を引き継ぐ。
  状態破損やsampling不具合は許容しない。未測定のBF16 full-model品質同等性を生成例やkernel oracleで認定しない。
- tools完全対応、vision、全モデル、TP、batchingは対象外。旧速度目標との差と未解明の原因を記録する。新たな数値下限は設定しない。
  完了時はcommit・pushし、公開CIを確認して必要な修正を行う。

## 基準と参照方針

- 正しさの出発点はPhase83公開HEAD `e1b8cfa349090fb343a24f20e80fadf4dfca7ed9`。
  Phase82 `63ef9057f6265d99e38b254b8fb31d0b426859a4`との比較も維持する。
  [Phase83の初回参考値](../../../../../history/2026/09/1-10/phase83-first-request-comparison.json)は単回値であり、正式反復の代用にしない。
- 入力token SHAは`855240c09609a19b9c1124043b763ecc97e3cfde84a16acfaba445a0f8c83b23`。
  同一model lock、tokenizer、GPU個体、toolchain、seed 123でMTPなし／ありを分ける。
- ユーザー指示により、基本の実装参照はllama.cppとする。最初の参照はローカルのcleanな
  `reference/llama.cpp` commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`へ固定する。
  speculative/MTPのbatch・state・sampling、HIP/CUDA MMQのtile／dispatch、FlashAttentionを調べる。
  upstreamのGGUF量子化形式・数値契約をsLLMのNVFP4／MXFP8へ無条件に置き換えない。
- 直接reuseを新規実装前に検討する。copy／adaptation／portを行った場合は
  [provenance方針](../../../../../provenance/README.md)に従い、実際の参照revision・source・変更・noticeを公開前に記録する。
  参照調査だけの段階でimportを行ったとは記載しない。
- 共通演算、GPU能力、shape/layout、encodingによる改善を優先する。モデル名だけの分岐は増やさない。

## 着手時の実行順

以下は実装時の手順であり、速度追求終了後の追加作業を指示するものではない。

1. **着手: 遅い箇所と最初の変更を確定する。** Phase83の既存trace／operator probeを再利用し、
   llama.cppとの差をMTP処理とprefill providerに分けて調べる。不足する区間だけ計時を追加する。
   全体を毎回長時間再測定せず、同じ症状・同じbuildの既存証拠は対応を確認して再利用する。
2. **prefillの既定経路を改善する。** ID87／ID89とQTILE4等の候補を既存の数値根拠・境界・性能で評価する。
   N2の経路を速いという理由だけで既定化しない。必要ならllama.cppのMMQ tile／load／reduction構造を移植する。
3. **MTP decodeの余分な処理を削る。** 小M target verify／draft matmul、MXFP8 attention、queue同期、
   hiddenコピー、state snapshot／replay、samplerの費用を切り分ける。既存ID88／90／92／93とGraph再利用も評価する。
   採用tokenやRNG位置、rollback、hidden更新の意味を維持する。MTPを無効化して目標達成とは扱わない。
4. **通常の公開経路へ統合する。** 候補単体で改善してもCLI/APIが従来経路のままにならないよう、
   capability・selector・metadata・実dispatchを揃える。必要な数値分類と切戻し条件を記録する。
5. **両GPUで正式測定し公開する。** 同条件のMTPなし／あり、1＋3測定、影響するlifecycleを確認し、
   試行・不採用理由・達成値・残差を履歴へまとめる。CIは実際の登録入口で確認する。

## 部分採用改善の設計記録（r8時点）

- r8では部分採用時にtarget全体をaccepted prefixで再実行している。llama.cppのrecurrent state rollbackを参考に、
  幅2のtarget M3検証中にrow0／row1後のGDN conv／recurrent stateを保存し、採用位置へ復元する案を次に評価する。
  KVは既存のlogical length巻戻しを使い、companion stateとsampling RNGの消費規則は維持する。
- Qwen3.8の48 stateでは追加2 planeが293.625 MiB、部分採用時の1 plane復元が146.8125 MiB。
  full acceptでも保存費用は発生する。全targetの再実行を省いてもD2D copyとHIP API呼出しは増える。
- 実装前にper-row activation quantization／scaleの因果性とgraph内pointerの寿命を確認する。
  private Rust/native bridgeを使い、installed C ABIは拡張しない。全stateの開始長・generation・容量・providerを
  実行前に確認し、非対応の場合だけ従来経路を使う。復元先を変更した後の失敗はrequestを失敗させcleanupし、
  部分的に上書きしたstateでreplayへ戻さない。採用planeは現在activeなM3終了slotへコピーし、
  inactiveなpre-block slotを保持する。初期案のpre-block slotへのコピーでは既存rewindが未来のM3状態を
  返してしまうため採用しない。commit後に開始位置へrewindしたpayloadも検査する。
- 正しさはM3中に計算されたprefix stateの保持、accepted length、KV長、hidden、次step、
  full／partial accept、cancel・再利用・解放で検証する。legacy M1/M2再実行とはmatmulの丸めが違い得るため、
  bitwise一致を先に保証せず差を測定・分類する。未測定のBF16品質同等性は認定しない。
  public経路・両target・8192/128の時間とVRAMを確認して採否を決める。上記Phase全体の目標は変更しない。

## 初期観測

Phase83の通常設定初回値はV620 prefill/decodeがMTPなし8.305／8.108、あり8.257／2.254 tok/s、
R9700がなし10.715／9.772、あり10.703／2.127 tok/sだった。機能の完成と速度の達成を区別する。
Phase83の高速opt-in探索は正式な既定値・速度達成値ではなく、候補の優先順位を決める参考とする。

## 最終検証（2026-09-10）

- 速度の追加追求は終了。旧目標とR56測定値の差を履歴に保持する。
- 両GPUの最終candidateでMTP有無1+3、公開CLIとAPI/lifecycleをPASS。host検査、lint/manifest修正と累積reviewを完了した。公開commitに対するhost/H3 CIを公開時に確認する。
- MTPあり中央値はV620 prefill216.571/decode25.409、R9700 541.402/34.541 tok/s。初回値と反復値を分け、速度条件の緩和決定を保持する。
- BF16 full-model品質同等性は未証明であり、kernel oracleや生成一致から主張しない。
- 次Phase84はMTP量子化。従来の他精度最適化は85、batchingは86へ繰り下げる。

詳細な変更・不採用の試行・測定と訂正: [Phase83.5履歴](../../../../../history/2026/09/1-10/phase83-5-llama-guided-performance.md)。
