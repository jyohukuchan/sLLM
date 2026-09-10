# Phase84: MTP重みのMXFP8量子化と条件付きMXFP6対応

> 状態: 実装・ローカル検証完了（2026-09-11）。MXFP8はdecode退行のためBF16既定を維持。MXFP6は接続と限定検証まで完了し、移行条件未成立により包括的比較を保留。公開後CIは当該commitのchecksで確認する。

## 決定と目的

まずMTP companionをMXFP8 E4M3 W8A8へ量子化する。採用率・実効速度に大きな問題がなければ、同じPhase84内でMXFP6 E3M2 W6A6の実装・比較・採否まで進める。MXFP6を最初からPhase85へ先送りしない。

目的はBF16 companionの重み転送・演算費用を減らし、採用率を含むMTP全体の実効速度を改善すること。Qwen3.8 27B NVFP4 target、標準MXFP8 E4 KV、固定T=1/P=.95/K20、MTP幅2、single GPU/batch1、V620 gfx1030／R9700 gfx1201を対象とする。Phase83.5の旧速度目標や新たな必達倍率は設定しない。

低bit draftでも採用率を維持した研究は試す動機として扱う。[ML-SpecQD](https://arxiv.org/abs/2503.13565)のMXFP4 weight-only draftは、本PhaseのQwen MTP companion・W8A8／W6A6と異なるため、採用率や速度を保証する根拠にはしない。MXFP4／NVFP4 MTP、汎用FP8 artifact対応、再学習・distillationは本Phaseへ追加しない。

## 形式・対象tensor

| 構成 | Weight／activation | Scaleと出力 | 役割 |
| --- | --- | --- | --- |
| BF16 companion | 現行BF16専用重み、共有headは既存FP8 | 現行のまま | 比較・切戻し基準 |
| MXFP8 companion | E4M3 W8A8 | K-axis block32／E8M0、FP32累積／BF16出力 | 最初に実装・評価 |
| MXFP6 companion | E3M2 W6A6 | K-axis block32／E8M0、FP32累積／BF16出力 | MXFP8に大きな問題がなければ同Phaseで評価 |

- 初期対象はcompanion専用の8行列：`mtp.fc.weight`、self-attentionのq/k/v/o projection、MLPのgate/up/down。normの7 tensorはBF16を維持する。
- targetのNVFP4/FP8重み、共有BF16 embedding／FP8 output head、KV encodingは維持する。companionだけに適用するrecipeを明示し、targetを暗黙に再量子化しない。
- W8A8／W6A6はweight-onlyではない。BF16 activationを各形式へ動的量子化する費用と誤差も測る。W6A8等への変更はこの形式決定へ暗黙に混ぜない。
- 8行列を個別に対応確認し、感度や性能に問題があるtensorをBF16へ残す場合は、その理由と実際の混合recipeを記録する。「companion全量子化」と説明しない。

## 実装順

### 1. 基準と既存providerの適用確認

- 共通化後source `c27e346d5f3332d6e8ce794b7913b87c42455c99` のBF16 companionを基準にする。docs-only commitの違いはsource/build identity対応を確認し、同じ証拠を再利用する。
- 取得済みprofileからdraft／target検証／sampling／state管理の費用を分ける。kernel時間の合計をwall時間と混同しない。
- 8行列のK/N、decodeのM=1、prefix準備のM、encoding/scale/layoutとGPU能力を既存MXFP8 providerへ照合する。実行可能性と高速variant採用条件を分けて確認する。
- 既存converter／codec／providerを流用し、llama.cpp reuseも検討する。形式名だけで新しい専用kernelが必要だと決めない。必要なkernel修正は演算・shape・target条件に基づく共通経路へ適用する。

### 2. MXFP8を通常経路へ統合

- BF16 companionからの変換、value/scale保存、source hashとrecipeを含むartifact identity、loader、weight plan、graph binding、GPU providerを一続きに接続する。
- targetとcompanionが異なるencodingを持てるよう、検証・resident共有・cache identityをcompanion recipeに対応させる。BF16固定の検査を単純に削除せず、対応するshape/range/scaleを検証する。
- 通常CLI/APIのmodel設定からcompanion形式を選択できるようにし、選択されたrecipeを記録する。初期比較には明示選択を使い、BF16への切戻しを維持する。性能採否後にGPU能力・対応recipeごとの既定を決める。
- 共通の投機的デコーディング制御を再利用する。head/hidden・KV/GDN状態はmodel adapterに残し、Qwen専用の新しい採否制御を作らない。
- 提案生成とp/q採否には同じ量子化draftの実際の分布を使う。旧BF16 draft分布を混在させない。固定GPU sampling、補正、commit／rollbackを維持する。

### 3. MXFP8の結果でMXFP6へ進む

「大きな問題がない」は、正しさと資源管理が成立し、採用率・検証負担・速度の同条件比較から実用上の退行が認められないこととして判断する。

- 採用率の小さな低下だけで中止しない。draft短縮が採用率低下や検証回数増を補い、decode/E2Eを維持・改善していればMXFP6へ進む。
- 反復のばらつきを超えるdecode/E2Eの低下、8192入力のprefill/TTFTの実用上の悪化、採用率の大幅減と検証負担増、非finite・sampling/state不整合があれば、該当条件の接続・数値・provider選択を調べる。疑わしい行だけを再測定し、必要最小限の修正・対象tensor調整を行う。
- 同条件で小差しかない場合も、MXFP6を試す目的は残る。新たな固定percentageや独立reviewを移行の必須条件にしない。速度だけでなく言語/タスク別の外れた悪化も確認し、判断理由を残す。
- 条件を満たした場合、再度の着手確認を挟まずMXFP6の変換・通常経路統合と比較まで進める。問題が残る場合はMXFP6を自動的に既定化せず、未実施理由を記録する。一方のGPUだけ問題がある場合はGPU別の結果を分け、成立する側の比較は進められる。

### 4. MXFP6を同じ仕組みで比較・採否

- MXFP8で作ったartifact/loader/graph接続を共有し、既存E3M2 packing・W6A6 providerを接続する。scale/layout/encodingで分岐し、生成制御を複製しない。
- BF16／MXFP8／MXFP6を同じGPU・prompt・seedで比較する。MXFP6が8bit系より速い、または十分な採用率を保つとは仮定しない。
- 採用率とdecode/E2E、prefix準備、VRAMを総合してGPU/recipeごとの採否を決める。MXFP6が不利ならMXFP8、両方が不利ならBF16を既定に維持できる。試行した形式と棄却理由を残す。

## 比較・検証

- **速度基準:** [共通化後再計測](../../../../../history/2026/09/1-10/phase83-common-qwen38-remeasurement.md)の8192 input／128 output、chunk2048／state8320、seed123、1 warmup＋3 measured。BF16 companionのprefill/decodeはV620 216.593/25.423、R9700 541.969/35.110 tok/s。旧Phase83.5は歴史比較に残し、共通化差を量子化差へ混ぜない。
- **採用率基準:** [言語・タスクsuite](../../../../../history/2026/09/11-20/mtp-language-task-acceptance.md)と同じ12 prompt×3 seed、最大128出力、各GPUで1 warmup。GPUごとにBF16と比べ、accepted/proposed/rejected、検証block数、確定出力数、seed別・条件別・全体token加重率を記録する。既存baselineの全体率はV620 66.59%、R9700 67.06%。GPU間の率一致を要求しない。
- draft時間、採用率、検証回数、prefill/decode/TTFT/E2E、resident/peak VRAMを分ける。棄却tokenはthroughputへ加算しない。初回とwarm値、短いAPI要求と8192速度行も分ける。
- 変換・kernelは独立した量子化/復号oracleとBF16参照で誤差・有限値・repeatを確認する。非整列N、行数境界、scale境界を含め、K非32倍は既存契約どおり拒否する。対象演算の数値許容差は既存MXFP WA oracleの絶対誤差0.5または相対誤差0.02以内を使用し（両方超過で失敗）、新形式の誤差を同形式の実装不具合と混同しない。
- 生成tokenの完全一致は品質条件にしない。既存coding等の入力で反復崩壊・不自然な出力が新規に生じないか確認するが、採用率や生成例だけでBF16 full-model品質同等性を認定しない。異常があれば同一prefixでの限定比較等で原因を切り分ける。
- 実HIP・CPU fallbackなし・非zero dispatchを確認する。通常CLI/API、SSE、MTP進行後cancel/recovery、要求再利用、unload、32GB級VRAM収容を採用候補で確認する。検査失敗をCPU emulationや別precisionの結果でPASSにしない。

## 完了範囲と記録

MXFP8の通常経路統合・比較・採否、移行条件を満たした場合のMXFP6の通常経路統合・比較・採否、正しさ/公開経路/資源管理、不採用理由の記録までをPhase84とする。MXFP6の採用や特定倍率の達成は必須ではない。移行条件を満たしているのにMXFP8だけで完了扱いにしない。MXFP6を進めない場合は問題・試行・残件を明記し、未実施を完了済みと説明しない。

詳細結果は対応する `docs/history` へ残し、main-plan・ロードマップと同期する。完了時に計画をarchiveへ移し、commit/push、CI確認と必要な修正を行う。docs-onlyの計画更新はPhase84実装完了とは区別する。

## 後続との境界

Phase85の他精度単一要求最適化、Phase86のbatchingは維持する。Phase84で実装・検証した共通MXFP8/MXFP6改善はPhase85へ引き継ぎ、同じ作業を重複計画しない。全モデル・全shapeのMXFP decode最適化やMXFP4 W4A8の残件までPhase84へ取り込まない。

正本: [main-plan](../../../../main-plan.md)、[ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。
前Phase: [共通化計画](../../../../archive/2026/09/1-10/phase83-common-speculation.md)、[Phase83.5履歴](../../../../../history/2026/09/1-10/phase83-5-llama-guided-performance.md)。
計画変更履歴: [2026-09-11の形式・順序決定](../../../../../history/2026/09/11-20/phase84-mtp-quantization-plan.md)。

実装・測定履歴: [Phase84量子化](../../../../../history/2026/09/11-20/phase84-mtp-weight-quantization.md)。

## 完了結果

- MXFP8/MXFP6の8行列sidecar変換・検証・loader・graph・resident共有と通常CLI/API選択を実装した。既定と切戻しはBF16。
- MXFP8の12条件×3 seed採用率は両GPUで概ね維持されたが、最終matched decodeはBF16比約17.2%／9.7%低下した。追加した共通M=1候補も比較し、採用せず撤去した。
- この速度退行によりMXFP6の包括的採用率・性能比較への移行条件は成立しなかった。MXFP6の両GPU operatorとAPI限定smokeは成功したが、包括的品質・性能評価とは区別する。未評価範囲はPhase85のprovider改善時に再検討する。
- 両GPU・両形式で各15ケースの数値oracle、APIのSSE・MTP進行後cancel/recovery・8192/128・再利用・解放・32GiB未満収容を確認した。最終Rust互換修正後のCLI generate（両形式）/chat（MXFP8）も成功した。
- ローカルH0 627件、H1 1564件、H2 38件が成功した。新binの依存台帳とRust 1.85非対応構文を修正した。source/build対応と未評価範囲は履歴に記録した。

完了履歴: [Phase84実装・測定](../../../../../history/2026/09/11-20/phase84-mtp-weight-quantization.md)。
