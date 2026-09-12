# Phase84.5: MTP経路の限定的な正しさ診断

> 状態: 限定診断完了（2026-09-12）。実行順は84→84.5→85→86。Phase84完了とBF16 MTP既定を維持する。

## 目的と範囲

現在のMTP採用率に、接続・実行制御・sampling・状態復元の不具合が寄与していないかを短い固定入力で確認する。モデル本体やMTP head自体の予測能力、量子化による分布の違いを改善するPhaseではない。採用率の最低値、他エンジンへの一致率、速度目標は設定しない。

対象はQwen3.8 27B NVFP4 target＋BF16 MTP companion（共有headは現行FP8）、標準MXFP8 E4 KV、T=1／P=.95／K20、幅2、batch1、single GPU。V620 gfx1030とR9700 gfx1201をGPUごとに評価し、相互のbit一致を求めない。MXFP8/MXFP6 MTPの全面比較は追加しない。

半日程度を診断の作業目安とし、既存検査を優先利用する。これは実行中の正常な処理を停止するタイマーでも、不具合を未解決のままPASSにする期限でもない。広い参照エンジン構築や全面再実装が必要と分かった場合は、未確認点と最小の次手を記録し、同じ作業単位の範囲を見直す。

## 実施順

1. **既存証拠の棚卸し。** Phase83／83.5／84の検査から、tokenとhiddenの位置合わせ、MTP入力norm／fusion／出力head、position、KV／GDN状態、prefix準備、固定p/q、commit／rollbackの確認済み部分を特定する。source/build/model identityが対応する証拠は再利用し、未確認箇所だけを追加する。
2. **接続仕様の独立確認。** 固定model revisionに対応する参照実装・metadataで、どの位置のtarget hiddenとtoken embeddingから何番目のtokenを予測するか、norm・head・positionの契約を確認する。通常経路と参照実行が同じ誤ったindex処理を共有するだけの比較にしない。参照元のrevisionと確認した式・位置対応を記録する。外部codeの扱いは既存provenance方針に従う。
3. **同一履歴での数値照合。** 既存suiteのcodingと日本語入力から短い固定token列を用意する。自由生成の分岐を避け、同じprefix・token列・状態で通常経路と単純な逐次参照を比較する。既存debug／replay入口を優先し、Graphやbuffer再利用を外すためだけに新しい推論エンジンを作らない。target hidden、MTP fusion入力、draft logits、固定sampling後のq、検証側pを必要な境界で照合する。参照のtarget/draft数値形式は通常経路と揃え、本体をBF16へ交換した差を経路不良と混同しない。
4. **採用・棄却と次状態。** 幅2の「0個採用」「1個採用」「2個採用」を小さいfixture／制御された乱数で通す。token、position、KV有効長・内容、GDN等の次計算に必要な状態を、確定tokenだけを逐次実行した参照と比較する。復元直後だけでなく次のproposalとtarget logitsまで確認する。強制fixtureの比率を実採用率として報告しない。
5. **固定samplingの照合。** top-k→top-pの順序、正規化、候補集合、採用確率min(1,p/q)、棄却時の補正分布、全採用時の追加tokenを小さい独立CPU/NumPy oracleと比較する。q=0／support差／同率候補と判定境界の両側を既存検査で確認し、不足だけを補う。CPU oracleをGPU実行成功の代わりにはしない。
6. **差が出た箇所だけ修正。** index・状態・分布の不整合があれば最初に差が生じた境界へ絞り、該当検査と影響する通常経路を再確認する。差がなければ追加の速度改善や採用率向上探索は開始しない。

入力は短い2ケースを基本とし、既存検査と合わせて非整列token長、実際のchunk／state境界の両側を小さいfixtureで含める。8192入力や12条件×3 seedの再実行は、今回の変更・失敗に具体的な必要性がある場合だけ選ぶ。

## 既存実装・検査の入口

- `crates/sllm-frontend/src/generation.rs`: MTP executor、hidden/token対応、commit/replay、固定K20接続。
- `crates/sllm-core/src/speculative.rs`: 共通verify／commit／abort契約。
- `native/hip/tests/phase83_5_linear_checkpoint_gpu_test.cpp`: 部分採用・rewindを含む状態検査。
- `native/hip/tests/phase83_5_token_selector_pq_gpu_test.cpp`: 採否・補正・境界乱数のGPU oracle。
- `native/hip/src/token_selector_pq_algorithm.hpp`: p/q計算と乱数domain。独立oracleはこの実装の単純複製にしない。

これらの存在だけで経路全体を検証済みとはしない。実行証拠と対象範囲を着手時に対応付ける。

## 判定と終了条件

- 離散的なtoken位置・採否・有効長・状態遷移は契約どおりであることを確認する。数値は適用可能な既存oracleの許容差を使用し、新しい境界では演算精度・累積順に基づき実行前に根拠を記録する。観測した失敗を通すために許容差を緩めない。近接logitsの順位変動だけを経路不良と断定せず、確率誤差と判定境界も見る。
- 両GPUで不足していた限定ケースの実HIP・非zero dispatch・fallbackなし・正常終了を確認し、既存証拠の再利用範囲と追加実測を区別する。未実施・timeout・crashはPASSにしない。
- 結果を「限定範囲で経路不整合を検出せず」「不具合を修正し同範囲で確認」「未解決／未確認」に区別する。未解決の経路不具合がある場合は正しさ確認完了としない。差がなくてもモデル固有の原因を証明した、全長・全モデルで正しい、という結論にはしない。
- 実装変更時は影響するhost／HIP／通常CLI/API検査だけを選ぶ。診断のみなら全面のAPI lifecycleや性能検査を繰り返さない。量子化形式の既定、sampling設定、幅2は維持する。
- 完了時に履歴へ確認範囲・参照仕様・数値差・修正・未確認点を記録し、本計画をarchiveへ移して相互リンクする。既定の完了方針に従いcommit・pushし、当該commitのCIを確認して必要な修正を行う。

## 対象外

採用率を80%等へ引き上げる目標、MTP head再学習、本体の精度改善、新kernel最適化、MXFP8/MXFP6再選定、全モデル・長文・多GPUの網羅、他エンジンの大規模benchmarkは含めない。

## 参照

- [main-plan](../../../../main-plan.md)
- [ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)
- [Phase84実測・追加MXFP6比較](../../../../../history/2026/09/11-20/phase84-mtp-weight-quantization.md)
- [既存言語・タスク採用率](../../../../../history/2026/09/11-20/mtp-language-task-acceptance.md)

履歴: [Phase84.5診断記録](../../../../../history/2026/09/11-20/phase84-5-mtp-path-correctness.md)。

## 完了結果

接続仕様、両GPUのMTP hidden／固定q、既存p/q・checkpoint oracleを確認した。V620は固定履歴のtarget M3/M1と採用0/1/2個後の状態がexact。R9700のtarget数値差はattentionのM1 wave／M3 genericへ切り分け、同じ演算の診断対照では状態と次計算までexactだった。通常attentionの独立oracleも8/8 PASSした。本番の演算・既定設定は維持する。

初期full-model 3ULP screen超過は履歴に残し、閾値を緩めてPASSにしない。最終判定は各演算oracleと同一演算の状態対照による限定診断である。既定MTP on/offの数値一致、BF16比品質、採用率低下の全原因は未証明で、今回の対象外とする。公開時は既定方針どおりcommit・pushし、当該commitのCIを確認する。

履歴: [Phase84.5診断結果](../../../../../history/2026/09/11-20/phase84-5-mtp-path-correctness.md)。
