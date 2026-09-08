# Phase 82: 不採用最適化の削除・条件付き既定採用

> 状態: 実装・検証完了（2026-09-08）。対象CIの確認を含む公開手順を適用。
> 作成日: 2026-09-08

## 目的と決定

2026-09-08のユーザー指示により、理由があってどの経路でも既定採用されなかった最適化を整理・削除し、
データ不足で採用が進んでいない候補は、確認できる範囲で条件付き既定採用する。
削除するものは、何を試し、どの条件で何がうまくいかなかったかを履歴から追えるようにする。
恒久方針は[main-plan](../../../../main-plan.md#最適化の共通化と既定採用の方針)を正本とする。

| Phase | 内容 | 変更 |
| --- | --- | --- |
| 82 | 本計画: 不採用最適化の削除・条件付き既定採用 | 新設・完了 |
| 83 | static FP8 KV／MTP／文章生成の実用closeout | 旧82を内容保持で繰下げ |
| 84 | 他精度の単一要求最適化 | 旧83を内容保持で繰下げ |
| 85 | NVFP4 GPU batching | 旧84を内容保持で繰下げ |

## 対象と境界

- 既存NVFP4／FP8／MXFP系kernel候補、activation量子化、attention／GDN、projection共有、deferred completion、Graph／KV chainを棚卸しする。
- 採用検証は手元のV620 `gfx1030`／R9700 `gfx1201`、既存対応shape・encoding・モデル経路を中心とする。
  MI300X等の実機が必要な未判定候補は不足事項を記録し、機材確保を本Phase全体の完了条件にしない。
- 固定GPU sampling（temperature 1.0、top_p 0.95、読込時のモデル別top_k）を前提とし、CLI／APIの通常起動から採用経路へ到達させる。
- 新しいKV形式・確率的MTP・batchingの実装はPhase83以降へ残す。全モデル／全形式の直積試験やモデル固有の追加速度探索へ広げない。
- `FORCE_*`の存在や一つのモデルで効果がなかったことだけで削除しない。他target／shapeでの既定採用、
  現行fallback、共有template、基準演算・切戻しとしての用途を調べる。
  既定採用済み経路に必要な実装を残し、不採用候補固有のinstantiationや分岐だけを除去する。
- 過去に採用されたcontrolは未採用候補と区別する。kernel IDと過去の監査出力の対応を保ち、ID変更が必要なら旧新対応を記録する。

## 作業

### 1. 現在の採否と依存関係を確定する

[Phase79一覧](../../../../../history/2026/09/1-10/phase79-selector-inventory.md)と
[既存ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)を起点に、現行sourceを照合する。
候補ID／フラグ、target／shape／encoding、未設定時の経路、共有consumer、既存証拠を記録し、次に分類する。

| 分類 | 処置 |
| --- | --- |
| 現行採用・必要な基準／共有処理 | 維持。既定OFF候補と混同しない |
| いずれの経路でも未採用で、不採用理由が確定 | 理由と依存関係を記録して候補固有実装を削除 |
| 証拠不足だが既存範囲で採用を判断できる | 必要な追加検証を行い、確認済み範囲で既定採用 |
| 今回判断できない | 不足データ・影響範囲・再検討に必要な条件を記録して保留 |

棚卸しで判明した不採用理由と単なるデータ不足を分ける。保留は未確認を成功と扱わず、候補ごとの未解決事項として残す。
Qwen3.5-4B／V620の既存deferredは既定ON、Gemma Denseの適合projection共有も既定ONである。
Gemmaでdeferredの追加利益がなかったことを、Qwenで利用する共通completion実装の削除理由にはしない。

### 2. 不採用候補を記録して削除する

初期候補はNVFP4 ID80 DP4A-K128、ID81 gfx1201 WMMA 128x32、MXFP系の棄却済みtile等。
これは削除済み・最終判定済みの宣言ではなく、sourceと全利用経路を確認するための入口である。
削除前に[Phase82履歴](../../../../../history/2026/09/1-10/phase82-optimization-cleanup-default-adoption.md)へ
次の情報を候補単位でまとめ、削除commitができた時点で対応を追記する。

- 候補ID、名称、環境変数、元の実装pathと削除前commit。
- 試した変更と狙い。比較対象のkernel／経路。
- GPU、モデル／演算shape、重み・KV形式、入力／出力長、warmup・反復条件など、当該比較に必要な条件。
- 数値正しさと性能を分けた結果。測定値、ばらつき、既存証拠へのリンク、未測定事項。
- 不採用理由。観測事実と原因推定を分け、特定shapeの退行を全shapeの証明へ広げない。
- 他の採用経路から参照されないことの確認、削除する範囲、残す共有処理／基準経路。
- 削除commit、実施した確認、将来再検討するなら何が変わる必要があるか。

元のarchive・測定履歴は保持する。過去の結果を削除後のbinaryの証拠へ読み替えない。
kernel本体とselector、symbol／workspace／grid／launch分岐、Graph／pack whitelist、候補専用test・probe・runnerを
対応させて整理する。一般の数値oracleや採用経路の回帰検査を候補と一緒に削除しない。
新しいraw trace・binary・モデルをGitへ追加せず、追跡済みの要約とGit履歴で試行内容を参照可能にする。

### 3. 有望候補を条件付きで既定採用する

既存証拠の不足が小さい候補から着手し、順序は測定費用と共通適用範囲に応じて調整する。

| 候補群 | 採用判断の要点 |
| --- | --- |
| Qwen projection共有、deferred、Graph spans、KV chain | Phase79／81の証拠を再利用し、未設定時の実選択を確認。Graphの短文費用、request所有権、取消・cleanup、FP16専用chainの制限を維持 |
| NVFP4量子化wave8、R9700 wave4、NVFP4／FP8 LUT | target・shapeごとの数値、境界、通常経路との追加性能比較。既定済みの別targetを変更しない |
| GQA6 split／prefill、GDN | KV長・head構成・target・encoding・workspaceと候補優先順位を確認し、利益のある範囲へ限定 |
| その他の残存候補 | 既存結果から少ない追加検証で採否を決められるものを扱い、未判定範囲は理由付き保留 |

GPU能力・演算・shape・encodingによる採用条件を優先する。成果物検証は維持し、モデル固定条件の除去に
新規kernelや未対応形式の実装が必要なら、その共通化は本Phaseへ拡大しない。
フラグを一律にONへ変えるだけで完了にせず、採用範囲外の既存経路、明示的な切戻し、選択理由の観測を保つ。
数値差の評価が残るstaging等は、速度データ不足だけの候補と分ける。

### 4. 通常起動・検査・文書を同期する

- 変更したselectorの未設定／明示ON／OFF／対象外条件をfocused host testで確認する。
- 影響するtargetのHIP compile、kernelの非整列・境界oracle、数値・性能比較を既存の検証入口で行う。
  既存証拠はsource・build入力等との対応が取れる範囲で再利用する。複数候補を同時に使った性能値を個別寄与率にしない。
- 最終候補では不要なopt-inを付けず、対応CLI／APIの固定samplingと代表的な短文／長文で実dispatch、
  TTFT／TPOT、prefill／decode、VRAM、取消・回復・cleanupを変更に応じて確認する。GPU失敗をCPUで代替して成功に数えない。
- 削除が確定した候補の廃止フラグを起動例・管理下の設定から除き、保持するforce／rollbackは残す。
  既定ONと運用上の明示ONを区別する。
  常駐serviceの再配置は計画更新と区別し、実施する場合は対象binaryと起動設定を揃える。
- 現行selector一覧、runtime／API説明、CI source inventory・契約・候補参照を同期する。
  採用経路の検査を弱めず、削除した候補への期待だけを整理する。

## 完了条件

- 対象候補の採用・削除・保留と理由が一覧から分かり、確認済みの不要候補が現行経路へ残っていない。
- 削除した候補の試行内容・結果・不採用理由・元のsource・削除commitを履歴から追える。
- 採用候補は確認済み条件で明示opt-inなしに選択され、通常CLI／APIの固定samplingと共存する。
- 影響する数値・性能・所有権・host／HIP検査を確認し、退行や未確認範囲を隠さず記録している。
- main-planと履歴を更新し、本計画をarchiveへ移して相互リンクを同期する。
  main-planの既存手順どおりcommit・pushし、最終公開HEADの対象CIを確認、必要な修正・再pushまで行う。

全候補の既定採用や全モデル／全KV形式への拡大は完了条件にしない。
数値・性能上不採用と分かったものは記録して削除し、機材・データ不足等で判断できないものは理由を残す。

## 完了結果

24 matmul候補と棄却済みattention専用経路を削除し、試行・不採用理由・削除commitを履歴へ記録した。
NVFP4量子化wave8、NVFP4／FP8 decode LUT、gfx1201 NV67、gfx1030 GDN row32、
既存Qwen3.8 projection／deferred／Graph／FP16 chainを確認済み範囲で既定化した。
数値判断が残るID62／64／72、P64／P128、rocBLAS F32は保留し、手動高速presetとの性能差を記録した。
両local targetのbuild／数値oracle、Qwen短文とV620長文、Gemma両target、R9700 API、host／CI契約を確認した。
詳細は[最終履歴](../../../../../history/2026/09/1-10/phase82-optimization-cleanup-default-adoption.md)と
[採用範囲](../../../../../history/2026/09/1-10/phase82-default-adoption-scope.md)を参照する。

## 工数と再計画

現対応範囲で24〜48時間（8時間／日換算3〜6日）の概算。分類3〜6、削除6〜12、
追加測定・採用12〜24、最終整合・文書・公開CI3〜6時間を見込む。確約や新しい時間gateではない。
全候補の再測定や新形式への拡大では1〜2週間以上になり得るため、既存の停止・再計画方針に従って範囲を見直す。

[メイン計画](../../../../main-plan.md) · [ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md) ·
[履歴・削除理由の記録](../../../../../history/2026/09/1-10/phase82-optimization-cleanup-default-adoption.md)
