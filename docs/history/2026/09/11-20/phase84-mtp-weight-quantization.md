# Phase84: MTP重み量子化の実装・測定

状態: 実装・ローカル検証完了（公開後CIは当該commitのchecksで確認する）。MXFP8の速度退行によりBF16既定を維持する。MXFP6の包括的な採用率・性能比較は移行条件未成立のため保留し、接続済み経路の限定的な正しさ検査を行う。

## 初期実装

BF16 companion専用8行列をMXFP8 E4M3 W8A8／MXFP6 E3M2 W6A6へ変換するsidecarを追加した。元artifactとsource hash、形状、value/scale範囲、recipeを検証し、元のweight planを参照情報として保持したまま実際のresident viewだけを置き換える。norm 7 tensor、共有embedding/head、target、KVは維持する。

通常generate/chat/serverの`--mtp-weights`からsidecarディレクトリを指定する。未指定はBF16。requestごとに量子化draft graphを再構築するときも同じsidecarを使用し、recipe digestをresidentと照合する。変換入口は`sllm-convert-qwen38-mtp --artifact-root ABS --encoding mxfp8|mxfp6 --output-dir ABS`。実際のencoding/digestを監査へ記録する。

実MXFP8成果物のhost graph検査も1件成功し、元plan identityの維持、共有tensor viewの一致、8行列だけの置換とresident byte数の減少を確認した。

## MXFP8初期候補 r1

共通化後BF16 companionを比較基準とする。8192 input／128 output、seed123、T1/P.95/K20、KV MXFP8 E4、MTP幅2、chunk2048/state8320、1 warmup＋3 measured。以下はmeasured中央値で、warmupは含めない。

| GPU | BF16 prefill | MXFP8 prefill | BF16 decode | MXFP8 decode | MXFP8 accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| V620 gfx1030 | 216.593 | 223.458 | 25.423 | 20.905 | 70/115 |
| R9700 gfx1201 | 541.969 | 542.036 | 35.110 | 32.312 | 77/102 |

速度単位はtok/s。各GPU内の3 measuredで採用数は一致した。V620のdecodeは約17.8%、R9700は約8.0%低下したため、この時点では既定採用しない。prefix短縮だけでdecodeの低下を隠さず、採用率と言語/タスク差、既存MXFP8 M=1 kernelの費用を調べる。

両GPUのmodel-free数値検査は13ケース・各2反復で成功した。MTP M=1形状、prefix境界、非整列N、scale境界を含み、独立oracle、non-finite分類、反復digest、実dispatch、fallbackなしを確認した。大prefixは境界点の数値oracleであり、全出力の数値一致とは説明しない。M=1の既定ID18はV620でfusion約2.19ms、gate/up片側約3.42msだった。これはoperator単体の値で、draft全体のwall時間ではない。後続候補ではprefix primingとdecode proposalのwall時間を別途計測する。

初期候補の両GPU buildではsource before/afterが一致し、実モデルのtarget/draftはHIPのみ、unload後のallocationとquarantineはゼロだった。R9700の初回測定後は既存serviceのunit/binary/configを変更せず復元し、health/readyとも200を確認した。

## 言語・タスク採用率 r1

英語/日本語/中国語×coding/reasoning/summary/creativeの12入力、seed123/456/789、最大128出力を同じGPUのBF16 companionと比較した。

| GPU | BF16 accepted/proposed | MXFP8 accepted/proposed | 率の変化 |
| --- | ---: | ---: | ---: |
| V620 | 2617/3930（66.59%） | 2619/3935（66.56%） | -0.034 percentage points |
| R9700 | 2628/3919（67.06%） | 2639/3899（67.68%） | +0.626 percentage points |

両GPUのBF16/MXFP8全144件を点検し、新たな反復崩壊・明白な言語/タスク逸脱は確認しなかった。128token上限での未完了コード等はBF16側にもあるため、完全なタスク品質比較とはしない。

全体では採用率がほぼ維持された。個別条件では両方向の差があり、この入力集合を超えた同等性や言語差の原因は認定しない。V620のsummary-en/seed789だけ125tokenで正常stopし、他の測定は128tokenに達した。短いAPI要求のwall時間は8192速度行とは分ける。

既存ID18のM=1 GEMVを調べ、FP32累積と既存E4M3/E8M0復号を保ちながら既存GEMVの複数列処理を再利用する候補を1本だけ比較する。これは観測したdecode低下への限定的な共通provider修正であり、Phase85の全shape最適化を完了条件へ取り込まない。不利なら候補を撤去し、初期候補の結果と理由を残す。

詳細: [数値・digest・計測条件](phase84-mtp-weight-quantization.json)。生成例や採用率だけでBF16 full-modelとの品質同等性は認定しない。MXFP6はこの段階で未測定であり、不採用と判断したものではない。


## 共通M=1候補の検査修正

r2は追加したrange境界fixtureの識別値が既存入力と衝突し、weight側の生成にも適用されていなかったため、両GPUで検査が停止した。これは検査データ生成の不具合であり、候補のGPU数値PASSや速度結果は得られていない。識別値を分離し、activationとweightの実際の生成入口を通すhost testへ修正した。FP16最大有限値65504を超える値が残ることも明示検査する。

あわせて候補の4byte読出しをbyte単位の安全な合成へ変更した。公開weight viewのoffsetは4byte整列を保証しないため、整列済み整数pointerへのcastには依存しない。修正後は別build identityのr3として比較する。

## 共通M=1候補 r3の測定

修正後のID95は両GPUで15ケース・各2反復の数値検査に成功した。候補dispatchは各18回、fallbackなし。8192/128の3 measured中央値はV620 prefill/decodeが222.396/22.644 tok/s、R9700が541.638/30.860 tok/sだった。accepted/proposedは各GPUの全measuredでそれぞれ70/116、71/111、decode proposal wall中央値は約781.06ms、553.01ms。draft以外を含むdecode全体とは分ける。

V620では初期MXFP8より改善したが、両GPUともBF16基準には届かない。R9700では初期MXFP8より低下しており、単体kernel時間だけで既定採用しない。生成digestと採用数もr1から変化したため、r1の言語suiteをこの候補の品質証拠へそのまま流用しない。ID95の候補kernel・selector・強制環境変数は撤去した。既存providerだけを使用する。MXFP6の包括的な採用率・性能比較は未実施である。R9700 serviceは測定後に元のunit/binary/configで復元し、health/ready 200を確認した。

## 既存providerへ戻した最終比較 r4

同じbuildからBF16／MXFP8を1 warmup＋3 measuredで比較した。

| GPU | BF16 prefill/decode (tok/s) | MXFP8 prefill/decode (tok/s) | BF16→MXFP8 E2E (s) | BF16→MXFP8 proposal wall (ms) |
| --- | ---: | ---: | ---: | ---: |
| V620 | 215.229 / 25.063 | 220.666 / 20.743 | 43.152 → 43.276 | 673.71 → 1208.61 |
| R9700 | 528.521 / 35.054 | 539.785 / 31.639 | 19.153 → 19.258 | 503.17 → 747.47 |

decodeは約17.2%／9.7%低下した。V620ではprefix primingが1582.11→454.27msに短縮し、8192入力のE2E差は小さいが、decodeの退行は残る。R9700のprefix primingは118.57→166.63ms。採用数はBF16が74/106・78/100、MXFP8が70/115・77/102で、各GPUの全measuredで一致した。

BF16測定の一部にはMSRV検査が重なり、V620 MXFP8測定の一部では後続buildが短時間重なったため、buildを停止して測定終了後に再開した。全sampleを保持し、V620 MXFP8 decodeのMADは0.120 tok/s。host負荷を完全に隔離した微差比較とはしない。初期r1でも同規模のdecode退行があり、僅差を根拠に採否を決めたものではない。

両GPU・両形式の既存provider数値検査は各15ケース・2反復で成功した。MXFP8/MXFP6の通常APIはtext、SSE、MTP進行後cancel、recovery、8192/128、要求再利用、unloadをすべて成功した。実HIP・非zero dispatch・fallbackなし・最終allocation/cleanup/quarantineゼロ・観測VRAM 32GiB未満を確認した。これはMXFP6の包括的な採用率・速度・品質評価ではない。

r1とr4はsidecar、core graph/resident、native演算sourceが一致する。通常設定で有効になる差分は主にwall時間の記録で、無効MTPとsidecarの併用拒否も追加した。8192/128の生成digestと採用数はr1とr4で一致した。言語suiteはr1の実測結果として保持し、新しいbuildで全suiteを再実施したとは扱わない。

最終r5はr4からRust 1.85対応の同値なOption条件記述とoracleの整形だけを変更した。両targetでsource before/after一致のbuildに成功した。量子化・graph/layout・native kernel・成果物は変わらず、r4 GPU数値・速度とr5の構文修正後確認を区別して記録する。

## 採否と評価範囲

MXFP8は明示選択の比較・開発経路として残し、両GPUの既定はBF16 companionを維持する。全体採用率の大幅な劣化は観測していないが、decode速度の低下が残るためである。初期ID18と追加ID95を試した結果を残し、不採用の追加kernelはcodeから削除した。既存MXFP8/MXFP6 codec/providerを再利用し、新たな外部kernel移植は行っていない。

MXFP6のconverter・sidecar・loader・graph接続と実artifactのhost検査は実装済みである。MXFP8の移行条件を満たさないため、MXFP6の12条件×3 seedの採用率suiteと3反復の速度比較は行わず、MXFP6が不利だったとは判断しない。公開される選択経路の数値・API・資源解放は限定smokeで確認し、包括的品質や速度の承認とは分ける。包括的比較はPhase85でproviderの残件を扱う際に再検討する。

CI事前検査で新converter binのtarget台帳漏れを修正した。依存packageとCargo.lockは増やしていない。最低対応Rust 1.85で使えないlet-chainは同じ条件のOption合成へ置換し、MSRVを含む依存検査に成功した。この構文修正は量子化・layout・演算処理を変更しない。

最終CLI r5ではMXFP8/MXFP6のgenerateとMXFP8のchatが成功した。generateは実HIP・MTP活動・選択されたencoding/digestを確認し、chatは指定sidecarを使用するproduction backendでturn/session完了を確認した。chat JSONL自体はbackend監査全体を公開しない。先行するCLI検証scriptのgenerateへの未対応context引数は起動前に拒否されたため、正しい引数で別記録に再実行した。

ローカルCIはH0 627/627、H1 1564/1564（74 deselected）、H2 38/38（9 deselected）で成功した。GPU検証とは区別する。公開後のhost-required/H3結果は当該commitのGitHub checksを参照する。

利用手順: [MTP companion量子化](../../../../development/mtp-companion-quantization.md)。

作業計画: [Phase84](../../../../plans/archive/2026/09/1-10/phase84-mtp-weight-quantization.md)。
計画決定: [量子化順序](phase84-mtp-quantization-plan.md)。
比較基準: [共通化後再計測](../1-10/phase83-common-qwen38-remeasurement.md)。

## 公開後CIの修正

初回commit `9af13362` のGitHub H1で、既存の `client_disconnect_cancels_active_generation` が失敗した。HTTP headerを受信してもbackendの生成開始は保証されず、CIでは開始前の切断が先行し得た。この場合にbackend内のcancel観測flagを要求していたことが競合の原因だった。

テストにbackend開始の通知と切断許可の同期を加え、active generationを確認してから切断する。キャンセルのassertionと2秒の待機上限は維持し、runtimeは変更していない。該当testは10回、HTTP契約全12件も成功した。公開CIの最終結果は修正commitのchecksを正とする。production source・GPU binary・量子化成果物は変わらないため、このhost test修正を理由としたGPUの再計測は行わない。
