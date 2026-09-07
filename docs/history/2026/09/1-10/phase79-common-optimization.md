# Phase 79: 共通化・条件付き既定採用の実装記録

> 状態: 完了（2026-09-07）。確認済みscopeの条件付き採用であり、全モデル・全KV形式への一律採用ではない。

## 2026-09-07 開始と担当範囲

ユーザーgoal「Phase79を完了する」により開始した。旧Phase79〜81の作業は80〜82へ移し、
既存NVFP4/FP8候補のselector、共通prepared execution、projection activation共有を対象とする。
モデル固有の追加速度探索、新しいKV形式、MTP、batchingは今回の範囲に含めない。
既存frontend/server/KV evidenceの未コミット変更を保持して進める。

## 候補別Gemma探索

既存Phase78 CLIを同一binaryで候補だけ切り替えた。最終Phase79 buildの証拠ではない。

- GPU: V620 gfx1030、UUID `GPU-76a08c022586fed6`、単独可視device 0。
- CLI SHA256: `497526210a20b97f8399485180f2b0f12d24e15fde8930f5ca27842117d88be6`。
- model: 既存`phase20-final-gemma4-nvfp4.gguf`、NVFP4/FP8混合、FP16 KV。
- 条件: 17入力/17出力、greedy、EOS無視、各1 warmup＋3 measured、同一GPUで逐次実行。

| 候補（他の設定は固定） | decode中央値 tok/s | min〜max |
| --- | --- | --- |
| 既定 | 1.6275 | 1.5991〜1.6423 |
| NVFP4 DP4A wave4 | 1.8882 | 1.8842〜1.9002 |
| NVFP4 DP4A columns | 1.7109 | 1.7046〜1.7160 |
| FP8 half2 | 6.0617 | 5.9772〜6.0830 |
| FP8 dword8 | 6.8076 | 6.7333〜6.8448 |

全runがHIP-only、fallbackなし、正常終了。全候補の生成列は既定と異なるため、
独立演算oracleと固定入力logitsによる数値分類を進める。最初の生成差は0始まりで
NVFP4 wave4が10、columnsが8、FP8 half2/dword8が1。生成列差だけを品質劣化とも品質同等とも判断しない。
この測定はdecode候補の個別比較であり、過去のDP4A prefill倍率とは分離する。

raw、argv、PID/終了状態、binary/raw SHA256は`.local-artifacts/phase79-model/`の
`explore.py`、`explore-*.execution.json`、`exploratory-summary.json`へ保存した。
最終結果とsource/build identityは本記録末尾のリンクを参照。

operator harnessの数値・性能結果は
[operator evidence](phase79-operator-evidence.json)へ記録した。5つのoperator runは
production ABIで独立encoded oracle、repeat、比較、境界shape、性能を確認したが、
ハーネス自体はallocatorの最終cleanup量を報告しないため、各runの `cleanup` は
`not_reported` としている。したがってoperatorの `PASS` は数値・比較・性能の結果を示し、
cleanup保証を含まない。固定logits collectorと最終model evidenceのcleanup0は別の証拠として
扱う。

## 固定入力logitsの採取・比較

`crates/sllm-hip/src/bin/sllm-phase79-gemma-logits.rs`は検証済みGGUFを読み、
同じpromptとteacher-forced continuationでlogitsを採取する。GPU dispatch、finite、cleanupを確認するが、
出力状態は`CAPTURED`とし、品質PASSを主張しない。
`ci/tools/phase79_compare_logits.py`はmodel/target/KV/fixture一致と全比較位置の存在を検査し、
KLD、top1、最大絶対誤差、RMSE/NMSEを出力する。数値分類・品質採否は自動付与しない。
解析解を持つKL、大きなlogit offset、欠落位置、fallback、cleanup異常のhostチェック3件がPASS。
collectorのhost cargo checkと両targetのdraft1 buildもPASS。GPU captureの結果は次節に記録する。

## 固定入力logitsの初回結果

同じdraft1 binary内で、3入力/17入力/65入力と各3 teacher-forced decodeの12位置を比較した。
これは既存selector候補の比較であり、編集中の共通化全体の最終証拠ではない。
初回にbase用lockを指定したcaptureはsemantic identity検証で拒否され、正しいIT用lockに修正した。
拒否runは`wrong-lock`として保持し、GPU PASSへ含めない。

| GPU | 比較（参照→候補） | top1一致 | 最大KLD | 最大logit絶対差 |
| --- | --- | --- | --- | --- |
| gfx1030 | 旧NVFP4→既定 | 11/12 | 2.144631 | 8.718750 |
| gfx1030 | 既定→NV wave4 | 12/12 | 0.347253 | 6.609375 |
| gfx1030 | 既定→FP8 dword8 | 8/12 | 5.834113 | 23.687500 |
| gfx1201 | 旧NVFP4→既定 | 7/12 | 2.852354 | 15.019531 |
| gfx1201 | 既定→NV wave4 | 12/12 | 0.000000 | 0.000000 |

全captureでfinite、実HIP dispatch、fallbackなし、cleanup0を確認した。
差の大小だけでN1/N2を分類せず、演算順と独立oracleで分類した（最終採否は下記）。
特に旧NVFP4→既定のprefill差はM63/64/65境界で演算単体の切り分けを行う。
既存R9700サービスは接続なしを確認後SIGINTで正常終了し、終了監査の全current bytes/cleanupが0。
R9700 capture後に同じunit/binary/configを再開し、`/healthz`がreadyであることを確認した。

## prefill数値差の切り分け（修正前）

M63/64/65 × (K48,N37)/(K3840,N15360) の本番ID11/59比較は、合成encoded入力で
全BF16出力一致、独立host oracle、repeat再現性を確認した。実モデルlogits差の否定には使わない。
静的にはID59のlane逐次和がID11より長く、既存ID59を誤差非増加のN1として採用する根拠はない。
重み共有を保ちつつID11と同じK分割・加算順へ戻す8 accumulator修正を実装した。
受入条件は同じencoded入力での基準演算一致、境界・repeat・finite・cleanupの維持とし、
既存の数値基準を緩和しない。これは共通経路の数値差の解消であり、モデル固有速度探索ではない。

修正版はgfx1030 production ABIの同じ6境界ケースで全BF16出力・repeat・独立oracleがPASS。
Gemmaの3/17/65入力と各3 teacher-forced decode、計12位置でID11と全logits一致
（top1 12/12、KLD 0、maxabs 0）。decode候補は両armとも明示無効にしてprefill差を切り分けた。
この比較はdraft2同一binary内であり、projection共有の最終採否とは分離する。
旧ID59との比較では加算依存深さ低減のN1、ID11との比較では有限encoded入力で加算順復元のN0。
NaN payload互換やモデル品質一般の証明は主張しない。

## decode採用範囲の境界（draft2）

gfx1030 NV ID67はM1、既存support内のK>=1024/N>=1024、FP8 ID68はK>=128/N>=64を候補範囲とした。
NV K1008/1024/1040、N1023/1024/1025、K17408、FP8 K64/128/192、N63/64/65の
本番ABI比較で独立encoded oracle、全BF16出力、repeat一致を確認した。
NV ID58参照はwave4 flagを0へ固定し、新既定が参照armへ混入しないようにした。
範囲外は既存経路、明示flag0は切戻し、未対応shapeの強制選択は既存対応経路を維持する。
selector host検証と以下の最終モデル確認を完了した。

## 統合候補の共通化確認とreview

両targetのGemma固定入力比較で、default対pack-off／deferredは全12位置の全logits一致。
各caseの3 decodeに対するpack submissionは144、prefillは0。pack-offは全区間0で、
対応別モデルへの実適用と明示切戻しを確認した。gfx1201も修正ID59とID11の全12位置が一致した。
V620-AのGemma形状native packは3 dispatch、workspace2160 bytes、通常3反復と
入力変更後の3-node Graph replay 2回が独立oracleと一致し、cleanup0。
Qwen3.8短文17/17のGraph ON/OFFは生成列hash一致、ONで128 span/1128 capture kernel nodes/
1920 replay、OFFで全Graph counter0。要求終了後request/workspace0、fallbackなし。
1 warmup＋1 measuredの機能・再利用確認であり、正式な速度倍率の証拠とはしない。

これらは最初の統合候補の記録。1回のintegration reviewで以下3件をcorrectness/ownershipとして発見し修正した。

- Gemma pack loweringでbackend/target capabilityを確認し、非対応targetをordinary matmulへ戻す。
- Gemma共通deferred条件へ実際のMTP/KV encodingを渡し、対象外を除く。
- Gemma並行requestでresident queueのcompletion modeを共有しないよう、queue所有権を分離する。

上記3点だけfocused再確認し、残存指摘なし。新しい全体reviewは追加していない。
最初の統合候補のR9700確認後は元のserviceを再開し、healthz readyを確認した。

## 最終性能と採否

V620-B、同一review済みbinary、FP16 KV、17/65入力・9出力、greedy、各1 warmup＋3 measured。
rollbackはNV ID67/FP8 ID68/projection共有を同時に無効にした比較で、個別寄与率ではない。
測定時は別GPUで固定logits採取を並行した。GPUは専有したが、共有host資源の影響は測定限界に含める。

| 入力 | 経路 | prefill tok/s | decode tok/s | TTFT ms |
| --- | --- | ---: | ---: | ---: |
| 17 | selector＋共有rollback | 10.9795 | 1.5516 | 1565.642 |
| 17 | 新既定 | 11.0032 | 15.5622 | 1562.204 |
| 17 | 新既定＋deferred opt-in | 10.9482 | 15.3575 | 1570.978 |
| 65 | selector＋共有rollback | 13.3788 | 1.5445 | 4880.302 |
| 65 | 新既定 | 13.4330 | 15.3531 | 4860.315 |
| 65 | 新既定＋deferred opt-in | 13.4486 | 15.3419 | 4857.180 |

全requestがHIP-only、fallbackなし、request dropとcleanup0。ばらつきとmemoryは構造化証拠に記録した。
ID67/68と適合Gemma packを採用する。Gemmaの追加deferred効果は小さいためopt-inを維持する。
Qwenの既存個別opt-inも維持し、共通化を理由に全候補を一律既定化しない。

小型Qwenは33入力/9出力、各1 warmup＋3 measuredでprofiled/deferredを比較した。
MXFP8重み＋FP16 KVはdecode 14.7044→16.0682 tok/s、BF16重み＋OCP MXFP8 E4 KVは
39.2862→45.0105 tok/s。各比較内で全生成列一致、dispatch数一致、fallbackなし、cleanup0。
これはcompletion方式の比較であり、旧Phase全体との改善率ではない。
MXFP8重み＋MXFP8 KVのCLI指定は既存の対応scope制限で拒否されたためGPU PASSには含めず、
対応済みBF16重みでKV検証を行った。新しいKV/weight互換scopeは追加しない。

prefill数値修正には費用がある。ID59のM65/K3840/N15360は旧探索24.584→修正29.499 ms（約20%増）。
修正時のID11 34.436 msよりは速い。誤差boundを小さくして基準logitsへ復元するN1修正として受け入れ、
この低下を隠さない。追加のモデル固有速度探索は行わない。

## 最終確認と完了audit

- selector: 適用一覧、prepare時trace、supported/enabled/adopted/reason、境界とoverrideのhost試験を確認。
- 数値: 独立operator oracle、repeat、両targetの固定logits比較、N1/N0分類とrollbackを台帳へ記録。
- 共通化: Gemmaの実pack144/無効時0/prefill0、Qwen Graph replay、FP16/MXFP8 completion、要求再利用・終了処理を確認。
- 所有権: core host 561 PASS/20 ignored、public-runtime host PASS。cancel/失敗時publication、pending owner保持・drop、context境界を既存focused契約で確認。
- review修正後の両target buildとGemma再比較がPASS。shared/通常/deferredの全12位置が一致し、R9700のID11比較も一致。
- Qwen証拠は変更のないQwen/common/native sourceへ対応付けた。review修正はGemmaとhost期待値のみであり、Qwen経路の全面再測定は追加しない。
- runtime契約、selector一覧、数値台帳、source/build/model/raw hashを記録。R9700 serviceは元のunit/configで復帰しhealthz ready。

Phase79の完了条件を満たした。Phase80はstatic FP8 KV/MTP/文章生成、81は他精度、82はbatchingを引き継ぐ。
全モデル・全KV形式での全Graph化、全候補の既定採用、一律速度倍率は完了条件へ追加していない。
実装完了時点ではcommit/pushは未実施だった。後続のユーザー指示により公開対象とし、
公開前の整形と測定時sourceとの対応はbuild identityに記録した。稼働serviceの新binary配置は行っていない。

[完了計画](../../../../plans/archive/2026/09/1-10/phase79-common-optimization.md) ·
[runtime契約](../../../../architecture/runtime.md) ·
[数値変更台帳](../../../../compatibility/numerical-output-changes.md) ·
[selector一覧](phase79-selector-inventory.md) ·
[採用範囲](phase79-adoption-scope.json) ·
[operator証拠](phase79-operator-evidence.json) ·
[model証拠](phase79-model-evidence.json) ·
[build identity](phase79-build-identity.json)
