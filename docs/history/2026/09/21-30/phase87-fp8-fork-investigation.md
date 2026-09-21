# Phase 87 FP8 fork調査と採否の再評価

2026-09-21のユーザー指示により、元の直列処理も正しいとは仮定せず調査した。
**過去N0の原因は未特定。ユーザー判断でその結果を採否から除外し、追加の数値・速度・KLD検証に基づいてFP8 M1だけを採用した。M3は直列を維持する。**

## 不一致の保存証拠

段階6の `final-gfx1030-r1/mtp-off/report.json` はwarmup 1回＋計測3回すべてが
同じ列で、baselineとoutput index 8から異なる。保存reportのhashは実行記録と一致する。
凍結した両pack並列binaryのSHA-256は
`5848594b8e4ec9e5ca19b3e911fb4f77c3c49f14344d0ec5da550d1204b0210b`。
直列baselineとのGPU fatbinは同一だが、当時のlogits・control・device weight内容は保存されていない。
実行記録の環境変数はSLLM設定が中心で、すべての外部条件を復元できる証拠ではない。

## 追加検証

主対象はV620 `GPU-76a08c022586fed6`、比較には第二V620
`GPU-08b2ddcbd6e6b36c` も使用。モデル・8192 prompt・MXFP8 E4 KV・chunk 2048・
state capacity 10240・seed 123・MTPなしを固定した。診断による同期、転送、入力介入を含むため速度評価には使わない。

| 検査 | 観測と限界 |
| --- | --- |
| 保存binary再実行 | 元のwarmup／計測条件、短い16-token条件、新規5プロセスを含め、過去の不一致を再現できていない。 |
| logits／sampler | 直列と両pack並列でprefill後・eager decode後・126 replayのBF16 logitsがbit一致。独立CPU top-k／top-p／SplitMix64も126行一致。観測用D2Hは時間を変える。 |
| 過去列のteacher forcing | 同じ8192 prefill後に過去の列を入力。eager・直列graph・並列graphの128位置のlogitsがbit一致。現在の分布では過去の選択12箇所が再現しない。単一counter shiftやseed 0..65535への置換では全列を説明できない。 |
| FP8数値参照 | M=1、K=5120、N=10240/6144の4 fixture。量子化前BF16からFP8 code／scaleはCPUとexact一致。131,072 matmul出力はFP32累積・scale・BF16丸めの誤差予算内。直列／並列差0。4 ULPの検出閾値自体を正しさの根拠にはしない。 |
| 実graph依存関係 | 1172 nodes、NV56＋FP48の104組だけがforkし、他の辺はbaselineと一致。実captureの1275 edgesを使う整数DAGは10プロセス×3構成×2同期方式×128 replayで全nodeのCPU oracle一致。実モデルのmemory accessを再現する試験ではない。 |
| メモリ初期値 | 新規hipMallocを0／0x3fで埋めた実行、MALLOC_PERTURB_=85でも126 replay logitsがbit一致。 |
| allocation境界 | 直列・並列とも1537 allocations/freesで256-byte redzone破壊0、検査API失敗0。診断器は意図的1-byte越境を検出した。allocation内の誤書込みは検査しない。 |
| H2D／D2H | pageable/pinned、fresh/reuse、非整列サイズを含む24条件、計36.0156 GiBの転送でbyte差0。さらに実モデルのH2D 894回、21,650,582,876 bytesを直後に全byte照合し差0。過去のdevice内容を復元する証拠ではない。 |
| LDS初期値 | 各kernel前に64 KiB volatile LDSをNaN bit patternで埋める補助kernelを挿入。直列・並列の生成列は一致し、並列の再実行ではprefill後＋126 replay logitsも元の直列とbit一致。node数は2342へ増えるため、スケジューリングも変わる。 |
| source寿命／同期監査 | provider 82/71、GDN／causal prefill、control/ring、upload、projection-pack raw workspaceを調査。具体的な原因は見つからなかった。workspace所有planはRust cache/captured ownersとnative graph pinで保持され、GraphExec破棄後に解放される。 |

最初の分岐位置の現在の乱数は `0.6198402433`、先頭候補321のCDFは `0.6193568137`。
小さいlogit差でも303から321へ反転し得るが、これは差の発生原因を説明しない。
LDS試験の初回logit observerは、引数なしの補助kernelを誤って不正と判定した。
observerを修正して再実行した結果だけをLDSのlogit一致証拠に用いた。

## 未解決点

過去に何が異なったかを示す中間値がなく、現在の凍結binaryでも不一致が再発していない。
元の直列処理が誤っていた、FP8並列化が演算結果を変えた、ROCmやhardwareが原因だった、
のいずれも断定しない。当時の未記録の実行条件についてユーザーへ確認し、変更の心当たりはないとの回答を得た。
ユーザーは、再現しない過去の不一致を採否根拠から除外して採否を再評価することを許可した。
raw証拠は保持し、bit反転等を原因と断定しない。原因特定は再評価の完了条件から外した。
この原因調査段階ではproduction修正を行わず、その後の再評価で下記のM1限定候補を作成した。commit／pushは行っていない。

## 再評価で観測したHIP signal停止

NVFP4-only基準版の通常計測（warmup 1回＋計測3回）の最後のrequestがdecode step 106で停止した。
`rocgdb`では実行中kernelがなく、HSA queueに未処理packetが残っていた。
queue 5の先頭 `BARRIER_AND`（index 292712）はsignal `0x71ec5a5d7b80`の0を待つが、値は`-1`。
producerはqueue 3の直前の処理済みpacketだった。同signalの値だけを`-1→0`へ書き換えると
queueが解放され、既にtimeoutとなったrequestのcleanupが完了した。この実行はFAIL／介入済みとして速度集計から除外した。

**この停止の直接原因はnegative completion signalの0待ちである。**
HIPのlaunch間signal pool再利用が関与する可能性は強いが、過剰decrementを生んだraceの細部は未確定。
先行する再現しないN0不一致と同因とは断定しない。
[pool導入commit](https://github.com/ROCm/rocm-systems/commit/bced4e4d8dcf52c635c9e679c56aa9a5fba1761c)と
[使用pinの実装](https://github.com/ROCm/rocm-systems/blob/2b22ab0195cc1461cd9abf3b969e9dd7c10af350/projects/clr/hipamd/src/hip_graph_internal.cpp#L398)では、
producer/consumerへsignalを共有し、完了callbackが値を1へ戻してpoolへ返す。
確認できたupstream修正版はなく、独自CLR差替えも行っていない。

既存の `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0` でこのsegmented経路を外すと、
同じ基準binaryのwarmup＋計測3回が完走し、全runのtoken列が一致した。
同SDK/rootを保つprocess-local workaroundとしてgfx1030の計測に使用する。
簡略graphの100 requests×128 replayではdefaultでも再現しなかったため、これを障害不存在の証拠にはしない。

R9700はclassicでMTPなし21.0521／あり35.0821 tok/sへ約1.5〜1.9%低下した。
従来設定へ戻すと21.4663／35.6270 tok/s、以前比+0.07%／−0.01%で全run N0一致。
回避設定はV620へ限定し、R9700へ一般化しない。既存activation/benchmark runnerにtarget別defaultと
runtime controlの記録を加え、明示overrideは保持する。詳細は[software契約](../../../../compatibility/software.md)。

## FP8 M=1の採用結果

classic条件の同一process AB/BAを3process×6roundで取り直した。
M1は全18roundで改善し、1pair短縮の中央値22.54 µs、最小16.76 µsだった。
48 pairs×126 replay÷127 decode transitionsを掛けると、中央値約1.074 ms/token、
最小約0.798 ms/tokenとなり、通常TPOT約59.93 msの1%を全roundで超える。
この換算は本番形状の単体寄与であり、全モデルの実測短縮そのものではない。
M3は1roundが退行したため採用せず、M1だけを追加採用した。NVFP4 M1/M3は維持する。

MTPなしの通常ABBA（各1 warmup＋3 measured、診断library/profileなし）は次の結果だった。
A1は同じbinary/configの直前classic対照を再利用した。

| 順序 | 構成 | 計測3回の中央値 tok/s |
| --- | --- | ---: |
| A1 | NVFP4だけ並列 | 16.688149 |
| B1 | FP8 M1も並列 | 16.813443 |
| B2 | FP8 M1も並列 | 16.830720 |
| A2 | NVFP4だけ並列 | 16.685315 |

各構成6 measuredを合わせた中央値は16.686732→16.813859 tok/s、**+0.76185%**。
全16run（warmupを含む）のtoken列が一致した。遅い単発sampleもraw reportへ保持し、
事後削除していない。採否は既存原則の単体AB/BA基準と合わせて判断する。
MTPありはFP8 GDNがM3なので変更対象ではなく、同条件A/Bは30.719020→30.717491 tok/s（−0.005%）で同等だった。warmupを含む全8runのtoken列が一致した。

品質は同一固定列の128位置で、BF16 referenceに対する平均KLD **0.026570786734159246**、
p95 **0.12232757291789409**、最大 **0.2185782916732342**、top1 agreement **0.921875**。
NVFP4-onlyと最終M1候補のcaptured logitsはbit一致し、KLD差は0。
これはcoding8192の固定継続列128位置の結果であり、既存2632位置corpusの集計とは区別する。
referenceはllama.cpp BF16／FP16 KV／batch64、比較側はNVFP4＋FP8／MXFP8 KV／prefill2048である。

最終候補の実DAGは1172 nodes、NV56＋FP48の104 forkで他の辺は不変。
GPU fatbin SHA-256は基準と同じ `8f9bcb28b95ecf7d813a922f0137ea9f1728c629e886b3f0f3107e2117d863ad`。
両targetのrelease build、C++ format、Rust format/clippy、87 host tests、JSON/H3契約がPASS。
新しいsourceに合わせて個別・集約・参照のCI hash連鎖を更新した。

再評価の生成物: `.local-artifacts/phase87/fp8-reevaluation/`。
`normal-classic/off-summary.json`、`unit/summary.json`、`quality/kld-final-vs-bf16.json`、
`quality-final-capture/dag-check.json`、`r9700-default/summary.json`、
`a1/mtp-off/queues-and-signals.json` と `signal-intervention.txt` を参照する。

生成物は `.local-artifacts/phase87/fp8-cause/`。主要な集計は
`logit-comparison.json`、`fresh-process-r1/comparison.json`、
`teacher-forced/output-secondary-r1/analysis.json`、`forced-graph-r1/comparison.json`、
`forced-graph-serial-r1/comparison.json`、`guards-summary.json`、
`upload-verify-r1/summary.json`、`lds-pairs-r2/comparison.json`。

計画: [調査・採否再評価](../../../../plans/archive/2026/09/21-30/phase87-fp8-fork-cause.md)。
先行履歴: [段階6](phase87-stage6.md)。

最終source SHA-256: `7cb87884b44959f815934e6f1fbf4552988af645fb26a4157ba554a1022b6180`。
最終gfx1030 binary SHA-256: `433b1b95ce97ab11e61e73ad0e527b8be6602eea0a32c587fffb4e3bab39dd68`。
計測準備中にsection抽出用llvm-objcopyが基準binaryのELFをrewriteしたことを検出した。
未変更のCargo build artifactから元のSHA `cf271c...` へexact復旧し、採否の計測は復旧後に行った。
この操作は過去N0より後であり、その原因にはしない。記録は `fp8-cause/identity-recovery.json`。

