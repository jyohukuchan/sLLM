# Phase 5: エンジン性能baseline履歴

## 2026-08-11: 計画作成

- Phase 5として、OpenAI-compatible APIより前にdirect engineの性能基準を取得する計画を作成した。
- model-resident lifecycle再利用、request-local state再作成、TTFT/TPOT/token-s/E2E/peak VRAMの定義、
  255/256/257境界、warmup/sample/summary方針を固定した。
- llama.cpp比較は固定commit`f5919bf458ef190468b5c329bb293f8a54a1e69c`と同一4B source revisionの
  BF16 artifactを使い、変換identityとmetric差を明示する方針とした。
- instrumentation実装と性能測定はまだ開始していない。

## 2026-08-12: P1 timing・memory instrumentation完了

- `sllm benchmark --lane direct`を追加し、modelをprocess内で1回だけloadして、correctness control 1回、
  warmup 3回、measured request 10回を同じresident modelへ順番に実行する経路を確定した。各requestでは
  request-local stateだけを作成・破棄し、timing有効時と無効時のtoken、stop、dispatch signatureを
  exact比較する。
- model load、request start、prefill submit/complete、first token、後続token、stop、cleanupを
  monotonic eventとして記録し、TTFT、prefill、request単位median TPOT、decode token/s、E2Eを導出する。
  allocationは`model_resident`、`request_state`、`workspace`へ分類し、current/high-water bytes、resident
  VRAM、peak VRAMをruntime allocatorのexact bytesで記録する。
- direct 22 row、render/tokenize 2 row、llama.cpp wrapper 14 rowの固定matrix、schema、bounded runner、
  aggregate JSON/CSV、digest sidecarを追加した。aggregateはrow完全性、source/build/model/cache/GPU identity、
  HIP-only、fallbackなし、sample数、timeout、health、process ownership、cleanup、raw digestをfail-closedに
  再検証する。
- runnerはexact BDF/UUID/target、ROCm loader path、実行前後health、1秒cadenceのAMD-SMI metric、
  throttling/ECC、GPU process、process group cleanupを記録する。長期稼働中のGIMPS PID `1325127`は、
  GIMPSが実計算中のV620以外のGPUをPhase 5に使用できるというユーザー指示に基づき、対象GPU上で
  VRAM 32,000 bytes、GTT 2,088,000 bytes、GFX/SDMA activity 0のinert contextである場合だけ
  明示allowlistへ入れる。
  identityまたはresource上限が変わった場合は成功扱いにしない。
- Phase 5 focused host contractは113/113を`PASS`した。最終host suite reportはH0 454/454、
  H1 328/328 selected、H2 35/35 selectedがfailed/skipped 0で`PASS`し、各report digest sidecarも一致した。
- GPU benchmark buildの共通semantic treeは`f1fd321a0a051137d548cae04fd4f8b660bcd39f`である。
  `gfx1030` binary SHA-256は
  `7870da34dd87aafe2a7c035d611ec7dc9dfc6eac42f2750a28cf9b011c86df4d`、`gfx1201`は
  `9698e152e29098643d7d6988a5b51b65d1885dde74ff98473827dc172afead60`とした。
- P2実GPUbaselineは同日に開始した。測定完了前の途中値はbaselineとして履歴へ固定せず、全22 rowと
  aggregateが`PASS`した後にP2結果を追記する。
- R9700最小rowの初回runは、AMD-SMIのlegacy aggregate `throttle_status`が負荷中に`THROTTLED`となりfail-closedに停止した。終了後VRAMはpre/postとも257 MBで、process cleanup、ECC、温度、powerは正常だった。無負荷10回でも8〜16 W、hotspot 42〜43℃で`THROTTLED` 6回と`UNTHROTTLED` 4回が混在し、violation accumulatorは全て`N/A`、CLIもreason表示をMI300以降に限定していた。そこでaggregate bit単独をhard gateから外し、公開されたactive violation、ECC、slowdown温度、socket power limit、profile/limit/performance-level driftをfail-closedにする契約へ修正した。修正後のfocused 113件と同じR9700最小rowはPASSし、負荷中最大133 W、hotspot最大46℃、終了後257 MB、cleanup/loader/process ownership全項目PASSだった。
- R9700 boundary-255はbenchmark自体がexit 0、cleanup/ECC/終了後VRAM 257 MB、最大観測218 W・hotspot 91℃だったが、約20分の1秒監視中にsocket powerが1回だけ`N/A`となりFAILした。dynamic sensorの単発欠落だけを100 ms間隔・最大3回で再取得し、連続欠落はFAIL、identity/process/ECC/明示violationはretryしないbounded contractを追加した。失敗rowはquarantineへ保持し、同じrowだけを再取得する。
- R9700 boundary-256はbenchmark自体がexit 0、cleanup/ECC/終了後VRAM 257 MBだったが、socket powerが設定上限300 Wちょうどへ達してFAILした。default profileで定格へ到達すること自体は上限超過やhealth defectではないため、温度はslowdown値到達でFAILを維持し、powerはlimit超過だけをFAIL、limitちょうどはbaseline観測値として保持する契約へ修正した。
- R9700 boundary-257はbenchmark自体がexit 0、cleanup/ECC/終了後VRAM 257 MBだったが、GIMPSのallowlist済みinert contextがpreでは存在しpostでは消えていたためauthorization driftとしてFAILした。外部processは各観測時点で存在するrecordをPID、VRAM/GTT、GFX/SDMA等のinert contractへ厳密に検証し、contextの存在有無そのものは干渉としない契約へ修正した。同種のhealth契約調整が3回続いたため、追加full rowの前にfocused host contractを再実行してから再開する。

## 2026-08-13: P2 sLLM canonical baseline完了

- semantic source tree `f1fd321a0a051137d548cae04fd4f8b660bcd39f`からtarget別binaryを作成した。
  `gfx1030` binary SHA-256は`7870da34dd87aafe2a7c035d611ec7dc9dfc6eac42f2750a28cf9b011c86df4d`、
  `gfx1201`は`9698e152e29098643d7d6988a5b51b65d1885dde74ff98473827dc172afead60`である。
- direct matrixは22/22 row、220/220 measured sampleがPASSした。aggregate summary SHA-256は
  `2fdf7b2fec8a50a0322b28d6be04effd40d68c203bceda7ed5438249fa490b7f`、graph CSVは
  `32dd7643e8cdcff20afb400a957b21b2367e4ee0c546af74b0e5e279f2a2d30c`、completed-bundle markerは
  `89a9d382cdfbae878cf0313e112213236925a93c161c4c1682ce04b89c9e4527`である。
- render/tokenize laneは2/2 rowがPASSした。aggregate summary SHA-256は
  `adb2cd63bf57af78f4d84a0bce64e6121eef1a041085512202df847cf672c553`、graph CSVは
  `d0e60d555eb343e494bd37ff2e75e8ebfe1107662ca2922e438094dd4f6bf42f`、completed-bundle markerは
  `573bcd21470ccfef68205df85305cfc48422c4c0346b7ab4252bd16c468d89ca`である。

Qwen3.5-4B direct laneのmedianは次の通り。VRAMはruntime allocatorのpeakであり、GiBへ換算した。

| GPU | case | TTFT (s) | E2E (s) | prefill tok/s | decode tok/s | peak VRAM (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V620 | minimum | 1.173 | 1.177 | 0.856 | N/A | 7.89 |
| V620 | short-odd | 7.550 | 25.838 | 2.253 | 0.876 | 7.95 |
| V620 | boundary-255 | 96.691 | 176.064 | 2.638 | 0.787 | 8.96 |
| V620 | boundary-256 | 97.513 | 177.873 | 2.626 | 0.784 | 8.96 |
| V620 | boundary-257 | 97.859 | 177.241 | 2.627 | 0.783 | 8.97 |
| V620 | prefill-long | 385.995 | 603.950 | 2.653 | 0.582 | 12.20 |
| V620 | decode-long | 14.841 | 322.461 | 2.157 | 0.829 | 8.02 |
| R9700 | minimum | 0.590 | 0.595 | 1.708 | N/A | 7.89 |
| R9700 | short-odd | 2.878 | 12.445 | 5.921 | 1.674 | 7.95 |
| R9700 | boundary-255 | 36.115 | 83.754 | 7.064 | 1.323 | 8.96 |
| R9700 | boundary-256 | 36.176 | 83.871 | 7.080 | 1.321 | 8.96 |
| R9700 | boundary-257 | 36.249 | 84.021 | 7.093 | 1.320 | 8.97 |
| R9700 | prefill-long | 98.320 | 251.148 | 10.417 | 0.831 | 12.20 |
| R9700 | decode-long | 5.766 | 178.608 | 5.557 | 1.475 | 8.02 |

size scalingの代表値では、2B short-odd E2EがV620 34.860秒、R9700 9.505秒、2B
boundary-257がV620 190.399秒、R9700 56.723秒だった。9B minimumはV620 3.058秒、R9700
0.694秒、9B short-oddはV620 79.777秒、R9700 19.149秒だった。2Bが4Bより常に速いという単純な
model-size比例にはなっておらず、V620 short-oddの逆転は最適化時の調査候補とする。

255/256/257のTTFT、E2E、VRAMは両GPUとも滑らかで、256境界そのものに大きなcliffは見つからなかった。
一方、1024-token prefillはV620で約604秒、R9700で約251秒、32/256 decodeはV620で約322秒、
R9700で約179秒を1 requestに要し、full matrix反復時間の支配項になった。

## 2026-08-13: P3 dedicated wrapper比較完了

- Qwen3.5-4B固定revisionから公式変換toolで作ったBF16 GGUFは8,424,393,568 bytes、SHA-256
  `636158bd8a217374134cc2455aa40603f7579366fda0f0f5efcbf8bcba37c045`である。変換manifest SHA-256は
  `09fceef231a65ea8793b0749fd1340f9eaffd00562aeffad7beaac74d1991f21`、llama.cpp固定commitは
  `f5919bf458ef190468b5c329bb293f8a54a1e69c`、treeは`e9b6173953477054a4068884aa5fc9aeef6475e8`とした。
- 初回wrapperは`llama_batch_init`がcapacityを確保しても`n_tokens=0`で返す契約を誤解し、exact capacityと
  token countを比較してminimumで停止した。allocation後に`batch.n_tokens`を設定してからtoken/positionを
  格納するよう修正し、wrapper source SHA-256を
  `43e7db595d5cc739021af6285b41b5bcf3d26d6bd25e4af70e0bf2732248296e`へ固定した。
- wrapper binary SHA-256は`gfx1030`が
  `b5f5aad53543ad3ecc92d4c19cfe000b7c43eb751b1d5f3ada4896cd4093d865`、`gfx1201`が
  `460cbbaae9577d4c9dcf85254af8aa0402b1bb1ae94d533b285c5cb14738dfd5`である。14/14 row、140/140
  measured requestがPASSし、sLLM aggregateとの比較bundle SHA-256は
  `53845c6501e78357b9b75ddcd8f960b2499cdbd64c94a2799afb95043799dccf`となった。

exact token、batch 1、3 warmup、10 measuredを一致させた比較可能指標では、sLLM TTFTはllama.cpp wrapperより
V620で49.4〜278.5倍、R9700で31.4〜742.1倍長かった。差は短いminimumより255〜1024 token prefillで
拡大し、とくにR9700のabsolute性能向上をsLLMが十分に利用できていない。TPOT、decode token/s、E2Eは
realized output identity/countとcleanup意味をcross-engineで同一証明していないため、context-onlyとして
保存し、比率へ変換していない。

公式`llama-bench`のprompt-processing、decode、paired 7 caseも各GPU 9/9 commandでPASSした。これらは
random/zero-initialized tokenと1 warmupを使うためratio比較には混ぜず、V620 context SHA-256
`b8800ab5ed92aa083accdddd78e626e83562a2c8bbeb07b05d50098aed85c2c1`、R9700 context SHA-256
`ecd52c69322ad68503d5146793eef453e0c3e9c350299b7c53d50fb74d44adc4`の補助evidenceとした。
`-pg`は指定pairに既定512/128 testを加算するため、`-p 0 -n 0`を明示して意図しない測定を除いた。

health evidenceの取得中に二つの過剰gateも検出した。ROCm componentは最初のGPU operation後に固定root内で
遅延mapされ得るため、loader path集合の完全一致を要求せず、各distinct集合のroot、canonical path、library
content digestを独立検証する。R9700では300 W cap、330 W公開maximum、default profileを変更していない
prefill中に最大362 Wが観測されたが、温度、ECC、明示violation、profile/limitは正常だった。socket powerは
全値を監査用に保持するが値単独をhard gateにせず、slowdown温度、ECC、明示violation、設定driftを停止条件とする。

## 2026-08-13: P4 optimization計測lane確定

最適化1件ごとにfull baselineを再実行しない。次の段階laneを採用する。

| lane | 実行対象 | 用途 |
| --- | --- | --- |
| O0 smoke | 変更対象GPU、4B short-odd、warmup 1 + measured 3 | correctness、fallback、cleanup、方向性だけを数分で確認 |
| O1 iteration | O0 + 32/32の短縮prefill/decode case、warmup 1 + measured 3 | 通常の最適化差分。target 10〜25分 |
| O1-boundary | 255/256/257のうち変更境界と両隣、warmup 1 + measured 3 | tiling、dispatch、KV/page境界へ影響する変更だけ |
| O2 integration | 変更対象GPUの4B canonical 7 case、3 + 10 | 複数最適化の統合またはbaseline更新 |
| O3 release/nightly | dual-GPU 4B full、必要時だけ2B/9B、llama比較、render/service | release candidate、定期回帰、architecture/model意味変更 |

現baselineで毎回実行する明確な価値が低いものは、minimum 1/1、255/256/257の無条件3点セット、2B/9B
size scaling、render/tokenize、公式llama.cpp比較、prefill-long/decode-longの10 sampleである。minimumはstartup
correctnessには有効だが最適化差の検出力が低い。境界3点は今回cliffがなく、境界に関係する変更だけでよい。
2B/9Bとllama.cppはarchitecture/model semanticsまたはbaseline identityが変わった時だけ再実行する。
long caseはO1では短縮surrogateを使い、canonical 10 sampleはO2以降へ限定する。

最適化backlogの優先順位は、(1) 全model/GPU共通のprefill GEMM・operator dispatch・同期削減、(2) 全model共通の
decode step再利用・fusion・kernel launch削減、(3) RDNA4で拡大したprefill差のtarget tuning、(4) V620の2B/4B
size-scaling逆転、(5) 境界固有cliffが再現した場合だけ境界専用kernel、とする。Phase 6ではdirect laneを再実行せず、
同じrequest identityにservice laneだけを追加してHTTP/JSON/SSE overheadを差分化する。

[対応する計画](../../../../plans/archive/2026/08/11-20/engine-performance-baseline.md)
