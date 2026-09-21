# FP8 GDN並列化の調査と採否再評価（完了）

2026-09-21のユーザー指示: 元の直列処理が誤っている可能性も含め、原因が分かるまで調べる。

## 結論

- ユーザーの追加判断により、再現しない過去N0は採否から除外した。原因は未確定のまま記録する。
- gfx1030のFP8 GDNはM1だけ採用。単体全18round改善、通常MTPなし16.686732→16.813859 tok/s（+0.76185%）。
- FP8 M3は速度が退行するroundがあるため直列を維持。MTPあり30.719020→30.717491 tok/sで同等。
- 固定128位置のBF16比mean KLDは0.026570786734159246で、NVFP4-onlyとの差0。通常計測のtoken列も一致。
- 別途観測したHIP completion signal=-1の停止はgfx1030のclassic scheduler設定で回避する。
  R9700は従来設定を維持し、21.4663／35.6270 tok/s、token列一致を確認した。
- 両target build、87 host tests、format/clippy、CI hash連鎖、DAG／KLD検証がPASS。commit／pushは行っていない。

## 当初の受入条件（下記の追加ユーザー判断で更新）

- 保存済み候補の不一致を再現し、入力・model・seed・binaryを固定する。
- 最初に異なる中間値を特定する。prefill、投影、後続状態更新、samplingを混同しない。
- 問題の演算を独立数値参照と照合し、直列側も正しいとは仮定しない。
- 原因を説明できる反証可能な再現例または介入実験を残す。差を消すだけの待機追加を原因説明にしない。
- 原因に対応する修正が必要なら、関連する数値・反復・実モデル条件で検証する。

## 調査中の検証状態

- 保存済みの両pack候補とNVFP4限定候補、直列baselineを保持している。
- 両pack候補のV620 MTPなしではtoken index 8から不一致。同一processの4回は同じ不一致列。
- GPU fatbin、model、prompt、sampling seedは同一だった。これは演算全体の正しさを証明しない。
- 保存binaryの再実行では不一致を再現できていない。原因特定・修正の受入条件は未達。
- GPUはmain agentが管理する。subagentは独立したkernel監査と診断入口の調査を担当する。

生成物: `.local-artifacts/phase87/fp8-cause/`。採用済みNVFP4限定経路と既存の未commit変更は保持する。
この調査ではcommit／pushしない。

関連履歴: [段階6](../../../../../history/2026/09/21-30/phase87-stage6.md)。

## 追加観測（原因は未確定）

- 保存した不一致binaryを再実行すると、16-token、128-token、元と同じwarmup 1回＋計測3回のいずれもbaseline列に一致した。過去の不一致reportとbinaryの保存hashは確認済み。
- 同一streamでlogits/control/supportを採取する診断では、直列と両pack並列のprefill後・eager decode後・126 replayのBF16 logitsがbit一致。独立CPUのtop-k/top-p/乱数計算とも126行一致した。採取による時間変化がraceを隠す可能性は残る。
- 過去の最初の差はtoken index 8（graph replay 7回目）。本日の乱数0.6198402433は先頭候補321のCDF 0.6193568137に近く、微小なlogit差でも303から321へ反転し得る。この近さだけでは原因を説明できない。
- 新規hipMalloc領域28,297,645,584 bytes（1,537回）を0x3f／0で埋めた2実行、ホストMALLOC_PERTURB_=85の実行も126 replay logitsが直列とbit一致。LDS/registerの未初期化や時間依存性までは除外できない。
- 凍結した両pack並列binaryの実captureは1,172 nodes、NVFP4 56組＋FP8 48組だけがforkし、それ以外の辺は保存baselineと一致した。
- 独立整数oracleを使う同形状HIP DAG再現器と、過去の不一致列をteacher forceする同条件prefill診断を準備している。単純な再実行成功を解決とは扱わない。

詳細生成物: `logit-comparison.json`、`logits-serial-r1/analysis.json`、
`logits-bad-r1/analysis.json`、`host-poison-dag-r1/dag-check.json`（上記生成物directory配下）。

- 実captureと同じ1,172 nodes／1,275 edgesを使う整数DAG probeは、10 fresh processes ×3 variants ×2同期方式 ×128 launchesで全nodeのCPU oracle一致。初版probeはFP8の4-node範囲を誤って4分岐にしていたため棄却し、実captureの辺を使う版へ修正した。これは実モデルkernelのmemory accessを再現する試験ではない。
- FP8 M=1、K=5120、N=10240/6144の4入力fixtureでは直列／並列差は0。131,072出力は独立FP64参照に対してFP32累積＋scale＋BF16丸めの誤差予算内だった。検出用4 ULP閾値そのものを正しさの根拠には用いない。matmul参照はquantized activationを入力にしているため、quantizerの数値検証は別途行う。
- control/ring/stagingとresident uploadの寿命監査で具体的欠陥は見つかっていない。upload完了待ちはあるが、productionでdevice weight全量をD2H checksum照合する処理はない。過去のdevice上の内容を復元できる証拠ではない。

- matched prefill 8192／chunk 2048／state 10240のeager teacher forcingを実行。GPU fatbinは保存benchmarkと同一。過去bad列を入力すると128位置中12位置の選択が現在の分布＋seed123と不一致だった。最初の9 logits（分岐直前まで）は既存wholegraph snapshotとbit一致。単一counter shiftやseed 0..65535の置換では全列を説明できない。
- GPU graphにも同じ列を入力する診断は、snapshot後にControlV1.pending_tokenだけを同じstreamで上書きする。selector/result ringは保持し、元の選択結果を観測する。これは入力介入実験であり、通常の生成性能として扱わない。

- wholegraphに過去bad列をteacher forceした直列・両pack並列も、全128位置のlogitsがeager teacher forcingとbit一致した。入力の自己回帰分岐を除いても、現在の実行では両者の差は出ない。
- 次はallocation redzoneとHIP transfer全byte照合で、両経路に共通し得るmemory corruptionを切り分ける。ROCm/legacy-rocm-build issue 6123（別version／RX6900XTの未確定報告）は転送検証の参考であり、今回の原因として採用しない。

- device allocation guard検査は直列・両pack並列とも1,537 allocations/frees、guard破壊0、検査API失敗0、生成列もbaseline一致。診断器は513-byte allocationの1-byte意図的越境を書き込み、検出感度を確認した。これはallocation内の誤書込みやLDSを検査しない。
- HIP transfer単体はpageable/pinned、fresh/reuse、direct/16 MiB chunk、4097/20 MiB/64 MiB+3の24条件、H2D+D2H計36.0156 GiBでbyte差0。実モデル全H2Dの直接照合を続ける。
- 共通prefillのGDN column、causal qtile4/qtile8、FP8 provider71のsource監査で具体的raceは見つからなかった。GDN B/A N=48はFP8 provider71ではなくBF16 matmul。現在のprofileにも外部BLAS GEMMは現れていない。

追加検証の集計と未解決点: [調査履歴](../../../../../history/2026/09/21-30/phase87-fp8-fork-investigation.md)。

## 追加ユーザー判断と採否の再評価

2026-09-21、ユーザーは設定変更の心当たりはないと回答し、bit反転等の可能性もあるため、
再現しない過去の結果を採否根拠から除外して再評価することを許可した。
原因特定を再評価の完了条件にはしない。raw証拠は保持し、原因をbit反転と断定しない。

- baselineは採用済みNVFP4-only、候補はNVFP4＋FP8 GDN。source差はfork条件の`!fp8_gdn`だけ。
- exact V620 primaryでA/B/B/A順に通常8192/128、MTPなし／あり、1 warmup＋3 measured。
  diagnostic interposer・profilerを入れない。各mode内のtoken一致とcleanupを確認する。
- 採用判断は既存Phase87原則に従い、同一process AB/BAの本番形状短縮が通常TPOTの1%以上、
  全roundで改善、他条件で退行しないことも確認する。単発の全モデル速度だけで採用しない。
- 独立数値oracleと既存DAG検証はsource・device codeが同一なので再利用する。
  固定列の実graph logitsをBF16 referenceとも比較し、KLDを記録する。
- 最終scopeに変更があれば、該当compile／host check／CI hash参照を更新する。commit／pushはしない。

再評価生成物: `.local-artifacts/phase87/fp8-reevaluation/`。

### 再評価中に見つかった別の停止

NVFP4-only基準版の4request目、decode step106で停止した。GPUには実行中kernelがなく、
HSA queue先頭のbarrierがvalue=-1のsignalを待っていた。採取後にその値だけを0へ戻すとqueueが解放された。
過去N0と同因とは断定しない。元のABBA A1はFAIL／介入済みとして速度集計から除外した。

現在のgfx1030比較は既存HIP control `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0` で行う。
同SDK・同binaryの基準版はこの条件でwarmup＋計測3回が完走／N0一致した。
R9700はclassicで速度が下がるため従来の1を維持する。globalなROCm install/symlink変更は行わない。

### 最終候補と検証範囲

- fresh FP8単体3process×6roundでM1は全18round改善、M3は1round退行。
  M1だけを追加候補にし、NVFP4のM1/M3とR9700の経路を維持する。
- MTPなしは既存の同一classic基準A1を再利用しB1/B2/A2の順でABBAを完了する。
  MTPありはFP8 GDN M3で、最終候補はこのgraphを変えないため同条件A/Bを記録する。
- BF16 referenceは同じ8320 token列・128評価位置、llama.cpp BF16／FP16 KV／batch64。
  比較側はNVFP4＋FP8／MXFP8 KV／prefill2048で、両候補間の条件を揃える。
- CI source inventoryの個別hashと集約hash、H3からの参照hashを更新する。
