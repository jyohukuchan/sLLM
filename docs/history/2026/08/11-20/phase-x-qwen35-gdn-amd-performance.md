# Phase X Qwen3.5系GDN llama.cpp AMD性能調査・修正履歴

## 2026-08-17: Phase割当と詳細計画作成

- ユーザーの明示指示により、Qwen3.8-27B/Qwen3.5 architectureのllama.cpp HIP性能低下を調査・修正し、
  sLLMへ還元する独立phaseとして`Phase X`を割り当てた。
- Phase Xの`X`は数値roadmapから独立した横断調査・修正を表す。Phase 20のGGUF統一を繰り下げず、両Phaseはsource/model containerの
  接点を持つが、Phase 20の完了条件、実行順、状態を変更しない。
- 2026-08-17の開始baselineとして、同じQwen3.8-27B Q5_K_XLと約9.4k tokenの実code-generation promptで、
  V620 HIP 59.6/約5.2、R9700 HIP 68.95/約12.0、V620 Vulkan 203.69/33.41、R9700 Vulkan
  718.08/48.16 prefill/decode tok/sを記録した。全runはEOS前に中断しており、最終MTP acceptance証拠ではない。
- R9700 HIP prefillが2,048 tokenの251.74 tok/sから9,377 tokenの68.95 tok/sまで低下したこと、upstream
  #18823/#20218/#20292で大量の小GEMM dispatchとhipBLASLt solution lookupが報告されていることから、
  GDN chunked prefillを第一仮説にした。GDN decode、MTP、Q5_1/262k memory、Harness overheadも独立仮説として残した。
- 最新llama.cppのfused/HIP/Vulkan GDN、open chunked prefill/MTP/KV issueを再監査し、profile、ablation、
  llama.cpp patch、sLLM reuse/port/native decision、focused GPU evidenceまでをPX-A0〜I0へ具体化した。
- sLLMが既にPhase 9でllama.cpp GDN state layoutをadaptしていることを反映し、新しい直接reuseは既存noticeを
  上書きせず新しいprovenance eventにする。Vulkanは比較controlだけとし、sLLM backend supportへ追加しない。
- この時点では計画文書とmain plan導線だけを変更し、llama.cpp/sLLM source、local subagent runtime、GPU build、
  model artifact、backend選択は変更していない。

## 2026-08-17: Phase X完了（fixed）

- update skillに従い`reference/`をupstream release/issueと照合した。llama.cppはb10227からb10453、commit
  `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`へ更新した。TensorRT-LLM v1.2.1、ATOM v0.1.5、
  LMDeploy v0.15.0、KTransformers v0.6.4はcurrentだった。vLLM v0.27.1とSGLang v0.5.17は、latest側に
  GPU crash/wedgeまたはsilent corruptionのopen regressionがあるため更新を保留した。
- 実行用llama.cppはbuild 901、commit `4df29be4f4c3673f428170fda944a5b19f743bb8`へ固定し、
  Qwen3.8-27B Q5_K_XLのmodel SHA-256、9,435-token実Python code prompt digest、software/GPU tupleを固定した。
- fresh four-way baselineはV620 HIP 60.99/6.81、R9700 HIP 69.50/12.50、V620 Vulkan 207.18/36.56、
  R9700 Vulkan 726.48/51.91 prefill/decode tok/sだった。MTP、context reserve、ubatch、Harnessを切り替えても
  historical GDN pathologyを説明せず、R9700 profileではcurrent fused GDNが1,152 dispatch、kernel time 6.16%だった。
- 根因はHIP baseline buildの`GGML_CUDA_FA_ALL_QUANTS=OFF`でQ5_1 K/VがFlash Attention対象外だったことである。
  `ON`で両targetをfresh buildすると、full promptの1 warmup + 5 measured中央値はV620 340.80/33.42、
  R9700 779.06/41.93 tok/sとなり、baseline比5.59x/4.91x、11.21x/3.35xへ改善した。
- Qwen exact shapeのQ5_1 Flash-Attention testをlocal llama.cppへ追加し、head dimension 256、GQA比6、
  KV長113/512/1024、query batch 1/3/17をCPU numerical oracleへ照合して`gfx1030`/`gfx1201`各18/18 PASSした。
  GDN operatorは変更しておらず、CPU fallback、mixed backend、GTT spillはなかった。
- userの明示停止条件に従い、decode数tok/sのslow baselineは各一回で停止し、計画上のbaseline 5 measuredと
  counterbalanceは適用しなかった。survivor candidateは両targetで規定の1 warmup + 5 measuredを完了した。
- #18823/#20292 closed、#20366/#20334 merged、#26001/#20377 openを再確認した。Q5_1 HIP Flash Attentionの
  all-quant build coverageにexact一致する既存issue/PRは見つからず、外部投稿は行わなかった。
- spare V620のlocal Qwen subagent runtimeを新HIP buildへ切り替え、Responses-compatible endpoint、DeepSeek Harness、
  context 262,144、Q5_1 model/draft KV、MTP幅3で実taskを完走した。旧HIP buildはrecoverable trashへ移し、Vulkan buildはcontrolとして残した。
- sLLMはFP16 KVを使用し、`linear_attention.gdn.v1`は原因でなかったためsource変更を行わない。新規llama.cpp
  code import/adaptation、provenance event、notice追加は不要である。Phase 20の状態と完了条件も変更しない。
- 再現identity、全反復値、profile、correctness、scope limitは
  [bounded summary](../../../../../ci/matrix/phase-x-qwen38-amd-summary-v1.json)へ固定し、Phase Xを`fixed`として完了した。

## 2026-08-17: post-closeout local subagent運用更新

- main Codex agentからの発見性を上げるため、`AGENTS.md`にlocal Qwen subagentの入口を追加し、起動、DeepSeek Harness接続、
  bounded task、検証責任、障害時のfail-closed手順を
  [運用正本](../../../../development/local-qwen-subagent.md)へ集約した。
- 524,288 contextはV620 32 GiBで全65 model layer、target/draft Q5_1 KV、MTP幅3を同時に保持できず、compute buffer OOMになった。
  52 layer GPU offloadなら起動できたが、ユーザー指示によりCPU layer offloadを採用しない。
- contextを368,640へ変更すると全model layer、target/draft context、MTP幅3がV620に収まった。llama.cppがnative
  262,144でslotをcapしないよう`qwen35.context_length`を同値へ明示overrideし、`--fit off`と`/props` actual-context照合で
  自動縮小を拒否する。batch/ubatchは512/128である。
- idle/validation中VRAMは約33.13/33.16 GB（sysfs total 34.34 GB）、GTTは約23 MBで、全model layer GPU offload、
  actual context 368,640、localhost Responses-compatible DeepSeek Harness tool loopを確認した。この運用値はnative windowを
  超える出力品質、368,640 token実入力のcorrectnessまたは長時間安定性を証明しない。
- 更新後の`AGENTS.md`だけを読むread-only Harness taskは、main agent用の正しいdelegation commandを返して完了した。
  9,597-token requestのprefill/decodeは255.47/30.74 tok/s、MTP acceptanceは0.64865だった。

## 2026-08-17: V620×2・1M context非運用ベンチマーク

- ユーザー指示により、通常のlocal-subagent運用とは分離した一時構成で、canonical/spare V620の2基をllama.cpp
  `--split-mode tensor --tensor-split 1,1`へ割り当てた。contextは1,048,576、Q5_1 target/draft KV、MTP幅3、
  batch/ubatch 512/128、parallel 1とし、`/props`でactual context 1,048,576を確認した。
- 9,435-token実Python code prompt、128-token出力、1 warmup + 3 measuredの中央値はprefill 416.80、decode
  47.90 tok/s（平均416.80/47.92）だった。MTP acceptanceはmeasured 3回とも0.78761である。
- request中のsysfs観測peakはBDF `0000:03:00.0`が33,274,519,552/34,342,961,152 byte、
  `0000:43:00.0`が33,286,418,432/34,342,961,152 byte、合計66,560,937,984 byte（61.99 GiB、96.91%）だった。
  合計headroomは2,124,984,320 byte、GTTは40,599,552 byteで、materialなGTT spillは観測しなかった。
- 2基間は2-hop PCIeで、internal AllReduce初期化に失敗してmeta-backend butterflyへ移行した。またtensor splitでは
  backend samplingが非対応のためtoken samplerだけCPUへ移行した。全model layerとtarget/draft contextはGPU residentだが、
  GPU-only end-to-endの証拠ではない。
- 1Mはcapacity確保を確認しただけで、1M-token実入力、native 262,144超の品質、長時間安定性を検証していない。
  262,144 context・batch/ubatch 2048/512のsingle-V620 Phase X中央値に対する1.223x/1.433xは条件差があるため、
  tensor-parallel scalingの証拠にしない。詳細値は
  [TP2 1M bounded summary](../../../../../ci/matrix/phase-x-qwen38-v620-tp2-1m-summary-v1.json)へ固定した。
- 一時serverを停止後、通常のspare V620 1基、actual context 368,640、全layer GPU、Q5_1 target/draft KV、MTP幅3の
  serviceを復旧した。TP2/1M構成は通常wrapper、skill、起動手順へ追加していない。

## 2026-08-17: multi-GPU profile selection follow-up

- 同じQwen3.8-27B Q5_K_XL、target/draft Q5_1 KV、MTP幅3、batch/ubatch 512/128で、現行の
  11,058-token Python code promptを2要求同時に送り、独立V620 server 2基、V620×2 layer/tensor、
  R9700+V620×2 layer/tensorを比較した。各survivorは1 simultaneous warmup + 1 measured、3基tensorだけは
  明確な非候補として1 measuredに限定した。
- 独立V620 2基は各368,640 contextでmeasured 45.58/47.01秒、prefill 263.33/254.92、decode
  35.66/35.21 tok/sだった。各GPU peakは33.20/34.34 GBでCPU model fallbackとGTT spillはない。最大aggregate
  throughputの候補だが、現行Harnessは単一endpointであり2 server dispatcherを実装していない。
- V620×2 layer splitは1,048,576 total・parallel 2でMTP compute-buffer OOMとなった。917,504 totalへ縮小すると
  actual 458,752/slotで起動したが67.31/69.56秒だった。V620×2 tensorはactual 524,288/slotで59.14/60.78秒、
  peak 33.35/33.34 GBだった。internal AllReduceなしのmeta-backend butterflyとCPU samplingを使うため、exact
  0.5M×2 capacityが必要な場合だけのexperimental profileとした。
- R9700+V620×2用に`gfx1030;gfx1201` multi-target HIP buildを作り、layer比率`1,1,1`、`3,2,2`、`2,1,1`、
  `5,2,2`を比較した。`5,2,2`がactual 524,288/slot、45.82/47.90秒で最良となり、peak VRAMはR9700
  30.97/34.21 GB、V620 16.14/21.52 GBだった。これ以上R9700へ寄せるとheadroomを約3.24 GB未満へ削るため止めた。
  3基tensorは63.45/65.08秒で、異種GPU同期と3-way butterflyのため棄却した。
- upstreamではlayerを低速interconnect向けの互換既定、tensorを高速interconnect依存のexperimental、rowをdeprecatedと
  している。現行guideはtensorと量子化KVを非対応とする一方、固定commitはQ5_1を受理したため、local成功を一般supportへ
  昇格しない。1M/0.5Mはcapacityであり、native 262,144超の品質や最大長入力を証明しない。
- 終了後は全一時serverを停止し、通常のspare V620 1基、actual context 368,640 serviceを復旧した。結果は
  [multi-GPU selection summary](../../../../../ci/matrix/phase-x-qwen38-multi-gpu-selection-v1.json)へ固定した。
- 事前のlocal Qwen read-only reviewは約8,192 internal tokenを生成して可視回答なしで終了したため、その報告を判断根拠に
  使用しなかった。これはGPU runtime failureではなく、長いreview taskをlocal subagentへ渡す際の出力制御上の制限である。

## 2026-08-17: DeepSeek Harness local profile hardening

- latest npm `@deepseek-ai/dsh` 0.1.0-rc.6とcomposed headless profileを監査した。upstream bundleはAGENTS/CLAUDE instruction
  自動読込、filesystem read/write/edit、glob/grep、sandboxed Bash、jobs、todo、Code Modeに加え、web、skills、recursive
  subagent、workflow、goal/Ralphを既定で持つ。MCP clientは同梱されるがheadless bundleへ自動接続されない。
- single-slot local Qwen用途ではbuilt-in filesystem/search/Bashでrepository taskを完結できるため、MCP serverは追加しなかった。
  web、skills、subagent、workflow、goal/Ralph、duplicate editorをlocal profileで無効化し、recursive delegationと不要schemaを除いた。
- profile personaと専用`$DSH_HOME/AGENTS.md`を追加し、既にsubagentであること、search-first、小window read、scope維持、
  concise reportを固定した。structured readは300 line/24,000 byte、retained result pruningは4,096文字からとした。
- wrapperはread-onlyを既定にし、editing時だけ`--workspace-write`を要求する。native toolsを既定、Code Modeを任意control、
  one-shot deadlineを既定900秒とした。Harness model catalogはactual context 368,640、max output 8,192へ同期した。
- native read-only taskは30.85秒、Code Mode controlは37.37秒、2-file Python editing + 4 unittestは54.87秒で完了した。
  main agent再実行も4/4 PASSし、read-only write denialは12.86秒で対象file不在を確認した。通常llama.cpp serviceのGPU、
  context、KV、MTP、offload構成は変更していない。

## 2026-08-17: V620×2 TP2の通常subagent運用への昇格

- ユーザーの明示決定により、単一V620のlocal Qwen経路を廃止し、V620×2 tensor split `1,1`、parallel 2、
  non-unified KVを通常構成へ昇格した。524,288 context/slotの比較構成から491,520/slot、983,040 totalへ縮小し、
  Q5_1 target/draft KV、MTP幅3、batch/ubatch 512/128、全model layer GPU offload、fit無効を維持した。
- DeepSeek Harnessは単一endpointを継続して使用する。endpointが2 slotを公開し、独立した2つのHarness processが各slotを
  利用する。wrapperへprocess全体を保持範囲とする2本の非待機leaseを追加し、3本目はqueueせずCodex subagent利用を促す。
- main-agent規則はboundedな委譲でQwenを優先する一方、Qwen利用不能・不適切、2 slot使用中、または追加並列性が有用な場合に
  native Codex subagentを躊躇なく使用する方針へ変更した。Qwen待ちによる直列化と単一V620へのfallbackは禁止した。
  main taskがsLLM GPU作業でV620を必要とする場合もQwenを利用不能として扱い、idle serviceを停止してpairを解放する。
- model catalogのcontextはcombined totalではなくper-slot 491,520へ同期した。native 262,144超はcapacityであり品質保証ではない。
  既存524,288/slot benchmarkは旧条件の比較証拠として保持し、新しい通常構成の性能値へ読み替えない。
- 通常serviceを再起動し、`/props`由来491,520/slot、2 slot、non-unified KVを確認した。起動直後のVRAM headroomは
  約2.48 GB/GPU、2 Harness task後は約2.40 GB/GPUで、GTTは約17/23 MBに留まった。2本のread-only Harness taskは
  26.85/24.58秒で別slotを使用して完了し、3本目は即時status 75、完了後は両lease解放を確認した。
- internal AllReduceは従来どおり初期化できずmeta-backend butterflyを使用し、tensor split非対応のtoken samplingだけが
  CPUとなった。この既知制限をmodel layer CPU offloadやGTT spillの成功扱いへ読み替えない。
- readinessはhealth/contextだけでなく`/props.total_slots == 2`、managed PID、V620 pair環境、TP2/non-unified KV/Q5_1/MTPの
  必須argvをfail-closedに検証するよう強化し、通常status/startup出力からlocal endpoint addressを除いた。

## 2026-08-17: Pi coding agent比較と通常経路への採用

- 同じV620×2 TP2 endpointでPi coding agent 0.84.2を試し、ResponsesではHarnessと同じduplicate tool ID/argument不整合を
  再現した一方、Chat Completionsでは複数tool callのIDと引数が一意かつ正しく保たれた。このため通常agent loopを
  DeepSeek Harness ResponsesからPi Chat Completionsへ変更し、Harnessは明示的な互換・診断経路として残した。
- Fast/Standard/DeepをsLLM Phase 20相当の段階課題で比較した。Fast read-only監査は144.92秒・tool 10回で完了した。
  Standard metadata parserは600秒で未完となり、6,144 output tokenの一括writeが切れたため、通常Standardを8,192へ増やし、
  大規模新規moduleをDeepへ振り分け、write/editを約12 KB未満へ分割する規則を追加した。
- Deepはreasoning budget 4,096、output 8,192、hard deadline 1,800秒の試験で1,605.56秒・tool 57回を要したが、compile/test
  errorを自力修正し、GGUF tensor-table/range parserのfocused 25/25とcrate全testを通して最終報告した。Qwenは課題に与えた
  NVFP4 `32/17`と正本の`64/36`の矛盾を報告した。比較artifactは実装採用を目的とせず、main treeへmergeしていない。
- native Codexは同等Deep課題を概ね5分で完了し、O(n log n) overlap検査を実装した。独立reviewではQwen案の65,536 tensor時
  O(n^2) scan、allocation順序、両案が課題側の誤ったNVFP4値へ従った点をfindingとした。Qwen Deepはboundedな中規模実装と
  second opinionへ使用できるが、format/security acceptanceはCodex mainが正本照合、計算量確認、focused testを行う。
- 通常profileはFast 300秒、Standard 900秒、Deep 3,600秒とした。Deepは進捗と明示理由があればdeadline overrideまたは
  no-timeoutを許容する。Pi wrapperは2本の既存leaseをHarnessと共有し、Landlockでhostをread-only、per-run scratchと
  workspace-write時のcurrent workspaceだけを書込可能にした。read-only/workspace-write smokeとLandlock denialを確認した。
- model、TP2、491,520 context/slot、Q5_1 target/draft KV、MTP幅3、batch/ubatch、全layer offloadは変更していない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase-x-qwen35-gdn-amd-performance.md)
