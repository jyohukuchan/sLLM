# Phase 52: R9700 100k KV物理コミットOOMの解消

> 状態: complete（2026-08-24）
> 対象: Radeon AI PRO R9700 exact `gfx1201`、単一GPU・単一要求、Qwen3.5-4B BF16 weight／FP16 KVの`100,000/2`
> 順序: ユーザー指示によりPhase 51を一時保留し、Phase 52を先に実施する。

## 目的

Phase 50後に修正した自動prefill capacity tierを使っても残ったR9700 `100,000/2`のOOMを、
失敗箇所を隠すchunk固定やsilent fallbackではなく、KV memory providerの物理コミット契約まで切り分けて解消する。
最終状態では自動prefill選択のまま同一caseを完走し、生成結果、HIP-only実行、資源後始末を確認できることを目的とする。

このPhaseは2026-08-24のユーザー指示によるR9700 OOM解消計画である。性能lane全体、Paged Attention、
複数要求・複数GPU、他モデル、MI300X最適化へ範囲を広げない。共通sourceへ変更が及ぶ場合だけ、影響targetの
focused regressionを追加し、その結果をPhase 51のcandidate入力へ渡す。

## 正本と固定入力

- 全体順序と共通の7行契約は
  [Phase 37以降の性能・機能ロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を使う。
- KV memory kind、opaque view、virtual-contiguous providerの現行契約は
  [KV memory方式](../../../../../architecture/kv-memory.md)と
  [runtime architecture](../../../../../architecture/runtime.md)を正とする。
- Phase 50の7行結果は
  [`phase50-r9700-summary-v1.json`](../../../../../../ci/matrix/phase50-r9700-summary-v1.json)を変更せず履歴として維持する。
  Phase 52の再実行でこのsummaryやschemaを上書きしない。
- 再現元sourceはcommit `159bc526cb26d180161f2cd7abcc22abb7e67e84`、fresh release binary SHA256は
  `9cf464705404cb263b166390461c85032cb49c39c9d7df01691a6cc964da4f63`である。
- modelは`phase20-audit-qwen35-bf16.gguf`、SHA256
  `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675`、model lock SHA256
  `425151d06832347a01b946b27336ceffac074eb7f6932af61e8c9821edc1e318`を固定する。
- inputはtoken ID `23066`を100,000個、input CSV SHA256
  `bb4f408734f3d5abc71b2740e8a1e4a7dc5351676783fc18be5dabd3b9ef0bec`、greedy 2 output、
  context length 131,072、BF16 weight、FP16 KV、MTPなし、batch／parallel 1とする。
- 実機tupleはR9700 BDF `0000:07:00.0`、UUID `GPU-a8e9ddefa2d60f55`、exact `gfx1201`、
  ROCm 7.14.0／HIP 7.14.60850、Code Object V6、wave32である。別tupleの結果は同一証拠へ混ぜない。
- 2026-08-24の自動選択再実行rowは
  `/home/homelab1/.local/share/sllm-evidence/phase50/r9700/rerun-auto-100000-20260824-159bc526/sllm/raw/phase50-r9700-sllm-long-100000/row.json`
  （SHA256 `a23ad8685a31368fac1a506beabe1dd8c4dbe9fe1750034a7cc9d3af8e9c8c58`）である。

## 既知の失敗と問題定義

- Phase 50最終候補は旧selectorでlayer 31のKV commitに失敗し、HBM peakは`26,414,587,904` bytesだった。
- capacity tier修正後、32 GiBの自動候補列は`[2048, 512]`となった。同一caseは約`152.867`秒後、
  `layer.23.kv_append`の`grow virtual KV physical commitment`でHIP status 260 OOMとなり、
  HBM peakは`13,160,554,496` bytesへ約50.18%低下した。GTT peakは`61,796,352` bytesだった。
- 失敗rowの`prefill_chunk_tokens`は自動指定を表す`null`であり、実際に2,048または512のどちらが採用されたかを
  失敗前に保存していない。従って、capacity tier修正でworkspaceだけが減ったことと、KV commit失敗の根因は分けて扱う。
- stderrにはOOMに続いて`HIP session cleanup failed: execution resource is busy`がある。一方、process groupは消滅し、
  HBM／GTTはbaseline `60,055,552`／`61,739,008` bytesへ復帰した。このcleanup errorを恒久leakと同一視せず、
  append途中のcommitとin-flight resourceの解放順を測る。

## 仮説と判定順

1. **H1: 複数planeのgrowがtransactionalでない。** 現行VMM growは物理pageを1個ずつ作成・mapし、K、V、scale planeを
   順に拡張する。途中の`hipMemCreate`／map失敗時に、そのappendで先に追加したpageや先行planeのdeltaが残り、
   logical appendだけがrollbackされている可能性を最優先で検証する。shared pageを先に置換するcopy-on-writeも同じ
   transactionに含め、旧shared mappingとhandle ownershipまで復元対象とする。
2. **H2: VMM handle／extent資源またはfragmentationが上限になっている。** HBM peakが総容量を大きく下回るため、
   1 page 1 handleの拡張回数、allocation granularity、map数、driver資源上限を総HBM不足と分離する。
3. **H3: provider固有の物理量をpreflightが見積もれていない。** model resident、graph state、workspace、reserveに加え、
   page丸め、mapped／committed bytes、同時grow deltaを比較し、`required <= available`の計算を選択providerと一致させる。
4. **H4: prefill workspace圧力が残っている。** 自動実効chunk、明示2K、明示512を必要最小回数だけ比較し、
   512でも同じcommit位置で失敗するかを観測する。明示512は診断であり、最終解決条件にはしない。
5. **H5: cleanup busyは失敗後の派生事象である。** profiled executionのabortがpending completionをwait／drainせず
   ownerだけ解放している可能性を含め、append event、view、mapping、handleの解放順を記録する。元のOOMをcleanup errorで
   上書きせず、OOM修正後もbusyが独立して残る場合はbounded drainを別のcorrectness修正として扱う。

## 作業単位

### A. 失敗前に確定情報を保存する

1. selectorのcapacity tier、candidate列、各candidateのexact memory見積りと棄却理由、最終実効chunkを、
   model execution開始前にrunnerが取得できる形で出力する。失敗rowにも必ず残し、明示指定と自動選択を区別する。
2. KV stateごとにmemory kind、virtual reservation、physical page bytes、mapped token capacity、K／V／scale各planeの
   append直前・要求delta・成功delta・失敗後delta、page／extent／handle数、HIP status、layerとtoken範囲を記録する。
3. 既存のopaque ABIを維持し、まず内部診断とevidenceへ追加する。public C ABIを拡張する場合は、必要性を確認した後にだけ
   `struct_size`／ABI version、Rust binding、layout probe、ABI test、runtime／KV文書を同時更新する。
4. pointer、VMM handle値、raw promptをsummaryへ含めない。token列はcountとdigestで表し、診断量はboundedにする。

### B. commit失敗をhostで再現し、rollback契約を固定する

1. VMM allocation／mapへ失敗注入点を設け、最初・中間・最後のpage、K成功後のV失敗、V成功後のscale失敗を検証する。
2. 失敗時はappend前から存在したmappingを維持し、そのappendで追加した全planeのmappingとphysical handleだけを解放する。
   copy-on-writeで置換したshared pageは旧mappingとownershipを復元し、単純な`mapped_bytes`差分だけで済ませない。
   logical length、mapped token capacity、accounting、snapshot/view公開値をappend前と同一に戻す。
3. rollback後のretryとstate releaseを成功させ、二重unmap、二重release、poisonの見落とし、handle／mapping leakを検出する。
4. 24／35／60／160 GiB境界のselectorに加え、capacity 65,535／65,536／65,537、chunk
   2,047／2,048／2,049、page境界の前後と非整列tokenを対象にする。巨大な全組合せmatrixは作らない。
5. profiled executionにpending completionを残してabortし、bounded wait／drain後に元のOOMとcleanup診断を分離して返し、
   owner、event、KV stateを安全にreleaseできることをfake nativeとfocused integrationで確認する。

### C. R9700で根因を最小回数で分離する

1. instrumentation入りcurrent candidateを自動選択で1回実行し、実効chunkと失敗時のper-plane deltaを取得する。
2. H4を分離する必要がある場合だけ明示2Kと512を各1回実行する。OOM後はprocess消滅、HBM／GTT復帰を確認してから次を始める。
3. H1〜H3が示された場合、VMM transactional grow／rollback、pageをまとめたextent作成、またはpreflight補正を
   個別candidateとして比較する。複数修正を一度に入れず、原因と採否を対応付ける。
4. R9700長contextだけで`contiguous-resident`を選ぶcandidateも比較対象とする。V620の65,536以上で既に使う方式を参考にするが、
   gfx1201へ無条件転用せず、full-capacity事前割当量、作成成功、E2E、peak HBM、後始末をVMM candidateと比較する。
5. OOM後にchunkを自動縮小して最初から再実行する処理、CPU/backend fallback、要求を分割して意味を変える処理は追加しない。

### D. 根因に対応する最小修正を採用する

- H1なら、単一appendに必要な全plane deltaを準備してから公開し、途中失敗時にdeltaを逆順で完全解放するtransactional growを採用する。
- H2なら、同じvirtual-contiguous契約を保つextent coalescingまたはhandle数上限の事前検査を採用し、driver failureを
  不明なOOMではなく測定済み理由へ変える。
- H3なら、選択providerのpage丸めと同時commit peakをgraph memory見積りへ加え、reserveを差し引いたavailable memory内でのみ開始する。
- H5が独立して再現するなら、profiled abortでpending completionをbounded drainし、drain failure／timeoutを元のappend errorと
  分離して保存する。cleanup成功のためにOOM原因を捨てたり、無制限waitを追加したりしない。
- `contiguous-resident`だけが安全に完走する場合は、exact `gfx1201`かつ長capacityへ限定したselector変更を採用候補とし、
  VMMの一般contractを削除しない。
- どのcandidateも根因と対応しない場合は採用せず、ROCm VMM／driver resource制約として証拠を固定して再計画する。

### E. 統合確認と記録

1. 最終自動経路で`100,000/2`を1 warmup＋3 measured実行する。各要求でtoken ID `[23066, 23066]`、
   HIP-only、fallback 0、timeoutなしを確認し、E2E／TTFT／peak HBMは性能gateではなく比較値として記録する。
2. `10,001/2`をfocused regressionとして実行し、thresholdより短い既存経路の正しさと重大な退行がないことを確認する。
   decode側へ共通変更が及ぶ場合だけ`32/20,000`も追加する。
3. gfx1201固有selectorだけならgfx1030／gfx942の非選択host testとproduction HIP compile/linkを行う。
   VMM growの共通sourceを変更した場合はV620の対応KV boundary testと、長contextへ影響するときだけ`100,000/2`を再確認する。
4. Phase 51と共通sourceが重なる場合は、Phase 52の最終commitをcandidate identityへ取り込み、MI300Xの既存
   `contiguous-resident`選択とwave64を変えていないことを再確認する。

## 完了条件

- 固定R9700 tupleの最終自動経路で`100,000/2`の1 warmup＋3 measuredが全て完走し、生成token列、HIP-only、
  fallback 0、timeoutなしを満たす。明示512指定を必要条件にしない。
- 各runがcapacity tier、candidate列、実効chunk、memory kind、page bytes、per-plane mapped／committed量、
  required／available／reserve、HBM／GTT peakを失敗時を含めて直列化する。
- 失敗注入した全地点でappendがatomicに失敗し、logical lengthを公開せず、追加mapping／handleを残さず、
  同じstateのretryまたはreleaseが成功する。
- 全要求後とprocess終了後にGPU process 0、cleanup error 0、HBM／GTTのbaseline復帰を確認する。
- `10,001/2`と変更範囲に応じたfocused regressionが正しさを維持する。llama.cppとの性能同等はPhase 52のgateにせず、差を記録する。
- 採用source、棄却candidate、exact identity、全反復値、既知制約をPhase 52専用の追跡済みsummaryへ固定し、
  Phase 50の7行summaryを変更しない。

## 証拠契約

- 実装時に`phase52-r9700-kv-commit-summary-v1`と対応schemaを追加し、runner、matrix、path-to-suite、validator登録を同期する。
- summaryはsource／binary／model／lock／input／runner digest、GPU tuple、selector decision、KV provider、per-plane commit、
  HBM／GTTのbefore／peak／settled、process、HIP dispatch、fallback、生成digest、各反復値を持つ。
- raw log／traceはrepository外へ置き、追跡summaryからSHA256で参照する。OOM、timeout、cleanup failure、0 caseはFAILとする。
- KV provider選択または契約を変更した場合は、`docs/architecture/kv-memory.md`と`docs/architecture/runtime.md`を
  実装と同じ変更単位で同期する。resource／selector／VMM transactional変更は採否に応じて
  `docs/compatibility/numerical-output-changes.md`へN0として記録する。
- Phase 52の成功・未達のどちらでも、closeout時に`docs/compatibility/gpu.md`、`amd-gpu.md`、`software.md`の
  R9700 100k statusを最終summaryへ合わせる。性能数値の詳細はcompatibility表へ複製せずhistory／summaryを参照する。

## 停止・再計画条件とrollback

- instrumentation後も二つのclean runで失敗位置・実効chunk・commit量が再現せず、コード原因を特定できない場合は、
  環境差を含む未再現として記録し、推測によるprovider変更を行わない。
- 明示512と`contiguous-resident`の両方が低いHBM使用量で失敗する場合は、ROCm VMM／driver／process resourceを優先して調べ、
  capacity tierをさらに下げるだけの変更を採用しない。
- 解決にPaged Attention、public ABI全面変更、allocator全体の再設計が必要になった場合は、Phase 52では診断と安全な後始末までを固定し、
  拡大範囲を別計画としてユーザーへ提示する。
- gfx1201限定selector candidateは一変更で戻せる形に保ち、未知targetの既定値を変えない。共通VMM変更は失敗注入testを
  rollback条件とし、1回の100k成功だけで全targetへ一般化しない。

## Phase closeout

完了または再計画時にこのplanをarchiveへ移し、採用source、棄却candidate、GPU証拠、残課題をmatching historyへ追記する。
Phase 51を待つこと、またはPhase 51がPhase 52を待つことを完了条件にしない。

## 完了結果

- source commit `3ed002c476b49417cc702119e37c2389cefb96bc`、exact `gfx1201` release binary SHA256
  `79b0099f0c8981c46d1629debaf2aacfe551107adb13ec00465f4ebce11c8f81`で固定tupleを実行した。
- 自動候補`[2048,512]`から2,048を選択した。`10,001/2`は従来の`virtual-contiguous`で3 warmup＋10 measured、
  `100,000/2`はcapacity 131,072の`contiguous-resident`で1 warmup＋3 measuredを全て完走した。
- 両行の生成tokenは`[23066,23066]`、HIP-only、fallback 0、cleanup failure 0、process group消滅、HBM/GTT baseline復帰だった。
  100kのKVは8 layer、logical/mapped 131,072、observed 100,001、K/V合計4,294,967,296 bytesである。
- 100k E2E/TTFT中央値は`325.593963905/325.526989625`秒、sysfs HBM/GTT peakは
  `15,388,794,880/106,524,672` bytesだった。性能同等はこのPhaseのgateにしていない。
- VMM extent集約、明示512、runtime retryは採用しなかった。resident providerの自動2K経路で完走したため、追加診断は不要と判断した。
  VMM transactional rollbackとbounded abort drainは、短いvirtual経路を含むfailure correctness修正として維持する。
- 追跡証拠は[`phase52-r9700-kv-commit-summary-v1.json`](../../../../../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)
  （SHA256 `9206f7d900b3656ff951c69f25fb36ea589cc752f7e5bf8ae9b08a4ddb82a771`）を正とする。

[全体計画](../../../../main-plan.md) / [対応する履歴](../../../../../history/2026/08/21-31/phase52-r9700-100k-kv-commit-oom.md)
