# SQ8 P8-C sampling, cancellation, and resident worker

日付: 2026-07-10

## 前回の要点

P8-B2で選択したM=128経路はclean correctness、3584+512 deep boundary、formal TTFT/decodeの全固定gateに合格した。P8-Cではrequest batchingを追加せず、deterministic sampling、cross-thread cancellation、active1/waiting0 resident workerを実装する。

## 今回の変更点

- `sq8_sampling` moduleを追加し、`rand_chacha=0.3.1`と`rand_core=0.6.4`をCargo.toml/Cargo.lockで固定した。
- signed i64 seedをtwo's-complement u64 bit patternへ変換し、request-local `ChaCha8Rng`を初期化する。
- temperature 0はfull-vocabulary greedyとし、同率では小さいtoken IDを選び、RNG drawを消費しない。
- temperature > 0は全logitのfinite検査後、f64でtemperature適用、top-k、stable softmax、最短top-p prefix、再正規化の順に処理する。tie-breakはtoken ID昇順である。
- samplingを`propose`/`commit`の2段階に分けた。破棄したproposalはRNG stateを進めず、stochastic tokenはcommit時だけ1 drawを確定する。これはcancelで未公開tokenを破棄する前提である。
- greedy一致、transactional proposal、同一seedの固定32-token列、異なるseed、top-k/top-p境界、NaN/Inf/all-masked相当、signed seedを8 testsで固定した。
- engine 270 tests、gfx1201 serving example checkは合格した。`clippy -D warnings`は既存の`ullm-runtime-sys/build.rs:86` needless borrowのみで停止した。

## 次の行動

1. 選択済みM=128に合わせてprotocol progress cadenceを仕様書で固定する。
2. sessionのtokenをprepared/publishedの2段階にし、cancel storeとstdout flushを同じ排他境界で線形化できるAPIへする。
3. sampler proposalをsessionへ接続し、未公開tokenのcancelでRNG/scheduler/generated counterが進まないことを決定的testで固定する。
4. strict bounded JSONL parserとresident workerを実装する。

## M=128 progressとtoken publication境界の固定

### 前回の要点

P8-B2で製品prefillはM=128に決定したが、P8-A時点のworker/session仕様にM=8と8-token progressが残っていた。また現sessionはtokenを返す前に内部counterをcommitするため、stdout flushを公開点とするprotocolとcancel raceを線形化できなかった。

### 今回の変更点

- protocol progressは各M=128 chunk完了時とprefill/decode遷移時に固定した。M=1 tailは毎step cancelを確認するが、旧8-token milestoneは合成しない。
- sessionはprepared token作成時にRNG、scheduler generated count、completion countを進めない。cancel時はproposalを破棄し、公開する場合だけcommitする。
- readerのcancel storeとinferenceのtoken write/flush/commitをactive-slot publication mutexの同じ排他区間に置く。flush成功後にだけRNG/counterをcommitし、mutexを解放する。cancelは公開前か、flush・commit完了後のどちらかに線形化される。
- write/flush失敗は生成stateを進めずfatalとし、flush後の内部commit失敗もfatalとして`released`を返さない。

### 次の行動

1. `sq8_serving_runtime` にprepared token stateとcommit/discard APIを追加する。
2. stochastic/greedyの両方で、prepare後cancelがRNG/counterを進めないtestを追加する。
3. そのAPI上にbounded JSONL workerとpublication mutexを実装する。

## Transactional session checkpoint

### 前回の要点

samplerのproposal/commit境界は完成していたが、serving sessionはtokenのstdout flushとcancelを一意に順序付ける2段階APIをまだ持っていなかった。

### 今回の変更点

- `TokenPrepared`状態と`prepare_advance_synchronized`、`publish_prepared_token`を追加した。
- prepare時はRNG draw、scheduler generated count、request generated count、feedback token、terminal stateを更新しない。publish callbackのflush成功後だけ、同じcancel mutex内で一括commitする。
- cancel先行ではcallbackとcommitが0回、publisher失敗ではcommitが0回、flush中のcancelはflushとcommit完了まで待つことをtransaction testで固定した。pending、RNG draw、active generated、scheduler generatedを同時に検査し、terminal tokenのcancel前後とflush後commit失敗も含めた。
- 仕様書の状態遷移へ`TokenPrepared`を追加した。既存P8-B diagnosticはno-op publisherを使う互換wrapperで維持した。
- `cargo test -p ullm-engine --lib`は279件、gfx1201 serving example check、Python serving tests 132件とsubtest 14件が合格した。

### 次の行動

1. strict bounded JSONL command/event codecとactive-slot制御をCPU test付きで実装する。
2. resident inference threadへ接続し、prepared token callback内でevent write/flushを行う。
3. prepare直後、flush中、terminal token前後のcancelを実GPU sessionで検証する。

## Strict bounded protocol checkpoint

### 前回の要点

transactional sessionは完成したが、stdinのbounded framing、strict command decode、ordered event flush、active1 ownershipは未実装だった。

### 今回の変更点

- payload 4MiBとoptional CRだけを保持する固定4,194,305-byte bufferを追加した。oversize terminated lineはLFまでbounded drainして次行へ復旧し、short/oversizeを問わず未終端EOFはfatalにする。
- depth 16、全階層duplicate key、UTF-8、trailing JSONを検査した後、shallow type inspectionとstreaming typed decodeを分離した。active中の5000-token generateでもfull decodeよりbusyが先行する。
- commandの構造検査とgenerateの意味検査を分離し、invalid_command、invalid_request、busy、unknown_requestの境界を固定した。
- exact event、毎行flush、writer permanent poison、M=128 progress、generation付きactive1 control、first cancel reasonを追加した。
- cancelはcontrol lockを外してからpublication mutexを待つ。ready/released/rejected requestのslot遷移は、ordered writerがflush成功後だけ生成できるprivate ackを必須にした。
- protocol 32 tests、engine全311 tests、gfx1201 serving example、Python 132 tests + 14 subtestsが合格した。独立レビューはP0/P1なしだった。
- `clippy -D warnings`は既存のdecoder/package等の警告で停止した。新protocol moduleで報告された指摘は修正した。

### 次の行動

1. capacity-1 synchronous channelと単一stdout writer threadを実装し、flush結果をcallerへ返す。
2. stdin reader loopへclean/active EOF、oversize/malformed recovery、fatal framing、busy/cancel/shutdownを接続する。
3. inference threadを追加し、HIP context/stream/sessionをそのthreadだけに所有させる。

## Resident CPU topology checkpoint

### 前回の要点

bounded codecとactive1 controlは完成していたが、stdin reader、常駐inference、単一stdout writerは未接続だった。

### 今回の変更点

- inferenceとeventの両channelをcapacity 1とし、単一writer threadが全JSONLのwrite/flushを所有する。
- backendをinference thread内で生成・破棄し、`!Send`なHIP/runtime/session ownerをthread間で移動させない。
- request-scoped publisherでactive generation、request ID、M=128+M=1 tail進捗、token index、EOS、terminal outcomeを固定した。
- fatalはatomic poisonを先に立て、best-effort eventをnonblocking送信し、後続token/releasedを拒否する。idle inferenceも50ms以内にpoisonを検知する。
- `released(A)`直後のB admit、B queue直後のEOF、terminal flush中EOF、operator cancel後EOF、startup前join、load/runtime/shutdown failureを決定的testで固定した。
- worker protocol/runtime 61 tests、engine全体、gfx1201 feature build、Python serving 132 tests + 14 subtestsで回帰確認した。新moduleのclippy警告は0件である。

### 次の行動

1. 実Qwen3 SQ8 backendを追加し、HIP context/stream/sessionをinference threadに常駐所有させる。
2. prepared-token callbackをrequest-scoped publisherの実flush境界に接続する。
3. worker binaryとprocess exit順序を実装後、R9700でcancel/reset競合、連続request、メモリ増加を検証する。

## Real session backend checkpoint

### 前回の要点

常駐CPU topologyは完成していたが、実`Qwen3Sq8ServingSession`をloadしてprotocolへ接続するbackendは未実装だった。

### 今回の変更点

- `Qwen3Sq8WorkerBackend`を追加し、gfx1201 feature、全HIP guard、isolated R9700、canonical artifact、packageをReady前にfail-closedで検査する。
- session、stream、contextをinference thread上で安全な破棄順に常駐所有し、M=128固定chunkと16 MiB upload chunkでloadする。
- prepared tokenのJSONL flush callbackをsession commit境界へ接続し、reset summaryのrequest ID、outcome、prompt/generated count、`reset_complete`をrelease前に完全一致で検査する。
- HIP ownerとprivate generic session adapterを分離し、製品とfakeが同じstart/prepare/publish/finish/abort変換を通るようにした。
- Generate開始後のprepared barrierでcancelし、token callback 0回を確認した。Length、EOS Stop、Cancelledの全cleanupでreset/abortがreleaseより先であることもtraceで固定した。
- engine 349 testsとgfx1201 feature buildは合格した。clippyは既知18件だけで、新moduleの警告は0件だった。

### 次の行動

1. CPU test可能なprocess runnerと`ullm-sq8-worker` binaryを追加する。
2. stdin block中のinference/writer fatal、clean EOF/shutdown、単一stdout ownerの終了順を統合testで固定する。
3. R9700で実adapterの2 token、prepare後cancel、連続2 requestを検証する。

## Worker process checkpoint

### 前回の要点

実session backendは完成したが、stdin reader、inference owner、stdout writerを一つの実行可能processへ束ねるentrypointは未実装だった。

### 今回の変更点

- generic process runnerとfeature必須の`ullm-sq8-worker` binaryを追加した。backendはinference thread上でbuildし、Ready後だけreaderを起動し、stdoutはwriter threadだけが所有する。
- mainはreaderではなくinferenceを待つ。inference/writer fatal時にstdin block中のreaderをjoinせずdetachし、writerのfatal処理完了だけを回収して非zero終了する。clean経路では全ownerをjoinする。
- CLIを`--artifact PATH --package PATH`だけに固定した。help、version、CLI error、process logはstderrへ出し、stdoutをprotocol JSONL専用にした。
- idle EOF、明示shutdown、active EOF cancel、load/shutdown/framing failure、stdin block中のbackend/writer failure、stdout単一owner、backend load 1回の連続2 requestを9 process testsで固定した。
- engine 358 tests、CLI unit 4 tests、実binary subprocess 3 testsが合格した。subprocessではOS exit code、help/CLI failureのstdout空、load failureのstdoutがprotocol JSONだけであることを確認した。
- 独立runtime reviewはP0/P1なしだった。

### 次の行動

1. release workerをbuildし、canonical artifact/packageと全HIP guardでR9700へ接続する。
2. 実2 token、prepare後cancel、連続2 requestのJSONLとreset証拠を取得する。
3. resident baselineへのVRAM/RSS復帰を確認してP8-Dへ進む。

## Terminal cleanup watchdog checkpoint

### 前回の要点

実workerのnormal/cancel resetは成功したが、HIP resetが停止した場合にworkerを強制終了する5秒watchdogは未実装だった。

### 今回の変更点

- finish/resetとabort/resetの両方を固定5秒watchdogで包んだ。watchdog threadはHIP objectへ触れず、cleanup completion channelだけを観測する。
- timeout時はcontrolをFailedへし、`cleanup_deadline_exceeded` fatal eventをnonblockingで試行し、releaseを禁止してprocessをexit 1にする。
- fast cleanup、timeout注入時のfatal codeとrelease禁止、test harness子processの実exit 1を3 testsで固定した。
- engine 361 testsとgfx1201 worker buildは合格し、新規clippy警告は0件だった。
- commit `ad6f2b6`のR9700機能smokeはresident A/B、completion 0/1 cancel、直後recovery、clean exit、最終VRAM復帰に合格した。これはformal latency/100-request gateを代替しない。

### 次の行動

1. watchdogをcommitし、release workerを再buildする。
2. cancel 2 warmup + 10 measuredのstore→flushed release p95を測定する。
3. cancel混在100 requestでpost-release RSS/VRAMの単調増加がないことを検証する。
