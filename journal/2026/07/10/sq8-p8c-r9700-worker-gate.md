# SQ8 P8-C R9700 worker gate

日付: 2026-07-10

## 前回の要点

`Qwen3Sq8WorkerBackend`と`ullm-sq8-worker`はCPU testと独立監査を通過したが、実HIP owner、連続request、cancel後reset、process終了時のVRAM解放は未検証だった。

## 今回の変更点

### 固定した実行条件

- git commit: `ad6f2b6e9bfc22c5f6a2b2ae5b4bf65a7ebd41d5`
- tracked source: clean
- excluded untracked path: `.rocprofv3/`
- release worker SHA-256: `eaab33501dec47d20911707f589ba1451dc9da946911fac8f1d82cba726a7278`
- artifact manifest file SHA-256: `23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2`
- package manifest SHA-256: `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb`
- device isolation: `HIP_VISIBLE_DEVICES=1`
- isolated runtime device: HIP device ID 0、`AMD Radeon Graphics`、compute 12.0、34,208,743,424 bytes
- physical R9700: PCI `0000:47:00.0`、DRM `card1`、rocm-smi `card2`
- required HIP guards: 10件すべて`1`
- request batching: なし、active 1 / waiting 0

artifact CLI引数はmanifest fileではなくartifact directoryを受ける。誤ってfile pathを渡した最初の試行はReady前に`load_failed`で停止し、VRAMを確保しなかった。この試行でprocess error詳細が欠けている運用上の問題を発見し、`ad6f2b6`で構造化stderrの`detail`へrunner errorを保存するよう修正した。

### 連続2 request

同じworker process、同じbackend loadでprompt `[1,2,3,4,5,6,7,8]`、greedy、最大2 tokenをA/Bへ順に送った。

| request | event列 | token IDs | indices | outcome | reset | elapsed | RSS | R9700 VRAM |
|---|---|---|---|---|---|---:|---:|---:|
| A | started, progress, token, token, released | `[353,10]` | `[0,1]` | length | true | 442 ms | 174,858,240 | 18,363,596,800 |
| B | started, progress, token, token, released | `[353,10]` | `[0,1]` | length | true | 280 ms | 174,858,240 | 18,363,596,800 |

A/Bでtoken列、index初期化、release summary、RSS、VRAMが一致した。workerをrequestごとに再起動せず、session reset後に同じresident backendを再利用した。

### Cancelと回復

cancel raceは二つの実経路を確認した。

1. R9700 busy 80%をrocm-smiで観測してからcancelした試行では、token index 0のflushが先に線形化され、その後`Cancelled`、completion 1、`reset_complete=true`になった。
2. 正しいR9700 sysfs、PCI `47:00.0` / DRM `card1`を使った試行では、prompt 128の`started`から2.007 ms後、busy 3%時にcancelした。token eventは0件、completion 0、cancel reason `operator`、`reset_complete=true`だった。

2件目の直後にprompt 8のrecovery requestを送り、token `[353,10]`、indices `[0,1]`、Length、`reset_complete=true`を確認した。workerはfatal化せず、明示shutdownでexit code 0になった。

CPU barrier testはprepare返却後かつpublish callback前のcancelを決定的に固定している。実GPU試行はGPU busy中のcancelとzero-token破棄、または先行flush後の1-token commitを確認する。両者を組み合わせ、publication mutexの前後どちらへ線形化されてもreset後に再利用できることを確認した。

### Resource結果

- 起動前R9700 VRAM: 87,384,064 bytes
- Ready後resident VRAM: 18,362,908,672 bytes
- A/B release後VRAM: 18,363,596,800 bytesで一致
- A/B release後RSS: 174,858,240 bytesで一致
- clean process exit: 0
- KFD解放直後は一時的に12,404,903,936 bytesが残ったが、後続観測では87,384,064 bytesへ復帰
- stderrは`ullm.worker.log.v1` JSONだけで、request ID、token count、elapsed、outcomeを含み、prompt/token内容を含まない
- stdoutは`ullm.worker.v1` event JSONだけだった

このsmokeは長期soakではない。P8-Cではworker単体のcancel 2+10回と100 request resource gateを先に実行する。HTTP gateway・systemd・再起動segmentを含む完全な64 MiB上限とTheil-Sen slopeの製品gateはP8-Fで再実行する。

### 5秒terminal cleanup watchdog

- commit: `b68f616ffedbf282f55795c4fff27a701a1356d7`
- finish/resetとabort/resetの両方を同じwatchdogで囲んだ。HIP処理は従来どおりinference threadが所有し、watchdogはcompletion channelだけを監視する。
- 5秒の絶対期限はwatchdog thread作成前に固定し、arm handshake後にだけcleanupを開始する。threadの起動遅延は5秒を更新しない。
- 期限超過時はordered writerをatomicにpoisonし、controlは`try_lock`でFailedへの変更を試すだけにし、最後にprocessをexit code 1で停止する。control mutexを別threadが保持した子process testでも2秒以内のexit 1を確認した。
- 正常cleanup、watchdog arm遅延、期限の前後の完了時刻、timeout後のno-release、control lock保持中のprocess exitを含むengine 363 testsが合格した。gfx1201 worker buildも合格し、clippyは従来の18警告だけだった。
- 2系統の独立監査でP0/P1なしと判定された。

## 次の行動

1. watchdogをGitに保存し、clean commitからrelease workerを再buildする。
2. cancelのatomic storeからflushed releaseまで2 warmups + 10 measuredを計測し、p95 2秒以内を独立に再計算する。
3. cancelを混ぜた100 sequential requestでRSS・VRAM・reset後再利用を計測し、合格後にP8-Dへ進む。

### P8-C standalone計測契約

- commit: `681a8c7`
- P8-Fの`ullm-openai.service`・gateway・再起動を含む610 sample schemaと分離し、P8-C worker単体専用の`ullm.sq8.worker_acceptance.raw.v1`を固定した。
- cancelはstdin write開始時刻からmatching releaseの読取時刻までを測る。これはatomic store起点より広い保守的上界であり、上界p95が2秒以内なら元の契約も満たす。
- latencyは2 warmups + 10 measuredでprompt/decode cancelを交互に実行し、各cancelの直後にnormal recoveryを完了させる。
- resourceは10 warmups後のbaseline 5 samplesと100 request×5 samples。各5 requestの4件目をcancel、5件目をnormal recoveryとする。worker VmRSSとAMD SMI process VRAMを主系列、KFD VRAMを完全一致cross-checkとする。
- Theil-Senは100点の全4950組を使い`<=262144 B/request`、最終点のbaseline差は`<=67108864 B`とする。

### 正式計測ツールとrelease build

- commit: `acbc60ea04b5e4b9a4f2374681623e5d36dfe236`
- release worker SHA-256: `145a5351db3957130200276314853e394d0fd206a69e2eab260c01141411b950`
- producer/validatorはraw command/eventの型厳密再構成、全34 cancelの5秒上限、135回のKFD隔離確認、505 resource samples、開始/終了時git identityを独立検証する。
- 証拠出力はrepo外に限定し、no-follow directory FDで作成・公開する。失敗や割り込みでは`raw.jsonl.incomplete`へ戻す。
- Python 78 tests、`py_compile`、`git diff --check`が合格し、独立の攻撃的監査でP0/P1は残っていない。
- gfx1201 release buildは合格。build後のsource statusは`.rocprofv3/**`以外clean。

## 次の行動

1. repo外の新規出力先でR9700正式runを実行する。
2. cancel latency 2 warmups + 10 measuredと100 sequential requestのraw証拠を独立validatorへ渡す。
3. 全gate合格後にP8-Cを完了し、P8-Dへ進む。

### 正式run 1回目のKFD snapshot競合

- evidence: `/home/homelab1/datapool/ullm/evidence/sq8-worker-acceptance-acbc60e-2026-07-10`
- latency 2 warmups + 10 measuredと各normal recoveryはproducer gateを通過した。
- resource warmup/baselineとrequest 1..86の全5 samples、request 87のsample 0まで完了した。
- request 87のsample 1で、KFD PID `1180214`を列挙した直後にdirectoryが消え、`vram_51545` readが`ENOENT`になった。短命のKFD process entryとlist/readの間のsnapshot競合であり、モデル処理や資源gateの不合格ではない。
- producerはfail-closed停止し、`raw.jsonl` は公開されず`raw.jsonl.incomplete`のままである。部分点の再試行や置換はしない。
- producer、worker、worker process groupは残留せず、AMD SMIはR9700にrunning processなしと報告した。

#### 次の行動

1. KFD取得を、PID directoryのinode identityを両端で検査する1秒以内のstable double-collect intervalとして仕様化する。
2. 破棄attemptを含むrawを保持し、不安定attemptでもworker以外の正VRAMを読んだ場合は即失敗する。
3. producer/validatorと負例を再監査・commit後、新規worker processで全runを最初から実行する。
