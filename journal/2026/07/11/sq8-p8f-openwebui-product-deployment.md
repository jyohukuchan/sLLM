# SQ8 P8-F OpenWebUI product deployment

Date: 2026-07-11

## 前回の要点

- P8-EでOpenAI互換SSE、単一active request、429、実socket切断後の同一worker復旧をR9700上で固定した。
- gateway codeは`3dcd1cb24ef7102abd0eecfd42ca47e47dc0d202`、P8-E evidenceは`b022523`まで保存済みだった。
- 次の目的はbatchを追加せず、systemd常駐gatewayを実OpenWebUIから通常利用できる状態にすることだった。

## 今回の変更点

### 配備コード

- commit `806fb8a1578e63d1cd199b7402747575b2a92ccc`で次を追加した。
  - `deploy/systemd/ullm-openai.service`
  - `deploy/systemd/ullm-openai-firewall.service`
  - `deploy/systemd/ullm-openai.env.example`
  - `deploy/nftables/ullm-openai.nft`
  - `deploy/nftables/ullm-openai-firewall`
  - `deploy/openwebui/compose.yaml`
  - `deploy/openwebui/configure.py`
  - `deploy/openwebui/browser-smoke.cjs`
  - `deploy/README.md`
- gatewayは`User=homelab1`、`RuntimeDirectory=ullm`、`CacheDirectory=ullm`、`KillMode=control-group`、`Restart=on-failure`で起動する。
- model、artifact、package、tokenizerは既存の永続product pathを使い、worker ready eventの固定identity照合を変更していない。
- `/etc/ullm/openai-api-key`は`root:homelab1 0640`、OpenWebUI session署名鍵は`root:root 0600`で管理し、値は証跡へ記録していない。

### Networkとfirewall

- Docker network ID: `79bb7cfca31cb5d76978cbbb229c946662c137b93ea647b5ae6c205af9126dc8`
- subnet/gateway: `172.20.0.0/16`, `172.20.0.1`
- bridge: `br-79bb7cfca31c`
- gatewayは`172.20.0.1:8000`だけでlistenする。
- dedicated `inet ullm_openai` tableは、port 8000宛ての非bridge入力と不正source subnetをdropする。
- OpenWebUI bridgeから正規keyで`GET /v1/models`は200、誤keyは401だった。
- hostからbridge addressへのprobeとLAN address port 8000へのprobeはともにHTTP `000`だった。nft counterでも非bridge packet 1件のdropを確認した。

### OpenWebUI

- OpenWebUI v0.9.4をdigest `sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff`で固定した。
- 既存external volume `open-webui`と既存providerを削除せず、uLLMをprovider index 1、`connection_type=local`として追加した。
- SQLite backup APIを使う設定toolを実データcopyで2回反復し、idempotent性を確認してから実volumeへ適用した。実volumeには変更前backupが3件ある。
- `title`、`follow_up`、`tags`の背景生成はDB persistent configで全てfalseになっている。
- OpenWebUI v0.9.4にはOpenAI互換providerの機能的なcontext-length設定がない。`num_ctx`はOllama専用で、OpenAI経路では上流へ転送されるため設定しない。
- `context_length=4096`と`n_ctx_train=4096`はmodel metadataとして記録した。機能上の唯一の正はgatewayであり、超過時は400を返す。
- model paramsは空にした。`max_tokens=256`をOpenWebUI model paramへ置くとrequest固有値を上書きするため使わない。未指定時の256とhard max 512はgatewayが所有する。
- session署名鍵を`/etc/ullm/openwebui-secret-key`からread-only bindした。containerを強制再作成した後も、再作成前JWTで認証200になった。
- CORS originはLANの正式URLとlocalhostだけに限定した。

### 実機smoke

- OpenWebUI API経由:
  - English: `OpenWebUI path works.` / `stop`
  - Japanese: `青` / `stop`
  - system prompt: `SYSTEM_OK`
  - multi-turn: `K9X`
  - code block: fenced Pythonを破損なく返した
  - `max_tokens=1`: content `1` / `length`
  - context overflow: 400、その直後に`RECOVERED.` / 200
  - fixed seed 4242: `Frosty`が2回一致
  - default sampling: `Valid.` / `stop`
- clientを最初のcontent後にcloseした試験では、first content 977.0 ms、busy probe 1回の後、valid generation込み784.2 msで`AFTER_STOP.`へ復帰した。
- direct gateway collisionはleader中に1.3 msで429、`Retry-After: 1`、`request_busy`だった。
- OpenWebUI v0.9.4のlegacy synchronous wrapperは同じ上流429をHTTP 400へ変換する。detailは`The model is serving another request.`として見え、leader完了後の次requestは200だった。gatewayの429 contract自体は維持されている。
- Playwright browser smokeはLAN URLでmodel名を表示し、assistant本文`BROWSER_OK`を描画した。page errorは0件だった。
- workerをSIGKILLするとgatewayもfailure終了し、systemd restart countは1になった。readinessはload中503、44.533秒後に200へ戻り、OpenWebUIから`FATAL_RECOVERY_OK`を返した。

### 検証

- gateway pytest: 114 passed
- gateway source strict mypy: passed
- gateway Ruff check/format: passed
- deploy `configure.py` strict mypy、Ruff、実DB copy反復: passed
- Compose config、systemd-analyze verify、nft `--check`、shell syntax、Node syntax: passed
- OpenWebUI DB `PRAGMA integrity_check`: `ok`
- machine-readable core evidence:
  - `uLLM-project/benchmarks/results/2026-07-11/sq8-p8f-openwebui-product-smoke-v0.1/summary.json`
  - browser screenshot SHA-256 `107b3461c4ac1410bf05c40f86e8d46b5473585b68d5aa0e3d277a875c257b82`

## 次の行動

1. full release evidence schemaと`tools/validate-sq8-openwebui-release.py`を先に固定する。
2. 実Stop button automationとpost-header injected failureを閉じる。
3. 20 sequential chatと5段階cancelの予備soakを行う。
4. 100 sequential HTTP chatのMemoryCurrent、RSS、VRAM、FD、thread、child、allocator、KV sampleとTheil-Sen slopeを記録する。
5. fixed fixtureのHTTP TTFT matrixを実行し、独立validatorで最終release gateを判定する。

通常のOpenWebUI text chatは現時点で利用可能である。上記はrelease evidenceを閉じる作業であり、batch追加や任意最適化ではない。

## P8-F release gate追加作業

### 不変契約と障害表示

- commit `5af479d`でgateway lifecycleを`ullm.gateway.lifecycle.v1`の構造化journalにし、request/completion identity、絶対monotonic時刻、cancel、release、worker fatalを相関可能にした。gateway 116 tests、Ruff、mypyが通過した。
- commit `a49d747`で`docs/specs/sq8-openwebui-release-v0.1.md`を固定し、OpenWebUI v0.9.4のpost-header provider error後に誤ったnormal doneを送らない派生imageを配備した。
- 実worker SIGKILLでbrowserはerror 1件、cancel 1件、error後done 0件となり、部分本文と失敗状態が保存された。systemd復旧後のOpenWebUI chatも200で成功した。

### Docker内raw HTTP client

- `tools/sq8-openwebui-http-client.py`を追加した。API keyを定型値やハッシュとしても出力せず、request body、response header、raw SSE chunk、SHA-256、`time.monotonic_ns`を有界JSONLで記録する。
- 単一active requestとし、外部closeと最初の非空`delta.content`でのsocket closeの両方を備えた。SSE分割、role除外、切断後のclient再利用、secret非漏洩を含む8 tests、Ruff check/format、`py_compile`が通過した。
- 実Docker networkで完走は200/eof、初回content切断は200/client_closedとなり、全body SHA-256が一致した。gateway journalは`client_disconnect`と`request_released(outcome=cancelled, reset_complete=true)`を記録した。
- HTTPのclient_closed直後はgateway reset完了前のため、次requestが429になる場合がある。対応する`request_released(reset_complete=true)`後に復旧requestを出した試験は200/eofで成功した。正式collectorは固定sleepではなくjournal releaseを同期点にする。

### Phase-1独立validator

- commit `b93dd9b`で`release-matrix.json`のtop-level、exact role/path/bytes/SHA-256、schedule/thresholds同一性、自己参照除外を規範文書に追記した。
- commit `8402f27`で`tools/validate-sq8-openwebui-release.py`を追加した。bundle layout・SHA256SUMS・matrix、strict JSON/JSONL、gateway lifecycle、global active-1、release後admission、resource `1+610+4`、segment別median/final delta/Theil-Sen、CLI trusted commit/worker hashを独立再計算する。
- 重複key、非finite数、invalid UTF-8、symlink、欠損resource、不正request相関、release前admission、資源増加、重複journal cursor、壊れた構造化journalを含む18件の改変testが通過した。
- 現段階は`--phase1-only`でのみexit 0とし、常に`release_status=incomplete`を出す。browser、5段階cancel、SSE/latency、post-header表示、全identity、最終validation発行は未実装gateとして明示的に残した。

### Cancel trigger観測経路

- `exact-p3584`の実pilotで、`journalctl -f`および`systemd.journal.Reader`からprogress=128を見てsocket closeすると、journaldの追従通知が約217 ms遅れ、progress=256まで進むことを確認した。HTTP close送信後のgateway cancel認識は1 ms未満であり、原因は切断処理ではなくjournal followerの観測遅延だった。
- commit `e1b3499`で、同じ非secret lifecycle JSONを`/run/ullm/lifecycle-observer.sock`へbest-effort/nonblocking AF_UNIX datagramとしてミラーする経路を追加した。公開HTTP contractにはテストfieldやendpointを追加していない。observer不在時は通常稼働であり、journalが引き続き正の証跡である。
- gateway全118 tests、Ruff、strict mypyが通過した。service更新後のreadinessは38秒で200へ戻り、OpenWebUI API chatも200/stopで成功した。
- observerでの128-token再pilotは、socket mode 0600、SO_PASSCREDのgateway UID/PID一致、event生成から受信242,344 ns、close送信からcancel観測824,696 ns、progress列`[128]`、cancel-to-release 218,032,745 ns、`reset_complete=true`となった。release後の回復chatは200/eof/lengthで、observer payload 10件は後続journal `MESSAGE`と全件byte一致した。

### 次の行動

1. phase-1独立validatorとrelease evidence collectorを統合する。
2. journal release同期を使い、5段階cancelと各回の即時回復を閉じる。
3. browser Stop 1件と20連続chatの予備運転後、resourceとlatencyの本計測へ進む。

### OpenWebUI Stopを覆う更新通知

- 最初の実browser Stop pilotは、生成中に右下へ表示されたOpenWebUI更新通知がStopボタンを覆い、Playwrightのクリック安全性判定が10秒でtimeoutした。同じrequestはgatewayの256 token lengthまで正常完了したため、SQ8/worker/gatewayの失敗ではない。
- OpenWebUIのlocal sourceから`ENABLE_VERSION_UPDATE_CHECK`と通知の表示条件を確認した。本配備はbase image digestとpatch hashを固定して手動更新するため、Composeでこの確認をfalseにした。
- containerを再作成後、health=healthy、`/api/config` false、再作成前JWTでの認証200を確認した。gatewayは同じPIDで`NRestarts=0`のままである。
- 再試験では実Stop clickが完了し、gateway cancelはclick完了から3.52 ms後、Socket.IO cancelは13.6 ms後に観測した。gatewayは`outcome=cancelled` / `client_disconnect` / `reset_complete=true`でreleaseし、正常doneとcancel後contentは0件だった。診断runは意図的にrelease controlを渡していないため、復旧turnの前にtimeout終了した。次はobserver releaseとcontrolを同期するformal orchestratorで同一chat回復まで閉じる。

### OpenWebUI 20 chat予備運転

- `firecrawl-playwright-service`を1 containerずつ起動し、temporary chatを直列20回実行した。全件で期待本文`BROWSER_OK`、model表示、page error 0を確認し、失敗は0件だった。
- gateway journalは20 admission / 20 releaseが一対一で、全releaseは`outcome=stop`、`completion_tokens=4`、`reset_complete=true`だった。次のadmissionは常に前requestのrelease後で、cancel/busy/fatalはない。
- 終了後もgateway PID 712139 / `NRestarts=0`、OpenWebUI health=healthyである。これは製品安定性の予備運転であり、formal release bundleでは専用のredacted browser actionとauthoritative journalを同時収集し直す。

### Phase-1 collectorの独立review

- subagent初版はcollector 16 tests、validator/HTTP client込み42 tests、Ruff、`py_compile`を通過した。一方で独立reviewにより、実HTTP clientが`shutdown_complete`を返すのにcollectorが`shutdown`を求める契約不整合を検出した。fake client testも誤ったcollector側に合わせていたため、実campaignは最後に必ず失敗する状態だった。
- その他、journal `_PID`のactive gateway未結合、final probe後のartifact copy中のfatal取りこぼし、任意HTTP client/fake event emitterの受理、任意fixtureでもresource gateを通せるvalidator不足、HTTP時刻順序未検査、restart hookの一括展開、artifactの後段TOCTOUを指摘した。
- 実GPU campaignの前に、actual client process統合test、raw request bodyの独立再構築、segment別gateway PID結合、final seal順序、single-fd secret/artifact処理を修正する。修正前collectorはcommitしない。

### Formal OpenWebUI Stop gate結果

- `tools/run-openwebui-stop-gate.py`で実browser、Socket.IO、gateway lifecycle observer、authoritative journal、release control、同一chat回復を一つのgateに統合した。token/scriptはsingle-FD private snapshot、browser imageはcontent ID限定、出力はatomicである。
- OpenWebUI 0.9.4が出す`chat:active`と`chat:outlet`は状態通知であり、本文0 byte / `done=false`の場合だけ受理する。この実動作差を検出するまで、失敗runは全てpass artifactを残さずcleanupした。
- 最終run `/home/homelab1/datapool/openwebui-stop-formal-pilot-20260711-180303` はexit 0。target cancel/releaseと回復stop/releaseの2 request、maximum active 1、restart 0を確認した。click→cancel 2.95 ms、cancel→release 5.14 msだった。
- observer/journalは11/11件で全byte一致、cursorは全一意、PID/unit/bootは単一だった。screenshot SHA-256は`d68acebb01b7ef9acfeeca0b32064e17366f822bab3753dca3f2fdc2015e236b`で、更新通知に覆われないStop controlを目視確認した。

### Phase-1 collectorの修正完了

- 実HTTP clientの`shutdown_complete`とcollectorを整合し、早期/非0終了、ack後extra eventを拒否した。localhost fake SSE serverに実clientを接続する統合testでBearer、request raw、event順序、completion、shutdownを確認した。
- 任意`client_command`を廃止し、derived image content ID、`open-webui-network`、`172.20.0.1:8000`、single-FDで固定したclient/key snapshotからDocker commandを構築する。`--interactive`、UID/GID 1000、read-only、cap-drop、PID/memory上限を固定した。実派生imageと実network/keyの`ready -> shutdown_complete`も成功した。
- validatorはresource positive bodyの同一fixture、sampling type/value、warmup/normal/restart、3 negativeの固定overflow/malformed body、HTTP時刻順序、release後admission、gateway PID epochをrawから再構築する。resource metric windowと重なる別phaseのHTTP/lifecycleも拒否する。
- restart hookは逐次parse、64 records、1 record 256 KiBに限定した。artifactはstaging後もidentity/hash/secretを再検査する。`SHA256SUMS.incomplete`はpost-seal drain/probe/drainが成功するまでrenameしない。
- 二段階の独立改変監査で、短いoverflow偽装、PID/probe同時改変、final中fatal、negative後admissionのphase偽装、resource gapのforeign trace、SHA公開前異常などを実bundleで再現し、全て負testで閉じた。
- 最終のcollector/validator/HTTP client関連76 tests、Ruff check/format、`py_compile`、`git diff --check`は合格した。Phase 1は引き続き`release_status=incomplete`であり、最終releaseを主張しない。

### Collector readiness URLの是正

- direct cancel gateのpreflightで、phase-1 collectorの固定readiness URLが`/ready`になっていることを検出した。現配gatewayは`/ready`=404、`/readyz`=200であり、実campaignは必ず失敗する不整合だった。
- commit `492967a`で固定URLを`http://172.20.0.1:8000/readyz`に是正し、旧`/ready`をconfigで与えると拒否する回帰testを追加した。collector 33 tests、Ruff、format、`py_compile`、`git diff --check`は合格した。

### Formal OpenWebUI 20-chat gate結果

- commit `2092eab`のformal browser soak gateを実行し、temporary chat 20/20、browser actions 100、Socket.IO events 300、page/cancel/provider error 0で合格した。
- gatewayは20 admission / 20 release、maximum active 1、全件`stop` / `reset_complete=true`だった。observer/journalは100/100 byte一致、gateway PID `712139`、restart 0のままである。
- atomic bundleは`/home/homelab1/datapool/openwebui-soak-formal-20260711-193714`、summary SHA-256は`8f027aacbfc70347f1a648f0bedeca1b7b40e0e9e1a6e223a2d3a6814ca4f3aa`である。

### Formal direct cancel gate結果

- commit `9539d98`の4段階direct cancel gateを実行し、4 target + 4 recoveryの8/8 requestが合格した。maximum activeは1、observer/journalは55/55 byte一致だった。
- cancel→releaseはstarted 149.386 ms、prefill128 218.734 ms、prefill2048 1.297 s、decode first-content 32.069 msで、全件が5秒以内の`client_disconnect` / `reset_complete=true`だった。各回の後続2-token回復も成功した。
- atomic bundleは`/home/homelab1/datapool/sq8-direct-cancel-formal-20260711-194417`、summary SHA-256は`60e6f49158ed2da54a54cc072c0b257547f668db71ffeca578486ed81c4dc646`である。

### Formal post-header failure gate結果

- commit `35908f1`で、部分本文表示後のworker PIDをpidfdで停止し、OpenWebUI失敗表示、systemd自動再起動、Docker network内readiness、同一temporary chatの回復を一つのatomic gateにした。
- 1回目は実worker fatalとrestartには成功したが、`journalctl -u`に含まれるPID 1のsystemd行をgateway行として誤検証して失敗した。commit `5f4c359`で`_SYSTEMD_UNIT=init.scope` / `UNIT=ullm-openai.service`の管理行を厳密に分離し、実journal 14行の再生testを通した。commit `83f1236`では秘密を出さない固定内部エラーだけを診断表示するようにした。
- 正式bundle `/home/homelab1/datapool/openwebui-failure-formal-20260711-195916` は合格した。summary SHA-256は`920486a8f209710ff2bfb10113cc09120ccbf0ee4e506f842d5755aef815bc35`である。
- signal→fatal 29.688 ms、fatal→UI error 19.009 ms、error→cancel 8.217 msだった。error 1、cancel 1、error後done/content 0、page error 0で、部分本文と失敗表示を目視確認した。
- restartは1→2で差分1、fault→readinessは53.753秒だった。同一chatの別messageは`FAILURE_RECOVERY_OK`に一致し、done 1、error/cancel 0、gatewayは`stop` / `reset_complete=true`でreleaseした。
- 最終gateway PIDは`1452201`、worker PIDは`1452625`で、OpenWebUI/gatewayはhealthy、observer socketと一時containerは残っていない。

### 次の行動

1. standalone formal bundle 4種はrunner回帰用pilotとして保持し、最終証跡へ単純結合しない。
2. 合成rawで完全phase順とgateway epochを独立validatorへ先に実装する。
3. 検証済みrunnerを同一campaign内で`soak20→cancel4→Stop→normal100→failure/restart→restart20`の順に再実行するorchestratorを作る。
4. HTTP TTFT/decode matrix、最終seal、独立publicationを閉じる。

### 単一campaign基盤の実装

- commit `2d7fafc`で、9 phaseの完全順序、OpenWebUI smoke+20、5組のcancel/recovery、唯一のfault/restart、normal/restart PID epochを検証する独立helperを追加した。
- commit `b359efb`でformal failure gateをcollector hookへ適応した。gate/browser sourceとimage/service入力をbundle hashへ結合し、action 0..3→fault→action 4..8の実時系列で10 recordsを出す。16 testsと既存formal bundle再検証が合格した。
- commit `196a955`で、5 TTFT fixture各2 warmup+10 measuredとdecode64の2+10、合計72件を実行するHTTP latency gateを追加した。raw chunk時刻、first-content close、64 content/63 interval、exact percentile、restart epoch、observer/journal相関を35 testsで固定した。
- commit `a11e8fe`で既存exact20 browser runnerのdefaultを維持し、明示時だけsmoke 1件→soak 20件を同一browser processで実行するcombined modeを追加した。combinedは21 chats/105 actionsとexact scheduleを証跡へ残す。
- commit `1f706e5`と`2dae4cd`で、campaign開始cursorからfinalまでのglobal journalを専用threadでstreaming保存するcaptureを追加した。2048 lifecycle/16 MiBの有界queue、bundle/resource claim、唯一のPID切替、secret scan、phaseを進めないnegative quiet window、run_end後row拒否、exclusive publicationを21 testsで閉じた。
- commit `0b9aa69`でresource normal/restartを`ResourceSegmentCollector`へ抽出した。phase-1はadapter経由で従来のraw順序を維持し、full campaignはcontinuous journal adapterを差し替えられる。collector/validator 89 testsが合格した。
- commit `c3fc266`でresource requestのHTTP応答から得た`completion_id`をcontinuous journal claimへ明示的に渡すadapterを追加した。異なるcompletion/PID、同じcaseの二重claimをsessionへ一部書き込む前に拒否し、collector 42 tests、Ruff、`py_compile`が合格した。
- commit `178a5bf`で実`CampaignJournalCapture`からresource adapterと`SessionWriter`までの統合testを追加した。関連113 testsが合格し、global journalの4 lifecycle rowsが連番のcampaign session recordになることを確認した。
- commit `75d3216`で、`GET /v1/models`、認証3種、query、malformed/duplicate JSON、`n=2`、不明modelの固定10件だけをOpenWebUI network内から検査する非GPU API contract gateを追加した。exact status/message/header/body、auth-before-parse、observer/journal lifecycle 0件、各case後identity、gateway source、secret scan、FD固定exclusive publicationを28 testsとgateway契約4 testsで閉じた。実Docker/service gateはfull campaignまで実行していない。
- commit `379bdeb`で、API gate moduleをimportせずraw HTTPから同じ10件のmethod/auth/body/status/header/model/errorを再構築する独立validatorを追加した。API phaseのworker lifecycle 0件も完全campaign順序helperで必須化し、関連154 testsが合格した。
- commit `eafa227`でcombined smoke+20 bundleを105 browser actionsと105 journal claimsへ変換するingestを追加した。実pilotとrunnerを照合し、`browser/openwebui-soak-summary.json`のみ`0400`、他を`0600`とするexact layoutに修正した。全action単調時刻、source/image/service/boot/PID/uid/gid/restart、observer/journal byte一致、実`CampaignJournalCapture`でのclaim、TOCTOU、secret scanを検証し、未実体stderrは0 bytesだけに限定した。
- commit `398e98d`で非GPU API contract gateの6-file bundle ingestを追加した。raw HTTP 10 caseを再検証して40 campaign recordsへ変換し、13 quiet checks、journal lifecycle 0件、source/service/image/network identity、producer summary、独立SHA256SUMS、FD固定seal、secret非混入を検証する。専用と既存gateの44 tests、Ruff、strict mypy、`py_compile`が合格した。実Docker/service gateはまだ実行していない。
- commit `6ed6a3a`でresource計測用`SystemRuntimeConfig`をphase-1のartifact設定から分離し、深い不変snapshotにした。`capture_journal=False`では内部journal readerを生成せず、HTTPとresource計測だけをglobal campaign journalへ接続できる。起動失敗時とreader/client closeも補強した。collector/campaign 72 testsと関連182 tests、Ruff、format、`py_compile`が合格した。collector全体のstrict mypyには既存51 errorsが残る。
- commit `d008fd5`でfull campaignのprivate stagingと一度だけのatomic publication境界を追加した。component workを証跡外に隔離し、独立validatorと同じ20入力path、private mode/owner、secret scan、stream copy、source/destination TOCTOU、validation前後のexact layout、既存destination非上書きを検証する。12 tests、Ruff、strict mypy、`py_compile`が合格した。
- commit `fd86d1a`で最終directoryへrenameした後のparent `fsync`失敗時に、同一inodeを確認してstaging名へ戻すrollbackを追加した。失敗runは公開名を残さず、bundle管理13 testsが合格した。
- commit `9f56b5d`でdirect cancel 4 target + 4 recovery bundleを32 HTTP session recordsと55 lifecycle claimsへ変換するingestを追加した。exact source/fixture/body/schedule、observer/journal/correlation、service epoch、producer summary、TOCTOU、secretを再検証し、global journalで55 claimsを全消費した。関連108 testsが合格し、旧pilotはcollector source SHA不一致のため条件付き回帰だけをskipした。
- commit `41b8ed6`でrestart epoch上のHTTP latency bundle ingestを追加した。72 requestのraw HTTP/SSE時刻、lifecycle、observer/journal相関、TTFT/decode64 samplesとpercentile、manifest、summary、SHA256SUMSを再構築する。全claimsのglobal journal消費、link/source置換拒否を追加し、専用14 tests・関連112 tests、Ruff、strict mypy、`py_compile`が合格した。
- commit `b45ea7e`で`environment.json`と`model-identity.json`のpreflight generatorを追加した。live OS/kernel/systemd/Docker/ROCm/GPU/service/OpenWebUI、full-payload promotion receipt、artifact/package manifests、tokenizer、worker binary、tracked sourceをFD固定で束縛し、巨大payloadは再読込しない。専用12 tests・関連40 testsとread-only実機captureが合格した。full orchestratorはcampaign前にfull promotion validatorを実行する必要があり、oracle/toolchainと今後のsourceは完全source-state統合時に追加する。
- commit `e959290`でpreflight identityへ固定argvから取得するPython・rustc・Cargo version lineと、`tokenizer_config.json`のchat template UTF-8 byte数・SHA-256を追加した。関連40 tests、Ruff、strict mypy、`py_compile`が合格した。vLLM oracleと今後追加するrunner/adapter sourceはfull orchestrator統合時に閉じる。
- commit `d9a203c`でpreflight identityへtracked serving fixture manifest、chat-template manifest、runtime oracle validationを追加した。oracleが記録したvLLM package/version・execution/device identityをmodel identityへ束縛し、chat template hashとmanifestも相互照合する。関連40 tests / 30 subtests、Ruff、strict mypy、`py_compile`と実tracked manifestのread-only確認が合格した。
- commit `06c5585`でOpenWebUI Stop gateの専用ingestを追加した。正式pilotのexact 6-file bundle、9 browser actions、11 lifecycle claims、target→recovery順、source/service epoch、observer/journal byte一致、PNG signature/CRC/IDAT展開を再構築する。screenshotは再encodeせず最終bundleへstream copyできる。正式pilotはsource一致でskipなしに合格し、関連6 suitesは130 tests / 33 subtests、Ruff、strict mypy、`py_compile`が合格した。
- commit `6d13836`でpost-header failure gateの専用ingestを追加した。producerの`passed`を信用せず、fault/readiness/browser/journalからsummary全体を再構築する。旧PID 5 claimsと新PID 5 claims、fault、restart probe、9 actions、failure screenshotをcampaign recordへ変換し、PNGはframing/CRC/IDAT zlib/scanlineまで逐次検証する。正式pilotはgate/hook/browser source全てが現行と一致し、関連7 suitesは144 tests / 75 subtests、Ruff、strict mypy、`py_compile`が合格した。
- commit `f24942d`でfailure browser actionsのcase割当を実時系列に合わせ、0..4をfailure、5..8をrecoveryへ分離した。最終rawから失敗表示5 actionsと再起動後回復4 actionsを独立再構築できる。関連71 tests / 26 subtestsとRuff、strict mypyが合格した。
- commit `a98a730`で`environment.json`と`model-identity.json`の独立validator helperを追加した。generatorをruntime importせず、canonical完全schema、source role/group aggregate、promotion receipt、artifact/package、tokenizer/chat template、oracle/vLLM、worker、GPU、OpenWebUI image/networkのfrozen v0.1値を再構築する。session header、initial probe、run_endとのcross-check APIも追加した。72 tests / 38 subtests、Ruff、format、`py_compile`は合格した。strict mypyは追加error 0で、既存baseline 38 errorsは不変である。
- commit `e16f0c8`で、各ingestorとresource rawから最終の派生6成果物をcanonical JSONとして生成する境界を追加した。resource 610 samples / 4 GPU metricsを逐次読込し、baseline、最終signed delta、全pair Theil-Sen、process countを再構築する。独立reviewで検出したfrozen command/tool版数の完全一致と`stop|length|cancelled`のoutcome契約も是正した。関連suiteは85 tests / 52 subtests合格、1 pilot source mismatch skip、Ruff、strict mypy、`py_compile`が合格した。
- commit `b088862`で、独立validator前の20入力をdirfd / `O_NOFOLLOW`で開き、inode・mode・owner・link・size・mtime・ctime・streaming SHA-256を一度だけsealするようにした。validator後とrename後に入力全件を再照合し、`release-validation.json`もvalidatorが返すbytes/SHAと結合する。同一size改変、rename直後改変、validation差替え、abort二次失敗をnegative testで拒否した。
- commit `8116673`で、preflight→API→smoke+20→direct cancel+Stop→normal resource→post-header failure/restart→restart resource→latency→finalの9 phaseを一つのcontinuous journalとatomic bundleで管理するfull orchestratorを追加した。新旧PID epoch、4 probes、2 PNG、raw commit/journal seal、derived、validator、publish順を固定し、失敗時は全runtime ownerのcleanupを必ず試行する。production backend未配線のCLIはexit 2でfail closedのままである。関連77 tests / 37 subtests、Ruff、strict mypy、`py_compile`が合格した。
- commit `7f75eeb`で、派生6種、17件の`release-matrix.json`入力、19件の`SHA256SUMS`入力、非判定の`summary.md`を決定論的に作る最終rendererを追加した。既存11 filesはdirfd / `O_NOFOLLOW`で二重stream hashとidentity再照合を行う。resource normalの実HTTP/releaseからsampling index 5..100/5の20件を`ResourceSegmentResult`へ明示的に返し、実orchestrator contextからrendererへ接続した。関連78 tests / 22 subtests、Ruff、strict mypy、`py_compile`が合格した。
- commit `5d8caae`で、完全campaignのraw session順序、API結果、HTTP request/response、SSE、browser action、faultを独立validator内の有界projectionとして保持する基盤を追加した。response本文は1 active分だけに限り、終了時に時刻・status・bytes/hash・SSE content/ID hash・usage存在をcompact化して解放する。SSE item数、record数、ID、selector、header件数/名/値/総bytesを固定上限で拒否し、空chunk、phase欠落、第2 fault、browser時刻/index改変をnegative testで閉じた。92 tests / 43 subtests、Ruff、`py_compile`が合格し、strict mypyの新規診断は0である。
- commit `0cface8`で、identity generatorと独立validatorのsource契約を63 roleへ拡張した。full campaignのbundle/orchestrator/renderer/views、全gate/ingestor、3 specs、5 TTFT fixtures、2 independent derived modulesを明示groupと`all`へ結合し、TTFTの固定bytes/SHAも独立検証する。public helperはtrusted commitの`git cat-file blob`を有界streamし、dirfd / `O_NOFOLLOW`で開いた対象worktree fileとenvironment source/group aggregateを照合する。無関係なdirty/untrackedは拒否せず、header `input_files`がsource全件を含むこともcross-checkする。関連111 tests / 74 subtests、Ruff、identity strict mypy、`py_compile`が合格した。
- commit `8736cf9`でnormal/20-chat/Stop recoveryのbrowser応答を`trim()`後の固定marker完全一致へ強化した。従来の部分一致では、別本文を含む応答でも成功証跡になり得たためである。Node syntaxとbrowser静的12 testsが合格した。
- commit `c66cb92`でsession/resource JSONLをASCII・key順固定のcanonical JSONへ統一した。JSON escape後のbyte走査だけでは引用符・backslash・controlを含むcredentialを見逃すため、JSON key/valueの意味上のUTF-8も有界再帰走査し、raw lineはstrict decode後にも同じ検査を行う。collector 51 tests / 13 subtestsとproducer view testsが合格した。
- commit `b6ac463`でAPI、sampling、cancel、OpenWebUI smoke/soak browserの最終成果物をproducer importなしでraw projectionから再構築する独立moduleを追加した。browser action全123件、成功marker、fault command、PNG framing/CRC/zlib/scanlineとFIFO置換を検証し、API quiet 13件を転記journal列へ結合する。専用16 tests / 19 subtestsと関連suiteが合格した。
- API quiet checkの13という件数だけを派生成果物へ固定していた不足を修正中である。componentのservice journal全列をindex/cursor/time/PID/MESSAGE bytes/hashとして最終rawへ転記し、quiet累積件数とglobal journalの完全連続区間へ照合する実装・negative testsは合格した。full validator接続と同じcommit単位で保存する。
- metrics独立reviewで、latency 72件のrequest body bytes/SHAが固定TTFT fixture・decode requestへ未結合であることを検出した。実GPU campaign前に、5 fixtureとdecode64のcanonical body identity、resource fileのFIFO置換拒否を修正する。
- commit `76c72b4`でlatency 72件とresource 1+610+4件をproducer/validator importなしで再構築する独立metrics moduleを追加した。tracked 5 TTFT fixtureとdecode64の6 canonical request body identity、HTTP/SSE/lifecycle、resource identity/window/threshold、O_NONBLOCKでのFIFO置換拒否を検証する。専用7 tests / 11 subtests、strict mypy、Ruffが合格した。
- commit `5efb771`でgatewayの`request_first_token`が最初の非空SSE content観測より後になる改変をTTFT/decode双方で拒否した。専用8 tests / 13 subtestsとstrict mypyが合格した。
- commit `9e3d200`で正式latency ingestor fixtureが持つ同一SSE object内の`finish_reason=length`とusageを独立metricsが過剰拒否していた不整合を修正した。latency/resourceとも`content < finish <= usage < [DONE]`とし、cardinality、usage object/token count、終端は維持した。専用11 tests / 13 subtestsが合格した。
- commit `5ad12cf`でAPI quiet 13件の元service journal全列を最終rawへ転記し、global journalの完全連続区間へ照合した。full validatorはidentity/source checkout、session/resources、独立6成果物、summary、matrix/SHAを再構築し、最後だけ`release-validation.json`を0600/O_EXCLで作る。BaseException時も同一inodeだけを削除し、validation残留を防ぐ。
- 同commitの完全fake campaignは、実component ingestor群→raw bundle→producer renderer→独立validator→validation→atomic publishをGPU/serviceなしで完走した。derived、quiet、journal、source、headerの代表改変5種は全てvalidation fileを残さず拒否した。最終の境界横断確認は253 tests / 165 subtests合格である。
- 実機campaign前の残作業はproduction backend配線である。`run-sq8-full-openwebui-campaign.py`のCLIは現在も意図的にexit 2し、既存gate/collectorを実引数・secret・service epoch・renderer・validatorへ結合していない。read-only preflight付きでこの境界を閉じてからだけR9700 campaignを実行する。
- commit `27689fe`で、長時間campaign中に最終出力名が作られる競合を検出し、公開と失敗時rollbackを`renameat2(RENAME_NOREPLACE)`へ変更した。競合先を置換しないnegative testを追加し、promotion validatorはartifact/package root自身も非symlink・非書込directoryと確認する。関連24 testsとRuff、`py_compile`、差分検査が合格した。
- commit `cbe1b8e`で、promotion validatorがimportする`tools/sq8_canonical_artifact.py`もenvironmentのsource roleとmodel identity receiptへ結合した。validator本体とcanonical helperのどちらが改変されても独立に拒否する。関連122 tests、完全fake campaign 2 tests、Ruff、identity strict mypy、`py_compile`が合格した。
- commit `950ec40`で、root:execution-gid/0640のAPI keyとexecution-user/0600のOpenWebUI tokenを単一FDで固定するsnapshot、0700 directory内の0600 master、両secretのraw/semantic JSON/chunk跨ぎscanner、非blockingの全campaign `flock`を追加した。FIFO停止、multiline/oversized JSON、大規模nodeのOOM、lock取得中のpath差替え、tamper時のsecret/FD残留をnegative testで閉じた。orchestratorとcollectorの75 tests、Ruff、strict mypy、`py_compile`、独立security reviewが合格した。production配線ではlock pathをcanonicalに固定する。
- commit `017f6c6`で、serial resourceの固定chat fixture、sampling、25/50/75 request後のoverflow/malformed negative、canonical config、environment全sourceと生成2入力のseal、session/resource headerを純粋関数で構築した。実collector型と独立validator APIで再検査し、source重複・順序・fixture・model・sampling・schedule・threshold・runtime identity・secret改変を拒否する。24 testsとRuff、strict mypy、`py_compile`が合格した。
- commit `73486fb`で、明示40hex HEADとNUL区切りGit porcelain statusのexact bytesを固定し、HEAD blobからpromotion validatorとcanonical helperのみを0700/0600 private snapshotへ復元するread-only preflightを追加した。full payload validatorをbounded subprocessで実行し、duplicate/nonfinite JSON、stderr/nonzero/timeout/oversize、source/product drift、symlink、`KeyboardInterrupt`中のcleanupを検査する。18 testsとRuff、strict mypy、`py_compile`が合格した。
- commit `625e1be`で、systemdのactive/running/success・restart/start-limit/900秒age、OpenWebUI container/image/network/health、gateway/OpenWebUIのexact GET body、observer path不在、R9700のAMD+/proc/KFD単一worker所有を前後で確認するread-only operational probeを追加した。commandはabsolute allowlist、sanitized env、実行中stdout/stderr/time上限で拘束し、mutation argvを構成できない。28 tests、ResourceWarning error、Ruff、strict mypy、`py_compile`が合格した。
- commit `4716fe1`で、production、operational、resourceの3 moduleとoperationalが再利用するworker-acceptanceをsource contractへ追加し、campaign identityを68 sourceへ拡張した。identity/validator 122 tests、新規preflight 55 tests、完全fake campaign 2 testsが合格した。
- commit `e62ec15`で、host firewallにより直接到達できないGateway `/readyz`を、既存OpenWebUI containerの検査済みPIDが持つnetwork namespaceから確認するread-only経路を追加した。固定`sudo -n nsenter`後にUID/GID 1000へ権限を落とし、container生成・`docker exec`・任意script/URLを許さない。container discoveryから前後identity検証までの37 testsと実機read-only preflightが合格し、Gateway/OpenWebUI 200、AMD/KFD VRAM一致、positive KFD ownerがworker 1 PIDのみと確認した。
- commit `018390d`で、campaign flockの取得直後またはpost-lock identity再検査中の`KeyboardInterrupt`で、lock・descriptor・parent descriptorを必ず解放して元割込みを再送出するよう修正した。orchestrator、operational、collectorの113 testsが合格した。
- commit `f74526c`で、固定deployment path/OpenWebUI/R9700/tool契約、GitAnchor前後再検査、full promotion receipt、worker SHA/service epoch、独立identity/source checkoutを一つのsecret-free production identity preflightへ合成した。prepare自身をsource roleへ追加し、69 sourceに拡張した。関連193 tests、完全fake campaign 2 tests、Ruff、strict mypy、`py_compile`が合格した。
- gatewayはPID `1452201`、workerは同epoch、`NRestarts=2`のまま稼働し、OpenWebUI healthも正常である。ここまでの新規testではGPU requestとservice restartを実行していない。

### 次の行動

1. API keyとOpenWebUI tokenのprivate snapshot、複合secret scanner、全campaign lockの安全primitiveを保存する。
2. HEAD固定promotion検証、Git/source identity、service/OpenWebUI/GPU状態をread-only preflightに配線する。
3. `--preflight-only`がrequest・signal・restart・gateを一切実行しないことをnegative testと実機の読取りのみで確認する。
4. 固定HEADでR9700実機campaignを1回実行し、独立検証とatomic publication後にOpenWebUI製品releaseを確定する。
