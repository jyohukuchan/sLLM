# AQ4 P2 resident profile diagnostic ready v0.1

## 前回の要点

通常のone-case maintenance chainはサービス停止、zero-owner gate、launcher実行、外側`finally`復旧を保証していたが、ROCTx markerとrocprofv3 captureを信頼鎖へ束縛していなかった。

## 今回の変更点

runner `e93a2c162eb059cb2db883953d331f7a158d3a16`、validator `82635456825503c535ce0b662e72a7a233d18c40`、B `7e59baee0c1ac93a350da58a4292a84fbfde9f1c`、launcher `eec6922fa9c90267213d2749c5dc816be54de527`、harness `7f1ad5c1785c50d99e0952afa7ce018a0204d29c`の順に再固定した。停止資源pollは開始時に固定した30秒のmonotonic absolute deadlineを使い、観測・probe前後の期限確認、2秒と残時間の小さい方へのsubprocess timeout制限、probe間の期限付きsudo keepaliveをbase/profile共通で適用する。

profile-ready artifactは`resident-one-case-smoke-profile-ready-v1`である。`ready-binding.json` SHA-256は`b1f2fff89a123fd6bc88ac5a35d61df0eeda116e0137d83d1e9e3bdfafe3e2ed`、`harness-trust.json`は`06aa87641e605897b74f0931b37b464236f23007376e2366bbb9238650233f76`、`qa-attestation.json`は`bac8dcc71368d5463bc9e6d6589236014560129292d0b69700c4246bb68cf15c`、target manifestは`61d6db729023e2cbf62737c25c7c9042fda8431ac7b108606ce577998d3d58f4`、`SHA256SUMS`は`73d7dcba5fe2c62a609afaaa175a6be627e5b8c96b5724ba1a82db0d0464671a`である。

profile modeの事前・復旧確認もbase modeと同じcontainer名前空間の正式health gateを使う。固定済みDocker、OpenWebUI container/image/network、container curlを検証し、認証済み`/v1/models`のAuthorization headerはstdinだけで渡す。host直結HTTPは診断に限定する。

artifactは`execution_mode=profile_diagnostic`、actual one-case最大1回、measurement/promotion不可、全output no-reuseである。profile以外の直接実行を拒否し、capture tool、profiler、Python、launcher、target manifestのpath、identity、SHA-256とexact commandを`before-start`、`capture-before`、`capture-after`、`finalize-before`の4段階で検査する。

実行所有順は`outer maintenance harness -> capture tool -> rocprofv3 -> launcher -> runner`である。harnessがサービスを停止してからcaptureを子process groupとして起動し、timeout、kill、capture起動失敗、launcher失敗、子残留の全経路で外側`finally`が復旧を試行する。capture toolはcommit `b4d515f9908136fa773f957775beab79edc3065d`、tree `228bbbd0d05b8055640bd47dd3ed95952e504eef`、Git blob `5197f7a2607da2ec281ab8a013ce1476178bf1b1`、SHA-256 `605a68d308bf4336fc96d23d0ba9f819799ef24b169e3f49ae6a377638ab6cf8`である。profilerは`/opt/rocm-7.2.1/bin/rocprofv3`、SHA-256 `13060810d6b80653631b14f0f5e33ea160c2b79a6a3a4c6850142010b48b8ec8`である。

target manifestはexact launcher argvを固定し、absolute argvのうちPythonとlauncherを入力file SHA-256、launcher evidenceとrunner outputをfresh output pathとして重複なく分類する。capture commandはmanifest raw file SHA-256を引数でも受け取り、manifest内のself-hashと二重検証する。profilerとmanifestは同一read-only FDをinitial、spawn直前、終了後に再検証する。

launcherはrunnerへ`--profile-roctx-ranges --roctx-library /opt/rocm/lib/libroctx64.so.4 --roctx-library-sha256 22bbc6946fdf5d7d8b1755cbd738c42a63f3795d18ac3ed1285b09cc772dee17`を`--driver-command`より前に透過する。resolved libraryは`/opt/rocm-7.2.1/lib/libroctx64.so.4.1.70201`へ固定する。launcherは12個のbalanced range、同一PID/thread、audit SHA、run ID、resident session ID、case ID/SHA、library invocation/resolved path/SHAを検査する。

capture outputは`p3/aq4-p3-diagnostic-rocprof-capture-v1`、resident summary/rawはprofile専用runner outputへ固定した。capture toolはrawのsession IDとmarkerのsession ID、run ID、case ID/SHAを照合し、warmup 0–1を除外してmeasured 2–11を分割する。

canonical dry-runは`resident-one-case-smoke-profile-ready-dry-run-v1`である。evidence SHA-256は`c13b18dac2eeebce48eb09533f21162e73de8f7be58068c214698b4f7022c822`、`SHA256SUMS`は`7b97ce657de200300eaaafeedf789a65a9ff22c91fcc2973da4c6898b649d689`である。sudo/keepalive、stop/start、launcher、rocprof、capture tool、docker、docker exec、container curl total/version/endpoint、stopped-gate poll/probeのprocess countsは全て0で、service/GPU/modelは未操作である。

回帰は主要セット194 tests、marker chain 55 tests（25 subtests）、diagnostic capture 11 testsが通過した。capture関連集合85 testsと独立marker QAの手動境界15件も通過している。

## 次の行動

profile diagnostic actual runには別の明示承認と、artifact内のexact capture commandを使う。同一PTY sudo cache、rocprof wrapper、fresh output、pre-stop/live gateのいずれかが不成立なら実行しない。取得結果は性能昇格に使わない。
