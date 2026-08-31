# Phase 48: minimal WebUI prototype implementation history

## 結果

2026-08-30に、sLLMの既存公開HTTP APIだけを利用する最小WebUI prototypeを`webui/`へ追加した。単一画面でmodel選択、
chat stream、separate reasoning、generation設定、runtime要約、model load／unloadを扱う。既定はdeterministicなsafe demoで、
利用者がendpointとbearer keyをmemory stateへ入力した場合だけlive serverへ接続する。

2026-08-31の追加指示でsource treeの起動を統合した。現在は`sllm-server`の標準起動がlocal WebUIをdefault enabledで起動し、
runtime API URLを受け取った画面が自動でlive接続する。standalone／hosted buildではruntime設定を空にして従来のsafe demoを維持する。
credentialは統合設定へ含めず、認証済みserverでは従来どおりbrowser memoryだけのuser／admin key入力を使う。
WebUI childは親環境をclearして`PATH`と非secretのAPI URL／integration markerだけを受け取り、API key用envや`HF_TOKEN`を継承しない。

prototypeはPhase 48のUI方向とAPI接続境界を確認する成果物であり、Phase全体の完了、production service、live API／GPU実行、
correctness／品質／性能evidenceではない。

2026-08-30の追加指示により初回表示をライトテーマへ変更し、ヘッダーに明示的なライト／ダーク切替を追加した。ライト用のsurface、
border、text、status、reasoning、composer tokenを独立定義し、ダークテーマも任意選択として維持する。選択はsession内だけに作用し、
reload時はユーザー指定どおりライトを既定にする。

2026-08-31の追加指示により、chat中心の構成からGPU／throughput管理中心の構成へ切り替えた。default viewはGPU identity、
model residency、prefill／decode token/s、TTFT、token window、直近runを表示するperformance dashboardとし、chatはnavigationから開く
secondary viewとして残した。右inspectorはdashboardではbenchmark prompt／model／output token設定、chatでは従来のgeneration設定を表示する。

live benchmarkは1件のstreaming Chat Completions requestの前後で`GET /metrics`を取得し、model／`stream="true"`のcounter差分から
`prefill = prompt_tokens / TTFT`、`decode = (completion_tokens - 1) / (E2E - TTFT)`を計算する。TTFTはqueue、prefill、first deltaを含むため
kernel-only速度ではなくserver metric由来の推定値と表示する。successful request差分が1でないwindow、不完全なtoken／timing差分、
`/metrics`未提供はfailureにする。同時failed request等の混入を完全には識別できないため、UI上でもidle時の単独runを要求する。

起動統合時に`/props.hardware`へserver側HIP device queryのGPU vendor、device name、gfx target、VRAMを追加した。照会できない場合は
未提供と表示し、browser WebGPUをsLLM実行GPUの代用にしない。demoのR9700／gfx1201とthroughput値は引き続き明示的なfixtureであり、
GPU／性能evidenceではない。

## 実装した契約

| 面 | prototypeの実装 |
|---|---|
| server discovery | `GET /healthz`、`GET /readyz`、`GET /v1/models`、`GET /props` |
| performance | `GET /metrics`の単一request前後差分によるprefill／decode推定、TTFT、token window、直近run |
| GPU identity | server側HIP device queryの`/props.hardware`だけをlive表示し、照会不能時は未提供を明示 |
| generation | `POST /v1/chat/completions`、SSEの`delta.content`／`delta.reasoning_content`分離、strict `[DONE]` |
| model lifecycle | admin keyを使う空bodyの`POST /admin/models/{alias}/load|unload` |
| model acquisition | loopback adminの固定`hf` CLI検索、完全SHA付きcommand copy、server側download job |
| credential | endpoint、user key、admin keyをReact memoryだけに保持し、URL／storage／fixtureへ永続化しない |
| failure | HTTP error、SSE error envelope、malformed JSON、`[DONE]`欠落、cancel、empty modelを明示状態にする |
| responsive | desktop 3-column、tablet inspector drawer、mobile model／settings drawer |
| integrated startup | `--webui true|false`、default `localhost:65457`、独立`--webui-port`、runtime API URL注入、graceful process-group shutdown |

demo fixtureにはreviewed Qwen3.5-4B BF16 dense text、gfx1201、standard OCP MXFP8 E4既定、FP16 rollbackという現行計画を反映し、
fixtureであることを画面上に明示した。この初期実装ではserverへhidden API、filesystem access、credential persistence、tool execution、
upload、独自session state machineを追加しなかった。後述のmodel-library admin surfaceだけは、その後の明示指示で境界を改定した。

2026-08-31の追加指示でmodel source選択を起動optionからWebUIへ移すprototypeを追加した。`sllm-server`はmodel sourceを省略して
loopback dynamic modeで起動でき、`POST /admin/model-library/browse|select|rescan`と`GET /admin/model-library`がserver filesystem上の
folder参照、選択、再走査を行う。これはbrowser filesystem APIではなく、loopback listenerでだけ構成されるadmin surfaceである。
credential-free serverではloopback adminを許可し、credentialを構成したserverでは従来どおりadmin bearer keyを要求する。

選択pathは`$XDG_CONFIG_HOME/sllm/model-library.json`、未設定時はuser config directoryへatomic保存し、再起動時に復元する。走査対象は
選択folder直下の最大256件のregular non-symlink `.gguf`である。`general.architecture`が`qwen35`、`qwen35moe`、`gemma4`、
`gemma4moe`のいずれかで、
同名`*.derived-lock.json`、reviewed model lock、GGUF identity、runtime weight-plan、device 0 resident capacityを検証できたmodelだけを
Phase 45 lifecycleへalias登録する。未対応architecture、lock欠落、identity／plan不一致、VRAM超過、alias競合はloadせず理由付きでグレー表示する。
folder変更／再走査時に現在のlibrary modelがresidentなら、先にunloadするよう明示errorにする。
`--metrics true`はmodel未選択のdynamic serverでも起動でき、operator-controlled catalogへ追加されたaliasだけを最大16件の固定seriesへ
登録する。request由来labelによるcardinality増加は引き続き許可しない。

この追加は従来の「filesystem操作を追加しない」prototype境界を、ユーザー承認によりmodel libraryだけへ狭く改定した。推論requestと
既存load／unload routeは引き続きalias-onlyであり、pathをgeneration requestへ入れない。upload、再帰走査、symlink追跡、network model、
tool実行は追加していない。既存`--models`とsingle-model CLIは互換経路として維持する。後続の明示指示でWebUI process起動は統合したが、
model-library／Hugging Faceの全機能を持つ経路は引き続きmodel source省略のloopback dynamic serverである。

その後の2026-08-31の明示指示で、開発者向けHugging Face取得だけを上記境界へ追加した。`GET /admin/hugging-face/status`、
`POST /admin/hugging-face/search|files|downloads`、`GET /admin/hugging-face/downloads/{id}`はdynamic loopback serverのadmin認証を再利用する。
serverに導入済みの`hf` CLIでGGUF modelを検索し、解決済み完全commit SHAでrepository rootの`.gguf`と同名derived lockを列挙する。
WebUIはserver生成のPOSIX commandをcopyでき、download buttonは同じmodelをserver側の選択済みfolderへ非同期取得する。

download APIはcopy文字列、任意destination、token、追加argumentを受け取らず、boundedな`repo_id`、SHA-40、root file名だけを検証して
shellを介さない固定argvへ変換する。同時実行は1 job、状態履歴は最大32件とし、完了時にmodel libraryの再走査を試行する。任意command runner、
任意URL proxy、Phase 47のtool/MCP実行には拡張しない。serverの`hf auth whoami`が未ログインを確認した場合は匿名rate limitと
gated／private取得失敗の可能性だけを警告し、`hf auth login`の案内／copyやWebUI token入力は追加しない。検索結果は未検証候補であり、
download後の実際のload可否は既存GGUF／derived-lock／reviewed identity／weight-plan／capacity検査で決定する。

## 検証

- `npm run typecheck`: PASS
- `npm test`: PASS（SSE parser 4件、API client 2件、Prometheus model／stream metric差分1件の計7件）
- `npm run lint`: PASS（prototype sourceを対象）
- `npm run build`: PASS
- local preview `GET /`: HTTP `200`
- owner-only [private UI preview](https://sllm-console-prototype.jyohukuchan.chatgpt.site): deployment succeeded。safe demoが既定で、
  sLLM backend／GPU／production inference serviceは含まない。
- `npm audit --omit=dev`: high／critical 0、low 1。残件はWindows上のdevelopment serverに関するtransitive `esbuild` advisoryで、
  Linux上のprototype build／private previewおよびbrowser client runtimeの受入blockerとはしない。
- live sLLM／GPU接続: 未実行。safe demoの結果をlive／GPU PASSへ読み替えない。
- model-library追加後: `cargo test -p sllm-server`全件、server全target clippy、WebUI API／metric／SSE test 8件、WebUI
  format／typecheck／lint／production build、markdown local link検査をPASSした。modelなし・metrics有効のloopback起動、
  model-library snapshot／folder browse、graceful shutdownをlocal smokeで確認した。WebUI dev serverは`http://localhost:3000/`で継続し、
  追加deployは行っていない。
- Hugging Face取得追加後: `cargo test -p sllm-server --lib` 101件、WebUI API／metric／SSE test 9件、対象frontend lint、typecheck、
  production build、local preview HTTP `200`をPASSした。一時loopback serverで実`hf` CLIの認証状態、20件のGGUF検索、完全SHAでの
  root file一覧とcopy commandを確認し、存在しないGGUFを指定したjobが`failed`へ遷移してbounded errorを返すことも確認した。
  実際の数GB model downloadとlive GPU loadは実行していない。
- 起動統合後: `sllm-server` binary test 18件、WebUI test 12件、typecheck、lint、production buildをPASSした。一時dynamic serverを
  API `127.0.0.1:18080`／WebUI `localhost:65458`で標準起動し、runtime URL、HTTP `200`、`/props`のhardware field／metrics／CORS／admin／model-library、
  model-library snapshot、実`hf` auth status、localhost CORS preflightを確認した。Ctrl-C後はdynamic shutdown audit `clean=true`で、
  npm／Vinext子processが残らないことを確認した。dummy API key envはnpm／shell／Nodeの全childで不在、明示envは4件の非secret値だけだった。
  npm group leaderを先にSIGKILLした異常系でもsLLM終了時に残るshell／Nodeを回収した。model load、generation、throughput実測、downloadは
  この起動smokeでは行っていない。
- Phase 55 actual-model統合後: R9700 exact `gfx1201`、18.8 GB `gemma4moe` GGUFをmodelなし標準起動のlibrary APIから選択し、
  supported／compatible登録、17.6 GB resident load、Unicode非stream chat、code SSE、raw Completions、prefix再利用、metrics差分、
  client disconnect／recovery、unloadをPASSした。WebUI requestの出力上限はstrict serverでも動くcanonical
  `max_completion_tokens`へ修正し、request body回帰testを含む12 testとlintをPASSした。終了は
  `shutdown_audit.clean=true`、両port閉鎖、npm／Vinext子process 0、GPU process 0、VRAM基準値復帰を確認した。実downloadは行わず、
  `hf` status、20 model検索、25 GGUF fileとcopy command生成までを確認した。
- Phase 56 actual-model統合後: 同じmodelなし標準起動でGemma 4 12B targetと公式MTP assistantをcompanion pairとして選択し、
  targetだけをload操作して両方を10,046,932,204 bytes常駐させた。raw CompletionsとChat SSE、resident metrics、unload、clean shutdownを
  exact `gfx1201`でPASSした。dynamic metricsがstatic registryだけを参照して0を返す統合漏れを修正し、loaded lifecycleのbackend snapshotを
  表示する回帰まで確認した。

## 継続項目

Phase 48本来のsession save／resume、adapter、slot cancel、key status、log download、large-conversation検証、server binaryへのasset組込み、
versioned配布は未着手である。統合起動はlocal WebUI用のexact CORS originだけを追加し、TLSとoperator指定の追加CORSは既存server契約を使う。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/active/2026/08/21-31/phase48-minimal-webui-prototype.md) /
[Phase 37以降のロードマップ](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
