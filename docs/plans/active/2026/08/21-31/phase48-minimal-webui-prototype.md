# Phase 48: minimal WebUI prototype

## 目的

sLLMのHTTP APIを利用するWebUI prototypeを作り、GPU認識状態、prefill／decode throughput、server状態、model管理を主画面で確認する。
chat、reasoning表示、generation設定は副画面として維持する。通常推論は公開APIだけを使い、model folderの参照だけはloopback server上の
admin APIへ分離する。Hugging Face取得はユーザー承認済みの固定`hf` subcommandだけを同じloopback admin境界へ追加し、browserへ直接
filesystem access、任意command実行、独自session state machineは追加しない。

## 受入条件

- working surfaceの最初の画面にserver接続状態、GPU認識状態、model選択、prefill／decode throughput計測を表示し、chatは明示切替の
  secondary surfaceにする。
- live接続は`GET /healthz`、`GET /readyz`、`GET /v1/models`、`GET /props`、`GET /metrics`、
  `POST /v1/chat/completions`を使う。model folder管理はloopback dynamic serverだけの`/admin/model-library*`へ限定する。
- live throughputは同一modelの単一streaming request前後で`/metrics`を差分し、prompt token／TTFTからprefill、
  first tokenを除くcompletion token／post-TTFT時間からdecodeを推定する。これはkernel-only benchmarkではなく、同時requestが混ざった
  windowを成功値として表示しない。
- GPU identityは`/props.hardware`のserver側HIP device queryだけをlive表示する。対応AMD GPUを照会できない場合は未提供と明示し、
  browser WebGPUやfixtureをsLLM実行GPUの認識結果へ読み替えない。
- Chat Completions SSEの`delta.content`と`delta.reasoning_content`を分離表示し、`[DONE]`欠落、error event、HTTP error、取消しを失敗として扱う。
- endpointとuser/admin keyはprototypeのmemory stateだけに保持し、URL、localStorage、log、tracked fixtureへ保存しない。
- model load／unloadはPhase 45のalias-only admin routeへ空bodyで送る。path、URL、artifact payload、credentialを要求bodyへ入れない。
- model source optionを省略した`127.0.0.1` serverを許可し、WebUIでserver側folderを参照・選択・再走査する。選択pathはserverの
  user configへatomic保存し、browser storageへ保存しない。走査は直下のregular non-symlink `.gguf`へ限定する。
- GGUF architecture、対応するcanonical `*.derived-lock.json`、reviewed model lock、runtime weight plan、resident byteをload前に検証する。
  対応architectureはalias-only lifecycleへ動的登録し、未対応または検証不能なmodelは理由付きでグレー表示する。
- `--metrics true`はmodel未選択でも起動可能にし、WebUI catalogが登録したbounded aliasだけをmetric seriesへ追加する。
- Hugging Face検索はserverに導入済みの`hf` CLIで明示実行し、GGUF tagのmodel、解決済み完全commit SHA、repository root直下の
  `.gguf`と同名`*.derived-lock.json`だけをboundedに返す。copy用commandは現在のserver側model folderと完全SHAを含める。
- download buttonは`repo_id`、完全SHA、root file名の構造化requestだけを受け、表示commandや任意destination／追加argumentを実行入力にしない。
  `sh -c`を使わず固定argvで`hf download`を1 jobずつ実行し、queued／running／completed／failedをpolling表示する。
- server側Hugging Face認証を確認し、未認証時は匿名requestの低いrate limitとgated／private失敗可能性を警告する。公開modelの実行は妨げず、
  token入力、token表示、`hf auth login`案内／copyは追加しない。
- `sllm-server`はWebUIをdefault enabledで起動し、`--webui false`で無効化、`--webui-port PORT`でAPIとは独立にportを変更できる。
  default portは`65457`とし、port 0、APIと同一、使用中の場合は別portへ移動せず起動errorにする。
- WebUI有効時の省略時metrics、localhost／127.0.0.1 exact CORS、runtime API URL注入、自動live接続を一つの起動経路で構成する。
  credentialは注入せずmemory-only入力を維持し、sLLM終了時はWebUI process groupも回収する。
- demo modeは画面確認専用と明示し、live API成功やGPU実行の証拠として扱わない。
- desktopとmobile幅、keyboard送信、stream取消し、loading／empty／error／successを実装し、production serviceへ自動deployしない。

## prototype外

- Phase 47の汎用tool/MCP実行、任意command／任意URL／任意destination実行、browser側credential永続化、upload、
  mid-generation checkpoint resume、再帰走査、symlink追跡、browser側filesystem access。承認済みの固定`hf`検索／取得subcommandは含めない。
- release packaging、sLLM server binaryへの静的asset埋込み、Node非依存配布、TLS自動構成、release-gradeのGPU性能証拠。

## 検証

- frontend build、lint／type check、API client unit test、静的preview responseを実行する。
- live GPU serverがない場合はdemo modeとmock HTTP contractで画面・SSE parserを検証し、live接続PASSへ読み替えない。

## 実装結果（2026-08-30）

- `webui/`へsingle-route prototypeを実装し、safe demoとlive接続を同じworking surfaceで切替可能にした。
- 公開route、alias-only model action、memory-only credential、reasoning分離、strict `[DONE]`、取消しの受入条件を満たした。
- desktop／mobile layout、loading／empty／error／success、keyboard送信を実装した。
- typecheck、4件のSSE parser testと2件のAPI client test、対象source lint、production build、local HTTP `200`をPASSした。
- live sLLM／GPU接続は今回実行しておらず、prototype表示とbuild結果をGPU correctness／品質／性能証拠へ使わない。
- owner-onlyの[private UI preview](https://sllm-console-prototype.jyohukuchan.chatgpt.site)を公開した。これはWebUI artifactだけであり、
  sLLM backendやproduction inference serviceのdeployではない。
- 2026-08-30の追加指示でライトテーマを初回既定にし、ヘッダーからダークテーマへ切り替えられるようにした。
  credentialと同様にtheme切替でserver契約を変更せず、reload時はライト既定へ戻る。
- 2026-08-31の追加指示でGPU／throughput dashboardをdefault viewへ変更し、chatをsecondary viewへ移した。safe demoでは固定fixtureを
  明示する。起動統合時にserver側HIP device queryを`/props.hardware`へ追加し、live modeはvendor、device名、gfx target、VRAMを表示する。
- live benchmarkは`/metrics`のmodel／stream counterとTTFT／E2E histogram sumの前後差分を使う。1件のsuccessful request以外が
  windowへ混ざった場合や、不完全なtoken／timing差分は失敗とし、推定値をkernel-only測定として扱わない。
- metric parser testを追加し、typecheck、7件のtest、lint、production build、local HTTP `200`をPASSした。live sLLM／GPU接続は
  今回も実行していない。
- これはPhase 48全体の完了ではない。session、adapter、slot、key status、log、server binary組込み／配布は継続項目である。
- 2026-08-31の追加指示で、sLLM serverをmodel sourceなしでloopback起動できるようにし、server側folder browser、選択pathの永続化、
  direct GGUF走査、architecture／derived lock／reviewed identity／weight-plan検証、動的alias登録を実装した。WebUIは対応GGUFを操作可能な
  modelとして表示し、非対応modelを理由付きでグレー表示する。既存`--models`／legacy single-model起動は互換経路として維持する。
- 2026-08-31の追加指示で、同じmodel library画面へHugging FaceのGGUF検索、完全commit SHA固定のdownload command表示／copy、server側
  download buttonを追加した。検索、file一覧、認証確認、downloadは導入済み`hf` CLIへ限定し、実行requestからcommand／destination／tokenを
  受け取らない。未認証はrate-limit警告だけを表示し、`hf auth login`案内やcopy機能は追加しない。
- 2026-08-31の追加指示で起動を統合した。`sllm-server`は既定でlocal Vinext WebUIを`localhost:65457`へ起動し、API URLだけを
  `/api/runtime-config`へruntime注入して自動live接続する。`--webui false`、`--webui-port`、metrics省略時有効化、exact CORS追加、
  port衝突のfail-closed処理、gracefulな子process group回収を実装した。static asset埋込み／release packagingは引き続き対象外である。
- Phase 55 actual-model統合で`gemma4moe`のfolder選択、load、非stream／SSE／raw生成、metrics、prefix、cancel／recovery、unload、
  clean shutdownをR9700 exact `gfx1201`でPASSした。WebUIの出力上限はstrict profileのcanonical
  `max_completion_tokens`を送るよう修正し、旧`max_tokens` aliasへ依存しない。

[全体計画](../../../../main-plan.md) / [Phase 37以降のロードマップ](phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[対応する履歴](../../../../../history/2026/08/21-31/phase48-minimal-webui-prototype.md)
