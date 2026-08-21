# Phase 43: Responses・Anthropic Messages・function/tool protocol

## 状態

- 2026-08-22開始、実装中。
- host/API/frontend/runtimeの機能Phaseであり、MI300X実機検証はユーザー指示によりdeferredとする。
- 組込みtool、MCP、shell/network/filesystemアクセス、credential解決はPhase 47の明示的なtrust-model承認まで開始しない。

## 目的

OpenAI ResponsesとAnthropic Messagesを、Chat Completionsのaliasではない別々のstrict compatibility profileとして公開する。
両transportはtyped internal item、tool definition/choice/call/result、生成中のstructured argumentsだけを共有し、wire DTO、
header、ID、usage、stop reason、error、SSE event state machineを混同しない。function/toolはmodelからcallを生成し、clientが
resultを返して次のgenerationを継続できるprotocolまでとし、sLLM process内でtoolを実行する経路は作らない。

## 仕様・参照pin

### OpenAI Responses

- OpenAI OpenAPI `2.3.0`、repository commit
  `010421dcbd0475277ea8c3e6c1e1cbca4659c4bd`の`POST /v1/responses`をPhase 43 profile v1へ固定する。
  2026-08-22にOpenAI Developer DocsのResponses create、function calling、streaming guideと照合した。
- 対応subsetは`model`、stringまたはtyped item arrayの`input`、top-level `instructions`、`max_output_tokens`、
  `temperature`、`top_p`、`stream`、`tools`、`tool_choice`、`parallel_tool_calls`、`reasoning.effort`、`metadata`、
  `store:false`と、明示的なsLLM extension `sllm.resumable`とする。`previous_response_id`、background mode、hosted/built-in tool、remote MCP、conversation/store、
  include expansion、multimodal input/output、file/url解決は拒否する。
- input itemは`message`の`input_text`、assistantの`output_text`、`function_call`、`function_call_output`を受理する。
  tool call/resultの`call_id`を保持し、隣接itemを勝手に結合しない。未知item/content typeはGPU admission前に拒否する。
- non-streamは`object:"response"`、status、typed `output` item、`output_text`、error/incomplete details、usageを返す。
  stateful storeは実装しないためresponse IDは観測用のrequest-local IDであり、後続参照には使えない。
- streamはnamed SSEで、`response.created`、`response.in_progress`、output item/content partのadded/delta/done、
  `response.output_text.delta/done`またはfunction-call arguments delta/done、最終`response.completed`の閉じた順序を持つ。
  post-header failureは`error`または`response.failed`を一度送り、成功terminalを後続させない。`[DONE]`は送らない。

### Anthropic Messages

- 公式Anthropic API version `2023-06-01`へ固定し、2026-08-22にAnthropic公式API overview、versioning、Messages create、
  streaming、stop reason、client tool、parallel tool useの各文書と照合した。
- `POST /v1/messages`は`anthropic-version: 2023-06-01`と`content-type: application/json`を必須とする。
  認証は既存sLLM key policyを利用するが、compatibility headerを黙って補完しない。
- requestは`model`、`max_tokens`、`messages`、任意top-level `system`、`stream`、`stop_sequences`、`tools`、
  `tool_choice`、明示的なsLLM extension `sllm.resumable`を受理する。message内system role、assistant prefill、sampling拡張、beta header、thinking block、画像・document、
  citations、prompt cache、server/builtin tool、MCP、programmatic tool useはprofile v1で拒否する。
- content blockはuser/assistant `text`、assistant `tool_use`、直後のuser message先頭に置かれる`tool_result`だけを受理する。
  各call IDへexactly one resultを対応させ、未知・重複・欠落ID、誤role、誤順序を拒否する。
- non-streamはAnthropic `message` envelope、content blocks、`stop_reason`、`stop_sequence`、input/output usageを返す。
  profile v1が生成するstop reasonは`end_turn`、`max_tokens`、`stop_sequence`、`tool_use`に限定する。
- streamは`message_start`、各content blockのstart/delta/stop、`message_delta`、`message_stop`のnamed SSEとする。
  tool argumentsは`input_json_delta.partial_json`、textは`text_delta`で表す。`ping`をsemantic stateへ数えず、stream error後に
  success terminalを送らない。Anthropic streamにも`[DONE]`は送らない。

### llama.cpp技術参照

- llama.cppはtag `b10453`、commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`へ固定する。
  Responses/Anthropic converter、serializer、SSE順序、tool parser、compatibility testsを技術参照とする。
- Rust側はsLLM固有のtyped internal normal formとadapterを新規実装し、llama.cpp sourceを直接reuseしない。
  直接翻訳・抜粋へ方針変更した場合だけfile単位のMIT provenanceを追加する。

## 共通internal protocol

- transport非依存の内部表現は、ordered message/content item、reasoning text、tool definition、tool choice、tool call、tool result、
  assistant prefill、parallel policyを型で保持する。wire field名やtransportのstop/error/event enumは内部型へ混ぜない。
- tool definitionは1〜128件、nameは`^[A-Za-z0-9_-]{1,64}$`、descriptionは最大16 KiB、schemaは最大1 MiB、
  request全体は既存96 MiB上限とする。call/result IDは最大256 bytes、arguments/result textは各16 MiB、並列callは最大16件とする。
- tool choiceは`auto`、`none`、`required`、specific toolを共通意味へlowerする。Responsesの`parallel_tool_calls:false`と
  Anthropic choiceの`disable_parallel_tool_use:true`は最大1 callを強制し、true/defaultは最大16件まで受理・生成する。
- tool-enabled generationはcanonical JSON envelopeを使う。message branchは
  `{"type":"message","reasoning":"...","text":"..."}`、tool branchは
  `{"type":"tool_calls","reasoning":"...","calls":[{"name":"...","arguments":{...}}]}`とし、`reasoning`は
  requestが有効化した場合だけ許可するoptional fieldとする。
  tool nameごとに`const`とそのargument schemaを結び、top-level choice/parallel policyをJSON Schemaへlowerする。
- Phase 40の`CompiledGrammar::from_json_schema`をgeneration開始前にcompileし、実token samplingへ適用する。
  完了後parseだけ、schemaに合わないraw textのtool call化、unsupported schema keywordの無視は禁止する。
- rendererはtool definition/historyをJSON-escapedなuntrusted dataとしてtyped promptへ埋め、instructionとして再解釈しない。
  verified capabilityのあるQwen production rendererだけを有効化し、Gemmaと未検証modelは`unsupported_parameter`で拒否する。
- tool outputはprotocol itemとしてpromptへ戻すだけで、内容をshell、URL、path、MCP request、dynamic library、credential名として
  解決しない。tool callを生成したことを実行許可と解釈しない。

## Work units

1. **P43-A0 profile/identity lock**: 本plan、machine-readable profile fixture、Draft 2020-12 schema、dependency-free validator、
   official reference identity、rejection matrix、limits、no-execution boundaryを実装前に固定する。
2. **P43-A1 internal protocol**: frontendへordered protocol item、tool definition/choice/call/result、validation/lowering、
   canonical generation envelope、tool argument schema compiler、Qwen typed rendererを追加する。既存no-tool renderer bytesを変えない。
3. **P43-B1 Responses parser**: strict duplicate/unknown/type/range/capability validation、input lowering、tool history、usage/status/error
   mappingを専用moduleに実装する。既存Chat parserのknown-unsupported contractは変更しない。
4. **P43-B2 Responses transport**: `/v1/responses`を既存bounded schedulerへ接続し、non-stream responseとprofile固有SSE、cancel、
   post-header error、Phase 39 resumable event storeを実装する。request-local IDとitem/call IDをstream全体で安定させる。
5. **P43-C1 Anthropic parser**: version header、strict message/content/tool validation、tool result ordering、tool choice/parallel policy、usage/
   stop mappingをResponsesとは別moduleに実装する。
6. **P43-C2 Anthropic transport**: `/v1/messages`を同じschedulerへ接続し、Anthropic固有non-stream/SSE event order、block index、
   cumulative usage、cancel、mid-stream error、resumable eventを実装する。
7. **P43-D1 production integration**: common generation requestをQwen production promptとPhase 40 grammarへ接続する。
   toolなしrequestは既存generation pathを使い、tool requestだけcanonical envelopeを生成・decodeしてvisible text/tool itemへ変換する。
8. **P43-D2 shared state semantics**: Phase 41 assistant-prefillはResponsesの対応subsetだけに接続し、Anthropic v1では拒否する。
   Responses multi-choiceはwireが単一responseのため拒否し、既存choice stateを暗黙反復しない。disconnect/cancel、usage、stop、
   reasoning、replayをtransport-specific state machineへ写像する。
9. **P43-E verification/closeout**: host unit/integration、synthetic backend HTTP/SSE、production compile、workspace test/clippy/fmt、
   profile/Markdown/link validationを実行し、archive/history/main-plan/API/runtime docsを更新して1 Phase 1 commitでpushする。

## Acceptance

- machine-readable fixture/schema/validatorは両profileのpositive/rejection matrix、limits、official pin、event transition、no-execution
  boundaryを一致させる。未知・duplicate field、wrong type、非finite値、境界の両側、unsupported capabilityをGPU admission前に拒否する。
- Responsesはraw HTTPのtext generation、tool definition、auto/none/required/specific choice、single/parallel call、tool result roundtrip、
  non-stream/SSE、stable ID、usage、stop、cancel、mid-stream error、resume範囲内/外をPASSする。
- Anthropicはversion header、system/text、tool_use/tool_result roundtrip、single/parallel policy、non-stream/SSE block index/event順序、
  cumulative usage、stop reason、cancel、mid-stream error、resume範囲内/外をPASSする。
- tool-enabled generationは実際にcompileされたgrammarでinvalid tool名、required call欠落、schema不一致、最大call数超過を生成中に
  禁止する。post-parse validatorもdefense in depthとして同じschemaとID/orderを確認する。
- no-execution testはmalicious tool名/description/arguments/resultへshell command、URL、path、credential markerを含めても、
  scheduler generation以外のprocess spawn、network、filesystem read/write、environment/secret lookupが到達不能であることを確認する。
- Qwen typed rendererはUnicode、JSON/XML delimiter相当文字、tool result error、parallel history、assistant prefillをlosslessに扱う。
  既存no-tool Chat/Completionsのrenderer bytes、parser rejection、response/SSE contractは回帰しない。Gemma tool requestはfail closedにする。
- workspace tests、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo fmt --check`、Markdown/link、profile validatorをPASSする。
  GPU kernel、operator、dtype、KV layout、selectorは変更しないため新規GPU性能PASSは要求せず、affected production buildを確認する。
  MI300X実行はdeferredであり、compile-only結果をruntime PASSと表記しない。

## Error・security contract

- malformed JSONは400 `invalid_json`、unknown/invalid valueと順序違反は400 `invalid_value`、既知未対応・model capability mismatchは
  400 `unsupported_parameter`、version header欠落/不一致は400、unknown modelは404、body超過は413、auth失敗は401、queue fullは429。
- request body、prompt、tool description/schema/arguments/result、token IDs、credentialをlog、metric、props、slot snapshot、error detailへ
  含めない。metric labelはprofile/endpoint/statusのbounded vocabularyだけを使う。
- CPU parse/grammar compileを含む全上限をGPU admission前に適用する。scheduler、timeout、cancel、single-owner runtimeを迂回しない。
- SSEはprofileごとのclosed state machineを持ち、重複terminal、terminal後event、block/index不整合、tool argument completion前のcall完了、
  error後successを禁止する。resumable replayは別profileのeventを混在させない。
- ResumableはPhase 39の64 KiB/event・256 KiB/sessionと設定event数を拡張しない。Responsesのofficial-style snapshot/done eventを
  分解して意味を変えず、`max_output_tokens` / `max_tokens <= 40`をscheduler admission前に要求する。128-byte token-piece上限、
  worst-case JSON escaping、bounded metadataからsnapshot/done eventが余裕をもって収まることを固定し、成功event batch全体も事前確認する。防御的確認に
  収まらない生成結果は部分successを残さず単一error terminalにする。

## Non-goals

- Chat Completionsへtoolsを追加すること、OpenAI Agents/Assistants/Realtime、Responses store/conversation/previous response、background、
  hosted search/code/image/computer tool、remote MCP、Anthropic server/builtin tool、Tool Runner、prompt caching、beta API。
- process内tool execution、worker/sandbox、shell/network/filesystem、credential/token broker、permission UI、human approval。これらはPhase 47で
  trust modelの明示承認後にだけ再計画する。
- generic Jinja/reasoning UX/interactive CLI（Phase 44）、adapter/router（Phase 45）、conversion/bench（Phase 46）、WebUI（Phase 48）。
- multimodal Responses/Anthropic、audio/image/document/citation、arbitrary provider-specific content block、MI300X最適化、multi-GPU/parallel。

## Stop/replan条件

- official pinと実装subsetの意味が両立しない、同じtransport state machineが2回integration rejectとなる、tool実行または外部I/Oへの
  reachabilityが生じる、validationがGPU admission後へ漏れる、SSEがterminalを重複する場合は同じwork unitを追加実装せず再計画する。
- Phase 44/45/47のscopeが必要になった場合はfallbackで取り込まず、Phase 43のunsupported boundaryを維持して後続へ渡す。

## Closeout

- 2026-08-22にP43-A0〜Eをhost-onlyで完了した。Responses/Anthropicのstrict parser、profile別non-stream/SSE、Qwen
  grammar-constrained tool call、client-owned result roundtrip、assistant-prefill subset、redacted error、40-token admission付きbounded replayを実装した。
- malicious tool payloadは生成・返却されてもdataのままであり、process/network/filesystem/environment/credential/MCP/workerへ
  到達しない。Phase 47のapproval-required境界は変更していない。
- machine profile validator、core/frontend/server unit、raw HTTP/SSE、tool single/parallel/result、mid-stream failure、replay、legacy
  regression、workspace fmt/clippy/testをrelease gateとする。GPU kernel/providerは変更しておらず、MI300X実機検証はdeferredのままである。
- [history](../../../../../history/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)と相互リンクし、
  `docs/plans/main-plan.md`、Phase 37+ roadmap、OpenAI/Anthropic compatibility、runtime architectureを同期する。
