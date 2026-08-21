# Phase 44: template・reasoning control・interactive UX

## Status and authority

- 状態: complete（2026-08-22、host/frontend/CLI統合）。MI300X real executionはユーザー指示によりdeferredであり、GPU
  provider/kernelのruntime PASSは主張しない。
- 上位authorityは`sLLM.md`、`AGENTS.md`、`docs/plans/main-plan.md`、Phase 37+ roadmapとする。Phase 41のcheckpoint/
  assistant-prefill、Phase 42のFIM/infill、Phase 43のtool protocolを再実装しない。
- closeoutでは本planをarchiveへ移し、対応history、main plan、roadmap、API/runtime/model-lock文書を同期する。commit/pushは
  release laneで別途行う。

## Fixed references and dependency decision

- llama.cpp behavior referenceは`b10453` / commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`へ固定する。
  Jinja context、kwargs、reasoning budget、reverse promptの意味比較だけに使い、wire/CLI aliasの完全互換はclaimしない。
- Generic rendererはMiniJinja `2.24.0`をexact pinする。default feature setを使わず、
  `builtins`、`fuel`、`json`、`macros`、`multi_template`、`serde`だけをallowlistする。`macros`と`fuel`を同時に使う
  2.24.0のコンパイル契約上`multi_template`は必要だが、include/import/extendsは構文検査で拒否する。dynamic loader、stack
  growth、custom syntax、URL/process/filesystem integrationは有効化しない。
- MiniJinjaのofficial APIが提供するper-render fuel、runtime recursion limit、strict undefinedを必須設定とする。dependency closure、
  checksum、license、MSRVは既存Rust dependency policyへ固定する。
- llama.cpp sourceの直接reuseは予定しない。fixtureまたは実装をreuseする場合は`docs/provenance/README.md`に従い、本Phaseの
  closeout前にcommit/license/pathを記録する。

## Frozen product surface

### Generic template profile v1

- 既存Qwen reviewed rendererはdefaultのままbyte/token identityを維持する。generic rendererは別の明示opt-in providerとし、
  model lockのreviewed defaultやGemma raw-text capabilityを暗黙置換しない。
- custom templateはUTF-8 source、利用者が指定した`sha256:<64 lowercase hex>`、renderer profile versionを全て照合してからcompileする。
  CLIのfile readerはregular fileだけを開き、symlink、special file、size raceを拒否する。renderer自身はpathを受け取らず、
  filesystem、environment、network、processへ到達しない。
- contextはJSON-likeの`messages`、`tools`、special-token文字列、`add_generation_prompt`、`enable_thinking`、
  `reasoning_effort`とbounded custom kwargsだけである。host object、method callback、secret、credential、path、clockは公開しない。
- supported Jinja subsetはinterpolation、if/elif/else、for、set、macro、標準の安全なcollection/string/JSON filterへ限定する。
  unknown variable/filter/test/function、include/import/extends、dynamic loader、unrestricted method/attribute、`__*` accessはfail closedとする。
- template sourceは64 KiB、rendered outputは16 MiB、messagesは1,024、kwargsは64 key・合計1 MiB・depth 32、recursionは32、
  fuelは1,000,000 instructionを上限とする。上限はcompile/render/tokenizeとscheduler/GPU admissionより前に適用する。
- output identityはprofile version、template digest/source size、kwargs digest、rendered bytes digestを持つ。checkpoint/cache identityには
  exact template digestを結合し、custom templateとreviewed defaultのstateを共有しない。

### Reasoning control v1

- modeは`disabled`、`enabled`、`template-default`を既存`ThinkingModeV1`へ写像する。optional budgetは1..=4,096 generated
  reasoning tokenで、disabledとの併用、raw prompt、Gemma/raw-text、reasoning markerを持たないtemplateではadmission前に拒否する。
- Qwenのassistant generation prefixがreasoning-active stateを開始する。modelがclosing markerを先に生成した場合は通常生成へ遷移し、
  budgetへ達した場合は既存token selector maskを通じてbounded closing-token列を強制する。別decode loop、host-only token fallback、
  tokenの後置書換えは作らない。
- closing sequenceを含めても`max_new_tokens`内に収まることをadmission前に確認する。grammar、stop、device selector、sampling、
  cancellationとmaskを交差し、全候補禁止や不一致はfail closedとする。強制tokenもusage/generated/decode historyへ通常tokenとして記録する。
- CLI/API adapterは同じfrontend controllerを使う。Phase 43 Responses `reasoning.effort`はprofile-defined budget mappingへlowerするが、
  Anthropic thinkingは現行profileどおりunsupportedのまま維持する。stream/non-stream reasoning分離は既存splitterを再利用する。

### Interactive CLI/session v1

- 既存`generate` JSON reportと引数は変更しない。新しい`chat` commandだけがinteractive UXを所有する。
- `--prompt-file PATH`はbounded regular fileを1回読み、terminal stdinと混同しない。`--prompt`、`--message`、`--prompt-file`、
  interactive stdinはclosed conflict matrixにする。NUL、invalid UTF-8、symlink、special file、16 MiB超過を拒否する。
- interactive inputはline-oriented UTF-8で、TTY固有raw modeを要求しない。empty line、EOF、SIGINT/cancel、maximum turn/message/
  transcript bytesを明示する。outputはversioned JSON Lines eventで、prompt/tool payloadをlog/diagnosticへ複製しない。
- reverse promptは最大4件・合計1 MiB、生成visible output上のturn-return boundaryである。`--stop`のgeneration finish semanticsへ
  読み替えず、matched reverse promptを次turn inputへ含めない。
- conversationはsystem/user/assistantのtyped transcriptとして最大1,024 message・16 MiBへboundedにserializeする。save/resumeは
  Phase 41 `SessionCheckpoint` / `CheckpointStore`のconversation、token history、model/renderer/tokenizer/target/plan/KV identity、
  atomic write、0700/0600、quota、checksumをそのまま使う。独自session format、implicit global session、mid-generation resumeは作らない。
- freshとresumeは同じexact transcript/template/optionsから同じprompt token列を作る。checkpoint restore後も既存runtime ownerが
  KV/GDN stateをtransactionalに復元し、CLI adapterはopaque state planeを解釈しない。

## Work units

1. **P44-A0 profile lock**: machine-readable fixture、Draft 2020-12 schema、dependency-free validator、limits、surface conflict matrix、
   llama.cpp/MiniJinja pinを実装前に固定する。
2. **P44-A1 sandbox renderer**: `sllm-frontend`へgeneric provider、bounded output writer、strict context/kwargs、digest identityを追加し、
   reviewed Qwen defaultを維持する。
3. **P44-A2 template adapters**: tokenizer utility、generation input、Phase 42 apply-template/input-token-countとCLIへgeneric providerを
   capability-gatedで接続する。custom sourceのfile readはCLI boundaryだけに置く。
4. **P44-B1 reasoning controller**: frontend generation config/stateへreasoning budgetとforced closing sequenceを追加し、host/device selector、
   grammar、stop、assistant-prefillへ統合する。
5. **P44-B2 protocol adapters**: existing Chat/Responses fieldsを同じreasoning configへlowerし、strict reject、redacted error、
   non-stream/SSE splitを検証する。Anthropic/Gemma unsupported boundaryを変えない。
6. **P44-C1 prompt file/reverse state**: shared bounded CLI input readerとreverse-prompt matcher、deterministic interactive state machineを追加する。
7. **P44-C2 chat/checkpoint integration**: new `chat` commandを既存production ownerへ接続し、typed transcriptとPhase 41 checkpointの
   save/resume identityを統合する。既存`generate` execution/reportを変えない。
8. **P44-D verification/closeout**: host unit/integration/process tests、machine validator、dependency policy、workspace fmt/clippy/test、
   Markdown/link validationを実行し、archive/history/main-plan/roadmap/API/runtime/model-lock docsを同期して1 commitでpushする。

## Acceptance

- sLLM canonical templateとllama.cpp-compatible fixtureでrole、special token、tool/reasoning block、Unicode、JSON kwargs、
  add-generation-promptをexact rendered bytes/token IDsへ一致させる。malformed syntax、unknown variable/filter/function、include/import、
  recursion/fuel/output/depth/countの境界両側をfail closedにする。
- custom templateはmissing/wrong/uppercase digest、empty/oversized/raced/symlink fileを拒否し、reviewed default未指定時の既存Qwen
  rendered bytes、digest、token列を維持する。Gemmaへverified capabilityなしで適用しない。
- reasoning disabled/default/enabled、budget 1/非aligned/4,096、early close、exact budget、multi-token forced close、max-output不足、
  grammar conflict、stop/cancel、host/device selectorでfixed token oracleをPASSする。forced close後のfinal answer token順を維持する。
- Chat/Responses non-stream/SSEとCLIはreasoning/visible content、usage、finish reasonを同じ生成結果から写像する。Anthropic thinkingと
  unsupported backendはscheduler/GPU admission前に拒否する。
- prompt fileとstdin、reverse promptとstopを混同しない。scripted interactive/non-interactiveおよびfresh/resumeは同じtoken列を生成し、
  successful turnだけをconversation/checkpointへatomic publishする。wrong model/template/tokenizer/target/plan/KV、corrupt/truncated、
  quota超過、cancelled turnはstateを公開しない。
- `cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test --workspace`、Phase 44 validator/
  mutation tests、Rust dependency policy、Markdown links、`git diff --check`をPASSする。
- GPU kernel/provider/selector ABIは変更しない。host testやgfx942 compile-only結果をMI300X runtime PASSと表記せず、実機実行はdeferredにする。

## Security and non-goals

- template/runtimeからfilesystem、environment、network、process、credential、secret、MCP、tool execution、host callbackへ到達させない。
  Phase 43 generated tool callはdataのままで、approval-required Phase 47を開始しない。
- full Jinja2/Python object compatibility、arbitrary extension/plugin、remote template、template hot reload、browser/terminal UI library、
  server WebUI、mid-generation checkpoint、distributed session ownershipは非対象である。
- model architecture、new dtype/KV format、MI300X tuning、multi-GPU、adapter/router lifecycle（Phase 45）、conversion/quality tools
  （Phase 46）を本Phaseへ混ぜない。

## Stop/replan conditions

- generic templateがhost capabilityへ到達する、existing reviewed renderer byte identityを維持できない、reasoning controllerが
  selector/grammar/stopを迂回する、checkpoint identityなしにstateを再利用する、または同じintegration work unitが2回rejectされた場合は、
  追加実装を止めて同じPhase内の分割/契約を再計画する。
- real interactive runtime integrationが既存production ownerの全面複製を必要とする場合は、owner抽出を先に行う。機能をfake-onlyで
  完了扱いにせず、mid-generation resumeを暗黙追加しない。

## Closeout

- P44-A0〜A2をhost/frontend/CLIへ実装した。MiniJinja 2.24.0 exact pinのgeneric providerは、digest-bound UTF-8 source、JSON-only
  context、strict undefined、bounded output/depth/kwargs/messages、fuel/recursion、closed filter/test/globalとinclude/import/extends
  拒否を持つ。reviewed Qwen rendererのdefault bytes/token identityとGemma raw-text capabilityは変更していない。
- P44-A2はtyped `GenericTemplateMessagesInputV1`と`TokenizerUtilityServiceV1`へ接続し、CLIのcustom templateはregular non-symlink
  file、`O_CLOEXEC|O_NOFOLLOW`、64 KiB bounded read、size-race、UTF-8/NUL、lowercase SHA-256をGPU/backend初期化前に検証する。
  kwargsはduplicate/non-finite/non-objectを拒否し、reportはtemplate/kwargs/rendered identityだけをdata-onlyで返す。raw/Gemma generic inputと
  未対応backendはtokenize/GPU work前にfail closedとした。
- P44-B1/B2はreasoning mode/budgetを既存generation selector・grammar・stop・cancelと交差させ、1〜4,096 token、multi-token close marker、
  early close、max-output不足、mask conflictをbounded controllerで処理する。Chat/Responses/CLIは同じfrontend semanticsへlowerし、
  Anthropic thinkingとGemma/raw-textはunsupportedのまま維持した。
- P44-C1/C2は`chat`のclosed prompt source matrix、regular prompt file、bounded typed transcript、reverse prompt、JSONL events、
  successful-turn-only publishを追加した。Persistent Qwen chatはreviewed history semanticsでhidden reasoning、selected stop、matched reverse markerを
  除外したcanonical history prefixへrebaseし、fresh resident ownerへre-prefillしてopaque checkpointをcaptureするため、next-turn/fresh-resume
  exact prefixを維持する。load時はmodel/renderer/tokenizer/target/plan/KV identityをtransactionalに検証し、conversation+KVのpending/current commit
  rollbackを行う。CLIはsourceとbounded inputをpreflightしてからmodel openし、SIGINTはin-flight turnのcancellation laneだけをcancelする。既存one-shot
  `generate` reportは変更せず、mid-generation resume・WebUI・Phase 47 tool/MCP executionは開始していない。
- focused Phase 44 frontend/CLI testsに加え、workspace全体のfmt、warnings-denied clippy、test、Rust dependency policy、exact Rust 1.85
  offline workspace/all-target build、machine profile、Markdown link validationを実行してPASSした。MI300XはVM削除済みのためreal
  correctness/performanceはdeferredであり、feature-pinned gfx942 compile/host evidenceをruntime PASSへ昇格しない。

対応historyは[Phase 44 history](../../../../../history/2026/08/21-31/phase44-template-reasoning-interactive-ux.md)であり、
main plan、Phase 37+ roadmap、API/runtime/model-lock docsと相互に同期する。

## References

- [Phase 37+ roadmap](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
- [Phase 41 archive](phase41-prefix-session-speculation.md)
- [Phase 42 archive](phase42-inference-modes-public-endpoints.md)
- [Phase 43 archive](phase43-responses-anthropic-tool-protocol.md)
- [Runtime architecture](../../../../../architecture/runtime.md)
- [Model lock](../../../../../models/model-lock.md)
- [OpenAI compatibility](../../../../../api/openai-compatibility.md)
- [Provenance policy](../../../../../provenance/README.md)
