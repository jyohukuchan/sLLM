# Generic reasoning Phase 0 audit

## Scope

`docs/plans/generic-reasoning-thinking-budget-production-plan-v0.1.md` に従う
reasoning / thinking budget production workの開始監査。

## Read-only handoff audit

- workspace root: `/home/homelab1/coding-local/ultimateLLM`
- repository root: `/home/homelab1/coding-local/ultimateLLM/uLLM-project`
- branch: `main`
- source HEAD: `f1c0c08`の後続作業中。`origin/main`より先行
- worktree変更は既存の未追跡`.rocprofv3/`のみ。reasoning作業では変更しない
- `ullm-openai.service`: enabled / active / `NRestarts=0`
- `llama-qwen35-udq4.service`: enabled / active / `NRestarts=0`
- OpenWebUI container: `ullm/open-webui:0.9.4-ullm.1`, healthy
- active manifest: `ullm.served_model.v1`, public model `ullm-qwen3.5-9b-aq4`
- active manifest context: 4096, completion limit: 512
- active worker protocol: `ullm.worker.v1`
- active promotion source commit: `4be10d007c48db444f6b18fe6ac22e09dc17f168`

Source HEAD、稼働binary、active manifestのpromotion source commitは一致すると仮定しない。
secret、API key、会話本文、OpenWebUI DB、`.rocprofv3/`の内容はこの記録へ保存しない。

## Current implementation boundary at audit start

- Gateway schemaは`reasoning_effort`をknown-but-unsupportedとして扱う。
- Gateway tokenizerはmanifestの`enable_thinking`を固定値として使用する。
- Gateway worker requestはreasoning phase / budgetを持たない。
- Rust `InferenceRequest`と`ullm.worker.v1`にもreasoning controlはない。
- AQ4 sessionには`prepare -> publish -> commit`境界があり、forced token queueの候補境界である。
- OpenWebUI側は`delta.reasoning_content`を扱えるため、UI patchは計画対象にしない。

## Verification

実行コマンド:

```text
cd uLLM-project/services/openai-gateway
uv run --frozen pytest -q
```

結果: `207 passed in 9.33s`

通常のsystem Pythonによるpytestは、依存とpackage pathがないためcollection errorになった。
以後はgatewayの指定どおり`uv run --frozen`を使用する。

## Next

Phase 0のmachine-readable baselineとv0.2 contractを固定し、synthetic multi-token
ReasoningDialectでtoken-level segmenterを先に検証する。production service、manifest
activation、OpenWebUI DBは変更しない。

## Implementation progress in this work unit

- Added model-independent Python/Rust reasoning state machines.
- Added synthetic multi-token natural-close, forced-close, budget-zero, EOS,
  cancellation/error, and raw output split tests.
- Added proposed v0.2 OpenAI, served-model, and worker specifications.
- Added v2 reasoning dialect parsing to the Python and Rust served-model loaders;
  v1 fixtures remain unchanged.
- Added Gateway request normalization, dynamic `enable_thinking`, assistant
  `reasoning_content` history policy, non-stream/stream field separation, and
  worker v2 reasoning command construction. Production activation is not done.
- Verification: Gateway `uv run --frozen pytest -q` reports `218 passed`;
  Rust targeted tests for inference and served-model/reasoning pass.

## Implementation progress after the audit

- `ullm.worker.v2` strict parserを追加した。v1のreasoning fieldは拒否し、v2の
  nested objectは未知field・重複key・bounded token arrayを拒否する。
- loaded served-model profileへworker schemaとReasoningDialectを伝播し、Rust側でも
  dialect identity、終了token列、予約token、budget上限を照合するようにした。
- AQ4 sessionへReasoningStateとforced-token queueを接続した。budget到達後はモデルの
  追加top-1を行わず、forced end sequenceを`prepare -> publish -> commit`で通す。
- v2イベントのschema versionをloaded worker profileから出すようにした。active v1
  production serviceは再起動・変更していない。
- 検証: Rust libの全回帰698件成功・1件ignored、AQ4 session 31件、
  worker profile snapshot 3件、Gateway 218件が成功した。
- `cargo check -p ullm-engine --all-targets`は既存の壊れた
  `examples/sq8_ck_serving_performance.rs`で失敗する。今回の変更とは無関係で、libと
  対象testは通過している。

Phase 0のreal AQ4 HTTP/SSE baselineはまだ未取得であり、既存の2026-07-12/13のAQ4・
OpenWebUI evidenceを参照できる状態に留めている。

取得済み証跡と不足項目は、
`benchmarks/results/2026-07-13/generic-reasoning-phase0-baseline-v0.1/inventory.json`
へ`status: incomplete`として固定した。これをPhase 0 gateの合格証跡とは扱わない。

## Generator and validator follow-up

- `tools/generate-served-model.py`をv1/v2 profile対応へ拡張した。reasoning profileがある
  場合は`ullm.worker.v2`を要求し、生成manifestを`ullm.served_model.v2`として出力する。
  v2 workerでreasoningがないprofileは拒否する。
- `tools/validate-served-model.py`とgenerator内のloaderは、Gateway packageの起動処理を
  importせずに相対importを含むserved-model validatorを読み込めるようにした。
- synthetic multi-token reasoning profileを実際に一時manifestへ生成し、v2 strict loaderで
  読み込むtestを追加した。v1 generator/validator fixtureの結果は維持した。
- 検証: `tests/test_generate_served_model.py`と`tests/test_validate_served_model.py`は
  `25 passed`、変更対象のRuff checkも通過した。
- この作業単位は`a4b9152 feat: support v2 reasoning in manifest generator`へ保存した。

Phase 0 inventoryのsource commitは取得時点の記録として維持し、同一HEADのAQ4 HTTP/SSE
baselineが得られるまでgateをcompleteへ変更しない。

## Live v1 probe

- active manifestをvalidatorで再確認した。`ullm.served_model.v1`、AQ4_0、worker v1、
  binary SHA-256は稼働中の値と一致した。
- OpenWebUI container経由のread-only相当probeで、reasoning controlを省略した非streamは
  HTTP 200、prompt 18、completion 2、total 20、`finish_reason=stop`だった。
- streamはHTTP 200、`role -> token -> stop -> usage -> done`、usage 18/2/20で、
  `delta.reasoning_content`は観測されなかった。request/response本文は保存せず、metadataと
  SHA-256だけを`benchmarks/results/2026-07-13/generic-reasoning-phase0-baseline-v0.1/active-v1-http-sse-probe.json`
 へ保存した。
- このprobeはactive promotion source commitが現HEADと異なり、長いpromptとgenerated token
  IDを含まないため、Phase 0 gateの合格証跡ではない。

Generator変更後のGateway全体回帰も再実行し、`uv run --frozen pytest -q`で`218 passed`
（9.26秒）を確認した。repository worktreeは既存の未追跡`.rocprofv3/`だけを残している。

## Follow-up contract hardening

- Gatewayの`WorkerConfig.from_settings()`が`ullm.worker.v2`を起動時に受け付けるようにし、
  v2 manifestのreasoning dialectがworker configまで伝わることをtestした。
- unbounded (`thinking_budget_tokens=-1`)でもforced end sequenceと最低回答tokenの予約を
  生成前に検証するようにした。non-stream/streamのreasoning分離、usage詳細、synthetic
  multi-token SSEの再結合testを追加した。
- Python/Rust served-model loaderでschemaとworker protocolの版を一致必須にし、区切りtokenの
  重複・start/end prefix衝突・manifest最大budgetの予約超過をfail closedにした。
- AQ4 sessionでforced-close中のcancel、publish callback未実行、reset後の次request再利用を
  testした。
- 検証: Gateway全体`222 passed`、Gateway対象test `154 passed`および`126 passed`、Python
  reasoning/manifest `53 passed`、Rust reasoning関連9件、AQ4 cancellation test 1件。
- 本番用`target/release`は上書きせず、`CARGO_TARGET_DIR=/tmp/ullm-reasoning-target`
  で`ullm-aq4-worker` release buildを完了した。active serviceは再起動・変更していない。
- 変更commit: `c9b2897`、`c57b5f5`、`029e27f`、`d21fb31`。

## v2 release accounting and unactivated candidate

- AQ4 `ReleaseSummary`がReasoningUsage（reasoning token数、forced-end token数）を保持し、
  `ullm.worker.v2`の`released` eventへ`reasoning_tokens`と`forced_end_tokens`を記録するようにした。
  v1 eventは従来の形を維持し、v2 Gatewayは両方の数値とcompletion token数の整合性を検証する。
- 合成resident worker routeのv2 testでtoken列`[7, 20, 21, 2]`、reasoning `1`、forced-end `2`、
  completion `4`を確認した。Gateway fake v2 workerでも同じrelease accountingを検証し、worker testは
  `30 passed`になった。
- `docs/specs/aq4-reasoning-openwebui-release-v0.1.md`へrelease accountingの契約を追記した。
  `deploy/served-models/qwen35-9b-aq4-reasoning.profile.json`を未activateのv2候補として追加し、
  READMEに現行v1 receipt/binaryとは互換性がなく生成がfail closedになることを記録した。
- 候補profileのgeneratorは実際に`generator_status=1`、`output_absent=yes`、
  `served-model generation failed: AQ4 promotion evidence worker protocol differs`となった。
  現行v1 service、active manifest、OpenWebUI image、production binaryは変更していない。
- Rust targeted worker-driver tests 3件とresident v2 reasoning route 1件は成功した。
  なお`cargo test -p ullm-engine --lib --no-run`ではRust lib test compileまで成功したが、workspaceの
  既存example `sq8_ck_serving_performance.rs`は別途壊れているためall-targets相当は不合格である。
- 追加監査で、forced tokenをprepareした後にcancelされた場合の未commit tokenがusageへ混ざる余地を
  見つけた。`Qwen35Aq4PreparedToken`へprepare前のreasoning/forced countersを持たせ、cancelまたは
  publisher failure時に復元するよう修正した。AQ4 session test 32件が成功した。
- 変更は`073bf5a`（v2 release accounting）、`d5138b5`（未commit usage除外）、`c6809a7`（v2仕様追記）、
  `9dd761d`（計画status更新）へ保存した。worktreeには既存の`.rocprofv3/`だけを残している。
- Gateway release parserへbudget overshoot、forced sequenceの部分release、dialect上限の検証を追加し、
  worker test `30 passed`を再確認した。変更commitは`b6a8f84`である。
- 既存のAQ4 promotion evidence toolがready/generate/shutdown schemaをv1固定していたため、residentは
  manifest protocol（v1/v2）、legacy compatibility routeはv1として扱うよう修正した。v2 fixtureを含む
  evidence tool testは`3 passed`である。変更commitは`cb06ae7`。
- 最新ソースで隔離release buildを行い、`/tmp/ullm-reasoning-target-v2/release/ullm-aq4-worker`を生成した。
  binary SHA-256は`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`で、help表示と候補profileの
  ephemeral manifest（`ullm.served_model.v2` / `ullm.worker.v2`）生成を確認した。本番`target/release`は変更していない。
- Phase 0用のhash-only HTTP/SSE collectorを追加し、OpenWebUI network上の短命Docker curlからactive v1へ
  公開fixtureを送信した。非streamでprompt `18/1024/2048/3072` tokenを全てHTTP 200・completion 2で取得し、
  short streamではusage `18/2/20`とSSE event sequenceを記録した。証跡は
  `benchmarks/results/2026-07-13/generic-reasoning-phase0-baseline-v0.1/active-v1-http-long-baseline.json`
  （SHA-256 `cff8f7107bef0be6ac53838fa031d66a7dd60424058cd333e92a0286e1d5700b`）で、本文は保存していない。
  active promotion sourceとHEADの不一致、generated token IDs未取得のためstatusはpartialのままである。
- collectorのfake tokenizer/transport testを追加し、hash-only出力・exact prompt grid・SSE usage抽出を
  `5 passed`（promotion evidence 3件を含む）で確認した。collector testのcommitは`8b5970f`。

次の不足は、same-HEADのAQ4 Phase 0 HTTP/SSE/token-ID baseline、実GPU v2 workerと一致するpromotion
evidence、OpenWebUI browser E2E、異常系・soak・benchmark、activation/rollback証跡である。

- Phase 0計画、AQ4 release spec、inventoryの証跡表現を見直し、HTTP本文を保存する記述を削除した。
  現行collectorのhash-only metadata・SSE列・usage cross-check方針と、機密情報禁止方針を一致させた。
  inventoryの未達項目も、same-HEAD identity整合、generated token IDs、hash-only HTTP/SSEとraw committed
  tokenの突合へ整理した。これらの文書変更は次の小単位としてcommitする。
- promotion evidence runnerがv2 candidate時にbudget-zero reasoning caseをresident側だけへ追加し、
  forced-end token数と予約answer tokenをrelease accountingで検証するようにした。legacy v1比較は従来の
  no-reasoning raw casesに限定した。fixtureを含むtestは`3 passed`、collectorとのcombined testは
  `5 passed`、Ruffとpy_compileも成功した。変更commitは`cc2a300`。
- v2追加後のRust lib regressionを再実行し、`cargo test -p ullm-engine --lib`で`700 passed、1 ignored`
  を確認した。
- Phase 0 HTTP証跡の独立validatorを追加し、現行の長文証跡を構造検証した。`case_count=4`、本文なし、
  usage/SSE整合は通過したが、source identity不一致とgenerated token IDs欠落により
  `gate_eligible=false`となった。`--require-complete`は期待どおりexit `2`。変更は次のcommitへ保存する。
- `services/openai-gateway`全体を`uv run --frozen pytest -q`で再実行し`229 passed`、証跡・generator周辺の
  combined testは`34 passed`を確認した。
- 未activateのv2 candidate generatorは、現行v1 promotion evidenceとのprotocol不一致で非ゼロ終了し、
  manifestを出力しないfail-closed挙動を再確認した。
- Gatewayのnon-stream/stream reasoning分離で、worker releaseの`reasoning_tokens`と`forced_end_tokens`を
  token-level再計算値と突合するようにした。正常reasoning testとusage不一致のfail-closed testを含む
  app/worker testは`75 passed`、Ruffも成功した。
- Gateway全体回帰を変更後に再実行し、`uv run --frozen pytest -q`で`230 passed`を確認した。
- generic reasoningのengine production codeをtoken ID検索で監査した。Qwen固有IDの出現はworker
  protocol parserのfixture testに限定され、実行時のreasoning dialectはmanifestから供給されている。
- `deploy/openwebui/configure.py`のmanifest readerをv1/v2 strict対応にし、v2 reasoning dialectの
  必須field、token列、effort budget、policyを検証するtestを追加した。configure testは`14 passed`、
  Ruffも成功した。OpenWebUI DB・provider設定・production serviceは変更していない。
- 稼働中OpenWebUIのfrontend資産をread-onlyで確認し、model custom parameterとして
  `reasoning_effort`を送信する実装を確認した。DB、session、credential、会話本文は読んでいない。
- 既存OpenWebUIのstop/failure/soak browser基盤と各ingest gateを静的testで再実行し、`71 passed`
  を確認した。これはreasoning表示の実E2E合格を意味せず、v2 activation後の追加試験が必要である。
- activation/served-model validatorとOpenWebUI v2 manifest readerの回帰をまとめて実行し、`28 passed`
  を確認した。activation自体は実行していない。

## Release evidence validator

- hash-only generic reasoning release evidence用の`tools/validate-generic-reasoning-release.py`と
  `tests/test_validate_generic_reasoning_release.py`を追加した。manifest、worker binary、tokenizer、
  OpenWebUI image identity、5つのbenchmark mode、HTTP/SSE metadata、raw token accounting、usage
  cross-check、budget overshoot、resource、qualityをstrictに検証する。
- prompt/response本文、authorization、API keyなどの禁止fieldを再帰的に拒否する。構造検証とproduction
  gate判定を分離し、source identity不一致、必須mode不足、`status=incomplete`では構造が正しくても
  `gate_eligible=false`とする。`--require-complete`は未達時にexit `2`を返す。
- validator単体testは`6 passed`、Ruff、py_compile、`git diff --check`が成功した。本番の証跡生成、
  activation、service restartは実行していない。
- planの再利用tool一覧とAQ4 release specへvalidatorの責務を追記した。
- この作業単位は`c522fcc`（`test: validate generic reasoning release evidence`）へcommitした。
  `.rocprofv3/`は既存の未追跡状態のまま保全している。
- commit後の回帰としてGateway全体`uv run --frozen pytest -q`は`230 passed`、Rustは
  `cargo test -p ullm-engine --lib`で`700 passed、1 ignored`となった。`ullm-openai.service`はread-only
  確認で`active`、MainPIDは`3263234`であり、service・manifest・OpenWebUIは変更していない。
- release evidenceのsource commitをlowercase 40文字のGit SHA、OpenWebUI imageをcontent-addressed
  digestへstrict化し、証跡schemaを`docs/specs/generic-reasoning-release-evidence-v0.1.md`へ分離した。
  validator testは`8 passed`となった。この変更は`4ca0a6d`（`docs: specify generic reasoning release evidence`）へ
  commitした。
- 現行`open-webui` containerのpayload変換をread-only相当の一回限りのisolated importで確認し、
  `custom_params.thinking_budget_tokens="32"`がprovider bodyのinteger `thinking_budget_tokens: 32`へ
  変換されることを確認した。frontend資産には`reasoning_effort`と`reasoning_content`の経路があり、
  OpenWebUI独自のreasoning表示patchは不要である。本番DB、設定、container lifecycleは変更していない。
- `tests/test_openwebui_configure.py`へ既存model params内の`custom_params.thinking_budget_tokens`を
  configure処理が保持する回帰を追加し、configure testは`14 passed`となった。
- 最新のread-only runtime監査ではactive manifest validationが成功した。activeはschema v1、model
  `ullm-qwen3.5-9b-aq4`、worker protocol v1、manifest SHA-256
  `7589b9db7734d176bef21130b31e1ba679d1e0599e9a3c0d8af6699f86eded80`、worker SHA-256
  `e31cd887ddc92e1f7c9ceecc28b2dfd70f2eabcfc96179ddc1c007b9009e3a7f`である。`ullm-openai.service`は
  `active`、OpenWebUIはpinned image `sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409`
  でhealthyだった。production stateは変更していない。
- `source_commit_aligned`をsource SHAとactive promotion source SHAの比較結果から再計算し、偽装した
  booleanを拒否するようvalidatorを強化した。validator testは`9 passed`となった。この変更は
  `05d925f`（`fix: recompute release source alignment`）へcommitした。
- release validatorがstream/non-streamのSSE chunk整合、`quality.correct`、主要timing指標の欠落を
  production gateへ反映するよう強化した。異常系を含むvalidator testは`11 passed`、Ruffとdiff checkも
  成功した。この変更は`ccf22cd`（`fix: fail release gate on incomplete metrics`）へcommitした。
- validatorの入力ファイル、case数、SSE chunk数へ上限を追加し、`12 passed`を確認した。この変更は
  `694285e`（`fix: bound release evidence validation`）へcommitした。
- 変更後の関連回帰（release/Phase 0 collector・validator/promotion/generator/served-model/OpenWebUI
  configure）は`64 passed`となった。
- Gateway全体回帰も現HEADで再実行し、`uv run --frozen pytest -q`は`230 passed`となった。
- AQ4 release contractへ、既存OpenWebUIのmodel `custom_params`から正確なbudgetを渡す経路と、
  uLLM-specific UI patchを追加しない条件を明記した。この変更は`3241b0b`（`docs: define OpenWebUI reasoning budget path`）へcommitした。
- Phase 5用の`deploy/openwebui/browser-reasoning-smoke.cjs`を追加した。reasoning detailsのtoggle、
  通常回答分離、refresh後の2回目requestにassistant `reasoning_content`が再投入されないこと、page
  error、provider request schemaを検証する。prompt/response本文とscreenshotは保存せず、hash・件数・
  booleanだけを出力する。静的・補助testは`3 passed`、Node syntax checkも成功した。この変更は
  `be5de8a`（`test: add OpenWebUI reasoning browser smoke`）へcommitした。
- 既存のOpenWebUI browser stop/failure系を含む静的ブラウザ回帰を再実行し、reasoning smokeを含めて
  `21 passed`となった。
- browser smokeのstdoutを独立検証する`tools/validate-openwebui-reasoning-browser-smoke.py`とtestを
  追加した。hash-only schema、reasoning details展開、page error 0、hidden reasoning再投入なし、禁止
  fieldを検証し、validator testは`6 passed`、Ruffとpy_compileも成功した。この変更は`fb9e24f`
  （`test: validate OpenWebUI reasoning smoke evidence`）へcommitした。
- OpenWebUI provider requestに`reasoning_content`が必ず含まれるとは限らないため、validatorがそのキーの
  存在を要求していた条件を削除した。details toggleと次turnのassistant historyだけを実証条件に保ち、
  validator testは`6 passed`となった。この修正は`e670690`（`fix: align OpenWebUI smoke evidence with request flow`）へcommitした。
- browser smokeがdetails toggleだけでなく、展開前後の本文長差と展開後本文hashを記録・検証するよう
  強化した。runner/validator testは`9 passed`、Node syntax、Ruff、diff checkも成功した。この変更は
  `2e2bd26`（`test: require visible reasoning details`）へcommitした。
- 独立validatorにも展開後の可視本文が回答本文より長いことを追加で要求し、validator testは`6 passed`
  となった。この変更は`1ca0751`（`fix: validate expanded reasoning details`）へcommitした。
- AQ4 release contractへ`expanded_view`のhash/byte-count条件を追記した。この変更は`a30da30`
  （`docs: describe expanded reasoning browser evidence`）へcommitした。
- release/browser/Phase 0/promotion/generator/manifest/OpenWebUI configureのcombined testは`73 passed`
  となり、worktreeは既存`.rocprofv3/`だけの未追跡状態である。
- release validatorのreportへmode別のtiming p50/p95/p99をraw caseから線形補間で再計算する機能を
  追加した。validator testは`12 passed`、Ruffとdiff checkが成功した。この変更は`1d44bac`
  （`feat: report reasoning benchmark percentiles`）へcommitした。
- release validator reportへmode別のquality total/correct/accuracyも追加し、validator testは`12 passed`
  を維持した。この変更は`c7205c4`（`feat: report reasoning quality summaries`）へcommitした。
- raw caseを複数にしたpercentile再計算testを追加し、validator testは`13 passed`となった。この変更は
  `411abe1`（`test: cover reasoning percentile aggregation`）へcommitした。
- 最終combined回帰は`74 passed`、`git diff --check`成功。read-only確認で`ullm-openai.service`は
  `active`、OpenWebUIは`running/healthy`、worktreeは既存`.rocprofv3/`だけの未追跡状態である。
- release evidence validatorをduplicate JSON field拒否・strict non-finite値拒否・image digest名前必須へ
  強化し、validator testは`15 passed`となった。この変更は`a0dac2e`（`fix: make release evidence JSON strict`）へcommitした。
- strictness変更後のrelease/Phase 0/promotion/generator/manifest/OpenWebUI combined testは`76 passed`、
  diff check成功、worktreeは`.rocprofv3/`だけの未追跡状態である。
- release validator reportへmode別RSS/VRAM/GPU温度/電力のp50/p95/p99/maximumを追加し、validator testは
  `15 passed`となった。この変更は`e12c0b5`（`feat: report reasoning resource summaries`）へcommitした。
- reasoning browser runner、validator、既存browser stop/failure系のcombined testは`27 passed`、正しい
  対象でのRuffと`git diff --check`も成功した。
- runnerとvalidatorを実行可能bitに設定し、`52d9464`（`build: mark reasoning browser tools executable`）へ
  commitした。
- 直近の変更後に関連combined testを再実行し、`76 passed`となった。Gatewayのrepository root全体を
  そのままpytest対象にすると、PyTorch未導入の`reference-src/aiter` testがcollection時に`SystemExit`
  するため、正式なGateway回帰は`services/openai-gateway/tests`を明示して実行した。この結果は`230 passed`
  で、Rustは`cargo test -p ullm-engine --lib`で`701 passed、1 ignored`となった。実サービス、active
  manifest、OpenWebUI container lifecycleは変更していない。
- 未activateのreasoning候補profileがactive v1 binaryを参照していたため、active `target/release`を変更せず
  `target/reasoning-v2/release/ullm-aq4-worker`へ現HEADのv2 workerを再buildした。binary SHA-256は
  `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`で、ephemeral manifest preflightは
  schema `ullm.served_model.v2`、protocol `ullm.worker.v2`、dialect `qwen3.5-thinking-v1`を確認した。
  候補profileのbinding変更は`8e8c831`（`fix: bind reasoning candidate to v2 worker`）へcommitした。
- 同じephemeral manifestをGatewayのserved-model loaderでも検証し、model
  `ullm-qwen3.5-9b-aq4`、protocol `ullm.worker.v2`、reasoning dialectの読み込み成功を確認した。
- v2候補profileがactive v1 binaryへ戻る退行を防ぐ静的testを追加し、generator testは`24 passed`、
  Ruffとdiff checkも成功した。この変更は`1ed607a`（`test: pin reasoning candidate worker binding`）へ
  commitした。
- candidate binaryのv2 binding変更に伴い、served-model READMEの説明を「v2 binaryは別targetに存在するが、
  v1 receiptのためmatching evidence取得まで生成不能」と修正した。この変更は`4274b3b`
  （`docs: describe v2 candidate binary binding`）へcommitした。
- v2候補binaryを同一checkoutから再buildする`cargo build --release --bin ullm-aq4-worker
  --target-dir target/reasoning-v2`手順をREADMEへ追加し、再現手順を固定した。この変更は`c79b763`
  （`docs: record reasoning candidate build command`）へcommitした。
- 現HEAD `c79b7633372dc21f17539bd96135e551e805e956`で候補workerの再build commandを再実行し、binary
  SHA-256 `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`を確認した。候補manifest
  preflightと関連回帰は`55 passed`、Ruff、diff checkも成功した。
- 既存のOpenWebUI Stop・worker failure・20-chat soak gateと対応browser scriptが、既定のSQ8 modelを
  保持したまま`ULLM_MODEL_ID`/`ULLM_MODEL_NAME`でreasoning候補へ切り替えられるようにした。既定値の
  gate回帰は`56 passed`、Node syntax、Ruff、diff checkが成功し、AQ4 release specへ使用条件を追記した。
  この変更は`3cb66e8`（`feat: parameterize OpenWebUI lifecycle gates by model`）へcommitした。
- 計画の通常100 request・restart後20 request要件に合わせ、OpenWebUI soak gateを既定20のまま
  `ULLM_OPENWEBUI_SOAK_COUNT=100`へ切り替えられるようにした。Python gateとbrowser scriptのcase数・mode・
  summary identity・Docker環境を同じcountから検証し、20の回帰`62 passed`、count=100のschedule/import
  preflight、Node syntax、Ruff、diff checkが成功した。この変更は`0f2ef43`
  （`feat: support configured OpenWebUI soak counts`）へcommitした。
- candidate model/nameとsoak countが実際のbrowser Docker commandへ渡ることを、候補値
  `ullm-qwen3.5-9b-aq4` / `uLLM Qwen3.5 9B AQ4` / `100`でread-only command-builder preflightした。
  実container、OpenWebUI、Gateway、GPUは起動していない。
- activation toolの候補preflight、atomic replace、check/reconcile/final-check失敗時のmanifest復元、rollback
  hook、symlink/権限拒否を再確認し、activation/systemd関連testは`13 passed`となった。activationやservice
  restartは実行していない。
- reasoning候補profileが旧v1 promotion receipt pathを参照していたため、v2専用の
  `promotion-reasoning-v2-v0.1.json`へ修正した。v1 receiptを再利用しない条件と静的binding testを追加し、
  generator/promotion関連testは`27 passed`、ephemeral v2 manifest preflightも成功した。この変更は
  `c99a752`（`fix: bind reasoning candidate to dedicated receipt`）へcommitした。
- v2 receipt未取得の現状態では、candidate manifest generationがexit `1`でfail closedし、出力manifestを
  作成しないことを確認した。placeholder receiptやv1 receiptの流用は行っていない。
- v2 promotion evidenceをproduct directoryへ`.incomplete`で取得し、成功後に同一filesystem内でrename、
  専用receiptを発行する実行順序をREADMEへ追加した。既存v1 receiptを上書きしないrunbookであり、変更は
  `db39428`（`docs: add v2 promotion evidence runbook`）へcommitした。実行自体はR9700専有前のため未実施。
- fake v2 workerでpromotion runnerの実出力からreceipt writer、served-model generatorまでを接続する
  統合testを追加した。v2 pipeline testを含むpromotion/generator/manifest関連は`32 passed`、Ruffとdiff
  checkも成功した。この変更は`04d38f4`（`test: cover v2 promotion receipt pipeline`）へcommitした。
- 最新read-only確認でもGPU[2] R9700（gfx1201）はactive `ullm-aq4-worker`とllama.cppが使用中で、GPU[0]/[1]
  V620（gfx1030）は候補identity不適合だった。active serviceとmanifest、candidate/active binary hashは
  変更されていない。
- Gatewayのv2 usage契約を監査し、HTTP usageは`reasoning_tokens`を公開する一方、forced-endは内部delimiter
  accountingとしてworker released eventで保持する設計を確認した。強制終了tokenのproduction evidenceは
  HTTP collectorで推測せず、worker promotion evidenceを正とする。
- OpenWebUI containerのread-only identityはconfig tag `ullm/open-webui:0.9.4-ullm.1`、image ID
  `sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409`、container state healthyだった。
  release evidenceではtagだけでなくこのcontent digestを記録する。
- read-onlyのROCm確認では、GPU[0]/[1]はgfx1030のRadeon Pro V620、GPU[2]はgfx1201のR9700だった。
  active `ullm-aq4-worker`とllama.cppはGPU[2]を使用中で、V620はcandidateの`gfx1201` resident
  identityを満たさない。したがって、現時点で同一GPU上のv2 promotion evidenceを安全に並行取得できる
  隔離GPUはない。service停止やactive GPUの奪取は行っていない。
- active manifestのread-only validationは成功し、schema/protocolはv1、manifest SHA-256は
  `7589b9db7734d176bef21130b31e1ba679d1e0599e9a3c0d8af6699f86eded80`、worker SHA-256は
  `e31cd887ddc92e1f7c9ceecc28b2dfd70f2eabcfc96179ddc1c007b9009e3a7f`だった。active promotion source
  commitは`4be10d007c48db444f6b18fe6ac22e09dc17f168`で、現HEAD `4274b3b31eb77fdd745629398d20b168c9989b22`とは
  不一致のため、same-HEAD gateは未達のままである。
- generic release evidence、validator report、OpenWebUI browser evidence/report、promotion evidence/receiptを
  結合する`validate-generic-reasoning-release-bundle.py`と回帰testを追加した。bundleは各artifactを再ハッシュし、
  generic/browser validatorをraw evidenceから再計算し、v2 promotion receiptのevidence bindingとsymlink境界を
  検証する。`status=incomplete`は構造検証を通したうえでgate不合格として返す。bundle関連を含むrelease/browser
  validator回帰は`26 passed`、Ruffとdiff checkも成功した。実GPU promotion、OpenWebUI実E2E、activationは未実施。
- `46b8034` commit後にPhase 0、promotion、manifest、OpenWebUI browser/lifecycle、activation、generic release
  validator、release bundle validatorの関連回帰をまとめて実行し、`164 passed`、diff check成功となった。
  read-only再確認では`ullm-openai.service=active`、OpenWebUIは`running/healthy`、active manifest SHA-256は
  `7589b9db7734d176bef21130b31e1ba679d1e0599e9a3c0d8af6699f86eded80`、active worker SHA-256は
  `e31cd887ddc92e1f7c9ceecc28b2dfd70f2eabcfc96179ddc1c007b9009e3a7f`で変化がなかった。.rocprofv3/のみ未追跡である。
- 現HEAD `46b803491323a65c5e680c3bdccac3a87fd47cd0`から候補v2 workerを
  `target/reasoning-v2`へ再buildし、SHA-256 `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`を確認した。
  active v1 workerのSHA-256は`e31cd887ddc92e1f7c9ceecc28b2dfd70f2eabcfc96179ddc1c007b9009e3a7f`のままである。
  v2専用receipt未発行のcandidate manifest生成はexit 1（`failed to read promotion receipt`）でfail closedした。
- Phase 0 active v1 HTTP/SSE baselineを実取得し、product directoryの
  `phase0-http-baseline-v1.json.incomplete`へ保存した。4つのprompt target（18/1024/2048/3072 token）と
  stream caseを含み、artifact SHA-256は`92aab87df688903ba86bc23e8fcdf321e04be90224a84b1ac9a54735eb71e37d`である。
  独立validatorは`structurally_valid=true`、`gate_eligible=false`を返し、理由はactive promotion sourceとの
  commit不一致とAQ4 generated token IDs不足だった。collectorがAPI keyをcurl argvへ展開していたため、
  一時curl configへ渡す方式へ修正し、process argumentsへsecretを置かない回帰testを追加した。この変更は
  `4a32214`（`fix: keep baseline API key out of curl argv`）へcommitし、関連testは`9 passed`となった。
- Phase 0 baseline validatorへgeneric release validatorと同じstrict JSON（重複key、非有限値、16 MiB超過を拒否）と
  `token` forbidden fieldを追加した。validator/collector関連testは`12 passed`、Ruffとdiff check成功で、変更は
  `a2fb320`（`fix: make phase0 baseline JSON strict`）へcommitした。
- 現HEAD `a2fb320ed89b560d5ebe0e09b05dc264a99a7ea7`から候補v2 workerを再buildし、binary SHA-256は
  `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`で変化しなかった。candidate binaryを
  現HEADのsource commitへ再結合した状態を、実promotion evidence取得前の最終buildとして固定した。
- Gateway workerの`request_released` lifecycle logがv2 reasoning requestの`reasoning_tokens`と
  `forced_end_tokens`を欠落させていたため、reasoning requestだけ2 fieldをmirrorするよう修正した。v1/SQ8
  release event形状は変更していない。lifecycle logへtoken IDやdecoded contentを入れないspec説明とv2 worker testを
  追加し、worker testは`30 passed`、Gateway全体testは`230 passed`、Ruffとdiff checkも成功した。この変更は
  `02ef7f9`（`feat: record reasoning accounting in lifecycle evidence`）へcommitした。
- lifecycle eventへv2の2フィールドを追加した結果、OpenWebUI Stop/failure gateのstrict field validatorが候補v2を
  拒否することを検出した。`request_released`だけv1のbase fieldsまたはv2 accounting fieldsの組を受理し、値の
  非負・completion上限・片側欠落を検証するようStop/failure gateを修正した。soak gateはStop supportを通じて同じ
  validatorを使う。関連回帰は`55 passed`、Ruff/diff check成功で、`ac2aa41`（`fix: accept reasoning accounting in lifecycle gates`）へcommitした。
- Phase 5のprovider切り替え要件へ対応し、hash-only browser smokeをschema v2へ更新した。各provider requestの
  model ID hashを記録し、uLLM → llama.cpp → uLLMの一時チャット切り替え、回答分離、最終requestのmodel bindingを
  実ブラウザー実行時に検査する。validatorはv1証跡も読み戻せるようにし、v2では初期2件・切り替え後・復帰後の
  model hashを厳密に検証する。bundle validatorもv1/v2 browser evidenceを受理する。
- v2 provider switch fixture、v1互換、初期model mismatch、switch model mismatchのtestを追加した。browser・bundle
  関連17件、Ruff、Node syntax、diff checkが成功した。変更は次のcommitへ保存する。
- `uv run pytest -q`の無指定全体収集は、既存`reference-src/aiter`とPyTorch未導入により収集時失敗した。
  `reference-src`と既知のPyTorch依存4件を除いた回帰は245件成功後、`tools/build-sq-fp8-w8a16-artifact.py`の
  torch importで停止した。今回の変更対象17件は独立して完走している。
- 現HEAD `4c3ad2e`で隔離v2 workerのcargo buildを再確認した。Rust依存に変更がないため既存artifactが再利用され、
  binary SHA-256は`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`のままだった。active
  `target/release`には触れていない。
- 現HEADでcandidate manifest generationを再度preflightし、v2専用promotion receiptが未発行のためexit `1`、
  output absentとなるfail-closedを確認した。active manifestはread-only validatorで検証成功、serviceはactive、
  OpenWebUI containerはhealthyのままである。
- 実装後のbrowser smokeと実container E2Eを計画内で区別する文書修正を`3b0ab55`（`docs: distinguish browser smoke from live E2E`）へ保存した。
- browser smokeが4件を超えるchat completionを黙って捨てないよう、再送・post body欠落・余分なrequestを失敗扱いに
  した。切り替え先model IDが初期uLLM modelと同じ場合もfail closedにし、関連testは`18 passed`、Ruff、Node
  syntax、diff check成功となった。変更は`d1a3951`（unexpected request拒否）と`bcc9bb1`（distinct model要求）へ保存した。
- 現HEADで`cargo test -p ullm-engine --lib`を再実行し、`700 passed、1 ignored`を確認した。HIP deviceを要する
  ignored testは実機promotionの代替にしていない。
- promotion runnerを単独shellから呼んだ場合に全GPUを継承する余地があったため、resident v2 workerとlegacy
  comparisonの両方へ`HIP_VISIBLE_DEVICES=1`を固定注入した。systemd配置と同じR9700 isolationを使い、V620へ
  誤って流さない契約を追加した。promotion/generator/served-model関連testは`33 passed`、Ruffとdiff check成功。
  変更は`64cfc0c`（`fix: pin HIP visibility for promotion evidence`）へ保存した。
- promotion runnerへ`rocm-smi --showpids --json`のR9700排他preflightを追加した。対象GPUにpositive VRAMのKFD
  processがあればworkerを起動せず、process identityを含むエラーでexit `1`、output absentとなる。fixtureを含む
  promotion/generator/served-model関連は`35 passed`、Ruffとdiff check成功。現ホストで実コマンドをread-only実行し、
  active workerとllama-serverを検出してfail closedすることを確認した。変更は`df6f0c1`、runbook更新は`8e7e4ba`。
- promotion、manifest、release bundle、browser smokeの横断回帰を`53 passed`で再確認し、Ruff、Node syntax、diff
  checkも成功した。preflightの契約説明は`ff6c08e`（`docs: clarify promotion exclusivity preflight`）へ保存した。
- runnerが記録したGPU排他結果を、manifest generator、receipt publication、release bundle validatorでも必須化した。
  `gpu_exclusive_preflight`の欠落、positive-VRAM偽装、bundle内欠落をfail closedにし、promotion/generator/bundle
  関連回帰は`43 passed`、Ruffとdiff check成功。変更は`5a30307`（`fix: bind GPU exclusivity into promotion artifacts`）へ保存した。
- release evidence schemaにGit worktree状態を追加した。`git_worktree_clean`と、既存`.rocprofv3/`をscope外として
  hash化した`git_worktree_status_sha256`を必須化し、dirty worktreeはstructural validでもproduction gate不合格とする。
  release validator/bundle回帰は`23 passed`、promotion/generator/bundleを含む横断回帰は`56 passed`。変更は
  `f498a66`（`feat: bind Git worktree state to release evidence`）へ保存した。
- `tools/prepare-generic-reasoning-release-evidence.py`を追加した。事前に機密本文を除去した測定case配列を受け取り、
  manifest、worker、manifest記載tokenizer files、OpenWebUI image、source commit、worktree status hashを結合し、
  独立validatorを通してからatomic publishする。`status=complete`はsource alignment、clean worktree、全mode・
  quality・timing gateまで通らなければ出力しない。assembler関連回帰は`27 passed`、Ruffとdiff check成功。
  計画/spec/runbookへ手順を追記し、`af69a6d`（`feat: add generic release evidence assembler`）へ保存した。
- release evidence assemblerがhash-only情報だけでなく、既存の`validate-served-model.py`を通したserved-model契約へ
  結合されるようにした。相対tokenizer rootの解決もmanifest位置基準へ統一し、壊れたworker protocolを含むmanifestを
  証跡化前に拒否する。関連テストと横断回帰は`32 passed`、`96 passed`、Ruffとdiff checkが成功した。
- 最終read-only確認ではactive manifestのserved-model validationが成功し、manifest SHA-256は
  `7589b9db7734d176bef21130b31e1ba679d1e0599e9a3c0d8af6699f86eded80`、active v1 worker SHA-256は
  `e31cd887ddc92e1f7c9ceecc28b2dfd70f2eabcfc96179ddc1c007b9009e3a7f`であった。`ullm-openai.service`と
  `llama-qwen35-udq4.service`はactive、OpenWebUIはhealthy、R9700はresident `ullm-aq4-worker`と
  `llama-server`が使用中である。`.rocprofv3/`以外のGit差分はなく、実GPU作業は保留した。
- bundle JSONの手作業生成をなくすため、`tools/prepare-generic-reasoning-release-bundle.py`を追加した。
  同一bundle directory内のrelease/browser/promotion 6 artifactを相対・非symlink参照へ変換し、旧active
  manifest、systemd unit、environment fileのrollback hashを結合する。既存bundle validatorを一時出力へ
  実行してからatomic publishし、`status=complete`は再計算gate不合格なら拒否する。関連回帰100件、Ruff、
  diff checkが成功した。実測artifactがまだないため、実production bundleは生成していない。
- `tools/activate-served-model.py`のv2 activationへrelease bundle preflightを追加した。v2候補はcomplete
  bundle、candidate manifest/worker identity、promotion source、現在active manifest、systemd unit、environment
  fileのhashが一致しなければatomic replace前に拒否する。v1 activationは互換維持し、activation関連回帰は
  `13 passed`、関連release/manifest回帰は`53 passed`、Ruffとdiff check成功となった。
- activation testへ実際のgeneric release bundle validator、v2 candidate manifest、promotion evidence/receipt、
  rollback対象ファイルを結合したintegration caseを追加した。実bundle結合を含むactivation回帰は`14 passed`、
  Ruffとdiff check成功となった。production bundle自体はまだ実測artifact未取得のため作成していない。
- root read-only auditで`/etc/ullm/ullm-openai.nft`、live `inet ullm_openai` table、repositoryの
  `deploy/nftables/ullm-openai.nft`が8000/8001双方の同一drop ruleへ一致することを確認した。active serviceは
  unchangedで、systemd unit SHA-256は`f0239713b16b3bf31cfd12a98f506e77e55af9b31abf58352f4e437e1cdee552`、
  environment file（`/etc/ullm/openai-gateway-manifest.env`）SHA-256は
  `68dd3a027fa86aaa8f5649bf55f34c32b818afb49a9e35e272f5dc6a1e5fb835`である。秘密値は記録していない。
- generic release evidenceへ`lifecycle` objectを追加し、sanitized `request_released` eventをcase IDへ
  結合した。validatorはprompt/completion token、stream/outcome、reset完了、reasoning/forced-end accounting、
  lifecycle timingをcaseのraw evidenceと再計算し、complete gateでは全caseのeventを要求する。assemblerは
  `--lifecycle`を受け付ける。release/activation/browser横断回帰は`116 passed`、Ruffとdiff check成功となり、
  `8fa77ae`（`feat: bind lifecycle accounting to release evidence`）へ保存した。
- lifecycle追加後の対象回帰は`79 passed`、横断回帰は`116 passed`、Rust `ullm-engine` lib回帰は
  `701 passed; 1 ignored`となった。OpenWebUIコンテナ内からllama.cpp healthは成功し、Gatewayの認証なし
  `/v1/models`は401で拒否された。promotion preflightはR9700上のactive worker/llama-serverを検出して
  `output_not_published=true`で停止し、実GPU測定とproduction状態変更は発生していない。
- 稼働中の`target/release/ullm-aq4-worker`を上書きしないよう、`target/reasoning-v2/`とは別の隔離
  target directoryへ現HEAD `8fa77ae7a29d05bc72a817b7e082f1a2d0f4799d`のv2 workerをbuildした。
  候補binary SHA-256は`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`で、既存
  `target/reasoning-v2` binaryと一致し、active v1 binary hashは変化していない。
- 実tokenizerをservice環境のTransformers `5.12.1`で再読し、Qwen2Tokenizer、chat-template SHA-256
  `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`、`<think>` token `248068`、
  `</think>` token `248069`を確認した。thinking有効/無効のtemplate出力は本文を保存せずtoken数と
  u32 token列hashだけを記録し、候補profileのReasoningDialectと一致した。
- 現行v1 serviceへhash-only Phase 0 HTTP/SSE baselineを取得し、
  `benchmarks/results/2026-07-13/qwen35-9b-aq4-phase0-http-baseline-v0.1/`へ保存した。18/1024/2048/3072
  tokenのnon-streamと18 tokenのstreamはすべてHTTP 200で、validatorは構造検証を通した。active promotion
  source commit不一致とAQ4 raw token ID未取得によりgateは不合格のままだが、本文・credentialを保存しない
  比較基準として`36c74d7`（`test: capture v1 phase0 http baseline`）へ保存した。
- 次回のmaintenance windowでモデル取り違えを起こさないよう、`deploy/served-models/README.md`へv2
  OpenWebUI gate runbookを追加した。`ULLM_MODEL_ID`/`ULLM_MODEL_NAME`、immutable image identity、
  soak 100/20、Stop、failureの出力分離を明示し、gate回帰`68 passed`後に`62c29ae`
  （`docs: document v2 OpenWebUI gate runbook`）へ保存した。
- runbookのbrowser imageを実行前検査し、OpenWebUI本体imageにはNodeがなく不適切だったため、
  `firecrawl-playwright-service`のSHA-256 `dbd552f6c831816050a1381a54cdb8d37df56df7f6559c82aba451d2ea93e0aa`
  へ修正した。実コンテナ内でPlaywright import成功、probe用curl imageは`8.12.1`であることを確認し、
  `cf106c6`（`docs: pin browser gate image`）へ保存した。v2 gate自体はまだ実行していない。
- `tools/run-openwebui-reasoning-browser-smoke.py`を追加した。immutable Playwright image、候補/比較
  model identity、hash-only v2 schema、validator gateをrunner内で結合し、成功時だけatomic publishする。
  token fileはread-only mount、stdoutは1 MiB bounded、stderr/bodyは保存しない。runnerと関連browser
  回帰は`72 passed`、Ruff/diff check成功で`03f8435`（`feat: add safe reasoning browser smoke runner`）へ
  保存した。実containerでのv2 smokeは未実施である。
- 後続commitでRustコードを変更していないが、source commitの曖昧さをなくすため現HEAD
  `03f8435b62f99f3c3e730b3579c8c132678c3ef6`から`target/reasoning-v2/release/ullm-aq4-worker`を再buildした。
  候補hashは`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`のままで、稼働中v1
  binary hashは`e31cd887ddc92e1f7c9ceecc28b2dfd70f2eabcfc96179ddc1c007b9009e3a7f`から変化していない。
- v2 profileのpromotion receipt、promotion evidence、candidate manifestはいずれも未作成であることを
  再確認した。receiptなしの`generate-served-model.py`はrc=1、`failed to read promotion receipt`で
  出力を公開せず停止した。R9700は引き続きactive worker/llama-serverが占有しているため、これは
  期待されたfail-closed状態である。
- browser smoke runnerへ`--manifest`を必須化し、served-model validatorとraw manifestの両方でv2、
  公開model ID、`ullm.worker.v2` protocol、reasoning dialectを確認してからDockerを起動するようにした。
  現在のactive v1 manifestを渡すテストはbrowser起動前に拒否し、関連回帰は`18 passed`、Ruff/diff check
  成功となった。`552670d`（`fix: bind browser smoke to v2 manifest`）へ保存した。
- activation、release evidence、bundle、manifest、promotion preflight、browser smoke/異常系の横断回帰を
  再実行し`121 passed`、Ruff、diff check成功となった。実v2 promotionとOpenWebUI実container gateは
  R9700占有中のため実行していない。
- v2 active後の実測を一括取得する`tools/run-generic-reasoning-release-campaign.py`を追加した。disabled、
  budget 32/128/256、unboundedのHTTP/SSEをfixture IDで走査し、response/lifecycle/resourceをhash-only
  artifactへ原子的に出力する。manifestのv2・gfx1201・served-model validatorを先に確認し、R9700の
  positive-VRAM processが`ullm-aq4-worker`一つだけであること、さらに稼働worker executableのhashが
  manifestのworker hashと一致することを要求する。llama-serverや別binaryなら停止し、本文・credentialは
  保存しない。純粋回帰、既存release/activation/browser gateを含む`161 passed`、Ruff、py_compile、diff checkが
  成功した。`f886259`（`feat: add generic reasoning release campaign collector`）へ保存した。現active v1 manifestへ
  実行すると`served-model manifest is not v2`で出力を公開せず停止した。
- 計画のstream/non-stream一致要件をcollectorへ反映し、5 modeそれぞれをstream/non-streamで収集する
  合計10 caseへ拡張した。non-streamのJSON response、usage、reasoning_content、timingsを同じcase契約へ
  変換し、両transportのstatus、finish reason、usage、reasoning/answer本文を保存前にメモリ上で比較する。
  不一致はatomic publish前に停止する。回帰は`163 passed`、Ruff、py_compile、diff check成功で、
  `71eb60a`（`feat: collect paired reasoning transports`）へ保存した。実v2実測はR9700排他待ちである。
- 現行active v1 manifestでpaired transport collectorを実行し、v2 preflightで`rc=1`・output未公開となった。
  v2 reasoning profileのpromotion receiptなしmanifest生成も`rc=1`・output未公開であり、service、active manifest、
  OpenWebUI、GPU状態に変更はない。
- paired collectorが作る5 mode×stream/non-streamの10 caseと10 lifecycle eventを、release validatorへ渡す
  offline統合テストを追加した。hash-only artifactで構造検証を通し、fixtureの回答本文がartifact JSONへ
  入らないことも確認した。collector回帰は`8 passed`、Ruff成功で`ebe7515`（`test: validate paired release evidence`）
  へ保存した。
- 主要release/activation/browser/phase0横断回帰を再実行し`164 passed`、Ruff、diff check成功となった。`.rocprofv3/`
  以外の作業tree差分はない。
- activationの循環（v2 activationはcomplete bundle必須、release/browser evidenceはv2 activeを要求）を
  解消するため、`tools/activate-served-model.py`へ明示的な`--bootstrap-v2`を追加した。通常v2 activationの
  bundle gateは維持し、bootstrapだけはv1 active限定、外部backup、systemd/environment hash、全対象serviceの
  inactive確認を要求する。証跡取得後にbackup v1へ戻してcomplete bundleを組み立てるrunbook/specを追加した。
  activation/release/browser関連回帰は`98 passed`、Ruff、py_compile、diff check成功で、`2d782d1`
  （`feat: add gated v2 evidence bootstrap`）へ保存した。実機bootstrapはまだ実行していない。
- browser smokeがuLLMとllama.cppの4要求を一つのcontainer実行内で行うため、共有R9700の排他条件と
  provider切替が両立しない点を解消した。`run-openwebui-reasoning-browser-smoke.py`の任意の
  `--alternate-r9700-services`で、uLLM→llama.cpp→uLLMの間にhost Unix socketを介して停止・起動を同期し、
  各境界でGPU ownerを確認する。transition失敗時はllama.cppを停止してuLLM復帰を試みる。Node script、
  host coordinator、runbookを更新し、browser/activation/release回帰`73 passed`、Ruff、Node syntax、diff check
  成功で`cbdf6c4`（`feat: serialize browser provider switching on R9700`）へ保存した。実container gateは未実施。
