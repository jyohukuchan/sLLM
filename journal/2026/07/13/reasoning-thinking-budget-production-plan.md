# Reasoningとthinking budgetの本番計画

## 前回の要点

- Qwen3.5 9B AQ4ではreasoningが配信設定、Gateway、worker protocol、出力分離の各境界で
  未接続だった。
- OpenWebUI 0.9.4は`delta.reasoning_content`を表示できるため、UI patchは不要だった。

## 今回の変更点

- `docs/plans/generic-reasoning-thinking-budget-production-plan-v0.1.md`へ、モデル非依存の
  reasoning dialect、厳密なbudget、OpenWebUI E2E、性能・品質benchmark、release evidence、
  rollbackまでの計画を保存した。
- 見積もりは試作2〜4人日、AQ4 beta累計8〜12人日、production候補累計15〜25人日とした。
- 他AIへの引き継ぎ用として、repository/runtime snapshot、関連実装境界、現行AQ4/llama.cpp/
  OpenWebUI配備、性能証跡、network未確認事項、test資産、開始時checklist、secret禁止事項を
  計画のSection 13へ追記した。
- repository HEADとactive workerのpromotion source commitが異なること、firewallの8001 ruleは
  live照合が必要なことを既知の不一致として明示した。
- この作業では実装、配信設定、サービス状態を変更していない。

## 次の行動

実装指示を受けたら、現行AQ4のAPI、token列、prefill/decode性能をPhase 0の基準証跡として
保存し、versioned API/manifest/worker仕様を固定する。

## 2026-07-13 継続作業

### 前回の要点

- v2 manifestをactiveにしたQwen3.5 9B AQ4で、HTTP/SSE、OpenWebUI reasoning、soak、
  restart、Stopの実測ゲートは成功していた。
- worker failureゲートだけは、準備プローブの実行方式とreasoning時の証跡上限が未整合だった。

### 今回の変更点

- `run-openwebui-failure-gate.py`のDocker readiness probeを、固定probe imageのcurl専用仕様に
  合わせた。HTTP 200を`{"ready":true,"status":200}`へ変換し、read-only、capability drop、
  retry、timeoutの制約を維持した。
- failure gateのSocket.IO証跡上限を、browser側の2048件と一致させた。実測の最終証跡は539件
  だった。
- raw service journalに含まれる公開モデルIDを秘密情報として誤検出しないようにした。一方で
  API token、URL、prompt、復旧マーカーは引き続きcleartext検査対象とした。
- reasoning UIのDOMに合わせて、reasoning blockを除いた回答本文をfailure/stop/soakの検証にも
  適用した。reasoning browser smokeは通常チャットURLを使い、検証後に生成チャットを削除する。
- 実機worker failure gateが成功した。worker停止、systemd一回復旧、Docker network内ready=200、
  OpenWebUI失敗表示、入力復帰、復旧チャット、Socket.IO証跡、secret scanを確認した。
- 検証中にsystemd restart rate limitへ到達したため、ゲートを停止して`reset-failed`とstartで
  uLLMを復旧した。最終状態は`active/running`、R9700はuLLM workerのみ、ready=200である。

### 次の行動

関連Python回帰71件、Ruff、Node構文、`git diff --check`を再確認した。最終HEADに対応する
resident promotion evidence/receipt、candidate manifest、HTTP/SSE 10ケースを追加した。
candidate用release evidenceは構造検証済みだが、producer statusはincompleteである。
交互provider browser runnerは切替後のservice transitionで失敗し、uLLM復帰は確認できたが、
candidate manifestに結び付く新browser evidenceは未取得である。active manifestは旧identityの
まま保持し、サービスを`active/running`、ready=200、R9700=uLLM workerへ戻している。

## 2026-07-14 最終候補activation

### 前回の要点

- 最終候補のbrowser evidence、HTTP/SSE 10ケース、promotion evidence/receipt、release evidenceと
  validatorは`ff51d85`のsource identityへ結合済みだった。
- 最初のcomplete release bundleは、実際のactive manifestではなく別の旧manifestをrollback targetへ
  記録していたため、activation preflightが安全に拒否した。

### 今回の変更点

- 稼働中の旧active manifest SHA `e9875a08f801d8604585a6b2f5ee21f257e545be468a3f9c6ff84ba071ac1226`を
  rollback targetへ取り直し、`release-bundle-ff51-active-e987.json`を再生成した。bundle validatorは
  `gate_eligible=true`、`structurally_valid=true`、`reasons=[]`を返した。
- `tools/activate-served-model.py`へcomplete bundle、現行systemd unit、environment fileを渡して、
  candidate SHA `e6f749654e85a5f69f2d077bd55d4e27aff869d71803809386c5d36865183e72`をatomic activationした。
  activation結果は成功し、source commitは`ff51d851d724d290eeb01108c09875c4c3bd0d29`へ更新された。
- activation後の`ullm-openai.service`は`active/running`、`NRestarts=0`、OpenWebUI containerの固定curl
  imageから`/readyz`=200を確認した。`/v1/models`は`ullm-qwen3.5-9b-aq4`を返し、reasoning chatは
  `reasoning_content`と回答本文を分離して返した。ROCm確認ではcard2だけがVRAMを使用していた。
- 回帰確認は関連Python unittest 71件、release bundle/activation pytest 21件、Ruff、CJS構文検査、
  `git diff --check`が成功した。host namespaceからのHTTPは環境の到達制御でtimeoutしたが、正式なOpenWebUI
  network namespace経路ではAPIとreadinessを確認できた。

### 次の行動

本番候補の切り替えと基本運用確認は完了した。計画のProduction昇格基準にある品質・性能の正式benchmark
（特にreasoning無効時との比較、品質閾値の固定と正式run）は、既存の10ケースrelease campaignとは別の
残課題として扱う。実運用をこの候補で継続する場合は、rollback targetとbundle pathを運用手順へ転記する。

## 2026-07-14 追加benchmark監査

### 前回の要点

- activation後の10ケースrelease evidenceは`gate_eligible=true`だったが、通常100 requestの集計と
  同条件p95比較は未取得だった。
- Phase 0 HTTP baselineは現行manifestへ取り直せるが、HTTP経路だけではAQ4 generated token IDsを
  証跡へ含められず、validatorは構造valid・gate不適格のままだった。

### 今回の変更点

- 現行active manifest `e6f74965…`に対してPhase 0 HTTP baselineを再取得した。source commitとactive
  promotion source commitは`ff51d85…`で一致し、4ケースのHTTP/SSE metadataは構造validだった。
- 固定fixtureの10ケースcampaignを10回実行し、合計100ケースを取得した。10 runすべてでmanifestと
  worker identityが一致し、lifecycle 100/100、正答100/100、budget overshoot 0、empty answer 0、
  `reset_complete` 100/100だった。
- 100ケースのdisabled p50は旧e987 baseline比でlatency `+0.41%`、prefill `-0.49%`、decode `-0.41%`
  だった。3%/5%のp50基準には収まるが、旧baselineはdisabled 2ケースだけなのでp95基準の判定は
  保留した。mode別p50/p95、RSS/VRAM上限は`http-soak-100-analysis-20260714.md`へ記録した。
- `ULLM_OPENWEBUI_SOAK_COUNT=100`のbrowser gateを現行candidateで試したが、OpenWebUIの
  `/api/v1/auths/`がbrowser tokenへ401を返し、最初のchat case開始前に終了した。サービスは変更されず、
  `active/running`と`NRestarts=0`を維持した。既存の20-chat、restart、Stop、failure、provider-switchの
  成功証跡は保持している。
- 計画と`aq4-reasoning-openwebui-release-v0.1.md`の冒頭状態を、v2 candidate activation後の実状態へ
  更新した。100-chat認証状態、p95のidentity-matched baseline、AQ4 generated token IDsは未完了として
  明示した。

### 次の行動

OpenWebUIの運用認証を既存の安全な手順で修復できる場合は、100-chat gateを再実行する。p95基準を
正式判定するには、旧v2 baselineを同じfixture数で再取得する必要がある。AQ4 generated token IDsは
prompt/response本文や秘密情報を保存しないworker-side hash-only collector経路を設計し、Phase 0 gateを
閉じる。これらが揃うまでは、candidateをactiveのまま維持しつつ、正式な全Production昇格完了とは宣言しない。

## 2026-07-14 curated quality repetition

### 前回の要点

- 現行active candidateはHTTP 100ケースで、正答・budget accounting・lifecycle resetが全件成功した。
- ただし、通常OpenWebUI 100-chat、identity-matched p95比較、Phase 0のAQ4 generated token IDsは未完了だった。

### 今回の変更点

- 正式fixture `tests/fixtures/generic-reasoning-release-v0.1/prompts.json`を使い、独立campaignを3回実行した。
  合計30ケース（各mode 6件、stream/non-streamを含む）を現行manifest `e6f74965…`とworker hash
  `177f3106…`へ結合した。
- 正答は30/30、empty answerは0、budget overshootは0、lifecycle outcomeは全件`stop`、
  `reset_complete`は30/30だった。mode別p50 latencyはdisabled 768.935ms、budget-32 1314.782ms、
  budget-128 2764.933ms、budget-256 3238.422ms、unbounded 3385.638msだった。
- 自作の算術・論理・コードfixtureは、収集器が要求するmode別固定応答契約と一致せず、完了メタデータ不足で
  生成物を公開しなかった。これはサービス状態を変更しない入力側の失敗であり、正式fixtureへ切り替えて再取得した。
- 詳細は`quality-curated-analysis-20260714.md`へ記録した。これは品質の補助的な反復確認であり、広い品質benchmark、
  p95正式比較、Phase 0、OpenWebUI 100-chatの未完了項目を閉じるものではない。

### 次の行動

OpenWebUI認証を安全な運用手順で復旧できる場合は100-chat gateを再試行する。引き続きcandidateをactiveのまま
維持し、AQ4 generated token IDsをhash-onlyで採取できるworker-side経路と、同一fixture数の旧v2 p95 baselineを
整備するまでは、全Production昇格完了とは宣言しない。

## 2026-07-14 現行ソース証跡の再構築

### 前回の要点

- 現行 active は旧 `ff51d85` candidate で、Phase 0 の HTTP 証跡は AQ4 generated token IDs 不足により gate 不適格だった。
- 100-chat OpenWebUI gate は `/api/v1/auths/` の HTTP 401 で開始前に停止していた。

### 今回の変更点

- Gateway の completion length 境界で、forced end token と自然終了 token の ID が同じ場合に worker の release accounting を失わないよう修正した。stream/non-stream の回帰試験を追加し、Gateway test 47件を通過した。
- activation tool に、同一 worker hash・同一 model ID の v2 candidate間だけを許す明示的な `--bootstrap-v2` 経路を追加した。これを使い、現行 source `ae8b2bb7c2735f4dc761773957bf45f470dd5a8c` の manifest SHA `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44` を一時 active にした。旧 active は `previous-active-reasoning-v2-v0.1-ae8b2bb.json` に保存した。
- resident promotion evidence を Phase 0 collectorへ安全に結合し、prompt/response本文や秘密情報を保存せずに worker-generated token evidence を取り込めるようにした。現行 Phase 0 artifact は `status=complete`、source aligned、validator `gate_eligible=true` になった。
- 現行 source の HTTP/SSE campaignを6回、合計60ケース実行した。正答60/60、budget overshoot 0、empty answer 0、lifecycle reset 60/60、最大VRAM delta 0だった。disabled p50 は旧比較値に対して latency `-0.65%`、prefill `+0.29%`、decode `+0.65%`で、計画の閾値内だった。旧比較のdisabled標本が2件だけのため、p95は未判定とした。
- 算術fixtureを使った診断で、length境界の forced-end accounting修正後もサービスが再起動せず、10/10 resetを確認した。このfixtureは正式品質fixtureではないため、品質合格証跡には使っていない。
- 現行 source で release evidence は独立 validatorを通過したが、OpenWebUI browser smokeは引き続き `/api/v1/auths/` の HTTP 401 で失敗した。したがって、現行 candidateを正式な complete production bundleとしては扱っていない。

### 次の行動

OpenWebUIの認証状態を安全な運用手順で修復できる場合は、現行 sourceへ結合した browser evidence を取得し、100-chat gateを再実行する。同時に旧v2を同一fixture数で測定してp95比較を閉じる。それまでは一時 bootstrap candidateを active/running、`NRestarts=0`で維持し、正式昇格完了とは宣言しない。

## 2026-07-14 最終evidence window

### 前回の要点

- 現行 `ae8b2bb` candidateのPhase 0、browser reasoning smoke、release evidenceは完了していたが、100-chat gateとidentity-matched p95比較が残っていた。
- 100-chat gateの実行では、1 requestあたり最大5 lifecycle eventに対してobserver上限256が不足していた。

### 今回の変更点

- `tools/run-openwebui-stop-gate.py`のobserver上限を100-chat soakを収容する4096へ引き上げ、対応する回帰テストを追加した。Gate testは47件すべて通過した。
- root runnerで現行candidateの100-chat soakを再実行し、100/100 chat、500 lifecycle actions、500 observer/journal records、17,512 socket events、restart 0で完了した。
- OpenWebUI reasoning browser smokeは正しいJWT・表示名・ROCm probeを使って4 provider requestsを完了し、validator `gate_eligible=true`になった。
- 現行candidateを含むcomplete release bundleをbundle-bound activationし、browser evidenceとresident promotion evidenceを結合した。
- 旧v2 manifest `e6f74965…`を`--bootstrap-v2`で測定窓だけ一時復帰し、同一fixtureの10 campaign、合計100ケースを収集した。その後、旧manifestをrollback targetにした`release-bundle-ae8b2bb-after-p95.json`で現行candidateへ復帰した。
- 現行candidate側も追加4 campaignを実行して合計100ケースへ揃えた。旧版・現行とも正答100/100、empty 0、budget overshoot 0、lifecycle reset 100/100、stop 100/100だった。
- identity-matched p95では、disabledの現行差分がlatency `+1.14%`、prefill `-0.21%`、decode `-0.83%`で、全modeの差分が計画の3%/5%閾値内だった。詳細は`http-identity-matched-p95-analysis-20260714.md`へ記録した。

### 次の行動

現行candidateは`feb3190d…` manifest、`NRestarts=0`、complete bundle-bound activationで稼働中である。今回のproduction evidence windowで、計画上のPhase 0、browser smoke、100-chat soak、identity-matched p95比較を閉じた。以後は通常の運用変更時に同じvalidator、bundle、rollback手順を再利用する。

## 2026-07-14 Git finalization audit

### 前回の要点

- core reasoning実装、current-source evidence、100-chat soak、identity-matched p95は完了していた。
- 最終bundleとその依存証跡が未追跡で、Gitだけではrelease判定を再現できなかった。

### 今回の変更点

- user判断によりllama.cppとの性能比較をrelease gateから除外した。既存の互換性証跡は履歴として保持する。
- 計画とAQ4 reasoning release仕様に残っていたbrowser未取得、receipt未発行、complete bundle未生成の
  古い記述を、active `feb3190d…` / source `ae8b2bb…`の実績へ更新した。
- OpenWebUI managed modelのmanifest markerと厳密な`thinking_budget_tokens` UI検証は、core
  Gateway/worker evidenceとは分離した運用上の残件として明記した。
- 再監査に必要な最終証跡94ファイルを`b6da487`で保存した。中間証跡103ファイルはSHA-256
  `f030bf1c76888956fe6e685900e00795ff55f5bd0128a5c98f4c526173633107`の外部archiveへ退避した。

### 次の行動

更新した計画、release仕様、journalをcommitし、`origin/main`へpushした後にremote HEADとの一致を確認する。
