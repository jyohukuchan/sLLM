# Phase 42: inference modeと基本public endpoint

## 状態

- 完了（2026-08-22開始・完了）
- MI300X実機検証はユーザー指示によりdeferred。Phase 42の開始・完了gateにはしない。

## 目的

OpenAI CompletionsとEmbeddings、sLLM固有のRerank、token utility、input-token-count、FIM/infillを、
transport-independentなfrontend/runtime modeとして実装し、HTTPとCLIから同じmodel identity・tokenizer・generation
state machineを利用できるようにする。既存Chat Completions profile v1のrequest、reject、response、SSE semanticsは変更しない。

## 仕様・参照pin

- OpenAI互換面の正本はOpenAI OpenAPI `2.3.0`、commit
  `117ce5680e4269f6656a4fd70d28f9755630d938`の`POST /v1/completions`と`POST /v1/embeddings`に固定する。
  Phase 42は下記の明示subsetだけをprofile v1として対応し、未知fieldと既知未対応fieldを区別して4xxで拒否する。
- 2026-08-22にOpenAI Developer Docsのcurrent OpenAPI `2.3.0`で両operationが引き続き存在することを確認した。
  current driftは固定profileを暗黙更新しない。
- llama.cpp比較参照はtag `b10453`、commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`へ固定する。
  `/v1/completions`、`/v1/embeddings`、`/v1/rerank`、`/tokenize`、`/detokenize`、`/apply-template`、
  input-token-count、`/infill`の技術事実だけを参照し、alias、任意template、classifier意味論を自動採用しない。
- llama.cpp sourceの直接reuseは予定しない。変更した場合だけprovenance正本へfile単位で記録する。

## 公開profile v1

### OpenAI Completions

- `POST /v1/completions`を公開する。requestは`model`、`prompt`、`max_tokens`、`temperature`、`top_p`、
  `stop`、`presence_penalty`、`frequency_penalty`、`seed`、`stream`、`n`、`logit_bias`、`logprobs`を対応する。
- `prompt`はstring、string array、token ID array、token-ID-array arrayを区別して受理する。arrayは非空、batch item数、
  text bytes、token数、`n`との積を有界にし、prompt-major、choice-minorの決定的indexを返す。
- `max_tokens`は1〜4096、`n`は1〜8、stopは最大4、`logprobs`は0〜5とする。Phase 40のsampler、stop、
  choice-local seed/logprob stateを再利用する。
- `best_of`、`echo`、`suffix`、`stream_options`、`user`とその他の未対応fieldは黙って無視しない。
  FIMはOpenAI legacy `suffix`へ暗黙接続せず、下記sLLM endpointで所有する。
- non-streamは`text_completion` envelope、streamは`text_completion` chunkとdata-only SSE、終端`[DONE]`を返す。
  usageは各prompt/choiceの実token数をchecked addし、stopで非表示になったtokenも既存generation contractどおり数える。

### OpenAI Embeddings

- `POST /v1/embeddings`を公開する。requestは`model`、`input`、`encoding_format`、`dimensions`を対応し、`user`は拒否する。
- `input`はstring、string array、token ID array、token-ID-array arrayを受理する。空text、空token列、混合array、
  model context超過、token ID範囲外をGPU処理前に拒否する。
- profile v1のmodel representationは最終RMSNorm後の全token hidden row、poolingはarithmetic mean、normalizationはL2、
  accumulatorはf64、公開dtypeはfinite f32とする。special tokenを含む実token列をpool対象にする。
- `dimensions`省略時はmodel hidden size、指定時は同じ値だけを受理し、任意truncateは行わない。
  `encoding_format`は`float`と、little-endian f32 bytesのstandard Base64である`base64`を対応する。
- input順、duplicate、dimensionを保持し、responseはOpenAIの`list`/`embedding` envelopeと
  `prompt_tokens == total_tokens`のusageを返す。full logitsや生成tokenを公開・利用しない。

### sLLM Rerank

- `POST /v1/rerank`を`sllm-rerank-v1`として公開し、OpenAI互換を名乗らない。
- requestは`model`、`query`、1〜256件の`documents`、任意`top_n`、`return_documents`だけを受理する。
  query/documentはtextだけとし、外部file、URL、pathを解決しない。
- scoreはPhase 42 embedding profile v1で得たL2-normalized vectorのcosine dot productをf64で計算してfinite f32へ丸める
  `cosine-embedding-v1`と明示する。専用classifier/rank headの出力とは称さない。
- score降順、同点は元index昇順のstable order、`top_n`省略時は全件、1〜document数だけを受理する。
  usageはqueryと全documentの実token数を合計する。

### sLLM token utility・input-token-count

- `POST /v1/tokenize`、`POST /v1/detokenize`、`POST /v1/apply-template`、`POST /v1/input-tokens`を公開する。
  対応CLI commandは`tokenize`、`detokenize`、`apply-template`、`input-tokens`とし、既存`decode`/`render`は互換aliasとして残す。
- 全surfaceは同じ`TokenizerUtilityServiceV1`を利用し、verified tokenizer/model-lock fingerprintを返す。
- tokenizeは`text`をmodel既定special-token policyでencodeし、count、token IDs、任意pieceを返す。pieceがUTF-8でなければ
  byte arrayとして返す。任意special-token policy overrideはprofile v1では受けない。
- detokenizeは1〜1,048,576件のu32 token IDと`skip_special_tokens`を受理し、未知IDを拒否する。
- apply-templateはtyped messagesと既存Qwen renderer option subsetだけを受理し、prompt、template version/digest、token IDs/countを返す。
  reviewed templateを持たないmodel、任意Jinja、custom kwargsは拒否する。
- input-tokensはraw textまたはtyped messagesをrender/tokenizeするだけで、GPU allocation/generationを起動せずcountとidentityを返す。

### sLLM FIM/infill

- `POST /v1/infill`とCLI `infill`を`sllm-infill-v1`として公開する。requestは`model`、`prefix`、`suffix`、generation
  sampler/stop/stream/`n` subsetだけを受理する。
- frontendのversioned `FimTemplateV1`がverified prefix/suffix/middle token IDと順序を持ち、model registry capabilityが
  同じdigestへ結合される場合だけpromptを構築する。markerをvisible outputへ含めない。
- 現在のverified Qwen/Gemma lockにFIM contractはないためproductionではfail closedにする。synthetic capability fixtureで
  render、generation共有、SSE、stop、usageを検証し、raw completionへのfallbackは行わない。

## Work units（完了）

1. P42-A0: 本plan、endpoint fixture、JSON Schema、rejection matrix、OpenAI/llama reference identityを固定した。
2. P42-A1: frontendへtoken utility、prepared raw/token/chat/FIM input、embedding pooling/normalization、rerank score contractを追加した。
3. P42-A2: Qwen/Gemma executionへfinal-normalized hidden readback modeを追加した。hidden shape、finite値、exact HIP、cleanupを監査し、
   generation tokenやfull logitsのD2HをEmbeddingの結果として利用しない。
4. P42-A3: runtime backend capabilityとbounded scheduler jobをgeneration、embedding、rerank、CPU-only utilityで型付けし、
   model registry identityとcancel/timeout/queue semanticsを共有した。
5. P42-A4: Completions、Embeddings、Rerank、token utility、input-tokens、Infillのstrict parser、HTTP handler、non-stream/SSE responseを実装した。
6. P42-A5: CLIを同じfrontend/runtime serviceへ接続し、既存command/reportを壊さずPhase 42 modeを公開した。
7. P42-A6: host unit/integration、native/GPU focused test、profile validator、docs/main-plan/historyを完了し、1 Phase 1 commitでpushする。

## Acceptance

- strict fixture/schemaはunknown field、wrong type、nonfinite、境界の両側、body/item/token/context上限、model/capability mismatchを
  status、error type、param、codeまで固定してPASSする。
- CLI/HTTP differentialはUnicode、byte fallback、special token、empty/large、unknown token、template digest、raw/chat countを一致させる。
- Completionsはsingle/multi prompt、token prompt、`n=1/2/8`、non-stream/SSE、stop、usage、logprobs、cancel/errorをPASSし、
  Chat Completions既存fixtureとHTTP testsがbyte/semantic regressionなしでPASSする。
- Embeddingsはinput順・duplicate、mean/L2、float/base64、dimension、usageをscalar f64またはNumPy oracleへ一致させる。
  Qwen/Gemma model integrationはfinal-normalized hidden shape、finite、determinism、exact target、fallbackなし、cleanupを確認する。
- Rerankは固定vector oracle、score tolerance、降順、stable tie、`top_n`境界、finite拒否、usageをPASSする。
- Infillはsynthetic supported capabilityとproduction unsupported Qwen/Gemmaの両方を検証し、fallbackやmarker露出がない。
- workspace test/clippy/fmt、Markdown/link、profile validator、affected native build/testをPASSする。既存GPU routeを変更した場合は
  V620 `gfx1030`とR9700 `gfx1201`をfocused testし、MI300X `gfx942` realはdeferredのままexact feature compileだけを行う。

## Error・security contract

- malformed JSONは400 `invalid_json`、unknown/invalid valueは400 `invalid_value`、既知未対応/capability mismatchは400
  `unsupported_parameter`、unknown modelは404、body超過は413、auth失敗は401、queue fullは429とする。
- prompt、token IDs、embedding vector、document、credentialをlog、metric、props、slot snapshotへ含めない。
- CPU utilityを除くmodel workはbounded scheduler、timeout、cancel、single-owner runtimeを迂回しない。
- SSE header後のerrorはerror eventを一度返してcloseし、成功chunkや`[DONE]`を後続させない。

## Non-goals

- Responses、Anthropic Messages、function/tool protocol（Phase 43）、任意Jinja/reasoning/interactive UX（Phase 44）、
  adapter/model router（Phase 45）、組込みtool/MCP実行（Phase 47）、WebUI（Phase 48）。
- classifier-head reranker、multimodal embedding/rerank/infill、wire session resume、OpenAI全field、llama endpoint alias、
  multi-GPU/parallel、MI300X性能最適化、generic quantization。

## Closeout

- matching historyへ実装、検証、known limitation、source/semantic identityを記録した。
- exact GPU evidence: `ci/matrix/phase42-inference-gpu-summary-v1.json`
- history: `docs/history/2026/08/21-31/phase42-inference-modes-public-endpoints.md`
