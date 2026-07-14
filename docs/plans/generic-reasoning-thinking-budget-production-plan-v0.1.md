# Generic reasoning and thinking budget production plan v0.1

Status: core implementation and release evidence complete for the current production candidate; OpenWebUI metadata and exact custom-budget UI reconciliation remain

Date: 2026-07-13

Last handoff audit: 2026-07-14

Current implementation status: the v2 schema/manifest contract, Gateway path,
synthetic reasoning state machine, AQ4 forced-close path, release accounting,
promotion evidence, release evidence, atomic activation, and rollback path are
in place. The forced-end accounting path at the completion-length boundary was
fixed and covered for both stream and non-stream responses. The current service
is active/running with `NRestarts=0` and the bundle-bound candidate: manifest
SHA `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`, source
commit `ae8b2bb7c2735f4dc761773957bf45f470dd5a8c`. Phase 0 is complete and
gate-eligible because resident promotion evidence supplies sanitized worker-side
generated-token evidence without prompt or response text. Current-source
browser reasoning smoke is complete and gate-eligible, and the 100-chat
OpenWebUI soak passed with 100/100 chats, 500 lifecycle actions, zero restarts,
and no failed request. The identity-matched HTTP/SSE comparison collected 100
cases for both the previous v2 manifest and the current candidate; correctness,
accounting, reset, and stop outcomes were 100/100 for both. Current p95 deltas
were within the plan thresholds in every mode. The retained complete bundle is
`release-bundle-ae8b2bb-20260714-final.json`; the intermediate post-comparison
restoration bundle is preserved in the external evidence archive.

## 前回の要点

- Qwen3.5 9B AQ4のdecode性能は実用域に達したが、配信プロファイルでは
  `enable_thinking: false`が固定され、reasoningは有効になっていない。
- OpenAI Gatewayは`reasoning_effort`を認識するが、未対応パラメータとして拒否する。
- worker protocolにはthinking budgetを渡す契約がなく、生成結果はすべて`content`として
  返される。
- OpenWebUI 0.9.4はOpenAI互換SSEの`delta.reasoning_content`をthinking表示できるため、
  表示用のOpenWebUI独自パッチは不要である。
- Qwen3.5のchat template自体はthinkingとassistant履歴の`reasoning_content`に対応する。
  必要なのは、template、Gateway、worker、推論sessionを一貫した契約で接続することである。

## 今回の変更点

- reasoning表示だけでなく、厳密なthinking budget、モデル非依存のreasoning dialect、
  OpenWebUI統合、障害試験、性能・品質ベンチマーク、昇格証跡までを一つの本番計画にする。
- Qwen固有の条件分岐を推論loopへ追加せず、モデル固有情報をserved-model manifestの
  宣言データへ限定する。
- 既存の凍結済み契約を直接変更せず、OpenAI subset、served-model manifest、worker
  protocolを新しいversionとして追加する。
- reasoningを指定しない既存clientと既存modelの挙動、生成token列、性能を回帰gateで守る。
- 他のAIがこの文書だけを起点に再開できるよう、repository、runtime、配備、性能証跡、
  security、既知の不一致、開始時checklistをSection 13へ追加する。

## 次の行動

現行candidateのbundle-bound activation、Phase 0、browser reasoning smoke、100-chat soak、
identity-matched p95比較は完了している。llama.cppとの性能比較はrelease gateから除外する。
今後はOpenWebUI managed modelのmanifest hashを現行active manifestへ再同期し、
`thinking_budget_tokens`を使う厳密budget UIを検証する。通常の運用変更時は同じvalidatorとbundle
activation手順を再利用し、rollback時にはactive manifestとサービス状態を再確認する。

## 1. 目的

OpenWebUIとOpenAI互換APIからreasoningを明示的に有効化し、thinking token数を制限できる
状態を作る。reasoningと通常回答を別々にstreamし、budget到達時にも通常回答へ確実に移行する。

同時に、今後追加するmodelで次を再実装しない構造にする。

- reasoning開始・終了区切りの検出
- thinking budgetの計数と強制終了
- reasoningと回答のstream分離
- UTF-8安全な増分decode
- usage計数
- cancel、worker failure、session reset
- OpenWebUI表示とrelease evidence

## 2. 現状と原因

### 2.1 配信設定

`deploy/served-models/qwen35-9b-aq4.profile.json`はchat template optionとして
`enable_thinking: false`を指定する。Gatewayのserved-model loaderも許可するtemplate optionを
固定しているため、request単位のthinking切り替えには新しい契約が必要である。

### 2.2 APIとGateway

`services/openai-gateway/src/ullm_openai_gateway/schemas.py`は`reasoning_effort`を既知だが
未対応のfieldとして扱う。stream/non-streamの双方が生成tokenを`content`だけへdecodeするため、
thinkingを有効にするだけでは`<think>`相当の区切りと内部思考が通常回答へ漏れる。

### 2.3 workerと推論session

現行worker requestはprompt token、最大生成数、sampling、EOS等だけを持つ。reasoning phase、
budget、強制終了token列を表現できない。AQ4 sessionにはtokenをpublish前にprepareし、publish後に
commitする境界があるため、この境界へ汎用のforced-token queueを追加すれば厳密な終了token列を
挿入できる。

### 2.4 OpenWebUI

配備中のOpenWebUI 0.9.4は`delta.reasoning_content`をreasoning UI itemとして扱い、最初の
通常`content`でreasoning表示を完了する。uLLM向けに表示patchを追加しない。

OpenWebUIはuLLM OpenAI providerの過去のreasoningを次turnへ自動再投入しない。この挙動を既定とし、
内部思考によるcontext肥大化を避ける。明示的な製品要件が出た場合だけ、モデル単位で再投入を
有効化する。

## 3. 公開API契約

### 3.1 request field

`/v1/chat/completions`で次を受け付ける。

- `reasoning_effort`
  - `none`はreasoningを無効にする。
  - `low`、`medium`、`high`はreasoningを有効にし、served-model manifestで定義したbudgetへ
    対応付ける。
- `thinking_budget_tokens`
  - `0`はthinkingを開始して直ちに終了区切りへ移る。
  - 正の整数はreasoning本文の最大token数とする。
  - `-1`はreasoningを有効にするが、強制的なbudgetを設けない。
- `reasoning_effort`と`thinking_budget_tokens`の同時指定は400とする。
- 両方を省略した場合はserved modelの既定値を使う。最初のQwen3.5 AQ4 profileは、後方互換の
  ためreasoning無効を既定とする。

budgetはreasoning本文のtoken数として数える。budget到達時に挿入する終了token列はbudgetへ
含めないが、実際に生成・commitしたtokenとして`completion_tokens`へ含める。

### 3.2 contextと生成上限

Gatewayは生成開始前に次を検証する。

- prompt、最大生成数、model context上限の整合
- budgetが最大生成数を超えないこと
- 強制終了token列全体と、最低1個の通常回答tokenを予約できること
- effort-to-budget対応がmodelの最大budgetを超えないこと

予約できない要求はsilent clampせず、理由を含む400で拒否する。

### 3.3 response field

非stream応答は次を返す。

- `message.reasoning_content`: reasoning本文
- `message.content`: 通常回答
- `usage.completion_tokens`: reasoning、区切り、通常回答を含む実生成token総数
- `usage.completion_tokens_details.reasoning_tokens`: reasoning本文のtoken数

stream応答は次の順序を守る。

1. assistant role delta
2. 0個以上の`delta.reasoning_content`
3. 0個以上の`delta.content`
4. finish chunk
5. request時だけusage chunk

budget到達はthinking phaseの終了であり、requestの`finish_reason`にはしない。`<think>`、
`</think>`、その他manifestで宣言した区切りはreasoning本文と通常回答の双方から除外する。

### 3.4 assistant履歴

Gateway schemaはassistant messageの`reasoning_content`を受理する。ただし既定の
`history_reasoning_policy`は`omit`とし、promptへ再投入しない。将来のprofileでは明示的に
`preserve`を指定できるようにする。

## 4. Versioning

既存の凍結済みschemaを暗黙に拡張しない。次を新設する。

- `docs/specs/openai-chat-subset-v0.2.md`
- `docs/specs/served-model-manifest-v0.2.md`またはschema ID `ullm.served_model.v2`
- `docs/specs/sq8-worker-protocol-v0.2.md`または共通schema ID `ullm.worker.v2`
- `docs/specs/aq4-reasoning-openwebui-release-v0.1.md`

v1 parserと既存fixtureは残す。v1 modelとv1 workerはreasoning未対応のまま正常に動作し、未知field
拒否を維持する。

## 5. モデル非依存設計

### 5.1 ReasoningDialect

served-model manifestへ、model固有のreasoning表現を宣言する。

- chat templateでthinkingを有効にするoption
- reasoning開始token列
- reasoning終了token列
- budget到達時に挿入する終了token列
- 初期phase
- reasoning中のEOS方針
- effort-to-budget対応
- 最大budget
- 通常回答用に予約する最小token数
- history reasoningの`omit`または`preserve`

区切りは文字列や単一tokenを仮定せず、token列として表現する。profile load時にtokenizerとの
整合、空配列、重複、prefix衝突、context予約量を検証する。

### 5.2 ReasoningRequest

Gatewayでclient表現を次の正規形へ変換する。

- enabled
- hard budgetまたはunbounded
- history policy
- model dialect identity
- maximum completion and reserved-token contract

workerやsessionは`reasoning_effort`というAPI文字列を解釈しない。

### 5.3 ReasoningState

推論sessionは次の汎用状態を持つ。

- disabled
- reasoning
- forcing-end-sequence
- answer
- finishedまたはcancelled

reasoning中に自然な終了token列が完成したらanswerへ移る。budgetへ到達したら通常samplingを止め、
forced-token queueから終了token列を順番にprepare、publish、commitしてからanswerへ移る。

### 5.4 出力分離

分離はtoken ID段階で行い、stream chunkへdecodeした文字列の単純な`<think>`検索へ依存しない。
開始・終了区切りが複数token、chunk境界、UTF-8文字境界をまたいでも正しく処理する。

reasoning用とanswer用に独立した増分decoderを持ち、各field内のstream連結結果がnon-stream結果と
一致することを契約にする。

### 5.5 architecture依存を持たせない境界

generic inference loopはQwen、AQ4、SQ8という名前を知らない。model固有情報はmanifestと
tokenizer adapterへ限定する。量子化方式やGPU kernelはreasoning phase制御へ影響させない。

## 6. 実装フェーズ

### Phase 0: 基準証跡と仕様固定、1日

- 現行AQ4でreasoning未指定時のprompt token列、生成token列、HTTPのhash-only metadata、SSE列を保存する。
- 短文、長文、1,024/2,048/3,072-token promptのprefill/decode基準を保存する。
- budgetの計数、予約token、EOS、history方針、usageの定義を仕様へ固定する。
- AQ4 sessionの生成token計数と`publish_prepared`周辺を監査する。

Gate:

- 後続の回帰試験で比較できるmachine-readable baselineがある。
- APIとbudgetの曖昧な項目が残っていない。

### Phase 1: schemaとmanifest、2日

- OpenAI subset v0.2を追加する。
- served-model v2とworker v2を追加する。
- generator、validator、fixtureをv1/v2対応にする。
- Qwen3.5 AQ4用ReasoningDialectをprofileへ追加する。
- request field、値域、排他関係、context予約のschema試験を追加する。

Gate:

- Qwen以外のsynthetic multi-token dialect fixtureを通過する。
- v1 fixtureと既存validator結果が不変である。

### Phase 2: Gateway、3〜4日

- requestをReasoningRequestへ正規化する。
- chat template optionをrequest単位で安全に切り替える。
- assistant履歴のreasoning fieldをschemaへ追加する。
- raw tokenからreasoningとanswerを分離する。
- stream/non-stream、usage、UTF-8増分decodeを実装する。
- fake workerを使ったOpenWebUI向けSSE試験を追加する。

Gate:

- streamをfield別に連結した結果がnon-streamと一致する。
- 区切り漏れが0件である。
- reasoning未指定時のbaselineが不変である。

### Phase 3: workerと推論engine、4〜6日

- worker v2 requestへReasoningRequestの実行表現を追加する。
- InferenceRequestへ汎用reasoning controlを追加する。
- AQ4 sessionへReasoningStateとforced-token queueを実装する。
- 自然終了、budget到達、EOS、length、stop、cancelを一つの状態機械で処理する。
- forced tokenとreasoning tokenをusage/eventへ記録する。
- session release、worker error、次requestへの状態漏れを試験する。

Gate:

- 指定budgetの超過が0 tokenである。
- forced終了後に最低1 tokenの通常回答を生成できる。
- cancel/error後の次requestが正常である。

### Phase 4: 汎用化と異常系、2〜3日

- 複数token区切り、prefix不一致、区切りの重複を試験する。
- reasoning中EOSのmodel policyを実装する。
- prefill、reasoning、forced close、answer各phaseのcancelを試験する。
- slow client、SSE切断、worker kill、Gateway restartを試験する。
- SQ8と将来backendが共通requestを読める境界を固定する。

Gate:

- Qwen tokenizer IDをhard-codeしたengine branchがない。
- synthetic dialectで同じ状態機械を検証できる。

### Phase 5: OpenWebUI E2E、3〜4日

- uLLM model設定から`reasoning_effort`を送信する経路を確認する。
- 正確なbudgetはOpenWebUI custom parameterの`thinking_budget_tokens`で指定する。
- reasoning panel、完了状態、通常回答分離をbrowserで検証する。
- Stop、refresh、複数turn、provider切り替えを検証する。
- uLLMの既存providerとmodel rowを維持する。llama.cpp providerは任意の互換性確認とし、
  reasoning release gateには含めない。

Gate:

- OpenWebUI独自patchなしでreasoningが表示される。
- 過去のhidden reasoningが意図せず次turnへ入らない。
- Stop後にworkerとGatewayがreadyへ戻る。

### Phase 6: benchmarkとrelease evidence、3〜5日

- reasoning無効の回帰benchmarkを実施する。
- budget別の性能・品質benchmarkを実施する。
- OpenWebUI soak、stop、failure、restart gateを実施する。
- llama.cppとの性能比較は実施しない。GPU排他性のpreflightだけを維持する。
- raw evidence、validator、promotion evidence、receiptを生成する。

Gate:

- Section 9の昇格基準をすべて満たす。
- raw evidenceからproducerの自己申告に依存せず合否を再計算できる。

### Phase 7: 文書、配信、rollback、1日

- API、manifest、worker、運用手順、OpenWebUI設定を文書化する。
- `journal/2026/07/`へ実装と検証の作業単位を記録する。
- 前のactive manifestを保持してproduction profileをactivateする。
- rollback後にreasoning無効の従来経路へ戻ることを確認する。

## 7. 試験行列

### 7.1 schemaとvalidation

- field省略
- `reasoning_effort`の全許容値と未知値
- budget `-1`、`0`、`1`、`2`、`8`、`32`、`128`、`256`
- `-2`以下、整数以外、model上限超過
- effortとbudgetの同時指定
- prompt/context/max completion境界
- v1/v2のstrict parseと未知field拒否

### 7.2 tokenizerとsegmenter

- thinking有効・無効のprompt suffix
- assistant履歴のomit/preserve
- 自然終了
- budgetの直前、ちょうど、直後で終了
- 開始・終了区切りが複数token
- 区切り候補のprefix不一致
- 区切りがdecode chunkをまたぐ場合
- 日本語、絵文字、結合文字のUTF-8境界
- reasoningなしで直接answerを返すmodel behavior

### 7.3 sessionとworker

- budget 0の強制終了
- forced-token queueの全token commit
- reasoning中のEOS
- 最大生成数到達
- prefill/reasoning/forced-close/answerでのcancel
- publish前後のerror
- worker killとrestart
- 正常release後と異常release後の次request
- active request 1、waiting request 0の既存制約

### 7.4 APIとOpenWebUI

- stream/non-streamの内容とusage一致
- `<think>`等の表示漏れ0件
- reasoning panelの開始と完了
- Stop操作後のready復帰
- browser refresh
- 複数turn
- provider切り替えは任意のOpenWebUI互換性確認とし、release gateには含めない
- slow clientと切断
- busy responseと既存OpenWebUI mapping

### 7.5 soak

- 通常HTTP chat 100回
- browser chat 20回
- worker restart後20回
- request中worker kill後1回以上の正常chat
- long-contextを含むuLLM実行
- RSS、VRAM、GPU温度、電力、zombie processの監視

## 8. Benchmark計画

### 8.1 比較対象

1. 現行uLLM Qwen3.5 9B AQ4
2. 実装後uLLM AQ4、reasoning未指定
3. 実装後uLLM AQ4、budget 32
4. 実装後uLLM AQ4、budget 128
5. 実装後uLLM AQ4、budget 256
6. 実装後uLLM AQ4、unboundedまたはhigh effort
llama.cpp Qwen3.5 9B UD-Q4との性能比較は2026-07-14のuser判断で対象外とした。既存の配備・
provider切り替え証跡は履歴として保持するが、品質、性能、production昇格の合否には使わない。

### 8.2 prompt分類

- 64 token以下の短い通常会話
- 日本語と英語の約512-token会話
- 1,024/2,048/3,072-token履歴
- 算術、論理、コード追跡
- 長くthinkingしやすいbudget負荷問題
- reasoning不要の事実問題
- 日本語、英語、絵文字の混在stream

正答を機械判定できる問題を中心にし、自由記述は固定rubricとblind評価を使う。モデルjudgeだけを
唯一の品質判定にしない。

### 8.3 記録指標

- prefill tok/s
- 最初のreasoning tokenまでの時間
- 最初の通常回答tokenまでの時間
- reasoning decode tok/s
- answer decode tok/s
- 総decode tok/s
- end-to-end latency
- reasoning token数とanswer token数
- budget overshoot
- forced close回数
- empty answer率
- HTTP status、SSE chunk数、finish reason
- p50、p95、p99
- RSS、VRAM、GPU使用率、電力、温度
- worker failureからreadyまでの時間

reasoning modelでは、最初のreasoning tokenと最初の可視回答tokenを分離して測定する。従来の
TTFTだけではOpenWebUI上の待ち時間を評価しない。

## 9. Production昇格基準

### 9.1 後方互換性

- reasoning未指定時のAPI response shapeとSSE順序が既存契約と一致する。
- greedyまたは固定seed fixtureで、reasoning無効時の生成token列がbaselineと一致する。
- v1 manifestとworker requestが引き続き動作する。

### 9.2 正確性

- budget overshootが全試験で0 tokenである。
- completion usageがraw generated token記録と一致する。
- stream/non-streamをfield別に連結した内容が一致する。
- reasoning/answerへ開始・終了区切りが漏れない。
- smokeとrelease gateでempty answerが0件である。

### 9.3 性能

- reasoning無効時のprefill/decode中央値の低下が3%以内である。
- reasoning無効時のp95 latency悪化が5%以内である。
- reasoning分離処理によるdecode速度低下が同条件で5%以内である。
- 無効時のRSS増加を32 MiB以内とし、model/KV以外の恒常的なVRAM増加を発生させない。

### 9.4 品質

- reasoning有効のcurated suiteがreasoning無効より3 percentage pointを超えて悪化しない。
- budget 128/256の正答率がhigh/unboundedの95%以上を目標とする。
- 強制終了後に回答不能、区切りだけ、空文字になる問題が0件である。

品質閾値はPhase 0〜6の校正runで分布を確認し、正式release runより前に固定する。正式runの結果を
見た後で合格線を変更しない。

### 9.5 安定性とOpenWebUI

- 通常100 request、restart後20 requestが成功する。
- OOM、worker zombie、session状態漏れが0件である。
- worker failureからの復旧が既存の45〜60秒程度の運用範囲を悪化させない。
- reasoning表示、Stop、refresh、複数turn、provider切り替えがすべて成功する。

## 10. Evidenceとrollback

release evidenceに次を含める。

- Git commitとworktree status
- uLLM binary hash
- model、tokenizer、manifest hash
- OpenWebUI image digest
- systemd unitとenvironment fileのhash
- llama.cpp比較時のbinary、model、起動条件
- request fixture IDとprompt hash
- 全benchmark raw data
- resource sampleとjournal lifecycle
- validatorによる合否再計算結果

evidenceは`.incomplete`へstreamingで書き、検証成功後に原子的に正式名へ変更する。producerが
書いた`passed`値を信用せず、validatorがraw evidenceからgateを再計算する。user prompt本文やAPI keyは
証跡へ保存しない。

generic reasoningのrelease evidenceは`validate-generic-reasoning-release.py`で構造、token accounting、
usage cross-check、budget overshoot、必須benchmark mode、source identityを再計算する。構造検証に
成功しても、identity不一致、mode不足、または`status=incomplete`の場合はproduction gateを不合格とする。
complete evidenceではhash-onlyの`request_released` lifecycle eventをcase IDへ結合し、prompt/completion
token数、reasoning/forced-end accounting、reset完了、release timingを再検証する。lifecycle eventを欠く
artifactは構造検証を通してもproduction gateを満たさない。

activation前のactive manifestとsystemd設定を保持する。rollbackは旧manifest/profileへ戻して
serviceを再起動し、reasoning未指定のbaseline smokeを再実行する。OpenWebUIへ恒久patchを入れないため、
UI imageの特別なrollbackは不要である。

v2 candidateの最終activationは、`tools/activate-served-model.py`へcomplete release bundleと
current systemd/environment fileを渡し、candidate identityとrollback hashの一致を確認してから
atomic replaceする。release evidenceを取得するための一時v1→v2切替は、同toolの明示的な
`--bootstrap-v2`だけで許可する。この経路はcomplete bundleを免除する代わりに、v1 active manifestの
外部backup、systemd/environment hash、全対象serviceのinactive確認を必須とし、実production gateの
合格を宣言しない。evidence収集後はbackupしたv1へ戻してからcomplete bundleを組み立て、通常のv2
activationを行う。v1 active pathは従来どおり維持する。

## 11. 見積もりとmilestone

- OpenWebUIへreasoningを分離表示する試作: 2〜4人日
- AQ4で厳密なthinking budgetが動くbeta: 累計8〜12人日
- 汎用化、障害試験、benchmark、release evidenceを含むproduction候補: 累計15〜25人日
- 1人で連続して進めるcalendar time: 約3〜5週間

GPUを排他的に使う長時間benchmarkとsoakは1〜2晩を見込む。実装と独立な文書・fixture・validatorは
並行化できるが、同じR9700上の性能測定は並列実行しない。

## 12. 主要risk

1. budgetで強制終了した直後の回答品質が落ちる可能性がある。
   - budget別品質試験、回答token予約、empty-answer gateで検出する。
2. 複数token区切りでは部分一致とrollbackが複雑になる。
   - 実model導入前にsynthetic dialect fixtureで固定する。
3. 既存AQ4生成token計数に二重加算の疑いがある。
   - Phase 0で独立監査し、usage gateの前提を確定する。
4. reasoningにより同じ`max_completion_tokens`内の通常回答枠が減る。
   - 生成前予約と明示的な400エラーでsilent truncationを避ける。
5. history reasoningを再投入するとcontextと機密性の負担が増える。
   - 既定を`omit`とし、必要なmodelだけ明示的に`preserve`へする。
6. R9700上の別workerはGPU排他性を壊し、結果を汚染する。
   - benchmark collectorはpositive-VRAM processをfail closedで検出する。llama.cpp比較自体は行わない。

## 13. AI引き継ぎcontext

このsectionは2026-07-13時点のhandoff snapshotである。仕様方針はSection 1〜12を正とするが、
service状態、binary、manifest、Git HEAD、GPU使用量は変動する。次の担当AIは作業開始時に
Section 13.10のread-only確認を再実行し、snapshotとの差分をjournalへ記録する。

### 13.1 作業範囲と最初に読む文書

- workspace root: `/home/homelab1/coding-local/ultimateLLM`
- Git repository root: `/home/homelab1/coding-local/ultimateLLM/uLLM-project`
- repository scope: `uLLM-project/`以下
- agent instruction: workspace rootの`AGENTS.md`
- 継続context: workspace rootの`memo-for-AGENT.md`
- 本計画: `docs/plans/generic-reasoning-thinking-budget-production-plan-v0.1.md`
- 関連する推論最適化計画:
  `docs/plans/generic-production-inference-optimization-plan-v0.1.md`
- 現行API仕様: `docs/specs/openai-chat-subset-v0.1.md`
- 現行served-model仕様: `docs/specs/served-model-manifest-v0.1.md`
- 現行worker仕様: `docs/specs/sq8-worker-protocol-v0.1.md`
- 現行OpenWebUI release仕様: `docs/specs/sq8-openwebui-release-v0.1.md`

作業記録は`journal/YYYY/MM/DD/`へ保存する。長期間の作業ではworkspace rootの
`進捗確認_for_user.md`を数行の最新状態だけに保つ。既存変更はuserまたは別作業の所有物として扱い、
無関係な差分を戻さない。

### 13.2 Git snapshot

2026-07-13の計画追記前snapshot:

- branch: `main`
- plan保存commit: `16e9e0e docs: add production reasoning budget plan`
- `main`は`origin/main`より98 commit先行していた。
- 未追跡は`.rocprofv3/`だけだった。

`.rocprofv3/`は既存のprofiling証跡であり、本計画の作業では変更、削除、add、commitしない。
source HEADと稼働binaryは一致すると仮定しない。2026-07-13時点でrepository HEADは`16e9e0e`だが、
active manifestが記録するpromotion source commitは`4be10d0`だった。

reasoning実装時は、別の汎用prefill/decode最適化作業が同じAQ4 session、worker profile、backend
registry周辺を変更している可能性がある。古い計画上のline numberや関数配置を信用せず、現HEADへ
rebaseした境界を再確認する。特に`qwen35_aq4_session.rs`とserved-model profileを一括して古い状態へ
戻してはならない。

### 13.3 hardwareとsoftware

reasoningのproduction対象はWRX80 hostである。

- CPU: AMD 3995WX、64 core
- memory: 16 GiB x 8、8 channel
- GPU: Radeon AI PRO R9700 x1、Radeon PRO V620 x2
- reasoning/AQ4 production GPU: R9700、`gfx1201`
- 監査時software: ROCm 7.2.1、systemd 255、OpenWebUI 0.9.4系
- host OpenWebUI URL: `http://192.168.0.66:3000`

GPUのordinalはhost表示、KFD、`HIP_VISIBLE_DEVICES`適用後のprocess表示で一致しない場合がある。
証跡では単なるdevice番号だけを使わず、`gfx1201`、PCI identity、visible-device設定、process内ordinalを
併記する。

OOMを避ける。長いprompt suite、raw event、GPU sampleはstreamingで保存し、全runをmemory上に保持
しない。GPUを使うruntime test、real-package smoke、performance benchmarkを無制限に並列実行しない。

### 13.4 現行production baseline

2026-07-13のread-only監査では次の状態だった。

- service: `ullm-openai.service`、enabled / active、監査時`NRestarts=0`
- bind: `172.20.0.1:8000`
- OpenAI base URL: `http://172.20.0.1:8000/v1`
- active manifest: `/etc/ullm/served-models/active.json`
- public model ID: `ullm-qwen3.5-9b-aq4`
- model: Qwen3.5 9B、format `AQ4_0`
- context上限: 4096
- completion上限: 512
- 既定sampling: top-k 1のgreedy path
- worker identity: `gfx1201` / `rdna4_aq4_resident`
- worker process: resident `ullm-aq4-worker` 1個
- 同時実行: 1 request
- waiting queue: 0
- busy応答: HTTP 429 `request_busy`
- prompt + requested completionが4096を超える場合: HTTP 400
- tokenizer runtime: Qwen2Tokenizer / transformers 5.12.1
- chat template: `enable_thinking: false`
- worker protocol: `ullm.worker.v1`

active manifestはworker binary、model package、tokenizer、promotion receiptのSHA-256を固定する。
legacy environment fieldとmanifest fieldを混在させず、manifestを配信identityの正とする。manifestの
置換は`tools/activate-served-model.py`による原子的なactivation手順を使う。

worker環境では`HIP_VISIBLE_DEVICES=1`とmanifest由来のHIP kernel必須guardが有効である。Gatewayは
`/run/ullm/r9700.lock`を排他取得する。reasoning実装はこのsingle-request、resident-session、
publish-before-commit、fail-closed kernel guardを維持する。

### 13.5 llama.cppの任意互換性context

llama.cppは過去にuLLMと同時にOpenWebUIへ登録された外部providerである。2026-07-14以降、
性能比較とprovider切り替えはreasoning release gateに含めない。

- service: `llama-qwen35-udq4.service`、enabled / active、監査時`NRestarts=0`
- bind: `172.20.0.1:8001`
- OpenAI base URL: `http://172.20.0.1:8001/v1`
- model alias: `llama-qwen3.5-9b-ud-q4`
- model: Qwen3.5 9B UD-Q4_K_XL GGUF
- context: 4096
- parallel: 1
- GPU layers: 999
- `--fit off`
- text-only、`--no-mmproj`
- `HIP_VISIBLE_DEVICES=1`、process内ではR9700が`ROCm0`

3 GPUを可視にしてR9700をprocess内`ROCm1`として使う構成では`libamdhip64` GPFが発生したため、
この構成へ戻さない。uLLMとllama.cppは同じR9700を共有するが、llama.cppは
`/run/ullm/r9700.lock`を共有しない。uLLM benchmark時はllama.cppを停止し、R9700の排他性を確認する。

2 worker常駐時の過去証跡ではR9700使用量は合計約13.38 GBだった。2026-07-13監査時の概算は
uLLM worker約7.35 GB、llama.cpp約5.97 GBだった。値は変動するため、正式runで再取得する。

llama.cppの短い2-token回答でdecode 131.61 tok/sという記録があるが、生成tokenが少なすぎるため
代表性能に使わない。別の短文実測はprefill 131.94、decode 74.23 tok/sだった。量子化とruntimeが
異なるため、これらの値はreasoning release判定に使わない。

### 13.6 OpenWebUIとnetwork

2026-07-13の監査時:

- container image: `ullm/open-webui:0.9.4-ullm.1`
- health: healthy
- publish: host `0.0.0.0:3000`
- data: external persistent `open-webui` volume
- session key: read-only bind mount
- signup、title generation、tags、follow-up、telemetry: composeで無効
- uLLM provider URL: `http://172.20.0.1:8000/v1`
- llama.cpp provider URL: `http://172.20.0.1:8001/v1`

過去の配備記録ではuLLM providerはindex 1、llama.cpp providerはindex 2である。ただし現在のDBは
資格情報と会話を含むため、handoff監査ではprovider rowを直接再読していない。変更前には
`deploy/openwebui/configure.py`のread-modify-write、backup、idempotency手順を使い、既存providerと
model rowを保持する。

planning auditではOpenWebUI 0.9.4 middlewareがOpenAI SSEの`delta.reasoning_content`をreasoning
itemとして扱うことを確認した。その後、`deploy/openwebui/browser-reasoning-smoke.cjs`へhash-onlyの
reasoning panel、完了状態、refresh、複数turn、hidden-history omission、provider切り替え検証を
追加した。現行candidateでは実containerのbrowser validatorが`gate_eligible=true`、100-chat soakが
100/100、Stop/failure/restart gateも成功した。provider切り替えは既存証跡に含まれるが、今後の
release必須条件ではない。

networkはDocker bridgeから8000/8001へ到達させ、bridge外から遮断する設計である。2026-07-13の
read-only root auditで、`/etc/ullm/ullm-openai.nft`、live `inet ullm_openai` table、repositoryの
`deploy/nftables/ullm-openai.nft`が同じ8000/8001 drop ruleで一致することを確認した。なお、正式benchmark
前にはbridge interfaceとcounterを再確認し、8001の比較workerも同じ隔離条件で測定する。

OpenWebUI v0.9.4はlegacy `/api/chat/completions`でupstream 429をvisible 400へ変換する既知挙動がある。
busy contractの正否は直接Gatewayの429と`Retry-After`で判定し、UI表示だけで判定しない。

### 13.7 reasoning固有の確認済み実装context

#### Gateway

- `services/openai-gateway/src/ullm_openai_gateway/schemas.py`
  - `reasoning_effort`と`thinking_budget_tokens`を排他的に正規化し、served-modelの
    `ReasoningDialect`からbudget、history policy、予約tokenを解決する。
  - assistant messageは`reasoning_content`を受理し、既定のhistory policyは`omit`である。
- `services/openai-gateway/src/ullm_openai_gateway/app.py`
  - stream/non-streamの双方でtoken IDからreasoningとanswerを分離し、
    `reasoning_content`、`content`、usage detailsを生成する。
  - v2 workerの`reasoning_tokens`と`forced_end_tokens`をraw splitと突き合わせ、
    不一致をfail closedにする。v1 requestの既存経路は維持する。
- `services/openai-gateway/src/ullm_openai_gateway/tokenizer.py`
  - v2 requestの`enable_thinking`をtemplate adapterへ渡し、assistant履歴の
    `reasoning_content`はdialectのpolicyに従ってomitまたはpreserveする。
- `services/openai-gateway/src/ullm_openai_gateway/worker.py`
  - v2 worker requestへ汎用reasoning controlを渡し、v1 requestの形状は維持する。

#### Tokenizer

現行Qwen3.5 tokenizerのplanning auditとv2 fixtureでは次を確認した。

- chat templateは`enable_thinking`を受け付ける。
- `enable_thinking=false`では空のthinking区間をpromptへ形成する。
- thinking有効時はassistant generation prefixでreasoning開始区切りを形成する。
- assistant履歴の`reasoning_content`を扱える。
- 観測したthinking開始token IDは`248068`、終了token IDは`248069`だった。
- これらは`skip_special_tokens=True`だけでは除去されない。v2 candidate profileの
  dialectへtoken列として結合し、engineへhard-codeしていない。

token IDはengineへhard-codeしない。active tokenizerのhashとtokenization結果をPhase 0で再検証し、
manifestのReasoningDialectへtoken列として保存する。別modelでは複数token区切りを許容する。

#### Workerとengine

- `crates/ullm-engine/src/worker_protocol.rs`は量子化非依存のv1/v2 protocol入口であり、v2の
  reasoning request、strict response、nested array、duplicate/unknown fieldを検証する。
- `crates/ullm-engine/src/inference_api.rs`とworker runtimeは汎用reasoning controlと
  `ReasoningUsage`を扱う。v1はreasoningなしの既存形状を維持する。
- `crates/ullm-engine/src/worker_driver.rs`はStop、Length、Cancelledに加えてreasoningの
  release accountingをlifecycle evidenceへ渡す。
- `crates/ullm-engine/src/qwen35_aq4_session.rs`はsampled tokenとforced tokenを同じ
  prepare、publish、commit境界で処理する。`crates/ullm-engine/src/reasoning.rs`の
  `ReasoningState`がbudget到達、自然終了、EOS、cancel/resetを量子化非依存に管理する。
- `crates/ullm-engine/src/session_worker_backend.rs`はsessionとworker protocolを接続する。

AQ4 sessionの`publish_prepared`周辺にあった生成token計数の監査上の疑いは、offlineのv2 worker、Gateway
usage、lifecycle accounting回帰に加え、source commit `ae8b2bb`へ結合したresident promotion、Phase 0、
HTTP/SSE evidenceで二重加算なしを確認した。forced-end境界はstream/non-stream回帰でも固定済みである。

#### Manifestとdeploy

- Python loader: `services/openai-gateway/src/ullm_openai_gateway/served_model.py`
- Rust loader: `crates/ullm-engine/src/served_model.rs`
- generator: `tools/generate-served-model.py`
- validator: `tools/validate-served-model.py`
- AQ4 profile: `deploy/served-models/qwen35-9b-aq4.profile.json`
- OpenWebUI config: `deploy/openwebui/configure.py`
- systemd unit: `deploy/systemd/ullm-openai.service`

v1はPython、Rust、generator、validator、OpenWebUI設定の各所でexact-key schemaとして扱われ、
v2も全loader、generator、validator、fixtureへ反映済みである。
`qwen35-9b-aq4-reasoning.profile.json`から生成したmanifest SHA `feb3190d…`は、専用promotion receipt、
worker SHA、tokenizer SHA、complete release bundleへ結合され、active manifestとして稼働している。

### 13.8 現在の性能context

user観測では、OpenWebUI経由のuLLM AQ4 decodeは短文で約70〜80 tok/s、長文では約40 tok/sまで
低下した。これはcontext長増加に伴うattention costと一致する傾向だが、reasoning計画のPhase 0で
prompt長、生成長、sampling、warm/coldを固定して再取得する。

`journal/2026/07/13/qwen35-aq4-full-native-prefill-evidence.md`の限定smokeでは、full-native prefillの
最大値は116.61 tok/sだった。数千tok/sには到達していない。これは限定的なresident smokeであり、
reasoning無効時のproduction回帰baselineとして単独使用しない。

reasoning実装では次の2種類の性能変化を分離する。

1. reasoning機能を使わないrequestに追加処理が与える純粋な回帰
2. reasoning tokenを実際に生成することで増える意図したlatency

既存の1011-token AQ4 OpenWebUI smokeは
`benchmarks/results/2026-07-12/qwen35-9b-aq4-resident-openwebui-v0.1/summary.json`にある。
EOS、length、cancel、post-cancel、browserの基準候補だが、reasoning別TTFT、最初の可視回答時刻、
budget overshoot、field別stream一致を記録しない。Phase 0で新しいraw evidence schemaを作る。

Hash-only generic release evidenceのschema、validator、bundle validator、assemblerは実装済みである。
assemblerはserved-model契約、worker/tokenizer/manifest identity、Git worktree状態を検証してから
atomic publishする。現行candidateのrelease validator、browser validator、promotion evidence/receiptを
束ねた`release-bundle-ae8b2bb-20260714-final.json`は`status=complete`で、active promotion sourceと一致する。
旧v2と現行candidate各100ケースのidentity-matched p95比較も全modeで閾値内だった。

### 13.9 再利用するtestとtool

Gateway test:

- `services/openai-gateway/tests/test_schemas.py`
- `services/openai-gateway/tests/test_app.py`
- `services/openai-gateway/tests/test_tokenizer.py`
- `services/openai-gateway/tests/test_served_model.py`
- `services/openai-gateway/tests/test_worker.py`

Rust/manifest test:

- `crates/ullm-engine/src/qwen35_aq4_session.rs`内unit test
- `crates/ullm-engine/tests/worker_profile_snapshot.rs`
- `tests/test_generate_served_model.py`
- `tests/test_validate_served_model.py`
- `tests/test_openwebui_configure.py`
- `docs/specs/generic-reasoning-release-evidence-v0.1.md`

OpenWebUI/release tool:

- `tools/run-openwebui-soak-gate.py`
- `tools/run-openwebui-stop-gate.py`
- `tools/run-openwebui-failure-gate.py`
- `tools/run-sq8-http-latency-gate.py`
- `tools/run-sq8-api-contract-gate.py`
- `tools/collect-sq8-openwebui-release.py`
- `tools/activate-served-model.py`
- `tools/run-aq4-resident-promotion-evidence.py`
- `tools/run-generic-reasoning-phase0-http-baseline.py`
- `tools/run-generic-reasoning-release-campaign.py`
- `tools/validate-generic-reasoning-phase0-http-baseline.py`
- `tools/validate-generic-reasoning-release.py`
- `tools/prepare-generic-reasoning-release-evidence.py`
- `tools/prepare-generic-reasoning-release-bundle.py`
- `tools/validate-generic-reasoning-release-bundle.py`
- `tools/validate-openwebui-reasoning-browser-smoke.py`
- `tools/run-openwebui-reasoning-browser-smoke.py`
- `tools/write-aq4-resident-promotion-receipt.py`
- `deploy/openwebui/browser-reasoning-smoke.cjs`

`deploy/openwebui/configure.py`もserved-model v1/v2をstrictに読み分ける。v2ではreasoning
dialectの宣言を検証するが、OpenWebUIのDBへreasoning履歴や会話本文を保存しない。

SQ8用toolをAQ4 reasoningへ流用する場合は、schema名、model identity、worker identity、prompt suite、
gateをAQ4 reasoning用に明示的に分ける。既存SQ8 evidenceを名前だけ変えて再利用しない。

### 13.10 次のAIが最初に行う確認

実装を始める前に、次をこの順で行う。

1. workspace rootの`AGENTS.md`と`memo-for-AGENT.md`を最後まで読む。
2. repository root、branch、HEAD、`origin/main`との差、worktree statusを確認する。
3. `.rocprofv3/`と既存未追跡・変更済みfileの所有権を保全する。
4. `ullm-openai.service`、`llama-qwen35-udq4.service`、OpenWebUI containerの状態をread-onlyで確認する。
5. `/etc/ullm/served-models/active.json`をvalidator経由で読み、public model、context、completion、
   protocol、binary/package/tokenizer/receipt hash、promotion source commitを記録する。
6. 稼働binary hashとsource HEADが一致するかを確認し、不一致ならbaselineのidentityを稼働側で固定する。
7. `HIP_VISIBLE_DEVICES`、`gfx1201`、worker process、GPU使用量、必須kernel guardを確認する。
8. root権限でrepository、`/etc`、live nftablesの8000/8001 ruleを照合する。
9. OpenWebUI DBを変更する前にbackupを作る。DB内容を標準出力やjournalへdumpしない。
10. reasoning未指定のAPI/SSE/token/performance baselineを、本文を保存しないmachine-readable evidenceとして取得する。
11. v2 manifestがactiveになった後、`tools/run-generic-reasoning-release-campaign.py`でdisabled、budget
    32/128/256、unboundedをstream/non-streamの両方で実行し、HTTP/SSE、lifecycle、resourceを一括取得する。
    収集器はv2 manifest検証後にgfx1201/R9700の排他性を確認し、llama.cppが常駐していれば停止する。
12. v0.2仕様とfixture、Gateway/worker/AQ4のoffline契約は固定済みである。production serviceを
    変更する前に、candidate identityを実機evidenceへ結合し、unit testと全production gateを通す。
13. 実装中は小さい意味単位でcommitし、各commitの検証結果をjournalへ記録する。

現行サービスは稼働中である。userが明示した停止は正常操作として扱い、停止履歴だけを障害と判定しない。
baseline取得以外の推論、service restart、manifest activation、OpenWebUI DB変更は、それぞれのPhaseで
必要になってから行う。

### 13.11 secretと証跡の禁止事項

次を計画、journal、Git、benchmark raw、tool outputへ保存しない。

- API key本体
- Authorization header
- OpenWebUI session secret
- OpenWebUI DBの`openai.api_keys`
- DB backupの内容
- userの会話本文
- 機密promptとresponse
- host password

必要なidentityはsecretそのものではなく、secretを含まない設定field、file path、SHA-256、model ID、
binary identityで記録する。prompt suiteは公開fixture IDとhashを使い、個人会話をbenchmark corpusへ
取り込まない。Gateway lifecycle logはprompt、response、credentialを出力しない既存方針を維持する。

### 13.12 既知の文書不一致

- `deploy/README.md`とGateway README冒頭にはSQ8/Qwen3-14Bを既定とする古い説明が残る。
  現行productionはAQ4_0/Qwen3.5 9Bのactive manifestである。
- repositoryのfirewall定義は8000/8001を含むが、過去journalには8000だけの記録がある。
- repository HEADとactive workerのpromotion source commitは一致しない。
- current OpenWebUI provider indexは機密DBを再読していないため、過去記録からの推定である。
- native prefill 116.61 tok/sは限定smokeであり、product-path full benchmarkではない。

この不一致を推測で解消しない。着手時のread-only監査とPhase 0 evidenceで現在値を確定し、文書を
更新してからreasoning実装へ進む。
