# Generic reasoning and thinking budget production plan v0.1

Status: proposed; implementation not started

Date: 2026-07-13

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

## 次の行動

実装指示を受けたら、現行AQ4のAPI・token列・性能基準を保存し、Phase 0の仕様固定から
開始する。Gatewayだけの表示試作やQwen専用の強制終了処理を先行してproductionへ入れない。

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

- 現行AQ4でreasoning未指定時のprompt token列、生成token列、HTTP body、SSE列を保存する。
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
- uLLMとllama.cppを同時登録した現在の構成を維持する。

Gate:

- OpenWebUI独自patchなしでreasoningが表示される。
- 過去のhidden reasoningが意図せず次turnへ入らない。
- Stop後にworkerとGatewayがreadyへ戻る。

### Phase 6: benchmarkとrelease evidence、3〜5日

- reasoning無効の回帰benchmarkを実施する。
- budget別の性能・品質benchmarkを実施する。
- OpenWebUI soak、stop、failure、restart gateを実施する。
- llama.cpp UD-Q4を外部参考として交互に測定する。
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
- uLLM/llama.cpp provider切り替え
- slow clientと切断
- busy responseと既存OpenWebUI mapping

### 7.5 soak

- 通常HTTP chat 100回
- browser chat 20回
- worker restart後20回
- request中worker kill後1回以上の正常chat
- long-contextを含む交互実行
- RSS、VRAM、GPU温度、電力、zombie processの監視

## 8. Benchmark計画

### 8.1 比較対象

1. 現行uLLM Qwen3.5 9B AQ4
2. 実装後uLLM AQ4、reasoning未指定
3. 実装後uLLM AQ4、budget 32
4. 実装後uLLM AQ4、budget 128
5. 実装後uLLM AQ4、budget 256
6. 実装後uLLM AQ4、unboundedまたはhigh effort
7. llama.cpp Qwen3.5 9B UD-Q4

llama.cppは量子化方式とruntimeが異なるため、uLLMの絶対的な品質合格基準にはしない。API挙動、
OpenWebUI表示、性能傾向を確認する外部参考値とする。2つのworkerは同じR9700を使うため、生成は
同時に走らせず交互に測定する。

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

activation前のactive manifestとsystemd設定を保持する。rollbackは旧manifest/profileへ戻して
serviceを再起動し、reasoning未指定のbaseline smokeを再実行する。OpenWebUIへ恒久patchを入れないため、
UI imageの特別なrollbackは不要である。

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
6. uLLMとllama.cppの同時benchmarkはGPU競合で結果を汚染する。
   - workerは同時常駐可能だが、測定requestは交互に実行する。
