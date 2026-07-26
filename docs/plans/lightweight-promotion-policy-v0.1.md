# 軽量昇格方針 v0.1

> Status: active policy, 2026-07-26 JST. This is a user-directed policy change for this
> development-only machine. A final activation does not require human approval. The
> machine-executable <code>--yes</code> flag is only an accidental-invocation guard, not an
> approval channel.

## 目的

通常の候補は、実際の推論で生成した文章の品質が明らかに崩れていなければ、速く昇格する。
後からより厳密な数値検証や修正版を出せることを前提にする。一方で、壊れた状態を長く
残さないため、昇格前の正確な active manifest と、実証済みの 1 コマンド・ロールバックは
必須とする。

この文書は、過去の専用 campaign、authorization、sealed-plan、literal confirmation
経路を新規候補の必須条件から外す。過去の昇格記録や、その再現用の専用ツールは履歴として
残すが、将来の通常昇格を拘束しない。

## 通常昇格の必須条件

### 1. 動作可能性

- 候補 manifest を gateway の validator で検証し、参照する worker、tokenizer、product、
  promotion receipt が存在し整合することを確認する。
- worker が service 経由で起動し、<code>/readyz</code> とモデル一覧が候補 manifest の
  model ID を返すことを確認する。
- 固定 prompt suite の実行中に worker/gateway が失敗しないこと。現行 worker protocol の
  生成経路は non-finite logits を正常な token として完了できないため、全 suite request の
  正常完了を logits NaN/Inf 非発生の実行上の確認として記録する。将来の worker がこの
  契約を持たない場合は、その点を <strong>未確認</strong> として記録し、別途その worker
  の有限 logits probe を追加する。
- service を再起動後、実際の completion 応答を受け取ること。待機は固定 sleep ではなく、
  単調時計の期限内で bounded exponential backoff により再試行する。

### 2. 実際の文章生成と比較

固定 suite は
<a href="lightweight-promotion-prompt-suite-v0.1.json">lightweight-promotion-prompt-suite-v0.1.json</a>
である。日本語、英語、コード生成、長文要約、多ターン会話を含む 10 件を使う。

1. active runtime から suite の全応答を取得する。
2. candidate を原子的に active に切り替え、service を再起動する。
3. candidate runtime から同じ suite の全応答を取得する。
4. prompt、現行出力、候補出力、各自動判定を同じ証跡ディレクトリに保存する。

出力の完全一致は要求しない。次の明白な崩壊は自動検出し、候補側で起きた場合は同じ
transaction 内でロールバックする。

- request failure、空の completion、gateway/worker の停止;
- 同一語句または同一文の反復ループ;
- replacement character、制御文字、読めない文字列などの文字化け;
- 現行出力に対して極端に短い/長い出力への偏り;
- コード要求にコードらしい構造が全くない、または要求言語の文字が完全に失われる、といった
  高信頼度の応答放棄。

言語の自然な混在、内容の正しさ、説明の良し悪し、創造性の差は機械合否にしない。
比較 Markdown に実際の文章を並べ、後から人間が読めるようにする。これは人間の
承認待ちではなく、監査可能な証跡である。

### 3. 安価な診断指標

取得が数分で済む指標は保存する。例は output exact-match 率、文字数比、生成 token 数、
worker が公開する場合の top-1 一致率である。

これらの値に昇格停止のしきい値を設けない。情報として記録し、明白な崩壊の自動検出と
人間可読な並列出力を補助するだけである。

## ロールバックは必須

ロールバックは昇格を遅くする仕組みではなく、速い昇格を成立させる仕組みである。

- atomic swap 前に active manifest の生バイト列と SHA-256 を保存する。
- rollback は active の現在バイト列が候補 manifest と厳密に一致し、保存済み rollback
  bytes と異なることを確認してから実行する。
- rollback も原子的な swap、service restart、ready/model/actual response の確認を行う。
- 応答確認に失敗した candidate は自動 rollback する。
- transaction ごとに、時刻、候補、元 manifest、保存先、結果、service 操作回数を
  append-only ledger に追記する。

通常 rollback は generic rollback tool 1 コマンドで実行でき、そのコマンドを昇格検証で
実際に通す。

## 明示的に通常昇格へ持ち込まないもの

新アーキテクチャまたは新候補の通常昇格に、次を必須にしない。

- 1 モデルあたり CPU 64 コアで 10 時間以上を要する FP32 参照 corpus;
- bitwise equality を昇格 gate にすること;
- campaign、authorization、candidate 固有の sealed intent、plan SHA 照合、literal
  confirmation token;
- browser campaign や重い bundle assembly。

これらは既存本番を最適化回帰から守る研究・監査用の手段として、必要な場合に別途実施して
よい。ただし通常昇格の前提にはしない。

## 実行経路

通常の昇格には <code>tools/promote-served-model.py</code> を使う。候補 manifest、
evidence directory、state directory を引数で与え、<code>--yes</code> を渡す。候補固有の
確認文字列や人間との対話はない。対応する
<code>tools/rollback-promoted-served-model.py</code> は、昇格 outcome を引数として
同じ rollback 記録を再利用する。

どちらの tool も active manifest と service を操作するため、GPU を使う他の計測がある場合は
その計測が終わるまで待つ。これは承認ではなく共有資源の衝突回避である。
