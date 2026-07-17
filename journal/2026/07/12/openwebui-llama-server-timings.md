# OpenWebUI llama-server互換timings

## 目的

uLLMの生成速度をllama-serverと同じ基準で計測し、OpenWebUI 0.9.4の
応答情報欄へtoken/sと詳細な終了理由を表示する。

## 確定した基準

- `predicted_n`は第1生成tokenとEOS tokenを含む全生成token数。
- `predicted_ms`は第1tokenのsample確定直後から最終tokenのsample確定直後まで。
- `predicted_per_second = 1000 * predicted_n / predicted_ms`。
- 1 tokenだけの完了では`predicted_ms`を`0.001 ms`へclampする。
- TTFT、terminal cleanup、reset、worker release、gateway detokenize、HTTP送信は
  `predicted_ms`へ含めない。
- prompt cache reuseはないため`cache_n=0`。

## 実装

- Rust workerは`started` flush完了後にprompt計時を開始し、各
  `DriverAdvance::Token`の直後にsample時刻を記録する。
- 正常な`released(stop|length)`へllama-server互換の9 timing fieldsを付与した。
  `cancelled`には成功性能値を付けない。
- gatewayはtimingsのfield set、token count、有限正値、rate式を検証する。
- non-stream responseはtop-level `timings`を持つ。
- streamはllama-serverと同じく、`include_usage=false`ではfinish chunk、
  `include_usage=true`では最後のusage chunkへtop-level `timings`を置く。
- 公開timingsへ標準`finish_reason`と詳細`termination_reason`を追加した。
  対応は`stop/eos_token`、`length/max_tokens`、`length/context_length`。
- OpenWebUI modelの`capabilities.usage=true`を設定した。OpenWebUI本体の追加patchは
  不要で、既存middlewareが`usage`と`timings`を統合し、情報tooltipが全項目を表示する。

## 検証

- `cargo test -p ullm-engine --lib -- --test-threads=1`: 365 passed。
- gateway full pytest: 123 passed。
- worker acceptance/configure tests: 103 passed、22 subtests passed。
- gateway mypy strict、ruff check、ruff format check、cargo fmt check、diff check: passed。
- release worker build: passed。
- live SSEで次を確認した。
  - `predicted_n=3`
  - `predicted_ms=71.65092800000001`
  - `predicted_per_second=41.869660083118525`
  - `finish_reason=stop`
  - `termination_reason=eos_token`
- OpenWebUI browser smokeで情報tooltipに
  `predicted_per_second`、`finish_reason`、`termination_reason`が表示された。
  同じ画面で標準usage、9 timing fields、正規化済みinput/output token数も確認した。

## 配備

- `target/release/ullm-sq8-worker`を再buildし、`ullm-openai.service`を再起動した。
- OpenWebUIを停止して`configure.py`を再実行し、DB backupを生成してから再起動した。
- 最終確認時点でgatewayはactive、OpenWebUIはhealthy、再起動回数は0。

## 制約

- promptと要求生成上限の合計が4096を超える要求は生成前HTTP 400になるため、
  assistant messageの情報欄は作られない。
- cancel、client disconnect、worker failureは成功final/usageを送らないため、
  token/sや成功終了理由を捏造しない。
- contextをちょうど使い切って`length`になった成功応答は
  `termination_reason=context_length`とし、通常の生成上限は`max_tokens`とする。
