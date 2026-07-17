# P2 pure-prefill fixture/driver implementation

## 前回の要点

- expanded P2 prefill casesは`generated_tokens=0`であり、通常の`Qwen35Aq4InferenceSession`は最低1生成を要求するため、そのままでは純粋なprefillを実行できなかった。
- 既存のworker/sessionや実行中サービスは変更せず、driver専用のresident model runtime経路に限定する方針を採用した。

## 今回の変更点

- `ullm-aq4-p2-full-model`は`step_count=0`を純粋prefillとして受理し、prompt chunkを`ColdPrefill`でdispatch、chunkごとに同期、最後にrequest stateを一度だけresetする。
- `M1`および`M>1`のtail幅1を明示的にtoken dispatchへ分岐し、decode/publicationを生成しない`prefill_complete`結果と構造化audit/lifecycle/timingを発行する。
- expanded caseから、served-modelのcontext/vocab/EOS/reasoning/tokenizer hash契約に束縛したdeterministic token-id fixtureをケースごとに生成する`tools/generate-aq4-p2-fixtures.py`と、subset件数・symlink・重複キー・上書き拒否のテストを追加した。公開indexにはtoken列を出さず、fixture SHAとtoken列SHAだけを残す。

## 次の行動

- parent agentがP2 identity/policy/resource observerを束ねたrun-rootで、smoke全件およびrepresentativeの対象点を実行する。
- 実行後、短いpromptで`actual_token_batch_width`が契約上のresolved Mと異なるケース（full stageのprompt 1等）がある場合は、ケース契約かadapterの幅表現を別途判断する。

## 検証

- `cargo check -p ullm-engine --bin ullm-aq4-p2-full-model`
- `cargo test -p ullm-engine --bin ullm-aq4-p2-full-model pure_prefill -- --nocapture`
- `cargo test -p ullm-engine --bin ullm-aq4-p2-full-model fixture_loader_rejects_symlink -- --nocapture`
- `pytest -q tests/test_aq4_p2_fixture_generator.py`
- `python3 -m py_compile tools/generate-aq4-p2-fixtures.py tests/test_aq4_p2_fixture_generator.py`

