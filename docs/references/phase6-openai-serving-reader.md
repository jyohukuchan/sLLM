# Phase 6 OpenAI serving facts-only reader

観測日は2026-08-13。次のsourceはno-copyのfacts-only readerであり、code、test body、型定義、近接した
pseudocodeをsLLMへ持ち込まない。正確なcommitとpath一覧は
[`ci/contracts/phase6-a2-v1.json`](../../ci/contracts/phase6-a2-v1.json)へ固定した。

## 観測source

- vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`: chat/model protocol、serving、router、engine protocol。
- SGLang `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`: protocol、serving base/chat、SSE、usage processor。
- TensorRT-LLM `376f7e1bd8ed543f75014309e3fd4b237e9b0e73`: OpenAI protocol/server/service、postprocess handler。
- LMDeploy `f4b8140ba19cd823c541241cbb113cc32f854e6a`: protocol、API server、chat completion serving、utility。

## 抽出した技術的事実

- validation、request rendering、generation、response変換を分離できる。
- streamとnon-streamは共通の正規化generation結果から構築できる。
- SSEはrole/content/finish reason/usage/terminatorを順序付きeventとして扱える。
- usage計算を一箇所に置くとstream/non-streamの不一致を防げる。
- HTTP disconnectをgeneration ownerのabort/cancelへ伝播できる。
- response header前のJSON errorと、header後のstream failureは別のwire behaviorを必要とする。
- bounded queue/channelでadmissionとslow consumerのbackpressureを分けられる。

これらは各engineがOpenAI仕様のoracleであるという意味ではない。sLLMの外形は固定profileを正本とし、
readerにある拡張fieldや独自finish reasonを持ち込まない。

## implementationへ渡すacceptance cases

1. generation開始前にstrict validationを完了する。
2. 固定requestでstream/non-streamのvisible text、usage、finish reasonを一致させる。
3. role delta、content delta、final finish reason、exact `data: [DONE]\n\n`の順をbyte testする。
4. prompt/completion/total tokenを共通accountingから生成する。
5. disconnectは同一requestを一度だけabortし、request-local stateをcleanupする。
6. pre-header JSON errorとpost-header terminal stream behaviorを別testにする。
7. active/queued/stream channelをboundedにし、queue fullとslow consumerをtestする。
8. unknown fieldとprofile外fieldをfail closedにする。
