# OpenWebUI single-worker製品計画

日付: 2026-07-10

Git commit: `dfc63de Add OpenWebUI single-worker product plan`

## 前回の要点

SQ8 P0-P7は完了し、Qwen3-14B-FP8の固定raw promptからR9700上で8 tokenを実生成できる。一方、canonical runtimeは固定prompt/context/greedyの監査経路であり、tokenizer、OpenAI HTTP、SSE、実行中cancel、常駐workerは未実装だった。次はrequest batchを実装せず、OpenWebUIから通常利用できるB=1製品を優先する方針になった。

## 今回の変更点

- `uLLM-project/docs/plans/openwebui-single-worker-product-plan-v0.1.md`をactive planとして保存した。
- v0.1はactive 1件、waiting 0件、同時requestは即時429とした。複数requestのbatch、waiting queue、prefix cache、自動履歴切捨て、request stop文字列は対象外とした。
- まずM=1 prompt経路をP7とvLLMの独立oracleで検証し、その後に同一request内だけの固定M=8 prefill chunkを追加する。このM=8はrequest batchingではない。
- 4096 contextは、512/4095 tokenの独立oracle、3584 prompt + 512 generationの境界試験、32/128/512/2048/3584 promptのhard TTFT gateを通すrelease条件とした。
- Rust resident workerがGPU/modelを所有し、Python FastAPI gatewayがQwen3 tokenizer、OpenAI Chat Completions、SSEを担当する境界を固定した。
- cancel後はKV reset完了を示す`released(reset_complete=true)`を受けるまで次requestを開始しない。worker hangもprogress eventとstartup/request/no-progress/cancel deadlineで検出し、gatewayごと終了してsystemdが再起動する。
- OpenWebUI接続は既存`open-webui-network`のhost gateway `172.20.0.1:8000`へ限定し、Bearer keyとbridge firewallを必須にした。2026-07-10時点ではnetworkとlocal imageは存在するが、OpenWebUI containerは稼働していないためP8-Fで再確認する。
- runtime、OpenAI/OpenWebUI、acceptanceの3観点でsubagent reviewを2回行い、phase依存、slow client、fatal recovery、persistent artifact、長文oracle、resource判定の指摘を統合した。

## 次の行動

P8-Aから開始する。最初に`/tmp`のcanonical artifactとthin packageを`/home/homelab1/datapool/ullm/product/`へストリーミングコピーし、hashを再検証してread-only化する。その後、serving session、worker protocol、OpenAI subset、oracleの4仕様と固定fixture/validatorを作る。これらが合格するまでruntime stateは変更しない。

