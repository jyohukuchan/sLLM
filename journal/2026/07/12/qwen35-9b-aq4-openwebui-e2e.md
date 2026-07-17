# Qwen3.5-9B AQ4 OpenWebUI E2E

## 実装結果

- `ullm-aq4-worker`を追加し、既存AQ4 CLI生成を`ullm.worker.v1`へ接続した。
- gatewayのmodel ID、context、vocab、EOS、top-k、identity、HIP設定を環境設定化した。
- Qwen3.5-9B tokenizerをファイルhashとchat template hashで固定するprofileを追加した。
- OpenWebUI登録とbrowser smokeをmodel設定可能にした。
- AQ4 packageを`/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1`へ配置した。

## 実機で見つけた問題と修正

- `HIP_VISIBLE_DEVICES=1`でもuLLM runtimeのdevice 0はCPU fallbackであり、R9700はdevice 1だった。
  AQ4 workerの配備既定をdevice 1へ修正した。
- AQ4が生成完了後にprompt全体を1 unitとして通知すると、既存progress trackerが非連続として拒否した。
  128-token unitと1-token末尾unitへ分けて通知するよう修正した。
- gateway tokenizer identityがQwen3-14B専用だったため、Qwen3.5-9B固定profileを追加した。

## 検証

- R9700直接生成: `verified=true`、device 1、`AMD Radeon Graphics`。
- worker JSONL: started/progress/token/releasedを完走し、4-token生成で
  `predicted_per_second=96.22287913047234`。
- OpenAI非stream HTTP: `AQ4_OK`、finish reason `stop`、termination reason `eos_token`、
  `predicted_per_second=95.19042749830608`。
- OpenAI SSE: `STREAM_OK`、最終usage/timings、`predicted_per_second=106.95689348408979`。
- OpenWebUI browser smoke: `AQ4_BROWSER_OK`、モデル表示、
  `predicted_per_second`、`finish_reason`、`termination_reason`を確認した。
- gateway tests: 133 passed。
- OpenWebUI configure tests: 3 passed。
- Rust AQ4 backend、worker CLI、worker protocol testsとAQ4/SQ8 worker checkが成功した。

## 現在の配備状態

- `ullm-openai.service`はAQ4 profileでactive。
- OpenWebUIはhealthy、restart count 0。
- `/etc/ullm/openai-gateway.sq8.env`に切替前のSQ8設定を保存した。

## 制約

- AQ4 packageは要求ごとに再読込する。
- token eventは生成完了後にまとめて通知するため、HTTP上の逐次表示は遅延する。
- AQ4 CLIはgreedy top-1であり、temperature/top-pは生成へ反映しない。
- 次の優先作業はAQ4 model loopのlibrary resident session化とtoken逐次callbackである。
