# Qwen3.5-9B AQ4 worker backend

## 実装

- `ullm-aq4-worker`を追加し、既存の`ullm.worker.v1` reader/writer/controlを再利用した。
- 現段階は互換backendであり、常駐workerから要求ごとに既存の
  `ullm-engine package-token-ids-bench`を子プロセスとして起動する。
- AQ4 CLIのJSON reportをtoken、EOS/length、llama.cpp互換timingsへ変換する。
- cancelを10ms間隔で確認し、子プロセスをkill/waitしてからcancelled terminal eventを出す。
- stdoutは別threadで最大8MiBまで読む。pipe詰まりと無制限なメモリ使用を避ける。
- worker profileを環境変数で設定可能にし、従来SQ8値を既定値として維持した。
  AQ4 worker自身の既定値はvocab 248320、context 4096、EOS 248044/248046、top-k 1。
- gateway互換の`--artifact AQ4_PACKAGE --package COMPAT_PATH`と直接実行用の
  `[--engine PATH] --package AQ4_PACKAGE`を受理する。

## 制約

- worker processは常駐するが、AQ4 weightsは要求ごとに読み込む暫定fallbackである。
- AQ4 CLIはgreedy top-1生成であり、温度・top-p samplingはまだ実生成へ反映しない。
- token eventは子プロセス完了後にまとめて通知する。真の逐次streamingにはAQ4 model loopの
  library resident session化が必要である。

## 検証

- `cargo test -p ullm-engine aq4_worker -- --test-threads=1`: 2 passed
- `cargo test -p ullm-engine sq8_worker_protocol -- --test-threads=1`: 33 passed
- `cargo check -p ullm-engine --bin ullm-aq4-worker`: passed
