# MI300X 外部 engine benchmark 実行計画

Date: 2026-07-26

## 前回の要点

- 親の existing-engine-benchmark-plan v0.1 は vLLM / SGLang / ROCm-ATOM を R9700 smoke/representative baseline、MI300X を future grid としていた。
- uLLM 側には SQ8_0 CDNA3 MI300X A′ 実機検証チェックリストがあり、preflight 5--10 分、fragment/lane 2--5 分で最短 go/no-go を 10--20 分に置いている。
- Qwen/Qwen3-14B-FP8 の local artifact は revision 9a283b4a5efbc09ce247e0ae5b02b744739e525a、4 shard + tokenizer、約 16.34 GB である。

## 今回の変更点

- MI300X×1 用の外部 engine runbook を新設し、uLLM gate の後に vLLM、SGLang、llama.cpp を一 engine ずつ接続する順序と hard timebox を固定した。A′ fail は uLLM 後続だけを止め、外部 engine の hardware-target 計測は続ける。
- vLLM/SGLang は同一 FP8 source revision、llama.cpp は公開 Qwen3-14B Q8_0 GGUF を使う。llama.cpp の format 差と KV dtype 差を隠さず result に残す。
- common OpenAI-compatible HTTP client を primary metric にし、1024 prompt / 20 output、C=1..128、各 leg 3 trial、prefix/prompt cache 無効化、TTFT/ITL/総 throughput/telemetry を同じ定義で採るようにした。llama-bench は supplementary microbenchmark に限定した。
- 現時点で確認できた vLLM v0.26.0 ROCm image、SGLang v0.5.16 MI30X ROCm image、llama.cpp b10107 を planning snapshot として記録した。lease 時には latest release、help、image digest、commit を再記録する。
- GPU、service、active manifest、/opt/ullm、既存 kernel、既存 result は操作していない。

## 次の行動

1. GPU lease 前に FP8 artifact と Q8_0 GGUF を persistent storage へ配置し、同一路 transfer 実測と SHA-256 を記録する。
2. shared 1024-token workload を uLLM comparison harness でも読めることを CPU-only で確認する。
3. 借用時は uLLM preflight -> fragment/lane を先に実施し、続けて外部三 engine の image/device admission と C=1..128 sweep を runbook の cap 内で消化する。
4. raw JSON、AMD SMI telemetry、version/digest、normalization output を持ち帰り、physical HBM/L2 counter は metadata が確認できた場合だけ別欄で解釈する。
