# SQ8 P8-A serving契約とfixture

日付: 2026-07-10

状態: 完了

## 前回の要点

P0-P7で固定raw promptのSQ8実生成と監査経路は完成した。次はrequest batchingを追加せず、B=1でOpenWebUIから使える製品を作る。P8-Aではruntime stateを変更する前に、永続artifact、serving/worker/OpenAI/oracle契約、fixture、独立validatorを固定する。

## 今回の変更点

- canonical artifact 13.21GB/280 pairとthin package 3.11GB/163 payloadを`/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/`へstreaming copyした。
- artifact content SHA-256 `2243acf1...b9147`、package manifest SHA-256 `c2133dfe...a0eb`、全payloadを再検証し、artifact/package treeをread-only化した。
- `f036a8d`でproduct promotion独立validatorと6 testsを追加し、全16.3GB payload hash検証まで合格した。
- OpenWebUI local image v0.9.4、digest `sha256:a6da0c29...dbff`、revision `f51d2b02...70`を一時containerで起動し、実`/api/chat/completions` proxyのstream/non-stream forwarded bodyを捕捉した。
- 実captureでは`metadata`が上流bodyから除去され、`max_completion_tokens`が`max_tokens`へ変換された。Authorization、cookie、tokenはfixtureへ含めていない。
- `0ef7cc6`でOpenAI Chat Completions subsetを固定した。
- `e360c05`でraw prompt 1/8/32/128/512/4095、OpenWebUI capture、oracle placeholder、独立validator、trusted manifest anchorを保存した。23 tests + 8 subtests、全17 checksumが合格した。
- `2ab2bf6`で実Qwen tokenizerによるchat fixture 10件を固定した。exact 32/128/512/2048/3584 tokenを含み、独立再計算validatorを合わせた28 tests + 8 subtestsが合格した。
- `d22ff9c`でserving sessionとworker JSONL protocolを固定した。active 1/waiting 0、slow-client即時cancel、cancelからreleaseまでの5秒hard deadline、固定モデルidentityを含む。
- `4bb684a`でrelease測定契約を固定した。systemd cgroup v2 `memory.current`、R9700 process VRAM/KFD照合、percentile、Theil-Sen、normal/restart別baselineを含む。
- `b951c85`で実vLLM oracle exporter、独立validator、22 testsを保存した。g1/g8/g64に加えて、feasibleな5 promptでg512 `ignore_eos`も取得する21-run契約である。
- R9700上の実captureは6 prompts/21 runsに成功した。metadata SHA-256は`1710ebf5...c122b`、payload manifest SHA-256は`5972a024...7405c`で、全44 file checksumと独立validatorが合格した。
- 実oracleを`/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/oracles/vllm-source-v0.1/`へ複製し、tree外anchorでpromotion検証後にread-only化した。
- `521c70f`で実oracleをchecked-in fixtureへpromotionした。完成rootは94 files、manifest SHA-256は`3b6362fd...f83`で、active placeholderは除去した。
- 初期入力と未取得placeholderは`provenance/bootstrap-input-v0.1/`へ、実測値は`oracles/vllm-source-v0.1/`へ固定し、completed root・bootstrap・real oracleの3段階anchorを検証する。
- 完成fixture、実oracle、chat templateの独立validatorを含む関連61テストが合格し、`git diff --check`も合格した。

## 次の行動

P8-Bのvariable-length lean serving sessionへ進む。HTTP workerはまだ追加せず、可変長prompt、M=1 prefill/decode、4096-token KV、reset/abortの正しさを先に固定する。P7 audited pathの出力schemaは変更しない。
