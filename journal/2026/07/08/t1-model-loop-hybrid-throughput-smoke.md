# T1 model-loop hybrid throughput smoke

## 前回の要点

- full-package logical gridは保存できたが、real request-batch性能ではない。
- `package-self-attn-mlp-block-model-loop-smoke` にはselected layer stackとdecode ready batchの足場があった。
- ただしtimed throughput fieldsとJSONL parserが不足していた。

## 今回の変更点

- model-loop smokeにprefill/decode/end-to-end token数、wall time、TPSを出すfieldを追加した。
- 空白入り配列でparserが壊れないよう、`layers_csv`、`prompt_tokens_csv`、`generated_tokens_csv` なども追加した。
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` を追加した。
- R9700で layers `3,7`、sequence_len `3` のJSONL smokeを実行した。

## 結果

- `batching.mode=hybrid`
- `prefill_real_batch=false`
- `decode_real_batch=true`
- `decode_executor_request_parallelism=2`
- prefill total tok/s: `78.702126`
- decode generated tok/s: `78.214266`
- end-to-end tok/s: `78.492300`
- verified: `true`

## 次の行動

1. このhybrid rowはselected-layer stack guardとして扱う。
2. 次はtoken-id full package pathとmodel-loop stack runnerの接続点を作る。
3. prefillもrequest-batch化できた段階で `batching.mode=real` へ昇格する。
