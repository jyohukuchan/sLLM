# Qwen3.5-9B AQ4 P2 source oracle

## 前回の要点

P2 baselineはsource oracleと同一artifact all-M=1 path oracleを分離して比較する必要がある。既存のQwen3 vLLM oracleはQwen3-14B-FP8用で、Qwen3.5-9B AQ4の契約にはならない。

## 今回の変更点

`tools/qwen35_aq4_p2_oracle.py` にbounded streaming payload、strict manifest、hash/path/symlink/finite/coverage検証とsource/path比較を追加した。capture CLIとvalidator CLIはsource、same-artifact path、共通linkを別rootで扱う。合成fixtureはvalidatorを通るが promotion不可である。

実環境のsource checkpointは `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B` に存在し、configは `qwen3_5`、text dtypeはBF16、safetensors indexと4 shard、tokenizerを確認できる。独立source forwardのbounded hidden/logit payload、固定revision、source runnerの出力はまだ存在しない。したがってcheckpointだけをproduction oracleとして扱わない。

## 次の行動

`tools/export-qwen35-aq4-source-oracle.py` を追加し、installed official `transformers` 5.12.1 + torch 2.12.0+cpu で実source forwardを取得した。GPU・networkなし、Qwen3.5-9B BF16 sourceを1 processでロードし、fixture 3 rowsを15.7秒で生成した。MemAvailableは62.9GB、checkpointは19.3GB、preflight要求は28.96GB、実行RSSは約15.6GBだった。`accelerate` は未導入のため low_cpu_mem_usage=True は使えず、low_cpu_mem_usage=Falseをruntime metadataへ記録した。

QAで、v1がcheckpoint全4 shardとruntimeをvalidator側で再hashせず、torch top-kの同値境界も規範化されていない問題を確認した。修正版は `source-oracle-v2/` として新規生成し、v1を上書きしていない。v2はconfig/index/全4 shardのstreaming SHA-256とcanonical checkpoint aggregate、tokenizer 5 files aggregate、runtime.json、CPU/BF16/package versions/thread数/preflight/row count/SHA256SUMSを相互検証する。greedyは全語彙最大logitの同値最小token ID、top-kはlogit降順・token ID昇順をstable full-vocab sortで取得する。

v2 manifestはrevision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`、checkpoint aggregate `22e7f39b92a49483961dd9ba316ff1b52d57f8c6b63a45e31449fca8fac807f0`、tokenizer aggregate `d31f0be10b0c77018130eea34a65078e0a37ca20c873e8be9ddfbda95b604ffc`、payload hash `7055590cc6f5d068d06edfa85a0464b38af42c7adb6eabfa219444cabd41d8a5` を固定した。validatorは `usable_as_source_evidence=true` かつ `promotion_eligible=false` を返した。path/link未実施のため、最終promotionは不可である。
