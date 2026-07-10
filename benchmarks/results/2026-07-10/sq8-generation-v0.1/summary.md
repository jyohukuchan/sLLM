# SQ8 Qwen3-14B real generation evidence

日付: 2026-07-10

## 判定

P7の最低限の実生成経路は合格とする。Qwen3-14B-FP8の固定raw prompt `[1,2,3,4,5,6,7,8]` から、R9700上のuLLM SQ8経路がvLLM oracleと同じ8 token `[353,10,4999,1725,15,16,17,18]` を生成した。全8 stepの数値gate、40層paged KV、token feedback、allocator release、fallbackなしを昇格結果と独立validatorで確認した。

定常測定10回も全回で同じtoken列、`finish_reason=length`、feedback、allocation releaseを再現した。これはB=1、prompt 8、generation 8、context 16の最小offline generationであり、tokenizer、複数request batching、streaming API、production servingは対象外である。

## 固定条件

- model: `Qwen/Qwen3-14B-FP8`, revision `9a283b4a5efbc09ce247e0ae5b02b744739e525a`
- artifact content SHA-256: `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`
- package manifest SHA-256: `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb`
- device/profile: R9700 `gfx1201`, `rdna4_w8a8_block_ck`, HIP kernel guard 10件
- generation: greedy、temperature 0、EOS早期終了有効、BOS/chat template/detokenizeなし
- benchmark: warmup 3回、測定10回、model load除外、次要求に必須の40層KV resetを主要時間へ算入

## 正しさ

- current-HEAD P6: 40 layers、280 projections、160 activation quantizations、top-1 `353`
- P7 promotion: 8/8 token一致、全step top-1一致、top-10 overlap最小9
- 最悪relative L2: step 5 logits `0.1000727613`、最低cosine `0.9951947454`
- final KV length: 全40層で15、decode positionは8から14、feedback 7回
- 10 measured repeats: exact token、token hash、feedback、allocation releaseが全回一致
- fallback: false、host tensor staging: false、終了後KFD process 0、VRAMは基準値へ復帰

## 定常測定

| runner | p50 latency | p95 latency | request/s | generated token/s | total token/s |
|---|---:|---:|---:|---:|---:|
| uLLM audited SQ8 cycle | 337.610 ms | 338.322 ms | 2.9626 | 23.7011 | 47.4021 |
| vLLM `LLM.generate` | 312.966 ms | 314.758 ms | 3.1944 | 25.5551 | 51.1103 |

uLLMの生成throughputはこの測定でvLLMの`92.745%`、vLLM/uLLM比は`1.0782x`だった。uLLMのp50はvLLMより`7.874%`長い。

ただし、これはproduction engine性能の同等比較ではない。uLLM側はresetに加えて各stepのfull hidden/full logits readback、host top-10 scan、hash・runtime contract検証を含む監査経路であり、vLLM側はlogprobsなしの`LLM.generate` wall timeである。vLLMの`RequestOutput.metrics`が取得できなかったため、uLLM内部TTFT `38.449 ms`とdecode `25.155 token/s`は直接比較しない。

## 証跡

- `p6-current-head.json`: `71ec45a2ea9edc7d7e7b23d66b23e171ad452e08252f4a358aa38521d0f514c5`
- `generation-run-01.json`: `cafd46e09d7f42e95dc021fc5d1a45e2dc54ab78f8f2afabfe261dac4971be04`
- `generation-benchmark-promotion.json`: `a9a1a4158a55cbb04a8da411b2dee5f676b149654df88f29926878bdaf9b28e0`
- `ullm-throughput-m8-g8.json`: `ec79d624888909bbd0f018993116859ea8fe611db61cdda58eae9d62a59c13b3`
- `vllm-throughput-m8-g8-v0.2.json`: `e5aaf99a37c5ca683d24fac038566bc37186d5ebcb89adbe45e6e36b0c44c1be`
- `ullm-throughput-run.log`: `025dd4a30fcf20e746229c9fd1b7a1396a1582459efa7ed50bc89ed12bb23df2`
- measurement commit: `63f5b0cdefc1e0f44cf513ecdc36734ec711e372`
- release binary SHA-256: `fe6ce16a6c8985deae2cc8eee9b33df239e792a9cf8432ef523d48bd0276a2d1`

独立validatorはbenchmark/generation/P6の合計129改ざんテストを通過した。新しいtiming resultは固定promotion trust anchorではないため、generation validatorでは`--contract-only`で検証し、benchmark validatorがpromotion result SHA-256を実ファイルへ照合した。

## 次の行動

P7の最低限実装はここで完了とする。次の開発単位は監査readbackを外したlean generation path、tokenizer/API統合、または複数request batchingのいずれかとして別計画に分ける。今回の23.70 token/sをproduction throughput基準には昇格しない。
