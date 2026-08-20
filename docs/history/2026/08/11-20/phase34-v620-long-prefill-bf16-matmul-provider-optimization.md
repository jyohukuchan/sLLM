# Phase 34: V620長行prefill BF16 matmul provider比較・最適化

> 状態: 完了（shape-aware hipBLAS限定採用、MTP correctness修正）
> 実施日: 2026-08-20

## 結論

exact gfx1030 V620の長い内部BF16 projectionへ、既存の`hipblasGemmEx` providerをshape-awareに限定採用した。
Phase 9の短い`M=17`判断を全`M>8`へ一般化していたselectorが10,001行にもtiled16を使わせ、16x16 tile、scalar FP32 K loop、
多数barrierを248回繰り返していたことが主因だった。新しいGEMM kernel、graph複製、weight repack、runtime autotunerは追加していない。

担当AI裁量では、採用scopeのdevice絶対短縮が約51.4秒、full-model短縮が約54.6秒と大きく、既存provider再利用、静的な6 shape、
二つのM threshold、容易なrollbackで表現できるため採用が妥当である。small-Nと未知shapeの悪化・不安定性は既存providerへ隔離した。

## 比較とcrossover

同一buffer/stream上でcurrent tiled16とhipBLASを比較するprivate toolを追加した。screenは7 production shape ×
`M=17/256/2048/10001`の28 pair、refinementは主shapeの32/64/128、K/VとN=32の追加点、final boundaryは
127/128/129および1023/1024/1025を測定した。採用判断へprofiler wallは使わず、HIP event device timeを使った。

- 10,001行の248-call加重予測: 62.525958秒 → 11.081425秒、82.277%短縮。
- final routeのrocprof確認: hipBLAS 200 call / 11.683135秒、tiled16 48 call / 0.076276秒。48 callは意図したN=32 complementだった。
- profile projection合計: Phase 33 baseline 66.5609秒 → 11.759411秒、82.333%短縮。
- representative stress `M=128,K=2560,N=4096`: 2.532 ms → 0.517 ms、79.58%短縮。
- representative stress `M=10001,K=2560,N=9216`: 412.646 ms → 74.120 ms、82.04%短縮。
- hipBLAS handle作成は約0.08〜0.17 ms、最初のlibrary callは約0.17〜0.26秒だった。

主shapeは`M=64`でもsteady-stateでは勝ったが、約0.18秒のfirst-call費用を248-call加重利益で明確に上回り、noiseや将来solution driftにも
余裕を持つ境界として`M>=128`を選んだ。`K=2560,N=1024`はM=512の差が小さく不安定で、M=1024から約81%短縮したため
thresholdを1024とした。`N=32`はMを増やしてもwinnerが非単調で絶対寄与も小さいためtiled16を維持した。

## Production routing

exact gfx1030で次だけ`matmul.hipblas.gemm_ex.v2`へ送る。

- `M>=128`: `(K,N)=(2560,9216),(9216,2560),(2560,8192),(2560,4096),(4096,2560)`
- `M>=1024`: `(K,N)=(2560,1024)`

`(2560,32)`、`(2560,248320)` all-logits、未知shape、短Mは既存selectorのcomplementを維持する。gfx1030 contextへhipBLAS handleを
一つ追加したが、hipBLASLtは作らない。gfx1201/gfx942は従来どおりである。実行失敗後のsilent retry fallbackは追加していない。

rollbackは`phase34_gfx1030_hipblas_shape` routeとgfx1030 hipBLAS handle作成条件を除く二点で、旧tiled16へ戻せる。

## 数値分類

rocprofで選択solutionを確認した結果、GSU1でglobal split/atomic combineは使われていなかった。両providerはBF16 input/weight、
FP32 compute、BF16 RNE outputを維持し、同じK項を一度ずつ含む。演算順は異なるためbit exactのN0ではないが、決定的な並べ替えであり、
Phase 8の保守的な`gamma_K * sum(abs(a_i*w_i)) + BF16 half-ULP` worst-case boundは増えないためN1とした。

符号と指数を混ぜたstress入力では`M=128,K=2560,N=4096`で519要素、`M=10001,K=2560,N=9216`で84,764要素が異なり、
最大provider差は16/32 BF16 valueだった。両providerのrepeat digestは一致し、sampled F64 oracleのbound違反は0だった。
final matmul G1はgfx1030/gfx1201とも18/18 PASS、fallback/cleanup 0だった。

## Full model

fixed Qwen3.5-4B BF16、FP16 KV、10,001 prompt / 2 output、one chunkの結果は次のとおり。

| case | Phase 34 baseline | final | 差 | audit |
| --- | ---: | ---: | ---: | --- |
| V620 gfx1030 | 89.249 s | 34.684 s | 61.14%短縮 | `[2064,5686]`、HIP-only、fallback false、cleanup 0 |
| R9700 gfx1201 control | Phase 33 final 75.553 s | 75.316 s | 0.31%短縮相当 | route不変、同token/audit |

V620 finalのworkspace arenaはbaselineと同じ5,278,049,280 byteだった。32 prompt controlは4.473秒から4.652秒へ4.00%長く見えたが、
short-M providerは変更されずhandle作成は0.1 ms未満なので、単一fresh processのmodel-load/DPM noiseとして速度claimに使わない。

同じfinal gfx1030 serverで10,001-token FP16 KVのOpenAI non-stream/SSEを各1回実行し、どちらも1 token `It`、usage
10,001+1、SSE terminal `[DONE]`を返した。別の10,001-token SSEを1秒でdisconnectするとshutdown auditは`cancelled`、
直後のsmall recoveryは`Hello`だった。shutdown後のcurrent/request-state/workspace byte、retryable cleanup、durable quarantineは0だった。

## MTP verify row不具合

R9700 long controlの初回実行は`target verify row count differs from draft width`で停止した。Phase 24のterminal-row compactionが、
実際には2行だけのspeculative verify blockにも10,001行target graph容量を基準として`Last`を選んだことが原因だった。

`run_transition`へ`force_all_terminal_rows`を追加し、MTP targetの`decode_block_with_mtp_state`とevidence variantだけtrueにした。
通常prefill、通常decode、partial replayはfalseのままである。`large_target_graph_preserves_every_speculative_verify_row`を追加し、
大きいtarget graphでも2行のhidden stateとtokenを保持してresolveできることを固定した。これは正常経路の数値式を変えず失敗を正すN0である。

## Verification

- `cargo +1.97.1 test --locked --offline -p sllm-core -p sllm-hip`: PASS。
- `cargo fmt --all --check`: PASS。
- exact gfx1030/gfx1201 release build: PASS。
- exact gfx942 ROCm 7.14 / Code Object V6 / wave64 compile-only: PASS。
- gfx1201 binaryをV620へload: exit 1、`requested device gcnArchName does not match exactly`でfail-closed。
- final GPU runはHIP-only、fallback false、cleanup 0。raw trace、binary、modelはGit追跡対象外。
- gfx1030 OpenAI 10,001-token non-stream/SSE/disconnect/recovery/graceful shutdown: PASS。
- cumulative integration review: blockerなし。selector境界、handle lifetime、MTP call site、scope外route、summary/compatibilityの整合を再確認した。

## 限界と再検討条件

結果はROCm 7.14、exact V620/R9700、Qwen3.5-4Bの固定shapeに限定する。Tensile solutionはROCm更新で変わり得るため、software tuple変更時は
actual solution、operator boundary、10k full-modelを再取得する。別model/SKU、concurrent throughput、dynamic FP8 KV wall、N=32用custom kernel、
universal gfx1030 crossoverへ一般化しない。

[対応するarchive plan](../../../../plans/archive/2026/08/11-20/phase34-v620-long-prefill-bf16-matmul-provider-optimization.md)
[bounded summary](../../../../../ci/matrix/phase34-v620-prefill-matmul-summary-v1.json)
[数値・出力影響変更台帳](../../../../compatibility/numerical-output-changes.md)
[メイン計画](../../../../plans/main-plan.md)
