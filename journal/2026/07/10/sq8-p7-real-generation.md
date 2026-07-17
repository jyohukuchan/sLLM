# SQ8 P7 real generation

日付: 2026-07-10

## 前回の要点

P6ではQwen3-14B-FP8のM=8 prefillを40層resident実行し、final hidden、全151,936 logits、top-1を独立CPU SQ8 oracleとsource vLLM oracleへ照合した。最終resultは`bcc7355ed9deed102ae620bf237466d2aebd78b897dd776fcb088377d4241647`で、device/CPU/vLLMのtop-1はすべて`353`、fallbackと測定中host stagingは0だった。

## 今回の変更点

- schedulerにprefill logits由来の最初のtokenだけを生成数へ加え、KV長を増やさない遷移を追加した。`max_new_tokens=1`、二重記録、prefill前、decode進行後、未知requestを無変更で検査する。
- `PagedDecodeState`へdevice bufferからの連続KV書き込みとM=1 decodeを追加した。host copyと内部同期を挟まず、CPUテストで既存host経路とKV layout・attention出力が一致した。
- SQ8 layerへ、M=8 causal prefillを維持しながらK/Vを保存する経路と、`position == written_len`を要求するM=1 paged decode経路を追加した。40層stackではprefill 320 KV write、decode stepごとに40 KV write・40 paged attention・280 CK projection・160 activation量子化を型付きreportへ固定した。
- exact BF16 `model.embed_tokens.weight`をresident loadし、HIP `bf16_row_f32`でtoken rowをM=1 F32 device bufferへ取得するruntimeを追加した。package manifest、payload SHA-256、shape、R9700 identity、kernel guardをfail-closedで検証する。
- M=1 final norm、BF16 lm_head、HIP top-1と独立host top-1照合を追加した。P6 APIと常駐device buffer数は維持した。

### Frozen vLLM generation oracle

- directory: `/tmp/ullm-qwen3-14b-fp8-vllm-generation-m8-g8-v0.1`
- metadata SHA-256: `5fc03a28cd15409e84a7fd23fd51c0cbd6ec9cf8761a66d1f5ede7ddfe3226a0`
- model revision: `9a283b4a5efbc09ce247e0ae5b02b744739e525a`
- prompt token IDs: `[1,2,3,4,5,6,7,8]`
- generated token IDs: `[353,10,4999,1725,15,16,17,18]`
- forward token counts: `[8,1,1,1,1,1,1,1]`
- decode input/position: `353@8`, `10@9`, `4999@10`, `1725@11`, `15@12`, `16@13`, `17@14`
- generation: greedy、temperature 0、BOS/chat templateなし、固定8 token、EOS出現なし、finish reason `length`
- validator: trust anchor、全payload SHA/health、feedback、position、top-1/top-10、schema、source revision、TOCTOU、no-clobberを検査し、15 testsが合格した。

### Commits

- `4152c79`: prefill token scheduler transition
- `2d1459b`: device-resident paged attention
- `2e7bace`: trusted vLLM generation oracle
- `eefe089`: 40-layer paged stack execution
- `0d6f71a`: resident BF16 token embedding
- `5673fc1`: M=1 decode head and HIP top-1
- `2cebd09`: device-resident M=8 stack input
- `eac9094`: paged decode stateとM=1 head reportの直接結合
- `2fb4345`: fail-closed real generation loop
- `40afa10`: source vLLM比較付きgeneration gate
- `2bc9708`: 測定外resetによる反復実行API

### Current-HEAD P6 regression

- result: `benchmarks/results/2026-07-10/sq8-generation-v0.1/p6-current-head.json`
- result SHA-256: `71ec45a2ea9edc7d7e7b23d66b23e171ad452e08252f4a358aa38521d0f514c5`
- log SHA-256: `35be67cbfa6a48ce37c2f3ba462364540e6aeafc9077ef5ef875f7f4fdc6e5e4`
- independent validator: `passed=true mode=contract-only layers=40 projections=280 activation_quantizations=160 hash_stability_checks=13 top1=353`
- timing: full p50 `34.201610 ms`、stack p50 `29.996848 ms`、head p50 `4.199236 ms`

attention分岐導入前のP6 resultを現HEADへ流用せず、同じCPU SQ8 oracle、40層boundary、final head、source vLLM gateを再実行した。旧P6 full p50 `34.181663 ms`と同水準で、top-1と実行契約も維持した。

### P7 R9700 generation result

- result: `benchmarks/results/2026-07-10/sq8-generation-v0.1/generation-run-01.json`
- result SHA-256: `cafd46e09d7f42e95dc021fc5d1a45e2dc54ab78f8f2afabfe261dac4971be04`
- log SHA-256: `803434fb04a16e487a800aee5446da964d13425acae350f5b8b9f6252f2e0c9e`
- generated token IDs: `[353,10,4999,1725,15,16,17,18]`。vLLM oracleと8/8一致。
- decode inputs: `[353,10,4999,1725,15,16,17]`、positions `[8,9,10,11,12,13,14]`、全40層のfinal KV lengthは`15`。
- 全8stepでfinal hidden/logitsはfinite。source vLLM比の最悪relL2はstep 5 logitsの`0.1000727613`、最低cosineも同stepの`0.9951947454`。top-10 overlapは最小9、top-1は全step一致した。
- totals: embedding gather `15`、prompt embedding D2D `8`、stack input D2D `8`、projection `2240`、activation quantization `1280`、layer output D2D `320`、KV write `600`、paged attention `280`、head `8`、scheduler decode advance `7`。
- fallback `false`、host staging `false`、allocatorは実行前後ともfree block `1` / allocated block `0`。
- first-run TTFT `194.071994 ms`、request latency `515.875351 ms`、generated TPS `15.507622`、decode TPS `21.752414`。これはsource-correctness gateの1回目であり、steady-state比較値とは分離する。
- 実行後は全GPU VRAM `0%`、関連processなし。

### P7-E steady generation and vLLM reference

- uLLM benchmark: `benchmarks/results/2026-07-10/sq8-generation-v0.1/ullm-throughput-m8-g8.json`
- uLLM result SHA-256: `ec79d624888909bbd0f018993116859ea8fe611db61cdda58eae9d62a59c13b3`
- uLLM promotion SHA-256: `a9a1a4158a55cbb04a8da411b2dee5f676b149654df88f29926878bdaf9b28e0`
- vLLM benchmark: `benchmarks/results/2026-07-10/sq8-generation-v0.1/vllm-throughput-m8-g8-v0.2.json`
- vLLM result SHA-256: `e5aaf99a37c5ca683d24fac038566bc37186d5ebcb89adbe45e6e36b0c44c1be`
- workload: raw prompt 8、generation 8、context 16、B=1、greedy、temperature 0、EOS早期終了有効、BOS/chat template/detokenizeなし。
- warmup 3回、測定10回。uLLMはpromotion実行をwarmup 1回目として数え、追加2回後に測定した。
- vLLM側は旧v0.1の`min_tokens=8`、`ignore_eos=true`を採用せず、uLLMと同じ`min_tokens=0`、`ignore_eos=false`へ直してv0.2を再実測した。
- uLLMの次要求には40層KV resetが必須なので、主要throughputはreset開始から`run_fixed_synchronized`完了までの定常cycleで算出した。resetを除外した値をvLLM request/sへ直接比較する旧案は撤回した。
- uLLM p50/p95は`337.610266 / 338.321527 ms`、aggregate generated TPSは`23.701058`、request/sは`2.962632`。
- vLLM p50/p95は`312.965839 / 314.757984 ms`、aggregate generated TPSは`25.555142`、request/sは`3.194393`。
- uLLM/vLLM generated TPS比は`0.927448`、vLLM/uLLMは`1.078228x`、uLLM p50は`7.874%`長かった。
- uLLM measured 10回は全てtoken `[353,10,4999,1725,15,16,17,18]`、finish reason `length`、feedback、allocation releaseを再現し、fallback/host tensor stagingはfalseだった。
- uLLM側はKV reset、full hidden/full logits readback、host top-10 scan、hash/runtime contract検証を含む。vLLM側はlogprobsなしの`LLM.generate` wallで、`RequestOutput.metrics`は取得不能だった。この比較は同一token workloadの診断値であり、production engine throughput比較には昇格しない。
- 実行後はKFD process 0、R9700 VRAM `87,384,064 bytes`へ復帰した。
- `validate-sq8-generation-benchmark.py`はsource、promotion SHA、R9700/profile/guards、全sample、timing包含、linear percentile、ratio-of-sums、flagsを再計算する。benchmark/generation/P6 validatorの合計129 testsが合格した。
- `benchmarks/results/2026-07-10/sq8-generation-v0.1/SHA256SUMS`は証跡10ファイルを全て再検証した。

### Additional commits

- `7ac8c59`: independent SQ8 generation validator
- `ba39014`: vLLM generation throughput benchmark
- `f91f332`: vLLM EOS/detokenize semantics alignment
- `fe45259`: audited steady SQ8 generation benchmark
- `63f5b0c`: benchmark device/profile/promotion evidence binding
- `fb05503`: independent SQ8 benchmark validator
- `2bf3d16`: P6/P7 real generation、steady benchmark、environment、SHA256SUMS evidence

## 次の行動

1. P7の最低限実生成は完了とし、この範囲を追加最適化で延長しない。
2. 次の開発単位を、監査readbackを外したlean generation path、tokenizer/API統合、複数request batchingから選び、別計画として開始する。
3. 今回の`23.701058 generated token/s`は監査経路の診断値として保持し、production throughput基準には使わない。
