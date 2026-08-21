# Phase 35: long-context Full Attention・GDN構造最適化

> 状態: 完了（両track共通shape限定採用）
> 実施日: 2026-08-20

## 結論

Phase 34後の10,001-token profileで残った二大device familyを、exact gfx1030/gfx1201の共通sourceで改善した。
Full Attentionは4 query rowが同じKV headを共有するQ_TILE=4 provider、GDNは1,024 workgroup相当のcolumn-owned
recurrent-state pipelineを採用した。いずれも短い入力・decode・scope外shapeを既存providerへ残す静的routeである。

担当AI裁量では、両targetでoperatorとfull-modelの利益が一貫し、二候補ともN1、rollbackが明示的で、public ABI、
vAttention、KV encoding、state transaction、projection、arenaを維持できたため採用が妥当である。固定改善率は使用していない。

## Fresh baselineとroute

fixed Qwen3.5-4B BF16 GGUF、FP16 KV、messages、thinking disabled、exact 10,001 input / greedy 2 outputを固定した。
Phase 34 baselineはV620 `34.860543559 s`、R9700 `75.348556986 s`、生成tokenは`[2064,5686]`だった。

- Attention: exact gfx1030/gfx1201、Q heads 16、KV heads 4、head dim 256、`M>=128`。
  logical symbolは`causal_attention.prefill.gqa4_qtile4.v7`。`M<=127`、decode、別shape/targetはPhase 33以前を使う。
- GDN: exact gfx1030/gfx1201、Q/K heads 16、value heads 32、head dim 128、token count 128以上。
  logical symbolは`linear_attention.gdn.column_state.v2`。短prefill/decodeはPhase 28/29を使う。
- `SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`と`SLLM_GDN_FORCE_BASELINE=1`は同一binary比較とrollback診断用で、
  public APIでもruntime failure後のsilent retryでもない。

## Full Attention

一つの256-thread workgroupが4 query row × GQA 4 headを所有する。8 waveが二つずつlogical queryを担当し、K/Vを
一度だけdirect decodeして16 queryへ共有する。各queryは独立したcausal key上限、online maximum、denominator、weighted Vを持つ。
global score/partial scratch、追加dispatch、KV mirrorは作らない。

V620のoperator screenではQ_TILE=4をM=64/65へ使うと約11%悪化した一方、M=127/128/129で約18.5〜20.4%、
M=255/256/257で約42.2〜42.5%改善した。R9700ではM=128/129が33.9/35.2%、M=255/256/257が
54.0〜59.2%改善したため、共通境界を128に固定した。

両target × FP16/dynamic FP8/static FP8/NVFP4 × 29 case、計232 caseをfinal sourceでPASSした。最大絶対誤差は
FP16 `2.3841858e-7`、FP8 `4.7683716e-7`、NVFP4 `1.1641532e-9`、fallback/cleanup 0だった。
同じ256 QK項、FP32 accumulator、key順softmax、BF16 RNE stageを維持し、固定treeの依存深さがPhase 33 C2の概ね8段を
超えないためN1とした。

Attention-only 10,001/2はV620 `34.861→28.130 s`（19.31%）、R9700 `75.349→68.879 s`（8.59%）だった。
V620 final profileのFull Attentionは`10.819552→4.109614 s`、62.02%短縮した。

## GDN

preprocessはtoken × Q/K headの128-thread blockでQ/K normを一度だけ計算し、既存BF16 round stageをconvolution scratchへ
書き戻す。同時にvalue headごとのBF16 betaとFP32 decayを二つのchecked FP32 planeへ生成する。recurrent kernelは
wave32 × 4、grid 1,024 workgroupで各waveが一つのstate/output columnを所有し、lane当たり4 state rowをFP32 registerへ
保持して全tokenを進める。postprocessはraw BF16 projectionから既存output RMSNorm、norm weight、z SiLUを適用する。

stateのtarget別物理index mapping、previous/next slot、accepted-prefix publication、conv stateは変更せず、provider間で
state migrationを必要としない。10,001 tokenの追加scratchはbeta/decay計2,560,256 byte/layer、24 layer合計
61,446,144 byteで、既存linear-attention stateのchecked request-owned scratchを拡張する。1 layer当たりdispatchは2から4、
full modelは984から1,032へ増えたが、Phase 31 workspace arenaは5,278,049,280 byteのままだった。

両targetでtoken 1/3/17/127/128/129を独立oracleへ照合し12/12 PASSした。最大絶対/相対誤差は
`0.00390625`/`0.014705882`、next state一致、fallback/cleanup 0だった。`S^T k`/`S^T q`は同じ128 FP32項を
逐次依存127から4項local + wave treeの概ね8段へ変えるためN1とした。Q/K、beta、raw output、normalized outputの
BF16 round stage、decay/state update式は維持する。

GDN-only 10,001/2はV620 `34.861→27.945 s`（19.84%）、R9700 `75.349→69.949 s`（7.17%）だった。
V620 profileのGDN familyは約`7.67157→0.6176 s`、91.95%短縮し、fixed llama.cppの0.622秒と概ね同等になった。

## Combinedとpeer残差

final sourceのcombined 10,001/2は次のとおり。

| target | baseline | final | 短縮 | audit |
| --- | ---: | ---: | ---: | --- |
| V620 gfx1030 | 34.861 s | 22.683 s | 34.93% | `[2064,5686]`、HIP-only、fallback false、cleanup 0 |
| R9700 gfx1201 | 75.349 s | 65.214 s | 13.45% | `[2064,5686]`、HIP-only、fallback false、cleanup 0 |

V620 profileではprojection 11.627秒でPhase 34の11.683秒を維持した。fixed llama.cpp E1 peerはprojection 12.772秒、
Full Attention 0.462秒、GDN 0.622秒である。Phase 35後はprojectionがpeerの0.91倍、GDNが0.99倍、Full Attentionが
8.90倍となった。したがってGDN gapは実質解消し、残る主要最適化余地はFull Attention 4.11秒である。

R9700 rocprofには既存のtoken-by-token MTP target laneが含まれ、decode系kernel callが多い。Phase 35のprefill候補時間へ
このresidualを混ぜず、MTP laneは別の既存診断課題として扱う。

## Provenance

AttentionはsLLM既存Phase 33 sourceと独立oracleから実装し、新しい外部source expressionを再利用していない。
GDNはfixed llama.cpp `f5919bf458ef190468b5c329bb293f8a54a1e69c`の`gated_delta_net.cu`にあるcolumn ownership、
register state shard、wave reductionの近接構造をbounded adaptationした。既存layout-only noticeを上書きせず、
`llama-cpp-phase35-gdn-column-state-001`を追加した。import commitは
`bca482251bd21b144d950956af39a769c4211417`、導入時local file SHA-256は
`cf8e8aafa5e7e64c8fe5bc082912b5b8a328d0a9ed407965d6782cad72b3bc4a`へ固定した。

## Verification

- `cargo +1.97.1 test --locked --offline -p sllm-core -p sllm-hip`: PASS（core 185、HIP lib 96、関連bin testを含む）。
- `cargo +1.97.1 test --locked --offline -p sllm-server`: PASS（lib/bin/http contract計32）。
- `cargo +1.97.1 fmt --all --check`、`git diff --check`: PASS。
- exact gfx1030/gfx1201 release build、gfx942 ROCm 7.14 / Code Object V6 / wave64 compile-only: PASS。
- gfx1201 binaryをV620へload: exit 2、`requested device gcnArchName does not match exactly`でfail-closed。
- final Full Attention 232 case、GDN 12 case、両target combined full model: HIP-only、fallback false、cleanup 0。
- final gfx1030 serverで10,001-token OpenAI non-stream/SSEを実行し、どちらも`It`、usage 10,001+1、SSE `[DONE]`、
  HIP-only、fallback 0だった。graceful shutdown後のcurrent/request-state/workspace byte、retryable/durable cleanupは0。
- cumulative integration review: blockerなし。threshold、metadata/actual symbol、state layout、scratch overflow、transaction、
  numerical classification、provenance、compatibilityを再確認した。

OpenAI transport、default FP16 KV、vAttention、GGUF/model lock、public ABIは変更していないため、service確認は代表gfx1030の
non-stream/SSE lifecycleへ限定した。raw trace/DB、model、binary、full token/input列はGit追跡対象外である。

## 限界と次の探索条件

結果はROCm 7.14、canonical exact V620/R9700、fixed Qwen3.5-4B BF16 long-context shapeに限定する。別SKU/model/head shape、
別ROCm、concurrent throughputへ一般化しない。GDNはpeer parity後のため追加scan/layout候補の優先度を下げる。
Full Attentionはなおpeerの約8.9倍であり、次のwork unitは残るbarrier、vector FP32 QK/PV、query/K tile、matrix innerを
fresh profileで分離し、Phase 35のQ_TILE=4を新baselineとして比較する。

[対応するarchive plan](../../../../plans/archive/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)
[bounded summary](../../../../../ci/matrix/phase35-attention-gdn-summary-v1.json)
[数値・出力影響変更台帳](../../../../compatibility/numerical-output-changes.md)
[provenance](../../../../provenance/README.md)
[メイン計画](../../../../plans/main-plan.md)
