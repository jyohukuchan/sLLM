# AQ Runtime Decode TPS Fix

## 前回の要点

- pre-SQ TPS測定で `512/256` の長時間runは完走したが、decodeはR9700/V620とも約 `0.14 tok/s` だった。
- その後の内訳確認で、主因はGPUのAQ/FP32演算そのものではなく、CPU/chunked `lm_head` top-kとruntime/smoke経路の固定オーバーヘッドだと分かった。

## 今回の変更点

- `package-token-ids-generate-smoke` / `package-token-ids-bench` に `gpu_resident_f32` lm_head modeを追加した。
- timing JSONにprefill/decode/layer/lm_headの内訳を追加した。
- 線形注意decodeで、gate/betaをGPU常駐化し、value-headごとのGPU RMSNormループをホスト計算へ置き換えた。
- runtimeの純粋なHIPカーネルを、起動直後に同期しない形へ変更した。
- `matvec_f32` にrocBLAS SGEMVの動的ロード経路を追加した。失敗時は既存HIPカーネルへfallbackする。
- incremental self-attn decodeから、smoke-onlyのq/k/RoPE/attention期待値再計算を外した軽量prepare pathへ切り替えた。

## 結果

- R9700/RDNA4 `prompt=16 generated=8 gpu_resident_f32`:
  - verified: true
  - decode: `5.254 tok/s`
  - prefill: `4.712 tok/s`
  - decode p50: `191.185 ms`
  - layers p50: `182.609 ms`
  - lm_head p50: `7.175 ms`
- V620/RDNA2 `prompt=1 generated=2 gpu_resident_f32`:
  - verified: true
  - decode: `3.725 tok/s`
  - prefill/decode疎通は成功

## 次の行動

- R9700のFP32短時間目標 `5 tok/s` は達成済みとして扱える。
- ただしR9700で期待される `15-20 tok/s` にはまだ遠い。次の主対象は、全layerで繰り返されるGEMVとhost/runtime orchestrationの削減。
- V620は、現段階では長いdecode測定を繰り返さず、RDNA2疎通確認の位置づけに留める。

## 追記: Warmupを考慮した再測定

## 前回の要点

- `prompt=16 generated=8` ではR9700で `5.254 tok/s` まで改善したが、期待値の `15-20 tok/s` には遠かった。
- ユーザーから、GPU warmupの影響でtoken/sが低く見える場合があるため、計測時に注意するよう指摘があった。

## 今回の変更点

- decode JSONに `step_wall_summary` を追加し、全ステップTPS、1/2ステップ除外TPS、後半4/8ステップTPS、min/p50/maxを出すようにした。
- `prompt=16 generated=16` で短い再測定を行い、warmup除外後のTPSを確認した。
- 直近比較で遅かったため、`matvec_f32` のrocBLAS SGEMVはデフォルト無効にし、`ULLM_ENABLE_ROCBLAS_MATVEC=1` または `ULLM_REQUIRE_ROCBLAS_MATVEC=1` の明示指定時だけ使う方針にした。

## 結果

- R9700/RDNA4 `prompt=16 generated=16 gpu_resident_f32`:
  - verified: true
  - all-step decode: `5.220 tok/s`
  - warmup skip-1: `5.210 tok/s`
  - warmup skip-2: `5.199 tok/s`
  - last-4: `5.136 tok/s`
  - step p50: `191.523 ms`
  - first step: `186.422 ms`
  - last step: `194.852 ms`
- 初回付近のステップが最速側なので、今回の低TPSはGPU warmupでは説明できない。
- 平均でlinear-attn層だけが約 `154.0 ms/token` を消費しており、`20 tok/s` に必要な `50 ms/token` を単独で超えている。

## 次の行動

- 現方式の小修正だけでは `20 tok/s` 到達は難しい可能性が高い。
- 続けるなら、次はlinear-attnの複数GEMV統合、GPU常駐の層間受け渡し、host readback削減のどれを本線にするか決める。

## 追記: Direct AQ4 Matvec Prototype

## 前回の要点

- materialized FP32経路ではAQ本来のon-the-fly dequant性能を測れていなかった。
- 低遅延なdequantを正しく実装できるかが、uLLM固有の本質問題だと整理した。

## 今回の変更点

- runtimeに `aq4_matvec_f32` を追加した。
- packed 4-bit index、u8 scale index、codebook、scale table、row-scale overrideを別bufferとしてGPU常駐させ、matvec中に統合する形にした。
- kernelはgroup単位でraw sumを作り、最後にscale/tensor_scaleを掛ける形にした。
- `package-aq4-matvec-smoke` を追加し、materialized FP32 matvecとの差分と速度を同時に出すようにした。
- linear-attn resident decodeの量子化projectionを、materialized FP32 matrixではなくdirect AQ4 matvecへ置き換えた。

## 結果

- 単体matvec:
  - layer0 qkv: `1.754x`、max abs diff `4.77e-7`
  - layer0 mlp.up: `1.687x`、max abs diff `1.34e-7`
  - layer0 mlp.down: `1.498x`、max abs diff `2.38e-7`
  - layer6 out(row-scale overrideあり): `0.873x`、max abs diff `2.38e-7`
- decode `prompt=16 generated=16`:
  - verified: true
  - generated tokens: materialized FP32 gen16と一致
  - all-step decode: `5.517 tok/s`
  - warmup skip-1: `5.508 tok/s`
  - warmup skip-2: `5.498 tok/s`
  - last-4: `5.465 tok/s`
  - step p50: `182.178 ms`
  - linear-attn mean: `143.32 ms/token`
  - linear-attn p50 sum: `143.28 ms/token`

## 次の行動

- direct AQ4 matvecの品質は壊れていないが、1:1置換だけでは `20 tok/s` には届かない。
- 次に本当に効くのは、qkv/a/b/zやMLP gate/upのprojection統合、host readback削減、linear-attn block全体のGPU常駐化。
- 現時点での壁はAQ formatの表現力ではなく、現行decode実行形のkernel粒度とhost介在。

## 追記: AQ4 fused decode follow-up

## 前回の要点

- direct AQ4 matvecは、raw valueとscale tableを分けて扱う実行方向が正しいことを確認した。
- ただし、projectionを1つずつAQ4 matvecへ置き換えるだけでは `5.517 tok/s` で、R9700に期待する `15-20 tok/s` には遠かった。

## 今回の変更点

- MLPの `gate` / `up` AQ4 matvecとSiLU-mulを1つのruntime kernelへ畳み込んだ。
- linear-attnの `a` / `b` AQ4 matvecとgate/beta変換を1つのruntime kernelへ畳み込んだ。
- qkv projection後にhostへ戻していたconv history更新、depthwise conv、SiLU、q/k/v split、q/k L2正規化をGPU runtime buffer上で処理する `linear_attn_qkv_prepare_f32` を追加した。
- linear-attn resident decodeからqkvのGPU->host->GPU往復を削った。

## 結果

- R9700/RDNA4 `prompt=16 generated=16 gpu_resident_f32`:
  - verified: true
  - generated tokens: direct AQ4 gen16と一致
  - all-step decode: `5.736 tok/s`
  - warmup skip-1: `5.727 tok/s`
  - warmup skip-2: `5.714 tok/s`
  - last-4: `5.667 tok/s`
  - step p50: `174.999 ms`
  - layers mean: `166.244 ms/token`
- direct AQ4 matvec比:
  - all-step decode: `5.517 -> 5.736 tok/s`
  - p50 step: `182.178 -> 174.999 ms`
  - layers mean: `172.807 -> 166.244 ms/token`

## 次の行動

- 今回の改善は有効だが、まだ `20 tok/s` の `50 ms/token` には届かない。
- 残る主対象は、層全体のGPU常駐化、attention normのGPU化、層間readback削減、qkv/zなどのより大きいprojection/workflow fusion。
- 小さい個別fusionだけを続けても改善幅は逓減し始めている。

## 追記: attention RMSNorm GPU化

## 前回の要点

- qkv prepareまでGPU化し、R9700 `prompt=16 generated=16` は `5.736 tok/s` まで上がった。
- まだrecurrent outputをhostへ戻してvalue-headごとのRMSNormを行い、再度GPUへ戻す経路が残っていた。

## 今回の変更点

- runtimeに `segmented_rmsnorm_f32` を追加した。
- linear-attn resident decodeで、recurrent output後のvalue-head RMSNormをGPU常駐buffer上で実行するようにした。
- attention norm weightもGPU bufferとして保持するようにした。

## 結果

- R9700/RDNA4 `prompt=16 generated=16 gpu_resident_f32`:
  - verified: true
  - generated tokens: direct AQ4 gen16と一致
  - all-step decode: `5.785 tok/s`
  - warmup skip-1: `5.774 tok/s`
  - warmup skip-2: `5.761 tok/s`
  - last-4: `5.706 tok/s`
  - step p50: `173.191 ms`
  - layers mean: `164.706 ms/token`
- direct AQ4 matvec比:
  - all-step decode: `5.517 -> 5.785 tok/s`
  - p50 step: `182.178 -> 173.191 ms`
  - layers mean: `172.807 -> 164.706 ms/token`

## 次の行動

- host readback削減は引き続き効いているが、単発の小kernel追加による改善幅は小さくなっている。
- 次は層間のGPU常駐化、linear-attn blockのより大きなworkflow fusion、またはqkv/z/out/MLPをまとめた大きいprojection設計へ移るほうがいい。

## 追記: recurrent decode fast path

## 前回の要点

- segmented RMSNorm後もR9700 `prompt=16 generated=16` は `5.785 tok/s` で、短文目標の
  `15-20 tok/s` には遠かった。
- projectionのAQ4 matvec単体は、qkv `0.215 ms`、MLP down `0.247 ms` 程度で、1層
  `~5.6 ms` の主因を単独では説明できなかった。

## 今回の変更点

- consecutive linear-attn layerをdecode中だけdevice bufferで接続した。
  - `5.785 -> 5.826 tok/s` と改善は小さく、層間readbackは主因ではなかった。
- `linear_attn_recurrent_f32` のHIP kernelがvalue headごとに1 threadしか使っていないことを確認した。
- `sequence_len == 1` のdecode専用fast pathを追加し、`(value_head, value_dim)` ごとに1 block、
  `key_dim` 方向を256 threadでreduceするようにした。
- `sequence_len > 1` は既存の直列pathを残し、HIP seq1/seq3 testを通した。

## 結果

- R9700/RDNA4 `prompt=16 generated=16 gpu_resident_f32`:
  - verified: true
  - generated tokens: direct AQ4 gen16と一致
  - all-step decode: `15.800 tok/s`
  - warmup skip-1: `15.752 tok/s`
  - warmup skip-2: `15.704 tok/s`
  - last-4: `15.335 tok/s`
  - step p50: `63.518 ms`
  - layers mean: `55.149 ms/token`
- 長めdecode:
  - `prompt=16 generated=128`: all-step `13.650 tok/s`, last-4 `12.263 tok/s`
  - `prompt=16 generated=256`: all-step `11.792 tok/s`, last-4 `9.153 tok/s`
- gen128/gen256はNaN/Infなしでverified。ただしsynthetic `len:16` promptなので、意味品質の評価ではなく
  smoke品質確認として扱う。

## 次の行動

- 短文decodeではR9700の下限目標 `15 tok/s` に到達した。
- 数百tokenではself-attnのcache長依存が支配的になり、後半は `~9 tok/s` まで落ちる。
- ここから先の主対象はAQ4 matvecではなく、古いf32 self-attn runtimeのhost readback/sync削減と
  paged decode attentionの長コンテキスト最適化。

## 追記: self-attn output-only fast step

## 前回の要点

- recurrent decode fast pathで短文は `15.800 tok/s` まで到達した。
- ただしgen256はall-step `11.792 tok/s`、last-4 `9.153 tok/s` で、self-attnのcache長依存が
  次の壁になっていた。

## 今回の変更点

- `PagedDecodeState` にdevice出力のまま戻るdecode stepを追加した。
- self-attn blockで、paged attention出力をhostに戻して再コピーする経路を避けた。
- self-attn block outputをGPU bufferのままpost RMSNorm/MLP residualへ渡すようにした。
- package token生成経路では、self-attn層のデバッグ用中間Vecを読まず、最終layer outputだけを読む
  `step_output_only` を使うようにした。

## 結果

- R9700/RDNA4 `prompt=16 generated=16`:
  - `15.800 -> 17.154 tok/s`
  - last-4: `15.335 -> 16.585 tok/s`
  - layers mean: `55.149 -> 50.134 ms/token`
- R9700/RDNA4 `prompt=16 generated=128`:
  - `13.650 -> 14.113 tok/s`
  - last-4: `12.263 -> 13.087 tok/s`
  - layers mean: `65.088 -> 62.408 ms/token`
- R9700/RDNA4 `prompt=16 generated=256`:
  - `11.792 -> 12.586 tok/s`
  - last-4: `9.153 -> 9.787 tok/s`
  - layers mean: `76.426 -> 71.237 ms/token`
- gen16/gen128/gen256はいずれもverified trueで、生成prefixも維持。
- 途中のgen16 1回目は `11.14 tok/s` の外れ値だったため採用しない。rerunでは `17.154 tok/s`。

## 次の行動

- output-only fast stepは採用できる。
- ただしgen256後半はまだ `~9.8 tok/s` なので、次はpaged decode attention kernel自体の
  cache長依存最適化、またはself-attn prepare/projectionのGPU常駐化が主対象。

## 追記: paged decode attention score reuse

## 前回の要点

- self-attn output-only fast stepで短文は `17.154 tok/s` まで上がった。
- ただしgen256はall-step `12.586 tok/s`、last-4 `9.787 tok/s` で、長いdecodeではまだ
  paged self-attentionのcache長依存が支配的だった。

## 今回の変更点

- `ullm_paged_decode_attn_f32_kernel` が同じq·k scoreをvalue次元ごとに再計算していたことを確認した。
- `head_dim <= 256 && value_dim <= 256` の場合に、query headごとに1 blockで起動するfast pathを追加した。
- q·kはsource timestepごとに1回だけ256 threadでreduceし、そのsoftmax weightをvalue laneで共有する。
- 範囲外のshapeでは既存のoutput-element並列pathに戻す。

## 結果

- R9700/RDNA4 `prompt=16 generated=16`:
  - `17.154 -> 20.306 tok/s`
  - last-4: `16.585 -> 20.367 tok/s`
  - p50 step: `58.138 -> 49.205 ms`
- R9700/RDNA4 `prompt=16 generated=128`:
  - `14.113 -> 20.103 tok/s`
  - last-4: `13.087 -> 19.880 tok/s`
  - p50 step: `69.499 -> 49.819 ms`
- R9700/RDNA4 `prompt=16 generated=256`:
  - `12.586 -> 19.710 tok/s`
  - last-4: `9.787 -> 19.143 tok/s`
  - p50 step: `77.418 -> 50.662 ms`
- 追加確認として `generated=512` も実行した。
  - all-step: `18.957 tok/s`
  - skip-2: `18.954 tok/s`
  - last-4: `17.858 tok/s`
  - last-8: `17.741 tok/s`
  - p50 step: `52.926 ms`
- gen16/gen128/gen256/gen512はいずれもverified trueで、既存accepted baselineと同じ生成prefixを維持した。

## 次の行動

- この時点でR9700のAQ4 prototype pathは、君が想定していた `15-20 tok/s` に到達した。
- 今回の遅さはAQ format/dequantそのものの壁ではなく、paged f32 attention kernelの重複計算が主因だった。
- ここから先は緊急のTPSデバッグではなく、実promptでのprefill/decode、出力品質、SQ候補との同条件比較に移るのが妥当。

## 追記: text prompt wrapperと実prompt計測

## 前回の要点

- `len:16` synthetic promptではR9700のdecode TPS目標に到達した。
- ただしsynthetic token列だけでは、出力品質が壊滅的に崩れていないかの観察として弱い。

## 今回の変更点

- `tools/run-package-token-prompt-bench.py` を追加した。
- Hugging Face tokenizerでpromptをtoken IDsへ変換し、既存の `package-token-ids-bench` を呼び出す。
- 生成token IDsを同じtokenizerでtextへ戻し、prompt text、tokenizer情報、decoded text、engine timingを1つのJSONへ保存する。
- `--target-prompt-tokens` で短いpromptを指定token数まで繰り返して切り出せるようにした。

## 結果

- plain text prompt, prompt_tokens=16, generated=128:
  - all-step: `20.102 tok/s`
  - skip-2: `20.099 tok/s`
  - last-8: `19.888 tok/s`
  - prefill: `17.238 tok/s`
  - decoded textは段落として始まるが、後半にcontrol-token風の反復が出る。
- chat template prompt, prompt_tokens=26, generated=128:
  - all-step: `20.026 tok/s`
  - skip-2: `20.023 tok/s`
  - last-8: `19.669 tok/s`
  - prefill: `19.163 tok/s`
  - 数値崩壊はないが、thinking-outline風の文体になる。
- repeated plain text, prompt_tokens=256, generated=256:
  - all-step: `18.413 tok/s`
  - skip-2: `18.409 tok/s`
  - last-8: `17.874 tok/s`
  - prefill: `22.189 tok/s`
  - prompt反復に引っ張られて生成も反復するが、途中にwarmup説明の文は出る。

## 次の行動

- 実promptでもTPSは `~18-20 tok/s` を維持した。
- 品質は「壊滅的崩壊ではない」が、greedy decode、stop-token policy、prompt/template policyが未整備なので、品質評価としてはまだ弱い。
- 次は小さい実promptセットとstop-token/template policyを入れて、SQ候補比較前の観察条件を安定させる。

## 追記: stop token policy

## 前回の要点

- plain text promptでは前半に自然な段落が出たが、後半にcontrol-token風の続きが出ていた。
- tokenizerで確認すると、切れ目になっていたtoken `248044` は `<|endoftext|>` だった。

## 今回の変更点

- `package-token-ids-generate-smoke` / `package-token-ids-bench` の末尾に任意の
  `STOP_TOKEN_IDS_CSV|none` を追加した。
- stop tokenが出た場合、指定された最大生成token数を待たずに早期終了する。
- JSONに `stop.token_ids`、`stop.stopped`、`stop.stopped_on_token_id`、`stop.reason` を出すようにした。
- wrapperには `--stop-token-ids`、`--stop-on-eos`、`--stop-on-special-tokens` を追加した。

## 結果

- plain text prompt, prompt_tokens=16, generated upper bound=128, `--stop-on-special-tokens`:
  - actual generated: `81`
  - stop reason: `stop_token`
  - stopped_on_token_id: `248044`
  - all-step: `20.232 tok/s`
  - skip-2: `20.232 tok/s`
  - last-8: `19.748 tok/s`
  - decoded textはcontrol-token風の続きに入る前に、GPU warmup説明の1段落として完結した。

## 次の行動

- 実prompt観察にはstop policyを使うべき。
- 次はprompt setを数本に増やし、temperatureなしgreedyだけでなく、少なくともstop条件つきの品質観察を標準化する。

## 追記: prompt suite runner

## 前回の要点

- 単発promptでは、実promptでのTPSとstop policyの有効性を確認した。
- ただし単発promptだけでは、品質観察としてもTPS観察としても選び方の偏りが大きい。

## 今回の変更点

- `tools/run-package-token-prompt-suite.py` を追加した。
- `benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.1.json` に4件の小さいpre-SQ prompt suiteを追加した。
- suite runnerは既存のtext prompt wrapperを順番に呼び、promptごとのJSON、`summary.json`、`summary.md` を出す。
- summaryにはprefill/decode TPS、stop reason、p50 step、unique generated token ratio、生成previewを残す。

## 結果

- R9700/RDNA4 suite, stop-on-special-tokens:
  - `warmup_measurement`: prompt `16`, generated `81`, decode `20.158 tok/s`, stop `stop_token`, verified true
  - `memory_vs_compute`: prompt `26`, generated `128`, decode `19.985 tok/s`, verified true
  - `throughput_checklist`: prompt `16`, generated `128`, decode `19.921 tok/s`, verified true
  - `long_prefill_warmup`: prompt `256`, generated `192`, decode `18.585 tok/s`, prefill `21.928 tok/s`, verified true
- suite summary:
  - mean decode TPS: `19.662`
  - min decode TPS: `18.585`
  - max decode TPS: `20.158`
  - verified all: true
- 短いtechnical promptは有限で局所的には読める英語を出した。
- `long_prefill_warmup` は意図的にpromptを反復しているため、timing用であり、意味品質の根拠にはしない。

## 次の行動

- このsuiteをSQ候補比較前の最小観察セットとして使える。
- 次の改善は、stopに届かないケースの生成上限/stop条件の調整、またはprompt suiteを日本語・コード・短いQAへ拡張すること。

## 追記: prompt suite output-health flags

## 前回の要点

- prompt suiteで4件の実promptを同条件で測定できるようになった。
- ただしsummaryだけでは、生成上限で切れたのか、反復が強いのか、prompt echoが出たのかがすぐ分からなかった。

## 今回の変更点

- `tools/run-package-token-prompt-suite.py` に `--summarize-existing` を追加した。
- 既存のper-case JSONを再利用して、summaryだけを再生成できるようにした。
- summary schemaを `package-token-prompt-suite-summary-v0.2` に上げた。
- case summaryに `output_status` と `output_warnings` を追加した。
- warningは以下を検出する。
  - `hit_generation_limit`
  - `low_unique_token_ratio`
  - `prompt_echo`
  - `control_marker_text`
  - `missing_terminal_punctuation`

## 結果

- 既存R9700 suiteを再実行せずsummaryだけ再生成した。
- output ok: `1 / 4`
- warning:
  - `memory_vs_compute`: `hit_generation_limit`, `missing_terminal_punctuation`
  - `throughput_checklist`: `hit_generation_limit`, `missing_terminal_punctuation`
  - `long_prefill_warmup`: `hit_generation_limit`, `low_unique_token_ratio`, `prompt_echo`, `missing_terminal_punctuation`
- `warmup_measurement` はstop tokenで止まり、warningなし。

## 次の行動

- 今後のSQ候補比較では、TPSだけでなく `output_status` と `output_warnings` を併記する。
- 次はstopに届かないpromptの生成上限を調整するか、QA/日本語/コードpromptを追加して観察範囲を広げる。

## 追記: prompt suite v0.2 category expansion

## 前回の要点

- prompt suite summaryにoutput-health flagsを入れた。
- 次の改善候補は、QA/日本語/コードpromptを追加して観察範囲を広げることだった。

## 今回の変更点

- suite runnerに `category` を追加し、summary JSONに `category_metrics` を出すようにした。
- `benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.2.json` を追加した。
- v0.2 suiteには、既存technical/checklist/timingに加えて以下を追加した。
  - `japanese_runtime_summary`
  - `python_stop_helper`
  - `short_qa_bandwidth`

## 結果

- R9700/RDNA4 v0.2 suite, stop-on-special-tokens:
  - case count: `7`
  - mean decode TPS: `19.858`
  - min decode TPS: `18.637`
  - max decode TPS: `20.325`
  - mean prefill TPS: `18.687`
  - verified all: true
  - output ok: `2 / 7`
  - output warn: `5 / 7`
- 新規case:
  - `japanese_runtime_summary`: decode `19.931 tok/s`, warning `hit_generation_limit`, `control_marker_text`, `missing_terminal_punctuation`
  - `python_stop_helper`: decode `20.008 tok/s`, stop tokenで終了、warningなし
  - `short_qa_bandwidth`: decode `20.060 tok/s`, first answer後に追加QAへ続き、warning `hit_generation_limit`, `missing_terminal_punctuation`

## 次の行動

- v0.2でもTPSは `~18-20 tok/s` を維持した。
- 品質面では、コードpromptは良い観察例になったが、日本語と短QAはprompt/template/stop条件の改善対象。
- 次に進めるなら、task-aware stop（例: `Question:` 再出現で止める、またはdecoded text stop substring）をwrapper側へ入れるのが有効。

## 追記: task-aware stop sequence

## 前回の要点

- v0.2 suiteでは、`short_qa_bandwidth` が最初の回答後に次の `Question:` へ続いていた。
- これはAQ4 runtimeの数値崩壊ではなく、token capだけに頼ったstop policyの問題だった。

## 今回の変更点

- Rustの `package-token-ids-generate-smoke` / `package-token-ids-bench` にmulti-token stop sequenceを追加した。
- stop sequenceは既存の `STOP_TOKEN_IDS_CSV|none` の次の位置引数として `SEQ1;SEQ2` 形式で渡す。
- JSONの `stop` に `token_sequences`、`stopped_on_token_sequence`、`reason=stop_sequence` を追加した。
- `tools/run-package-token-prompt-bench.py` に `--stop-token-sequences` と `--stop-text` を追加した。
- `tools/run-package-token-prompt-suite.py` にsuite共通 `--stop-text` とcase別 `stop_texts` を追加した。
- decoded reportには `generated_without_stop_sequence` を追加し、summaryは停止delimiterを除いた本文で品質warningを判定するようにした。
- `benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.2.json` の `short_qa_bandwidth` に `stop_texts: ["\nQuestion:"]` を追加した。

## 結果

- R9700/RDNA4 v0.2 suite再実行:
  - case count: `7`
  - mean decode TPS: `19.834`
  - min decode TPS: `18.616`
  - max decode TPS: `20.300`
  - mean prefill TPS: `18.893`
  - verified all: true
  - stopped count: `3 / 7`
  - output ok: `3 / 7`
  - output warn: `4 / 7`
- `short_qa_bandwidth`:
  - generated tokens: `35`
  - stop reason: `stop_sequence`
  - stopped_on_token_sequence: `[198, 14162, 25]`
  - decode: `20.146 tok/s`
  - warnings: none
  - 表示用生成文から末尾の `Question:` delimiterは除去された。

## 次の行動

- task-aware stopで短QAは比較用の観察ケースとして改善した。
- 日本語promptの `<think>` marker、英語promptのgeneration limit、timing用repeated promptのechoは残っている。
- 次に進めるなら、template/stop policyをさらに詰めるか、SQ候補比較に必要な出力観察条件としてこのv0.2を採用するかを決める。

## 追記: controlled prompt suite v0.3

## 前回の要点

- v0.2では短QAはtask-aware stopで改善した。
- ただし日本語promptの `<think>` marker、英語promptのgeneration limit、timing用repeated promptのechoが残っていた。
- これらはAQ runtimeの数値崩壊というより、prompt設計と品質評価母数の混在だった。

## 今回の変更点

- `tools/run-package-token-prompt-suite.py` に `output_health` を追加した。
  - `false` のcaseはTPSやverifiedを残すが、output warning評価からは外して `not_evaluated` と表示する。
  - summary schemaは `package-token-prompt-suite-summary-v0.3` にした。
  - JSONで `output_health` がboolean以外ならエラーにする。
- `benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.3.json` を追加した。
  - technical/checklist/japaneseは短い直接回答promptへ変更した。
  - 直接回答promptには `\n\n` stop_textを入れた。
  - short QAはv0.2の `\nQuestion:` stop_textを維持した。
  - long prefill timingは残しつつ `output_health: false` にした。

## 結果

- R9700/RDNA4 v0.3 suite:
  - case count: `7`
  - mean decode TPS: `19.796`
  - min decode TPS: `18.500`
  - max decode TPS: `20.166`
  - mean prefill TPS: `19.850`
  - verified all: true
  - stopped count: `6 / 7`
  - output ok: `6 / 7`
  - output warn: `0 / 7`
  - output not evaluated: `1 / 7`
- 品質評価対象6件:
  - `warmup_direct_answer`: `20.009 tok/s`, warningなし
  - `memory_vs_compute_direct`: `20.001 tok/s`, warningなし
  - `throughput_checklist_direct`: `19.938 tok/s`, warningなし
  - `japanese_direct_answer`: `19.975 tok/s`, `<think>` markerなし
  - `python_stop_helper`: `19.982 tok/s`, warningなし
  - `short_qa_bandwidth`: `20.166 tok/s`, warningなし
- timing probe:
  - `long_prefill_warmup_timing`: decode `18.500 tok/s`, `not_evaluated`

## 次の行動

- SQ候補比較の標準prompt suiteはv0.3を優先するのが良さそう。
- v0.2は荒いprompt stress用として残せる。
- 次の改善候補は、R9700だけでなくV620/RDNA2でv0.3を軽く通し、RDNA2でも同じ出力観察条件が崩れないかを確認すること。

## 追記: V620/RDNA2 v0.3確認

## 前回の要点

- R9700/RDNA4ではv0.3 controlled prompt suiteがmean decode `19.796 tok/s`、品質評価対象6件すべてokだった。
- 次の確認対象は、同じ観察条件がV620/RDNA2でも成立するかだった。

## 今回の変更点

- `benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.3.json` をV620/RDNA2 `device_index=1` で実行した。
- output dirは `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-v620-v0.3`。
- R9700と同じpackage、tokenizer、stop policy、output-health ruleを使用した。

## 結果

- V620/RDNA2 v0.3 suite:
  - case count: `7`
  - mean decode TPS: `15.434`
  - min decode TPS: `14.553`
  - max decode TPS: `15.668`
  - mean prefill TPS: `16.503`
  - verified all: true
  - stopped count: `6 / 7`
  - output ok: `6 / 7`
  - output warn: `0 / 7`
  - output not evaluated: `1 / 7`
- 品質評価対象6件:
  - `warmup_direct_answer`: `15.661 tok/s`, warningなし
  - `memory_vs_compute_direct`: `15.585 tok/s`, warningなし
  - `throughput_checklist_direct`: `15.549 tok/s`, warningなし
  - `japanese_direct_answer`: `15.592 tok/s`, `<think>` markerなし
  - `python_stop_helper`: `15.432 tok/s`, warningなし
  - `short_qa_bandwidth`: `15.668 tok/s`, warningなし
- timing probe:
  - `long_prefill_warmup_timing`: decode `14.553 tok/s`, `not_evaluated`

## 次の行動

- 現時点で、controlled pre-SQ prompt suiteについてはR9700/RDNA4とV620/RDNA2の両方で動くと言える。
- 次の改善候補は、発表用に「対象範囲・未対応範囲・既知の制約」を1ページでまとめること。
- その後にSQ format設計へ移るなら、v0.3をSQ候補比較の標準suiteとして扱う。

## 追記: AQ4 RDNA prototype status brief

## 前回の要点

- R9700/RDNA4とV620/RDNA2の両方でcontrolled v0.3 prompt suiteが完走した。
- R9700はmean decode `19.796 tok/s`、V620はmean decode `15.434 tok/s` だった。
- 品質評価対象6件は両方でwarningなしだった。

## 今回の変更点

- `docs/research/aq4-rdna-prototype-status-2026-07-06.md` を追加した。
- 外に出せるclaimを「single-request Qwen3.5-9B AQ4 runtime prototype」に限定した。
- 出せないclaimとして、SQ未実装、tensor parallelなし、batchingなし、server APIなし、final logits/generated-token reference guard未完了を明記した。
- SQ設計へ進む時のbaselineとして、R9700/V620 v0.3 suiteとreference guardを使う方針をまとめた。

## 結果

- 発表用には、以下の狭い言い方なら根拠がある。
  - RDNA4/RDNA2で動くlocal AQ4 prototype path
  - Qwen3.5-9B controlled prompt suite
  - R9700約 `20 tok/s`、V620約 `15 tok/s`
  - single-request greedy decode
- まだ避けるべき言い方も明確になった。
  - production inference
  - SQ実装済み
  - batching/tensor parallel対応
  - 出力品質の完全検証済み

## 次の行動

- 次はSQ format設計へ戻れる状態になった。
- その前にもう一段改善するなら、final logitsまたはgenerated-token reference guardをprompt suite側に足すのが妥当。

## 追記: cross-device generated-token guard

## 前回の要点

- AQ4 RDNA prototype status briefでは、generated-token reference guardが次の改善候補だった。
- R9700/RDNA4とV620/RDNA2のv0.3 suiteは、preview上は同じ内容を出していたが、token列として機械比較していなかった。

## 今回の変更点

- `tools/compare-package-token-prompt-suite.py` を追加した。
- 2つのprompt suite summaryを読み、caseごとに以下を比較する。
  - prompt token IDs
  - generated token IDs
  - stop reason / stop token / stop sequence
  - per-case `verified`
  - output status
- R9700/RDNA4 v0.3 summaryをreference、V620/RDNA2 v0.3 summaryをcandidateとしてguardを実行した。
- 結果artifact:
  - `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-token-guard.json`
  - `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-token-guard.md`

## 結果

- guard result: passed
- compared cases: `7`
- prompt token matches: `7`
- generated token matches: `7`
- stop matches: `7`
- both verified: `7`
- output status matches: `7`
- generated-token hash prefix:
  - `japanese_direct_answer`: `f9ba6ba327e9e382`
  - `long_prefill_warmup_timing`: `6be141962dcb081a`
  - `memory_vs_compute_direct`: `3876cd8b6a716ab0`
  - `python_stop_helper`: `37c03e076f19178d`
  - `short_qa_bandwidth`: `d30b2709cb879de4`
  - `throughput_checklist_direct`: `c855ce236defcb73`
  - `warmup_direct_answer`: `1775ce6734cb10c0`

## 次の行動

- cross-device generated-token guardは追加済みになった。
- まだ残る正しさgateは、CPUまたは外部referenceに対するfinal-logits guard。
- SQ候補比較時は、v0.3 suite summaryに加えてこのgenerated-token guardも通す。

## 追記: short-QA final top-logits guard

## 前回の要点

- cross-device generated-token guardはv0.3全7件でpassedになった。
- 残りの正しさgateは、CPUまたは外部referenceに対するfinal-logits guardだった。

## 今回の変更点

- `package-token-ids-logits-smoke` をv0.3 `short_qa_bandwidth` prompt tokensで実行した。
- R9700/RDNA4とV620/RDNA2のlogits JSONを保存した。
- `tools/compare-package-token-logits.py` を追加した。
  - prompt token IDs
  - top-k token IDs
  - top-k logit値
  - per-report `verified`
  を比較する。
- CPU backendで同じfull-model logits smokeも試したが、2分半以上かかって未完走だったため中止した。routine guardとしてはまだ重すぎる。

## 結果

- short QA final top-8 logits guard:
  - prompt tokens: `25`
  - top-k: `8`
  - R9700 total wall: `49479.525 ms`
  - V620 total wall: `49945.966 ms`
  - top token IDs match: true
  - max abs logit diff: `0.0`
  - both verified: true
  - passed: true
- top tokens:
  - rank0 token `79612`, logit `8.736091614`
  - rank1 token `8938`, logit `8.176540375`
  - rank2 token `1473`, logit `7.934105396`
  - rank3 token `271`, logit `7.861293793`

## 次の行動

- GPU間のfinal top-logits guardは1件追加できた。
- ただしCPU/external reference guardはまだ未完了。今後は全層CPU再計算ではなく、短いlayer subsetや既存golden fixtureからfinal logitsに近い軽量referenceを作る方向が良い。

## 追記: suite-wide top-logits guard

## 前回の要点

- short QA 1件では、R9700/V620間のfinal top-8 logitsが完全一致した。
- ただしv0.3 suite全caseのlogits比較はまだgenerated-token guardに含まれていなかった。

## 今回の変更点

- `tools/compare-package-token-prompt-suite.py` を拡張した。
- 既存のprompt/generated token、stop、verified、output status比較に加えて、各caseの以下も比較するようにした。
  - `prefill.top_logits`
  - `decode.last_top_logits`
- `--logit-atol` を追加し、既定値は `1e-6` にした。
- R9700/RDNA4とV620/RDNA2のv0.3 guard artifactを再生成した。

## 結果

- suite-wide token/logits guard:
  - compared cases: `7`
  - generated token matches: `7`
  - top logits matches: `7`
  - max prefill top-logit abs diff: `0.0`
  - max decode last top-logit abs diff: `0.0`
  - passed: true
- 追加ベンチなしで、既存v0.3 reportから全caseのtop-logits一致を確認できるようになった。

## 次の行動

- v0.3 suite guardは、generated-tokenだけでなくtop-logitsも見る形になった。
- まだ独立CPU/external referenceではないが、RDNA2/RDNA4間のdeterminism確認としては十分強くなった。

## 追記: guard tool regression tests

## 前回の要点

- `compare-package-token-prompt-suite.py` はv0.3 suite全caseのgenerated tokenとtop logitsを比較できるようになった。
- `compare-package-token-logits.py` は単独logits smoke JSONのtop-k比較をできるようになった。

## 今回の変更点

- `tests/test_compare_package_guards.py` を追加した。
- temp dir上の最小JSON fixtureで、以下をunittestするようにした。
  - prompt suite guardがmatching token/logitsでpassする。
  - prompt suite guardがlogit mismatchでfailする。
  - logits guardがmatching top logitsでpassする。
  - logits guardがtop token mismatchでfailする。

## 結果

- `python3 -m unittest tests/test_compare_package_guards.py`: `4` tests passed.
- `python3 -m py_compile tools/compare-package-token-prompt-suite.py tools/compare-package-token-logits.py tests/test_compare_package_guards.py`: passed.
- 実artifactのsuite guard/logits guard再実行もpassed。

## 次の行動

- guard tool自体の最低限の回帰テストが入った。
- 次の改善候補は、SQ候補比較時にこのguard一式をまとめて実行するchecklistまたはdriverを作ること。

## 追記: guard bundle driver

## 前回の要点

- prompt suite token/logits guardとstandalone logits guardは別々に実行できた。
- SQ候補比較では、実行漏れを避けるために同じguard setをまとめて走らせる入口が欲しかった。

## 今回の変更点

- `tools/run-package-prompt-guard-bundle.py` を追加した。
- 入力:
  - reference summary
  - candidate summary
  - 任意のreference/candidate logits JSON
- 出力:
  - prompt suite token/logits guard JSON/Markdown
  - standalone logits guard JSON/Markdown
  - bundle summary JSON/Markdown
- R9700/RDNA4 vs V620/RDNA2 v0.3 artifactでbundleを実行した。

## 結果

- bundle summary:
  - passed: true
  - prompt suite token/logits: passed
  - standalone short-QA logits: passed
- artifact:
  - `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/guard-bundle-summary.json`
  - `benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/guard-bundle-summary.md`

## 次の行動

- SQ候補比較時は、このbundle driverを標準のcorrectness/output guard入口として使える。
- 次の改善候補は、SQ format設計入力docへv0.3 suiteとguard bundleを正式な比較gateとして追記すること。

## 追記: SQ design input gate update

## 前回の要点

- v0.3 prompt suite、suite-wide token/logits guard、standalone short-QA logits guard、guard bundle driverが揃った。
- 次の改善候補は、SQ format設計入力docへそれらを正式な比較gateとして反映することだった。

## 今回の変更点

- `docs/plans/sq-format-design-input-v0.1.md` を更新した。
- 現行AQ4 prototype baselineとして以下を追加した。
  - R9700/RDNA4 v0.3 mean decode `19.796 tok/s`
  - V620/RDNA2 v0.3 mean decode `15.434 tok/s`
  - output ok/warn/not evaluated: `6 / 0 / 1`
  - guard bundle passed
- SQ candidate acceptance gateを追加した。
  - compact resident bytes
  - materialized working-set bytes
  - materialization granularity/wall time
  - v0.3 suite summary
  - guard bundle summary
  - short golden prefix guard
  - row-scale override policy changes
- Minimum pass criteriaを追加した。

## 結果

- SQ候補は、TPSだけでは比較対象として扱わない方針になった。
- first sq candidateは、TPSがAQ4と同等でも、compact residency / bounded working setを大きく改善し、同じguard bundleを通すなら有用と扱える。
- 独立CPU/external final-logits referenceはdeferredのまま明記した。

## 次の行動

- 次はSQ format v0.1の具体設計に進める。
- 実装前にもう一段整理するなら、SQ run record schema案を作るのが良い。

## 追記: SQ candidate runtime result schema

## 前回の要点

- SQ design inputにv0.3 suiteとguard bundleを正式gateとして追記した。
- 次の改善候補は、SQ候補比較時のrun record schema案を作ることだった。

## 今回の変更点

- `docs/specs/sq-candidate-runtime-result-v0.1.md` を追加した。
- SQ候補run rowの必須項目を定義した。
  - candidate
  - model
  - hardware
  - workload
  - storage
  - timing
  - quality
  - guards
  - artifacts
  - baseline
  - decision
- `docs/plans/sq-format-design-input-v0.1.md` から新schemaを参照するようにした。

## 結果

- SQ候補は、単なるTPS表ではなく、compact resident bytes、materialized working-set bytes、materialization time、v0.3 suite summary、guard bundle resultを同じrowに残す方針になった。
- `decision.comparable_to_baseline` と `decision.accepted_for_next_iteration` を分け、比較可能性と採用判断を混同しない形にした。

## 次の行動

- 次はこのschemaに沿ったAQ4 baseline rowを作ると、SQ候補が出た時の比較基準がさらに明確になる。
