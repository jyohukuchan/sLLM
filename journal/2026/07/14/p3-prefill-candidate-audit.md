# P3 prefill candidate audit (2026-07-14)

## 前回の要点

- P2のCPU validatorは有効だが、P2のidentity/policy/path-oracle bundleと、同一caseへ束縛されたproduction traceはまだ揃っていない。
- P1 live-v4は `production_server` の診断証跡として有効だが、P2 case と `case_id`/`case_sha256` で結ばれていないため、candidate promotionのbaselineではない。
- サービスとGPUは変更せず、P2のR9700測定前にP3候補の所有範囲と計測条件を監査した。

## 読み取り根拠

- `docs/plans/aq4-production-prefill-decode-optimization-plan-v0.1.md` の候補順、P3 lane所有権、CPU/source → HIP component M grid → full model offline → direct worker → production server の昇格順を確認。
- `crates/ullm-engine/src/qwen35_aq4_model_runtime.rs` は native prefill width を `2..=128` に制限し、model-wide linear/self-attention sequence workspace と ping/pongを確保して各layerを順に実行する。長いpromptはこの幅へchunk分割される。
- `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs` のlinear sequence pathは、AQ4 qkv/z/a/b projection、`LinearAttentionQkvPrepareBatch`、gate/beta、`GatedDeltaRuleSequence`、postprocess/O projection/MLPを実行する。self-attention sequence pathはQ/K/V projection、Q/K norm+RoPE、paged KV chunk write/read、O projection、MLPを実行する。
- `crates/ullm-engine/src/backend_operation_registry.rs` と runtime API/sourceは、各operationについてtyped geometry、feature guard、workspace admission、HIP/CPU pathを持つ。BM8 registerはgfx1201、group16、rows divisible by32、cols divisible by128、batch>=8の制約を持つ。
- P1 `bottleneck-audit-live-v4.json` は `cold_prefill` M=128 2037.990432 ms (2 observations) を記録したが、operation wall timeは0で、launch/workspaceだけを示す。`GatedDeltaRuleScan`/`LinearAttentionQkvPrepare` は各48 launches、self-attentionの `FusedQkNormRopePagedKvWrite`/`PagedCausalGqaRead` は各16 launches、後者のplanned workspaceは各約1.61 GBである。これは仮説でありP2受入れ基線ではない。

## 候補の暫定順位（P2 baseline前の仮説。最終決定ではない）

| 暫定順 | 候補 | 主な所有範囲（候補実装者） | CPU oracle | HIP component gate | full-model gate | direct worker / production gate | OOM・rollbackリスク | 判断 |
|---:|---|---|---|---|---|---|---|---|
| 1 | recurrent attention + dense self-attentionのchunk execution | P3-A: runtimeのlinear-attn QKV/recurrent、paged chunk kernels。P3-C: `qwen35_aq4_layer_runtime.rs` のsequence dispatch（ABI/registry/sessionは凍結後に統合） | M=1とM=2/8/16/32/64/128で、token/logit/hidden、conv history、recurrent state、KV/cache位置、chunk境界、cancel/resetをall-M=1と比較 | `LinearAttentionQkvPrepareBatch`、`GatedDeltaRuleSequence`、`PagedKvWrite`/`PagedCausalGqaRead` の各Mでshape/dtype/finite、workspace、launch/sync/fallbackを確認 | prompt 128/512/1011/1024/1339/2048/3584を同一chunk policyでcold/cached-prefix測定。p50/p95、実M、state transactionを比較 | 同じworker/manifest/package/identityでdirect resident workerを1 caseずつ実行し、production traceでAPI/SSEのpublish/commit/discard/reset/EOS/lengthを再検証 | recurrent stateとconv historyはrequest-owned。chunk失敗後poison/reset漏れが致命。self-attention chunkのKV write後reader失敗でrollbackが必要。共有workspace拡大はVRAMを圧迫 | P1でlinear 48 + self 16 launchesが見えており、native M=128の主経路に直接効く。ただしwall time未計測なのでP2で再順位付けする |
| 2 | AQ4 BM8/register kernelのshape coverage、tail、scale metadata residency | P3-A: `runtime/src/ullm_runtime_api_aq4.inc`、`runtime/src/ullm_runtime_hiprtc_sources.inc` と専用kernel test。P3-C: layer dispatch/typed registryのbinding | CPU AQ4 matvec batchをM=1/8/16/32/64/128でsource/path oracleと比較し、scale index、row scale、tail、非有限を検証 | gfx1201/group16のみ。rows%32=0、cols%128=0、M>=8。対応外shapeはcanonical/tiledへ明示fallbackし、予期しないstaging/D2Hを拒否 | 全projection（linear/self MLP、QKV等）のimplementation ID、resolved/actual width、fallback、p50/p95とlogit/hiddenを比較。BM8単体改善は昇格根拠にしない | worker identity/runtime feature guard、production traceのimplementation/actual width/peak memoryを同一run rootへ束縛。V620はcapability smokeのみでR9700 gateの代替にしない | register pathのshape外誤選択、scale metadataのdevice residency不備、tailで未初期化出力。新kernel失敗時はimmutable legacy/tiled pathへrollbackし、guardをfail-closed | 実装の境界が比較的明確で、projection回数が多いため有力。ただしP2でprojection wall timeとfallbackを得るまで確定しない |
| 3 | paged KV block-table validation、D2H、stream sync削減 | P3-A: paged chunk/read/write runtime kernels。P3-C: self-attention layerのblock-table admissionとtiming（共有registryは統合窓口） | block tableの範囲・cache position・KV stateをCPUで全M比較。D2H無しのmetadata契約とreset/rollbackを検証 | `PagedKvWrite`、`PagedCausalGqaRead`、`FusedQkNormRopePagedKvWrite` のM/contexts 16/128/512/1024/1339/2048/3584、sync/D2H/peakを測定 | decode/prefill両方でKV/logit/greedy token、short-context canonical threshold、chunk境界を比較。読出しと書込みを分けてprofile | direct workerでblock-table identity、actual implementation、workspace、resource observerを束縛。productionでcancel/publish failure後のKV破棄/resetを検証 | P1のplanned workspaceはwriter/reader各約1.61 GB。sync削減のためのmetadata常駐がVRAMを増やすとOOM。KV write後の失敗はrequest reset必須 | 重要な安全候補だが、P1はwall timeを測っていない。OOMリスクが高いためP2 memory/transfer証拠が揃うまで実装順位を上げない |
| 4 | projection/norm/activation/residualのlaunch削減・安全なfusion | P3-A: 専用HIP fused kernels。P3-C: layer runtimeでphase/shape binding。CPU oracleは個別primitiveと全layer差分を担当 | hidden/logit、各中間のfinite・shape、residual/norm/SiLUのstreaming差分。state/positionは変更しない | 各fusionの入力/出力buffer alias、M tail、dtype、workspace、launch/syncを確認。fallbackは明示的に記録 | 全モデルでlogit/greedy/top-k、短文脈decode、MLP/attention別wall time。componentのlaunch削減だけでは不採用 | direct worker/productionでtraceのoperator graph、fallback、reset、EOS、length、publishを検証 | fusion境界の数値差・alias破壊・workspace増加。失敗時は既存primitiveへ即時rollbackできるregistry bindingが必要 | 改善余地はあるが、影響範囲が広く数値/rollbackリスクが高い。P2でlaunch/syncが支配的と判定された場合のみ候補化 |
| 5 | embeddingまたはLM head専用改善 | P3-C: `qwen35_aq4_model_runtime.rs`/sessionのembedding・final norm・LM head接続。P3-Aはtop1/logit kernel限定 | embedding gather、hidden/logit chunk、top-k/top1をCPU/source oracleとstreaming比較（全vocabを常時保持しない） | embedding gather、final norm、AQ4 LM head/top1のwall time、D2H、workspaceを分離測定 | prefill TTFTとdecode ITL、full logits/top-k/greedy tokenを比較。vocab 248320のhead計測を独立して記録 | direct workerのgeneration state epoch、production traceのgraph tail、SSE token/stop/resetを確認 | LM headの巨大workspace/D2HでVRAMと遅延が増えやすい。top1最適化の数値差はrollback対象 | P1 graph tailには含まれるが、prefill 2.0 sの主因とは未確認。実測支配時のみ着手 |

### 選抜ルール

P2で全caseの同一identity baselineを取得するまで、候補1を含めて最終candidateを決めない。P2 raw traceでoperator別 wall time、実M、launch/sync、H2D/D2H、workspace、fallback、peak VRAMを再計算し、最大のprefill寄与がある一つを第一候補とする。候補を同時に育てるのは原則二つまでとする。

各候補は以下の順で同じcase identityを束縛する。

```text
CPU/source differential
  -> HIP component M grid
  -> full_model offline
  -> direct resident worker
  -> production_server smoke
```

どの段階でも、shape/dtype/finite、hidden/logit/top-k/greedy、KV・recurrent・conv・cache・position・chunk、cancel/publish failure/EOS/length/reset、OOM/fallback/workspace、p50/p95非回帰が合格しない候補は停止する。componentの改善だけでproductionへ昇格しない。

## 残課題

1. P2 R9700 baselineが未実施で、P1のoperation wall timeは未観測。P2完了前に候補を実装・activationしない。
2. P2 caseにidentity/policy/path-oracle/production trace/resource observerをhash-boundで揃える。
3. P2のrawから候補別にM/contextsごとのwall-timeとmemoryを再順位付けする。特にwriter/readerの1.61 GB workspaceが実際にprefillかdecodeかをtraceで切り分ける。
4. CPU oracleはsource/path/state transactionをstreamingで比較し、full logits matrixを保持しない。direct workerはactive HIP gfx1201 identityのためCPU product caseをunsupportedとして扱う。

## 次の行動

- 親エージェントがP2 baseline/profileを完了した後、上表の選抜ルールで第一候補を確定する。
- 第一候補だけをP3-A/Bで並行実装可能な範囲へ分割し、P3-Cの共有registry/session統合とR9700実測は直列で行う。
