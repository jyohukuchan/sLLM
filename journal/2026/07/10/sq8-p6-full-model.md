# SQ8 P6 full model / prefill

日付: 2026-07-10

## 前回の要点

P5ではQwen3-14B-FP8の一層M=8について、4回のactivation量子化、7 projection、全intermediate、最終出力を独立CPU oracleと照合し、optimized p50 `0.7773185 ms`、reference比 `21.266652`倍、最終relL2 `0.003996148414`で通過した。P6では結果を見る前に、40層境界、final head、vLLM source oracle、実行カウンタのpromotion gateを固定し、resident stackとfinal headをfail-closedで接続する方針を確定した。

## 今回の変更点

- git `27e3bd242f15334f7f8af6d61af68942cd24de84`で、R9700の物理識別とHIP runtime識別を一つのfail-closed検査へ統合した。最初の実機runは空の`gcn_arch_name`を検出して終了し、識別検査を無効化せずに修正した。失敗ログのSHA-256は`a56ccb47e5a80f360d2ace9cc608fa164be7edac1b63fda7b1386e7a8e6a43cd`。
- 40層をresident weight、共有workspace 1個、resident hidden 1本、同一stream、各層1回のD2Dで接続した。測定経路はfallbackとhost stagingを使わず、40層後に1回同期する。layerwise auditは40回readbackする非測定経路として分離した。
- final RMSNormとsource BF16 `[151936,5120]`のlm_headをresident化し、stackのpromotion済みoutputと対応するexecution reportだけを受け付けるようにした。artifact、thin package、vLLM oracle、全passthrough payloadは名前、dtype、encoding、shape、byte数、SHA-256をfail-closedで検証した。

### Frozen source oracle

- directory: `/tmp/ullm-qwen3-14b-fp8-vllm-oracle-m8-v0.1`
- metadata SHA-256: `5caafcd2c976482dd01e51b537593d8924d381a8a9ab076b2082325e22fea39e`
- model revision: `9a283b4a5efbc09ce247e0ae5b02b744739e525a`
- input token IDs: `[1,2,3,4,5,6,7,8]`
- positions: `[0,1,2,3,4,5,6,7]`
- BOS/chat template: none
- final normalized hidden SHA-256: `a6772963cee66d8429eaa7b4e72e2594345b1a6613a06a1bf67660b4f02aa9a7`
- logits SHA-256: `24c93f3fbe0fc3d2a101c782f0e181be1206cabd56e900814a608d2a09fd268e`
- final-position top-10 IDs: `[353,3764,25010,220,5572,671,3014,374,262,16]`
- top-1/top-2 margin: `1.9375`

### Frozen numerical gates

結果を見る前に次を固定する。変更が必要な場合は、結果値ではなく演算契約上の理由を先に記録する。

#### Optimized GPU vs independent CPU SQ8 oracle

- 全40 layer boundaryでNaN/Infが0。
- 各layer boundaryでrelL2 `<= 0.10`、cosine `>= 0.995`。
- layer 39 outputでrelL2 `<= 0.08`、cosine `>= 0.997`。

#### Device final norm / BF16 lm_head vs CPU head oracle on the same hidden

- final normalized hiddenと全151,936 logitsでNaN/Infが0。
- logits relL2 `<= 2e-3`、cosine `>= 0.999999`。
- top-1 IDが一致する。

#### Optimized uLLM vs source-checkpoint vLLM oracle

- 最終positionのnormalized hiddenでrelL2 `<= 0.15`、cosine `>= 0.99`。
- 最終positionの全logitsでrelL2 `<= 0.15`、cosine `>= 0.99`。
- uLLM top-1 IDが`353`で、vLLM top-10との重複が5件以上。
- logits、top-k、output healthは同じ固定token/position/model revisionから導出する。

#### Execution contract

- layers `40`、projections `280`、activation quantizations `160`、layer D2D `40`。
- 全projectionがMとshapeに対応する測定済みCK implementation IDを返す。
- fallbackなし、測定中host stagingなし、層間同期なし。
- promptを8回のtimestep full-stackとして実行せず、M=8の1回のprefill stackとして実行する。

### Promotion result

- 最終resultは`benchmarks/results/2026-07-10/sq8-full-model-v0.1/full-model-m8-final.json`、SHA-256は`bcc7355ed9deed102ae620bf237466d2aebd78b897dd776fcb088377d4241647`で、`passed=true`。最終logのSHA-256は`23b3947aca23a501a05ab76b96758b94ec74eab570e13c373e2f3029ca9f3547`。
- 全40層がfiniteで、GPU対CPU SQ8の最悪値はlayer 39のrelL2 `0.030492298862366745`、cosine `0.9995673845541831`。layer 39の厳しいgateも通過した。診断用のGPU対vLLM層境界も、最悪値はlayer 39のrelL2 `0.06655599993460926`、cosine `0.9985139105064265`だった。
- device対CPU head oracleは、final hiddenがrelL2 `1.0960481842317075e-7`、cosine `0.9999999999999909`、logitsがrelL2 `2.060008209116002e-7`、cosine `0.9999999999999016`。device対vLLMは、final hiddenがrelL2 `0.04253753323548277`、cosine `0.9991131536875918`、logitsがrelL2 `0.041190031517998854`、cosine `0.9992311957756344`だった。
- top-1はdevice、CPU、vLLMのすべてで`353`。device top-10は`[353,3764,25010,220,5572,671,3014,374,368,262]`で、vLLM top-10との重複は9件だった。
- 測定1回はstack 1回、40層、280 projection、160 activation量子化、40 layer D2D、stack同期1回。headはD2D 1回、RMSNorm 1回、BF16 matvec 1回、readback 2回、同期1回。CK dispatchは`16x128x128`が160回、`16x128x256`が40回、K-padding `16x128x256`が80回で、fallbackと測定中host stagingは0。
- warmup 3回、測定10回で、full stack+headはp50 `34.1816625 ms` / p95 `34.235882600000004 ms`、40層stackはp50 `29.987294999999996 ms` / p95 `30.03552355 ms`、headはp50 `4.187013 ms` / p95 `4.232364 ms`だった。
- device VRAM `34208743424` byteに対し、最小accounted residentは`14775996928` byte。内訳はartifact weight/scale `13213670400`、layer norm `1679360`、共有workspace `3989504`、resident hidden `163840`、model head `1556493824` byteで、未計上余力は`19432746496` byte。allocator/backend overheadはこの計算に含めない。
- 初回成功resultのSHA-256は`f5656dfd85fbc6fafa7cf9790e7d5d60bde7e326a1bf14e9f73b80fec712725b`。初回成功と最終成功は、timingとCPU oracle elapsedだけを除外すると完全一致し、正規化JSONのSHA-256はともに`cb88838d4a698ca75ba1aeec2668c957491248a5048758b61edb5df326607210`だった。

## 次の行動

P6はfrozen gateを変更せずにgreenとなり、実装、実機result、環境、失敗履歴、再現性を証跡へ固定した。次はP7としてKV cacheとdecodeをresident runtimeへ接続し、実tokenizer入力から複数tokenを生成するreal generationを、vLLM oracleと実行契約の両方で検証する。
