# Phase 87 段階0: 計測と棚卸し

## 状態と対象

2026-09-19完了。対象はQwen3.8-27B Unsloth混合NVFP4、single GPU、
exact `gfx1030`／`gfx1201`。この記録は段階0だけを扱い、Phase 87全体の完了を意味しない。

受入項目は[計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)の段階0に固定する。

1. 両GPU、MTPなし／ありのdecode内訳、読み出し量、実効帯域、形状別copy帯域。
2. NVFP4 M=1の実行形式の確認、現行W4A4の速度・KLD基準。
3. 内訳と到達可能帯域に基づく優先順位、および過去の棄却候補の棚卸し。
4. W×A16の全利用箇所と、移行に必要なactivation global scaleの有無。

## 開始時点の観測

- 未コミットのscale選択修正、計画変更、ユーザーのREADME編集を含む作業ツリーから開始。
  既存変更は維持する。初期HEAD、status、差分は
  `.local-artifacts/phase87/stage0/initial-state.json`と`initial-diff.patch`へ保存した。
- ローカルQwenサービスは停止済み、R9700の`sllm-qwen38-r9700.service`もinactive。
  GPU測定では既存lease runnerの停止前状態・設定hash・performance levelの保存復元を使用する。
- 計画の「Qwen3.8の現行M=1はW4A16」という前提と実装の不一致を確認した。
  `build_unsloth_qwen38`系graphのNVFP4 bindingは`Encoding::Nvfp4W4A4`であり、
  `native/hip/src/matmul_runtime.inc`はこのdescriptorを`Nvfp4W4A4`へ送る。
  後述の実行時auditとkernel traceでも確認し、移行による速度・品質差とは扱わなかった。

### R9700の現行実行経路（非profile実測）

`baseline-gfx1201/baseline-mtp-off/report.json`の本測定はHIP、fallbackなし。
ID84 `matmul.nvfp4.w4a4.decode.scale_lut.v1`を14,224回（112回×127 decode transition）、
ID89 W4A4 prefillを448回選択した。したがって少なくともこの通常ベンチマークのM=1は、
計画作成時に想定したW4A16ではなく、既にW4A4である。
速度はdecode中央値19.1665 tok/s、TPOT中央値52.1744 ms、3本の測定すべて128出力token。
prefill中央値549.8210 tok/s。これは現在のW4A4基準値であり、移行前後の差ではない。

2026-09-19、上記の前提相違を報告したところ、ユーザーから
「現行W4A4を基準にして前提を訂正し、不要な移行比較は省く」と回答を得た。
計画の段階0項目2と段階1の既定切替の記述を訂正した。W4A16を診断用に追加する作業は行わない。

## 通常速度の基準

両GPUとも8192入力／128出力、1 warmup＋3 measured、固定GPU sampling、MXFP8 E4 KV、
chunk capacity 2048。MTP有効時はBF16 companion、draft width 2、catch-up既定無効。
decode速度の分子はprefillが返した最初のtokenを除く127 transitionであり、draft提案数ではない。

| GPU | MTP | prefill中央値 tok/s | decode中央値 tok/s |
| --- | --- | ---: | ---: |
| V620 `gfx1030` | なし | 226.8093 | 14.2425 |
| V620 `gfx1030` | あり | 214.9283 | 25.4067 |
| R9700 `gfx1201` | なし | 549.8210 | 19.1665 |
| R9700 `gfx1201` | あり | 540.0601 | 35.1134 |

全runが128出力、HIP-only、fallbackなし、終了後のallocation accountingは0、
retryable cleanup／durable quarantineは0。GPUは直列実行し、lease runnerで元のservice状態と
performance levelへの復元を確認した。各中央値・MAD・raw reportのhashは
`.local-artifacts/phase87/stage0/baseline-summary.json`、実行環境は各`baseline-gfx*/execution.json`にある。
通常測定binaryは`baseline-bin/gfx*/`へ保存し、`binaries.json`のhashとの一致を確認した。

V620のMTPなしでもID84を14,224回選択し、現行W4A4を確認した。両GPUの実測値を
W4A16のbaselineまたはW4A4移行の改善値として扱わない。

## KLDの基準

現行W4A4、R9700、chunk32。固定Qwen3.8コーパスの8種類・2,632位置を主集計、
同一入力の129位置をrepeat controlに用いた。基準は保存済みllama.cpp BF16／FP16 KVのlogits、
有効語彙248,077個、`KL(softmax(BF16)||softmax(candidate))`、temperature 1、単位nats。
`qwen38_kld_compare.py`でFP64正規化・集計した。

| KV | 平均KLD | 中央値 | p95 |
| --- | ---: | ---: | ---: |
| FP16 | 0.092217680 | 0.019251884 | 0.443169298 |
| MXFP8 E4（既定） | 0.103586493 | 0.019934413 | 0.477517121 |

9ケースすべてのcaptureを完了し、nonfiniteなし。FP16 KVのraw logitsは以前の同条件の9ケースと
SHA-256が全件一致した。入力・語彙hashは`kld-corpus.json`、各capture・比較・実行記録は
`.local-artifacts/phase87/stage0/kld-gfx1201/`に保存した。このKLDはW4A4移行の差ではなく、
後続最適化に対する現行数値基準である。異なるGPU・chunk・KV条件へ一般化しない。

## 棚卸し

- [W×A16の利用箇所とartifactの入力scale](phase87-a16-inventory.md)。
  Qwen3.8／Gemma 12B／Gemma 26B MoEの直接経路は既にW4A4であり、旧sidecarと
  MX A16 opt-in、公開ABI・provider・evidenceの残存箇所を区別した。
  NVIDIA Gemma 31Bはupstream indexで入力scaleを確認した参照対象であり、runtime対応済みとはしない。
- [過去の棄却・保留候補](phase87-rejected-candidates.md)。採用済みのID67／84／82等を
  棄却候補へ混ぜず、probe枝と製品IDの対応が未確定なものも区別した。

## attentionの対象追加

V620、MTPなしのprofileでfull attentionは13.3193 ms／確定decode token、
うちstage1は約13.18 msを占めた。通常速度のTPOTとは区別した診断値である。
計画の段階0項目3に従ってユーザーへ確認し、2026-09-19に
「attentionも候補に含め、帯域・時間から優先順位を決める」と承認された。
Phase 87の後続候補へdecode attentionを追加した。段階0では実装の最適化を開始しない。

## decode内訳

値は通常kernel traceの全decode区間を、実際に確定した127 transitionで割ったms/token。
MTPではdraft、verify、棄却・replay、sampling、finishを含むROCTX区間を使用した。
最後のsampler間隔を1 tokenとみなす方法は、MTPには使っていない。
profileは1 warmup＋1 measuredで、通常速度の1＋3測定とは分離した。

| GPU上の処理 | V620 MTPなし | V620 MTPあり | R9700 MTPなし | R9700 MTPあり |
| --- | ---: | ---: | ---: | ---: |
| FP8行列積 | 25.046 | 13.402 | 23.103 | 11.408 |
| NVFP4行列積 | 19.722 | 8.833 | 16.220 | 6.998 |
| full attention | 13.319 | 8.062 | 3.455 | 3.177 |
| GDN | 1.328 | 1.010 | 2.075 | 0.680 |
| 活性値量子化 | 2.364 | 1.137 | 1.966 | 0.854 |
| RMSNorm・head norm/RoPE前処理 | 1.818 | 0.871 | 1.925 | 0.839 |
| BF16行列積 | 0.447 | 2.033 | 0.453 | 1.633 |
| GPU実行時間の合計（残るsampling・KV等も含む） | 64.387 | 35.891 | 49.491 | 26.004 |
| profile区間のwall time | 74.021 | 41.938 | 57.366 | 31.326 |

GPU空白時間をHIP APIの区間と交差させた内訳は次のとおり。同期APIのwall time全体を
GPU時間へ加算せず、GPUが動いていない区間だけを数える。「HIP API外」はhost処理・
ライブラリ処理・profiler負荷等を含み、原因をlaunch overheadと断定しない。

| GPU空白時のhostの位置 | V620 MTPなし | V620 MTPあり | R9700 MTPなし | R9700 MTPあり |
| --- | ---: | ---: | ---: | ---: |
| launch API内 | 0.317 | 0.547 | 0.462 | 0.669 |
| 同期・poll API内 | 0.056 | 0.135 | 0.064 | 0.183 |
| transfer API内 | 0.494 | 0.325 | 0.605 | 0.321 |
| その他HIP API内 | 0.107 | 0.126 | 0.112 | 0.150 |
| HIP API外 | 8.661 | 4.914 | 6.631 | 3.999 |

全4条件でprofileの生成tokenは通常baselineと完全一致し、HIP-only、fallbackなし、cleanup 0。
GPU実行区間と空白区間の和がROCTX区間全体と一致することも確認した。

## 読み出し量・copy帯域・優先順位

別runのhardware counterで、GL2C/EAの32／64／128 byte要求と、V620の96 byteまたはR9700の
256 byte要求を収集した。これは当該interfaceのread-request量であり、物理DRAM bus量ではない。
counter runの遅くなった時間を帯域の分母にせず、通常kernel traceの時間と組み合わせた。
組合せ前に生成token、完全なkernel symbol＋grid、呼出し回数が一致することを検査した。

| GPU | MTP | read-request GB/token | 全binが揃ったdecode kernel数 |
| --- | --- | ---: | ---: |
| V620 | なし | 21.272763 | 148,717 / 148,717 |
| V620 | あり | 10.659153 | 69,129 / 69,129 |
| R9700 | なし | 21.445920 | 148,717 / 148,717 |
| R9700 | あり | 10.424325 | 64,923 / 64,923 |

形状別copyは各GPUで73条件。行列積のM=1／2／3、LM headを含む実K/N、活性値入力量、
attentionのKV代表payload、非整列境界を含む。`uint4` copy後に、使用した全bufferの全byteを
GPUで検査した。大きなcopyでclockを暖め、1 MiB以上のpayloadは512 MiB以上の領域を巡回する。
最初のcold-clock版と3-buffer版は保存したが、順位付けには使わない。3-buffer版ではV620の大きな
cacheにKV相当payloadが収まり、1 TB/s前後の値になったためである。

copyの帯域はread＋writeを合わせたdecimal GB/s。小さいpayloadはcache／launch latencyの影響を
含む。copyとmatmulのread-requestは異なるtraffic構成なので、copyは厳密な帯域上限ではなく観測参照である。
この違いによりmatmulがcopy参照を上回る行もあり、負のscoreを0に隠していない。

順位指標は計画どおり **GPU時間の割合 ×（copy参照帯域 − 実効read-request帯域）**。
時間割合の分母はdecode GPU kernel時間の合計で、host空白は上の別表へ分離した。
copy候補が複数ある場合は範囲を保存し、表のscoreはその最大値を使う探索用指標とする。
カーネルごとの全時間・byte量・帯域・候補形状・signed scoreは[結果JSON](phase87-stage0-results.json)にある。

| GPU・MTP | 最優先kernel | ms/token | 実効 GB/s | copy参照 GB/s | score |
| --- | --- | ---: | ---: | ---: | ---: |
| V620・なし | MXFP8 attention stage1、M=1 | 13.181 | 21.21 | 355.6〜357.6 | 68.85 |
| V620・あり | MXFP8 attention stage1、target M=3 | 7.268 | 47.54 | 355.6〜357.6 | 62.78 |
| R9700・なし | MXFP8 attention stage1、M=1 | 3.374 | 82.82 | 470.8〜491.1 | 27.83 |
| R9700・あり | MXFP8 attention stage1、target M=3 | 2.963 | 71.35 | 470.8〜491.1 | 47.83 |

attentionのcopy参照はcontext 8193／8256／8319のKV payloadを中心とした約17.3〜17.6 MB。
MTPのquery数・KV再利用による実traffic差はcounter側へ含める。queryごとにKV全体を読むという
仮定でbyte数を水増ししていない。copy参照の最小値を用いてもattention stage1が4条件すべてで最上位となる。
ただし、この指標は行列積の余地を過小評価していた。下記のレビュー補正を優先順位の正とする。

### レビュー補正（2026-09-19）

上の指標には、行列積の余地を小さく見せる偏りが二つある。

- **GL2C read-requestは物理DRAM量でも論理重み量でもない。** cache上の再利用を含むため、
  NVFP4 decodeでは論理重み量8.42 GB/tokenに対しcounterは9.79〜10.06 GB/token（約2割多い）。
  その結果、実効帯域がcopy参照を上回り、scoreが負になった。
- **copy参照はread＋writeの混合で、読むだけの処理の上限として低い。**

段階0の`shape_candidates`（MTPなし、M=1の実K/N）から重みの論理byte数を計算し直した。
FP8は1 byte/要素、NVFP4は0.5＋1/16 byte/要素。scaleの細部は含めない。

| GPU | FP8行列積 | NVFP4行列積 | full attention |
| --- | --- | --- | --- |
| V620（ピーク512 GB/s） | 25.05 ms、10.62 GB、424 GB/s（83%） | 19.72 ms、8.42 GB、427 GB/s（83%） | 13.32 ms、21 GB/s |
| R9700（ピーク640 GB/s） | 23.10 ms、10.62 GB、460 GB/s（72%） | 16.22 ms、8.42 GB、519 GB/s（81%） | 3.46 ms、83〜85 GB/s |

仮にピークの90%を読み出し上限とした短縮余地（ms/token、MTPなし）は次のとおり。
90%は到達を確認した値ではなく比較用の仮定であり、必達値ではない。

| GPU | attention stage1 | FP8行列積 | NVFP4行列積 |
| --- | ---: | ---: | ---: |
| V620 | 約12.4（copy参照） | 約2.0 | 約1.5 |
| R9700 | 約2.8（copy参照） | **約4.7** | 約1.6 |

V620ではattentionが最優先という結論は変わらない。R9700ではFP8行列積（hipBLASLt MT16x16系の
`K6144,N5120`、`K5120,N10240`等）がattentionより大きい。当初の「4条件すべてでattentionが最上位」は
R9700について訂正する。NVFP4にも約1.5 ms/tokenの余地があり、「正の帯域余地なし」は指標の偏りによる。
実際の読み出し上限は、読むだけのstreaming kernelで別に測るまで未確定とする。

### WU0実測による置換（2026-09-19）

WU0で同じ73 payloadを両GPU、copy/read各3起動、3 warmup＋9 measuredで測定した。
上の90%仮定を形状別read実測へ置き換えた結果は次のとおり。行列積は論理weight bytes（resident scale込み）、
attentionはcontext平均8256のunique K/Vを使い、GL2C/EA要求量は使わない。

| 対象 | signed余地 ms/token | 正の形状別余地 ms/token | 半分の線 ms/token |
| --- | ---: | ---: | ---: |
| V620 attention stage1 | 12.2506 | 12.2506 | **6.1253** |
| R9700 attention stage1 | 2.7031 | 2.7031 | 1.3516 |
| R9700 FP8 projection | 2.1657 | 2.3594 | **1.1797** |
| V620 FP8 projection | −2.4155 | 0 | 適用しない |
| V620 NVFP4 projection | −1.0560 | 0 | 適用しない |
| R9700 NVFP4 projection | −0.2249 | 0 | 適用しない |

read probeが既存matmulの論理帯域を下回る形状もあるため、負値は最適化不能の証明にしない。
これは起動・集約・アクセス／cache条件を含む観測参照で、物理DRAM上限ではない。
WU1はV620を主対象にR9700も比較し、その後WU2へ進む。
[WU0履歴](phase87-wu0-read-bandwidth.md)と[73 payload・kernel行の結果JSON](phase87-wu0-read-bandwidth-results.json)を
今後の参照値とし、以前の90%推定値は経緯として残す。

2026-09-20の[WU0再計測](phase87-wu0-read-bandwidth.md#再計測-クロック状態の補正2026-09-20)で、
上表はクロックが上がり切る前の計測だったと分かり、次へ置き換えた。

| 対象 | 正の形状別余地 ms/token | 半分の線 ms/token |
| --- | ---: | ---: |
| V620 attention stage1 | 12.4143 | **6.2072** |
| R9700 attention stage1 | 2.7869 | 1.3935 |
| R9700 FP8 projection | 4.0813 | **2.0406** |
| V620 FP8 projection | 1.052 | 0.526 |
| V620 NVFP4 projection | 0.832 | 0.416 |
| R9700 NVFP4 projection | 1.412 | 0.706 |

### GPU空白時間の内訳（レビュー追加）

profile中のGPU空白をkernel間の隙間の長さで分けると、長時間の停止ではなく、
kernelごとの小さな隙間の積み重ねが大半だった（profiler負荷を含む）。

| 条件 | kernel数/token | 空白 ms/token | 隙間の中央値 | 2〜10 µsの隙間 | 50 µs以上の隙間 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V620・MTPなし | 1,171 | 9.63 | 7.9 µs | 7.36 ms | 1.74 ms |
| V620・MTPあり | 544 | 6.05 | 4.8 µs | 3.06 ms | 2.17 ms |
| R9700・MTPなし | 1,171 | 7.87 | 4.5 µs | 5.80 ms | 2.06 ms |
| R9700・MTPあり | 511 | 5.32 | 4.4 µs | 2.12 ms | 2.73 ms |

- 50 µs以上の大きな隙間は主にtokenごとの往復である。sampler後のcopy（`__amd_rocclr_copyBuffer`）から
  次tokenのembeddingまでで、大半の時間はHIP APIの外（host側の処理）にある。
- MTPありでは、`sllm_kv_state_bf16_to_mxfp8_e4_token_major_v1`とattentionの間に約1.5 ms/tokenの隙間がある。
- 通常計測との差（TPOT − profileのGPU時間合計）はMTPなしでV620約5.8、R9700約2.7 ms/token、
  MTPありで約3.5／2.5 ms/token。profile中の空白の一部はprofiler負荷である。
- 対策候補は、kernel融合によるkernel数の削減、decode 1段全体のgraph化、sampling結果のdevice上保持である。
  同日のユーザー決定で、decode 1段全体のHIP graph化とMTPなし・ありのサンプリング経路の
  CPU-GPU間通信削減をPhase 87の段階5として追加した。

## 次の作業単位案（未着手、レビュー補正後）

1. **MXFP8 E4 decode attention stage1（V620優先）**。共通のGQA=6、head dim=256、M=1〜3の経路を対象に、
   KV復号・scale処理・head間再利用・laneごとの仕事量を調べる。copy参照まで近づける仮定での短縮余地は、
   MTPなしでV620約12.40 ms/token、R9700約2.81 ms/token、MTPのtarget stage1で約6.30／2.53 ms/token。
   これは帯域だけによる粗い見積りで、演算・復号・同期を無視した必達値ではない。
   次の着手時に具体的な一つの仮説と候補を決め、その影響範囲の見積りへ絞る。
2. **FP8 W8A8のprojection（R9700優先）**。R9700ではhipBLASLtの`K6144,N5120`、`K5120,N10240`等が
   ピークの約65〜75%に留まり、上の補正で最大の余地を持つ。V620は約83%で余地は小さい。
   M=1だけでなくverify M=3を同じ演算条件で扱う。
3. **活性値量子化・RMSNorm等の融合可能性**。kernel時間は合計でMTPなし約4 ms/token。
   byte量が小さく、帯域だけの指標ではlaunch／融合による利益を十分表現できないため、別仮説として扱う。
4. **NVFP4 W4A4 decode**は既存ID84等を基準とする。論理byte数では両GPUとも約81〜83%で、
   約1.5 ms/tokenの余地がある。上の項目の後に扱う。

各作業単位の前に、読むだけのstreaming kernelで実際の読み出し上限を測り、上の余地見積りを置き換える。

提案元はこの段階0の実測。対象は上記の演算条件、費用は次の1作業単位での限定調査と最大3候補の比較、
有効期限は次の測定またはkernel構造の変更までとする。旧Phaseの棄却案は
[棄却・保留台帳](phase87-rejected-candidates.md)と照合し、同じ前提の再試行にしない。
GDNの新しい最適化はこの段階0で対象へ追加していない。host実行制御は上記のとおり段階5へ追加した。

## 証拠と検査

- 通常baseline、profile、counter、KLD、copyのraw出力は`.local-artifacts/phase87/stage0/`。
  [結果JSON](phase87-stage0-results.json)に4組の対応する入力path/hashとkernel別集計を固定した。
- `phase87_decode_profile.py`はROCTX区間のcoverage、shape衝突、counter bin欠落を区別する。
  `phase87_profile_breakdown.py`はGPUとHIP APIの重複を差し引き、idleを二重加算しない。
  `phase87_stage0_summary.py`はtoken・kernel/gridの呼出し回数・counter coverageを照合して集約する。
- generic FP8 quantizerとBF16 MTPの一部では、symbolだけでKを一意に切れないため、
  候補集合とcopy帯域の範囲を残した。kernelごとのread-request量そのものは全dispatchのcounterによる。
- benchmarkへ追加したROCTXは`SLLM_PHASE87_PROFILE=1`のときだけ動的に読み込む。
  production kernelや推論の既定選択は変更していない。開始時に記録したgraph、matmul runtime、
  lowp launch／kernel sourceのhashも計測後に一致した。
- 最終copy probeは両targetでHIP compileと実GPU全byte検査を通過。
  profile集計のfocused host tests、Markdownのlocal link検査、diff whitespace検査を実施した。
  終了時は両V620・R9700ともGPU busy 0、performance level `auto`、
  R9700 service inactive、ローカルQwen service stoppedで、開始時の状態を維持した。
- 統合確認で、4条件のkernel/grid join、127 transitionの分母、token一致、counter domain、
  shape候補とscoreを独立照合し、重大な問題なしと確認した。結果JSONのevidence indexには
  実行metadata、binary hashを持つ記録、KLD比較、copy記録のpath/hashも含めた。
  これはdirty local treeでの段階0の証拠であり、release candidateの不変identityを主張しない。

段階0の4項目（両GPUの内訳と帯域、訂正後のW4A4速度・KLD基準、優先順位、W×A16棚卸し）は完了。
後続の最適化・W×A16削除・MTP形式変更は未着手である。

## 計測の扱い

- 非profileの代表8192入力／128出力、1 warmup＋3 measuredの中央値を速度基準とする。
- profiler計測は内訳の診断用。profile時のwall timeを通常速度と混ぜない。
- 論理byte数、counterによるinterface要求量、物理DRAM量を区別する。今回測ったcounterはinterface要求量である。
  copyのread＋write帯域とmatmulのread帯域の分母も明記する。
- 実行失敗、未取得、未確認の項目はPASSにしない。

計画: [Phase 87](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)
