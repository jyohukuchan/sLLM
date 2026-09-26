# Phase 87 段階3: MTP companionの形式

2026-09-24着手。同日に形式比較後の完了を差し戻し、WU-3Sで低bit companionの遅延原因、
受理率込み速度と公開経路を確認した。較正済みNVFP4 W4A4は明示sidecarとして追加し、
MXFP6 W6A6のMTP companion sidecarは退役させた。測定時の自動判定ではBF16を維持したが、
後述の同日ユーザー決定により、現行の既定companionはNVFP4とする。
比較対象はQwen3.8 NVFP4 targetに対するMTP companionのNVFP4 W4A4とMXFP6 W6A6。
target重み、shared embedding／縮小draft head、MXFP8 E4 KV、幅2のp/q受理規則は維持する。

## 実装と較正

- 8本のMTP BF16行列をsLLMの既存`quantize_nvfp4_weights`でE2M1 packed値、K軸block16のE4M3FN scale、
  F32 weight tensor scaleへ変換し、検証済みsidecarとしてgraph／residentへ接続した。
  既存MXFP8と測定時のBF16 control選択を維持する。MXFP6 sidecarの退役と最終既定は後述する。
- NVFP4の活性値scaleは[固定した較正入力](../../../../../ci/fixtures/phase87-stage3-calibration-v1/manifest.json)の
  英日中6条件から得る。これらは
  `mtp-bench-v1`の本文・凍結列・token列と重ならないことをhost toolで検査した。
  実行時はBF16 companionの5入力（fusion、Q/K/V、o_proj、gate/up、down）のproducer直後を
  GPUから読み、両targetのamaxの大きい方から`g=f32(amax/(6×448))`を計算する。
  MTP graphのlayer IDは64、weight名の`mtp.layers.0`とは別である。
- `g`はNVFP4 sidecar固有の**resident値**として保存・uploadする。
  Unsloth target artifactのreciprocal `input_global_scale`を流用しない。
  standalone RMSNormはNVFP4 producer出力へ未対応のため、MTP q/k/vとgate/upのRMSNormは
  BF16出力からmatmul前に量子化する。対応済みのSiLU×up producerだけNVFP4出力を使う。
- 較正入力manifest SHA-256は`cbdb70f1de9d2ed6d44c77cd5407934eb2f1dfb48156b5eed24219c828c28a93`、
  suite SHA-256は`54629f273d5213e259534000104b51a87bbec7da907b2ac802506f1fdd3e5978`。
  両GPU較正reportのSHA-256はV620 `48b202a8782953e0bd983e8b4a58b1c52dc7c8962529bd93a52adff25bf68cdb`、
  R9700 `8934c0029badfd34c58b0bce8b30c8b1eee3b82a51be65ff3acaa8fe1ffd05da`。
  6/6条件ずつHIP-only、fallbackなし、request／session cleanup zeroでPASSした。

| BF16 producer入力 | 両GPUのamax | 対応するNVFP4入力scale `g` |
| --- | ---: | ---: |
| `mtp.concat.output` | 55.5 | 0.02064732 |
| `layer.64.input_rmsnorm.output` | 61.75 | 0.02297247（q/k/v共通） |
| `layer.64.full.sigmoid_mul.output` | 23.25 | 0.00864955 |
| `layer.64.post_attention_rmsnorm.output` | 51.75 | 0.01925223（gate/up共通） |
| `layer.64.mlp.silu_mul.output` | 78.5 | 0.02920387 |

較正scale manifest SHA-256は`0ff19cf876f1ef2c475b0856f470dbf611583963fdb47bf1d1e5b8998841d9ce`。
生成したNVFP4 sidecarは追跡対象外の`.local-artifacts/phase87/stage3/mtp-nvfp4-v2/`に置き、
manifest SHA-256 `b6334312e4d64404d7a31618005a707d317a8ba5ff53a8feccb639c2cbbd32c4`、
payload SHA-256 `b2dd8a77b9820a4927af724b13680ff69ad8e761d41cc6a63acc69d23ca66905`、
combined recipe digest `sha256:d9698c41954ef7b53a2937c0f662ac2a273f1bdc40c602f77d4928b63de991e1`。
MXFP6 sidecarは既存`.local-artifacts/phase84/mxfp6/`をcurrent loaderで検証して使用する。

M1 campaignのbinaryはV620 SHA-256 `95771eaaa98164af394e4b1498ffa260b4a0c19082381e3a06088ac51d1b6186`、
R9700 `075ed60125f414bb794747211f97a6f917c1514e8c034e42d7fe0fbc43bc551a`。
campaign後のGPU実行source変更はsidecarの説明コメントとlint annotationだけである。
最終sourceから再buildしたbinaryとの`.text`、`.rodata`、HIP `.hip_fatbin`のbyte一致を両targetで確認し、
追跡対象外の`.local-artifacts/phase87/stage3/binary-semantic-map.json`
（SHA-256 `6e6034f0000d7ed79e9a44103db666bea0b400931ebd96a96aeede8815268678`）へ対応を記録した。
後続の通常速度はこの最終source binaryで測る。

## GPU smokeと測定状況

最初のNVFP4 smokeはMTP q/k/vのstandalone RMSNorm prequant拒否、次のsmokeは
post-attention RMSNormの同じ拒否で終了した。いずれもcleanupとR9700 service復元を確認し、PASSへ含めない。
BF16 RMSNormへ戻し、resident scaleの逆数誤りも修正した後、8192入力／128出力のNVFP4
MTP smokeはexact V620 `gfx1030`とR9700 `gfx1201`でHIP-only、fallbackなし、cleanup zeroをPASS。
単回の受理数・速度は生成列が分岐するため採否には使わない。

固定列M1は`mtp-bench-v1` Tier A 26条件、既定98,304語彙draft headでBF16／MXFP6／NVFP4を
同一binary・同一prefix campaign内で比較した。各値はprompt cluster平均で、M1はtop-1一致数ではなく
top-k20／top-p0.95後の期待p/q受理率である。
BF16凍結列に加え、独立のW8A8凍結列を
`.local-artifacts/phase87/stage3/prefixes-26-w8a8-v2.json`
（SHA-256 `4b339eb2822af3ca8e50d0ce375066c2c984ea163b8736b43b209fd1d05af129`）として固定した。
元suiteのpromptとtoken IDを再検算し、同じtoken数の本文改変も拒否する。双方の列でcandidateとBF16対照を
同じcampaign内に置き、V620／R9700各26条件をHIP-only・fallbackなし・cleanup zeroで完了した。

| GPU | 凍結列 | BF16 M1 | MXFP6 M1（BF16差） | NVFP4 M1（BF16差） | NVFP4−MXFP6 |
| --- | --- | ---: | ---: | ---: | ---: |
| V620 `gfx1030` | BF16 | 0.7745 | 0.7717（−0.284 pt） | 0.7620（−1.248 pt） | −0.964 pt |
| V620 `gfx1030` | W8A8 | 0.7745 | 0.7728（−0.173 pt） | 0.7637（−1.082 pt） | −0.909 pt |
| R9700 `gfx1201` | BF16 | 0.7733 | 0.7699（−0.341 pt） | 0.7619（−1.141 pt） | −0.800 pt |
| R9700 `gfx1201` | W8A8 | 0.7743 | 0.7707（−0.362 pt） | 0.7606（−1.371 pt） | −1.008 pt |

同一buildのBF16対照は[段階9の縮小head対照](phase87-stage9.md)の0.7745／0.7733と一致した。
NVFP4とMXFP6の差は両GPU・両凍結列とも5 pt未満。NVFP4のBF16差は、既定昇格に関する
[MTPベンチマーク規則](../../../../development/mtp-acceptance-benchmark.md)の1.0 ptを両GPU・両凍結列で超える。
条件ごとのNVFP4−MXFP6の符号は、V620で17/26、R9700で16/26が凍結列を変えても一致した。
一致した条件のうち負方向は16/17、15/16で、符号不一致の条件を除いた平均差も
V620 −1.152／−1.432 pt、R9700 −1.009／−1.412 pt（BF16／W8A8凍結列）である。
集計と実行契約の監査は追跡対象外の`.local-artifacts/phase87/stage3/m1-policy-audit.json`
（SHA-256 `f168ed6a4ae202b21f45e8479b5b83eb92e2e2359f9c7a0d79e9f8303cad5927`）に保存した。

## 実効速度と既定形式

通常8192入力／128出力、MTP幅2、固定GPU sampling、warmup1＋measured3を両GPUで
BF16→MXFP6→NVFP4→BF16の順に実行した。中央値はmeasured 3回のTPOTから計算した。
前後BF16対照のtoken SHAは各GPU内で一致し、TPOT差はV620 +0.183%、R9700 −0.017%。

| GPU | BF16先／後 tok/s | MXFP6 tok/s（BF16前後平均比） | NVFP4 tok/s（同） | 自由生成の受理／提案 BF16・MXFP6・NVFP4 |
| --- | ---: | ---: | ---: | --- |
| V620 `gfx1030` | 32.55／32.49 | 30.13（−7.34%） | 32.34（−0.56%） | 77/101・75/105・76/104 |
| R9700 `gfx1201` | 37.71／37.72 | 34.85（−7.59%） | 33.26（−11.80%） | 75/105・73/110・66/123 |

全8 jobがexact GPU UUID、HIP-only、fallbackなし、非有限terminal logit 0、
128出力、cleanup zeroでPASSし、性能levelとR9700 service状態を復元した。
監査結果は追跡対象外の`.local-artifacts/phase87/stage3/speed-policy-audit.json`
（SHA-256 `f54ce375598078f73ff70c1a12b950912c27dceb9162fc5cd117b68717b77af0`）。
最終source binary SHA-256はV620 `948d30278ca3ee53628d62b27511038f72b13c597c5d58353c261e192609d9fc`、
R9700 `4609811be737fe085cae53d3bfa7de9aad95ba67388a938c12c42be0f0917088`。
NVFP4のMTP常駐model byte数は両GPUで22,393,597,824、MXFP6は22,486,495,040、
BF16は23,004,065,600だった。

この当初の速度は[ベンチマーク仕様](../../../../development/mtp-acceptance-benchmark.md)のM5相当の
単一prompt自由生成参考値であり、候補間で生成token列と受理ブロック数が変わる。
Tier A 26条件のM1は上表の受理数とは別に固定列で判断した。
この最初の形式比較時点ではM3の実prompt別draft／non-draft時間と、M1から導くM4のprompt-cluster区間は未取得だった。
後述のWU-3Sで両方を取得した。

NVFP4−MXFP6のM1差は両GPU・両凍結列で−0.80〜−1.01 ptであり、計画の**−5 pt条件は不成立**。
MXFP6を追加で必須対応へ昇格させる条件は生じなかった。NVFP4はBF16比M1が
−1.08〜−1.37 ptで、当時の既定昇格の1.0 pt許容範囲を全比較で超えた。
この時点ではBF16を既定に維持し、NVFP4と既存MXFP6を明示sidecarとして残した。
2026-09-24のWU-3Sで旧1.0 pt規則を廃止し、MXFP6 sidecarも退役させた。
既定のtarget-only出力は両GPUの8192/128で[段階9](phase87-stage9.md)のtoken SHAと一致した。
NVFP4 sidecarの公開`generate --mtp-weights`も両GPUで16 tokenをHIP-only・fallbackなしで生成し、
recipe digest一致を確認した。

## 低bit companionが遅い原因（2026-09-24、WU-3S）

段階3の最終binaryを、BF16／MXFP6／NVFP4 companionでそれぞれrocprofv3のkernel traceにかけた
（8192入力／128出力、MTP幅2、warmup 1＋measured 1、両GPU、6 runともexit 0）。
measuredのdecode区間（`sllm_phase87_decode_mtp` marker）のkernelを名前・gridごとに「中央値×回数」で集計し、
MTP verifyのattention呼び出し数から数えたblock数で割って比べた。V620のNVFP4 runには1回84 msなどの
profiler由来とみられる外れ値があったため、合計ではなく中央値で集計している。
原票と集計scriptは追跡対象外の`.local-artifacts/phase87/stage3-speed-cause/`（`run_profiles.sh`、`analyze2.py`）。

| GPU | companionの行列積 ms/block（BF16） | MXFP6 | NVFP4 | kernel時間/block BF16 → MXFP6 → NVFP4 |
| --- | ---: | ---: | ---: | --- |
| V620 | 3.84 | 6.34（+2.50） | 1.74（−2.10） | 83.46 → 86.19 → 81.34 ms |
| R9700 | 3.23 | 約4.7（+1.5） | 約1.3（−1.9） | 61.15 → 62.64 → 59.34 ms |

**MXFP6: companionの行列積kernelが遅い。** MXFP6のM=1 kernel `sllm_mxfp6_w6a6_m1_col2_v1` は、
読むbyte数がBF16の約0.38倍なのに、1回あたりの時間はBF16 kernel（`sllm_matmul_bf16_fp32_decode_v4`）の1.1〜1.7倍かかる。

| 形状 | V620 BF16 | V620 MXFP6 | R9700 BF16 | R9700 MXFP6 |
| --- | --- | --- | --- | --- |
| gate/up K5120 N17408 | 360 µs（496 GB/s） | 606 µs（115 GB/s） | 285 µs（625 GB/s） | 432 µs（161 GB/s） |
| q＋gate K5120 N12288 | 254 µs（496 GB/s） | 431 µs（114 GB/s） | 202 µs（622 GB/s） | 308 µs（159 GB/s） |
| k/v K5120 N1024 | 24 µs（430 GB/s） | 43 µs（95 GB/s） | 35 µs（297 GB/s） | 38 µs（108 GB/s） |

BF16 kernelはほぼ読み出し帯域どおりに動くが、MXFP6 kernelは帯域の約2〜3割で、E3M2の展開と積和が律速していると推定する。
これで1 blockあたりV620 +2.5、R9700 +1.5 msとなり、受理数の差（V620 75/105対77/101、R9700 73/110対75/105）と合わせて
通常計測の−7.3%／−7.6%をおおむね説明できる。

**NVFP4: kernelは遅くない。** NVFP4 companionの行列積はBF16より1 blockあたりV620で2.10、R9700で1.9 ms短く、
kernel時間/blockも2.5〜3%短い。通常計測の遅さ（V620 −0.56%、R9700 −11.80%）は受理数の違いで生じている。
特にR9700の単一prompt自由生成では受理66/提案123（約54%）で、BF16の75/105より大きく低く、1 blockあたりのtoken数が約15%減った。
固定列M1ではNVFP4のBF16差は−1.1〜−1.4 ptにとどまるため、この大きな低下は単一promptの生成経路のばらつきとみられる。
ベンチマーク仕様どおり、自由生成（M5）ではなくM1とM3から導くM4で判断すべき量である。

**判断への影響**
- NVFP4は、当時のルールではM1がBF16比1.0 ptを超えて低いため既定昇格の条件を満たさなかった。
  2026-09-24にこのルールは廃止され、[main-planの採否ルール](../../../../plans/main-plan.md#変更の採否ルール2026-09-24ユーザー決定)で
  受理率込みのdecode速度により判断することになった。
- MXFP6は、M1のBF16差が−0.17〜−0.36 ptで1.0 pt条件の範囲内にある。遅いのはkernelの効率が原因なので、
  MXFP6 M=1 kernelを帯域近くまで改善できれば、M4がBF16を上回って既定昇格の条件を満たす可能性がある。
  BF16 kernel並みの実効帯域なら、MXFP6 companionの行列積は1 blockあたり約1.3〜1.4 ms（BF16比約−2.5 ms、V620のblockの約3%）になる見込み。

### NVFP4のprefix準備が遅い原因

通常8192入力／128出力のmeasured 3回で、BF16前後対照のprefill中央値平均はV620 37.95秒、R9700 14.88秒。
NVFP4は44.54／20.84秒だった。うちMTP prefix準備のhost wall中央値はBF16 1.56／0.11秒、
NVFP4 7.76／6.04秒で、prefill増分のほぼ全量に一致する。
target本体のprefill後、companionは1024行ずつfc・q・k・vまで実行してKVを作る。
そのK/N形状はtarget本体向けに採用済みのNVFP4高速prefill形状に合わず、汎用
`sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1`へ送られる。
既存rocprofのwarmup後の実測prefill区間では、このkernel群32起動の合計がV620約7.47秒、R9700約5.95秒で、
同区間のprefill増分をおおむね説明する。これはM=1のdecode用NVFP4行列積がBF16より速かった事実と矛盾しない。
raw traceは追跡対象外の`.local-artifacts/phase87/stage3-speed-cause/`、通常速度の監査は
`.local-artifacts/phase87/stage3/speed-policy-audit.json`に置く。
今回の既定採否だけでprefill／TTFT退行を無視するというユーザー決定と、後でこの経路を最適化する課題は
[backlogのP12](../../../../plans/backlog.md)に記録した。

## WU-3Sの最終結果

### MXFP6 companionの退役

MTP companion専用のpacked MXFP6とBF16 roundtrip MXFP6の変換・manifest読込み・graph選択を削除した。
旧MXFP6 sidecarを公開`generate --mtp-weights`へ渡すと、両GPUでexit 2と
`retired MTP MXFP6 sidecar encoding`を返し、別形式へfallbackしない。
converterでも旧encoding指定を明示拒否する。これは従来opt-inでMXFP6 companionを使っていた場合の
互換性変更である。MXFP6本体モデルの形式・量子化器・共用kernel
`sllm_mxfp6_w6a6_m1_col2_v1`は残す。kernel効率の改善は[backlog P1](../../../../plans/backlog.md)に残した。

### 26条件のM3とM4

`mtp-bench-v1` Tier Aの26条件を同じprompt token SHA、固定K20 p/q、seed 123、幅2、
98,304語彙draft headでBF16とNVFP4それぞれ1 warmup＋3 measuredに固定した。
whole-decode graphを無効にしてproposalのhost wall時間を測り、decode全体から差し引いて
non-draft wall時間を得た。以下のM3は各promptのmeasured中央値を出した後の26 prompt中央値であり、
通常のwhole-graph製品速度ではない。

| GPU | 形式 | M3 draft ms/block | M3 non-draft ms/block | M3 total ms/block |
| --- | --- | ---: | ---: | ---: |
| V620 `gfx1030` | BF16 | 8.0750 | 66.3158 | 74.4899 |
| V620 `gfx1030` | NVFP4 | 6.0852 | 66.7345 | 72.8156 |
| R9700 `gfx1201` | BF16 | 6.9225 | 55.6215 | 62.4908 |
| R9700 `gfx1201` | NVFP4 | 5.1063 | 55.5317 | 60.6543 |

M1のproposal 1歩目と2歩目の期待受理率から`1+a₁+a₁a₂` token/blockを各promptで求め、
M3のtotal ms/blockで割った。M4は同一promptのNVFP4/BF16速度比を26 promptで平均し、
prompt単位20,000回のbootstrapで95%区間を計算した。M1とM3のprompt token SHAは26/26一致した。

| GPU | 凍結列 | M4のNVFP4/BF16速度差 | prompt-cluster 95%区間 | 改善したprompt |
| --- | --- | ---: | ---: | ---: |
| V620 | BF16 | +1.704% | +1.173〜+2.199% | 23/26 |
| V620 | W8A8 | +1.759% | +1.194〜+2.326% | 20/26 |
| R9700 | BF16 | +1.748% | +1.271〜+2.205% | 25/26 |
| R9700 | W8A8 | +1.607% | +1.074〜+2.117% | 22/26 |

M3両GPUのBF16/NVFP4全52条件×4回はHIP-only、fallbackなし、finite terminal logit、
fixed-K20 p/q、request/session cleanup zeroでPASSし、GPU性能levelとR9700 service状態を復元した。
26条件suiteのSHA-256は`707be808c2121940d6d457bb95f79c97e58508de4e8e2cde8e76b0b81aeff2ca`、
縮小headは`24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0`。
M3公開binaryのSHA-256はV620 `ba1bae23c57e9833b2478bd39960e0ca16914e6888295ff2a38f9a1509595d47`、
R9700 `e7fefef2ff2c35bfbc37899b4595ceecfd0763196d786e0fbc061ca6eda2729a`。
M3原票は`.local-artifacts/phase87/stage3/m3-wu3s-gfx1030-v2/`と
`.local-artifacts/phase87/stage3/m3-wu3s-gfx1201-v1/`、M4派生値は同階層の
`m4-wu3s-{gfx1030,gfx1201}-{bf16,w8a8}.json`に置いた。
M3 report SHA-256（BF16／NVFP4）はV620
`6b436c50da11efdca19e296177ecbfe6a7a48bf4f4cb2e28a56094fd93912b06`／
`44928450773c39c8f9e8f735cb3e99983ae812903ce3f9bdc022cf5c42e0bae4`、R9700
`08beadc478b1a440600881df6820ce214dcd730aca2f6891f996956ca0f24230`／
`81a1dbedbe4fba83e153c741abdff2a6ea0cca801919480f8`。
最初のV620 M3 binaryはHIP runtime stubだったためGPU証拠から除外し、上の公開runtime binaryで再実行した。

### 通常whole-graphの同一process AB/BAと採否

同一processにtargetを1組、BF16/NVFP4のMTP companionを両方常駐させ、通常のcoding8192/128、
幅2、固定GPU samplingでABとBAを各1 warmup＋3 measuredにした。各値はmeasured TPOT中央値である。
両GPUともHIP-only、fallbackなし、request/session cleanup zeroでPASSした。

| GPU | 順序 | BF16 TPOT ms | NVFP4 TPOT ms | NVFP4の短縮率 |
| --- | --- | ---: | ---: | ---: |
| V620 | AB | 30.7372 | 30.9296 | −0.626% |
| V620 | BA | 30.7599 | 30.9074 | −0.480% |
| R9700 | AB | 26.5302 | 30.0537 | −13.281% |
| R9700 | BA | 26.5220 | 30.0425 | −13.274% |

V620のBF16／NVFP4自由生成の受理／提案は77/101・76/104、R9700は75/105・66/123で、
候補間の生成token列も異なる。M4は固定列上の期待受理率とwhole-decode無効時のblock時間から作った推定値で、
通常生成のTPOTは全roundで悪化した。

差の内訳はAB順の中央値で明確になる。V620は128 tokenに要したblockがBF16 51→NVFP4 52、
R9700は53→62だった。R9700では1 blockのdecode wallが63.572→61.562 msとNVFP4の方が約3.16%短いが、
9 block増えた費用が約554 ms、既存53 blockで節約した時間が約107 msで、差し引き約447 ms遅い。
V620も1 blockが76.542→75.540 msと短いが、1 block増えた約75.5 msが既存51 blockの節約約51.1 msを上回る。
固定列M1からの期待確定token/block平均はR9700でBF16 2.424、NVFP4 2.397（約1.13%減）だったのに対し、
この通常生成1条件の実値は2.415→2.065（約14.52%減）だった。初めて生成列が分岐した位置は
R9700で7 token目、V620で10 token目。M1は26条件の凍結committed列に対する期待p/qであり、
実際のp/q抽選で進む単一promptの生成列と保持状態を固定しない。AB/BAは同じseedで同じ差を再現したが、
この結果だけで他のpromptでも同じ受理低下が起こるとは断定しない。別入力・256出力のM3自由生成26条件では、
R9700の実際の確定token/block平均はBF16 2.449、NVFP4 2.456で、この大幅低下は平均では見えなかった。
companion residentはBF16 1,353,516,032 byte、NVFP4 743,048,256 byteで、
NVFP4は610,467,776 byte少ない。種類Bの共通ルールを機械的に適用すればBF16維持となる測定結果だが、
2026-09-24の追加ユーザー決定により、今回の段階3では共通ルールを適用せず、**NVFP4を既定にする**。
既知のTPOT低下とprefill／TTFT低下は受け入れたトレードオフとして記録する。draftだけの変更で
p/q補正後のtarget分布は維持するため、KLDはこの採否で要求しない。

AB/BA公開binaryのSHA-256はV620 `7b06abf524257a03e7ea3e470b8422b63131dd6a399d528610c1922401756a1b`、
R9700 `bf3002ac5b0c4cb26ec9c2c630aaf910675a6ddc912fba69aa33984882e52191`。
最終clippyで補助moduleが独立binaryとして検出される配置を修正し、両targetの公開runtimeを再buildした。
再build SHA-256はV620 `986a0d029320e4c420deca95fcda7f12ab1494ce3ee6f0ae26b3d212c5418291`、
R9700 `ad885c4466a9e24335e62041919fb9508d43d49dd4fda22d975700c8e5e34683`。
計測binaryとの全ELF sectionを両targetで比較し、異なるのはbuild-id、symbol tableとstring tableだけだった。
`.text`、`.rodata`、`.hip_fatbin`、`.data.rel.ro`を含む他のsectionはbyte一致し、
上の計測は最終sourceの実行意味へ対応する。対応表は追跡対象外の
`.local-artifacts/phase87/stage3/binary-semantic-map-wu3s-abba.json`
（SHA-256 `62eb8ea9d48ae062b788cbe0402f3aca3b4f385cc6a8f4c8f4844b41780c567f`）。
report SHA-256はV620 `c0dd95357c74a3526bc91977802a5dc15f170575129d902f9939b52899303843`、
R9700 `58ec24d1809c3789fdfb3701f7b423df2d9d07a18d4a645cd985fc112ae16f65`。
原票とGPU UUID、runner cleanup・性能level復元は`.local-artifacts/phase87/stage3/abba-wu3s-gfx1030-v1/`と
`.local-artifacts/phase87/stage3/abba-wu3s-gfx1201-v1/`に置いた。

両GPUの公開CLIは較正済みNVFP4 sidecarで16 tokenをHIP-only・fallbackなしで生成し、
combined recipe digest `sha256:d9698c41954ef7b53a2937c0f662ac2a273f1bdc40c602f77d4928b63de991e1`と一致した。
MTPなしの通常8192/128も両GPUでPASSし、段階9のtoken SHAとV620
`sha256:c9c0b4ee401b11544fe0faaec15d882cb2491c31a460a32a89dce28e437bcc0e`、
R9700 `sha256:75d36def8ff45d155373ebb05b885d0e4f0e7197ba9adb2123adfb1fe63b189d`で一致した。
今回だけprefill／TTFT退行を採否から除外するユーザー決定は[backlog P12](../../../../plans/backlog.md)へ
増分と原因を記録した。後続の採否へは引き継がない。

### ユーザー決定後の既定解決

今回のユーザー指示は、段階3のNVFP4 MTP draft companionについて共通の性能採否ルールをすべて適用せず、
NVFP4量子化を既定にして段階3を閉じるものとした。この例外はこのMTP companionの既定選択に限定し、
他の量子化形式・後続作業の採否ルールは変更しない。

MTPが有効で`--mtp-weights`を指定しない場合、production shared backendはartifact root配下の
`.sllm/mtp-nvfp4-v1/`を既定sidecarとして解決する。sidecarのNVFP4 encoding、source/model lockとの結合、
combined recipe digest `sha256:d9698c41954ef7b53a2937c0f662ac2a273f1bdc40c602f77d4928b63de991e1`を検証し、
欠落・破損・digest不一致は明確なエラーで終了する。`--mtp-weights`の明示指定は引き続きこの解決を上書きし、
MTP無効時の経路は変更しない。

検証済みsidecarの`manifest.json`／`payload.safetensors`をモデルrootの上記既定位置へ原本と
SHA-256一致で配置した。SHA-256はそれぞれ
`b6334312e4d64404d7a31618005a707d317a8ba5ff53a8feccb639c2cbbd32c4`／
`b2dd8a77b9820a4927af724b13680ff69ad8e761d41cc6a63acc69d23ca66905`。
生成物約228 MBはGitに追加しない。別のmodel rootで既定MTPを使う場合も、同じreview済みrecipeの
sidecarを`.sllm/mtp-nvfp4-v1/`へ配置する必要がある。

最終公開CLI binaryのSHA-256はV620
`5dbc63ad54e743f478917509ba70665b389caced0ab97987894be32335bb965e`、R9700
`4841dd864a157a88bec4ec2e6863b2747c3750774fd36b8e2b93db6b6640d2b0`。
両GPUで`--mtp-weights`省略の16 token生成がHIP-only、fallbackなしでPASSし、auditの
`mtp_weight_encoding=nvfp4-w4a4-e2m1-block16-e4m3fn-f32`とcombined recipe digestが既定identityに一致した。
最終CLI report SHA-256はV620
`3a5c02636023ac59951a66f4097098e5435a28c0d0651b77674de31f952cd91b`、R9700
`19846b045b1563006c6be892cadd40f50867538c01158b642887449f81c4b94e`。
V620の公開CLIではMTP無効の16 token生成（`mtp=false`、sidecarなし）と明示MXFP8 sidecar指定もPASSした。
R9700のHTTP serverは`--draft mtp-auto`・`--mtp-weights`省略で起動し、Chat Completionsの
16 token応答をHTTP 200で返して正常終了した。最終HTTP response SHA-256は
`725926ae36a219fcf286a5b33ed9af388d1498400c766a1fcd7d80a7c0d2c10f`。
R9700 service状態はinactiveへ復元し、診断portも閉じた。最終server binaryのSHA-256はV620
`dc5b3d54389fe0bf943376d9d6e621328b9caacbe3b04d046ff6ef4398b43bc5`、R9700
`4fcadca809154778209f4c7fa609c70cf536cf43f0f6196dd526810d49438587`。

同日追加で依頼された入力数・出力数を増やす27条件×2 seedの追試は、ユーザー指示により途中で停止した。
残った出力は探索的な部分証拠であり、既定化の証拠や全入力への受理率保証には使わない。

## 検証と制約

- `cargo test -p sllm-core --lib`: 681 passed、24 ignored。
- `cargo test -p sllm-cli --bin sllm-convert-qwen38-mtp`: 3 passed。
- `cargo check`（変更したcore／CLI／HIP benchmark）、`cargo fmt --check`、
  変更3 crateの全target clippy `-D warnings`、較正入力・scale集約のPython testと
  `mtp_expected_acceptance.py --self-test`: PASS。
- WU-3S後の`ci.tests.test_phase87_stage3_m4` 2件、Markdown local link validator、
  `git diff --check`: PASS。公開HIP runtime binaryを両targetでbuildし、ELFのHIP fatbinを確認した。
- NVFP4既定化後の`Qwen38Nvfp4BackendConfigV1` host選択テスト、変更したserver／CLIの
  全target clippy `-D warnings`、両targetの公開CLI・R9700 HTTP server smoke: PASS。
- 較正入力と`mtp-bench-v1`評価入力の本文・token列非重複を固定tokenizerで確認。
  sidecarの全8行列のresident input scaleは較正manifestの値とbit一致した。
- 旧smoke失敗と逆数scaleの旧sidecarは採用証拠へ含めず、修正後のsidecar v2と
  最終source binaryの結果だけを判断に使った。新しい外部source importはない。

計画: [Phase 87 段階3](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階3-mtp-companionの形式)。
