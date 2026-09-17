# Phase86: MTP catch-up条件付けの検証

状態: 完了（有意な改善を確認できず、既定採用せず）

計画: [Phase86計画](../../../../plans/archive/2026/09/11-20/phase86-mtp-catch-up-conditioning.md)

## 記録範囲と受入番号

本記録は設計比較、実装、状態検査、効果測定と採否をまとめる。Stage 0では幅2、受理数0/1/2について、target検証後にMTP KVへ入るhiddenの由来と行offsetを固定した。外部engineのコードは転載せず、固定checkoutのパス・revision・行番号だけを記録する。

Phase86計画の受入番号は次のとおりである。

| 番号 | Stage 0との関係 |
| --- | --- |
| 1 | Stage 0の差分を数値付きで記録する。否定的な結果も有効な完了とする。 |
| 2 | catch-up実装を行う場合もopt-inとし、採否まで既定経路を変えない。 |
| 3 | 効果判定は位置単位でなくprompt cluster単位とする。 |
| 4 | 後続GPU証拠はexact target・UUID、HIP-only、fallbackなし、非zero dispatch、cleanup 0で判定する。 |
| 5 | 既存M1がtarget条件付け・chain 1歩目だけを測った限界を独立に記録する。 |
| 6 | 比較は同一GPU内で行い、GPU間の採用率絶対値を比較しない。 |
| 7 | 既定採用で出力列が変わる場合は数値出力変更台帳へ記録する。 |
| 8 | BF16 companionを量子化companionより先に評価する。 |

## Stage 0の記号と共通offset

target検証ブロックの入力を `x0=現在のpending token`、`x1=draft1`、`x2=draft2` とする。`T-1` は `x0` の直前にtargetが出力したhidden、`T0/T1/T2` はtarget検証行0/1/2のhiddenである。受理したdraft数を `a` とし、確定target入力行数を `r=a+1` とする。

外部engine準拠の確定prefixは次の行対応になる。

| 受理数 `a` | 確定target入力行 | MTP KVを作るhiddenの由来 |
| ---: | --- | --- |
| 0 | `x0` | `x0:T-1` |
| 1 | `x0,x1` | `x0:T-1`, `x1:T0` |
| 2 | `x0,x1,x2` | `x0:T-1`, `x1:T0`, `x2:T1` |

`T2`は次のbonus tokenのseedに使われるtarget hiddenであり、`x2`のMTP状態を作るhiddenではない。複数requestをflattenした場合も、表のoffsetはrequest内のrow offsetで記録し、global offsetは`query_start_loc`との対応を別に残す。

## 固定参照と外部engineの実装対応

固定revisionは [source-lock](../../../../references/source-lock.md) の次の値を使う。

| engine | 固定revision | Stage 0の参照箇所とoffset |
| --- | --- | --- |
| vLLM | `v0.26.0`, `568afb3a13806beb53bb2e6bd518269357b237c0` | `reference/vLLM/vllm/v1/worker/gpu_model_runner.py:5166-5202` が拒否後の`target_token_ids`と同じoffsetの`target_hidden_states`を作る。`reference/vLLM/vllm/v1/spec_decode/llm_base_proposer.py:1192-1210` の拒否数は幅2で`2-a`、残る行数は`r=a+1`。初回forwardへのhiddenコピーは`llm_base_proposer.py:821-860`、IDを1行ずらして次proposalと融合する処理は`846-858`。初回forward内のtarget hidden row offsetは受理0/1/2で`[0]`/`[0,1]`/`[0,1,2]`。同一proposalの2歩目以降は`llm_base_proposer.py:712-754`で直前draft output hiddenを入力する。 |
| SGLang | `v0.5.16`, `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` | target verifyのhidden取得は`reference/SGLang/python/sglang/srt/speculative/eagle_worker_common.py:485-559`。`eagle_worker_v2.py:873-892`で`accept_lens=a+1`から`select_index=base+a`を作り、target hiddenの選択offsetは受理0/1/2で`0/1/2`。`eagle_worker_v2.py:971-1013`でそのhiddenを次入力へ渡す。`eagle_worker_common.py:104-185`と`eagle_info.py:307-315`の`num_accept_tokens`が受理prefixの行数を表す。draft proposal内部の再帰forwardは`eagle_worker_v2.py:623-707`でdraft hiddenを使う。 |
| llama.cpp | `b10453`, `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` | `reference/llama.cpp/common/speculative.cpp:1302-1305`が`verify_h`を直近target verification rowsとして定義。`1427-1490`で非共有KV時のcatch-upを行い、row0は`pending_h=T-1`、row1以降はtarget embeddingを1行ずらして`T0,T1`を使う。target hidden row保存は`1524-1540`、受理後のseed offsetは`1696-1708`の`i_h=min(a,n_rows-1)`。KV共有時だけ`1461-1462`でcatch-upを省略する。 |

vLLM/SGLangは初回・extend forwardへtarget hidden rowsを渡し、同じproposal内の追加draft stepだけdraft hiddenへ切り替える。llama.cppはtarget verification batch全体を先にcatch-upし、受理後のseedを`verify_h[a]`へ戻す。これは「全てのdraft proposal行が常にtarget hiddenで生成される」という意味ではなく、次の受理prefixに必要な行の再条件付けと、proposal内部のdraft再帰を分けた記録である。

## 現行sLLMの幅2比較

通常経路 `propose_mtp_draft`（`crates/sllm-frontend/src/generation.rs:2118-2152`）とdevice-selector経路 `propose_mtp_draft_with_device_selector`（`2066-2111`）はhidden ownershipが同じである。selectorはtoken選択とcounterだけを変える。

proposal開始時のrow0は`last_target_hidden_bf16=T-1`、row1はrow0のMTP出力hidden `D0`で条件付けされる。target検証後の`committed_rows`は`generation.rs:1969-1973`、不足分のrewindは`1996-1997`、全受理時のstate-only追加は`1985-1995`で決まる。

| 受理数 `a` | 現行sLLMが検証直後に保持するMTP状態 | 外部engineとの差分 |
| ---: | --- | --- |
| 0 | row0=`x0:T-1`。row1を1回rewind。 | 差分なし。 |
| 1 | row0=`x0:T-1`、row1=`x1:D0`。 | row1が外部の`T0`ではなくdraft hidden。 |
| 2 | row0=`x0:T-1`、row1=`x1:D0`。state-onlyでrow2=`x2:T1`を追加。 | row1が外部の`T0`ではなくdraft hidden。row2はtarget hidden。 |

上表はtarget verify直後のpending block基準である。後続の`finalize_mtp_draft`は`committed_input_rows`に応じて未消費行をさらにrewindするため（`generation.rs:2263-2304`）、Stage 0の受理0/1/2表と最終消費後の状態を混同しない。

## M1測定の限界

既存teacher-forced M1は `crates/sllm-hip/src/bin/sllm-phase78-qwen38-benchmark.rs` の`stage0_fixed_entries`で、強制行ごとに`draft.decode_mtp(token, &last_hidden)`を呼び、`last_hidden`を毎行target decode outputへ更新する。従って全位置がtarget hidden条件で、chain 1歩目だけを測る。本番の幅2では2歩目がdraft output hidden `D0`を入力にするため、M1のdraft忠実度値を本番の採用率や2歩目品質へ直接読み替えない。

## Stage 0数値と記録上の留意点

`stage0_state`由来の集計として、Tier A 26条件では保持行6,275、draft由来行1,828〜2,603（29.13〜41.48%）だった。全8条件では保持行2,042、draft由来行587〜862（28.75〜42.21%）だった。集計はfrozen reference MTP countersに依拠する。幅1終端補正ありで、受理数・位置別の厳密ヒストグラムは保存されていないため、これはStage 0の保持割合レンジであり、位置ごとの完全な分解ではない。

Stage 0数値の追記欄: （上記のstage0_state報告値を反映済み。厳密ヒストグラムは空欄）
- 端末表示のtruncationや要約だけを数値証拠にしない。行offsetは上記固定checkoutを再読し、request-local row、受理数、hidden sourceを同じmanifestへ対応付ける。
- raw output、logits、trace、buildは計画指定の`.local-artifacts/phase86-mtp-catch-up/`へ置き、tracked historyへコピーしない。
- この記録は設計比較の途中状態であり、実装、GPU測定、完了、既定採用の主張を含まない。

## Stage 2a: 分離catch-upと状態検査

`SLLM_QWEN_MTP_CATCH_UP=separate` をrequest構築時に固定し、通常とdevice-selectorの
両verify経路へ接続した。既定は従来経路。幅全体のdraft遷移をrewindした後、
確定入力と `[直前target hidden, verify hiddenの先頭r−1行]` をstate-only batchへ渡す。
catch-up時間は既存proposal時間へ含め、独立した累積時間getterも追加した。

両GPUでraw coding 127／Japanese 129 tokenの2入力×受理0／1／2を検査した。
KV有効payload、次hidden、次logits、次tokenがscalar state-only対照とexactに一致した。
3行catch-up後の3→1／3→2 truncationも同様に一致した。各GPUの集約dispatchは
ターゲット5,838／MTP 1,772、exact target、HIP-only、fallbackなし。
resident破棄後current bytes 0、poisoned false、retryable／durable cleanup 0。
これはcatch-up状態の限定検査であり、full-model品質全般の認定ではない。
rawは `.local-artifacts/phase86-mtp-catch-up/pilot-{gfx1030,gfx1201}/state-check/` に保存した。

## 自由生成の副測定（8192入力／128出力）

同一バイナリ・同一GPU内でoffとseparateを1 warmup＋3 measuredで比較した。
各系列内の生成列は再現したが、系列間の生成列は異なるため採用率の因果的な主評価にはしない。
速度は実committed decode token数127を、block数×実測per-block時間で割った値である。
終端打ち切りで `blocks + accepted` が128となる場合があり、127へ勝手に揃えない。

| GPU | 経路 | blocks | accepted/proposed | draft ms/block | 非draft ms/block | 導出decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gfx1030 | off | 53 | 74/106 | 12.605 | 82.475 | 25.202 |
| gfx1030 | separate | 55 | 73/110 | 13.910 | 81.156 | 24.285 |
| gfx1201 | off | 50 | 78/100 | 10.025 | 62.191 | 35.176 |
| gfx1201 | separate | 54 | 74/108 | 11.404 | 61.980 | 32.059 |

中央値の比較でV620は約3.6%、R9700は約8.9%低下した。
変更前binaryとのoffの生成token hashは両GPUで一致した。
rawは `.local-artifacts/phase86-mtp-catch-up/stage2a-free-{gfx1030,gfx1201}/`、
集計は同rootの `free-analysis.json` に保存した。

## 統合reviewとホスト検査

Stage2の統合reviewを1回行い、指摘箇所だけを再確認した。
診断のhidden offset、rewind済みownerの再利用、失敗集約、truncation audit、
allocator cleanup、および集計のoff/separate・fixture identity検査を修正した。
最終レビューにblocking findingは残っていない。coverageの最低値追加は採用していない。
frontend hostは102 PASS／1 ignored、測定器のブロック・hash・空計測契約6 PASS、統計・位置対応契約5 PASS。
両exact targetのrelease buildと対象Rustのformat checkを通した。
初回buildのROCm論理パス不一致とcodegen指定不足は環境引数を正して解消し、
失敗logもraw rootへ残している。

## 入力準備の初回失敗

`full-{gfx1030,gfx1201}/T` はmodel provision後、prompt digest表記の不一致で失敗した。
既存helperは`sha256:`付き、manifestはbare hexで、raw file hash自体は一致する。
続けてtoken digestの定義も確認し、manifestがCSV十進UTF-8、runtimeがLE i32という
違いを分離した。host照合で26件すべてのprompt、sequence file、CSV token digest、
token countが一致した。失敗はPASSへ読み替えず、post-error cleanup 0も記録した。
入力の補完をprepare-onlyへ分離してから、同じ26条件・同じ凍結列で再開する。

## Stage 3: Tier A 26条件の強制列比較

既存8列＋凍結BF16から補完した18列を同じprefix fileへ固定した。
SHA-256は `7d7be1a0ca3946fba592891d96eb4155c8c9235a714a448dbbb3a409f1b0c1a3`。
最大promptは8,284 token、出力列は最大256 token、自然終了した短い4列も維持した。
Tは各GPU6,286位置。既存8条件の全2,048位置のdraft logits hashは、各GPUの旧M1と一致した。
T／P／P+separateの全26条件は両GPUでPASS、HIP-only、fallbackなし、cleanup 0だった。

Pの受理は次の強制tokenへのtop-1一致で決める診断であり、本番p/q受理率ではない。
表はpromptごとの率を等重みで平均した値。区間はprompt clusterを再標本化した
20,000回のpaired percentile bootstrap、p値は同じ単位の両側sign-flip検定（seed 86）。

| GPU | P off | P separate | 差（ポイント） | 95%区間（ポイント） | p値 | 改善／同率／悪化prompt |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| gfx1030 | 67.3003% | 67.8358% | +0.5355 | −0.1406〜+1.2359 | 0.1462 | 11／6／9 |
| gfx1201 | 67.3284% | 67.5618% | +0.2333 | −0.4792〜+0.9909 | 0.5437 | 11／8／7 |

両GPUとも改善の有意性は確認できない。これは効果ゼロの証明ではない。
受理数でblock境界が変わるため、位置の単純な独立標本検定は行わない。
共通の絶対位置・同じproposal stepに限るmargin比較では、V620は5,072 pairの188 flip中
160件、R9700は4,996 pairの187 flip中159件がcontrol draft margin < 1に集中した。
margin >= 4にもそれぞれ4／5件あり、near-tieだけで全flipを説明しない。

R9700のT/P top-1一致は1歩目2,635/2,711（97.20%）、2歩目2,045/2,711（75.43%）だった。
TはM1 target、PはM3 verify/replayなので、差にはtarget演算形状も含まれる。
この比較を純粋なcatch-upの効果とはみなさず、従来M1の範囲を示す補助指標とする。

Stage2aで採用率改善を確認するという条件を満たさないため、Stage2bの融合catch-upと
MXFP8／MXFP6 companionへの拡張は実施しない。これらを検証済みとは記録しない。

## 最終速度比較と採否

最終campaignはT／P／P+separateと自由生成off／separateを、各GPU内で同じbinary、
同じ固定prefix、同じsampling設定で測定した。自由生成は1 warmup＋3 measured。

| GPU | 経路 | draft ms/block | 非draft ms/block | 自由生成の導出decode tok/s |
| --- | --- | ---: | ---: | ---: |
| gfx1030 | off | 12.524 | 81.657 | 25.445 |
| gfx1030 | separate | 13.822 | 81.125 | 24.322 |
| gfx1201 | off | 10.156 | 62.651 | 34.895 |
| gfx1201 | separate | 11.219 | 61.512 | 32.340 |

Stage3の強制列tokens/blockと上記per-block時間から計算したTier A M4は次のとおり。
各promptの相対速度差を等重みで平均し、prompt clusterで区間を求めた。

| GPU | M4相対差の平均 | 95%区間 |
| --- | ---: | --- |
| gfx1030 | −0.374% | −0.977〜+0.238% |
| gfx1201 | +0.305% | −0.307〜+0.952% |

両GPUとも0を跨ぎ、既定採用に必要な優位性は成立しない。
M4は強制top-1診断と観測した中央値のM3を組み合わせた推定であり、実運用p/qの
throughput保証ではない。区間は観測したM3中央値を条件とし、時間推定の不確かさを
すべて包含するものでもない。自由生成では系列間のtoken列が変わり、別経路の
端点時間からcatch-upの普遍的な性能を主張しない。

**Phase86は本範囲で有意な改善を確認できなかったという結論で完了する。**
BF16 companionとcatch-up無効の既定を維持する。分離catch-upは明示opt-inの診断経路として残す。
融合catch-upと量子化companionの比較は条件未成立により未実施であり、成功扱いにしない。

## 最終照合

- 6強制列report×26条件、4自由生成report、各GPU6状態case＋4truncationを照合した。
- exact UUID、HIP-only、nonzero dispatch、fallbackなし、非finiteの拒否、cleanup 0を確認した。
- 変更前と最終offの生成token hash、selected kernel counts、proposal／accepted会計は両GPUで一致した。
- R9700は開始時inactiveを維持し、unit／run.sh／binary hash不変。両GPUのperformance levelはautoのまま。
  稼働サービスを停止・再起動したcampaignではないためHTTP health/ready復元検査は適用しない。
- 最終計測後のsource差は、測定器の空計測をPASSにしないreport guardとそのhost testだけである。
  現在sourceを両targetで再buildし、既存reportも同じ非空条件で再照合した。GPU演算・sampling・時間計測は不変。
  測定binaryと現在binaryを同一と偽らず、`measurement-source-mapping.json`へ対応を記録した。
- この作業以前の変更は保持した。commit／push／公開CIは本Phaseの対象外。

## 追記: 期待p/q受理率による再解析（2026-09-17）

Stage 3の受理判定は「draftのtop-1が次の強制tokenと一致するか」であり、本番のtemperature 1.0 p/q受理ではなかった。
p/q受理では提案 `x ~ q` が確率 `min(1, p(x)/q(x))` で受理されるので、1提案行の期待受理率は
`Σ_t min(p′(t), q′(t))` になる（`p′`・`q′` はtargetとdraftの分布にengineのsupport変換を施したもの）。
これは本番の受理規則そのもので、サンプリング雑音が無く、top-1一致のようにnear-tieで離散化しない。

モードPはdraftとtargetの両logitsを全行保存していたため、**GPUを再実行せず**既存の最終campaignから計算した。
support変換は参照token selector（`native/hip/src/token_selector_kernel.hip.cpp` のtop_k／top_p分岐）に合わせ、
logit降順（同値は小さいtoken id優先）→top-k 20→softmax→累積がtop_p×総和（0.95）に達した候補まで→再正規化とした。
draft行とtarget行は同じrun内で `(sequence_index, block_row)` により対にした。
受理数でブロック境界が変わりPとseparateで行集合が異なるため、比較はprompt単位の平均で行う。
区間はprompt clusterの20,000回bootstrap、p値は同単位の両側sign-flip検定（seed 86）。

| GPU | 期待受理率 P off → separate | 差（ポイント） | 95%区間（ポイント） | p値 | 改善／悪化prompt |
| --- | --- | ---: | --- | ---: | --- |
| gfx1030 | 77.63% → 78.06% | **+0.428** | **+0.145〜+0.728** | **0.0057** | 19／7 |
| gfx1201 | 77.62% → 77.87% | +0.246 | +0.010〜+0.488 | 0.0595 | 16／10 |

同じデータのtop-1指標では両GPUとも区間がゼロを跨いでいた（gfx1030 −0.14〜+1.24、gfx1201 −0.48〜+0.99）。
期待受理率の区間はその約3分の1の幅で、**gfx1030では改善が有意、gfx1201でもほぼ同方向**だった。
したがって本文の「有意な改善を確認できず」はtop-1指標での結論として正しいが、
**catch-upの効果がゼロという意味ではない。小さい正の効果が存在する。**

提案段ごとの行プール平均（記述統計）では、効果は2歩目に集中した。

| GPU | 1歩目 off → separate | 2歩目 off → separate |
| --- | --- | --- |
| gfx1030 | 81.00% → 81.21%（+0.21） | 71.51% → 72.06%（+0.56） |
| gfx1201 | 81.05% → 81.07%（+0.02） | 71.28% → 71.77%（+0.49） |

catch-upが直すのは保存済みMTP KVの条件付けだけで、各ブロック2歩目の入力がdraft自身のhiddenであることは変わらない
（これはvLLM／SGLangでも同じ）。2歩目の方が弱い提案であるぶん、文脈の改善が効きやすいと解釈できるが、原因は確定しない。

### 実務上の大きさ

幅2では2本目のdraftは1本目が受理されたときだけ有効なので、期待tokens/blockを `1 + a₁ + a₁·a₂` で見積もった
（段間の独立を仮定した近似）。catch-upはtarget側に影響しないため、非draft時間は固定し、draft時間の増分だけを加えた。
最終campaignで記録されたseparateの非draft時間の低下（gfx1030 −0.53 ms、gfx1201 −1.14 ms）は測定の揺れとして扱う。

| GPU | tokens/block | 分離catch-upの時間増 | **分離catch-upの正味** | 融合catch-upの上限 |
| --- | --- | ---: | ---: | ---: |
| gfx1030 | 2.3892 → 2.3973（+0.34%） | +1.38% | **−1.02%** | +0.34% |
| gfx1201 | 2.3882 → 2.3926（+0.18%） | +1.46% | **−1.26%** | +0.18% |

分離catch-upはforwardが1回増えるため明確に正味マイナスであり、**既定不採用の判断は変わらない**。
融合catch-upの上限は追加forwardのコストをゼロと置いた値で、1歩目がM=1からM≈3になる実コストは含まない。
計画の「2aで採用率改善を確認した場合のみ2b」という条件は、正しい指標ではgfx1030で成立していたことになるが、
上限が+0.2〜0.3%にとどまるため、2026-09-17のユーザー指示により**融合catch-upは実施しない**。

### 教訓と後続

- temperature 1.0の推測デコードの採用率評価では、期待受理率 `Σ min(p′, q′)` を主指標とする。
  top-1一致はflip／margin解析の補助指標にとどめる。この変更を
  [MTP採用率ベンチマーク](../../../../development/mtp-acceptance-benchmark.md)へ反映した。
- 計算器は `ci/tools/mtp_expected_acceptance.py`（`--self-test` でsupport変換の境界とtie処理を検査）。
  結果は `.local-artifacts/phase86-mtp-catch-up/pq-expected-acceptance/final-{gfx1030,gfx1201}.json`。

集約: [Phase86結果](../../../../../ci/matrix/phase86-mtp-catch-up-v1.json)。
raw・build・source mapping・最終監査は `.local-artifacts/phase86-mtp-catch-up/` に保存した。
最終監査は `final-audit.json`、統計は `final-analysis.json`、実行台帳は
`final-{gfx1030,gfx1201}/execution.json` を参照する。

測定後の2026-09-17に、`crates/sllm-hip/src/bin/phase86_mtp_catch_up.rs`をcargoが単独binaryとして自動検出してビルドが失敗するため、
内容を変えずに`crates/sllm-hip/src/bin/phase86_mtp_catch_up/mod.rs`へ移し、`sllm-phase78-qwen38-benchmark.rs`の`#[path]`を更新した。
`.local-artifacts`のsource mappingは移動前のパスを記録している。

計画: [Phase86保存済み計画](../../../../plans/archive/2026/09/11-20/phase86-mtp-catch-up-conditioning.md)。
