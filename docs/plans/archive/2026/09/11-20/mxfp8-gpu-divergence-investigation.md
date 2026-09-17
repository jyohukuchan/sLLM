# MXFP8のgfx1201／gfx1030差の切り分け

> 状態: 完了（Stage A〜Dの結果を記録。codec仮説は棄却、残る演算経路差は未特定）
> 起点: `e25bcc07` ＋ 未commitの作業ツリー
> 2026-09-15ユーザー指示: 「gfx1201とgfx1030でMXFP8が違う理由を調べる」

## 背景（実測済みの事実）

[MTP採用率ベンチマーク](../../../../../development/mtp-acceptance-benchmark.md)の
teacher-forced測定（8条件・2,048強制位置・両GPU同一prefix）で次を観測した。

| 指標 | gfx1030 (V620) | gfx1201 (R9700) |
| --- | ---: | ---: |
| MXFP8 draft top-1 が自GPUのBF16 draftと一致 | 0.9731 | **0.9824** |
| MXFP6 同 | 0.9697 | 0.9673 |
| MXFP8対MXFP6のペア差（McNemar） | +0.34 pt, z=0.82（有意差なし） | +1.51 pt, **z=3.91（有意）** |
| near-tie `[0,0.25)` での MXFP8 flip率 | 0.420 | **0.238** |
| near-tie `[0,0.25)` での MXFP6 flip率 | 0.386 | 0.405 |

MXFP6のflip率は両GPUでほぼ同じなのに、**MXFP8だけが大きく動く**。これが調査対象である。

不一致は主にnear-tieに集中する。ただし既存margin集計には、gfx1030のMXFP8で
margin `[1,2)` と `[2,4)` に各1 flip、gfx1201のMXFP6でmargin≥4に1 flipがある。
「marginが1を超えるとflip 0」という当初の記述は誤りであり、Stage Aで再集計して訂正した。
したがって本件は「量子化が系統的に劣化させているか」ではなく、
「near-tieの引き直し方がGPUでなぜ違うか」の問題である。

## 問題は2つに分かれる（既に切り分け済み）

Stage 0の`target_hidden_sha256`を照合した結果:

| 比較 | 結果 |
| --- | --- |
| **GPU内**でのBF16／MXFP8／MXFP6の`target_hidden` | **8/8 完全一致**（両GPU） |
| **GPU間**での同一系列の`target_hidden` | **0/8 一致（全条件で相違）** |

- **Q1（GPU間）**: 両GPUはdraftへ**異なる入力**を与えている。
  よって `0.9824 対 0.9731` は同じ基準に対する値ではなく、companionの性質ではない可能性が高い。
  Phase84.5が既にgfx1201のtarget M3/M1差をattention加算順（M1 wave／M3 generic）へ切り分けている。
- **Q2（GPU内）**: GPU内では入力が完全に同一なので、
  **gfx1201でMXFP8がMXFP6より有意に忠実（z=3.91）、gfx1030では差なし（z=0.82）** という観測がある。
  有意差の有無の違いだけでは相互作用を証明できず、異なるtarget hiddenやBF16基準も
  残るため、companion固有の原因とはまだ帰属できない。**本調査の主対象はQ2。**

## 既に棄却済み（再実行しないこと）

`sllm_matmul_mxfp8_w8a8_m1_col2_body` の `#if defined(__gfx1201__)` 分岐（`make_wave_block32` による
block単位scale共有）をgfx1030へも適用した診断buildで、**logits 8/8 bit一致・top-1一致1.0000**、
near-tie flip率0.420のまま不変だった（[対照実験](../../../../../../ci/matrix/mtp-bench-decode-path-ablation-v1.json)）。
`decode_scaled(value, scale)` と `decode(value)*decode(scale)` は2の冪scaleで厳密に等価であり、
laneのK割当（thread `wave*32+lane` が要素 `wave*32+lane+256j`）も同一なのでFP32加算順も変わらない。

**ただしこの対照は半分しか潰していない。** gfx1030でwave分岐をcompileしても
`ScalarCodec<E4M3Fn>::decode` のソフト復号が使われる。gfx1201固有の
native `v_cvt_f32_fp8_e32` を使う経路は、調査開始時には未検証だった。
Stage C2の独立oracleとC1のモデル対照を追加し、この範囲のnative復号仮説を棄却した。

## 調査段階

### Stage A: GPU不要の解析（既存アーティファクトのみ）

保存済みの対象logits（2 GPU×3形式×8条件＝48ファイル、
各248,320語彙×256行のf32-le、計12,205,424,640 bytes）で以下を計算する。
当初の「64ファイル／約16 GB」は対象系列数の誤記だった。
`.local-artifacts/mtp-bench/claimb/fixed2-{bf16,mxfp8,mxfp6}/` と
`.local-artifacts/mtp-bench/claimb-gfx1201/fixed-{bf16,mxfp8,mxfp6}/` の
各report.jsonが `logits_file` と `logits_sha256` を持つ。**読む前に必ずSHA256を照合すること。**

1. **logit距離（閾値なしの指標）**: 各GPU内で `MXFP8 − BF16` と `MXFP6 − BF16` の
   行ごとの相対L2・最大絶対差・top1位置での差を出す。
   top-1一致率はnear-tieで閾値化された指標なので、連続量で見ると像が変わる可能性がある。
   **gfx1201のMXFP8のlogit誤差がgfx1030より小さければ、実体のある忠実度差**。
   同程度なら、差はnear-tieの分布側にある。
2. **margin分布の比較**: 各GPUのBF16 draftのtop1−top2 margin分布を比較する。
   gfx1201のmarginが系統的に大きければ、同じlogit誤差でもflipしにくい。
   これが真ならQ2は「MXFP8が忠実」ではなく「gfx1201のBF16がtieを作りにくい」になる。
3. **flip位置の重なり**: 両GPUでflipした位置集合の重なりを見る。
   同じ位置なら「難しい位置」効果、別位置ならGPU固有の摂動。

Stage Aで2または3が説明を与えるなら、Stage B以降は実施せず記録して終了する。

### Stage B: kernel選択の帰属（host検査のみ、GPU不要）

実MTPの6形状（K/N = 10240/5120、17408/5120、5120/17408、5120/12288、5120/1024、6144/5120）について、
両targetで実際に選ばれるkernel IDとdevice symbolを列挙し、**GPU間で異なる形状を特定する**。
`native/hip/tests/public_runtime_host_test.cpp` の既存selector検査と
`native/hip/src/matmul_kernel_internal.hpp` の述語を使う。

既知の相違（確認すること、前提にしないこと）:

- MXFP6は `phase85_mxfp6_gfx1030_m1_col2_shape` が `5120<K<10240` を除外するため、
  **o投影 `K=6144,N=5120` だけgfx1030は旧ID20、gfx1201はID100**となる
  （`static_assert(!phase85_mxfp6_gfx1030_m1_col2_shape(1U, 6144U, 5120U))`）。
- MXFP8の選択述語 `phase85_mxfp_m1_col2_shape` は両targetで同一。相違はbody内の`#if`のみ。

matmul以外（activation quantizer、companion側attention、KV append）でも
target別分岐があるかを同じ方法で洗い出す。

### Stage C: gfx1201側のミラー対照（GPU、Stage A/Bで未解決のときのみ）

Stage Aの1が「gfx1201のMXFP8のlogit誤差が実際に小さい」を示した場合のみ実施する。

1. **native FP8変換の寄与**: 診断buildだけでgfx1201の
   col2で使うE4M3復号だけを既存のソフト実装へ揃え、演算配置とscale共有を維持する。
   共通`ScalarCodec<E4M3Fn>::decode`全体の変更はtarget／KV側にも届くため、
   診断helperとcol2の3 load呼出しだけに限定する。
   単にcol2の`#else`側へ強制しても、
   `BlockCodec::load`→`decode_scaled`の例外側でgfx1201 native復号が残るので、
   それだけをnative復号の対照として扱わない。診断対象のcall chainを確認したうえで、
   同一強制prefix・同一BF16基準で再測定する。
   同一GPUでlogitsが変化すれば、この復号変更の寄与を数値化する。
   変化しなければ当該native復号仮説を棄却するが、matmul全体や他の算術差までは棄却しない。
   異なる入力を持つgfx1030のflip率0.42へ近づくことを因果判定条件にはしない。
2. Phase85のbottleneck診断は「E4M3全256 code、E3M2全64 code、E8M0全256 codeをGPU上で照合し、
   有限値bitwise一致・NaN class一致で両targetPASS」と記録している。
   **その検査がnative変換経路を通り、独立した数値oracleと比較していたかを確認する。**
   同じnative命令同士の比較だけなら、ソフト復号との等価性を証明しない。独立照合がなければ、
   subnormal・NaN・境界codeに限定した両target比較を追加する（全256 codeで足りる、モデル不要）。
3. 1がnegativeなら、Stage Bで特定したmatmul以外のtarget別分岐を1つずつ戻して二分する。

### Stage D: Q1のclose（低コスト）

Q1は「両GPUがdraftへ異なる入力を与えている」で説明できる見込みが高い。
Phase84.5のgeneric対照手法（r5ではM1/M3とprefillをgenericへ揃えた。
r7はM1／prefillを維持してM2〜M4をwaveへ揃えた別対照）を参照し、
`target_hidden_sha256` が両GPUで一致するかを確認する。一致すれば、
そのうえでMXFP8の一致率がgfx1030側の値へ寄るかを見る。
**一致しない場合でも本調査の失敗とはせず、target側の未解明差として記録する。**

## 受入基準（実装前に凍結）

1. 各Stageは、肯定・否定いずれの結論でも数値付きで記録して完了とする。
   「原因を特定できなかった」も完了であり、Stage Cの実施は成果条件ではない。
2. 既存logitsを読む前に `logits_sha256` を照合し、不一致なら停止する。
3. 診断buildは本番sourceを変更しない。build後に必ず復元し、
   **変更前後のファイルSHA256一致を記録する**（今回の対照実験と同じ手順）。
4. GPU実行はexact target・UUID、HIP-only、非zero dispatch、fallbackなし、cleanup 0を満たす。
   CPU emulation、timeout、crash、zero selectionはPASSにしない。
5. 比較は必ず**同一GPU・同一強制prefix・同一BF16基準**で行う。
   GPU間の一致率の絶対値を、異なる`target_hidden`のまま比較しない。
6. 本調査の結論で本番kernel選択・既定・常駐serviceを変更しない。変更提案は別作業とする。

## 非対象

本番kernelの採用、BF16既定の変更、採用率の再定義、ベンチマークv1の条件変更、
Tier A全26条件への拡大、greedy受理規則の実装、commit／push、公開CI。
新しい外部コード取込みは予定しない。行う場合は同じ作業中に
[import log](../../../../../../THIRD_PARTY_NOTICES.md#import-log)へ記録する。

## 実行と記録

- laneはDraft。Stage A・Bはdirty treeで可、GPUを使わない。
- 再計画条件（AGENTS.md準拠）: 同一work unitが2回却下、review時間が実装時間を超過、
  機能的進捗が1時間以上停止、検証/文書が作業の30%超、見積り1.5倍超、または受入基準の変更。
- GPU運用: exact UUID（V620 `GPU-76a08c022586fed6`、R9700 `GPU-a8e9ddefa2d60f55`）を確認する。
  V620使用時はローカルQwen serviceを停止して利用不可として扱う。
  R9700は `sllm-qwen38-r9700.service`（**user unit**）を測定時だけ停止し、終了時に
  unit／run.sh／binaryのSHA256一致とhealthz／readyz 200、performance level復元を確認する。
  既存の手順実装は `ci/tools/run_mtp_teacher_forced_r9700.py` を再利用してよい。
- 強制prefixは既存の `.local-artifacts/mtp-bench/claimb/gen2-bf16/prefixes.json` を再利用する。
  **新しく生成しないこと**（位置が変わると既存の全測定と比較できなくなる）。
- raw出力・build・profileは `.local-artifacts/mxfp8-gpu-divergence/` へ置き、Gitへ追加しない。
  集約は `ci/matrix/mxfp8-gpu-divergence-v1.json`。完了時に本計画を
  `docs/plans/archive/2026/09/11-20/` へ移し、対応する
  `docs/history/2026/09/11-20/mxfp8-gpu-divergence.md` を作って相互リンクする。

## 参照

- [ベンチマーク仕様と実測](../../../../../development/mtp-acceptance-benchmark.md)
- [2 GPU集約](../../../../../../ci/matrix/mtp-bench-teacher-forced-v2.json)
- [margin分解](../../../../../../ci/matrix/mtp-bench-margin-decomposition-v1.json)
- [復号経路対照（棄却済み仮説）](../../../../../../ci/matrix/mtp-bench-decode-path-ablation-v1.json)
- [Phase84.5のtarget M3/M1切り分け](../../../../../history/2026/09/11-20/phase84-5-mtp-path-correctness.md)
- [メイン計画](../../../../main-plan.md)


## 完了結果（2026-09-15）

| 段階 | 数値結果と判定 |
| --- | --- |
| A1 | MXFP8平均相対L2はgfx1030 0.074509／gfx1201 0.073947。BF16 top-1位置での平均絶対差は0.34427／0.35571で逆向き。一様な忠実度優位を認定しない |
| A2 | BF16 margin中央値4.375／4.28125。gfx1201側が系統的に大きいという説明は支持されない |
| A3 | MXFP8 flipは共通12、gfx1030のみ43、gfx1201のみ24。重なりだけでは原因を説明できない |
| B | MXFP8全6形状は両GPUでID99。MXFP6はK6144/N5120のみID20／100。activation／attention／KVのtarget別条件を記録 |
| C2 | 独立host数式oracleのE4M3FN全256 codeは両GPUで不一致0。有限値bit一致・NaN class一致 |
| C1／C3 | col2復号、activation encode、KV encodeの3対照すべてでtarget hidden／logits hashが8/8一致、top-1 2,048/2,048一致、最大絶対差・相対L2 0。BF16からのflipは36のまま |
| D | r5のwave無効化後もgfx1030とのtarget hidden一致は0/8。診断BF16／MXFP8間は8/8一致し、同一診断BF16基準のflipは38/2,048。target側の未解明差として記録 |

C1は共通decoder全体ではなくcol2の3 load呼出しだけをソフト復号へ置換した。
C3はactivation quantizerとKV appendのencodeを独立して置換し、scale決定を維持した。
Dはr5と同じnative／Rustのwave無効化であり、長いM1の共通staged32を変更する対照ではない。

## 受入基準の照合

1. Stage A〜Dの数値結果を記録した。残る演算経路への帰属は未特定だが、当初基準が認める否定・未特定の結果として完了する。
2. 48既存logitsと各比較の参照logitsで、配列を読む前にSHA256・サイズを照合した。Stage Aの再計算top-1は6系列×2,048位置でreportと一致した。
3. 各診断buildのsource before／after hash一致、終了時の本番source 336ファイルのhash一致を確認した。
4. 全5 model run・10,240強制位置でexact gfx1201／UUID、HIP-only、nonzero dispatch、fallbackなし、cleanup 0。独立codec検査はgfx1030でも同条件を確認した。
5. 同一GPU・固定prefix・BF16基準で比較した。Dは同一診断内のBF16を基準にし、入力が異なる旧BF16やGPU間の絶対保持率を品質比較に用いなかった。
6. 本番kernel・既定を変更せず、R9700 serviceのunit／run.sh／binary hash不変、healthz／readyz 200、performance level復帰を確認した。

ELFのISA抽出に伴う生成artifactの再直列化と実行hashの対応付けは履歴に記録した。
実行artifactのidentityとsourceの復元を混同せず、実際に実行したhashを集約へ保持する。

[集約](../../../../../../ci/matrix/mxfp8-gpu-divergence-v1.json)には各比較、source／binary identity、GPU監査、未解明範囲を保存した。

[調査履歴](../../../../../history/2026/09/11-20/mxfp8-gpu-divergence.md)
