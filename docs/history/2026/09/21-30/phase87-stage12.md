# Phase 87 段階12: MTP幅3・4

2026-09-25着手。幅2を既定に保ち、Qwen3.8 NVFP4で幅3・4を明示選択できる機能を先に実装した。
当初はWU-12Dで幅2・3・4の受理率込み速度を測って既定を決める計画だったが、同日の後続ユーザー決定で
GPU共通の既定幅2を維持し、段階12の追加の長時間ベンチマークを止めて重大退行の防止だけを残した。
対応する[計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)の段階10・11より前に行う。

## WU-12A: 選択機能（2026-09-25完了）

- CLI generate/chatとserverのQwen3.8設定に幅2／3／4を通す。CLIの0は従来どおりMTP無効を示す。
  `--mtp-weights`省略時の較正済みNVFP4 sidecar既定は維持する。Qwen3.8以外のdevice selectorは従来の幅3上限を維持する。
- Qwen target verifyのGDN checkpointを`token_count=width+1`、`checkpoint_rows=width`として2／3／4に一般化する。
  幅3・4の部分受理、全受理、batch失敗、pool／state解放をhostと両GPUのcomponent試験で確認する。
- 2026-09-25時点でRust coreの全lib test 687件PASS、変更したCLI／serverのhost選択test、全targetのcargo checkはPASS。
  native checkpointのcomponent GPU試験は両exact targetでPASS。device backingを要求行数だけ確保し、core preflightと一致させた。
  同じstateでwidthを変更する場合の再確保はgraph pin／進行中参照を拒否し、失敗時は旧stateを保持する。
- 公開CLIの16-token短生成を固定入力・seed17・較正済み既定NVFP4 companionで両GPUの幅2／3／4へ実行した。
  6条件とも`state=PASS`、exact target、HIP-only、fallbackなし、出力16 token、companion digest
  `sha256:d9698c41954ef7b53a2937c0f662ac2a273f1bdc40c602f77d4928b63de991e1`で一致した。

| target | 幅2 accepted/proposed | 幅3 accepted/proposed | 幅4 accepted/proposed |
| --- | ---: | ---: | ---: |
| V620 gfx1030 | 10/10 | 10/18 | 10/22 |
| R9700 gfx1201 | 10/11 | 10/15 | 11/14 |

  この単一短生成の受理数は幅の速度・品質判定に使わない。rawは
  `.local-artifacts/phase87/stage12/smoke-gfx1030/`と`smoke-gfx1201/`。
  CLI binary SHA-256はV620 `1031d19532104d20f4ee26182273b1d78f7d1bb0d0b618c244178b0afbf203d3`、
  R9700 `669320ffb81792030550b7f7c8e35de2282f66751dc86cff327883f98f89196c`。
  R9700 serviceはinactive、port 8000は閉じた状態を維持した。この時点では固定seedの反復決定性、
  stop／予算／context末尾、全幅のTier A受理率が未確認だった。

追加でR9700のserverを`--draft mtp-auto --mtp-draft-width 4`、`--mtp-weights`省略で起動し、
Chat Completionsの16 token応答がHTTP 200でPASS、graceful shutdownはexit 0だった。
response SHA-256は`5790858d9d84b1f1ce683fbe23de9598c60e081f063b25127b0c338124831c6d`。
server binary SHA-256は`226a20bdf4089e73afb685e73d45abef44eedf729eadc79a94846b9550e854ae`。
rawは`.local-artifacts/phase87/stage12/server-gfx1201-width4/`。R9700 serviceはinactiveのまま、
port 18087／8000は閉じた。

追加の機能境界では、production p/q selectorの独立long-double GPU oracleへ幅4の
3段受理→4段目拒否→残差ID 11を加えた。`widths=[1,2,3,4,8]`、11 valid caseで
gfx1030／gfx1201ともPASSし、fallbackは0。rawは
`.local-artifacts/phase87/stage12/pq-gfx{1030,1201}-v1.json`。
R9700の公開CLIでは固定seedの幅3・4反復が出力文、usage、提案・受理・拒否数、
proposal block数まで一致した。全幅で`max_new_tokens=1`と`幅+1`の予算、および
`--stop idempotent`を実行し、長さ／stop reason、全processのexit 0、service・
performance level復元を確認した。12追加runは
`.local-artifacts/phase87/stage12/tail-gfx1201-v1/`に保持し、実行記録のSHA-256は
`28e06eda3b2bdd8a8b62c1599de3a3c36d8575d6c42eeb6931a0a3b92665a7f0`。

両GPUのserverでは各幅2／3／4に`context_length=32`を指定し、同じ22-token入力から
10-token生成して`usage.total_tokens=32`、finish reason `length`を確認した。6ケースとも
HIP-only、fallbackなし、shutdown時のrequest／workspace残量0、graceful exit 0、
service・performance level復元をPASS。rawは
`.local-artifacts/phase87/stage12/context-tail/{gfx1030,gfx1201}-width{2,3,4}/`。
最初の試験要求はstrict APIで受け付けない`max_tokens`を送ったためHTTP 400、
修正後の別入力は2 tokenでEOSとなり、どちらも受入証拠から除外した。
最終の要求は`max_completion_tokens`を使い、上記32-token境界まで到達した。
V620の公開CLIも幅3・4の固定seed反復で出力文、usage、提案・受理・拒否数、
proposal block数が一致した。全幅の`max_new_tokens=1`／`幅+1`、
`--stop idempotent`がPASSした。13追加runのrawは
`.local-artifacts/phase87/stage12/tail-gfx1030-v1/`、execution JSON SHA-256は
`ca8131239ccd2b7290b0728805c6cf207974e895af8002d0ef183d82db4503e5`。
前後のQwen serviceはstopped、GPU performance levelはauto、全processはexit 0だった。

これらと幅2の変更前後一致、GDN checkpointの両GPU数値試験を合わせ、
**WU-12Aの機能受入を完了**とする。幅2を既定のまま維持し、幅3・4は明示選択として残す。
速度による既定幅の採否はWU-12B〜Dの後に判断する。

## WU-12B: 受理率と速度の基準

`ci/matrix/mtp-bench-v2.json`は凍結済みv1 Tier A 26条件のprompt／token列をSHA-256で参照し、
fixtureを複製しない。host側の`phase87_stage12_acceptance.py`と`phase87_stage12_m4.py`は、
幅2／3／4の各proposal stepのp/q期待受理率と
`1+a₁+a₁a₂+…+a₁…aₙ`によるM4を計算する。
M3用v2 suiteも26条件から生成した。M3とM=4／5 profileは未測定。

統合レビューでは当初、host集計が誤ったprompt／token列や欠けた生成hashを受け入れ得ると分かった。
v2 manifestへPhase86凍結prefixの26条件ごとのLE-i32 token hashを固定し、v1 fixture／reference列、
M1のprompt/output prefix、M3のprompt、measured生成hash、model/binary identityを照合するよう修正した。
焦点再レビューでcorrectness blockerは解消済み。hostのfocused testは10件PASS。

幅3のM1固定列は両GPUで26/26条件PASS。R9700のraw runnerは446.0秒、V620は715.5秒で完了した。
prompt平均のa₁／a₂／a₃はV620 `0.803375／0.697129／0.605005`、
R9700 `0.802280／0.695812／0.609786`。期待確定token数はそれぞれ
`2.728724／2.728544 token/block`。これはM1の固定列期待値で、通常自由生成の受理数やTPOTを保証しない。
rawは`.local-artifacts/phase87/stage12/m1-width3-gfx1030-v1/`と`m1-width3-gfx1201-v1/`。
両runnerはexact UUID、source binary hash、service状態、performance level復元、
fallbackなしを確認した。この時点では幅2のv2基準とM3は未測定だった。

R9700幅4 M1も26/26条件PASS（runner 1878.1秒）。prompt平均のa₁／a₂／a₃／a₄は
`0.792455／0.681387／0.600119／0.502845`、期待確定token数は
`2.869609 token/block`。幅3の約2.73/blockから増えるが、実行費用を含む速度採否には使わない。
rawは`.local-artifacts/phase87/stage12/m1-width4-gfx1201-v1/`、execution JSON SHA-256は
`441b98754a6e99d4ccb5e104309b4f32849e19485fb3c4885da4172103e4af4f`。
V620幅4も26/26条件PASS（runner 2958.2秒）。prompt平均のa₁／a₂／a₃／a₄は
`0.794054／0.683990／0.597066／0.504449`、期待確定token数は
`2.875917 token/block`。rawは`.local-artifacts/phase87/stage12/m1-width4-gfx1030-v1/`、
execution JSON SHA-256は`356865124ac6c5597e8fd266d944928b31336f2cdccf22d95f0935430d987082`。
この時点ではV620幅2とM3は未測定だった。

R9700幅2のv2 M1は26/26条件PASS（runner 452.0秒）。prompt平均のa₁／a₂は
`0.807726／0.713352`、期待確定token数は`2.391801 token/block`。
WU-3P採用済みの直前binary `ea20f8db8770d00d9a37646bf3963bd3aa8f874c48466376fea06beb57414eb2`
を同じ入力、seed、NVFP4 sidecar、幅2で再実行した対照は451.8秒でPASSした。
新旧26条件すべてでprompt/output prefix、target/draft logits、target hiddenのSHA-256、
accepted数、block数、tail omissionが一致した。R9700の既定幅2は今回の機能追加後もbitwise不変。
原票は`.local-artifacts/phase87/stage12/m1-width2-gfx1201-v1/`と
`m1-width2-prestage12-gfx1201-v1/`。runner execution JSON SHA-256は順に
`e978eb62caf84870bb0f68bbfef1dc7e44b4ab4cbf0087e750ac0c135cc0ab14`／
`84bb9d6a767b507af0bb5b4618da7545ce7f2086a4b3258960f9c84509a6f3ed`。
両runともexact UUID、service復元、performance level復元を確認した。R9700のM3幅2を開始した。

V620の通常8192入力／128出力、MTP幅2、1 warmup＋3 measuredも現行Stage12 binaryでPASS。
生成SHA-256 `sha256:8950b821b2aa47de2b6621e175a583cffb10d70734b19a6234fede16b28a73d0`、
受理／提案75/103は[段階1の最終run](phase87-stage1.md)と全runで一致した。
TPOT中央値は旧30.414→新30.398 msで同程度。rawは
`.local-artifacts/phase87/stage12/full-width2-gfx1030-v1/`、report SHA-256は
`73f98accfd3d5703fdeb80986e292b06dcc1fc3e1dc534a8d0113a92da3da995`。
このrunはR9700のM3時間測定と共有host上で一部重なったため、prefill／TTFTの微小差を
この比較だけで採否しない。幅2の生成内容・受理数の回帰確認に限定する。

V620幅2のv2 M1固定列も26/26条件PASS（runner 695.7秒）。prompt平均a₁／a₂は
`0.811690／0.715820`、期待確定数は`2.400262 token/block`。rawは
`.local-artifacts/phase87/stage12/m1-width2-gfx1030-v1/`、report SHA-256は
`c02d0b27c67bf9146ffcfc62048d093b86af9d13b61fc0da649a6d2b68da6dd6`。
現行validatorで凍結prefix、NVFP4 companion、縮小head、exact GPU UUID、binary hash、
service／performance level復元まで照合した。V620幅2のM3／M4は未測定。

R9700で最初に実行したStage12 M3幅2（26条件、1431.9秒）は、raw自体はPASSだったが
`draft_vocab_sha256`が欠け、98,304語彙の既定draft headではなく全語彙headを使っていた。
原因はStage12 M3 modeを縮小headのload条件へ含め忘れたこと。hostのM3 identity validatorが
これを拒否したため、**このrawをWU-12Bの速度・採否証拠へ含めない**。
誤条件の原票は`.local-artifacts/phase87/stage12/m3-width2-gfx1201-v1/`、report SHA-256
`b90b3fae045a5482588a6ea374b1dfc5d93076c376c0e36cb256dcfaf274a2ee`。
Stage12 M1／M3では縮小headの存在をfail-closedで要求するよう修正し、同じv2条件を再実行した。

修正後のR9700幅2 M3は26/26条件、1 warmup＋3 measuredでPASSした。rawは
`.local-artifacts/phase87/stage12/m3-width2-gfx1201-v2/`、report SHA-256は
`85440383f65af715537d5a6b4240f51e8456c5f680ccc9d9d77c29825d9496f6`、
binary SHA-256は`8ab8b7e66b4c7f770746a40d586429aee0923f68ace23bfc5cd3cb9bf5286fd1`。
全条件でmeasured 3回の生成hashが一致し、NVFP4 companion digest、98,304語彙head
`24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0`、
exact GPU UUID、HIP-only、fallbackなし、service／performance level復元をhost validatorで確認した。
prompt単位の中央値を平均すると、draft `5.0373`、non-draft `56.1494`、合計
`61.1932 ms/block`。R9700幅2 M1を現行validatorで再導出し、M4はTier A 26 prompt平均
`39.1052 token/s`、prompt-cluster bootstrap 95%区間`37.7502〜40.4758 token/s`だった。
これは幅2の対照値であり、幅3・4の採否は残りのM3／M4と同一process通常AB/BAの後に決める。
M4原票は`.local-artifacts/phase87/stage12/derived/m4-width2-gfx1201-v2.json`、SHA-256は
`b0d1fb88e792c231a95eae07ba97afcc9565254caeea363c840d32224c20fb49`。

R9700幅3 M3も26/26条件、1 warmup＋3 measuredでPASS（runner 1390.0秒）。rawは
`.local-artifacts/phase87/stage12/m3-width3-gfx1201-v1/`、report SHA-256は
`d8e360247350b5cbafa4912cfc5451a5c89c349f2429c7c56d6b3c81fdc944aa`、
binary SHA-256は`9add3ddf815c42923c53900b1be8edd1fb409e83401026fdb3ba7ea197ad0a8e`。
measured生成hashは各条件の3回で一致し、NVFP4 companion、縮小head、exact UUID、
HIP-only、fallbackなし、service／performance level復元をhost validatorで確認した。
prompt単位の中央値を平均したdraft／non-draft／合計は`7.5564／60.5622／68.1191 ms/block`。
R9700幅3のM4は26 prompt平均`40.0891 token/s`、幅2に対するprompt対の相対差は
平均`+2.1608%`、bootstrap 95%区間`+0.2719〜+4.0893%`（17 promptで正、9で負）。
M4原票は`.local-artifacts/phase87/stage12/derived/m4-width3-gfx1201-v1.json`、SHA-256は
`f44c8309c7606d5b4cccf509ccf506b1b9a479591b1d00d6aff3c3d0031f0899`。
これは当時のscreeningである。後のユーザー決定により、既定幅のための追加M4と通常AB/BAは省略した。

V620幅2 M3は26/26条件、1 warmup＋3 measuredでPASS（runner 2432.3秒）。rawは
`.local-artifacts/phase87/stage12/m3-width2-gfx1030-v1/`、report SHA-256は
`17e930ff50de9a2232ab9f3f127093d4151ec273393e4a09ce4823c31d708a95`、
binary SHA-256は`9671e868ce5077f37638477288d44202d5132ef12023633e0b528e54cb7ff34b`。
全条件のmeasured生成hash、NVFP4 companion、縮小head、exact GPU UUID、HIP-only、
fallbackなし、performance level復元をhost validatorで確認した。prompt単位の中央値平均は
draft／non-draft／合計`5.9815／67.7125／73.6985 ms/block`、
M4は`32.6325 token/s`（prompt-cluster bootstrap 95%区間`31.4123〜33.8464`）。
M4原票は`.local-artifacts/phase87/stage12/derived/m4-width2-gfx1030-v1.json`、SHA-256は
`2079520100953bdd2fa0e8754a572ca4d28ddf5fda7f6b511a766b2c354bc988`。

V620幅3 M3も26/26条件、1 warmup＋3 measuredでPASS（runner 2495.8秒）。rawは
`.local-artifacts/phase87/stage12/m3-width3-gfx1030-v1/`、report SHA-256は
`9e703f9d7e940435f40e64c46f2dd6692fe7cf8b468b479011a43de277437920`。
binaryは幅2と同じ`9671e868ce5077f37638477288d44202d5132ef12023633e0b528e54cb7ff34b`。
全条件の反復生成hash、NVFP4 companion、縮小head、exact UUID、HIP-only、fallbackなし、
performance level復元をhost validatorで確認した。prompt単位の中央値平均は
draft／non-draft／合計`9.0501／82.6493／91.7277 ms/block`。M4は
`29.9876 token/s`で、幅2へのprompt対の相対差は平均`−8.5538%`、bootstrap 95%区間
`−10.6671〜−6.4108%`（3 promptで正、23で負）。M4原票は
`.local-artifacts/phase87/stage12/derived/m4-width3-gfx1030-v1.json`、SHA-256は
`1ebc68b299948eab1c743892415f127897d692402e83d1fb74a60e06df417697`。
これは当時のscreeningである。後のユーザー決定により、WU-12Dの通常AB/BAは省略した。

R9700幅4 M3は26/26条件、1 warmup＋3 measuredでPASSしたが、runnerは7055.7秒を要した。
rawは`.local-artifacts/phase87/stage12/m3-width4-gfx1201-v1/`、report SHA-256は
`7f4fd8a12e66a494b36dc8fc6b514bc2d88996b4575e6e29a5eaa87a0ae6defd`、
binary SHA-256は幅3と同じ`9add3ddf815c42923c53900b1be8edd1fb409e83401026fdb3ba7ea197ad0a8e`。
全26条件でmeasured生成hashが一致し、NVFP4 companion／縮小head、exact UUID、HIP-only、
fallbackなし、service／performance level復元をhost validatorで確認した。
prompt単位の中央値を平均したdraft／non-draft／合計は
`10.0982／689.8554／699.9695 ms/block`。width3の合計`68.1191 ms/block`と比べ、
non-draftが約11.4倍になっている。幅4 M4は`4.0992 token/s`、幅2に対するprompt対の
相対差は平均`−89.5692%`、bootstrap 95%区間`−89.8785〜−89.2459%`。
M4原票は`.local-artifacts/phase87/stage12/derived/m4-width4-gfx1201-v1.json`、SHA-256は
`c20efed535d9be192826c36656a020de3da9169c59969d9b7c5a2ea7b0ccdbea`。
幅2／3／4のscreening集計は同じ`derived/m4-width-compare-gfx1201-v1.json`、SHA-256
`b7d6cfaabeddbe90232a4840c6dff846ffa869594e5f7d29745504fc7ba1b69a`。

幅4 M3の所要時間が当初の見込みを1.5倍以上超えたため、WU-12Bの次のGPU計測は
そのまま増やさず、まず代表入力のprofileでM=5経路を切り分ける。幅4は明示選択機能として
WU-12Aで受け入れ済みだが、この未最適化のR9700結果で既定幅を4へ変えない。
V620の幅4全26条件M3とWU-12C候補の順序は、profileの結果を見て再計画する。

### R9700のM=5 profileによる原因帰属

最初の1条件profile試行は通常M3 modeの26条件必須チェックで拒否され、GPU PASSや速度証拠へ含めない。
失敗原票は`.local-artifacts/phase87/stage12/profile-wu12b/gfx1201-width4-code-rust-bugfix-zh-invalid-onecase-m3/`。
診断専用modeを別に設け、通常M3のチェックを維持した。

凍結Tier Aの`code-rust-bugfix-zh`（8,284入力／256出力）で幅2／3／4
（verify M=3／4／5）を、同じprofile専用binary `6e37f65bedf5065f1ee75a1f437b6b1283fd7e11db8b06001c84140496988644`
でROCTX decode区間に限定して計測した。各幅は1 warmup＋3 measured、生成hash・proposal block数は
26条件の通常M3の同じcaseと一致した。exact gfx1201 UUID、NVFP4 companion、縮小head、
HIP-only、fallbackなし、cleanup、service／performance level復元を確認した。
profile原票は`.local-artifacts/phase87/stage12/profile-wu12b/gfx1201-width{2,3,4}-code-rust-bugfix-zh/`。

| 同じ入力のprofile値 | 幅2／M=3 | 幅3／M=4 | 幅4／M=5 |
| --- | ---: | ---: | ---: |
| proposal block | 95 | 81 | 71 |
| decode区間のwall ms/block | 70.261 | 79.990 | 716.540 |
| GPU busy union ms/block | 57.831 | 65.917 | 699.826 |
| NVFP4 MLP行列積のdispatch/block | 168 | 168 | 168 |
| NVFP4 MLP行列積のkernel ms/block | 17.784（ID94 small-M） | 20.550（ID94 small-M） | 640.287（ID59 row8） |

同じtraceの他のkernel合計（ms/block）は、FP8行列積（GDN投影を含む）が
`22.211／22.295／32.613`、attentionが`6.094／8.654／10.603`、
target lm_headが`2.342／2.355／2.370`、draft headが`1.605／2.407／3.203`、
明示的なGDN state／convが`1.661／2.423／2.217`、token selectorが
`0.449／0.637／0.894`、activation量子化が`2.375／2.460／2.585`だった
（順に幅2／3／4）。kernel時間は並列実行で重なり得るため、wallへの単純加算はしない。

幅4では同数の168 NVFP4行列積がID94から`prefill_row8_tiled256_v1`へ切り替わり、
その増分`619.738 ms/block`がprofile wall差`636.550 ms/block`の`97.36%`を占めた。
GPU busy unionとkernel時間の合計は近く、重複計上だけで大きく見えた差ではない。
これはprofile診断領域の原因帰属であり、通常M4の速度値へ直接加算しない。
3幅を揃えた集計原票は`.local-artifacts/phase87/stage12/profile-wu12b/m3-m4-m5-gfx1201.json`、
SHA-256は`0d8a8d79f7c96d9caab4f16dc72355731722a07cd73d1385984319bb41fd400c`。
profile専用の1条件入口は通常M3の26条件チェックと別modeにし、M4採否validatorは受け付けない。

### WU-12C: M=5 NVFP4の初期候補

M=5の168 NVFP4 MLP行列積を現行row8から置き換えるため、production selectorには触れず、
既存ID94算術bodyを5行へ拡張したC1と、既存ID94の4行＋ID84の1行を順に起動するC2を
test-only probeで比較した。入力はE2M1／E4M3FNの非一様な値、両実shape
`(K,N)=(5120,17408)`／`(17408,5120)`、M=5。独立FP32 oracleは全5行と境界列を含み、
全出力finite、guard、反復一致、現行row8とのbitwise一致を確認した。

最初のCMake構成は`CMAKE_BUILD_TYPE`が空で最適化されておらず、spillが多い静的情報と
数ms〜100 ms/callの単体時間を返した。この**debug構成の速度・資源値は採否証拠から除外**し、
`CMAKE_BUILD_TYPE=Release`で両targetを別buildして取り直した。

| Release probe（ms/call中央値） | R9700 gate/up | R9700 down | V620 gate/up | V620 down |
| --- | ---: | ---: | ---: | ---: |
| 現行ID59 row8 | 3.8401 | 3.8342 | 4.7827 | 4.8218 |
| C1 ID94算術5行 | 0.1126 | 0.1136 | 0.1254 | 0.1682 |
| C2 ID94 4行＋ID84 1行 | 0.2227 | 0.2438 | 0.1594 | 0.1661 |

C1／C2とも全5回のAB-BA-ABで現行より速く、独立oracle最大差0 ULP、
現行との全出力bitwise mismatch 0、候補の反復mismatch 0、cleanup PASSだった。
C1の資源はR9700 VGPR 121／LDS 1,056 B／private 0 B／active blocks 6、
V620 VGPR 100／LDS 1,056 B／private 0 B／active blocks 4。
Release原票は`.local-artifacts/phase87/stage12/m5-probe-gfx{1201,1030}-release-v1/`、
execution JSON SHA-256はR9700
`e447448a162d5a349993422fb46ec09782a5ca76a34e084a6d0cfa2281db0c74`、
V620`99217aff812413700ea7933f3cf97818017b3f4b4bb7caf134bb829f0d1c5850`。
いずれもexact UUID、binary hash、service／performance level復元を保持した。
C1を実M=5形状へN0候補として先に統合し、end-to-endと他幅・MTPなしへの影響で採否を決める。

### 段階12だけの測定簡素化（2026-09-25ユーザー決定）

ユーザーは、機能追加と関係のない長時間ベンチマークとGPU別の既定幅を避け、
過度な速度退行を防いで段階12を早期に閉じるよう指示した。既定幅は両GPUで2、
幅3・4はopt-inとして維持する。段階10・11は元の計画範囲を維持する。
R9700のC1候補幅4 M1は26条件で凍結prompt/output prefix、target hidden、target／draft logits、
proposal block、受理数、各block reportが旧binaryとすべて一致した（runner 468.3秒）。
原票は`.local-artifacts/phase87/stage12/m1-width4-gfx1201-m5-v1/`、report SHA-256は
`1c434cfb87d75934b26639af65620bd7de0053fc5010861f122074468b999e3a`、
候補binary SHA-256は`c091dbf28c25e1d685f44e7b640b051c08c6d5d1f27b59ab1a74eb473737165e`。
強化したhost validator、exact GPU UUID、HIP-only、fallbackなし、service／performance level復元もPASS。

その後開始した同候補のR9700幅4 M3は、ユーザーの測定縮小指示を受けて9/26条件で終了した。
runnerの状態は`failed`、child exit `-15`であり、**M3／M4の採否証拠には含めない**。
原票は`.local-artifacts/phase87/stage12/m3-width4-gfx1201-m5-v1/`に保持し、
R9700 service／performance level復元、VRAM解放、port閉鎖を確認した。
後続は、既にPASSした両GPUのC1数値oracle／単体AB-BA-ABと、同じ短い64出力要求の
幅2対幅4のserver時間中央値（各幅1 warmup＋2 measured）に限定する。
幅4が幅2の2倍以下、64 token完走、NVFP4 companion一致、HIP-only、fallbackなし、
cleanup・service復元を受入線とする。残る26条件M3、幅別既定値のAB/BAは行わない。

Phase 87全体の完了判断は、この段階と段階10・11の後にユーザーが確認するまで行わない。

### WU-12C／12Dの実運用確認と段階12完了

M=5をID94へ接続した本番binaryで、同じ47-token入力・固定seed・64-token出力を
各GPUで幅2と幅4に1 warmup＋2 measuredずつ流した。時間はserverのrequest auditの
`elapsed_ns`中央値を使った。両GPUとも全要求で64 tokenを完走し、NVFP4 companion digest一致、
HIP-only、fallbackなし、request／server cleanup、serviceとperformance levelの復元を確認した。

| exact GPU | 幅2の中央値 | 幅4の中央値 | 幅4／幅2 | 受入線 |
| --- | ---: | ---: | ---: | --- |
| R9700 `gfx1201` | 5.770秒 | 6.202秒 | 1.075倍 | 2倍以下、PASS |
| V620 `gfx1030` | 7.544秒 | 9.509秒 | 1.260倍 | 2倍以下、PASS |

原票は`.local-artifacts/phase87/stage12/safety-gfx{1201,1030}-v1/execution.json`。
SHA-256はR9700 `37aa19564b940f8eeadcfabc0fa6109b49c2d70cfdda8a445cbc44dc377ae7dd`、
V620 `47dfbc7077e0bd61a8f4ae284e5e2b0a6407bcda34fd353cb365fad0c0aa7886`。
本番server binary SHA-256はR9700 `90c41394982e48adb7ddedb443263a13c5b0231dba090005880d64bd17fcadae`、
V620 `12f540caff11cf49070e23133fcb9c4aa8fbea7169a1959cf7a3c3c836fdb055`。
これは短い要求で過度な遅さを防ぐ確認であり、幅4を既定にする速度証拠ではない。
R9700の未最適化幅4の26条件M4やV620幅3の26条件M4は上記に履歴として残す。
候補幅4の残り26条件M3と、GPU別既定幅を選ぶAB/BAはユーザー決定により実施しない。

**段階12は2026-09-25に完了**。GPU共通の既定幅2を維持し、幅3・4は明示選択として残す。
段階10・11は元の計画範囲で継続し、Phase 87全体はユーザー確認まで完了扱いにしない。

計画: [Phase 87 段階12](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。
