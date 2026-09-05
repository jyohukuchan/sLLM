# 数値・出力影響変更台帳

> 正本状態: active
> 適用開始: 2026-08-18
> 決定者: ユーザー明示指示

この文書は、同じmodel/input/sampling条件でもlogit、token列、visible outputへ影響しうる実装変更を一か所で追跡する正本である。
個別Phaseのplan/historyは詳細証拠を保持し、この台帳は変更理由、数値方向、観測された出力差、承認区分、rollback identityを索引する。

## 数値変更の承認規則

token完全一致は観測項目として残すが、それ単独を数値correctnessのhard gateにしない。candidateは次の区分で扱う。

### N0: 数値・token互換

- 実数式、浮動小数点演算順、丸めstageを維持するか、固定matrixで必要なtoken/logit一致を確認した変更。
- 通常のcorrectness、性能、resource、fallback、cleanup条件を満たせば通常承認できる。

### N1: 解析的に誤差非増加または低減

次をすべて満たす変更は、変更前とtoken列が異なっても**数値変更として自動承認**する。高精度providerや全modelのFP64比較を
新しい必須gateにしない。

1. real-number semantic equationを変更しない。
2. dtype、丸めstage、入力集合、加算項等の欠落がなく、差の原因を演算順・近似式・精度昇格等へ局所化して説明できる。
3. 標準的な浮動小数点誤差解析により、対象演算のworst-case boundまたは期待誤差が非増加となる。例として、同符号128項の
   FP32逐次和をbalanced pairwise/tree和へ変える場合、依存深さは`127`から概ね`ceil(log2(128))`へ減る。
4. race、未定義動作、非決定atomic、未初期化値、silent fallbackを誤差低減として扱わない。同一providerのrepeatは再現可能である。
5. 既存のfinite、tiny numerical oracle、state publication、padding、cleanup、unsupported inputのfail-closedを満たす。
6. token/logit差を隠さず、この台帳へ最初の分岐位置、対象scope、source/provider identity、rollbackを記録する。

N1の自動承認は数値互換性gateだけに適用する。性能採用条件、security/correctness defect、resource、ABI、fallback、cleanup条件を
免除しない。semantic equation自体、量子化recipe、sampling規則、stop/usageを変える変更はN1に分類しない。

### N2: 誤差が僅かに増加

- 既存oracle tolerance内だが、解析上の誤差bound、accumulator精度、近似誤差のいずれかが僅かに悪化する変更。
- 自動承認しない。scope、速度・memory効果、誤差bound、token/logit差、品質controlを提示し、人間が採否を決定する。
- 「僅か」は既存の演算別tolerance内かつfinite/state/tokenization contractを壊さない範囲に限定する。範囲を説明できない場合はN3とする。

### N3: 不明・非有界・意味変更

- 差の原因が説明できない、誤差方向を分類できない、非決定、入力依存で非有界、またはsemantic equationを意図せず変える変更。
- 採用せずreplanする。人間承認だけでcorrectness/security defectを相殺しない。
- 高精度referenceはN1の定常gateではないが、N2/N3の分類を解消するため人間が要求した場合や、解析が曖昧な場合に限定して作成できる。

## 台帳に必須の項目

- 日付、Phase/変更ID、対象model/op/dtype/target/scope。
- baseline/candidateの式、演算順、accumulator、丸めstage、provider/source identity。
- N0/N1/N2/N3分類と解析根拠。
- tiny oracle、state/fallback/cleanup、同一provider repeatの結果。
- token/logit差の有無、最初の分岐位置、品質control。未実施項目は未実施と明記する。
- 性能・resource結果、採否、target split、rollback identity。

## 変更履歴

### OUT-2026-09-05-P78-FP8-PREFILL-LOAD64: V620 FP8 prefill協調load（N0）

- scope: exact gfx1030 ID71、M1024/K6144/N5120およびM1024/K5120/N10240。
- change: 各threadへ隣接8-byteのFP8値を割り当て、2本の32-bit loadを1本の64-bit loadへまとめる。
  low/high wordを同じ変換helperへ渡し、LDSの最終配置・half2 dot順・FP32加算・scale・BF16 RNEを維持する**N0**。
  A/Wが8-byte非整列の場合は既存full-tile body、それ以外の形状も既存経路を使う。
- correctness: 両形状4-copy全BF16出力・repeat・FP64 tile境界期待値と、A/W offset1/2/3/4/7 fallback、
  guard／finite／checked cleanupをPASS。本番matmul TUの直接compile G1でも同じ比較をPASSし、実行前後SHAは不変。
  両target release buildを確認した。gfx1201に新経路を選択する変更はない。
- performance: private copy別中央値平均は`3826.618→3569.996 us`と`6134.939→5683.695 us`。
  r25 V620長文1 warm＋3 measuredはr24と全token／文章／停止理由／audit一致、HIP-only／cleanup0。
  prefill `308.308→312.115 tok/s`、TTFT約373 ms短縮、E2E約401 ms短縮を確認したが、Phase78目標は未達。
- identity: source manifest `12d409672170b2d3bbfa5db90a8afaf4d879a7b62463972796f668c3553514c3`、
  gfx1030 archive `f30b4921c755f2e3297ab069cff1dc696c355b0b933a0f8afba8c35a26c26aa7`。
- rollback: r24のfull-tile body。既存ID71 opt-in内に限定する。
- 詳細: [Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-FP8-GDN-Z-TUPLE: V620 GDN z投影の形状定数化（N0）

- scope: exact gfx1030、Qwen3.8 GDN zのM1/K5120/N6144、既存ID82 opt-in内だけ。
- change: 既存rolled P2 tuple bodyへ形状を定数として渡し、generic bodyの境界判定・address計算を削減する。
  LUT ingress、各laneのK順、4つのdot2、FP32加算・wave reduction、scaleとBF16 RNEを維持するため**N0**。
- correctness: private 4-copy比較で全6144出力一致、各copy64境界点の独立FP64期待値、両provider repeat、
  guard／finite／checked cleanupをPASS。本番r24 archiveのGDN共有・個別matmul・Graph replay数値比較もPASS。
  source／binaryの実行前後SHA不変、native host testと両target release buildをPASS。
- performance: privateのcopy別中央値平均は`0.090251→0.081091 ms`。VGPRは57→170、active blocksは8→2だが、
  実測時間は約10.1%短縮した。実モデルの効果は別途記録し、private速度をwhole-modelへ読み替えない。
  r24短文／長文は各1 warm＋3 measuredでr23と全token／文章／停止理由／audit一致、HIP-only／cleanup 0。
  長文decode `15.402→15.412 tok/s`はMAD範囲内であり、whole-model改善を確定せずopt-in候補に留める。
- identity: source manifest `65a600a23cb5be3bc52f1ddf490f6234364ef5b7744b85a9dc7137ce843ce4c7`、
  gfx1030 archive `0bb737e39826cd17a358fff757f56c7c60ecea0eb6b46e36ce9b20bf07400a1b`。
- rollback: r23のgeneric ID82 body。既存ID82 opt-in自体の省略も従来どおり利用できる。
- 詳細: [Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-FP8-GDN-SHARED: GDN qkv/zの量子化共有（N0・opt-in）

- scope: exact Qwen3.8のGDN 48 pair、gfx1030／gfx1201、M1 K5120、N10240／6144。
- change: 同じBF16 input viewを既存FP8 quantizerで一度だけ量子化し、bytesとrow scaleを2つの既存matmulで共有する。
  member別weight scale、provider／hipBLASLt algorithm、演算順とBF16出力丸めを維持する。
- 分類: **N0**。quantizerはweight非依存であり、全Kのmax／448とE4M3FN丸めを変更しない。
  request-owned workspaceは5124 bytes、M>1は従来どおり2 matmulへ分解する。
- correctness: r23の両target G1でmember別weight・scaleの独立期待値、全BF16出力一致、入力変更後比較、
  実repeat、3-node HIP Graph capture／replay、cleanup 0をPASS。共有3 dispatch、個別合計4 dispatch。
  source／archive／binaryの実行前後SHA一致を確認した。短文17/17と長文9435/128は両target各1 warm＋3 measuredで
  baselineと全token／文章／停止理由一致、HIP-only／cleanup 0。長文decodeはV620 `15.266→15.402 tok/s`、
  R9700 `18.719→18.828 tok/s`。最終4行3 warm＋10 measuredの性能比較は未完了。
- identity: source manifest `75c0177bb01da93e1efaaa4a6944eebccd7d89840c0f5f4881573c6aa518fe53`。
  archiveはgfx1030 `402a213f60731a59b453af6985b1f4551f9103db6fd974c94cc1950ac5551d35`、
  gfx1201 `50da39102fbdd5093c8a173fa4d65d37e7e91d7a12bd148cfa2a67a743d99c71`。
- 決定: `SLLM_QWEN38_FP8_GDN_PROJECTION_PACK2=1`のopt-in候補。省略がrollbackで、既存NVFP4 opt-inとは独立。
- 詳細: [Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-V620-PIPELINE: NVFP4次stage先読み（N0）

- scope: exact gfx1030、ID62、M1024/K5120/N17408だけ。single LDSの64x64/K32 tileを維持し、
  次stageのpacked activation／weightとscaleを現在stageのdot4演算中に読む。
  rawからLDSへの変換、block16演算順、scale乗算、FP32加算、BF16 RNEを維持するため**N0**。
- private correctness/resource: tiny M32/33/65 K48/N131全点FP64、M1024 wide/down全BF16一致＋
  64点FP64、sentinel付きrepeat、finite／cleanupをPASS。VGPR86／LDS6144／scratch 0／active 5。
- performance: 4-copy／128共通prewarm／3 warm＋10交互measuredでwide
  `8229.996→7391.363 us`（MAD `8.170/44.446 us`）。downは約8.2%遅いため適用しない。
  独立global symbolとexact shape判定を接続済み。r21 production native launcher G1で
  tiny M33/65全点FP64、実wide全17,825,792出力一致＋64点FP64、sentinel付きrepeat、finite／cleanupをPASS。
  exact／neighbor／transposed形状のmetadata判定も確認した。r21長文9435/128は旧r17と全token・文章・audit一致、
  HIP-only／nonfinite 0／cleanup 0。prefill `300.074→307.808 tok/s`、TTFT `30671.365 ms`。
  decodeは`15.266 tok/s`、TPOT `65.504 ms`。1 warm＋3 measuredの探索結果で、最終目標は未達。
  旧Index32 bodyとそれ以外のrouteを維持し、rollbackはr19のgfx1030 provider。

### OUT-2026-09-05-P78-R9700-LOAD-SHARING: 短文load指定とdecode activation共有（N0・短文実モデル確認済み）

- scope: exact gfx1201。ID64はM17/K5120/N17408のpacked A/Wを通常loadへ変更し、
  128x64/K32 tile、WMMA contribution、scale乗算、FP32加算、BF16 RNEを維持する。
  M17/K17408/N5120の既存split4 partialも同じload指定変更だけを行い、分割・reduction順は維持する。
  ID84はM1の(K,N)=(5120,17408)/(17408,5120)だけactivationをLDSへ共有し、
  P4 weight先読み、dot4順、FMA、wave reduction、tensor scale、BF16 RNEを維持する。双方**N0**。
- private correctness: ID64はtiny M17K48N37と実shapeの全BF16一致、64点FP64 oracle、
  repeat／finite／cleanup 56/56をPASS。ID84修正版r4はtiny全点FP64、実2shape全BF16一致、
  control/candidate双方のrepeat／finite／cleanup 168/168をPASSした。初回r3は測定実装不備で採用証拠から除外。
- private performance: 4-copy／128共通prewarm／3 warm＋10交互measuredで、ID64実shapeは
  `928.828→659.646 us`（MAD `3.020/9.301 us`）。ID84は通常中央値で
  wide `105.401→103.781 us`、down `107.166→105.786 us`。tinyは遅いため適用しない。
- integration: 独立device symbolとexact shape分岐を接続し、r20 gfx1201 release buildをPASS。
  r20 binary SHA-256
  `af92677c86c5c6ddefb1ab9c14f2d45d32b517e26bc504cfca368f76530bbdd1`。
  ID64 wideのproduction launcher G1はtiny／実shape全BF16一致、FP64／repeat／cleanup 56/56をPASS。
  ID84のr20 archive単独linkによるnative launcher G1はtiny／neighborの旧symbolと実2shapeの新symbolを確認し、
  全BF16一致、tiny全点／実shape64点FP64、repeat／finite／cleanup 224/224をPASSした。
  split4 ordinaryの私有比較も全BF16一致、64点FP64、双方repeat／cleanup 40/40をPASSし、約33%短縮した。
  r21 split4 native launcher G1も同じcorrectnessをPASS。r21実モデル17/17は旧r19と全token・文章・audit一致、
  HIP-only／nonfinite 0／cleanup 0。TTFT `236.314→179.119 ms`、prefill `79.325→107.655 tok/s`。
  長文9435/128も旧r15と全token・文章・audit一致、HIP-only／nonfinite 0／cleanup 0。
  prefill `1150.989 tok/s`、decode `18.719 tok/s`、TPOT `53.422 ms`。
  1 warm＋3 measuredの探索結果であり、decode目標と正式反復の確認は残る。
  rollbackはr19。ID72 N2の採用保留とPhase 78の既存完了条件を維持する。

### OUT-2026-09-05-P78-REQUEST-GRAPH: 要求開始時のhostコピー削減（N0）

- scope: residentからの要求生成。既存の3種類のrewrite gateがfalseなら所有済みgraphをmoveし、
  無効な変換による深いcloneを省く。WeightLoadPlanはresident/core間で不変のArc共有とする。
  graph/layoutの検証、adapter・環境条件の評価、要求ごとのstate／queue生成、GPU演算は維持するため**N0**。
- correctness: resident関連12件とresidual fusion 1件のhost test、両target release buildをPASS。
  r18/r19の両target短文17/17は旧版と全token一致、HIP-only／nonfinite 0／cleanup 0。
- performance: r18 gfx1201はsetup `26.255→21.664 ms`、TTFT `241.619→236.740 ms`。
  r19 gfx1030はsetup `23.431 ms`、TTFT `234.249 ms`、decode `16.186 tok/s`。
  r19 gfx1201はsetup `22.027 ms`、TTFT `236.314 ms`、decode `19.560 tok/s`。
  計画共有だけの追加速度差はばらつき内である。各1 warm＋3 measuredの探索値で、
  V620のhost時間差はばらつきを含み、確定改善率は主張しない。
  r18 gfx1030途中測定は別buildと重なったためhost性能判断に使わない。
- rollback: r17の要求開始経路。r19 binary SHA-256はgfx1030
  `90cf96f3cd5fe3abaa297496e1a73f7a4143651fbbde3143964c51331d76d162`、gfx1201
  `672a0581c4d389497b30b513a9082c232b45454de052e7fa2fc37b7c134b2dcb`。

### OUT-2026-09-05-P78-FP8-DECODE-TUPLE: gfx1030 ID82の3形状専用経路（N0）

- scope: M1、(K,N)=(5120,17408)/(6144,5120)/(5120,10240)。定数K/Nとrolled loopで
  境界判定を除き、LUT、load、dot、reduction、FP32 scale、BF16 RNEの順を維持するため**N0**。
  3つの独立global symbolから共通template bodyを使い、他shapeの旧ID82 bodyは維持する。
- correctness/resource: r17 production launcherの3形状と非一致K64/N32で、全BF16 bit一致、
  独立host prefix oracle、全256 E4M3 ingress、repeat、finite、cleanupをPASS。
  公開dispatch metadataも形状に対応したsymbolを確認。専用kernelはVGPR170／LDS544 bytes／
  scratch 0／active blocks 2、旧kernelはVGPR57／active blocks 8で、資源増加を専用symbol内に限定する。
  archive SHA-256は`1d286790bbbd0149de8de3b0623a4c9598556b086689b3287b8d97705b98fee1`。
- performance: private固定候補のK6144/N5120とK5120/N10240は、各launch前に計測外で128 MiBを
  読み同期する共通条件で、中央値`169.201→133.881 us`／`229.642→197.641 us`。
  r17 G1のK5120両形状はMADが差より大きく、その速度比だけでは改善を確定しない。
  r17実モデル9435/128（1 warm＋3 measured）はr15と全生成token一致、HIP-only／nonfinite 0／cleanup 0。
  decode `14.793→15.224 tok/s`（約2.9%改善）、TPOT `67.602→65.686 ms`、
  prefill `299.134→300.074 tok/s`。Phase 78の最終性能条件は未達である。
  candidate binary SHA-256は`15b3b3ef3ed9bec265a84efd916aadb6b92e01b8002769488b2e7f3079ad52d3`。
- rollback: r15の旧ID82 body。binary SHA-256
  `9bd0f0de7293cecb9aecdadd09a2555371ec49ff82093766875a30c872b280aa`。

### OUT-2026-09-05-P78-FP8-FULL-TILE: gfx1030 ID71の境界判定削減（N0）

- scope: M1024、(K,N)=(6144,5120)/(5120,10240)だけ、既存64x64/K32 tileの境界判定を省く。
  FP8→FP16 ingress、dot2の順、FP32 scale、BF16 RNEを維持するため**N0**。
  それ以外は旧bodyを使用し、既存ID71 symbolと同じ共有メモリを使う。
- correctness/resource: r15 production launcher対private旧bodyで、実2shapeとM219の全BF16 bit一致、
  FP64 oracle、全256 E4M3 ingress、repeat、cleanupをPASS。LDS8704 bytes、VGPR52、spill 0、
  active blocks 7を確認した。最初の二重LDS案は接続前に修正している。
  archive SHA-256は`aac223c6c805113ff0011c34aa2178bc1261d279b951ed52608215b18a9c306c`。
- output/performance: G1は`3581.838→3426.796 us`／`5681.006→5398.762 us`、M219はほぼ同じ。
  r15長文9435/128はr14 Graph-onと全生成tokenが一致し、HIP-only、nonfinite 0、cleanup 0をPASS。
  prefill `296.763→299.134 tok/s`、TTFT `31820.643→31565.963 ms`。
  decodeの観測値は`14.884→14.793 tok/s`であり、decodeの改善は主張しない。
  各1 warm＋3 measuredの探索値で、Phase 78の最終条件は未達。
  rollbackはr14 binary `6df7be6c22eb327173c9f0147a86807fd153113e47b3f7ff8e0e9a7841ca065c`。

### OUT-2026-09-05-P78-GRAPH-SPAN: gfx1030 stateless decode Graph（N0・opt-in）

- scope: exact Qwen3.8 M1 decodeの同一layer内stateless区間を、requestが保持する同じprepared plan／bufferで
  captureする。初回はeager、以後は同じkernel列をreplayし、stateful演算とterminal処理は区間外へ残す。
  式、dtype、演算順、丸めstageを変えないため**N0**。capture自体は出力を実行・更新しない。
- correctness: gfx1030 M1K35N37のRMSNorm＋BF16 matmul G1で5入力のeager／graph／数値oracle bit一致、
  capture時出力不変、1,000 replay、早期解放BUSY、cleanup 0をPASSした。
  r14 archive SHA-256は`454d1bba2796fdd589c4f8476b06ffc5726c69596fd1cc2058cc7c8c04166aad`。
- output: 同一r14 binaryの17/17と9435/128 off/onで全生成tokenが一致し、最初の分岐はない。
  logical submission、kernel identity/countも一致、HIP-only、nonfinite 0、cleanup 0、各build内再現性をPASS。
  各requestは128 span／1,176 kernel nodeを保持し、replayは短文1,920回、長文16,128回だった。
  binary SHA-256は`6df7be6c22eb327173c9f0147a86807fd153113e47b3f7ff8e0e9a7841ca065c`。
- performance: 作成費用込みの短文TPOTは`62.693→63.487 ms`で約1.3%退行、長文は
  `68.094→67.186 ms`、decode `14.686→14.884 tok/s`で約1.35%改善。各1 warm＋3 measuredの探索値であり、
  Phase 78の最終性能条件は未達。長文向けopt-in候補とし、短文での速度改善は主張しない。
  rollbackは`SLLM_QWEN38_GFX1030_GRAPH_SPANS`の省略または`0`。r14時点ではR9700未検証。
- r15でgfx1201の既存prepared FP8 Lt planへ同じcapture機構を接続した。rank／algorithmは変更しない。
  M1K6144N5120でcapture時出力不変、7入力のeager bit一致、1,000 replay、cleanup 0をPASS。
  R9700の17/17 off/onも全token・logical kernel監査が一致し、HIP-only／cleanup 0を確認した。
  TPOT `51.273→51.260 ms`、decode `19.503→19.508 tok/s`で、短文の速度改善は確認できない。
  長文9435/128も全token／logical kernel監査が一致し、TPOT `54.993→53.856 ms`、decode
  `18.184→18.568 tok/s`（約2.1%改善）、HIP-only／cleanup 0を確認した。各1 warm＋3 measuredの探索値。
  長文は既存ID72条件を保ったGraphのみの比較で、ID72のN2採用判断待ちは維持する。
  opt-in／rollbackは`SLLM_QWEN38_GFX1201_GRAPH_SPANS`。
  archive SHA-256は`cb9d40ac30287911f5aceaa35b7e1f5b54b03b3123c01c3b6139f074d1942dab`。

### OUT-2026-09-05-P78-FP8-SCHEDULE: gfx1030短文tileとID82 P2先読み（N0）

- scope: ID71 FP8 outer prefillの実測済みK/N・M=2..32を32x32／32x64 tileへ変更し、
  ID82 decodeではraw activation/weightを2反復分先読みする。FP16へのexact ingress、
  各出力のdot2順、wave reduction、FP32 outer scale、BF16 RNEは維持するため**N0**。
  それ以外のprefill shapeは既存64x64を維持し、公開grid metadataも同じshape判定へ揃えた。
- correctness: 最新production launch wrapperへ再リンクし、短文M17/31/32/33とtinyの
  oracle・全BF16一致を確認。ID82 P2は全256 E4M3 code、N37/N67の境界fixture、
  全8実shape×4 weight copiesで全BF16一致、finite、反復決定性、cleanupをPASSした。
  r7 archive SHA-256は`ae5e3c362806d9ff320f79a3169c4063d9cbd5fba96f027a0c912d1132e2cf01`。
  r9では未適用だった6実shapeへ短文tileを拡張し、production wrapper対private referenceの
  11 shape（M17/31/32/33と非整列tiny）で全BF16一致・oracle・cleanupをPASSした。
  r9 archive SHA-256は`cab4f8512c40b7563e212d1c1995ddddcea5fe688c1db31881e72cc0a1b39352`。
- output/performance: v3-r7の全4行でv3-r6と生成tokenが一致し、HIP-only、terminal nonfinite 0、cleanup 0。
  長文decodeは`13.647→14.658 tok/s`、TPOTは`73.279→68.220 ms`へ改善した。
  短文tileの効果はv3-r6に含む。いずれも最終3 warm＋10 measuredではなく各1 warm＋3 measuredの探索値。
  r9の17/17はr8と生成tokenが一致し、prefill `61.497 tok/s`、TTFT `328.600→304.616 ms`。
  M33以上とwhitelist外のshapeは既存64x64へ戻す。
- source/rollback: `native/hip/src/fp8_prefill_short_m32.inc`と`matmul_kernel.hip.cpp`。
  ID82の変更前archiveは`3201f34562edbc334cb37ab4aba1ae53674ec0c848fe0a8c2e83dc168a79dce0`、
  短文tileの変更前archiveは`a1d7a769ea7268e604f8212e61b1e4ddc96f2f6fee8f91e30d6c79b250c27d9d`。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-NVFP4-SCHEDULE: gfx1030短文tileとdecode P2先読み（N0）

- scope: gfx1030 NVFP4 W4A4。ID62のM=2..32では64行tileの無効な後半32行を除き、
  ID84 decodeではweight/scaleのraw bitsを2反復分先読みする。
  block16内の整数dot、FP32 scale適用、block間加算、wave reduction、BF16丸めの順序は維持するため**N0**。
  行同士の加算は存在せず、raw bits先読みも数値変換を追加しない。
- correctness: 変更前archive `a1d7a769ea7268e604f8212e61b1e4ddc96f2f6fee8f91e30d6c79b250c27d9d`
  と短文prototypeのM17/31/32/33、wide/down全BF16一致、tiny oracleをPASSした。
  統合後archive `3201f34562edbc334cb37ab4aba1ae53674ec0c848fe0a8c2e83dc168a79dce0`
  のproduction launch wrapperへ再リンクし、M2/16/17/31/32/33・K48/N37の全点oracle、
  実shape全BF16一致、cleanup failure 0を確認。decode P2もscale/tiny oracleとwide/down全BF16一致を確認した。
  targetはV620 PCI03、ROCR index 1。実モデルtokenと性能はv3-r6で確認中。
- performance/decision: 短文prototypeのM17 wide/down時間比は`0.746／0.854`。
  M33は退行するため短文tileを選ばない。P2の既存単体加重改善は約`1.095x`であり、
  実モデル全体の改善とは扱わない。rollbackは既存64行tileと同include内のbaseline ID73 body。
- source: `native/hip/src/matmul_kernel.hip.cpp`、`native/hip/src/nvfp4_decode_scale_lut.inc`。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-FP8-SHORT-RANK: M17のhipBLASLt選択（N1・combined model checkpoint）

- 対象: gfx1201 E4M3FN、M17/K6144/N5120だけ。zero-workspace heuristicを4件要求しrank3を選ぶ。
  M1 decodeの32件要求・環境overrideは維持する。
- 解析: control 123381／candidate 123380は同じCOMPUTE32F、FP32 WMMA、GSU1、LocalSplitU1。
  ISAでFP16 accumulatorとatomicがないことを確認した。同じFP8積のFP32丸め和として、
  標準の共通上界`gamma_6143 * sum(abs(products))`を共有するN1とする。
  tight boundの改善、pointwise誤差非増加、全入力bit一致は主張しない。
- 根拠: [TensileのGSU定義](https://github.com/ROCm/rocm-libraries/blob/develop/shared/tensile/Tensile/Common.py)、
  [AMDのTensile解説](https://rocm.blogs.amd.com/artificial-intelligence/reverse-hipblaslt-tensilelite/README.html)。
  両symbolを確認したROCm 7.14 code object SHA-256は
  `dfb7d5c735a92e23e3e9fb7ee0c6dc1f350844db22a3c2e92ef8053821ec4526`。
- 検証: private probeの全BF16 outputはcontrol／CPU oracleと一致、repeat／cleanup成功。
  単体中央値`0.115561→0.091641 ms`。production rank境界host検証はPASS。
  R9700 r13の17/17実モデルは1 warm＋3 measuredでPASSし、TTFT `241.01478 ms`、
  HIP-only、nonfinite 0、cleanup成功、反復生成一致を確認した。BF16 thin／NVFP4 split4も同時に含むため、
  速度・出力への単独寄与は未分離である。r11との最初の差は15番目の出力token（20→15）。
  証拠は`.local-artifacts/phase78-resume/fp8-gfx1201-m17-k6144-n5120-run.log`。
- rollback: 当該M17 policyを従来single-result/rank0へ戻す。r13 archive SHA-256は
  `9ec38a7c76602d3c162b69d72d29e7a8efd3f36b82d25e413e8ad14e7ec51120`。

### OUT-2026-09-05-P78-LOAD-HINT: prefillの通常load（N0）

- 対象: gfx1030 NVFP4 ID62 Index32 longと、FP8 ID71 long 64x64。
- 変更: activation／weightのnontemporal load hintを通常loadへ変更する。
  FP8はpacked/scalar計4箇所、NVFP4はpacked 2箇所。算術・codec・scale・tile・index幅は維持する。
- 検証: NVFP4は固定r10/r11の別binaryによる7 shape全BF16 dumpが一致し、tiny oracleをPASS。
  FP8は旧production対private候補、新production対同じprivate候補で7 shapeの全BF16一致とoracleをPASS。
  r11のV620全4行はr10と生成token一致、HIP-only、nonfinite 0、cleanup成功。
- identity: r11 gfx1030 archive
  `7e3e2720f7f238a2df90161fdeb7df572251d24254a7617fc26a77227df2913a`。
  証拠は`.local-artifacts/phase78-resume/nvfp4-index32-r10-r11-dump-summary.txt`と
  `fp8-gfx1030-half2-64x64-no-nt-r11-summary.txt`。
- 性能: r11探索測定の長文prefillは`283.231→296.387 tok/s`。最終性能条件は未達。
  短文NVFP4と特定shapeのFP8 decodeへ広げたr12はG1・生成token一致をPASSしたが、
  実モデル改善を確認できず採用を取り消した。このr11の採用範囲には含めない。

### OUT-2026-09-05-P78-NVFP4-INDEX32: 安全な32-bit offset（N0）

- 対象: gfx1030 ID62、M33以上かつ全tensor extentが32-bit内に収まるshape。
  端数tileのoffset余裕もguardし、対象外は従来64-bit kernelを使用する。
- 変更: offset/index幅だけを縮小し、tile、codec、scale、積和順、BF16変換を維持する。
- 検証: guard境界host検証、production GPUのM32/33/65・K48/N131 oracleと
  4実shapeの全output bit一致をPASS。r10全4行の生成tokenはr8と一致。
- identity: gfx1030 r10 archive
  `e7e6ee6e62651d81933e7fe494849ef0b68d7229958d0a2c083d8a2e714d527d`。
  証拠は`.local-artifacts/phase78-resume/nvfp4-index32-r10-production-g1*`。

### OUT-2026-09-05-P78-GDN-REGISTER: 短文recurrent stateの保持（N0）

- 対象: gfx1030/gfx1201、Qwen qk16/value48/head128、M2..32。
- 変更: 各threadのFP32 stateをtoken間でregister arrayへ保持する。
  scalar演算、reduction、物理state index、BF16出力変換は変更しない。
- 検証: 両GPUのproduction launchでM2/17/32/33 × zero/nonzero、全786,432 FP32 stateと
  BF16 outputがgeneric referenceとbit一致、cleanup成功。M33のgeneric分岐も確認した。
  r10のV620全4行とR9700短文で生成token一致、HIP-only、nonfinite 0、cleanup成功。
- identity: gfx1030 archiveは上記r10、gfx1201は
  `70f6c56ce2b0b8dd35cda5678bb6a0133abeebf441bb0672f9318b3ccf4acb84`。
  証拠は`.local-artifacts/phase78-resume/linear-gdn-register-state-r10-production-g1-summary.md`。
- 性能: 単体M17はV620約4.15倍、R9700約2.9倍。実モデル短文TTFTは
  V620 `285.236 ms`、R9700 `247.610 ms`。探索測定であり最終速度ゲートは未達。

### OUT-2026-09-05-P78-NVFP4-QUARTER: ID62のscale乗算移動（N0）

- scope: NVFP4 DP4A prefillのactivation scaleをLDSへ置く際に`0.25F`を掛け、
  各outputのblock sum側にあった同じ係数を除く。整数dot、weight scale、block間加算、BF16丸めは維持する。
  block16の整数dot絶対値は2304以下で、有限E4M3 scaleとの最初の積はFP32で厳密表現できる。
  二進の係数移動で後段のoperandは変わらないため**N0**。
- correctness: dot範囲と全256 E4M3符号の1,179,904組をhostで確認し、非NaN値のbit不一致0。
  両側NaNは同値扱いで、NaN payloadの一致はこの証明の対象外とする。
  V620 PCI43ではproduction archiveと旧演算のprivate referenceを比較し、M65/K48/N131の全点oracle、
  M128/1024のwide/down全BF16一致をPASSした。旧archiveとprivate referenceの比較もPASSし、
  scale loadのcache修飾とE4M3 decodeを揃えている。
  gfx1201はstrict compile-onlyをPASSした。これはgfx1201 GPU実行の証拠とは扱わない。
- identity: r8 gfx1030 archive SHA-256は
  `063d1001642b67882f15340e8b632b8615aec51a932054fc8b9b7d0e97974778`。
  ローカル証拠は`.local-artifacts/phase78-resume/nvfp4-quarter-g1-{old,r8}.log`。
- performance: 実shapeの単体時間は同時比較の旧演算から約3〜4%短縮した。
  v3-r8の全4行・各1 warm＋3 measuredでr7の生成tokenと一致、HIP-only、nonfinite 0、cleanup 0。
  長文prefillは`275.307→279.797 tok/s`へ約1.6%改善したが目標未達。decodeは`14.684 tok/s`でほぼ横ばい。
  rollbackはactivation scale stagingの係数を外し、block sum側へ戻す。

### OUT-2026-09-05-P78-P64-LDS: P64 attentionのFP16 LDSとnative変換（N0）

- scope: gfx1030/gfx1201のQwen3.8 GQA6 FP16 KV decode、partition 64、tile16。
  K/VをFP32へ変換してからLDSへ置く処理を、元のFP16 bitsをLDSへ置き利用時に同じFP32値へ変換する処理へ変更。
  FP16の有限値・subnormal・signed zeroはFP32で厳密表現できる。Inf/NaNは既存の符号・payload変換を明示保持する。
  partition、QK加算、softmax、V加算、merge、BF16丸めの順序は不変なので**N0**とする。
- correctness: native変換は両targetの全65,536 FP16符号で独立host oracleとbitwise一致。
  P64 prototypeはL=8191/8192/8193/9435×2 seedでproduction controlとの全BF16一致、
  FP64 stable-softmax oracle、repeat、nonfinite確認をPASSした。
- performance/resource: LDSは32 KiBから16 KiB、workspaceは不変。
  gfx1201 prototypeの8条件合計時間比は`0.820`。gfx1030のproduction実モデルv3-r5は
  9435/128の生成tokenがcontrolと一致し、decode `13.700 tok/s`、runtime cleanup成功。
  gfx1201もproduction再リンクG1をPASSし、v3-r6長文decodeは`17.607→18.092 tok/s`へ改善、
  同一行の生成token一致とruntime cleanup成功を確認した。単体の比率を全体改善とは扱わない。
- identity/rollback: sourceは`native/hip/src/causal_attention_kernel.hip.cpp`。
  gfx1201の変更前archive SHA-256は`9fa32e05bdfd764c96ee7085e023771068a60d39871adce0a1b5a290e0f55612`。
  rollbackは同関数のFP32 LDS branchであり、P128へ切り替えない。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-P128: gfx1030 GQA6 decode partition変更（N2・採用保留）

- scope: exact `gfx1030`、Qwen3.8 GQA6（query 24／KV 4 head、head dimension 256）、FP16 KV decode。
  P64からP128へpartitionを増やし、同じsoftmax／weighted V式のFP32部分和とmerge順序が変わる。
- classification: **N2**。現行production archiveへリンクした独立FP64 stable-softmax oracleで、
  L=8191/8192/8193/9435とseed 0/7919の8条件を検証した。P64はFP64 oracleのBF16丸めと全点一致、
  P128は合計4点で最大1 BF16 ULP差があった。例えばL8193／seed0のL2誤差は
  `9.752086430e-3→9.752088890e-3`、最大絶対誤差は`2.441160609e-4→2.441651891e-4`となった。
  この有限な観測誤差をworst-case上限の証明や実文章品質の判定へ一般化しない。
- numerical/resource: 両providerとも8条件の全反復がbitwise決定的、oracle／actual nonfinite 0、probe exit 0。
  targetはV620 PCI `0000:03:00.0`。archive SHA-256
  `af30b49d4dd7f813f3a91f303eab3a4e92a04499a57598cd1208e531ff0d3707`、probe SHA-256
  `29e305843234c944ddd7ea352f343f834932b1b1825454eb8f64d2367dfa7c1b`。
- output/decision: 既存長文探索ではP128だけを外すと5個目からの生成token分岐が解消した。
  P128は採用保留を維持し、Phase 78の継続測定にはP64を使う。P128採用をPhase完了の前提にしない。
  rollbackは`SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128`を外し、P64 opt-inを維持する。
- evidence: `.local-artifacts/phase78-resume/p64-p128-current-pci03-*`。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-ID72: gfx1201 NVFP4のFP16 staging（N2・判断待ち）

- scope: exact `gfx1201`、固定Qwen3.8-27B mixed-NVFP4のW4A4 prefill。
  ID64のblock16 WMMA＋FP32 scale適用から、ID72のexact FP16展開＋full-K FP32 GEMMへ切り替える。
  E2M1とE4M3 scaleの積はFP16で正確に表現できるが、積和の順序は変わる。
- classification: **N2**。生成tokenの変化に加え、独立FP64 scalar oracleで誤差増加を確認したためN1とはしない。
  M=17/127/128/129/219、K/N=5120/17408と17408/5120、rocBLAS solution 0/136321を調べた。
  positive finite E4M3の全127 codeと符号付きE2M1から合成入力を作り、各caseのBF16不一致点の先頭256点をFP64と比較した。
  例えばsolution 136321のwide形状では、確認点の最大`abs(error)/sum(abs(terms))`が
  ID64の`0.000748140701`からID72の`0.000748232564`へ増加した。既存oracleの許容範囲内だが、
  この差分点を選んだ標本は全出力の平均誤差や実文章品質の評価ではなく、worst-case上限の証明でもない。
- performance/output: 端数219-token chunkをID64で補完した1 warmup＋3 measuredの長文prefillは
  `522.150→1134.906 tok/s`、HIP-only／fallback 0／cleanup 0。benchmark v2の長文生成hashは
  `03a85b481c6a52b8bc029f3959a9b42987a46697a1b048e292ced4944984adf3`から
  `628ad99500e71de824a1e06fb918ec03a400587f7f9337ecc3225c07cbb591fb`へ変化した。
- decision: 2026-09-05に、この具体的な速度・丸め差を示してユーザー判断を依頼した。回答待ちで既定採用しない。
  rollbackは`SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING`を外し、既存ID64を選ぶ。
- evidence: Git管理外`.local-artifacts/phase78-resume/id72-short-shapes.cpp`／同`.log`、
  `r9700-id72-tail.json`。詳細と最終採否は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-BF16-GDN-THIN: M17 GDN薄型投影のrow-wise reduction（N1・production G1 PASS、combined model checkpoint）

- scope: exact `gfx1030`／`gfx1201`、Qwen GDN thin projectionの`M=17,K=5120,N=48`だけを対象とする。
  `K=70/71,N=47`のtiny確認はguard外の既存tiled16経路であり、新providerのproduction scopeには含めない。
- baseline/candidate: baselineは既存`matmul.bf16_fp32.tiled16.v2`の16x16 tile内K逐次FP32 accumulation。
  candidateは既存`matmul_bf16_decode_body<32U,8U>`を各`(row,column)` blockへ呼ぶ`grid=(N,M)` providerである。
  BF16 input、paired load、各FP32 product、wave32 tree、8-wave fixed tree、BF16 RNE outputを維持し、runtime ID／public ABIは追加しない。
- classification: **N1**。`K=5120`のcandidateは各threadが10 paired loads（20 product terms）を処理し、局所加算深さ19、wave32 tree深さ5、8 partialの有効な最終tree深さ3となる。
  構造上の深さは`27`、zero laneを保守的に含めても`<=29`で、baselineのK逐次深さ`5119`より小さい。
  同じ入力項、BF16→FP32 product、FP32 accumulator、BF16 RNE stageを維持し、race、atomic、未初期化値、fallbackはない。
  このboundはworst-case誤差上界の非増加を示すもので、pointwise非増加や全入力bit一致は主張しない。
- correctness/output: r13 archiveのpublic `PrefillTiled16` launcherを使ったproduction G1をV620 PCI `0000:03:00.0`とR9700 PCI `0000:07:00.0`で実行した。
  tiny K70/71を含む両target全shapeで独立FP64 oracle mismatch 0、production/reference BF16 diff 0、repeat mismatch 0、finite、cleanupをPASSした。
  実modelの個別kernel帰属は未分離であり、combined checkpointを下記へ記録する。
- performance/resource: common prewarm 128、warmup 3、measured 10のmean event時間は、V620が
  `0.190954→0.016484 ms`、R9700が`0.178870→0.017640 ms`（private tiled16 reference→production row-wise）だった。
  これは単体meanであり、実model速度の単独帰属ではない。
- combined model checkpoint: V620 r13（BF16 thinとNVFP4 split4を同時に含む）17/17は1 warm＋3 measuredで
  prefill `79.749667 tok/s`、`213.167033 ms`、TTFT `240.039708 ms`（r11 `281.070 ms`）、decode `15.826946 tok/s`だった。
  HIP fallback 0、nonfinite 0、cleanup 0、deterministic true。r11との差はzero-based 14（15番目、`15→20`）から分岐した。
  R9700 r13も17/17 PASS、prefill `79.014116 tok/s`、`215.151429 ms`、TTFT `241.01478 ms`（r11 `244.851 ms`）、decode `19.538382 tok/s`だった。
  RもHIP fallback 0、nonfinite 0、cleanup 0、deterministic trueで、r11との差はzero-based 14（15番目、`20→15`）から分岐した。
  V/Rは同じ位置で逆向きのtoken差になった。この分岐をBF16 thin単体またはsplit4単体へ帰属させず、両targetのcombined N1 output effectとして記録する。
- source/rollback/identity: sourceは`native/hip/src/matmul_kernel.hip.cpp`。rollbackはexact shape guardを外して既存tiled16へ戻す。
  r13 gfx1030 archive SHA-256は`9bbec6f8cd40c13934768037c90adb6320f25c4b3a7dba8da0eed9d75e1d0dd7`、
  gfx1201は`9ec38a7c76602d3c162b69d72d29e7a8efd3f36b82d25e413e8ad14e7ec51120`。
  evidenceは`.local-artifacts/phase78-resume/bf16-gdn-thin-production-g1-summary.md`と同`*-run-gfx1030.log`／`*-run-gfx1201.log`。

### OUT-2026-09-05-P78-NVFP4-SPLIT4: M17 stage split-K=4 reduction（N1・production G1 PASS、combined model checkpoint）

- scope: exact `gfx1030`ではID62の`M=17,K=5120,N=17408` wideと`M=17,K=17408,N=5120` down、
  exact `gfx1201`ではID64の`M=17,K=17408,N=5120` downだけを対象とする。tiny `M=17,K=48,N=37`とR9700 wideはguardで拒否する。
- baseline/candidate: scaled block16 terms、E2M1→E4M3 ingress、FP32 partial、tensor scale、BF16 RNEを維持する。
  candidateはK stageを4 contiguous partitionsへ分け、`p0,p1,p2,p3`を固定順でpartial-reduceし、最後にtensor scaleを一度だけ適用してBF16 RNEする。
- classification: **N1**。`B=K/16`、`S=ceil(B/2)`とすると、K5120は各partition 40 stage／80 termsで
  `79+3=82`、K17408は136 stage／272 termsで`271+3=274`のcandidate加算深さとなる。
  unsplit baselineはそれぞれ`B-1=319`／`1088-1=1087`、tiny K48はcandidate／baselineとも`2`である。
  標準boundは`gamma_82`対`gamma_319`、`gamma_274`対`gamma_1087`で非増加だが、pointwise非増加やbitwise equalityは主張しない。
- correctness/output: production G1はV620 PCI `0000:03:00.0`とR9700 PCI `0000:07:00.0`でPASSした。
  V620 wide/downはBF16 diff 0、FP64 oracle、repeat、nonfinite、cleanupをPASSした。R9700 downはBF16 diff 30をN1 report-onlyとし、
  control/candidateともFP64 oracle mismatch 0（max normalized error `0.0001631567`）、repeat、finite、NaN fixture（両方5120 nonfinite）、
  cleanup 16 alloc/freeをPASSした。bit diff 30は誤差boundを無効にするpointwise主張へ昇格しない。
- performance/resource: V620 wideは`401.064→392.204 us`、downは`758.727→499.205 us`、R9700 downは
  `1024.648→905.488 us`（各control→split4 production G1）。V620 split4はVGPR54／LDS4608／scratch0／active blocks 8、
  R9700 split4はVGPR97／LDS7680／scratch0／active blocks/CU 7、active waves/CU 56、occupancy 0.875だった。
  V620/R9700のcombined r13 model checkpointは上記BF16 thin entryに記録した。BF16 thin／split4のどちらがtoken分岐へ寄与したかは未分離であり、
  combined N1 output effectとして扱う。この単体結果だけで個別candidateのmodel帰属や性能採用完了とはしない。
- source/rollback/evidence: split4は既存ID62/ID64のexact shape guard内に限定し、scope外は従来providerへ戻す。
  production G1 evidenceは`.local-artifacts/phase78-resume/nvfp4-gfx1030-split4-production-g1-pci03.log`と
  `nvfp4-gfx1201-split4-production-g1-pci07.log`。tiny拒否、target identity、固定partial reduction、scale一回適用を同ログで確認した。

### OUT-2026-09-02-P73-GFX1201-MXFP8-WIDE-N: production selectorのN<=32768拡張（N1）

- scope: exact `gfx1201`、OCP MXFP8 E4M3 W8A8 block32/E8M0 prefill。ID31／34／36／37のN上限だけを
  16,384から32,768へ広げ、既存M/K、64／128列alignment、他target／format／decodeを維持する。
- arithmetic/classification: kernel、量子化recipe、E4M3 value、E8M0 scale、FP32 accumulation、BF16 RNEは変更しない。
  row8から既存WMMA treeへ選択が変わり得るため、Phase 63と同じ**N1**として扱う。
- verification/decision: N=17,408／32,000／32,768のhost selection、32,769／32,832のfallback、prepared providerと
  既存gfx1201 codec/provider testをPASSした。ユーザー指示により新規範囲のGPU数値oracle、生成token、性能測定は省略し、
  未実施を明記したうえでselector scopeを採用する。
- rollback: 3 predicateの上限を16,384へ戻す。
- details: [Phase 73履歴](../history/2026/09/1-10/phase73-gfx1201-mxfp8-wide-n-selector.md)と
  [追跡要約](../../ci/matrix/phase73-gfx1201-mxfp8-wide-n-selector-v1.json)。

### OUT-2026-09-02-P72-GFX1201-MXFP6-WIDE-N: ID45のN<=32768拡張（N1）

- scope: exact `gfx1201`、OCP MXFP6 E3M2 W6A6 block32/E8M0 prefill matmul。Phase 70 ID45のN上限だけを
  16,384から32,768へ広げる。M>=17、K>=2048、K%32=0、N>=1024、他target／format／decodeのcomplementは維持する。
- baseline/candidate: baselineはN>16,384をID25 tiled16へ戻す。candidateは既存ID45のE3M2→E4M3 exact ingress、
  K16 FP8 WMMA×2、E8M0 scale、block間FP32 accumulation、BF16 RNEをそのまま広幅Nへ選ぶ。量子化recipeやkernel算術は変更しない。
- classification: **N1**。Phase 70と同じ固定WMMA treeへのprovider切替であり、real-number equation、入力集合、dtype、
  scale、accumulator、丸めstageを維持する。測定8 shapeではID25／ID29とBF16 digestまで一致したが、全入力bit一致とは主張しない。
- correctness/output: N=16,384/16,385/17,408/17,409/24,576/32,000/32,767/32,768の各45 sampled pointを
  独立FP32 oracleでPASSした。最大相対誤差`0.0036457598`、非有限不一致0、各5 row top-1とoutput digestは両controlに一致、
  repeat不一致0。Qwen3.5-27Bの4 sampleも生成`[23066,23066,23066,23066]`で一致した。
- performance/resource/decision: ID45はID25比`3.0731〜10.6190x`。強制指定なしのQwen3.5-27B 512-token prefillは
  旧既定`81.746517`から`383.170165 tok/s`へ4.6873倍となり、resident／peakは`24,115,002,880 / 24,777,018,880` byte、
  HIP-only、fallback／cleanup 0だった。検証結果に基づきN<=32,768を採用した。
- rollback: selector上限を16,384へ戻す。運用上のcontrolは`SLLM_MXFP6_PREFILL_FORCE_TILED16=1`。N=32,769以上は
  現状もID25へfail closedに戻る。
- details: [Phase 72履歴](../history/2026/09/1-10/phase72-gfx1201-mxfp6-wide-n-selector.md)と
  [追跡要約](../../ci/matrix/phase72-gfx1201-mxfp6-wide-n-selector-v1.json)。

### OUT-2026-09-02-P70-RDNA-MXFP6-VIA-E4M3: packed E3M2 ingressとgfx1201 WMMA（N0/N1）

- scope: OCP MXFP6 E3M2 W6A6 block32/E8M0のprefill matmul。exact `gfx1030`のID43は明示benchmark専用、
  exact `gfx1201`のID45は`M>=17`、`K>=2048`、`K%32=0`、`1024<=N<=16384`へ限定採用する。decode M=1、
  scope外shape、別target、KV default、量子化recipe、GGUF encoding、sampling、stop/usageは変更しない。
- baseline/candidate: ID43はID29 col8のrow/column/K分解、E8M0 scale、FP32 accumulation、wave reduction、BF16 RNEを維持し、
  packed E3M2を実数値exactなE4M3FN bitへ変換して既存E4 decodeへ渡す。gfx1201 ID44/45は同じvalue/scale byteから
  K32 tileだけをE4M3へmaterializeし、K16 FP8xFP8-to-FP32 WMMAを2回、scale pair、block間FP32 accumulation、BF16 RNEの
  固定treeで処理する。ID45はID44と同じ算術treeのまま、同じ3-byte groupの4値を一括変換・32-bit LDS storeする。
- classification: ID43とID44→ID45は**N0**。E3M2→E4M3FNは全64 codeで実数値exactで、ID43はID29とのBF16 digest、
  ID44/45は相互の演算順と丸めstageを維持する。従来ID29→gfx1201 WMMA familyは**N1**。実数式、入力項、scale、FP32
  accumulator、BF16 RNEを維持し、差を固定K16 WMMA treeへ局所化できる。逐次K32 dotより加算依存深さを増やさず、race、
  atomic、未初期化値、silent fallbackを使用しない。
- oracle/state: 全64 E3M2 code×4 packed laneをexact `gfx1030`／`gfx1201` device oracleでbit exactに確認した。ID43は
  production 5 shapeでID29 digest一致。ID44/45/46は独立FP32 oracle、非有限位置一致、repeat determinismをPASSし、
  P70-Fの最大相対誤差は`0.003875792259350419`だった。selector境界、prepare freeze、別target非選択、HIP-only、
  fallback false、cleanup 0も確認した。
- output/quality: 固定Qwen3.5-4B MXFP6、FP16 KV、512／2,048 inputのID44／45全sampleで生成tokenは
  `[23066,23066,23066,23066]`だった。旧providerとWMMA familyのlarge-M operator BF16 digestは異なる。full-model logitの
  最初の差、top-1、KLD、perplexityは未収集であるが、recipe不変のN1 arithmetic変更なので旧KV default用`0.99` gateは適用しない。
- performance/resource/decision: exact gfx1201のID44→ID45は3 warmup＋10 measuredで512 input
  `1276.494→2157.868 tok/s`（1.690倍）、2,048 input`1506.933→2423.308 tok/s`（1.608倍）。ID45はLDS 6,912 byte、
  SGPR/VGPR 38/115、spill/private 0でshape限定採用した。N128 ID46はVGPR 167かつ両full-model行でID45より遅く
  benchmark-only。gfx1030 ID43も512／2,048で約22.7%／21.6%遅くbenchmark-onlyとした。
- rollback: ID44は`SLLM_MXFP6_PREFILL_FORCE_PHASE70=gfx1201-n64`、従来tiled16は
  `SLLM_MXFP6_PREFILL_FORCE_TILED16=1`。scope外は従来providerを維持する。
- details: [Phase 70履歴](../history/2026/09/1-10/phase70-rdna-mxfp6-mxfp8-path-reuse.md)と
  [追跡要約](../../ci/matrix/phase70-rdna-mxfp6-mxfp8-path-reuse-v1.json)。

### OUT-2026-09-01-P67-GFX1030-MXFP8-MMQ: staged col8 scoped default（N0）

- scope: exact `gfx1030`、OCP MXFP8 E4M3 W8A8 block32/E8M0のprefill matmul。`M>=128, K>=2048, K%32=0`かつ
  `2560<=N<=16384`または`M>=512 && N==1024`だけ既存ID27 col8を選ぶ。短M、M<512のN=1024、未計測N、語彙head、別target、decode M=1は
  既存providerを維持する。量子化recipe、KV default、sampling、stop/usageは変更しない。
- change: gfx1201 ID37のN方向再利用をsoftware-decode gfx1030へ転用し、ID38 col16／ID39 col32を評価した。
  両候補は既存ID27をfull-modelで上回らず明示benchmark-onlyとし、追加shape sweepで境界を確定したID27だけを限定採用した。
- classification: **N0**。ID22/27/38/39は各outputのMXFP8 value／E8M0 scale、FP32 accumulator、加算順、wave reduction、
  BF16 RNE stageを維持する。18 case×10回と追加M=`512/2048` caseのBF16 output digestはprovider間で一致した。
- oracle/state: K=`31/32/33` admission、M=1、M=`17/127/128/129/512/2048`、N tail、N=`32/1024/2560/4096/8192/9216`、
  target非選択、override優先順位、prepare-time freezeをhost／exact gfx1030 GPUで確認した。HIP-only、fallback false、cleanup 0である。
- output/quality: 固定Qwen3.5-4B、FP16 KV、512／2,048 inputの全sampleで生成tokenは
  `[23066,23066,23066,23066]`だった。operatorでbit一致しておりrecipe不変のN0なので、KV形式変更用top-1 `0.99` gateは起動しない。
- performance/resource/decision: 同一最終binaryでrow8→scoped defaultは512 inputが
  `72.1830 -> 207.6111 tok/s`（2.8762x）、2,048 inputが`71.2428 -> 208.2710 tok/s`（2.9234x）。
  resident／peakは不変。ID27/38/39はLDS `8,704/17,152/34,048` byte、VGPR `46/42/83`、spill 0である。
- rollback: `SLLM_MXFP8_PREFILL_FORCE_ROW8=1`。scope外は既存row8、ID38/39は明示overrideだけである。
- details: [Phase 67履歴](../history/2026/09/1-10/phase67-gfx1030-mxfp8-tile-transfer.md)と
  [追跡要約](../../ci/matrix/phase67-gfx1030-mxfp8-tile-transfer-v1.json)。

### OUT-2026-09-01-P66-GFX1201-LOWP-PROVIDER: N128 matrixとtyped provider移植（N0）

- scope: exact `gfx1201`のMXFP8 E4M3 W8A8 ID37、MXFP6 E3M2 W6A6、NVFP4 W4A16／W4A4、
  MXFP4 W4A4 prepared routing、およびFP16／MXFP8 E4 KVのtyped causal-attention候補。量子化recipe、GGUF encoding、
  weight/KV default、sampling、stop/usageは変更しない。
- change: ID36の各output列と同じFP32 arithmetic treeをN128 tileへ広げるID37を追加し、format/block/layout／activation pack／
  tile／inner productをprepare時にfreezeする共通providerへ各形式を接続した。attentionはq4k4／q4k8／q8k8のtyped loadを追加した。
  NVFP4／MXFP4 W4A4は既存device kernelへのprovider routing移植であり、数値式の異なる別candidate kernelではない。
- classification: **N0**。ID37は独立output列の同時処理数だけを変え、各outputの項、scale、FP32 accumulator、加算tree、
  BF16 RNE stageを維持する。attention control/candidateの全output digestと最大absolute error 0、matrix ID36/37のBF16 digest、
  NVFP4／MXFP4 routingのoracleが一致した。provider freeze自体はdevice arithmeticを変更しない。
- oracle/state: MXFP8はM=`127/128/129`、N=`64/127/128/129/256/512/1024`とproduction shapeを実行し、
  K31／33は期待どおりhost rejection、K32はGPU受理を確認した。attentionはFP16／MXFP8 KVのM=`128/512/2048`、
  MXFP6／NVFP4／MXFP4は形式ごとの非整列blockとM>1をexact gfx1201で実行した。
  全採用evidenceはHIP-only、fallback false、cleanup 0である。
- output/quality: 同一入力の数値差、token/logit分岐は観測していない。N0かつquantization recipe不変なので旧KV default用
  top-1 `0.99` gateを起動しない。MXFP8 full-modelの4 outputは全run `[23066,23066,23066,23066]`だった。
- performance/resource/decision: ID37はwide/down operatorをID36比14.45%／6.27%短縮し、exact gfx1201、Phase 65 family、
  N%128=0へ限定採用した。LDS 1,024 byte、SGPR/VGPR 40/164、spill 0、wave32、WMMA 16命令である。
  attention候補は全primary rowで4.3〜27.3%遅くproduction不採用。MXFP6、NVFP4、MXFP4は形式別の既存kernel／fallbackを維持し、
  MXFP4 full MoE productionはscope外とした。
- rollback: ID37 scope外はID36／既存provider、attentionは既存q4k1等、各低精度形式は従来device kernelである。
  persistent BF16/FP32 weight展開、FP32 attention/KV planeは追加しない。
- details: [Phase 66履歴](../history/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md)と
  [追跡要約](../../ci/matrix/phase66-gfx1201-low-precision-provider-summary-v1.json)。

### OUT-2026-09-01-P63-GFX1201-MXFP8-WMMA: 大規模prefill WMMA provider（N1）

- scope: exact `gfx1201`、OCP MXFP8 E4M3 W8A8 block32/E8M0、M>=128、K>=2,048、1,024<=N<=16,384、
  K%32=0、N%64=0のprefill matmul。M=1、N=32／LM head、`gfx1030`、`gfx942`、未知targetは既存providerを維持する。
- change: resident value/scaleを直接読み、従来row8のK32逐次FP32 dotを、N64 tileあたり8個のK16
  FP8xFP8-to-FP32 WMMA、block scale適用、block間FP32 accumulationへ変更する。入力E4M3/E8M0 byte、実数式、項、scale、
  FP32 accumulator、BF16 RNE outputは同一である。K32ごとのFP32 contribution LDS store/readは行わない。
- classification: **N1**。固定K16 treeを2個使う加算依存深さは従来のK32逐次FP32和より増えず、term/scale/dtype/rounding stageを
  欠落させない。差はmatrix instruction内を含む固定FP32 treeへ局所化でき、race、atomic、未初期化、fallbackではない。
- oracle: exact gfx1201でcandidate 7 case／21 submissionを3 repeatし、M=`127/128/129`、wide/down/output、N=1,024、
  N=32／M=1非選択を確認した。最大production相対誤差は`0.0036960265`。special E4M3/E8M0 byteと13 oracle点は
  非有限4/4一致、mismatch 0、relative `0.0004885198`、repeat digest一致、HIP-only、fallback false、cleanup 0だった。
- output: 同一MXFP8 artifactのrow8/candidate 10 case／20 rowはtop-1 `19/20=0.95`、KLD mean `0.0029974001`、
  p99/max `0.0153089212`、perplexity相対差`-0.516618%`。最初のlogit差は`b255` prefill position 254 index 0
  `6.25→6.21875`、最初のtoken差は`b511` decode position 511 `13→220`だった。旧KV default判定のtop-1 `0.99` gateは適用しない。
- performance/resource/decision: 3+10の512/1,024/2,048/4,096 prefill中央値は
  `1,727.595/1,814.619/1,722.844/1,588.366 tok/s`。model residentは従来と同一で、persistent workspaceは追加しない。
  exact gfx1201の上記shapeだけへscoped default採用し、明示row8およびscope外row8をrollbackとする。
- details: [Phase 63履歴](../history/2026/09/1-10/phase63-gfx1201-mxfp-matrix-prefill.md)と
  [追跡要約](../../ci/matrix/phase63-gfx1201-mxfp8-wmma-prefill-v2.json)。

### OUT-2026-08-31-P62-LOWP-CODEC: 共通low-precision codecと起動境界specialization（N0）

- scope: MXFP8 E4M3／MXFP6 E3M2 W/A matmul、MXFP8 E4/E5 KV append/attention、NVFP4の共通scalar/block read/write、
  exact `gfx1030`／`gfx1201`。数値recipe、accumulator、丸めstage、public encoding、defaultは変更しない。
- change: E4M3FN/FNUZ、E5M2、E3M2、E2M1、E8M0とMX block 32／NV block 16を共通device-inline codecへ抽出し、
  attentionのruntime encoding switchをgeneric、decode wave、GQA shared/qtile、scaled long-prefillのkernel起動境界へ移した。
- classification: **N0**。W/A value/scale byteとM=`1/3/17`の6 BF16 output hash、KV append byte、29-case attention output hashを
  beforeとbit exactに維持した。実数式、FP32 accumulation、BF16 RNE、OCP scale/packing、NVFP4 outer scaleは同一である。
- oracle: 両GPUでdecode 1,104 code、encodeのzero/subnormal/tie/max/Inf/NaN境界、MX `31/32/33/256`、
  NV `15/16/17/256`を独立host oracleへ照合した。W/A/KV/full-attentionはHIP-only、fallback false、cleanup 0だった。
- output: 固定Qwen3.5-4Bの3／4および17／4 full-model試行はbefore/afterで生成token列が一致した。
  数値差、最初の分岐、品質recipe変更はないため追加quality gateを起動しない。
- performance/decision: 共通codecと起動境界specializationを両targetへshared adoptionした。代表17-token FP16-KV prefillは
  gfx1030 MXFP8/MXFP6 `47.31/98.23→48.48/99.23 tok/s`、gfx1201 `36.67/32.72→72.87/115.30 tok/s`。
  MXFP8 KV=8,193 attentionはgfx1030 `5.248→2.462 ms`、gfx1201 `3.515→1.569 ms`だった。
- rejected: cross-plan activation cacheはbuffer generation/liveness identityがなくstale readをfail-closeできず、単純fusionはN tileごとに
  activation量子化を重複するため不採用。rollbackはPhase 61までのconsumer-local codecであり、public rollback optionは追加しない。
- details: [Phase 62履歴](../history/2026/08/21-31/phase62-reusable-low-precision-block-optimization.md)。

### OUT-2026-08-30-MXFP8-E4-DEFAULT: block16廃止とstandard OCP MXFP8 E4既定化（N2）

- scope: reviewed Qwen3.5-4B BF16 dense text／full attention／single GPU／head dim 256、exact `gfx1030`、`gfx1201`、
  `gfx942:sramecc+:xnack-`。対象外model/laneのfixed recipeは変更しない。
- change: `kv-fp8-e4-block16`／`kv-fp8-e5-block16`を全production admission境界で拒否し、省略時を
  `kv-mxfp8-e4-v1`（OCP E4M3FN、block 32、E8M0）へ変更する。gfx942でもFNUZへ再解釈しない。
- classification: KV resident量子化とattention decode値がFP16から変わるためN2。明示`fp16`をrollbackとして残す。
- direct GPU: ROCm 7.14.0のV620 exact `gfx1030`とR9700 exact `gfx1201`で、head dim
  `31/32/33/255/256/257`のvalue／scale byte oracle、append 6、head dim 256のpacked direct attention 1をPASSした。
  いずれもHIP-only、fallback 0、cleanup 0である。gfx942実機は未実施であり、他tupleへ一般化しない。
- full-model draft quality: 同じQwen3.5-4B lockと10 case／20 logit rowの一回測定で、KLD p99は両targetとも
  `0.004945428206833837`、KV request-state peakは`68,354,048`から`60,195,840` byte（11.935%減）だった。
  top-1一致はgfx1030 `1.0`、gfx1201 `0.85`で、gfx1201はfreeze済み`>=0.99`を満たさない。これは実行correctnessと
  memory効果のdraft evidenceであり、品質gate PASSへ読み替えない。report SHA-256はgfx1030
  `d342b7755857848741d67d3ae37a580dcc6bb6442fabfbe49321fa03f013b9c2`、gfx1201
  `1280e6e0172bc25be1c69a370d6f449e8455c3d4d6108aaaa91f96d8b6e471c0`である。candidateは明示形式ではなく
  省略時resolverからtarget-aware graphへlowerした。
- decision: gfx1201品質未達を隠さずN2として保持した上で、2026-08-30のユーザー明示決定を採用根拠としてdefault変更を維持する。
  release品質昇格は主張せず、明示`fp16`をrollbackとする。
- history: Phase 53/54のblock16 correctness／quality／early-stop evidenceは削除せず、今回の採用根拠には使わない。

### OUT-2026-08-27-P53: KV FP8 block16 target別default判定（N2・旧recipe非採用、follow-up active）

- scope: reviewed Qwen3.5-4B BF16 dense text/full attention/single GPU、exact `gfx1201`の
  `kv-fp8-e4-block16`とexact `gfx1030`の`kv-fp8-e5-block16`。標準OCP `kv-mxfp8-e4`／
  `kv-mxfp8-e5`はreference-onlyのexplicit比較である。
- baseline/candidate: baselineはFP16 KV。block16はtoken内head-dimension方向16値ごと、標準MXFP8は32値ごとに
  独立E8M0 scaleを持つ。append/outputはBF16、attention accumulatorはFP32を維持するが、KV量子化recipeを
  変更するため**N2**とする。
- superseded recipe: 以下のcorrectness、quality、metrics、summary、digestは、有限値を飽和させない最小scaleを使った
  `kv-fp8-e4-block16-v1`／`kv-fp8-e5-block16-v1`へ結合する。このevidenceは監査履歴として保持するが、
  descriptor v2のcorrectness、品質、default採否を決めない。
- correctness: gfx1201／gfx1030でblock16とMXFP8のpadded value／scale byte oracle、append 6、direct attention 1、
  HIP-only、fallback 0、cleanup 0をPASSした。gfx942はfresh Phase 53 reportがなく、standard OCP MXFP8はFNUZ非互換のためunsupportedである。
- quality: FP16→block16→MXFP8をresident完全解放付きで3 repeatした。gfx1201 block16はKLD p99
  `0.0038687249522990803`、top-1 `0.85`、long-context loss `0.08333333333333337`、gfx1030 block16はKLD p99
  `0.04331390780013198`、top-1 `0.8`、long-context loss `0.16666666666666663`だった。全repeatは同値で、
  reference-only MXFP8 KLD p99はgfx1201 `0.004945428206833837`、gfx1030 `0.03218873133110086`だった。
- E5 analysis: gfx1030の逆転はscale recipeが第一候補である。E5M2 block16は最大有限値`1.75 * 2^15`を飽和させないため、
  block amaxの仮数が`1.75`を超えるとscaleを標準MXFP8より一段大きくする。標準MXFP8は最大側をSATし得る代わりに残りへ
  2倍細かい刻みを保つ。仮数2 bitではこの差がblock16の局所性を上回ったと推定するが、scale/SAT/layer別countは未取得である。
- decision: freeze済みpolicyのtop-1 `>=0.99`とlong-context loss `<=0.02`を両targetが満たさず、gfx1201／gfx1030は
  旧recipeについて`retain-fp16`。gfx942は`insufficient-evidence`のまま将来のMI300X一括検証へ延期した。明確な品質FAIL後の
  early-stopにより旧recipeの7行performance/resourceを実行しない。
- follow-up: ユーザー決定によりblock16はblock size 16を維持し、descriptorを`kv-fp8-e4-block16-v2`／
  `kv-fp8-e5-block16-v2`、scale recipeを`StandardMxFloorPowerV1`へ変更する。有限amaxの最大2冪をE4では256、
  E5では32768で割ってE8M0範囲へ収め、RNE＋SATする。fresh correctnessはgfx1201／gfx1030でPASSしたが、block16 KLD p99は
  それぞれ`0.006562189165612111`／`0.03659844555378746`、top-1とlong-contextがthreshold未達で両方`retain-fp16`とした。
  default mappingは空、MI300Xは引き続きdeferredである。
- rollback: mapping候補は空で省略時FP16を変更していない。explicit新形式とstate identityは残し、unsupported scopeをsilent fallbackしない。
- evidence: v2 summary `external:phase53/phase53-kv-default-summary-standardmx-v2.json` SHA-256
  `c259e81bc76fb341e9dbba8cdcc0c132456a0762585b471556267cdc59165e10`、空mapping候補
  `external:phase53/phase53-runtime-mapping-standardmx-v2.json` SHA-256
  `283911d387c67d7ba25546ce702fd89ee6a25db482d9e62bf0b61854d5613e77`。個別report digestは
  [Phase 53履歴](../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)を正とする。

### OUT-2026-08-25-P51-GDN-W64: gfx942 wave64 column-state候補（N1・明示opt-in）

- scope: logical target `gfx942`でruntime実体が厳密に`gfx942:sramecc+:xnack-`、Q/K heads 16、value heads 32、
  head/state dim 128、BF16 activation、FP32 recurrent state、token count 128以上。`gfx1030`、`gfx1201`、suffixなし／別suffix、
  unknown target、shape外、127以下は既存providerを維持する。
- baseline/candidate: baselineはvalue head当たり128 threadで各output columnの128 state項を逐次走査する。candidateは
  256 threadの4 wave64でwave当たり1 column、lane当たり2 state dimensionをregisterへ保持し、tokenを逐次処理する。
  recurrent kernelはLDSとbarrierを使用せず、`S^T k`と`S^T q`だけをwave64 treeへ変更する。Q/K L2 normとoutput RMSNormは
  gfx942 baselineと同じ128項index順FP32逐次和、同じBF16-RNE stageを維持する。
- 分類: **N1**。real-number式、128項、FP32 accumulator、decay/state update、transactional state publicationを維持し、
  recurrent projectionの加算依存深さだけを127段からlane local 2項＋wave64 treeへ短縮する。数値差はこの2 reductionへ局所化する。
- selection/identity: `SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE=1`の明示opt-inだけで
  `linear_attention.gdn.column_state.gfx942_wave64.v3` / `sllm_linear_attention_column_state_wave64_v3`を選ぶ。
  `SLLM_GDN_FORCE_BASELINE=1`が常に優先する。compile sourceはCMakeがexact `gfx942:sramecc+:xnack-`にだけ設定する
  `SLLM_HIP_COMPILE_WAVE64=1`へ限定する。
- correctness/performance: host/Rust selector、127/128/129境界、shape、target suffix、force-baseline、metadata identityとHIP compileを
  確認した。MI300X operator 7 shapeと、同じstateへのcandidate 128 token→force-baseline 128 token継続はPASSし、second outputを
  256 token逐次scalar oracle、publication length/layoutを128/256境界へ照合した。最大絶対／相対誤差は
  `0.00390625`／`0.014705882`でbaselineと同一、fallbackなし、cleanup 0だった。full-model `10,001/2`は出力完全一致で、
  prefill中央値を`22.718162442`秒から`6.410255551`秒へ3.54403x短縮した。候補入り全7行は未取得のため、default採用せず
  exact gfx942の明示opt-in `target-separated`候補とする。
- rollback: opt-inを設定しないか`SLLM_GDN_FORCE_BASELINE=1`を設定する。恒久rollbackはv3 selectorと3 stage launch/symbolを除去する。

### OUT-2026-08-24-P52: RDNA長context KV providerとVMM append rollback（N0）

- scope: exact `gfx1030`/`gfx1201`のlogical capacity 65,536以上と、共通virtual-contiguous KV append/COW。
  exact `gfx942`の既存resident選択、65,535以下、unknown target、KV layout/encoding、attention kernelは変更しない。
- baseline/candidate: 長capacityのRDNAはpage単位VMM commitからlogical capacity全量の通常device allocationへ変える。
  virtual経路は各planeを逐次更新していたgrow/COWをappend transactionで包み、失敗時に追加mapping/handleと旧shared accessを戻す。
- 分類: **N0**。K/Vのtoken-major byte layout、append入力、attention式、dtype、演算順、丸め、logical publication時点は不変である。
  providerは同じcontiguous pointer contractの物理所有方式だけを変え、失敗rollbackは成功出力を変更しない。
- correctness/output影響: capacity 65,535/65,536/65,537のtarget selector、VMM createのfirst/middle/last、map/access、
  COWのfirst/cross-plane failureを注入し、logical/mapped/commit復元、retry、release、live resource baselineを確認した。
  R9700 `10,001/2`は13/13、`100,000/2`は4/4 PASSし、全requestの生成は`[23066,23066]`、HIP-only、fallback/cleanup 0だった。
- decision/rollback: exact targetとcapacity境界に限定採用する。rollbackは
  `RDNA_CONTIGUOUS_LONG_KV_MIN_TOKENS`のgfx1201選択を除去し、virtual providerへ戻す。VMM transactional rollbackは
  correctness修正なのでprovider selectorのrollbackとは分離する。
- resource: 100kは8 KV layerのK/V 4 GiBをresident確保し、HBM peak `15,388,794,880` bytes、終了後baseline復帰。
- 詳細: [Phase 52計画](../plans/archive/2026/08/21-31/phase52-r9700-100k-kv-commit-oom.md)、
  [summary](../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)。

### OUT-2026-08-24-P49-P50-BUNDLE: Qwen decode 3融合（N0・target限定採用）

- scope: 固定Qwen3.5-4B BF16 graph、text-only greedy、exact `gfx1030`/`gfx1201`、`M=1`、FP16 KV、adapter/control、
  MTP、multimodal、FP8 sidecarなし。Residual RMSNorm、GDN qkv/z/b/a projection、MLP gate/up/SiLUの3 familyを対象とし、
  `M>1`とscope外graphは既存opへ意味分解する。exact `gfx942`とunknown targetは選択しない。
- baseline/candidate: Residual RMSNormはF32 add→BF16-RNE intermediate→wave32 RMSNorm→BF16-RNEを1 kernelへまとめるが、
  中間roundと8 wave reduction順を維持する。GDNは4本のdecode matmulを一launchへ束ね、各columnのBF16 decode、K項のF32
  accumulate、wave32 tree、BF16-RNEを維持する。MLPはgate/upの独立F32 reductionとBF16-RNEを同じblockで実行し、丸め済み
  gateをSiLU→BF16-RNE、丸め済みupとF32 multiply→BF16-RNEする既存elementwise境界を維持する。
- 分類: 固定Qwenのfinite input/model scopeでは3件とも**N0**。real-number式、入力項、F32 accumulator、wave32 reduction、
  BF16 round stage、公開tensor境界を変更しない。GDN bundleのNaN payload canonicalizationはbaselineとのbit-exact比較を未実施であり、
  非有限payloadをN0へ一般化しない。説明不能なfinite差、state差、非決定差はN3としてcandidateを無効化する。
- correctness/output影響: Residualは`1x2560`、`2x255`、`3x256`、`3x257`のintermediate/output bitwise oracle、
  GDNはactual `K=2560`、width `8192/4096/32/32`を4 baseline matmulへ20 repeat照合した。いずれもgfx1030 evidenceである。
  MLP専用operator oracleと3件のgfx1201専用scalar oracleは未実施だが、gfx1201のcontrol/residual/GDN/MLP/fused3は3 warmup＋
  10 measured、HIP-only、fallbackなし、cleanup復帰、全sampleの生成token列一致をPASSした。5 candidate間でも生成token列は一致した。
- performance/resource: gfx1201 short 17/17のE2E中央値はcontrol `451.8648785` ms、Residual `446.7119175` ms、
  GDN `435.1726845` ms、MLP `436.8484785` ms、3融合 `410.794651` ms。3融合はcontrol比9.09%短縮し、dispatchは
  `108732`から`73372`へ減少した。各runはprocess終了後にHBM/GTT baselineへ復帰した。
- decision/rollback: exact targetごとの専用selectorで限定採用し、対応targetではunset/`1`を有効、`0`/unknownを無効とする。
  rollbackは`SLLM_QWEN_GFX{1030,1201}_RESIDUAL_RMSNORM_FUSION=0`、
  `SLLM_QWEN_GFX{1030,1201}_GDN_PROJECTION_BUNDLE=0`、
  `SLLM_QWEN_GFX{1030,1201}_MLP_GATE_UP_SILU_BUNDLE=0`。詳細は
  [Phase 50履歴](../history/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)を参照する。

### OUT-2026-08-24-P49-P50-P32: decode GQA4 32 partition（N2・target限定採用）

- scope: exact `gfx1030`/`gfx1201`、causal attention decode `M=1`、KV長4,096以上、Q heads 16、KV heads 4、
  head dimension 256、FP16 KV。KV `4095`以下、別head/dtype/target、force-baselineでは既存providerを維持し、gfx942は選択しない。
- baseline/candidate: baselineはkey順の単一online-softmax stateを持つ。P32はKVを32区間へ分け、128-thread/4-waveのstage 1で
  partition-local maximum、denominator、weighted Vを計算し、stage 2がpartition順に固定mergeする。real-number式、causal key集合、
  GQA mapping、F32 state、最終BF16-RNEは同じだが、QKの加算依存深さは概ね8段から12段へ増え、partition merge順も変わる。
- 分類: **N2**。Phase 33 C1およびPhase 49のユーザー承認済みGQA split数値方針と同じく、僅かなworst-case bound増加を
  token一致だけでN0/N1へ再分類しない。gfx1201への展開も同一algorithm/scopeをtarget別A/Bして採用し、別shapeへ拡張しない。
- correctness/output影響: gfx1030はKV `1023/1024/1025/4096/8192/16384`の独立scalar oracle 6/6と、境界・非有限を含む
  full candidate 22/22をPASSし、最大絶対誤差0、fallback/cleanup 0だった。gfx1201はhost selectorで`4095/4096/4097`、env、
  force-baseline、shape、gfx942非選択を確認し、4,096/256 full-modelを1 warmup＋3 measuredでcontrol/P32/P32+3融合とも
  HIP-only、fallbackなし、cleanup復帰、全sample token一致でPASSした。gfx1201専用scalar oracleは未実施であり、その点を隠さない。
- performance/resource: gfx1201 4,096/256のE2E中央値はcontrol `10417.448939` ms、P32 `7671.955934` ms、
  P32+3融合 `6969.896471` ms。P32単体は26.36%、統合候補は33.09%短縮し、統合時decodeは`42.6583` token/sだった。
  P32単体のdispatchはcontrol `504000`から`512160`へ増えたが、全runでHBM/GTTはbaselineへ復帰した。
- decision/rollback: exact target専用envのunset/`1`で既定有効、`0`/unknownで無効とする。
  `SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT_P32=0`または
  `SLLM_CAUSAL_ATTENTION_GFX1201_DECODE_GQA4_SPLIT_P32=0`でtarget単位に戻し、
  `SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`でattention candidate全体をbaselineへ戻す。wave32 block/partitionをgfx942へ直接使わない。

### OUT-2026-08-21-P36-MTP: gfx942 MTP width/state/admission拡張（N0）

- scope: Qwen3.5-4B、公開CLI、exact `gfx942`、greedy MTP draft width 1〜8。BF16 targetはFP16 KV、FP8 targetは
  dynamic FP8 target KVを使い、MTP side modelは既存どおりBF16 weights＋FP16 KVとする。
- baseline: 公開CLIはforced MTPをgfx1201、width 1へ固定し、request stateをvisible token budgetだけで確保していた。
  quantized GGUF plan schemaもMTP graph validationで拒否していた。
- candidate: proposal widthを1〜8へ一般化し、target verify rowsを`width+1`、allocated state capacityを
  `logical+width`へboundedにする。quantized GGUFは同じmodel fingerprint、tied embedding、MTP component/recipeを検証して
  admissionする。target側とMTP側のweight/KV encodingを別fieldでreportする。
- 分類: **N0**。targetが選んだtokenだけを既存順でpublishし、不一致draftはrewind/replayする。target equation、dtype、
  round stage、sampling、stop、visible-token budgetを変えず、追加capacityは未公開の投機stateだけを保持する。
- correctness/output影響: BF16 off/width 2/3/4/7/8とFP8 target off/width 3は、それぞれoffと同じ16 visible tokenへ一致した。
  proposal accountingは全rowでaccepted+rejected=proposed、fallback/cleanup 0、HIP-onlyだった。初回width 2のcapacity overflowと
  FP8 plan schema拒否は修正後のfocused rerunで解消した。
- performance/resource: Session Cはcorrectness runであり性能claimを行わない。追加state slackは最大8 tokenにboundedである。
  rollbackはforced gfx942/width拡張とquantized-plan admissionを除去し、target-onlyまたは従来width 1へ戻す。
- evidence: [Session C summary](../../ci/matrix/phase36-mi300x-session-c-summary-v1.json)。

### OUT-2026-08-21-P36-CHUNK: 公開prefill chunk overrideとMI300X partition確認（N0）

- scope: Qwen3.5 dense公開CLIの`--prefill-chunk-tokens 1..16384`、exact gfx942、BF16 targetのFP16/dynamic FP8 KV。
- change: auto selectorを維持しつつ、明示指定時は一つの候補だけを選び、resource fallbackで別chunkへ黙って変更しない。
  absolute position、KV/GDN state継続、terminal行だけのLM head/Argmax、量子化recipeは既存Phase 31 contractを維持する。
- 分類: **N0**。演算対象、dtype、round stage、terminal visible outputを変えないscheduling/resource指定である。
- correctness/output影響: auto/512/2K/4K/8K/16K × 上記2 KV encodingの12/12 rowで、入力ID`23066`×10,001から
  生成ID`[23066,23066]`へ一致した。全rowはHIP-only、fallbackなし、cleanup 0で、終了後HBM/GTT baselineへ復帰した。
- evidence: [Session B summary](../../ci/matrix/phase36-mi300x-session-b-summary-v1.json)。
- resource: arena high-waterはauto/16K `5,278,049,280` bytes、512 `270,209,024` bytes。これはmemory feasibility
  evidenceであり、single-run timingを性能claimへ使わない。rollbackは明示overrideを除去して既存auto selectorだけへ戻す。

### OUT-2026-08-21-P36-G: gfx942 GDN normのPhase 29 scope修復（N0）

- scope: exact `gfx942` / wave64、Qwen3.5 GDNの短いbaseline provider（token count 128未満）。
  Phase 29で承認したwave32 treeは引き続きexact `gfx1030`/`gfx1201`だけを対象とする。
- baseline: Phase 28までのgfx942はQ/K L2 normとoutput RMSNormを128項のindex順FP32逐次和で計算した。
  Phase 29の共通source変更により、文書化したtarget scope外のgfx942にも4個のwave32 partialを足すtree順が漏れていた。
- candidate: `SLLM_HIP_COMPILE_WAVE64=1`のbuildだけ128項の逐次和を維持し、wave32 buildはPhase 29のtreeを変更しない。
  real-number式、入力集合、dtype、BF16 round stage、recurrent state、kernel symbol、dispatch数、ABIは不変である。
- 分類: **N0**。新しい数値順序の採用ではなく、gfx942を承認済みのPhase 28順序へ戻すtarget-scope修復である。
  gfx1030/gfx1201のPhase 29 N1最適化と、gfx942の既存wave64 matmul providerは変更しない。
- correctness: gfx942のtoken 1/3/17 GDN独立oracleは3/3 PASSし、最大絶対/相対誤差は
  `0.00390625`/`0.014705882`、state publication一致、fallback/cleanup 0だった。Qwen BF16 `Hello`の5-tokenは
  修復前後とも`[11,353,2688,4313,310]`で、同一provider repeatも一致した。診断用wave32 matmul controlではGDN順序との
  組合せだけがreviewed RDNA token `[11,353,1044,4313,310]`への分岐を説明した。
- output影響: final gfx942 BF16とFNUZ FP8は3番目のtokenだけ`2688`/`1044`へ分岐した。BF16はwave64 BF16 reduction、
  FP8はhipBLASLt FNUZを使うため、このcross-dtype/cross-provider差をbit-exact gateにはしない。Unicodeとstop rowは一致した。
- performance/resource: 未測定。短GDNのdispatch、scratch、allocationを変更せず、Phase 29/35のRDNA performance scopeへ
  影響しない。rollbackはwave64条件分岐の除去だが、Phase 29の承認scopeを再びgfx942へ拡張するため採用しない。

### OUT-2026-08-20-P35-A: Full Attention Q_TILE=4 query-row共有（N1・限定採用）

- scope: exact `gfx1030`/`gfx1201`、Qwen系causal/full attention、`M>=128`、Q heads 16、KV heads 4、
  head dim 256、FP16/dynamic FP8/static FP8/NVFP4 KV。短M、decode、別shape/targetは既存providerを維持する。
- baseline: Phase 33 C2が1 query row × 1 KV head/workgroupでK/VをGQA 4 headへ共有する。
- candidate: 1 workgroupが4 query row × GQA 4 headを所有し、K/Vを16 logical queryへ共有する。各logical queryの
  causal key集合、key順online softmax、FP32 maximum/denominator/weighted V、BF16 RNE出力は独立に維持する。
- 分類: **N1**。QKは同じ256項を8 value/laneの固定treeとwave32 treeで加算し、Phase 33 C2の概ね8段を超えない。
  real-number式、入力集合、dtype、丸めstageは同じで、標準worst-case boundは非増加である。
- correctness: 2 target × 4 KV encoding × 29 caseの232/232 PASS。M=127/128/129、255/256/257、nonzero start、
  long-prefix decode、NaN/+Inf/subnormalを含み、最大絶対誤差はFP16 `2.3841858e-7`、FP8 `4.7683716e-7`、
  NVFP4 `1.1641532e-9`、fallback/cleanup 0だった。
- output影響: fixed 10,001 input / 2 outputはbaseline/candidateとも`[2064,5686]`。将来の差は同じ項の固定tree順へ
  局所化できる範囲だけN1とし、説明不能・非決定差はN3とする。
- 性能/resource: V620 profileのFull Attentionは10.820秒から4.110秒へ62.02%短縮し、Attention-only E2Eは
  V620 19.31%、R9700 8.59%短縮。global scratch、追加dispatch、KV mirrorは0、arena high-waterは不変。
- 決定: 担当AI裁量で両target共通のshape限定採用。M=64/65のV620候補悪化を避けるため境界を128とし、
  Phase 33 providerを明示complementにする。固定改善率は使用しない。
- rollback: `SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`相当のselectionへ戻し、Q_TILE=4 symbol/launchを除去する。
- 詳細: [Phase 35履歴](../history/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)、
  [bounded summary](../../ci/matrix/phase35-attention-gdn-summary-v1.json)。

### OUT-2026-08-20-P35-G: GDN column-owned recurrent state（N1・限定採用）

- scope: exact `gfx1030`/`gfx1201`、Qwen3.5 GDN、token count 128以上、Q/K heads 16、value heads 32、
  head/state dim 128、BF16 activation、FP32 recurrent state。短prefill/decodeはPhase 28/29 providerを維持する。
- baseline: value head当たり1 workgroupで、threadが1 output columnを所有し、128 state rowを逐次走査する。
- candidate: preprocessでQ/K normとbeta/decayを一度生成し、1,024 workgroup相当のcolumn-owned recurrent kernelで
  lane当たり4 state rowをregisterに保持し、postprocessで既存output RMSNorm/z SiLUを適用する。targetごとの既存物理state
  index mappingとtransactional previous/next publicationは変えない。
- 分類: **N1**。state transpose/migrationや項の欠落はない。`S^T k`/`S^T q`の同じ128 FP32項を逐次依存127から
  4項local + wave32 treeの概ね8段へ短縮し、標準worst-case boundは非増加である。Q/K、beta、raw output、normalized outputの
  BF16 round stage、decay/state update式は維持する。
- correctness: 両targetでtoken 1/3/17/127/128/129を独立oracleへ照合し12/12 PASS。最大絶対/相対誤差は
  `0.00390625`/`0.014705882`、next-state publication一致、fallback/cleanup 0だった。
- output影響: fixed 10,001 input / 2 outputは`[2064,5686]`を維持した。将来の差はrecurrent projectionの固定tree順へ
  局所化できる範囲だけN1とし、state/publication差はcorrectness blockerとする。
- 性能/resource: V620 GDN familyは約7.672秒から0.618秒へ91.95%短縮しfixed llama.cpp 0.622秒と概ね同等になった。
  GDN-only E2EはV620 19.84%、R9700 7.17%短縮。10,001 tokenでbeta/decay FP32 planeを2,560,256 byte/layer、
  24 layer合計61,446,144 byte追加し、
  1 layer当たりdispatchは2から4、full-modelは984から1,032へ増えたがarena high-waterは不変だった。
- 決定: 担当AI裁量で両target共通のshape限定採用。絶対短縮、N1、peer parity、既存state layout再利用、短経路complementを
  総合し、追加2 dispatchの費用を上回ると判断した。
- rollback: `SLLM_GDN_FORCE_BASELINE=1`相当のselectionへ戻し、preprocess/recurrent/postprocessの3 candidate kernelを除去する。
- 詳細: [Phase 35履歴](../history/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)、
  [bounded summary](../../ci/matrix/phase35-attention-gdn-summary-v1.json)。

### OUT-2026-08-20-P34: gfx1030長行BF16 matmul hipBLAS route（N1・限定採用）

- scope: exact `gfx1030`、Qwen3.5-4B内部BF16 projection。主要5 shapeは`M>=128`、`K=2560,N=1024`は
  `M>=1024`。N=32、未知shape、all-logits、短M、gfx1201/gfx942は既存providerを維持する。
- baseline: `matmul.bf16_fp32.tiled16.v2`が16x16 tile内でKをsource-level scalar FP32 accumulateし、BF16 RNE出力する。
- candidate: existing `matmul.hipblas.gemm_ex.v2`/`hipblasGemmEx`を使う。BF16 input/weight、FP32 compute、BF16 RNE出力、
  real-number equation、入力項集合、layout、publicationは同じ。観測Tensile solutionはGSU1でglobal split/atomic combineを使わない。
- 分類: **N1**。providerの固定reduction順は異なるためbit exactのN0ではないが、同じK項を一度ずつ含む決定的な並べ替えであり、
  Phase 8の保守的な`gamma_K * sum(abs(a_i*w_i)) + BF16 half-ULP` worst-case boundは非増加である。
- correctness: signed/exponent-mixed stressはprovider間差を観測したが、repeatは決定的だった。`M=128,K=2560,N=4096`と
  `M=10001,K=2560,N=9216`のsampled F64 bound違反は両provider 0、matmul G1は両target 18/18 PASS、fallback/cleanup 0。
- output影響: final 10,001 prompt / 2 outputはbaseline/candidateともtoken `[2064,5686]`。別入力でtoken差が生じても、
  同じ式・項・dtype・丸めstageと非増加boundへ局所化できる範囲はN1として扱う。説明不能な差はN3である。
- 性能/resource: V620 248-call加重projectionを62.526秒から11.081秒へ82.28%、full modelを89.249秒から
  34.684秒へ61.14%短縮。context-lifetime hipBLAS handle 1、hipBLASLt/workspace/weight repack/追加dispatch 0、arena不変。
- 決定: 担当AI裁量でshape-aware限定採用。small-Nの不安定性と未知shapeは既存providerへ隔離し、固定改善率は使用しない。
- rollback: `phase34_gfx1030_hipblas_shape` routeとgfx1030 contextのhipBLAS handle作成条件を除去してtiled16へ戻す。
- 詳細: [Phase 34履歴](../history/2026/08/11-20/phase34-v620-long-prefill-bf16-matmul-provider-optimization.md)、
  [bounded summary](../../ci/matrix/phase34-v620-prefill-matmul-summary-v1.json)。

### OUT-2026-08-20-P33-C1: decode wave8 KV split（N2・ユーザー承認採用）

- scope: exact `gfx1030`/`gfx1201`、Qwen系causal/full attention、`M=1`、KV長1,024以上、head dim 256、
  FP16/dynamic FP8/static FP8/NVFP4 KV。2026-08-20のユーザー承認によりproductionへ限定採用する。
- baseline: 256 dimensionのQK積を概ね8段のbalanced treeで加算し、key順に一つのonline-softmax stateとweighted Vを更新する。
- candidate: 8 waveへ連続したKV区間を割り当て、waveごとのpartial online-softmaxを区間順にLDS上で固定mergeする。
  各laneが8 dimensionを逐次加算してwave treeへ渡すためQK加算依存深さは概ね12段となる。real-number semantic、入力集合、
  FP32 accumulator、softmax式、BF16 RNE、state publicationは維持する。
- 分類: **N2**。QK sumの標準worst-case boundは概略`gamma_8`から`gamma_12`へ僅かに増える。weighted Vは短い
  partialと固定mergeになるが、QK側のbound悪化を相殺したと証明しない。承認後もN1へ再分類せずN2として追跡する。
- correctness: 4 KV encoding × 2 target × 29 caseを含むPhase 33 oracle 232/232 PASS。candidate範囲のKV=1,024
  NaN query/+Inf value、KV=4,097 signed mixed、KV=8,193を含む。最大絶対誤差はFP16 `2.3841858e-7`、FP8
  `4.7683716e-7`、NVFP4 `1.1641532e-9`。fallback/cleanup 0、測定範囲の生成token差なし。
- 性能: KV=1,024〜8,193のdevice中央値をgfx1201で約53〜58%、gfx1030で約64〜65%短縮した。scratch、追加dispatchは0。
- 決定: **ユーザー承認により採用**。大幅なdevice短縮、観測誤差、token一致、scratch/追加dispatch 0を確認したうえで、
  N2の僅かなworst-case bound増加を受容した。C1 symbol/routingをproductionに維持する。
- rollback: `use_decode_wave_split`をfalseとし、`causal_attention_decode_wave_split_kernel`とC1 metadata symbolを除去する。
- 詳細: [Phase 33履歴](../history/2026/08/11-20/phase33-full-attention-structural-optimization.md)、
  [bounded summary](../../ci/matrix/phase33-full-attention-summary-v1.json)。

### OUT-2026-08-20-P33-C2: prefill GQA4 K/V共有

- scope: exact `gfx1030`/`gfx1201`、Qwen系causal/full attention、`M>=64`、GQA ratio 4、head dim 256、
  FP16/dynamic FP8/static FP8/NVFP4 KV。
- baseline: query row/query headごとに1 blockを起動し、同じKV headへmapされる4 query headがK/Vを別々にdecode/readする。
- candidate: 1 query row/KV headごとに1 blockを起動し、K/V elementを一度だけdecodeして4 query headで共有する。各headの
  QK reduction、online maximum/denominator、weighted V、causal key順は独立に維持する。global scratch、追加launch、KV mirrorはない。
- 分類: exact `gfx1201`は既存wave providerと同じ32-lane partial + 8 partial固定treeで**N0**。exact `gfx1030`は
  256-thread LDS treeからwave32 + 8 partial treeへ順序が変わるが、加算依存深さは同じ8段で標準worst-case boundが
  非増加のため**N1**。real-number equation、入力集合、dtype、丸めstageは維持する。
- correctness: Phase 33 oracle 232/232 PASS。M=63/64/65、127/128/129、255/256/257、nonzero start、M=64の
  NaN query/+Inf valueを含む。fallback/cleanup 0、測定範囲の生成token差なし。
- 性能: M=64〜257のFP16 device中央値をgfx1201で約21〜47%、gfx1030で約38〜54%短縮した。M=37のgfx1201
  prototypeは8.22%悪化したためM>=64だけへstable routeする。R9700 10,000-promptのC1+C2候補はB0比28.96%全体短縮。
- 決定: 担当AI裁量でshared限定採用。全scoped patternの大幅改善、scratch 0、共通source、明示B0 complement、低い保守費用を
  総合し、固定改善率gateは用いない。gfx1201 matrix innerは同じ4-row tileへ適合せず別candidateとして棄却する。
- rollback: `use_prefill_gqa4`をfalseとし、M>=64をPhase 30 wave provider（gfx1201）またはB0（gfx1030）へ戻す。
- 詳細: [Phase 33履歴](../history/2026/08/11-20/phase33-full-attention-structural-optimization.md)、
  [bounded summary](../../ci/matrix/phase33-full-attention-summary-v1.json)。

### OUT-2026-08-19-P32: gfx1201 native FP8 KV append encode

- scope: exact `gfx1201`のdynamic/static FP8 KV append。gfx1030、FP16、NVFP4は既存経路を維持する。
- baseline: scale後のF32値をsoftware binary searchでOCP E4M3FNへRNE/saturation encodeする。
- candidate: NaN、Inf、signed zero、448 saturationを同じcontractへ明示補正し、通常finite値を
  `__builtin_amdgcn_cvt_pk_fp8_f32(value, value, 0, false)`でencodeする。kernel、workgroup、grid、scale、store、KV formatは不変。
- 分類: **N0**。全65,536 BF16 codeをK/Vで一巡したdynamic/static fixtureと19 token境界でpayload byte／F32 scale bit mismatch 0。
  production attention oracleもgfx1201/gfx1030 × dynamic/static FP8の68/68 caseをPASSした。
- output影響: 測定上もcontract上も変更なし。生成tokenはgfx1201 10,001/16,385、gfx1030 10,001 inputですべて`[1228, 1228]`。
- 決定: 担当AI裁量でC1 native scalarを限定採用。C2 packedはworkgroup/store/tail複雑性のため不採用。default KVはFP16のまま。
- rollback: `float_to_e4m3fn_fp8_append` callをsoftware `float_to_e4m3fn`へ戻す。public ABI、state migration、artifact変換は不要。
- 詳細: [Phase 32履歴](../history/2026/08/11-20/phase32-native-fp8-kv-append-revalidation.md)、
  [bounded summary](../../ci/matrix/phase32-native-fp8-append-summary-v1.json)。

### OUT-2026-08-19-P31: chunked prefillとworkspace arena

- scope: Qwen3.5 dense BF16 weight graph、text prefill、FP16/dynamic FP8/static FP8/NVFP4 KV、gfx1030/gfx1201。
- baseline: prompt全行を一graph実行し、request-owned dynamic tensorを個別bufferへ割り当てる。
- candidate: promptを連続chunkへ分割し、absolute positionとKV/GDN stateを継続する。中間chunkの未使用LM head/Argmaxを省略し、
  completion boundaryまで重なるtensorは別slotのまま、重ならないlifetimeだけを再利用する。
- 分類: **N0**。terminal行のreal-number equation、dtype、演算順、量子化recipe、丸めstageを変更しない。chunk境界で既存KVを
  再量子化せず、新規K/Vを一度だけappendする。static FP8のscale 1.0は明示設定のdescriptor完成でありdefault変更ではない。
- correctness: 10,001 tokenをgfx1030/gfx1201、16,385 tokenをgfx1201でHIP-only、fallbackなし、cleanup 0で実行した。
  16,385 tokenは16,384+1の2 chunkとなり、反復入力の生成tokenはone-chunk controlと同じ1228だった。dynamic FP8は両targetの
  10,001 tokenとgfx1201の16,385 token、static FP8はgfx1201の10,001 token、NVFP4は513-token spotをPASSした。
- output影響: 測定範囲でtoken差なし。chunk partition、arena reuse、intermediate terminal省略から説明できない差はN3 blockerとする。
- resource: 10,001 tokenのworkspace high-waterは39,950,821,120から5,278,049,280 byte、16,385 tokenでは
  65,448,547,584から8,646,688,768 byte相当となり、いずれも約86.79%削減した。
- 決定: shared chunked prefill/arenaを採用し、明示low-bit KV選択をCLI/serverへ接続する。defaultはFP16を維持する。
- rollback: source base commit `1def2b63cfb26cd71e7e1bf500235a6eb5c7ed9b`の一括prefill・個別allocation。
- 詳細: [Phase 31履歴](../history/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)、
  [bounded summary](../../ci/matrix/phase31-chunked-prefill-summary-v1.json)。

### OUT-2026-08-19-P30: RDNA4 causal-attention wave reduction

- scope: Qwen系generic causal/full attention、exact gfx1201、head dim 256、decode `M=1`とprefill `M>=32`、FP16/FP8/NVFP4 KV。
- baseline: 256 threadのLDS treeでQKの256項FP32積を固定balanced reductionし、keyごとに約11回のblock同期を行う。
- candidate: 8 wave × 32 laneの`__shfl_down` treeと8個のLDS partialを固定treeで合成する。online-softmaxのmax、denominator、V accumulation、FP32 accumulator、BF16 RNE output stageは維持する。
- 分類: **N1**。real-number equation、入力集合、dtype、丸めstageは同じで、QKの加算依存深さはbaselineの8段からwave内5段+wave間3段の8段を超えない。native E4M3FN readは全256 code（NaN 2 codeを含む）がsoftware contractと一致するためN0である。
- correctness: gfx1201/gfx1030 × FP16/FP8の各17 caseが全出力一致、fallback 0、cleanup failure 0。gfx1201 native decode probeは256/256 code PASS。full-model 29/267/4108 inputのbaseline/candidate token recordは一致した。
- output影響: 測定範囲ではtoken差なし。演算順が変わるため将来のlogit/token差は生じ得るが、原因は固定balanced tree間の丸め差へ局所化できる。
- 性能: gfx1201 operatorはFP16 decodeで6.12〜17.16%、FP8 decodeで0.64〜27.91%、prefill `M≈255`で約21.0%/31.5%短縮。Qwen3.5-4B BF16、4108 inputの3 process中央値はTTFT 9.60%、prefill 9.72%、E2E 9.16%、decode throughput 7.86%改善した。29 inputのTTFT -0.60%は1 processのsub-1% control noiseとして非悪化扱いとした。
- 決定: exact gfx1201の`M=1`と`M>=32`へ限定採用し、`M=2..31`とgfx1030はbaselineを維持する。native append encodeはchunk 256で68.69%悪化したため棄却した。
- candidate source SHA-256: closeout summaryの`kernel_source_sha256`を正とする。
- rollback: source base commit `1def2b63cfb26cd71e7e1bf500235a6eb5c7ed9b`のscalar/vector provider。
- 詳細: [Phase 30履歴](../history/2026/08/11-20/phase30-rdna4-native-attention-kv-optimization.md)、
  [bounded summary](../../ci/matrix/phase30-rdna4-attention-kv-summary-v1.json)。

### OUT-2026-08-18-P29: GDN norm wave32 tree reduction

- scope: Qwen3.5-4B dense BF16、通常target-only decode、GDN recurrent gated norm、gfx1030/gfx1201。
- baseline: Q/K L2 normとoutput RMSNormの128項FP32逐次和。依存深さ127。
- candidate: 4 wave × 32 laneの`__shfl_down` tree reduction後、4 partialを固定順で加算。最長加算依存深さは概ね8。
- 分類: **N1**。対象項は二乗値で全て非負、real-number sumとBF16 RNE stageは同じで、逐次和の概略誤差bound
  `gamma_127`をtreeの概略`gamma_8`へ縮小する。race、atomic、追加launch、scratchはない。
- correctness: model-free token 1/3/17は両GPU PASS、output 16の全formal runはtoken一致、同一provider repeatは一致、fallbackなし、cleanup 0。
- output影響: output 128はgfx1030/B0が105、B2が20、gfx1201/B0/B1/B2が111/112/108 token目からbaselineと分岐。
  gfx1030/B1は128 token一致。差はreduction順変更によるrecurrent stateへの丸め差蓄積として説明可能。
- 性能: GDN device p50はgfx1030で2.15〜2.20%、gfx1201で8.10〜9.21%短縮。全pattern非悪化、gfx1201の全patternが5%以上。
- 決定: 2026-08-18の改訂規則でN1としてshared adoption。token完全一致を理由に棄却しない。target splitなし。
- candidate source SHA-256: `62b5d6caab9e06044e29c5f043046a65eace243faafa034aca1b3f2ce8eb3dc6`。
- rollback: Phase 28 source SHA-256 `44e0b6a0b3e5bb01cd21423a2796f14dd20d8a40f5bbc7ac5c56535a38e3807f`。
- 詳細: [Phase 29履歴](../history/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)、
  [bounded summary](../../ci/matrix/phase29-gdn-device-summary-v1.json)。

### OUT-2026-08-18-P28: GDN state pass統合

- scope: Qwen3.5 GDN recurrent state、gfx1030/gfx1201。
- change: 初回state copy、decay、previous projectionを一passへ統合。key走査、FP32演算順、BF16 RNE、state publicationを維持。
- 分類: **N0**。代表output 128でbaseline/candidate token record一致、tiny oracle PASS。
- 決定: shared adoption。Phase 29のrollback provider。
- 詳細: [Phase 28履歴](../history/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)。

### OUT-2026-08-18-P24: prefill terminal-row projection

- scope: Qwen3.5 prefill terminal LM head/Argmax。
- change: 生成開始に不要な非terminal rowのLM head/Argmaxを省略し、terminal rowの式とproviderを維持。
- 分類: **N0**。生成に使用するterminal logits/tokenを維持し、all-logits要求とMTP経路は既存all-rowへrouteする。
- 決定: shared adoption。
- 詳細: [Phase 24履歴](../history/2026/08/11-20/phase24-prefill-terminal-row-projection-optimization.md)。

## 運用

- 今後、出力へ影響しうる変更はPhase historyだけで完結させず、この台帳へ一項目を追加する。
- token差がない変更も、数値順序、近似、quantization、state、sampling、stopへ触れる場合はN0として記録する。
- 台帳更新は実装と同じcommitに含める。raw model、full logits、生成全文は追跡せず、bounded aggregateとidentityだけを残す。

[メイン計画](../plans/main-plan.md)
