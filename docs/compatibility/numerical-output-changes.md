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
