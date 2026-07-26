# SQ8_0 数値ゲート v0.2 — FP32 参照に対する相対品質

Status: **凍結済み・未評価**（2026-07-26）

機械可読な正本は
[`sq8-numerical-gate-v0.2-relative-to-fp32-reference.json`](sq8-numerical-gate-v0.2-relative-to-fp32-reference.json)
である。SHA-256 は
`64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`。

このハッシュを記録した時点まで、候補の v0.2 評価は実行していない。閾値は
既存の失敗候補の観測値から決めず、固定した artifact-FP32 意味論、IEEE-754
binary32 の丸め幅、候補非依存のコーパス、相対非劣性、および被覆数から先に
設計した。以後、候補を測るためにこの JSON の数値や入力を変更してはならない。

## Goal

`SQ8_0` 最適化経路を、CK/direct との bitwise 一致だけで判定せず、**同じ
`SQ8_0` 成果物を厳密に F32 へ復元した full-model 参照に対して、対応する
CK/direct control と同等以上に近いか**で判定する。

ここで判定するのは「固定された量子化済み形式の上で、実行経路がどれだけ参照に
近いか」である。量子化前の配布モデルとの距離は重要だが、artifact を固定した
カーネル選択の正誤とは別の問題として扱う。

## Success Criteria

- artifact-FP32 参照、コーパス、評価式、全閾値、判定順序を JSON に凍結し、上記
  SHA-256 で再現できる。
- 全候補は、同一 artifact、同一 teacher-forced token stream、同一実行形態で
  matched CK/direct control と比較される。
- logits、final hidden、各 layer hidden、KL、top-1、top-10 を**別々に**
  判定する。ある指標の改善で別の悪化を相殺しない。
- 4096 primary decode positions、64 個の 64-token block、7 primary stream、
  512 hidden-layer probe と境界/tail/pre-fill coverage が揃わない実行は
  non-qualifying とする。
- `pass_relative_fp32_v0_2` は、全必須指標・全 scope・両 candidate repetition
  が通った場合だけに付く。参照が未実装/未適格なら `blocked` であり、推定値で
  代用しない。

## Non-Goals

- 既存の bitwise gate を緩めたり、過去の No-Go を遡及して Go にしたりしない。
- 量子化前の FP16/BF16（またはその source dtype）モデルを主参照にしない。
  それは artifact 化の損失まで含む別の release-quality 監査である。
- component/single-shot 比較、token 一致だけ、または candidate 自身の生成 token
  を full-model 品質の証拠にしない。
- 本計画では GPU、activation、campaign、systemd、active manifest、`/opt/ullm`
  を操作しない。性能・ABI・resource・昇格はこの数値ゲートの後の別判定である。

## Working Hypotheses

1. `SQ8_0` の payload/scales を artifact から F32 に復元して実行すれば、artifact
   量子化を固定したまま実行経路の誤差を測れる。
2. CK/direct と異なる reduction/fragment 順序は bitwise には異なり得るが、同じ
   artifact-FP32 参照に対して CK/direct より悪くないことは測定可能である。
3. single-shot/component test だけでは decode state、feedback token、KV、layer 間の
   累積差を検出できない。teacher forcing を使った multi-step full-model decode が
   必須である。
4. 連続値だけでは rank の局所的悪化を見落とし、top-1 だけでは小さいが系統的な
   ベクトル誤差を見落とす。したがって複合スコアではなく独立ゲートが必要である。
5. 4096 token position は互いに独立とは限らない。Wilson は位置レベルの必須診断に
   留め、固定 64-token block bootstrap と stream 別判定を併置する。

## 設計

### 1. 主参照: artifact-FP32

主参照 `artifact_fp32_strict_v1` は、評価対象の immutable `SQ8_0` artifact を
唯一の重みソースとする。

- E4M3 payload を canonical decoder で binary32 に復元し、保存された scale を
  宣言された dtype/byte order のまま読み、binary32 に変換して block ごとに掛ける。
- non-SQ parameter も、同じ bound artifact/package の宣言済み storage から F32
  化する。元 checkpoint を差し込まない。
- activation quantization は行わない。F32 activation、F32 causal KV、固定順序の
  F32 演算で 40 layer、final norm、LM head を実行する。
- 行列 reduction は K 昇順で固定し、reference build 全体で一種類に固定した F32
  multiply-accumulate primitive を使う。GPU/HIP、candidate 出力、CK/direct 出力、
  F64 accumulator は参照値として使わない。
- artifact hash、各 payload/scale hash、shape、block layout、finite 性、reference
  executable/source hash、CPU/thread 構成、出力 hash を receipt に結び付ける。

量子化前 source model の F32 実行は、artifact 作成による損失も含む secondary
audit としては有用である。しかしその差で candidate を通落させない。kernel
implementation の比較では、同一 artifact が主語でなければならない。

### 2. 現在の CPU 参照の実現可能性

調査済みの既存経路だけでは、必要な full-model strict-FP32 参照はまだない。

| 経路 | 確認できた範囲 | v0.2 主参照としての可否 |
|---|---|---|
| `crates/ullm-engine/src/sq_reference.rs` | canonical artifact を stream して projection を参照計算するが、F64 accumulator | 不可（projection-only / F64） |
| `crates/ullm-engine/src/sq_optimized_reference.rs` | dynamic activation quantization 後の projection、F64 accumulator | 不可（対象意味論が異なる / F64） |
| `crates/ullm-engine/src/cpu_reference_executor.rs` | generic F32 ModelGraph の deterministic CPU executor | 不可（SQ8 weight materialization なし、1 tensor 1,048,576 elements、総 8,388,608 elements、50,000,000 work-unit の保護上限） |
| `sq8_ck_full_model` 等の serving example | runtime full model と terminal CPU helper | 不可（full model は CPU artifact-FP32 executor ではない） |

従って「CPU で現時点の full-model FP32 参照を回せるか」は **未確認** ではなく、
**既存実装のままでは不可**である。新しい独立 executor の実装と適格化が必要である。
その executor は GPU を使用しない。

必要な CPU work は、primary decode だけでも 4096 position、さらに prefill と
boundary capture を含む。現時点でこの runner の token/s、総所要時間、必要 RAM は
**未測定**である。実装後、まず `raw-p0001` の 8 teacher-forced decode step を
CPU-only で二回実行し、reference hash 一致と per-step elapsed time を記録する。
この pilot は所要時間の測定だけで、品質 coverage を満たしたことにはならない。
full corpus 所要時間を推測で報告してはならない。

保存容量については、vocab 151,936 の 4096 F32 logits は約 2.32 GiB である。これは
candidate 値から導いた量ではなく、固定 corpus と tensor shape からの必要容量なので、
reference capture 前に空き容量を preflight する。

### 3. multi-step workload と入力固定

candidate 自身の greedy token が後続入力を変えないよう、順序を固定する。

```text
artifact-FP32 reference -> greedy token stream を hash 付きで保存
                          -> CK/direct control はその token stream を teacher-force
                          -> candidate も同じ token stream を teacher-force
```

primary decode は以下の 7 stream で 4096 positions を正確に構成する。

| prompt | forced decode | positions | 狙い |
|---|---:|---:|---|
| raw 1 | 1024 | 1024 | 短い seed からの長い feedback |
| raw 8 / 32 / 128 / 512 | 各 512 | 2048 | 小〜中 context と 128/512 境界 |
| exact chat 2048 / 3584 | 各 512 | 1024 | 長 context と 4096 到達 |
| **計** |  | **4096** | 7 strata |

加えて raw 127, 255, 511, 1023 は各 4 decode step、raw 4095 は 1 decode step
を必須にする。これらは 127/128、255/256、511/512、1023/1024、4095/4096 の
boundary/tail を直接横断する。sample count を増やす目的には使わない。

`sequential_m1` と `m128_chunks_with_declared_tail` の両 prefill mode を実行する。
M=128 では raw 128/512、chat 2048/3584、raw 4095 を使い、各完全 chunk 境界と
最終 tail で final hidden/logits を採取する。これにより decode 専用の変更だけで
なく M=128 prefill で有効な候補も同じ full-model contract に入る。

final hidden は全 required position、各 transformer layer と final norm の hidden
は 512 個の candidate 非依存 probe position で採取する。probe は first/final、全
boundary、全 prefill checkpoint を必ず含め、残りを固定 SHA-256 順で選ぶ。

### 4. 「同等以上に近い」の定量化

`R` を artifact-FP32、`C` を三回測った matched CK/direct control、`X` を二回
測った candidate とする。各連続誤差は F64 で再計算する。

```text
relative_l2(A, R) = sqrt(sum((A - R)^2) / max(sum(R^2), 1e-30))
max_abs(A, R)     = max(abs(A - R))
KL(R || A)        = sum(softmax(R) * (log_softmax(R) - log_softmax(A)))
```

上限型の各指標は、candidate の二回中の worst を control 三回の median に対して
次式で判定する。

```text
E_X <= median(E_C) * 1.05 + control_repeat_envelope + absolute_floor
```

`1.05` は既存誤差の 5% を超える劣化を許さない相対非劣性幅であり、約 0.42 dB
相当である。絶対出力誤差を 5% 許容する意味ではない。いずれも候補測定前に固定した。

| 指標 | absolute floor | 必須 scope |
|---|---:|---|
| relative L2 | `2^-20` | logits、final hidden、各 layer hidden |
| mean / P99 KL (nats) | `2^-20` | logits |
| max abs | `16 * ulp(max(1, max_abs(R)))` | logits、final hidden、各 layer hidden |

これらの floor は F32 表現と reference tensor の scale に由来する。control が偶然
0 誤差でも、F32 の末尾丸め以下を「悪化」とする不安定な判定を避けるための下限であり、
候補の過去誤差を基にした値ではない。

各 metric は aggregate だけでなく、7 primary stream の各々、各 boundary case、
prefill checkpoint set でも満たす必要がある。layer hidden は各 layer ごとに
aggregate relative L2、P99 position relative L2、max abs を独立に通す。平均化や
composite score による相殺はない。

### 5. 離散 quality と near-margin

logit top-1 は降順 logit、同値なら token ID 昇順で決める。reference top-1 が
candidate top-10 に残る率も別に測る。両者について 95% 片側 Wilson lower bound を
出し、candidate の worst repetition は control median の lower bound から
0.1 percentage point を超えて下がってはならない。

`0.1 pp` は 4096 primary positions で概ね 4 position 以下の幅である。ただし、
control が reference top-1 と一致しているのに candidate だけが異なる
non-near-margin swap は **0 件**でなければならない。

near-margin swap は次の全条件を満たす場合だけ policy-aware agreement に数える。

1. candidate top-1 が reference top-2 そのものである。
2. reference top-1/top-2 margin が、同一 position の最大 control logit max-abs と
   reference scale の 16 ULP から作った control envelope の 2 倍以下である。
3. 連続値および top-10 の全ゲートも通る。

これは AQ4 P2 の「僅差の top-1 入れ替わりを、数値品質全体と分けて扱う」考え方と
整合する。ただし AQ4 の calibration-derived absolute threshold は流用しない。

### 6. 統計的妥当性

4096 position 全一致なら 95% 片側 Wilson lower bound は約 99.934% になる。coverage
不足なら閾値は緩めず、teacher forcing により不足分を固定 stream として追加して
4096 に到達させる。

autoregressive positions は独立とは限らないため、Wilson を独立サンプルの厳密な
証明としては使わない。primary decode を 64-token ごとの 64 block に分け、stream
内の block 数を保存する stratified non-overlapping block bootstrap（10,000 回、固定
seed domain）も必須にする。candidate-control の policy-aware top-1 agreement 差の
95% lower bound は `-0.1 pp` 以上でなければならない。加えて各 stream が独立 scope
として全連続値ゲートを通すため、aggregate が一つの prompt の失敗を隠せない。

### 7. 現行 bitwise gate との関係

現行 CK/direct multi-step bitwise equality は廃止しない。これは同じ演算順序を維持する
実装に対する最強の Tier-S 保証として、`pass` / `fail` / `not_applicable` を別途記録する。

v0.2 は practical numerical admission である。v0.2 pass は bitwise 一致を主張せず、
既存 bitwise No-Go を書き換えない。bitwise が fail でも、凍結済み artifact-FP32
relative criteria を**後から変更せず**全て通れば、別の意味論で v0.2 pass になり得る。

## Decision Tree

```text
開始
 |
 +-- JSON の SHA-256 が凍結値と一致するか？
 |     |-- no  -> blocked（新 version を設計し直す）
 |     `-- yes
 |
 +-- artifact-FP32 full-model CPU reference は適格化済みか？
 |     |-- no  -> blocked_reference_or_capture
 |     `-- yes
 |
 +-- reference token stream / corpus / artifact / build hash は全て bind 済みか？
 |     |-- no  -> fail / blocked（欠損の種類を receipt に記録）
 |     `-- yes
 |
 +-- matched CK/direct control 3 repetition は完了したか？
 |     |-- no  -> blocked
 |     `-- yes
 |
 +-- candidate 2 repetition は 4096 positions + boundary + prefill を満たすか？
 |     |-- no  -> non-qualifying（閾値を変更しない）
 |     `-- yes
 |
 +-- finite性・identity・logits・final hidden・全 layer hidden・KL・rank が全て通るか？
 |     |-- no  -> fail_relative_fp32_v0_2
 |     `-- yes
 |
 `-- hard top-1 regression 0 件、Wilson / block bootstrap も通るか？
       |-- no  -> fail_relative_fp32_v0_2
       `-- yes -> pass_relative_fp32_v0_2
                      (性能・ABI・promotion は別ゲート)
```

## Risks

| Risk | Mitigation / stop condition |
|---|---|
| strict-FP32 full-model CPU reference が未実装 | 参照を代替値で埋めず `blocked`。CPU-only executor を独立タスクで実装・適格化する。 |
| CPU 時間や容量が不足 | 8-step CPU pilot と容量 preflight を先に行い、実測まで所要時間を主張しない。 |
| reference と runtime の共通バグ | canonical decoder 全 byte test、scale boundary test、reference source/build hash、二回の byte-identical CPU run を必須にする。 |
| candidate が自分の token 列で差を隠す | reference-first teacher forcing と input/token hash binding を必須にする。 |
| aggregate が局所的な破損を隠す | per-stream、boundary、prefill、every-layer の独立 gate と max/P99 を使う。 |
| token position を独立と誤解する | Wilson と block bootstrap を併置し、Wilson 単独を過大な信頼主張に使わない。 |
| failure を見て policy を調整する | JSON hash を evaluation 前に receipt へ bind。変更は新 version と新測定だけに有効。 |

## Next Actions

1. 別タスクで `artifact_fp32_strict_v1` を CPU-only で実装する。既存 projection-only
   F64 reference の流用だけでは適格化しない。
2. canonical decode、scale、F32 operation order、full 40-layer path の unit/integration
   validation を行い、8-step CPU pilot の実測時間と reference hash を記録する。
3. frozen corpus を materialize/hash し、full artifact-FP32 reference token stream と
   tensor capture を一回作る。
4. GPU 窓ではまず matched control を 3 repetition、その後 5 候補を一候補ずつ 2
   repetition で測る。候補が v0.2 pass した場合だけ、独立 confirmation を一回行う。
5. したがって必要な GPU 窓は、最低 **6 窓**（control 1 + 5 candidate）である。
   pass confirmation まで行うなら **6〜11 窓**。各窓の実時間は runner 未実装・未測定の
   ため **未確認**である。

この文書の作成時点では上記 Step 1 以降、GPU capture、候補再評価、activation、campaign、
`git push` は実行していない。
