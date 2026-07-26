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

## 2026-07-26 実現可能性 preflight（基準本文は不変）

この節は frozen JSON を変更しない実測前提の確認記録であり、v0.2 の閾値、corpus、
reference 意味論、判定順序を改訂するものではない。詳細な機械可読 receipt は
[`benchmarks/results/2026-07-26/sq8-fp32-reference/feasibility.json`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/feasibility.json)
に保存した。

- frozen JSON（SHA-256
  `64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`）の
  `scope.model_family` は `Qwen3-14B-FP8` である。本文の 40 layer、vocab
  151,936、4,096 logits の約 2.32 GiB という記述も、この binding と一致する。
- ローカルで v0.2 の artifact predicate を通った唯一の実体は
  `Qwen3-14B-FP8` の canonical `sq-fp8-artifact-v0.2` と同 product の raw
  passthrough package だった。`SQ8_0`、`full_model` coverage、280 pair、163
  passthrough payload、`[128,128]` `BF16` block scale、40 layer を満たすが、9B
  artifact ではない。
- 既存 canonical decoder を CPU-only で実行し、Qwen3-14B artifact の 280 weight/scale
  payload（weight 13,212,057,600 bytes、scale 1,612,800 bytes）を hash 検証した。
  `model.layers.0.mlp.down_proj.weight` の block `[0,0]` を F32 復元した値の SHA-256 は
  `7f48464a20b4ca17092c193a914a344be9b495fba09f9c5a572670136621b391` だった。これは
  decoder の再利用可能性の確認であり、9B full-model forward の測定値ではない。
- ローカルの Qwen3.5-9B artifact は
  `sq-fp8-artifact-v0.1` の部分 overlay であり、48 FP8 tensor、`row_block`、
  256-column、F32 scale である。Qwen3.5-9B text config は 32 layer / vocab
  248,320 で、v0.2 の model family、40-layer reference、canonical block-scale
  意味論と一致しない。従ってこれを入力にした CPU runner は v0.2 主参照にならない。
- このため、要求された 9B の 1-token strict-FP32 full-model forward、8-step
  pilot、peak RSS、4,096 position × 7 stream の外挿は**未実行・未確認**である。
  値を proxy や source model で補っていない。snapshot 時点の host
  `MemAvailable` は 83,132,428,288 bytes であり、Qwen3.5 package の宣言 element
  count から計算した F32 bytes は language model が 31,746,738,176 bytes、全 tensor
  が 38,612,417,472 bytes だが、これは allocation/RSS の実測値ではない。

結論は CPU 性能による可否判定の前段で `blocked_reference_or_capture` である。9B を
v0.2 として評価するには、同 model 用の canonical full-model SQ8_0 artifact と、その
32-layer / vocab 248,320 semantics を固定した新しい gate version が必要である。
既存の Qwen3-14B-FP8 canonical artifact を使う場合は v0.2 の model binding に整合するが、
本 task の「9B」対象を 14B へ変更する明示的な承認が必要である。部分 overlay を用いた
engineering-only CPU pilot、F64 主参照、layer-only reference、GPU reference、または
positions/streams の削減はいずれも v0.2 の適格 capture を満たさず、採るなら新 version の
freeze が必要である。

## 2026-07-26 14B canonical artifact strict-FP32 実測（基準本文は不変）

その後の明示的な対象確定に従い、この節では 9B proxy を一切使わず、v0.2 が bind する
Qwen3-14B-FP8 の既存 canonical artifact をそのまま測定した。frozen JSON、閾値、corpus、
activation、service、GPU は変更・操作していない。機械可読 receipt と content-hash manifest は
[`14b-full-model-feasibility.json`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-full-model-feasibility.json)
および
[`14b-full-model-SHA256SUMS`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-full-model-SHA256SUMS)
に保存した。

### 対象と実装

- artifact は `sq-fp8-artifact-v0.2` / `SQ8_0` / `full_model`、40 layer、280
  quantized pair、163 BF16 passthrough、`[128,128]` BF16 scale であり、content SHA-256 は
  `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147` である。manifest の
  source model は `Qwen3-14B-FP8`、vocab は 151,936 である。
- `crates/ullm-engine/src/sq8_fp32_reference.rs` と
  `ullm-sq8-fp32-reference` を追加した。runtime context、HIP、BLAS、activation
  quantization を呼ばない CPU-only path である。canonical E4M3FN を F32 に decode し、
  declared BF16 `[128,128]` scale を F32 にして各 element に掛け、K 昇順の F32
  `mul_add` で reduction する。F32 causal KV、RoPE、GQA、RMSNorm、residual、SiLU、final
  norm、BF16 LM-head まで 40 layer を実行する。
- 大行列は output-row partition ごとに artifact/package から stream する。保存する
  forward capture は logits、final hidden、全 40 layer post-residual hidden、greedy token と
  各 F32 little-endian payload hash である。seed は 0、thread 数は 64、final executable
  SHA-256 は `581bdc5222ef4c080adeaea5f7248e0cea23ca96e0eee43873f18e0f5fa19b97` である。

### 実測

- isolated `raw-p0001`（token ID 1、position 0）の full-model forward は
  **8.742120321 秒**だった。artifact/package 全 payload hash・finite scan を含む初期化は
  45.190505324 秒、同 process の `/proc/self/status` `VmHWM` は 560,384 KiB、外部
  `/usr/bin/time -v` の maximum RSS は **528,100 KiB**だった。二つは異なる計測器なので、
  片方で置換していない。swap と major page fault は 0 だった。
- `raw-p0001` の prompt 1 token + greedy feedback 8 token を二回 capture した。各 run は
  9 forward、各 forward は 8.81–9.21 秒（r1）および 8.93–9.14 秒（r2）だった。9 position
  全てについて logits、final hidden、40 layer hidden の SHA-256 が byte-identical だった。
  final build でも独立に二回（r3/r4）同じ 9 summary hash を得た。これは full corpus
  coverage ではなく、frozen JSON の feasibility pilot 要件だけを満たす。
- 実行前に artifact 280 pair と bound package 163 payload を全量 hash/finite scan した。
  全 256 E4M3FN byte と canonical finite scanner の一致 test、F32 FMA、RoPE position 0、
  1-token GQA、tie-break、`[128,128]` block boundary、BF16 matvec の解析解 test は PASS。
  さらに real artifact の layer 0 Q projection を既存 CPU F64 projection reference と比較し、
  5,120 values で nonfinite 0、max-abs `2.7418136596679688e-6`、relative-L2
  `1.0365190732913615e-6`、cosine `0.9999999999994604`（既存 projection threshold 内）だった。
  F64 path はここで decoder/scale semantics の cross-check にのみ使い、primary reference には
  使っていない。

### メモリ判定

quantized matrix 13,212,057,600 element と passthrough 1,556,249,600 element を全て F32 に
復元すると論理 weight size は **59,073,228,800 bytes（55.016 GiB）**である。preflight の
`MemAvailable` は 83,205,074,944 bytes（77.491 GiB）だったため、論理 byte 数だけなら
22.475 GiB の差はある。しかし all-resident allocation/RSS は live 開発 host では試しておらず、
実測済みではない。従って「常駐可能」は未確認であり、安全な実装判断は layer streaming である。
実際の strict runner RSS は上記の約 0.50 GiB だった。

### 時間外挿と判定

8.742120321 秒を position 0 の直接率として掛けた値であり、長 context の causal attention 増加、
filesystem contention、capture I/O は含まない。このため以下は全て**楽観値**である。

- 質問文を文字どおり `7 stream × 4,096 position = 28,672` forward とすると、
  **69.626 時間（2.901 日）**である。従ってこの解釈では「数日」であり、実行は不可と判断する。
- frozen v0.2 の実際の primary decode は 7 stream **合計** 4,096 position である。この部分だけの
  直接値は **9.947 時間**である。初期化、長 context、capture を足すため、数時間ではない。
- frozen corpus の prompt prefill + boundary を `sequential_m1` で展開すると 16,437 token-forward、
  要求された M=128 cases を現在の逐次 runner の token-forward 相当で数えるとさらに 12,416、計
  28,853 になる。同じ直接率では **70.066 時間（2.919 日）**である。これは M=128 batch を
  測った時間ではなく、現在の runner を逐次利用する場合の楽観的な規模見積りである。

よって strict artifact-FP32 reference は 1-token/full-model と 8-step feasibility pilot までは
実装・検証できたが、full frozen corpus capture をこの task で開始していない。M=128 checkpoint
capture scheduler も未実装である。v0.2 の status は `blocked_reference_or_capture` のままであり、
この結論は gate の改訂ではない。

### v0.2 外の代替案（提案のみ）

- primary 4,096 decode だけに絞ると prompt/boundary/M=128 相当の 24,757 forward を省く。これで
  4096 sample の Wilson lower bound `99.934%` は保てても coverage が変わるため、v0.2 pass には
  使えない。
- 1,024 / 512 position に縮めると、完全一致時の片側 Wilson lower bound はそれぞれ `99.737%` /
  `99.474%`（4,096 の `99.934%` から低下）であり、stream/position 条件の新 freeze が必要である。
- F64、layer-only、GPU reference は主参照の scalar arithmetic、full-model observables、または
  CPU-only 条件を変える。今回の single projection で F64-vs-F32 max-abs は上記
  `2.742e-6` だったが、これは full-model 差の上限ではなく、代替主参照を正当化しない。

## 2026-07-26 GPU F32 control の位置づけと測定前比較契約

この節は凍結済み v0.2 JSON を変更しない。`artifact_fp32_strict_v1`（CPU）を真値として
GPU F32 control の適格性を検証するための実装・比較契約であり、候補 kernel の評価ではない。
GPU control が CPU と一致しなければ control として使わず、候補の v0.2 判定にも使わない。

- control は `runtime/src/sq8_fp32_gpu_reference_gfx1201.hip.cpp` に隔離する。canonical
  `SQ8_0` の raw OCP E4M3FN byte と raw BF16 scale/parameter byte を GPU 上で直接 F32
  decode し、projection は標準 `hipBLAS SGEMM` の F32 のみ、KV cache も F32 とする。
  CK、WMMA、HIPRTC、production SQ8 dispatcher の API・buffer・kernel は呼ばない。RMSNorm、
  RoPE、GQA causal softmax、residual、SiLU はそれぞれ単純な独立 kernel であり、融合・
  タイル化・candidate kernel の再利用はしない。
- CPU/GPU は同じ immutable input の admission/hash/finite scan だけを共有する。CPU が
  decode した F32 scale、norm、activation は GPU に渡さない。GPU upload は raw file の
  SHA-256 を再確認しつつ行うため、数値 decoder の共有による循環参照にはしない。
- 実行は `HIP_VISIBLE_DEVICES=1` と `ULLM_HIP_VISIBLE_DEVICES=1` を必須とし、filtered
  ordinal 0 の exact `gfx1201` 以外なら fail closed とする。これは R9700 を指す pinned
  token であり、V620/gfx1030 と production service には触れない。

GPU の結果を測る**前**に、比較器
`tools/compare-sq8-fp32-reference-runs.py` の次の契約を固定した。

| 対象 | 判定 |
|---|---|
| input token ID | CPU capture と完全一致 |
| greedy token ID | 全 position で完全一致 |
| logits、final hidden、layer 0–39 hidden | 全 position・全 tensor を個別比較、nonfinite 0 |
| `max_abs` | `<= 2.0e-5` |
| `relative_l2` | `<= 1.0e-5` |
| cosine | `>= 0.999999` |

この数値は既存 SQ8 F32LE GPU differential control と同じ事前値である。CPU strict-F32 と
CPU F64 の実 artifact projection cross-check（max-abs `2.742e-6`、relative-L2
`1.037e-6`）より余裕を持たせ、CPU の K 昇順 FMA と hipBLAS の F32 reduction 順序の差だけを
許す。一方で量子化品質 gate より桁違いに厳しく、layer 間の誤差伝播や input/token の不一致を
平均値で隠せない。測定結果を見てこの値を緩めない。比較結果、GPU capture、CPU capture の
content hash はすべて `benchmarks/results/2026-07-26/sq8-fp32-reference/` に保存する。

## 2026-07-26 gfx1201 GPU F32 control 実装・CPU 突合（非適格）

この追記も凍結済み v0.2 JSON には変更を加えない。結果は **GPU F32 control は v0.2
control として使えない** である。以下の CPU/GPU 比較契約は GPU 実測前に上節で固定済みであり、
失格後に値を緩めて通すことはしなかった。

### 実装と隔離

- 実装は `runtime/src/sq8_fp32_gpu_reference_gfx1201.hip.cpp`、Rust wrapper
  `crates/ullm-engine/src/sq8_gpu_fp32_reference.rs`、専用 runner
  `ullm-sq8-gpu-fp32-reference` に隔離した。optimized SQ8 runtime、CK、WMMA、HIPRTC、
  production dispatcher、candidate kernel を呼ばないことを source test でも確認する。
- canonical artifact の raw OCP E4M3FN payload と raw BF16 scale を GPU 上で直接 F32 に戻し、
  projection は `hipblasSgemm` の標準 F32 のみで行う。raw BF16 passthrough parameter も GPU
  で直接 F32 化する。CPU が decode した値は GPU に渡さない。
- RMSNorm、RoPE、GQA causal attention の score/max、exp/sum、value の各 pass、residual、SiLU は
  simple な別 kernel にした。attention は query head ごとの serial F32 reduction、KV cache は
  device-resident F32 である。projection を融合・tile 化せず、被験経路と構造を共有しない。
- 実行前の device guard は `HIP_VISIBLE_DEVICES=1` と `ULLM_HIP_VISIBLE_DEVICES=1`、visible device
  count 1、filtered ordinal 0、exact `gfx1201` を要求する。実測した device は R9700
  `0000:47:00.0`、total `34,208,743,424` B、GPU upload/finalize 後 free
  `6,689,914,880` B だった。V620/gfx1030、service、active manifest、activation は操作していない。

### 事前固定の CPU strict-F32 比較と結果

CPU `14b-raw-p0001-g8-r1` の 9 position（prompt 1 + feedback 8）を同じ teacher-forced input に
GPU で再生した。比較範囲は各 position の logits、final hidden、layer 0--39 hidden、計
`9 × 42 = 378` tensor である。事前固定の判定は input/greedy token ID exact、nonfinite 0、
各 tensor の `max_abs <= 2e-5`、`relative_l2 <= 1e-5`、cosine `>= 0.999999` だった。根拠は
CPU の K 昇順 FMA と standard hipBLAS F32 reduction のみを許し、CPU F32/F64 projection
cross-check より余裕を持たせる、という測定前の契約である。

- input token と greedy token は全 9 position で完全一致した。GPU replay も 9 position の
  logits/final/layer output hash が完全一致し、nonfinite は 0 件だった。
- しかし各 position に少なくとも 23 tensor の失格があり、全 9 position が不合格となった。
  aggregate は max-abs 最大 `0.0048828125`（position 0 / layer 19）、relative-L2 最大
  `1.4888965980100416e-05`（position 2 / layer 23）、cosine 最小
  `0.9999999998906846`（同じ position/layer）、bit mismatch `3,210,190` である。
- 誤差は layer 0 の max-abs `1.192092896e-7`、relative-L2 `2.963378064e-8` から始まり、
  position 0 では layer 2 の max-abs `2.288818359e-5` で固定 max-abs 条件を初めて超え、
  layer 19 では `0.0048828125` になった。後者は CPU value `12539.7529296875` に対して
  GPU value `12539.748046875`、F32 spacing で 5 ULP である。一方、position 2/layer 23 の
  relative-L2 は上限も超えている。したがって static absolute threshold だけの見掛けの問題
  ではない。

最初の layer が極小差で、residual layers を通ると差が増幅する形は、CPU の固定 K-order FMA と
hipBLAS の内部 reduction order の差と整合する。`hipblasSgemv` に置換した診断は aggregate
max-abs `0.02978515625`、relative-L2 `2.3621352547551214e-05` とさらに悪化した。ただし、
GPU/CPU の libm（RoPE/exp/sqrt）差もあり、**reduction order だけが原因であることは未確認**
である。この原因を完全に分離できていないため、GPU result を CPU 真値の代替にしたり、
比較閾値を結果に合わせて再定義したりはしていない。

ゆえに「9 position では一致しない GPU control を 100 position / full corpus へ拡張しても
適格化の根拠にならない」という停止条件を適用した。CPU の追加 100 position も生成していない。
これは CPU reference の欠陥を意味せず、GPU control が事前の CPU-oracle contract を満たさなかった
ためである。

比較 receipt は
[`14b-gpu-fp32-cpu9-teacher-v1-vs-cpu-r1.json`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-gpu-fp32-cpu9-teacher-v1-vs-cpu-r1.json)
（SHA-256 `aea6640282e5ca78b16f7e145f77598ee4456142c5e46cd6298ead556db51d43`）、
GPU run receipt は
[`14b-gpu-fp32-cpu9-teacher-v1/run.json`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-gpu-fp32-cpu9-teacher-v1/run.json)
（SHA-256 `8bf3fd4996a9859a9f15a03195d9ef2303ef4eeed43767557fdccbff53ecf8e0`）に保存した。

### 性能診断と full coverage の見積り

これは capture を伴わない、非適格 control の性能診断であり、reference 生成ではない。
R9700 だけで 100 forward（position 0--99）を測った。初期化は `60.506401280` s、position 0
forward は `0.614119774` s、position 1--99 に対する最小二乗 fit は
`t(position) = 0.374549014 + 0.000497507391 × position` 秒（R² `0.996947573`、RMSE
`0.000786699` s）だった。実測平均は position 1--9 `0.376705886` s、75--99
`0.417650849` s であり、context cost を position-0 rate だけで無視していない。

28,853 forward を最大 context 4,096 の sequence に配置する場合、position 0 を毎回測った
rate でのみ掛ける context-free scenario は `4.922` h、7 × 4,096 + 181 の最長 sequence に
上の fit を当てる compute-only model は `11.118` h になる。後者の position 4,095 予測は
`2.411842` s である。ただし position 100 以降は未測定で、capture I/O も測っていないため、
この `4.922--11.118` h は保証値ではない。full capture の raw F32 payload は少なくとも
`38.894` GiB（metadata/JSON を除く）で、filesystem overhead は未確認である。空き容量
`2,716,785,868,800` B には収まる。

性能 receipt は
[`14b-gpu-fp32-sgemm-performance-p100-v1/run.json`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-gpu-fp32-sgemm-performance-p100-v1/run.json)
（SHA-256 `45f1868e42e13f8be46d1a3cb5a44b5be4ce1aef0b2453bc116510f5645e42ea`）に保存した。計算だけなら
実用的な見込みはあるが、CPU 突合に不合格なため full coverage reference は生成していない。

### 保存状態

GPU 9-position capture は logits/final/40 layer hidden、token ID、content hash を含み、各
`metadata.json` が全 F32LE payload を hash で束縛する。上位 manifest
[`14b-gpu-fp32-control-SHA256SUMS`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/14b-gpu-fp32-control-SHA256SUMS)
は run receipt、9 metadata receipt、比較 receipt、性能 receipt を束縛する。これは失格診断を
再検証可能に保存するものであり、v0.2 control の追加・置換ではない。

## 2026-07-26 CPU strict-F32 のプロセス並列 throughput と全 coverage capture

この追記は凍結済み
[`sq8-numerical-gate-v0.2-relative-to-fp32-reference.json`](sq8-numerical-gate-v0.2-relative-to-fp32-reference.json)
を変更しない。CPU strict-F32 artifact reference だけを使い、GPU、`ullm-openai.service`、
active manifest、activation/campaign、`/opt/ullm` には触れない。

### CPU-only 実測と選択

ホストは AMD Ryzen Threadripper PRO 3995WX、64 physical core / 128 logical CPU だった。
SMT を使わず logical CPU 0--63 の physical-core set 上で、disjoint affinity、nice 10、seed 0
の artifact-F32 runner を position 0--3 の四 forward / worker で測った。steady aggregate rate は
初期化を除く各 worker の四 forward 合計の最長 critical path、wall rate は full artifact/package
verification と初期化を含む。生の receipt は
[`cpu-parallel-throughput-v1/summary.json`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-parallel-throughput-v1/summary.json)
に保存した。

| threads x processes | steady aggregate forward/s | critical 4-forward s | RSS x processes KiB | 17 case の early-position ECT |
| --- | ---: | ---: | ---: | ---: |
| 64 x 1 | 0.118987 | 33.617 | 561,416 | 67.358 h |
| 32 x 2 | 0.234869 | 34.062 | 596,888 | 34.241 h |
| 16 x 4 | 0.383692 | 41.700 | 667,716 | 19.286 h |
| 12 x 5 | 0.563547 | 35.489 | 670,676 | 13.888 h |
| 10 x 6 | 0.489846 | 48.995 | 706,020 | 16.077 h |
| 8 x 8 | 0.599971 | 53.336 | 809,792 | **13.444 h** |
| 6 x 10 | 0.620157 | 64.500 | 969,584 | 15.777 h |
| 4 x 16 | **0.679429** | 94.197 | 1,556,784 | 21.447 h |

4 x 16 は aggregate forward/s 最大だが、case 内の causal dependency を短縮しないため、4,096
forward case の makespan が長い。frozen 17 case を largest-forward-count-first で配列し、worker
ごとの実測四 forward mean による earliest-completion-time assignment を行うと 8 x 8 が最小だった。
従って capture は 8 threads x 8 processes を選んだ。launch preflight では 768 MiB x 8 の worker
budget と 16 GiB reserve を要求し、`MemAvailable` 60,398,608 KiB から budget 後 37,329,936 KiB
が残った。benchmark RSS x 8 は 809,792 KiB であり、測定時・launch 時とも memory constraint を
満たした。

### coverage、所要時間、基準の扱い

frozen corpus の seven primary stream は合計 4,096 forced-decode position である（「各 stream
4,096」ではない）。prompt forward を加えると primary は 10,409 forward、boundary は 6,028、
sequential M=1 は 16,437、required M=128 chunks/tails は 12,416、合計 28,853 forward になる。
各 forward について logits、final hidden、40 layer hidden と token ID を保存する。F32 raw tensor
payload だけの見積もりは 41,762,524,672 B (38.894 GiB) で、JSON/hash/filesystem overhead は未確認
である。

選択構成の 13.444 h は early position forward-only の計算であり、後半 context の CPU cost、queued
job initialization、capture/hash I/O、filesystem contention は含まない。よって **8--12 h の一晩に
収まることは確認できない**。12 h を既に超えており、long-context CPU cost は未確認なので、この
数字を上限や完走保証としては使わない。

削減案は判断材料としてだけ記録する。M=128 全量を外せば 12,416 forward (43.03%)、long M=128
二 case を外せば 8,192 (28.39%)、primary decode を 4,096 から 1,024 / 512 に短縮しても
3,072 (10.65%) / 3,584 (12.42%) である。前二者は mandatory M=128 coverage を失い、後二者は
one-sided 95% Wilson lower bound を frozen 4,096 sample の `99.934%` から `99.737%` / `99.474%`
へ下げる。primary-only は 18,444 (63.92%) 削減するが boundary/M=128 を失う。いずれも v0.2
条件からの逸脱であり、基準を改訂せず non-qualifying と扱う。

### capture の保全と並列決定性

`ullm-sq8-fp32-reference-corpus` は各 independent case を所有する CPU-only worker である。各
position は `.staging` に content hash 付きで書いた後、atomic rename で publish し、`progress.json`
を position 単位で atomic update する。completion 時には case-local `SHA256SUMS` が全 capture,
token stream, plan, receipt を束縛し、`run.json` が manifest hash を記録する。`--resume` は immutable
launcher plan/binary/gate hash の一致を要求し、既存 capture の hash を検証して KV state を replay
復元するが、capture を再書込みしない。従って中断後に最初から output capture をやり直す必要はない。

8-thread serial (CPU 40--47) と、同時に八 worker を走らせた 8-thread parallel (CPU 56--63) で
`raw-p0001-g1024` の最初の四 causal forward を比較した。metadata 四 file と logits/final/40 layer
hidden の 168 payload、計 172 file は byte-identical (mismatch 0) であり、token は
`1 -> 25 -> 330 -> 16 -> 13` で一致した。receipt は
[`parallel-vs-serial-t8-v1.json`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-f32-parallel-reference-v1/parallel-vs-serial-t8-v1.json)
に保存した。これは process parallelism が fixed 8-thread serial output を変えていない実測証拠である。

実 capture root は
[`cpu-f32-parallel-reference-v1/`](../../benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-f32-parallel-reference-v1/)
である。全 17 case が `run.json` の `status=complete` を持つまで v0.2 full reference は未完成であり、
途中で終了した場合は completed case/position と hash manifest を明記して non-qualifying のままとする。

## 2026-07-26 v0.2 consumer harness と GPU capture 手順

この節は凍結 JSON を変更しない。CPU reference が完了した直後に、同じ artifact-FP32
reference と teacher-forced token stream を control/candidate が再利用できるよう、以下を追加した。

- `tools/evaluate-sq8-gate-v0.2.py` は frozen JSON を**実行のたびに** bytes で SHA-256
  検証する。期待値 `64a43c03…07f72bf`、schema、capture identity、tensor shape/dtype/byte
  count/SHA-256、finite 性のいずれかが違えば拒否する。進行中 reference root は read-only
  で、`index-reference` が外部に index を作る。consumer と launcher は frozen corpus から
  **4,210** position（4,096 primary + 17 boundary + 97 M=128 checkpoint）を再導出して
  set equality を要求するため、途中 index から GPU plan を作らない。
- evaluator は reference に対して control 3 repetition と candidate 2 repetition の logits,
  final hidden、40 layer hidden と final-norm hidden を F64 で再計算する。各 metric/scope の
  worst/threshold と P99/max-abs の position を JSON と Markdown に残すため、失格した tensor
  scope と position を後から追跡できる。Wilson/top-10/hard-top-1 の不一致 position も
  candidate repetition ごとに保存する。
- upper repeat envelope は frozen JSON の “median を超える excess” を
  `max(control) - median(control)`、lower envelope は “median への shortfall” を
  `median(control) - min(control)` として実装した。P99 の補間、bootstrap PRNG、median-control
  tie break は JSON に指定がないため、nearest-rank F64 / SHA-256 seed の NumPy PCG64 / lower
  repetition index を明示的に receipt に記録する。これは JSON の値を補う仮定であり、基準値を
  コードへ焼き込むものではない。
- `ullm-sq8-gate-capture` は private teacher-forced session API を使い、reference が保存した
  token stream のみを次 input にする。通常 serving の sampled token/default selector は変更しない。
  one capture process は sequential M=1 と M=128 chunk/tail の両方を走らせ、必要な 4,210
  position（4,096 primary + 17 boundary + 97 prefill checkpoint）で logits/final hidden、626
  layer-required position で 40 layer + final norm を保存する。raw F32 payload は一 repetition
  あたり 3,157,642,240 B（約 2.94 GiB、JSON/filesystem overhead を除く）である。

### 候補 selector と default 非干渉

`tools/prepare-sq8-gate-v0.2-capture.py` は plan を create-new で書き、`run` は子 process
だけの環境で既知の candidate selector をいったん scrub して plan の値だけを再設定する。control
plan はすべての selector を unset にして `enabled: false` を receipt に残す。従って
`/etc/ullm/served-models/active.json`、systemd、production default は変更しない。

capture manifest は plan SHA-256、capture executable SHA-256/git commit、artifact/fixture/reference
hash、selector fingerprint、device、Cargo feature、HIP guard environment と mode 別 elapsed time を
記録する。evaluator は selector 以外の executable/device/compiler/guard configuration が control と
candidate の全 repetition で一致しなければ fail とする。

| 候補 | capture selector | v0.2 status |
|---|---|---|
| Flash2 staged wave32 | `ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE=1` | full v0.2 capture 可 |
| paged source-tile 128 | tile `128` と `ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1` | full v0.2 capture 可 |
| paged source-tile 256 | tile `256` と同じ evaluation-only bypass | full v0.2 capture 可 |
| handwritten WMMA projection | `rocm-handwritten-projection-gfx1201` + fresh private session | 現状 M=1-only。required M=128 を満たせず blocked/non-qualifying |
| `SQ8_1` W8A8 | なし | frozen scope が `SQ8_0` のため v0.2 対象外。別 quality gate のまま |

source-tile の bypass は `decoder.rs` の exact-`1` opt-in だけである。tile selector 自体が
設定されていない場合には無効で、unset/`1` 以外では従来の direct containment fallback を保つ。
これは既知の multi-tile divergence を再評価するための私有経路であり、default を変更しない。

### 完了後の実行順序

全 reference case の `run.json: status=complete` と token stream hash を確認してから、次の順で走らせる。
artifact/package path は reference plan と同じ immutable input を指定する。

```bash
python3 tools/evaluate-sq8-gate-v0.2.py index-reference \
  --gate docs/plans/sq8-numerical-gate-v0.2-relative-to-fp32-reference.json \
  --reference-root benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-f32-parallel-reference-v1 \
  --output benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/reference-index.json

CARGO_BUILD_JOBS=8 cargo build -p ullm-engine --release \
  --features rocm-ck-gfx1201 --bin ullm-sq8-gate-capture
```

各 repetition は `prepare` で plan を作り、`run` で一つの create-new capture directory を作る。
既存 HIP guard は caller が `1` に設定したものだけを受け入れ、tool は guard/service を変更しない。

```bash
python3 tools/prepare-sq8-gate-v0.2-capture.py prepare \
  --gate docs/plans/sq8-numerical-gate-v0.2-relative-to-fp32-reference.json \
  --reference benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/reference-index.json \
  --artifact /immutable/artifact --package /immutable/package \
  --role candidate --candidate paged-decode-source-tile-128 \
  --output /capture-plans/tile128-r1.json

python3 tools/prepare-sq8-gate-v0.2-capture.py run \
  --plan /capture-plans/tile128-r1.json \
  --capture-binary target/release/ullm-sq8-gate-capture \
  --output /capture-results/tile128-r1
```

最後に capture manifest を evaluator へ渡す。`--allow-incomplete-test-coverage` は current
partial-reference self-test 専用で、full admission には使えない。

```bash
python3 tools/evaluate-sq8-gate-v0.2.py evaluate \
  --gate docs/plans/sq8-numerical-gate-v0.2-relative-to-fp32-reference.json \
  --reference benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/reference-index.json \
  --control /capture-results/control-r1/capture-manifest.json \
  --control /capture-results/control-r2/capture-manifest.json \
  --control /capture-results/control-r3/capture-manifest.json \
  --candidate /capture-results/tile128-r1/capture-manifest.json \
  --candidate /capture-results/tile128-r2/capture-manifest.json \
  --output-json /capture-results/tile128-v0.2-result.json \
  --output-markdown /capture-results/tile128-v0.2-result.md
```

### GPU window の整理

Flash2/tile128/tile256 は同じ `rocm-ck-gfx1201` executable/device/pre-fill configuration で、
candidate selector をすべて disabled にした control output は同一である。従って frozen
control rule を満たす限り control 3 repetition を三候補で共有できる。共有する場合は control
window にも `ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1` を含む全 HIP guard を設定する。
これは selector を有効化せず、tile candidate と guard configuration だけを一致させる。

- admission 最小は **4 isolation window**: standard control triplet 1、Flash2 candidate pair 1、
  tile128 candidate pair 1、tile256 candidate pair 1。各 candidate window 内で二 process repetition、
  control window 内で三 process repetition を連続実行する。
- handwritten M=1 diagnostic を必要とする場合だけ別 build/control を要するため **+1 window**。ただし
  M=128 不足のため v0.2 pass を出せない。`SQ8_1` は +0 window。
- pass した standard candidate が `P` 件なら、独立 confirmation は新しい standard control triplet
  1 window と candidate `P` window、すなわち **+1+P**。従って standard admission + confirmation は
  `4`（pass 0 件）から `8`（3 件すべて pass）window、handwritten diagnostic を含めれば `5`--`9`。

標準 admission は control 3 + candidate 6、すなわち **9 full capture repetition**（raw F32 payload
だけで 28,418,780,160 B / 約 26.47 GiB）である。時間の数値は**未確認**である。新 harness 固有の
full capture は reference 完成前に GPU を使わない方針のため、layer readback と write を含む実測はまだ
ない。既存 p512/g4 serving timing はこの full capture と同じ output/readback 条件ではないので、そこから
時間を推測しない。capture manifest の process total/mode 別 `elapsed_seconds` を使い、最初の standard
control repetitionを追加 window なしの calibration として残りの reservation を更新する。confirmation
追加分は control 3 + candidate `2P` repetition（payload 約 `8.82 + 5.88P` GiB）である。

### reference 完成前の harness verification

GPU は使わず、進行中 reference の read-only partial index（179 position）を control 3/candidate 2 の
test-only alias として consumer に入力した。最新 receipt
[`self-consistency-final-index/result.json`](../../benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/self-consistency-final-index/result.json)
は failure 0 であるが、coverage 179/4,210 のため status は
`test_only_harness_verification` であり admission ではない。最初の candidate logits を意図的に壊した
receipt [`intentional-failure-final-index/result.json`](../../benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/intentional-failure-final-index/result.json)
は expected failure（7 gate）となり、`chat-p2048-g512` M=128 prompt position 127 を明示している。
partial index から plan を作ろうとした receipt
[`reference-incomplete-blocked.json`](../../benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/reference-incomplete-blocked.json)
も、GPU を起動せず `blocked_reference_or_capture` を返す。
