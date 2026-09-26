# MTP採用率・実効速度ベンチマーク（mtp-bench-v1）

MTP companionの量子化形式やdraft側の変更を、再現可能に比較するための常設ベンチマーク仕様。
用途はcoding agentと翻訳で、創作は参考値としてのみ計測する。
一度きりの実験記録ではなく、以後の判断で繰り返し使う正本として維持する。

## なぜ作り直すか

従来のMTP比較は、単一promptの8192/128 runでdecode tok/sを直接測っていた。
[Phase85 A16実測](../history/2026/09/11-20/phase85-m1-a16-mtp.md)では、この設計で
10系列中R9700 BF16の1系列だけがcampaign間で 50→55 block、78/100→72/109 に飛び、
decode −9.71%が基準側だけに乗った。この1件が研究全体で唯一のプラス結果を作り、
Stage 3の着手条件も同時に成立させた。候補側の変動はいずれも2.1%以内である。

既存の言語・タスクsuite72 runの分散を分解すると原因が定量的に分かる。

| 量 | 値 |
| --- | ---: |
| 全体平均採用率（提案 平均109/run） | 0.685 |
| 観測SD（72 run） | 0.137 |
| 二項分布だけから期待されるSE | 0.045 |
| 分散比（観測/二項） | 9.54倍 |
| 条件間SD（12条件） | 0.128 |
| 条件間SD（創作を除く9条件） | **0.049** |
| 条件内SD（同一条件6 run） | 0.050 |

採用率のばらつきは出力token数の不足ではなく、条件の違いと生成経路の分岐が支配している。
出力長を伸ばしても二項成分（0.045）にしか効かず、経路分岐はむしろ増える。

## 設計方針

1. **創作を採用判断から外す。** 従来の12条件suiteでは、創作を除くだけで条件間SDが
   0.128→0.049（分散で6.8倍）に下がった。創作は0.441〜0.500、それ以外は0.695〜0.821で
   分布が分離しており、混ぜると平均が創作の混合比で動くためである。
   ただしv1の基準runで実測すると、**この分散削減幅はv1ではもっと小さい**（下記）。
   v1で創作をTier Bに置く主因は用途上の関連性と、最低採用率群を判断から外すことであって、
   分散削減の大きさではない。計測は続け、参考値として併記する。
2. **prompt数を増やし、出力長は伸ばさない。** promptは独立な抽出、
   同一生成内のtoken位置はそうではない。同じGPU時間ならprompt数へ回す方が分散が速く減る。
3. **teacher forcingで経路分岐を消し、受理は期待値で測る。** 固定した committed token列を全系列へ食わせ、
   各提案行で**本番のp/q受理規則の期待値** `Σ_t min(p′(t), q′(t))` を計算する（M1）。
   サンプリング雑音が無く、top-1一致のようにnear-tieを0/1へ離散化しない。
   比較の単位はpromptとする（受理数でブロック境界が動くため、系列間で行集合が一致しない）。
4. **速度と採用率を分けて測り、decode tok/sは導出する。** 下の恒等式が実測で成立している。

```
decode_tokens = Σ_block (1 + accepted_in_block)
decode_time   = blocks × (draft_ms_per_block + non_draft_ms_per_block)
decode_tok/s  = decode_tokens / decode_time
```

V620 BF16の実測で検算すると、blocks 53・accepted 74 → 127 token、
(12.847 + 82.808) ms/block → 127 / (53 × 0.095655) = 25.05 tok/s（実測 25.049）。
近似ではなく恒等式であり、不確かさは採用率だけに集約できる。
per-block時間はcampaign間で±4%以内に安定しており（経路が完全に変わったR9700 BF16でも
draft/blockは−2.5%）、長いrunを要しない。

M1から期待tokens/blockを出すときは、幅2では2本目が1本目の受理時だけ有効なので
`1 + a₁ + a₁·a₂`（`a_k` は提案段kのM1、段間の独立を仮定した近似）を使う。
候補がdraft経路だけを変える場合は、非draft時間を対照の値に固定して比較する
（Phase86ではcatch-upと無関係な非draft時間の揺れ −0.5〜−1.1 ms/blockが、見かけの正味を大きく動かした）。

## v1基準runで観測した分布

凍結したBF16基準列58本のうちBF16側29件から、条件ごとの採用率を集計した実測値。

| 集計 | 条件数 | 平均 | 範囲 | 条件間SD |
| --- | ---: | ---: | --- | ---: |
| 全条件 | 29 | 0.693 | 0.426〜0.933 | 0.1274 |
| Tier A のみ | 26 | 0.719 | 0.479〜0.933 | 0.1059 |
| coding agent のみ | 20 | 0.712 | 0.528〜0.869 | 0.0960 |
| 創作（Tier B） | 3 | 0.463 | 0.426〜0.485 | — |

創作の0.426〜0.485は従来suiteの0.441〜0.500をほぼ再現しており、Tier分けの前提は新しい実測でも成立する。
一方でTier A自体の条件間SDは0.1059と従来の非創作9条件（0.049）より大きい。
v1が生成言語10種とcode/散文軸を意図的に張ったためで、**設計どおりの広がり**である。
条件間分散はペア比較で相殺されるので検出力には影響しないが、
「創作除外で分散が6.8倍縮む」という従来suiteの数字をv1へ持ち込まないこと。

code/散文軸も設計どおり効いている。言語ごとに同一ソースを共有し、タスクだけを変えた比較で、
code優位10条件が平均0.775、散文側10条件が平均0.649だった。内容を固定したうえでの12.6 pt差であり、
単一の数値ではなく軸として読むべき理由がここにある。

## 計測する量

| ID | 量 | 方式 | 主/副 |
| --- | --- | --- | --- |
| M1 | **期待p/q受理率** `Σ min(p′, q′)` | teacher-forced（本番忠実なモードP）、prompt単位ペア | **主指標** |
| M1-top1 | draft top-1一致率 | 同上 | 補助。flip／margin解析専用で、採否判断に使わない |
| M2 | 採用率 | free-running、条件ごとペア | M1が使えない場合の代替／整合確認 |
| M3 | draft ms/block、非draft ms/block | free-running、少数条件 | 主指標 |
| M4 | decode tok/s | M1とM3からの導出（上式と `1 + a₁ + a₁·a₂`） | **採用判断値** |
| M5 | free-running end-to-endのdecode tok/s | 少数条件 | M4との整合の健全性確認のみ |

M5をM4より優先しない。M5とM4が乖離した場合はM4を採り、乖離自体を記録する。

**2026-09-17の定義改訂。** M1は当初からこの仕様で「engine自身の受理規則で測る、argmax一致ではない」と
定義していたが、実装（Phase85 Stage 0、Phase86モードP）はいずれもtop-1一致で受理を数えていた。
Phase86の再解析で、同じデータでもtop-1一致のprompt cluster区間は期待受理率の約3倍の幅になり、
小さい正の効果を見逃すことが分かった。これを受けてM1を期待受理率に確定し、top-1一致はM1-top1として補助指標へ移した。
**この日より前に記録したM1の値はtop-1一致であり、改訂後のM1と比較しない。**
fixture（prompt、凍結列、出力長、幅、KV）は変わらないので、versionは `mtp-bench-v1` のまま据え置く。

## 条件（v1で凍結）

Tier A 26条件、Tier B 3条件。条件表とhashは[manifest](../../ci/matrix/mtp-bench-v1.json)、
組み立て済みprompt本文は `ci/fixtures/mtp-bench-v1/prompts/` にある。

### Tier A — 採用判断に使う（26条件）

**A1 coding agent（20） = 生成言語10種 × 2タスク**

生成言語は C、HTML、Rust、C++(CUDA)、Python、C++(SYCL)、Go、TypeScript、Java、SQL。
各言語は**1本の凍結ソースを共有**し、2条件はタスクと指示言語だけが異なる。
これにより内容の効果とタスクの効果が分離される。

| 生成言語 | code優位条件 | 散文側条件 | ソース | prompt長 |
| --- | --- | --- | --- | ---: |
| C | `implement` / en | `explain` / ja | repo `header_c_compile.c` + `evidence_abi.h` | 約4,626 |
| HTML | `test` / ja | `review` / zh | authored dashboard | 約2,268 |
| Rust | `bugfix` / zh | `refactor` / en | repo `mtp_quantized_sidecar.rs` | 約8,265 |
| C++(CUDA) | `implement` / en | `explain` / ja | authored block-scaled GEMV | 約1,712 |
| Python | `test` / ja | `review` / zh | repo `ci/tools/common.py` | 約8,253 |
| C++(SYCL) | `bugfix` / zh | `refactor` / en | authored block-scaled GEMV | 約1,990 |
| Go | `implement` / en | `explain` / ja | authored inference gateway | 約1,550 |
| TypeScript | `test` / ja | `review` / zh | repo `sllm-api.ts` + `page.tsx` | 約8,254 |
| Java | `bugfix` / zh | `refactor` / en | authored BPE tokenizer | 約1,785 |
| SQL | `implement` / en | `explain` / ja | authored benchmark schema | 約1,561 |

タスクの出現数は implement 4／test 3／bugfix 3／explain 4／review 3／refactor 3。
指示言語は en 7／ja 7／zh 6。コードと識別子は言語を問わず同一で、
**指示文と期待する散文出力の言語だけ**を変える。
code/散文のtoken比が採用率を強く動かすため、各runでこの比を記録し、
単一の数値ではなく `code優位 → 散文優位` の軸として読む。

**A2 翻訳（6）**

`trans-en-ja-tech` / `trans-ja-en-tech` / `trans-en-zh-tech` / `trans-zh-en-tech` /
`trans-en-ja-general` / `trans-ja-en-general`。
技術文は本リポジトリの英語 `AGENTS.md` と日本語履歴文書を使う。
中国語技術文と一般文（en/ja）は対応物が無いため v1 で作成し固定した。prompt長は684〜1,045。

### Tier B — 計測のみ、判断に使わない（3条件）

`creative-en` / `creative-ja` / `creative-zh`。従来suiteと同じ短い書き出し（45〜55 token）を使い、
過去の採用率 0.441〜0.500 との比較可能性を保つ。
報告では必ず Tier A と分けて表示し、集計へ合算しない。

### 入力の与え方

**全条件をchat templateの単一userターンとして与える。** 素のcompletion promptでは、
reviewed modelがinstruction tunedであるため即座にEOSを出す
（v1の検証では `completion_tokens: 1`、`finish_reason: "stop"`、`output_text` が空になった）。
chat template経由が実際のcoding agentの使い方でもある。
`sllm generate --message user:<prompt>` を使い、`--prompt` は使わない。

したがってtoken数は2種類ある。manifestの `prompt_tokens` は素のprompt本文のtoken数（raw）、
runtimeが報告する値はtemplate適用後（rendered）で、templateが約11〜15 token加える。
比較で効くのはrenderedの方で、凍結時に `prompt_tokens_rendered` として記録する。

### prompt長の扱い

**prompt長は条件ごとに凍結するだけで、条件間で揃えない。**
本ベンチマークの全指標は条件内のペア比較なので、条件間の長さ差は比較に影響しない。
結果として684〜8,273 tokenの幅を持ち、short／medium／long contextの各régimeを同時に覆う。
長さを揃えるためのpaddingや文の途中での打ち切りは行わない
（コードは行境界、散文は段落境界で切る）。

### 出力長と反復

- 出力256 token。採用率0.75・幅2なら1 runあたり約102 block・205提案。
  Tier A 26条件で約5,300提案となり、pooledの二項SEは約0.42 pt。
- M1はcommitted列とsampling状態を固定するので決定的であり、条件ごと1回で足りる。
- M2を使う場合のみ 26条件 × 2 seed = 52 run とする。
- M3・M5は代表4条件、1 warmup＋3 measured。
  [Phase 87 WU-3S](../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階3-mtp-companionの形式)は、
  promptごとのM4を導くためM3だけTier A 26条件へ広げる作業固有の例外とする。M5の単一prompt値は採否へ使わない。

## prompt corpusの出所

**本リポジトリの凍結snapshotを優先し、対応物が無い言語だけ作成して固定する。**
リポジトリ由来を優先する理由は、MITの自前コードで来歴上の追加義務がなく、
commit固定で経年ドリフトせず、本プロジェクトのcoding agent利用そのものを代表し、
日本語文書と英語コード/AGENTS.mdという実在の対訳素材を翻訳条件へそのまま使えるためである。

| 区分 | 対象 |
| --- | --- |
| repo参照（`path@commit` とテキストSHA256を保持、本文は複製しない） | C、Rust、Python、TypeScript、翻訳の技術文 |
| v1で作成し固定（`ci/fixtures/mtp-bench-v1/corpus/`） | HTML、CUDA、SYCL、Go、Java、SQL、中国語技術文、一般文en/ja |

repo参照は作業ツリーではなく `git show <commit>:<path>` から解決する。
解決後のSHA256が一致しなければ実行を止める。
組み立ては `ci/tools/build_mtp_bench_prompts.py` が行い、同じcommitとtokenizerから
同じpromptを再生成できる。

## teacher forcing（M1）の定義

前提として、Qwen3.8 NVFP4のCLIは temperature=1.0・ペナルティ0・MTP幅0または2に固定されており、
greedyを選べない。したがって基準列は**固定seedのsampling**で作る。

1. 各条件について、**凍結したcommitted token列**を1本持つ。
   v1では BF16 companion・canonical V620 `gfx1030`・temperature 1.0・seed 123・幅2で生成し、
   fixtureへtoken列とSHA256で固定した。以後の比較で再生成しない。
2. 全系列に同じcommitted列を強制し、**本番忠実なモードP**（Phase86）で進める。
   各ブロックの1歩目はtargetのhidden、2歩目はdraft自身のhiddenで提案し、
   本番と同じ規則で状態を保持／rewindする。全位置をtargetのhiddenで条件付けするモードTは
   本番の条件付けではないので、M1には使わない（下の「Phase86で明確化したM1の条件付けの限界」）。
3. 各提案行で、**engine自身の受理規則の期待値** `Σ_t min(p′(t), q′(t))` を記録する。
   `p′`・`q′` はその行のtarget分布とdraft分布に、参照token selector
   （`native/hip/src/token_selector_kernel.hip.cpp` のtop_k／top_p分岐）と同じsupport変換を施したもの:
   logit降順（同値は小さいtoken id優先）→top-k 20→softmax→累積がtop_p×総和（0.95）に達した候補まで（最低1）→再正規化。
   期待値なのでサンプリング状態を固定する必要はない。
   計算には**draftとtargetの両logitsを全行保存したreport**が必要で、
   draft行とtarget行は同じrun内で `(sequence_index, block_row)` により対にする。
   計算器は `ci/tools/mtp_expected_acceptance.py`（`--self-test` でsupport変換の境界とtie処理を検査）。
4. モードPのブロック進行（どの提案を受理済みとして状態を保持するか）は、draft top-1と次の強制tokenの一致で決める。
   これはハーネスの駆動規則であって受理率の定義ではない。本番はp/q抽選で進むので、
   保持される状態の分布は本番と完全には一致しない（既知の近似）。
5. 受理数でブロック境界が動くため、**系列間でtarget行の計算配置と行集合が一致しない**
   （Phase86ではoffとseparateでdraft行数が236対228、target logitsのhashも異なった）。
   したがってM1はprompt単位の平均で比べ、位置を独立標本として扱わない。
6. 幅2の2本目はdraft自身の1本目に依存する。この依存も測定対象に含め、提案段ごとの内訳も記録する。
7. BF16基準がこの方式で100%になることはない。committed列はtargetを経た出力であって
   draft単体の出力ではないため、BF16 draftも本来の受理率を示す。

**凍結列の導出について。** Qwen3.8 NVFP4のCLI経路は `output_text` を返し token ID列を返さない。
そのため凍結列は `tokenize(output_text)` で導出し、`derivation` フィールドへ明記する。
v1の全58列で `reported_completion_tokens` と導出token数が一致し、
tokenizerのround tripも全列で完全一致した。
またCLIが報告する `prompt_tokens` は、組み立て時にオフラインtokenizerが数えた
manifestの `prompt_tokens` と全条件で一致しており、オフライン集計がruntimeと同じ分割であることを確認している。

**凍結列の来歴について。** 基準列は「forcedな入力」であって、あるbuildの再現対象ではない。
必要なのは、列が固定され、hashされ、全系列が同じ列を使うことだけである。
したがって生成時のbuild identityは来歴として記録するが、
列そのものの再生成可能性は受入条件にしない。
この性質により、`tokenize(output_text)` が元のdecode経路と1 tokenも違わないことも要求しない。

基準列をBF16から取ることでBF16寄りの偏りが入る可能性は残る。
これを打ち消すため、**W8A8基準の第2の凍結列**も併せて保持し、
両基準での受理率差の符号が一致することを確認する。符号が割れた条件は判断へ使わず、
その旨を記録する。

## 検出力

M2は条件内SD 0.050（創作を除く）を用いた保守的な見積り。
M1とM1-top1は、Phase86（Tier A 26条件、catch-up off対separate、prompt cluster bootstrap）で**実測した**95%区間の半幅である。

| 方式 | 条件数 | 95%区間の半幅 | 根拠 |
| --- | ---: | ---: | --- |
| M2 free-running | 9 | 約4.9 pt | 見積り |
| M2 free-running | 26 | 約2.8 pt | 見積り |
| M2 free-running | 52 | 約2.0 pt | 見積り |
| M1-top1 teacher-forced | 26 | 0.69〜0.74 pt | 実測（gfx1030／gfx1201） |
| **M1 期待受理率** | 26 | **0.24〜0.29 pt** | 実測（gfx1030／gfx1201） |

M1はM1-top1の約3倍狭い。Phase86ではこの差で、top-1では区間がゼロを跨いだ
catch-upの効果（gfx1030 +0.43 pt、p=0.006）を検出できた。
Phase85 Stage 0が24〜33%という安定した回復率を出せたのもteacher forcingによるもので、
Stage 2が単一promptに戻したことがその時の弱点だった。

## 実行profile

| profile | 内容 | 概算コスト |
| --- | --- | --- |
| `full` | Tier A 26条件のM1＋Tier B 3条件＋M3＋M5。全系列・両GPU | 5系列×2 GPUで約3〜4時間 |
| `quick` | Tier A から8条件のM1のみ、2系列・1 GPU | 約8分 |

`quick` の8条件はmanifestの `profiles.quick` に固定した
（`code-c-implement-en`、`code-cuda-explain-ja`、`code-rust-bugfix-zh`、`code-python-review-zh`、
`code-go-implement-en`、`code-java-refactor-en`、`trans-en-ja-tech`、`trans-zh-en-tech`）。
2026-09-15以降のteacher-forced測定で実際に使った組で、code優位／散文側、en/ja/zh、short/long context、翻訳を覆う。

`full`は既定変更の判断に使う。`quick`は開発中の退行検知に使い、単独で採否を決めない。
M1はdraftとtargetの全行logits（1行 248,320 f32）を保存するため、モードPのTier A 26条件で**1系列あたり約13 GB**を使う
（Phase86実測）。raw出力は `.local-artifacts/` に置き、Gitへ追加しない。
targetのprefillは系列間で同一なので、条件ごとにprefillを再利用する最適化を実装してよい
（MTP prefix primingはsidecarごとに異なるため再利用しない）。

## ガードレール

- **基準ドリフト検査**: 同一のmodel lock・sidecar digest・binary hash・GPU UUIDなら、
  同じ系列を再実行したBF16のTier A M1（期待受理率）は決定的に再現するはずである
  （Phase86では同一系列の反復で生成token hashが再現した。系列が違えばモードPではブロック配置が変わるので一致しない）。
  差が出たら比較を信用せず原因を先に調べる。free-runningのM2では、記録済み参照値から±1.5 ptを超えて動いたら同様に扱う。
  以前のR9700 BF16異常（free-running）はこの検査で捕捉できた。
- **campaign跨ぎ禁止**: 基準と候補は同一campaign・同一binaryで取る。
  異なるcampaignの基準を分母にしない。
- **経路分岐の明示**: M2/M5では系列間の生成token列一致/不一致を必ず記録する。
  不一致のまま採用率を比較した場合は、その旨を数値の脇に書く。
- **実行契約**: exact target・GPU UUID、HIP-only、非zero dispatch、fallbackなし、cleanup 0。
  CPU emulation、timeout、crash、zero selectionはPASSにしない。
- **version lock**: prompt参照・解決後SHA・凍結committed列・出力長・幅・KV encodingを
  manifestで固定する。いずれかを変えたらversionを上げ、旧versionと数値を比較しない。
  指標の計算定義の改訂（2026-09-17のM1改訂など）はfixtureを変えないのでversionを上げないが、
  manifestの `revisions` に記録し、改訂前後の値を比較しない。

## 採否

2026-09-24以降、量子化MTP形式の既定採否は
[main-planの変更の採否ルール](../plans/main-plan.md#変更の採否ルール2026-09-24ユーザー決定)で判断する。
draft側の変更はM4または同一process AB/BAの通常計測で受理率込みのdecode速度を測り、M1は採用率の診断値として記録する。
Tier Bは参考値として併記する。過去の形式比較と当時の判定は[Phase 87 段階3履歴](../history/2026/09/21-30/phase87-stage3.md)に残す。
WU-3SのNVFP4 companionに限るprefill／TTFT例外も[作業計画](../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階3-mtp-companionの形式)で管理し、このベンチマーク全体の既定条件にはしない。

WU-3SのM3では、Tier A 26条件を各1 warmup＋3 measuredの自由生成で測る。
draft／非draftの時間を分けるrunはwhole-decode graphを無効にし、各proposalの同期的なhost wall時間とdecode総時間を記録する。
M4は各promptのM1の1段目／2段目と、そのpromptのM3 block時間から導く。
このM3は製品のwhole-graph実行時間そのものではないため、採否では同じbinary・target・fixtureの
whole-graph同一process AB/BAも別に測る。M3とwhole-graphの時間差を隠して同一の速度と扱わない。

## v1のM1初回測定（gfx1030、2026-09-15）

> **旧定義の値。** この節と次の2 GPU比較の一致率は、モードTで量子化draftのtop-1がBF16 draftのtop-1と一致した割合
> （改訂後の呼び名ではM1-top1に近い量）であり、改訂後のM1（期待p/q受理率、モードP）ではない。
> 当時の観測として保持し、改訂後のM1と比較しない。

canonical V620 `gfx1030`（UUID `GPU-76a08c022586fed6`）で、Tier Aから8条件
（code優位／散文側、en/ja/zh、short/long context、翻訳2件）を選び、
BF16 companionで強制prefixを生成してから、BF16／MXFP8／MXFP6の3系列を同じ強制prefixで通した。
全系列PASS、非finite 0、HIP-only、state capacity 10,240。

| 条件 | 行数 | MXFP8一致 | MXFP6一致 |
| --- | ---: | ---: | ---: |
| code-c-implement-en | 256 | 0.984 | 0.969 |
| code-cuda-explain-ja | 256 | 0.965 | 0.941 |
| code-rust-bugfix-zh | 256 | 0.965 | 0.973 |
| code-python-review-zh | 256 | 0.945 | 0.965 |
| code-go-implement-en | 256 | 0.977 | 0.980 |
| code-java-refactor-en | 256 | 0.980 | 0.969 |
| trans-en-ja-tech | 256 | 0.980 | 0.977 |
| trans-zh-en-tech | 256 | 0.988 | 0.984 |

- **MXFP8 draftのtop-1がBF16 draftと一致: 1993/2048 = 0.9731（SE 0.36 pt）**
- **MXFP6: 1986/2048 = 0.9697（SE 0.38 pt）**
- 同一位置でのペア比較（McNemar）: MXFP8のみ一致40、MXFP6のみ一致33、
  差 +0.34 pt、SE 0.42 pt、**z = 0.82**。
  つまりこの8条件では**MXFP8とMXFP6のdraft忠実度に有意差は無い**。
  仮数3bitと2bitの差から予想されるほどの開きは観測されなかった。
- logitsは全条件で系列間bit一致せず（重みが違うので当然）。

検出力の実測比較（同じ8条件、同じGPU時間）:

| 方式 | ペアSE |
| --- | ---: |
| free-running（基準runで実測した条件間SD 0.062） | 2.19 pt |
| **teacher-forced（位置単位）** | **0.42 pt（5.3倍タイト）** |

解釈: 量子化によるdraftの忠実度低下は**位置の約3%**である。
採用率への影響はこの3%で上から抑えられるので、高々数ポイントであり、
free-runningの6 pt級のノイズに埋もれる。Phase85で効果が分離できなかったこと、
Stage 0の回復率24〜33%という値と整合する。
8条件の初回測定であり、全Tier Aへは一般化しない。

## gfx1201を追加した2 GPU比較（2026-09-15）

gfx1030で生成した強制prefixをそのまま`gfx1201`（R9700、UUID `GPU-a8e9ddefa2d60f55`）でも使い、
**両targetを同一位置で強制**して3系列を通した。R9700の常駐serviceは測定中だけリースし、
unit／run.sh／binaryのhash一致、healthz／readyz 200、performance level復元を確認した。
非finite 0、HIP-only、fallbackなし。

### GPU内（量子化draft vs 同一GPUのBF16 draft）

| target | MXFP8一致 | MXFP6一致 | ペア差（McNemar） |
| --- | ---: | ---: | --- |
| gfx1030 | 0.9731（SE 0.36 pt） | 0.9697（SE 0.38 pt） | +0.34 pt、SE 0.42、**z=0.82（有意差なし）** |
| gfx1201 | 0.9824（SE 0.29 pt） | 0.9673（SE 0.39 pt） | +1.51 pt、SE 0.39、**z=3.91（有意）** |

**形式の優劣はGPUに依存する。** gfx1201ではMXFP8がMXFP6より有意に忠実だが、gfx1030では区別できない。
gfx1201のMXFP8だけが`make_wave_block32`とnativeな`v_cvt_f32_fp8`を使う経路であり、
この差の候補として記録する（本測定で因果を確定したわけではない）。

### GPU間（同一形式・同一強制位置）

| draft形式 | gfx1030 vs gfx1201 のtop-1一致 |
| --- | ---: |
| BF16 | **0.9521** |
| MXFP8 | 0.9497 |
| MXFP6 | 0.9453 |

**これが本測定の最重要の結果である。**
2つのGPUは互いに **位置の約4.8%** で異なるtokenをtop-1に選ぶ。
一方、同一GPU内で量子化がBF16から外れるのは1.8〜3.3%にすぎない。
つまり **ハードウェア／kernelの差のほうが、BF16→MXFP8の量子化より大きくdraftを動かす。**
入力・位置・prefixは完全に同一で軌跡分岐も無いので、これは純粋に数値・演算順の差である。

この結果は運用上2つの意味を持つ。

- Phase84.5が測定した`gfx1201`のM3/M1 attention加算順差、および
  [言語・タスク比較](../history/2026/09/11-20/mtp-language-task-acceptance.md)で
  「V620の採用率が一貫して低いとはいえなかった」という観測と整合する。
  2つのGPUは単に別の数値を計算している。
- **採用判断ルールの「両GPUで成立」条件は、2つの異なる数値系での成立を要求している。**
  GPU間で採用率の絶対値を比較しない、という既存の方針は維持する。
  形式の採否は各GPU内のペア比較で行い、GPU間の一致は要求しない。

### 不一致の正体はnear-tie（margin分解）

強制位置ごとにBF16 draftのtop1−top2 logit差（margin）でバケットに分けると、
**不一致はすべて近接tie位置に集中している。**

| margin | 位置数 | gfx1030 MXFP8 flip | gfx1030 MXFP6 | gfx1201 MXFP8 | gfx1201 MXFP6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0, 0.25) | 約86 | 0.420 | 0.386 | 0.238 | 0.405 |
| [0.25, 1) | 約263 | 0.062 | 0.105 | 0.056 | 0.112 |
| [1, 2) | 約262 | 0.004 | 0.004 | 0.004 | 0.008 |
| [2, 4) | 約357 | 0.003 | 0.000 | 0.000 | 0.000 |
| [4, ∞) | 約1,080 | 0.000 | 0.000 | 0.000 | 0.001 |

**margin 1 を超えると量子化はdraftの選択を一切変えない。** 全体の53%を占める margin≥4 の位置では flip 0。
不一致位置のmargin中央値は0.125、一致位置は4.4〜4.8で35倍違う。
不一致の85〜96%が最小margin十分位に入る。GPU間のBF16不一致98件も同じ構造（85%、中央値0.219）。

つまり量子化もハードウェア差も、draftを**系統的に劣化させていない**。
ほぼ決めている位置では答えが変わらず、ほぼ引き分けの位置だけを引き直している。

### 復号経路の対照実験（仮説棄却）

「gfx1201のMXFP8が忠実なのは`make_wave_block32`のblock単位scaleとnative `v_cvt_f32_fp8`のため」
という仮説を対照実験で検証した。診断buildだけで`sllm_matmul_mxfp8_w8a8_m1_col2_body`の
gfx1201分岐をgfx1030にも適用し、同じ強制prefix・同じBF16基準・同じdeviceで比較した。
本番sourceはbuild後にhash一致で復元した（`f13689d3…`）。

| 比較 | 結果 |
| --- | --- |
| 対照：BF16 prod vs diag | logits 8/8 bit一致、top-1 一致 1.0000（診断はBF16経路に触れていない） |
| MXFP8 prod（scalar）vs diag（wave） | **logits 8/8 bit一致、top-1 一致 1.0000** |
| 同一BF16基準への一致率 | scalar 0.9731 ／ wave 0.9731（不変） |
| near-tie [0,0.25) flip率 | scalar 0.420 ／ wave 0.420（不変、gfx1201は0.238） |

**仮説は棄却された。** 2つの復号経路はgfx1030で完全にbit一致する。
理由は明確で、`decode_scaled(value, scale)` はE8M0の指数加算をFP32のbit列へ融合し、
`load_wave_block32` は `decode(value) * decode(scale)` を計算するが、
2の冪のscaleでは正常域で両者とも厳密であり、
laneのK割当も同一（thread `wave*32+lane` が両経路で要素 `wave*32+lane+256j` を担当）なので
FP32の加算順も変わらない。異なるのはscale byteの読み方だけで、値は同じである。

したがって**gfx1201 MXFP8の優位はM=1 matmulの復号経路では説明できない。** 残る候補は次のとおりで、未確定として残す。

- gfx1201のBF16基準自体が違う（GPU間で位置の4.8%が不一致）ため、
  各GPU内の一致率は**異なる基準に対する値**であり、GPU間で直接比較できない。
- MXFP8経路のmatmul以外のkernel（activation quantizer、attention route）。
- MXFP6のnear-tie flip率はGPU間で安定（0.386／0.405）なのにMXFP8だけ大きく動く（0.420／0.238）。
  このMXFP8固有の変動はmatmul復号では説明できない。

結果: [gfx1030初回](../../ci/matrix/mtp-bench-teacher-forced-gfx1030-v1.json) /
[2 GPU集約](../../ci/matrix/mtp-bench-teacher-forced-v2.json) /
[margin分解](../../ci/matrix/mtp-bench-margin-decomposition-v1.json) /
[復号経路対照](../../ci/matrix/mtp-bench-decode-path-ablation-v1.json)

## 成果物の置き場

| 種類 | 場所 |
| --- | --- |
| 条件・prompt参照・凍結committed列 | `ci/fixtures/mtp-bench-v1/` |
| 形状・実行契約manifest | `ci/matrix/mtp-bench-v1.json` |
| campaignごとの集約結果 | `ci/matrix/mtp-bench-results-<campaign>-v1.json` |
| 作成した素材（authored corpus） | `ci/fixtures/mtp-bench-v1/corpus/` |
| 組み立て済みprompt本文 | `ci/fixtures/mtp-bench-v1/prompts/` |
| 凍結committed列 | `ci/fixtures/mtp-bench-v1/sequences/` |
| 組み立て・凍結ツール | `ci/tools/build_mtp_bench_prompts.py`、`ci/tools/freeze_mtp_bench_sequences.py` |
| raw出力・profile・build identity | `.local-artifacts/mtp-bench/<campaign>/`（Git追跡外） |

model・binary・raw traceはGitへ追加しない。集約値・hash・文書のみ追跡する。

## teacher forcingは既存だった（2026-09-15の訂正）

当初「未実装」と記録したが、**Phase85 A16作業のStage 0 harnessが既にteacher forcingを実装していた。**
`crates/sllm-hip/src/bin/sllm-phase78-qwen38-benchmark.rs` の `Stage0FixedEntry` が契約を明記している。

```
teacher_forcing_contract:
  "row i consumes output_prefix_tokens[i] with target hidden after prompt plus output_prefix_tokens[..i]"
```

`SLLM_PHASE85_A16_STAGE0=1` ＋ `SLLM_PHASE85_A16_PREFIX_FILE`（schema `phase85-a16-mtp-prefix-v1`）で
強制prefixを与え、各行のdraft logits全量・top1・top5、HIP-only監査を記録する。
`SLLM_PHASE85_A16_STAGE0_GENERATE=1` ＋ suite file で強制prefixの生成もできる。
suiteの `message` にpromptを渡せば、runtime自身のchat templateで描画されるので
オフライン再現のズレが入らない（実測で benchmark と CLI の rendered token数は8条件すべて一致した。
一方 transformers のテンプレート適用は全条件で+40 tokenずれたので、オフライン描画は使わない）。

したがってM1は新規実装を要さない。残件は次の2点だけである。

- native benchmarkの `PromptFixtureKind` は `Legacy`／`Coding8192` の合成token列のみで、
  **M3（per-block時間）の測定に実promptを流せない**。M1はStage 0経路で取れるが、
  M1とM3を同じ経路で揃えるにはここが必要。
- code/散文token比の記録。

## greedyは使えない（測定済みの制約）

外部エンジン（vLLM／SGLang／llama.cpp）はいずれもgreedyで採用率と出力一致を検査している。
greedy投機デコードでは受理規則が「draft提案 == targetのargmax」に退化し、
committed列がdraftの質から独立するため、軌跡分岐が原理的に起きない。

**sLLMではgreedyと投機デコードを併用できない。** 2か所でgateされている。

| 場所 | 制約 |
| --- | --- |
| `crates/sllm-cli/src/model.rs` | Qwen3.8 NVFP4 CLIはtemperature=1.0、penalty=0、MTP幅0／2／3／4に限定する |
| `crates/sllm-hip/src/bin/sllm-phase78-qwen38-benchmark.rs:1077` | `SLLM_PHASE83_MTP=on requires SLLM_PHASE83_SAMPLING=gpu-fixed` |

`decode_mtp_argmax` はdraft側に存在するが、target検証側は
`decode_block_with_mtp_state_and_logits` でlogitsを読み戻す確率的受理規則で書かれている。
greedy受理規則の追加は本番decode経路の変更であり、本仕様の範囲外とする。

その結果、「greedy自由生成でdraft AとBのcommitted列が一致するか」という検査（主張B）は実行できない。
ただし**teacher forcingは強制prefixで位置を固定するので、主張Bを前提としない**。
全系列が同じ行・同じバッチ配置を通るため、軌跡安定性を仮定する必要がそもそも無い。

[MTP companion量子化](mtp-companion-quantization.md) /
[Phase85 A16実測](../history/2026/09/11-20/phase85-m1-a16-mtp.md) /
[言語・タスク採用率の従来suite](../history/2026/09/11-20/mtp-language-task-acceptance.md)

## Phase86で明確化したM1の条件付けの限界（2026-09-17）

既存Stage 0のM1は、すべての位置をtargetのhiddenで条件付けする1歩目型の
teacher forcingである。本番のQwen MTPは2歩目以降にdraft自身のhiddenを使い、
検証後もその条件付けで作った受理位置の状態を保持する。このため既存M1の
量子化draft対BF16のtop-1一致率は、本番のp/q受理率や2歩目の提案品質そのものではない。
本番と同じブロック遷移・保持／rewindを測るPhase86のモードPと区別する。
既存M1の測定結果は保持し、catch-upの効果や本番採用率へ読み替えない。

### Phase86の再実行入口

Phase86対応の `sllm-phase78-qwen38-benchmark` は次を受け付ける。

| 環境変数 | 動作 |
| --- | --- |
| `SLLM_PHASE86_PREPARE_PREFIX_ONLY=1` | GPU接続前に26条件のprefixを準備する。推論による新規生成は行わない。 |
| `SLLM_PHASE86_BENCH_MANIFEST` | `mtp-bench-v1.json` の絶対パス。prompt／sequence file／token列／件数を検証する。 |
| `SLLM_PHASE86_PREFIX_OUTPUT` | 準備したprefixの出力絶対パス。 |
| `SLLM_PHASE85_A16_PREFIX_FILE` | 補完時に保持する既存8条件のprefix。元のtoken列とseedを維持する。 |
| `SLLM_PHASE86_MODE=T` | 全位置をtarget hiddenで条件付けする従来M1型の基準。 |
| `SLLM_PHASE86_MODE=P` | 幅2のhidden連鎖・保持／rewindを測る強制列診断。 |
| `SLLM_PHASE86_PREFIX_FILE` | 準備済みの同一prefixを各mode／GPUへ渡す絶対パス。 |
| `SLLM_QWEN_MTP_CATCH_UP=separate` | P診断および通常Qwen MTPで分離catch-upを有効にする。省略時は既存動作。 |

prepare-onlyには既存の `SLLM_PHASE78_MODEL_PATH` も必要である。
manifestのtoken digestは十進CSV UTF-8、runtime reportのtoken digestはLE i32であり、
同一の文字列形式として比較しない。既存8列のうちPython review列はmanifestのBF16列と
異なるが、Phase86では指定されたgen2列を優先する。凍結列の自然終了による短い列も保持する。

[実行runner](../../ci/tools/run_phase86_mtp_catch_up.py) は `--target`、`--binary`、
`--jobs`、`--output` を受け取る。jobsは `name` と `env` を持つJSON配列で、
envは上記を含む `SLLM_*` の上書きだけを受け付ける。GPU UUID、実行設定、binary／report hash、
serviceの初期状態と復元、performance levelを記録する。raw出力先は新規directoryとする。

[集計tool](../../ci/tools/analyze_phase86_mtp_catch_up.py) の `--forced-pair OFF ON` と
`--timing-pair OFF ON` で、prompt単位の差・bootstrap区間・sign-flip検定と導出速度を得る。
`--target-reference T P` は同じ絶対位置の1／2歩目を比較する。
TはM1 target、PはM3 verify／replayを使うので、T/P差にはtarget演算形状の差も含まれる。
Pのoff/onでも受理数に応じてblock境界が変わるため、margin比較は共通位置・同じproposal stepに
限定し、そのcoverageを記録する。位置を独立標本として有意性を判定しない。

Pの受理数はdraft top-1と次の強制tokenの一致から決める診断値であり、本番のp/q受理数ではない。
導出速度もその診断値と自由生成のper-block時間からの推定であり、本番throughputの保証には使わない。

**2026-09-17追記。** モードPのreportはdraftとtargetの両logitsを保存しているので、
上記の受理数とは別に `ci/tools/mtp_expected_acceptance.py --baseline <P report> --candidate <P report> --out <json>`
で改訂後のM1（期待p/q受理率）を計算できる。受理数はブロック進行を決める駆動規則としてだけ使い、
採用率の評価にはM1を使う。Phase86の最終campaignでの結果は
[Phase86履歴の追記](../history/2026/09/11-20/phase86-mtp-catch-up-conditioning.md)にある。

## Phase87 段階12: 幅2・3・4のv2測定

[`mtp-bench-v2`](../../ci/matrix/mtp-bench-v2.json)はv1のTier A 26条件とfixture hashを参照し、
promptや固定列を複製しない。既定幅は2、明示選択は2／3／4。v1の幅2報告は履歴証拠として保持する。

- M1は`SLLM_PHASE87_STAGE12_M1=1`、`SLLM_PHASE86_MODE=P`、
  `SLLM_PHASE83_MTP_WIDTH=2|3|4`で同じ凍結prefixを実行する。targetは幅＋1行、draftは幅行。
  最後のtarget bonus行を提案受理率へ含めない。`row_matches`は強制列top-1診断値で、
  採否に使う各段の期待p/q受理率は[幅別M1集計](../../ci/tools/phase87_stage12_acceptance.py)でlogitsから求める。
- M3は[v2 suite生成](../../ci/tools/phase87_stage12_m3_suite.py)で26入力を固定し、
  `SLLM_PHASE87_STAGE12_M3=1`、幅2／3／4、各1 warmup＋3 measured、通常のGPU fixed samplerで実行する。
  suiteのschema・v2 manifest hash・幅集合が一致しなければ測定を拒否する。
- [幅別M4集計](../../ci/tools/phase87_stage12_m4.py)は段別受理率から
  `1+a₁+a₁a₂+…+a₁…aₙ`を計算し、M3のdraft／non-draft／合計時間で速度を出す。
  [幅間比較](../../ci/tools/phase87_stage12_width_compare.py)は同じ26 promptを対にして
  幅3・4を幅2へ比較する。これはWU-12Dの一次判定であり、既定の変更には通常の複数prompt
  AB/BAも確認する。
