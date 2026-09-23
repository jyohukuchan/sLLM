# Phase 87 探索: MTP draft専用の縮小語彙lm_head

2026-09-23。Codexの利用制限が戻るまでの空き時間を使った最適化余地の探索。
Cinference（NInfer）の調査で見つかった「MTP draft専用に語彙を縮小したlm_head」をsLLMへ適用した場合の効果を、
既存のlogitsとkernel単体計測から見積もった。**実装は行っていない。**

## Cinferenceが行っていること（参照のみ）

Cinference `b74044fb`（Apache-2.0）をsourceとして読んだだけで、コード・データは流用していない。
vLLM系と同じno-copy referenceの扱いで、import logへの記録は不要である。

- 変換時に`--proposal`を付けると、語彙の頻度順位の上位131,072行（既定）だけを本体のoutput headから抜き出し、
  Q4 g64で量子化した「proposal head」と、行→語彙IDの対応表（`proposal/token_ids`）を追加する。
  本体の全語彙headはそのまま残す。
- MTP／DFlashのdraftはproposal headでlogitsを出し、argmaxの後に対応表で語彙IDへ戻す
  （`draft.cpp`の`ProposalHead::Optimized`、`proposal_remap_token_ids`）。target verifyは全語彙headのまま。
- 頻度順位は同梱のcorpus集計（`tools/freq_corpus`）で、special tokenは必ず含める。
  公開性能表のMTP3・DFlashはいずれもこのheadを使っている。

## sLLMでの現状

MTPありでは、draftも本体と同じFP8全語彙lm_head（`K5120,N248320`、1.27 GB）を読む。
幅2なので1 blockあたりdraft 2回（M=1）とverify 1回（M=3）である。
段階6最終profileでの1回あたり時間は次のとおり。

| GPU | draft lm_head（M=1） | verify lm_head（M=3） |
| --- | ---: | ---: |
| V620 | 2.668 ms（`dword8_wave4col32`） | 2.620 ms（`fused_k5120n248320`） |
| R9700 | 2.013 ms（ID103 `dot4`） | 2.347 ms（hipBLASLt） |

同じprofileでは2 runの256 tokenが約103 blockで、1 blockあたり約2.49 token。
draftのlm_headだけで確定tokenあたりV620約2.14、R9700約1.62 ms、通常TPOTの約6.6%／5.8%を占める。
draftのMTP companion（BF16、約0.85 GB/step）よりlm_head（1.27 GB/step）の方が重い。

## 受理率への影響（CPUでの反実仮想計算）

### 方法

Phase86最終のモードP report（両GPU、Tier A 26条件、draft・target全語彙logits保存済み）を使い、
draftのlogitsを語彙集合Sの外だけ除いて`Σ min(p′, q′)`（M1）を計算し直した。
p′（target）は変えない。support変換は本番selectorと同じK20・top-p 0.95で、
`ci/tools/mtp_expected_acceptance.py`の関数をそのまま使った。

Sの行を語彙ID順に並べれば、S内の各行のlogitはM=1 GEMVの列ごとの計算なので全語彙headと同一になる。
selectorの同値時の順序（小さいID優先）も保たれるため、本番のdraft分布は
「全語彙のdraft logitsからS外を除いたもの」と一致し、この計算は本番の予測になる。
ただしモードPのblock進行はdraft top-1で決まるため、既知の近似はM1本体と同じである。

### 語彙順位

mtp-bench-v1の入力と重ならないcorpusだけで作った（sLLMリポジトリの文書・コードは使っていない）。
domainごとに相対頻度へ正規化し、等重みで足した。special tokenは常に含める。

| domain | 出所 | 量 |
| --- | --- | ---: |
| en_web | FineWeb CC-MAIN-2025-05 | 60 MB |
| chat | UltraChat 200k test_sft | 60 MB |
| ja／en_para | JParaCrawl（日英の各側） | 各60 MB |
| zh | `reference/`のLMDeploy・KTransformers中国語文書、Cinference中国語文書 | 0.57 MB |
| code_py／code_cpp／code_other | `reference/`配下のsource（repo間でround-robin） | 60／60／20 MB |
| zh_wiki（v2のみ） | Wikipedia 20231101.zh shard 2 | 60 MB |

v1はzh_wikiなし、v2はzh_wikiを加えたもの。

### 結果（Tier A 26条件の平均、基準は全語彙）

| N | v1 V620 | v1 R9700 | v2 V620 | v2 R9700 | v2最悪条件 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 全語彙 | 0.7763 | 0.7762 | 0.7763 | 0.7762 | — |
| 32,768 | −2.57 pt | −2.66 pt | — | — | — |
| 49,152 | — | — | −1.47 pt | −1.53 pt | −5.48 pt |
| 65,536 | −0.64 pt | −0.69 pt | −1.02 pt | −1.06 pt | −3.65 pt |
| 81,920 | — | — | −0.38 pt | −0.40 pt | −2.09 pt |
| 98,304 | −0.27 pt | −0.31 pt | −0.17 pt | −0.17 pt | −1.38 pt |
| 131,072 | −0.14 pt | −0.17 pt | −0.07 pt | −0.07 pt | −0.60 pt |

語彙ID順に単純に先頭N行を取った場合は、131,072でも−3.45 ptと大きく落ちる。頻度順位が必要である。

失う条件は、中国語の散文（review/zh、trans-en-zh）と、識別子が細かく分かれるcode（C、Go、SQL）に集中する。
v2はzhを補った分、65,536では他domainの枠を奪って悪化し、98,304以上では改善した。
小さいNほど順位の作り方に敏感で、98,304以上はどちらの順位でも損失が0.3 pt未満に収まる。

## 速度の見積り

draft lm_headの時間は、両GPUとも行数にほぼ比例する。V620はWU2 probeの本番経路で
248,320／131,072／98,304／65,536行を2.607／1.420／1.089／0.755 msと実測した。
R9700は本番ID103（`sllm_matmul_fp8_outer_gfx1201_dot4_v1`）を、許可形状だけ足したscratch buildのWU2 probeで
2.003／1.066／0.804／0.673／0.533 ms（N=248,320／131,072／98,304／81,920／65,536）と実測した。
いずれも独立oracle・repeat・guardがPASSで、比例による見込みとの差は1%程度である。

確定tokenあたりの短縮は `2 × 1回の短縮 ÷ 2.49 token/block`。
受理率の低下はtoken/blockを `(1 + a₁ + a₂) × Δ` だけ減らし（a₁=0.810、a₂=0.715）、速度では約1.06×Δとなる。
通常TPOTは段階6最終のV620 32.58、R9700 28.07 msを使った。

| N（順位） | V620 短縮 | R9700 短縮 | 受理率による低下 | 正味 V620 | 正味 R9700 | 追加VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 131,072（v2） | 3.1% | 2.7% | 約0.1% | 約+3.0% | 約+2.6% | 671 MB |
| 98,304（v2） | 4.0% | 3.4% | 約0.2% | 約+3.8% | 約+3.2% | 503 MB |
| 65,536（v1） | 4.8% | 4.2% | 約0.7% | 約+4.2% | 約+3.5% | 336 MB |

いずれも採用基準の1%を大きく超える。損失の小ささと順位への鈍さから、N=98,304を第一候補とする。
追加VRAMは、S行をFP8のまま別bufferへ集める場合の値で、MTP有効時だけ確保する。

## 数値分類の見込み

verifyのtarget分布とp/q受理規則は変わらないため、出力分布は変わらない。
変わるのはdraftの提案分布と受理率だけで、固定seedでの生成token列は変わりうる。
draftのS内の各行は全語彙headと同じ計算のため、本番のdraft分布はこの見積りの反実仮想と一致する。
N1相当と考えるが、draftだけの変更をどう分類するかは台帳の既存規則に合わせて実装時に確認する。

## 実装する場合の要点（未実施）

1. **語彙集合の作成と配布**: 2026-09-23のユーザー決定で、語彙集合は追跡せず生成手順だけを置く
   （Phase 87計画の段階9）。
   JParaCrawlは研究目的の利用条件があるため、本番用の順位では日本語Wikipedia等へ置き換える。
2. **load時にFP8の行とscaleをID順に集める**（MTP有効時のみ）。全語彙headは変更しない。
3. **draft graph**: lm_headを`N行`のheadへ差し替え、fixed K20 selectorを`vocab=N`で動かす。
   supportを書く前にN行の番号を語彙IDへ戻す。verify側のsupport record形式は変えない。
4. **exact shape**: R9700はID103の許可形状へ`K5120,N=|S|`を追加する。V620の`dword8`は任意Nを扱える。
5. **検証**: mtp-bench-v1のM1を実runで取り、この文書の反実仮想値と一致することを確認する（oracle）。
   あわせてM3のdraft ms/blockとM4を記録する。

## SWE-chatの検討（2026-09-23）

ユーザーが見つけた[SWE-chat](https://arxiv.org/abs/2604.20779)を、語彙順位のcorpusとして使えるか確認した。
中身はまだ取得していない（gatedのため、下記）。

- **内容**: 公開repositoryから集めた実利用のcoding agent session。論文時点で6,000 session、user prompt 63,000件超、
  tool call 355,000件。user prompt、agentの応答・tool call、diffとcommitを含む。
  sLLMの主用途（coding agent）と分布が近く、探索で損失が集中したcode（C、Go、SQL）と、
  UltraChatが代わりに担っていた対話の両方を1つで賄える見込みがある。
- **配布**: `SALT-NLP/SWE-chat`（英語、`conversations.parquet` 1.31 GB、`commits.parquet` 1.08 GB等）と、
  多言語のtag（en、ja、zh、ru、pt、ko）を持つ`cfahlgren1/SWE-chat`（transcript JSONL）。
  どちらもlicenseはODC-By、Hugging Faceのgated（同意すると自動承認）で、取得には認証が要る。
- **利用条件**: ODC-Byは帰属表示で利用でき、頻度の集計に使うことに支障はない。
  session内のコードは元repositoryのlicenseに従うが、sLLMは頻度だけを使い本文を再配布しない。
  生成手順には出典とrevisionを記録する。
- **注意点**: 継続的に更新されるdatasetなので、revision（commit）を固定する。
  公開repository由来のため、sLLMリポジトリのsessionが含まれていないかを`repositories.parquet`で確認し、含まれていれば除く
  （mtp-bench-v1の入力と重なるため）。中国語・日本語の量は多くない見込みなので、Wikipediaのdomainは残す。
- **取得に必要なこと**: ユーザーのHugging Face accountで利用条件に同意し、ユーザー自身が取得する
  （tokenをagentへ読ませない）。例: `hf download SALT-NLP/SWE-chat conversations.parquet repositories.parquet --repo-type dataset --revision <commit> --local-dir /home/homelab1/datapool/dataset/swe-chat`。
  取得後、探索と同じ反実仮想でv2順位と比べる。

## あわせて見えたこと

- **MTP幅3の再評価**。a₁=0.810、a₂=0.715で、3段目もおよそ0.65なら1 blockあたり約0.38 token増える。
  Phase 83.5では幅3の方が遅かったが、当時とはdraft費用も、M=4 verifyの経路も変わっている。
  縮小head（とstage 3のcompanion低bit化）でdraftが安くなれば損益が変わるため、両者の後に単回比較する価値がある。
  現行のexact-shape providerの多くはM1〜M3に限られており、M=4 verifyが未最適化経路へ落ちる点が主な費用になる見込み。
- R9700のverify lm_head（M=3、hipBLASLt 2.347 ms）はM=1のID103（2.013 ms）より遅いが、
  差は確定tokenあたり約0.13 ms（0.5%）で、単独では1%基準に届かない。

## 保存先

`.local-artifacts/phase87/mtp-shortlist/`（Gitへ追加しない）:
順位作成`build_ranking.py`・`add_zh_wiki.py`、評価`eval_shortlist.py`、domain別count、
順位`ranking-equal-domain.npy`（v1、SHA-256 `1724e91d…`）・`ranking-equal-domain-v2.npy`（v2、`c0f20d76…`）、
結果`eval-shortlist.json`・`eval-shortlist-v2.json`。
中国語Wikipediaのshardは`/home/homelab1/datapool/dataset/wikipedia-zh-20231101/`（SHA256SUMSとREADME付き）。

計画: [Phase 87](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md) ／
関連: [vllm-mxfp4の分析](vllm-mxfp4-optimization-analysis.md)（P1「MTP draft headの低bit化＋厳密rerank」）
