# Phase 88: Qwen3.8単一要求のkernel効率（llama.cpp比）

> 状態: 計画済み・未着手（2026-09-26作成）。Phase 87は2026-09-26に完了した。
> 番号: 2026-09-26のユーザー指示で新設した。旧Phase 88（NVFP4リクエストバッチ処理）はPhase 89へ繰り下げた。

## 目的

Qwen3.8-27Bの単一要求で、llama.cppに対して遅れている重いkernel本体の効率を、decodeとprefillの両方で上げる。
2026-09-26の比較では、decodeはkernel数とgraph内の空白でllama.cppより有利だが、kernel本体の合計が約9.1 ms/token長い。
prefillはllama.cppの約半分〜6割の速度で、行列演算命令（WMMA）の使い方に大きな余地がある。
数値形式（NVFP4／FP8混合とQ5_K_XL、MXFP8 E4 KVとF16 KV）は変えず、kernelの構造だけで差を詰める。
前半（段階1〜5）でdecode、後半（段階6〜11）でprefillを扱う。

根拠:

- [速度とdecode時間内訳](../../../../../history/2026/09/21-30/qwen38-q5kxl-llama-and-sllm-long-prefill-20260926.md)
- [3系統のコード比較](../../../../../history/2026/09/21-30/qwen38-llama-sllm-decode-code-comparison-20260926.md)
- [attentionの改善余地（Phase 87）](../../../../../history/2026/09/21-30/phase87-attention-headroom.md)

## 出発点

### decode（R9700 `gfx1201`、MTPなし、文脈8192、profiler下の1 tokenあたり）

| 区分 | llama.cpp | sLLM | 差 |
| --- | ---: | ---: | ---: |
| 行列積（投影／MLP／lm_head） | 32.006 ms | 36.048 ms | +4.042 |
| Full Attention＋KV append | 1.057 ms | 4.179 ms | +3.122 |
| GDN関連（融合範囲を揃えた値） | 1.414 ms | 2.061 ms | +0.647 |
| kernel本体合計 | 35.784 ms | 44.896 ms | +9.111 |
| GPU graph span | 44.063 ms | 49.598 ms | +5.535 |

通常測定ではllama.cppのdecodeはR9700 25.35 token/s（39.4 ms）、V620 19.93 token/s（50.2 ms）。
sLLMのMTPあり（段階11後、8192/17）はR9700 41.63 ms、V620 49.66 msで、MTPなしのllama.cppにR9700では届いていない。
V620のkernel別内訳はまだ取っていない。

### prefill（8192入力）

| GPU | llama.cpp（Q5_K_XL、F16 KV） | sLLM |
| --- | ---: | ---: |
| R9700 | 962 token/s | 約490〜550 token/s（通常8192/128で14.9秒） |
| V620 | 367 token/s | 約230 token/s（通常8192/128で36.2秒） |

sLLMの値はPhase 87 WU-3P後の通常benchmarkと、2026-09-26の長文診断（chunk 2048で493 token/s）。
長い入力ではattentionの2乗の費用が支配的になり、R9700で32768入力223、65536入力116、131072入力77 token/sまで下がる。

R9700のkernel別内訳（Phase 87段階6の最終profile、chunk 2048の本計測の回、2026-09-24に集計。WU-3PとPaged KV移行の前）:

| 系統 | 時間 | 割合 | 実効性能（概算） |
| --- | ---: | ---: | ---: |
| NVFP4の行列積（MLP 56層、`wmma128x64`） | 7.41秒 | 49.6% | 約33 TFLOP/s |
| full attention（16層） | 4.54秒 | 30.4% | 約2〜3 TFLOP/s |
| GDN（48層） | 1.10秒 | 7.4% | — |
| FP8の行列積（hipBLASLt） | 1.02秒 | 6.8% | 約150 TFLOP/s |
| NVFP4の活性値量子化など | 0.57秒 | 3.8% | FP8の量子化（0.06秒）の約10倍 |
| その他（BF16、RMSNorm、FP8量子化） | 約0.3秒 | 約2% | — |

prefill全体15.27秒（kernel稼働14.95秒）。実効性能は演算量（2×重み数×token数など）を時間で割った概算で、
R9700のFP8 WMMAは約325 TFLOP/s、FP16 WMMAは約160 TFLOP/s（vllm-mxfp4の測定）。
NVFP4の行列積を約2秒、attentionを約0.2〜1秒、NVFP4の量子化を約0.05秒へ縮めれば、R9700のprefillは約4〜5秒
（約1,700〜2,000 token/s）の見込みになる。V620はWMMAを持たないため、kernel別内訳を段階0で取ってから見込みを立てる。

## 目標

- **decode**: 両GPUのMTPなしTPOTでllama.cpp（同じ文脈長、`llama-bench`）と同等以上。MTPありはその上乗せとなる。
- **prefill**: 8192入力のprefill速度で両GPUともllama.cpp以上。R9700はWMMAの活用で1,500 token/s以上を目指す。
- 目標に届かなくても、各段階は共通の採否ルールで個別に採否する。目標未達を理由に形式・数値契約を緩めない。

## 共通の進め方

- 採否は[main-planの共通ルール](../../../../main-plan.md)に従う。新kernelは「重い変更」として、単体のAB/BA全roundで1%以上に加え、
  通常モデル（8192/128、MTPなし・あり）でも1%以上の短縮を要する。decodeの段階はTPOT、prefillの段階はTTFTで判定し、
  もう一方に有意な退行がないことを確かめる。
- 新kernelは還元順が変わるためbitwise一致を前提にしない。独立oracle、finite、BF16比KLDの非悪化で品質を確認する。
  KLDが悪化して大きく速くなる場合はユーザーが判断する。
- llama.cpp（MIT）の構造・コードは流用してよい。copy／adapt／portした場合は、同じ作業内で
  [import log](../../../../../../THIRD_PARTY_NOTICES.md#import-log)へ記録する（`AGENTS.md`）。
- 各段階はまず単体probeで効果を確かめ、見込みが出てから本番のproviderと選択表へ接続する。
  2回不採用になった同じ候補は再計画する。
- exact `gfx1030`／`gfx1201`の両方で扱う。片方だけで採用する場合は、target限定と理由を記録する。
- 融合kernelの開発は`AGENTS.md`の融合の方針（演算は共有device helper、開発と区分別計測は分解したkernel）に従う。

## 段階0: 計測の土台（最初に行う）

1. llama.cppとsLLMの同条件profile（rocprofv3、定常replay 8回、区分別集計）を、scriptとして再実行できる形にする。
   2026-09-26の解析scriptはignored `.local-artifacts/benchmark-20260926/decode-breakdown/`にあるため、追跡対象の`ci/tools/`か
   `docs/development/`の手順へ移す。モデルとtraceはGitへ入れない。
2. V620でも同じ区分表を取る。R9700と主因が違えば、以降の段階の順序をtarget別に入れ替える。
3. prefill（8192入力）の区分表を両GPU・両engineで取る。
4. 基準値として、両GPUの通常TPOT（MTPなし・あり）とprefill速度をsLLM・llama.cppの両方で記録する。

## 段階1: decode attentionのtile化とGQA共有（最優先）

差の比が最も大きく（約4倍）、原因がコード比較で分かっている。MTP verify（M=3）と長い文脈にも効く。

- 現状: 1 waveが1 Q head・1 splitを担当し、同じKV headを6つのQ headが別々に読んで展開する。softmaxの更新をkeyごとに行う。
- 候補A1: llama.cppの`fattn-tile`の構造で、1 WGがKV head 1つに対応するQ head（GQA比6）をまとめて担当する。
  MXFP8 E4のK/V tileを一度だけLDSへ展開し、32 key単位でscore→max→再スケールを行う。
- 候補A2: A1の積和をpacked FP16（half2）で行う。FP32累積のA1と品質・速度を比べ、KLDが悪化するならA1に留める。
- Phase 87段階11のC2-A（既存GQA共有kernel）は1%未満で不採用だった。共有だけではkeyごとの更新費用が残るため、
  この段階ではtile化と共有を必ず組にして評価する。段階11で採用したM=3の行間KV共有（C1）との統合も扱う。
- paged block table（128-token block）とsplit／mergeの既存契約は維持する。
- 見込み: R9700で約−2.5 ms/token。

## 段階2: NVFP4 matvecのK方向分割

- 現状: NVFP4の実効帯域は約490 GB/s（FP8 575、llama.cpp 593）。1 waveが4出力を持ち、VGPRは96。
- 候補B1: llama.cppの`mmvq`のように、1出力（または少数の出力）をWG内の複数waveでK方向に分担し、最後にLDSで還元する。
- 候補B2: 現行の出力方向の構成のまま、VGPRを減らしてoccupancyを上げる。
- まずoccupancyと帯域のcounterを取り、どちらが律速かを確かめてから候補を絞る。
- Phase 87段階1のR9700 M=1候補（SGPR基点hoist）は退行した。その記録を前提にする。
- 見込み: NVFP4が575 GB/sに届けば約−2.5 ms/token。

## 段階3: gate/upの融合

- llama.cppはgateとupの積を同じK loopで計算し、SiLU×upまで1 kernelで行う（Q5_K 64組）。
- sLLMはactivationの量子化を共有した後、gateとupを別kernelで起動する。
- NVFP4のgate/up（56組）を1 kernelにし、SiLU×upを出力側で行う。段階2の新kernelを土台にする。
- backlogのP13（dual-output bundle）をこの段階へ移す。GDNのqkv/z（FP8、48組）も同じ形で扱えるなら含める。

## 段階4: FP8 matvecの残り

- FP8 dot4は約575 GB/sで、llama.cppに近い。R9700のread参照値（Phase 87 WU0）との差を形状別に見て、
  余地が大きい形状だけを対象にする。見込みは約−0.5 ms/token。
- 段階2・3の結果（K方向分割の効果）をFP8にも当てはめられるかを先に確認する。

## 段階5: GDNの並列度

- 現状: 1 head／1 WG（48 WG）で、stateを2回走査する。llama.cppは1 headを32 WGに分けてstateをregisterに持つ。
- 範囲を揃えた差は約0.65 msで、sLLMは融合によって起動数が少ない利点もある。
- head内をWGに分けると、同じkernel内のRMSNormとz-SiLUを分ける必要がある。分割による起動増と並列度の利得を単体で比べる。
- backlogのP8（conv＋recurrentの追加融合）もこの段階で扱う。
- R9700のstate転置だけの変更は過去に退行した（再提案しない候補）。転置単独は候補にしない。

## 段階6: NVFP4 prefill行列積のFP8 WMMA化（prefillで最優先）

- 現状: R9700のNVFP4 prefill行列積（`wmma128x64`）は約33 TFLOP/sで、FP8のhipBLASLt（約150 TFLOP/s）の約4.5分の1。
  prefill時間の約半分を占める。
- 候補: NVFP4の値（E2M1）はFP8 E4M3で誤差なく表せ、NVFP4の16要素blockはFP8 WMMA 1命令のK=16と一致する。
  重みと活性値のE2M1をE4M3へ引き当ててFP8 WMMAへ入れ、blockごとのscale（E4M3 block scale×FP32 tensor scale）を
  FP32の累積時に掛ける。vllm-mxfp4がMXFP4で使う方式と同系統（設計のみ参考、コードはno-copy）。
- tileの形、LDSの二重buffer、weightのdirect-load（Phase 64〜65の知見）を順に評価する。vllm-mxfp4分析のP2
  （A-tiled producer-consumer、fragment order、non-temporal load、LDS padding）もここで扱う。
- V620はWMMAがないため、現行のDP4A経路の効率をcounterで確かめ、余地がある場合だけ候補を立てる。
- 見込み: R9700で7.4秒→約1.6〜2.5秒。

## 段階7: prefill attentionの行列演算化

- 現状: MXFP8 E4 KVのprefill attention（`gqa6_qtile8_w16`）は約1〜2 TFLOP/sで、scalarとwave内加算で計算している。
  8192入力でR9700 4.5秒（約30%）、V620 8.5秒（約24%）。入力が長いほど支配的になる。
- R9700候補: K/VをMXFP8からBF16へ展開し（E4M3の値×E8M0 scaleは、極端に小さいscaleでの下位桁を除きBF16で正確に表せる。端の扱いは着手時に独立oracleで確かめる）、Q（BF16）とともに
  BF16 WMMA（FP32累積）で計算するflash attention。Phase 87段階11のC3では、QをFP8へ量子化したFP8 WMMAが
  品質条件を満たさず不採用だったため、入力を丸めない形を第一候補にする。
- V620候補: 段階1のtile構造（GQA 6 headの共有、LDSへの一度だけの展開、32 key単位のsoftmax）をM>1へ広げる。
  段階11のpacked FP16 dot2は退行したため、dot2単独は候補にしない。
- paged block tableとchunk境界の既存契約は維持する。
- 見込み: R9700で4.5秒→約0.2〜1秒。V620は段階0の内訳を見て立てる。

## 段階8: NVFP4活性値量子化

- 現状: prefillのNVFP4活性値量子化は1回約1.3 ms（M=2048）で、FP8の量子化の10倍以上。R9700で計0.57秒。
- 原因（block amax、scale選択、書き出しのどれが律速か）をcounterで確かめ、FP8量子化並みの速度を目指す。
  段階6でWMMA側が活性値の配置を変える場合は、その配置で直接書き出す。
- 見込み: 約−0.5秒。

## 段階9: GDNのprefill

- 現状: R9700で1.10秒（7.4%）。chunk単位の並列形（delta ruleのchunkwise形式）を行列演算へ寄せられるかは未調査。
- まず現行kernelの構造と律速を調べ、行列演算化の見込みが立つ場合だけ候補を実装する。段階5（decodeのGDN）と共通のhelperを使う。

## 段階10: chunk幅と作業領域

- R9700はchunk 4096以上でprojection-packとMLP gateの作業領域の`hipMalloc`がOOMになる。chunk 2048は1024より3.10%速い。
- 作業領域の再利用・分割でより大きいchunkを通せるかを調べ、速度が上がる場合だけ既定のchunk選択を変える。
- 131072入力では残VRAMからchunk 512を選んでいる。長い入力のchunk選択も同じ規則で見直す。

## 段階11: prefillの残り

- 段階6〜10の後に区分表を取り直し、FP8行列積（hipBLASLt、約150 TFLOP/s）やその他の小さいkernelに余地が残るかを確かめる。
- 目標に届かない場合は、残りの差をllama.cppのprefill区分表と並べて記録し、次の候補をbacklogへ残す。

## 対象外

- 重み・KVの形式変更（NVFP4層の拡大、KVのFP16化など）。速度とKLDのトレードオフになるため、必要ならユーザー判断で別に扱う。
- リクエストバッチ処理（Phase 89）、複数GPU、EXL3、DFlash2。
- V620とR9700以外のtarget。
- 構造的sparsity（2:4など）の命令。Qwen3.8のweightはsparse化されておらず、使う前提を満たさない。

## 完了条件

- 段階0〜11の各候補の採否と、採用分の単体・モデル測定を履歴へ記録する。
- 最終の区分表（両GPU・両engine）と通常TPOT／prefill速度を、出発点と並べて記録する。
- h0と両GPUの公開GPU検証がPASSする。
