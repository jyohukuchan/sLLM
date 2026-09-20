# Qwen3.8-27B cross-engine KLD

2026-09-18完了。40条件のGPU captureと64組のKLD比較を取得した。単一位置のlogit比較ではなく、
共通teacher-forced列・全語彙分布を比較する。再実行は[再現手順](../../../../references/qwen38-kld-reproduction.md)を参照。

今回の比較にはengine/hardware/calibration/weight encodingの差が含まれ、量子化形式だけの誤差と断定しない。
KV拡張は同じ重み・engine内のcontrolを併設する。

## 第一巡

8種類の入力（英語、日本語、中国語、Rust、Python、SQL、算術、JSON）、計2,632位置を主集計とし、
英語129位置のrepeatを別途取得した。公式tokenizerの248,077 IDを全て正規化対象とし、
モデルの248,320出力中のpadding IDだけを除外する。KLDは `KL(BF16 || candidate)`、単位nats。

| 重み／engine | GPU | KV | 平均KLD | 中央値 | p95 | 最大 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| EXL3 3bpw | R9700 | FP16 | 0.05487677 | 0.01397972 | 0.22422886 | 4.76104079 |
| EXL3 4bpw | R9700 | FP16 | 0.01924106 | 0.00422471 | 0.07829403 | 1.88460612 |
| EXL3 5bpw | R9700 | FP16 | 0.00523280 | 0.00117938 | 0.01931599 | 0.45041290 |
| sLLM NVFP4 | R9700 | FP16 | 0.09337046 | 0.02066163 | 0.43735324 | 3.65344475 |
| sLLM MXFP6 | V620×1 | FP16 | 0.06345269 | 0.01500271 | 0.28152186 | 2.46743401 |
| sLLM MXFP8 | V620×1 | FP16 | 0.04695124 | 0.00867330 | 0.17832704 | 5.88682197 |
| vLLM FP8 | R9700 | BF16 | 0.01451873 | 0.00227863 | 0.05046123 | 1.88858868 |

baselineはllama.cpp `bc52a12b38941b0a690ade65fbc5749715224e30`、V620×2、BF16重み、
FP16 KV、layer split 1:1、全層GPU配置、batch/ubatch64、context2048。
EXL3はR9700、既検証gfx1201 patch付き `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`、chunk64。
3/4/5bpwともturboderp公開plain artifactでhead6bit、calibration250×2048、MTPなし。
全9caseのfinite full-logit dumpと正常終了を確認した。

BF16 GGUFは公式BF16 safetensorsのembedding、output、layer0/31/63の代表MLP重み計5tensorと
全byte一致を確認し、公式tokenizerの全248,077 ID対応も一致した。全重みtensorの再比較ではない。
同じBF16の短いchunk8対chunk64の先頭8位置では平均KLD `0.0001895624`、最大 `0.0004340510`。
実行順・精度による差が存在するため、上表を重み量子化だけの誤差と解釈しない。

追加の全case controlでは、BF16重み／FP16 KVを維持してchunk64からchunk1へ変えるだけで
平均KLD `0.0004960073`、p95 `0.0013647860`、最大 `0.1223956617`、top1一致率99.6201%となった。
sLLM NVFP4をchunk1 baselineに対して比べた平均KLDは `0.09262949`（top1 89.8176%）で、
主表のchunk64基準 `0.09337046`との差は小さい。主表の基準自体は変更しない。

sLLM NVFP4も全9caseを成功し、HIP-only、fallbackなし、repeat出力byte一致を確認。
主集計のtop1一致率はEXL3 3/4/5bpwが90.8435%／94.3769%／97.7204%、sLLM NVFP4が89.6277%。
NVFP4は既存verified Qwen3.8 mixed recipeを使用する。重みだけでなく活性値の量子化・処理順もEXL3と異なる。
NVFP4とBF16の実際の`text_config`全34項目は一致し、full attentionのsigmoid gate、GDNのSiLU gate、
RoPE、epsilon、layer配置が現在のnative実装と整合することを追加確認した。
既存NVFP4経路が参照するQwen3.5 semantic lockはgraphの共有構造に用い、実重みは固定hash検証済みのQwen3.8である。
ただしreportの固定recipe revision `57926baca9a82b4d6906b43f2750d55315f5b10f`とHF download metadataの
`9e3d73c76eddb75f795cc24ccfbc5affe41c66bd`は異なる。両revisionのmodel/config/tokenizer payloadは同一で、
READMEだけが異なるため数値比較には影響しないが、取得revisionと検証recipe revisionを同一視しない。
tokenizerの語彙IDは一致する一方、pretokenizer/decoder設定には差がある。今回の共通固定token-ID入力では
各engineの文字列tokenizeを介さないため、この違いは測定入力へ混入しない。
MXFP6/MXFP8は実際のQwen3.8 BF16から変換を完了し、全9caseも正常終了した。
主集計のtop1一致率はMXFP6 90.9195%、MXFP8 91.7933%。
R9700をvLLM診断へ割り当てたため、MXの第一巡にはV620を用いた。異なるGPUの値を同一GPUの比較とは呼ばない。
Qwen3.8独立lock/specを追加し、BF16保存のGDN `A_log`／`norm.weight`をnative ABIのF32へ正確に昇格する。
graphのsource dtypeとresident viewを別々に検証するための限定分岐も追加した。
生成済みGGUFのrecipe文字列は旧family namespace `qwen35:<Qwen3.8 fingerprint>`を持つが、実重み・source fingerprint・
derived lockはQwen3.8由来である。converterの今後の出力は`qwen38:`へ修正した。実行中artifactは変更していない。
core変更のレビューでは、今回使うGGUF／専用NVFP4経路のidentity・dtype処理を確認した。
追加lockの利用範囲は変換とこれらの測定経路であり、generic safetensors直読みのBF16実行は未対応である。
その経路にはGDN scalar昇格処理がなく、既存の転送サイズ検査で拒否される。今回の測定には使っていない。
focused core検査は`qwen38_ --lib`の36件を成功し、GPU artifact必須の5件はignoredとして区別した。
vLLM FP8の最初のGPU試行は重みloadでOOM（28.24GiB allocated、3.20GiB reserved/unallocated）となり、
CPUへの重み保持、GPU上での実行、FLA autotune候補制限、KV dtype指定を調整した。
attempt7のsmoke成功後、全9caseのraw-logit取得と正常終了も成功した。主集計top1一致率95.2508%、repeatはbyte一致。
vLLM `0.21.0+rocm722`、TP1、BF16 model dtype、KV backend引数`auto`で実KV BF16、chunk32、eagerで実行。
CPU weight offload指定8GiB（ログ8.09GiB）、GPU model weights20.08GiBを記録し、全重みGPU常駐とは呼ばない。
モデル演算はGPUで行い、FLA autotuneは各kernelの既存候補を1件へ絞った。採用configは各manifestに保存。
失敗したattemptをGPU PASSに含めない。これにより初期baseline＋候補7構成の取得・集計を完了した。

129位置repeatの同一engine内KLDはbaseline、EXL3 3/5bpw、NVFP4で0。
EXL3 4bpwは平均 `4.02e-9`、最大 `5.18e-7`で、全条件top1一致率100%。
比較器は手計算KL、同一分布、加算offset不変性、語彙padding除外、token/position不一致拒否、
最後の位置だけの場合のNLL欠如を含む7件の解析的テストを成功した。

## KV／処理単位の追加測定

llama.cppは同じBF16重み・V620×2・chunk64でKVだけを変更した。以下はFP16 KVをpとする全2,632位置のKLD。

| KV | 平均KLD | 中央値 | p95 | top1一致率 |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 0.00109742 | 0.00002469 | 0.00087714 | 99.2781% |
| Q4_0 | 0.01338772 | 0.00242388 | 0.04286351 | 95.4027% |

長さ4,097の2入力・末尾257位置ずつ（計514位置）では、llama.cpp Q4_0 KVの平均KLDは `0.00218868`、
top1一致率99.0272%。入力内容・選択位置も異なる長文条件として別集計した。

EXL3はR9700で短文8条件・長文2条件を完了した。次表はllama.cpp BF16に対する平均KLD。

| EXL3重み | KV16/16 | KV8/8 | KV4/4 |
| --- | ---: | ---: | ---: |
| 3bpw | 0.0548768 | 0.0547017 | 0.0610490 |
| 4bpw | 0.0192411 | 0.0191432 | 0.0311137 |
| 5bpw | 0.0052328 | 0.0054916 | 0.0193861 |

4bpwのKV6/6は `0.0200611`、KV4/8は `0.0248433`。
同じEXL3 4bpw・KV16/16をpとするKV設定変更のKLDは、8/8が `0.000144940`、6/6が `0.000945407`、
4/4が `0.0121508`。BF16基準との差を単純に差し引いた値ではない。
長文の4bpwはKV16/16がBF16基準 `0.00335370`、KV4/4が `0.00580072`、
同じ重みのKV16/16基準からKV4/4へのKLDは `0.00266283`。

MXはserial1とblock32/64を区別し、KV E4/E5は同じblock32のFP16 controlを併設する。
block処理は既存の全行logits取得APIを使う教師強制であり、MTP draft予測は行わない。
初回行はFP32 last-logits API、以降のblockはBF16 logitsをFP32へ正確にwidenして保存する。
1-token条件は既存のserial APIを維持する。

次はchunk32の追加条件をllama.cpp BF16に対して比較した平均KLD。非量子化KVはsLLMがFP16、vLLMがBF16。

| engine／重み | GPU | 非量子化KV | E4 KV | E5 KV |
| --- | --- | ---: | ---: | ---: |
| sLLM MXFP6 | V620 | 0.06707940 | 0.05870681 | 0.06680522 |
| sLLM MXFP8 | V620 | 0.04444137 | 0.04701790 | 0.04710216 |
| sLLM NVFP4 | R9700 | 0.09221768 | 0.10358649 | 非対応 |
| sLLM NVFP4 | V620 | 0.09375354 | 未測定 | 0.09450807 |
| vLLM FP8 | R9700 | 0.01451873 | 0.01564250 | checkpoint guardで拒否 |

同じengine・重み・GPU・chunkの非量子化KVをpとする平均KLDも別に計算した。
MXFP6はE4 `0.04182082`／E5 `0.04183631`、MXFP8はE4 `0.01915722`／E5 `0.02266889`、
NVFP4はR9700 E4 `0.05095653`／V620 E5 `0.05326541`、vLLM E4は `0.00535145`。
BF16基準の平均が量子化KVで下がった場合も、このKV設定変更による分布差とは区別する。

MXFP6/MXFP8ともchunk32対64は全caseでKLD 0・top1一致率100%だった。
NVFP4のV620 FP16 KVとR9700 FP16 KV（いずれもchunk32）間のKLDは `0.04646920`。
GPU・実装経路による差が含まれるため、重み形式だけに結果を帰属しない。

長文条件のMXFP6はFP16 KV `0.00861420`、E4 KV `0.00841005`（BF16基準）、
同じMXFP6 FP16 KV基準のE4 KLDは `0.00687248`。
vLLMはBF16 KV `0.00239956`、E4 KV `0.00203316`（llama.cpp BF16基準）、
同じvLLM BF16 KV基準のE4 KLDは `0.00208864`。

## 非対応条件・修正・確認範囲

- sLLM NVFP4＋MXFP8 E5 KVはgfx1201でbackend status 283により拒否された。
  同じGPU上のFP16 controlを含めてV620で実行し、E5の数値を取得した。R9700の成功とは扱わない。
- このvLLM imageはFP8 checkpoint＋E5 KVを明示的に拒否するため、full-model E5 KLDは得られない。
  別に見つけたROCm cache writerのE4書込／E5読出の不一致にはopt-inの限定修正を用意した。
  gfx1201上でE4の全254 finite code、E5の全248 finite code、scale 1/2を独立CPU参照と照合し、
  K/Vともbyte mismatch 0を確認。global platform dtypeはE4のまま維持した。
  符号化単体の成功をfull-model対応へ昇格しない。
- 起動失敗、model load OOM、autotune試行、誤ったGPU UUIDによる失敗を結果フォルダ・ログに保持した。
  失敗をKLD 0やGPU成功として数えていない。

## 完了監査と成果物

`capture-audit.json`は事前に列挙した40条件を検査し、全92,464 raw rows／91,842,641,920 bytesについて
入力token hash、位置、shape、全語彙finite、ファイルSHA-256を確認した。GPU capture後のCPU KLD処理はFP64。
この行数は各構成の合計であり、独立した評価token数ではない。
`comparison-summary.json`／`.csv`は64組のfull-corpus比較を集約し、smokeやprefix派生controlを除外する。
実行終了コードは各execution ledger／ログ、sLLMは各caseのHIP dispatch・fallbackなしでも確認した。
対応しない2条件は独立して記録した。コードは2026-09-18にcommitした。

raw logits、モデルrevisionとhash、各case結果、失敗logは
`/home/homelab1/datapool/qwen38-kld-20260918/` に保存。`SESSION.md`は再開用の作業状況であり、
実行継続の判定には実プロセスを確認する。

## 追記: EXL3 3/4/5bpwの速度（2026-09-19）

KLD測定と同じEXL3 3/4/5bpw（R9700、gfx1201 patch付き`550dcfed…`、コンテナ`qwen38-kld-exl3`）で、
[既存のrunner](../../../../../ci/tools/benchmark_rocm_exl3_large.py)の`exl3` modeを使って速度を測った。
3つのbpwで同じ入力（manifest sha256 `adfe6881…`）を使い、各行1 warmup＋3 measuredの中央値、FP16 cache 4096 token、
`DefaultSampler`・seed 1234、EOSで止めない。prefillは`prompt_tokens`を分子にした`time_prefill`、
decodeは128 tokenの入力の後の63 token（`max_new_tokens=64`）の`time_generate`で計算する。

| bpw | pp128 | pp512 | pp2048 | decode（128入力、64指定） |
| --- | ---: | ---: | ---: | ---: |
| 3.00 | 109.9 | 888.6 | 1494.3 | 28.36 |
| 4.00 | 106.3 | 886.0 | 1372.7 | 27.97 |
| 5.00 | 98.4 | 844.1 | 1475.0 | 24.26 |

単位はtok/s。pp2048は3回のばらつきが10〜16%と大きい。pp128はpp512より所要時間そのものが長く
（約1.2秒対約0.6秒）、Qwen3.5-9Bの前回測定と同じ傾向である。raw結果は
`/home/homelab1/datapool/qwen38-kld-20260918/speed-exl3/`に保存した。sLLM等との同条件比較ではない。

[計画](../../../../plans/archive/2026/09/11-20/qwen38-cross-engine-kld.md)
