# vllm-mxfp4 の参照追加・R9700実測

2026-09-20、ユーザー指定の [GGZ14/vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4) を
`reference/vllm-mxfp4/` へ取得した。source、モデル、imageの固定情報は
[identity](../../../../../ci/matrix/vllm-mxfp4-investigation-v1.json)、取得手順は
[source-lock](../../../../references/source-lock.md#vllm-mxfp4-独立調査追加2026-09-20)に記録する。
sLLMへのコード組込みや形式採用ではない。sLLMのMXFP4方針はユーザー回答に従いW4A6を正とし、
Git管理外のsLLM.mdの該当記述とmain-plan末尾の未解決事項を揃えた。

## 構成

- source `31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663`（VERSION 0.12.0）、image `stilldeadcode/vllm-radiance:0.9.3`。
- R9700単一可視、`gfx1201`、ROCm 7.14.0／HIP 7.14.60850、PyTorch 2.11.0、vLLM 0.27.1、TP=1。
- AMD `Qwen3.8-27B-Quark-AWQ-MXFP4` revision `5233554c5fa56afda40150556b95573c2d7d29c0`。
  本体はMXFP4 E2M1、K方向block32、E8M0 scale。実行時のactivationはper-token FP8 E4M3FN＋FP32 scaleであり、OCP MXFP8ではない。
- 上流 `fp8_mtp.py` でMTP 8行列をper-channel FP8へ変換した。target lm_headはBF16のまま。
  元モデルのMTP excludeのmodule/tensor名不整合を回避する上流手順。変換後モデル18.04 GiB。
  変換は既存 `rocm-exl3-investigation:tested` のCPU PyTorchで実行し、実行用imageとは区別してhashを残した。
- DFlash2 FP8 revision `ee0cb26a8279b7910cc28d82a8a3e15e4728d56f`。モデル2種とも公開・Apache-2.0のmodel cardを確認。
- libr4d `b9e42ab7202f53a3bc13d415f5d41481f9ca311b`＋上流rx9 patchを同imageでbuild。
  上流がGDN overflow修正必須とする構成を使用し、NaN sanitizerは0。
- 実際の304行列すべてでRadiance W4A8 kernelが選ばれた。GPU計算のFLA/Triton経路はCPU fallbackではない。
- 上流のsysfs走査は`GPUS=1`の表示をV620と誤認するため、canonical UUID `GPU-a8e9ddefa2d60f55`を明示。
  container runtimeで単一R9700／gfx1201、PCI bus7を確認し、実行時もR9700だけが使用された。
  `KV_MEM=0`で誤表示に基づく既定KV容量pinを使わない。
- 上流launcherのDocker呼出しをローカルwrapperで実行し、localhostの18084番だけへ公開。
  host home／credentialをmountせず、モデルと上流sourceはread-only mount。

## 速度

既存Phase83のtoken ID列（SHA-256 `855240c09609a19b9c1124043b763ecc97e3cfde84a16acfaba445a0f8c83b23`）を
そのまま `/v1/completions` へ送った。8192入力／128出力、temperature=1、top_p=.95、top_k=20、seed=123、
1 warmup＋3 measured。各responseのusageで入力・出力数を確認し、server側prefix cachingは無効。
EOSで早期終了しない固定長計測。API入力はtoken IDsのため上流chat template差は混入しない。
別のchat smokeでは「17×23」に「391」と回答した。

DFlash2 depth7、FP8 KV、BF16 conv／FP16 SSM、maxlen16384、maxseqs1、chunk4096、GPU utilization .90、
V2 model runner、graph有効。上流の動的投機幅とint2候補＋rerank headは既定値のまま。
DFlashのため内部の最大scheduled prefillは4090となる。

| 構成 | TTFT中央値 ms | E2E中央値 ms | decode tok/s | TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| MXFP4・投機なし | 2199.846 | 6168.134 | 32.00129 | 31.24874 |
| MXFP4＋DFlash2 | 2271.073 | 4033.119 | 72.06994 | 13.87541 |

DFlash側のTTFTの3回は2262.318／2273.371／2271.073 ms、E2Eは4024.610／4035.548／4033.119 ms。
decode throughputは `(128-1)/(E2E-TTFT)`、TPOTはその逆数。API SSEの受信時刻であり、sLLMの内部stage時間とは同一ではない。
最初のSSEは各回1 token。生成文全体やraw traceをGitへ含めない。
初回reportのdecode throughputの分子が128だった集計ミスを127へ修正し、同じraw時刻から再集計した。GPU再実行はしていない。

投機なしは同じモデル・KV／SSM・要求条件で、draftと高速verify headを無効にした。
このimageは投機なしではV1 model runnerを選ぶため、DFlashとの差にはrunner選択も含む。
投機なしのTTFTは2222.218／2199.846／2199.544 ms、E2Eは6194.163／6166.684／6168.134 ms。

直前の[Phase87 WU2](phase87-wu2-fp8.md)のR9700 NVFP4は、同じ8192/128でMTPなし21.4268、MTPあり34.4633 tok/s。
今回とは重み、KV（sLLMはMXFP8 E4）、投機方式（BF16 MTP対DFlash2）、計測境界が違うため、engineだけの倍率と解釈しない。

## BF16比KLD

[既存ベンチマーク](qwen38-cross-engine-kld.md)の8入力・2632位置とrepeat129位置、全248077有効語彙を使用。
`KL(llama.cpp BF16 || candidate)`、温度1、単位nats、FP64正規化。paddingを含む248320幅のraw FP32 logitsを保存してから
既存vocab mapでpaddingだけを除外する。BF16参照はV620×2／FP16 KV／chunk64。
保存済みbaselineの短文9本・長文2本すべてを過去のaudit hashと再照合した。
AMDモデルのtext_config全体、tokenizer語彙IDとadded tokensはBF16原本と一致した。

候補はR9700、chunk32、eager、V1 model runner、投機なし、prefix cacheなし。
`RADIANCE_VERIFY_HEAD=0`／`RADIANCE_FAST_DRAFT=0`で全語彙headを使い、部分候補への再正規化は行わない。
速度用のgraph／V2／DFlash経路そのものの品質保証ではなく、既存のteacher-forced KLD条件に合わせた測定である。

| 候補 | KV | GDN SSM | 平均KLD | 中央値 | p95 | 最大 | top1一致率 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vllm-mxfp4 | BF16 | FP32 | 0.12529074 | 0.03043594 | 0.51303832 | 7.38518519 | 86.8921% |
| vllm-mxfp4 | FP8 E4 | FP16 | 0.13202150 | 0.03131504 | 0.51053894 | 14.20231144 | 86.5122% |

比較用の過去値はvLLM公式FP8／BF16 KVが0.01451873、sLLM NVFP4／FP16 KV／chunk32が0.09221768、
EXL3 4bpw／FP16 KVが0.01924106。これらは以前の固定binary・重みの値であり、最新sLLMのKLDを測り直したものではない。
今回の差にはAWQ model artifact、量子化方式、engine、GPU、KV、演算順が含まれる。重み形式だけの誤差へ帰属しない。
また上の2行はKVとSSMの両方が変わるため、差分をKV単独の誤差とは呼ばない。

長文は同じ4,097-tokenの2入力、末尾257位置ずつ（計514位置）、context5120、chunk32。
BF16 KV／FP32 SSMは平均 **0.02066802**、FP8 KV／FP16 SSMは **0.01828082**。
どちらもtop1一致率97.6654%。自然文の平均は0.00007451／0.00006720、codeは0.04126153／0.03649444であり、
平均だけでは入力別の差を隠す。短文主表と混ぜない。

候補のlm_headとembeddingのpayloadはBF16原本と完全一致した。
全9caseのtoken hash／位置列も一致し、先頭8行の同位置MSEは±1位置ずらしより全caseで小さかった。
head／embeddingの取り違えやoff-by-oneはこの差の説明から除外できるが、量子化本体とruntimeの原因分解は行っていない。

4 capture・22 case・計6,550行のraw logitsを再SHA-256検査し、shape、input token列、位置、非有限0を確認した。
短文の英語repeatは両設定でbyte一致した。case別値と集約は[結果JSON](vllm-mxfp4-results.json)。

## 保存先と検証

- モデル: `/home/homelab1/.cache/sllm-models/vllm-mxfp4/`。
- raw logits／KLD JSON: `/home/homelab1/datapool/vllm-mxfp4-20260920/`。
- 実行コマンド、log、速度JSON、source／model hash検証: `.local-artifacts/vllm-mxfp4/`。
- 再起動は同ディレクトリの`env.sh`をsourceして固定upstream `serve-mxfp4.sh --no-enable-prefix-caching --language-model-only`、
  KLDは`run_kld.py bfloat16`／`fp8`（長文は追加引数`long`）。使用したargv全体も各`*-command.json`へ保存。
  KLD比較器は既存`ci/tools/qwen38_kld_compare.py`と固定`cases-v1.json`／`vocab-v1.json`を使用する。
- 既存KLD comparatorの解析的test 7件、Python構文・static検査、Markdown local linkとdiff whitespaceを確認。
- 両API構成と4 KLD captureは正常終了。試験用API containerを停止し、GPU使用を解放した。
- source／libr4dの外部評価用配置は[取込み一覧](../../../../../THIRD_PARTY_NOTICES.md#vllm-mxfp4-investigation-20260920)へ記録。

[対応する計画](../../../../plans/archive/2026/09/11-20/vllm-mxfp4-investigation.md)
