# 最新llama.cppのMTP量子化比較

2026-09-14のユーザー指示により、V620／R9700と計測開始時点の最新版llama.cppで、MTP量子化のBF16比を測定した。
**今回の範囲では、両GPUとも量子化MTPがBF16 MTPより高速だった。** 専用8行列だけでも改善し、MTP側のembedding/headまで量子化した場合は改善幅が大きかった。

## 結果の要約

数値は同条件の各3反復のdecode tok/s中央値から求めたBF16比を、条件間で幾何平均した高速化率。
「専用8行列」は12条件、「ファイル全体」は代表2条件で、BF16の基準となるembedding/head精度も異なるため分けて読む。

| 量子化範囲 | GPU | Q8_0 | Q6_K | Q4_0 |
| --- | --- | ---: | ---: | ---: |
| 専用8行列・共有部分固定 | V620 | +3.17% | +4.02% | +4.44% |
| 専用8行列・共有部分固定 | R9700 | +2.46% | +3.28% | +4.38% |
| MTPファイル全体 | V620 | +10.06% | +12.22% | +16.94% |
| MTPファイル全体 | R9700 | +10.93% | +10.75% | +13.94% |

BF16と量子化の84比較（専用行列72＋ファイル全体12）はすべて中央値でBF16を上回った。
同じMTP幅・同じ入力条件のBF16 MTPと、量子化MTPの最終生成token列も84比較すべてで一致した。
全128構成で3 measured反復のtoken列は安定していた。これは今回のfixture／seedにおける観測で、未測定条件の保証ではない。

E2Eの相対速度（BF16 HTTP wall / 量子化HTTP wall）も以下のとおり改善した。これらは時間短縮率そのものではない。

| 範囲 | GPU | Q8_0 | Q6_K | Q4_0 |
| --- | --- | ---: | ---: | ---: |
| fixed | V620 | +2.90% | +3.29% | +3.67% |
| fixed | R9700 | +1.71% | +1.89% | +2.94% |
| full | V620 | +6.80% | +7.09% | +10.38% |
| full | R9700 | +6.04% | +5.53% | +8.20% |

## 条件と量子化の範囲

- Source: [llama.cpp `bc52a12b38941b0a690ade65fbc5749715224e30`](https://github.com/ggml-org/llama.cpp/tree/bc52a12b38941b0a690ade65fbc5749715224e30)。2026-09-14開始時のmasterをremote照合し、変更しないcheckoutからexact HIP Releaseをbuildした。
- ROCm 7.14／AMD clang 23、exact gfx1030（V620）／gfx1201（R9700）、各UUIDで単独可視化。
- 本体は固定Qwen3.8-27B UD-Q5_K_XL。FA on、KV f16、context9216、batch2048／ubatch512、parallel1、全target/draft layer GPU指定、fit off。
- 各構成1 warmup＋3 measured。temperature1、top_k20、top_p0.95、min_p0、seed123、repeat_penalty1。固定出力長のためignore_eos=true。
- prefix cacheとcache RAMを無効化し、全requestでprompt/cache countと出力長を確認した。
- MTPはupstreamのgreedy draft経路。n_min0／p_min0、n_max1/2/4。幅のrequest単位overrideは無効なrevisionなのでserverを再起動して変更した。

専用行列比較では、original BF16 GGUFのblk.64にある8 matrixと7 F32 normを抽出した。
token_embd.weight（Q5_K）、output.weight（Q6_K）、output_norm.weight（F32）は固定本体からbyte単位でコピーした。
外部MTPモデルのglobal embedding/headはtargetの実体へのaliasではないため、全形式で同一payloadになるよう明示的に固定した。
Q8_0/Q6_K/Q4_0への変換後、8行列だけの型変更と、norm／共有3 tensorの計10 payloadが不変であることをhash照合した。

ファイル全体比較は、original BF16のembedding/headも含む18 tensorを基準に、10 matrixすべてを同時量子化した。
8個のnormはF32で維持した。通常の外部MTPファイルを丸ごと量子化する場合に近い比較で、専用8行列の結果と混同しない。

専用行列の条件は、入力128/2048/8192 × 幅1/2/4 × 出力128の9条件に、2048入力／512出力／幅2、
日本語推論と英語文章の2048入力／256出力／幅2を加えた12条件。ファイル全体は128/8192入力・128出力・幅2の2条件。
各laneでMTP offの一意条件も測定し、GPUごとに54＋10＝64構成、両GPU合計128構成・512 request（128 warmup＋384 measured）を完走した。

## 専用8行列の各条件

セルはdecode tok/sの中央値。採用数・MAD・範囲・E2Eは[全128行CSV](llama-mtp-quantization-results.csv)と[追跡JSON](../../../../../ci/matrix/llama-mtp-quantization-results-v1.json)に記録した。

| GPU | 条件（入力/出力/幅、task） | BF16 | Q8_0 | Q6_K | Q4_0 |
| --- | --- | ---: | ---: | ---: | ---: |
| V620 | 128/128/1 coding | 29.74 | 30.57 | 31.04 | 30.34 |
| V620 | 2048/128/1 coding | 30.67 | 31.49 | 31.64 | 31.34 |
| V620 | 8192/128/1 coding | 28.84 | 29.22 | 29.33 | 28.97 |
| V620 | 128/128/2 coding | 32.58 | 33.71 | 33.89 | 34.14 |
| V620 | 2048/128/2 coding | 34.61 | 35.83 | 35.38 | 34.98 |
| V620 | 8192/128/2 coding | 33.58 | 34.57 | 34.91 | 34.59 |
| V620 | 128/128/4 coding | 26.75 | 27.21 | 28.67 | 29.12 |
| V620 | 2048/128/4 coding | 27.04 | 28.10 | 28.42 | 29.45 |
| V620 | 8192/128/4 coding | 26.31 | 27.63 | 27.93 | 29.00 |
| V620 | 2048/512/2 coding | 34.62 | 35.71 | 35.46 | 35.70 |
| V620 | 2048/256/2 ja_reasoning | 31.30 | 32.73 | 32.64 | 33.20 |
| V620 | 2048/256/2 en_prose | 34.20 | 35.23 | 35.48 | 35.27 |
| R9700 | 128/128/1 coding | 35.45 | 36.37 | 36.37 | 36.66 |
| R9700 | 2048/128/1 coding | 35.19 | 36.05 | 36.19 | 36.33 |
| R9700 | 8192/128/1 coding | 34.84 | 35.75 | 35.84 | 36.50 |
| R9700 | 128/128/2 coding | 43.46 | 44.26 | 44.45 | 44.74 |
| R9700 | 2048/128/2 coding | 41.54 | 41.72 | 43.28 | 42.83 |
| R9700 | 8192/128/2 coding | 41.19 | 41.88 | 41.79 | 42.03 |
| R9700 | 128/128/4 coding | 34.97 | 36.26 | 35.69 | 36.81 |
| R9700 | 2048/128/4 coding | 35.21 | 36.41 | 36.54 | 37.76 |
| R9700 | 8192/128/4 coding | 34.80 | 35.92 | 36.21 | 35.85 |
| R9700 | 2048/512/2 coding | 38.40 | 39.32 | 40.07 | 41.26 |
| R9700 | 2048/256/2 ja_reasoning | 42.48 | 43.57 | 44.51 | 45.25 |
| R9700 | 2048/256/2 en_prose | 43.18 | 44.37 | 44.98 | 44.83 |

## MTPファイル全体の各条件

| GPU | 条件（入力/出力/幅、task） | BF16 | Q8_0 | Q6_K | Q4_0 |
| --- | --- | ---: | ---: | ---: | ---: |
| V620 | 128/128/2 coding | 30.13 | 33.16 | 33.50 | 35.60 |
| V620 | 8192/128/2 coding | 30.59 | 33.67 | 34.64 | 35.40 |
| R9700 | 128/128/2 coding | 40.32 | 44.46 | 44.30 | 46.42 |
| R9700 | 8192/128/2 coding | 37.52 | 41.87 | 41.88 | 42.30 |

## 解釈とsLLMとの関係

出力128のcoding条件では、両GPU・全形式・3入力長で、幅2が幅1/4より速かった。これはn_min0／p_min0の今回の幅比較に限定する。

MTP量子化がBF16より遅くなることは一般則ではない。今回のllama.cppでは、共有部分を固定した8行列だけでも改善した。
ただしビット数と速度は単調ではない。例えばV620の幅1ではQ6_KがQ4_0より速く、Q4_0の採用数低下も伴っていた。
一方V620の8192入力／128出力／幅4ではQ4_0が+10.20%で、採用率はBF16の44.51%から46.07%へ上がった。
R9700の2048入力／512出力／幅2ではQ4_0が+7.43%、採用率60.87%→62.83%だった。計算時間と検証回数の双方が実効速度へ影響する。

MTP側embedding/headまで量子化すると改善幅はさらに大きいが、これは同じ8行列だけの改善率ではない。
sLLMではshared headが既にFP8で、今回のLLAMA全体BF16 sidecarと基準が異なる。
またllama.cppのQ8_0／Q6_K／Q4_0はsLLMのMXFP8／MXFP6と別形式で、最新版にMXFP8／MXFP6のggml型／quantizerはない。
本体Q5_K_XL対NVFP4、KV f16対MXFP8、draft sampling、EOS設定も異なるため、絶対tok/sでengineを公平に順位付けする比較ではない。
比較できるのは各engine内のBF16 MTPに対する量子化効果であり、今回の改善率をsLLMのMXFPへそのまま移す根拠にはしない。

MTP offも測定したが、temperature1の今回の設定ではMTP on/offの生成token列が一致しない条件がある。
MTP offと一致したのは112 MTP構成中32構成で、off比は同一出力の比較とは扱わない。
主結果の量子化対BF16 MTPは同じ幅・条件で比較し、84/84で生成token列が一致した。
MTP draft自体の品質や別fixture・seed・samplingの結果はこの一致から一般化しない。

native APIにdraft専用walltimeはないので推定値を作っていない。主指標はnative predicted_per_secondで、
通常は128出力でも初回prefill選択分を除いた127 tokenをpredicted_msで割る。この値を128/時間へ置き換えていない。
各3反復の全値を保持し、warmupだけを除外した。MAD比較は記述的なばらつきの目安で、有意差検定ではない。

## 検証・保存先・復帰

両GPUでBF16／Q8_0／Q6_K／Q4_0の小規模MUL_MATをCPU backend参照と照合し、各17/17ケースをPASSした。
全モデル・全logitのCPU再現を行ったという意味ではない。実推論ではexact UUID、GPU telemetry、MTP提案／採用カウンタ、
入力・出力長・cache条件、終了codeを確認した。すべての128構成で3 measuredが有効で、失敗や条件違反はない。
全server groupは自PIDへのSIGTERMでexit0、R9700の元serviceはunit／binary／run.sh hash一致、healthz／readyz 200へ復帰した。
両GPUのperformance levelは元のauto。Qwenローカルserviceは開始時から停止中で維持した。

準備時の--include-weights指定はimatrixの選択であり、重み量子化の対象限定には使えなかった。失敗logを残し、
共有typeを明示する正しいCLIで生成し直した。失敗artifactは測定に使用せず、全8 GGUFと固定本体のhashを最終再照合した。
元モデル・sLLM runtimeは変更していない。upstream sourceは測定後もcleanで、commit／pushは行っていない。

再現用command、model/source hash、raw response、telemetry、検証scriptは `.local-artifacts/llama-mtp-20260914/`。
生成した8 GGUF（計22.84 GB）は内容とinodeを保ったままcheckout外の
`/home/homelab1/.cache/sllm-benchmarks/llama-mtp-20260914/models/` へ移し、従来のlogical pathはsymlinkで維持した。
原本はdatapool上で不変。Gitには集約値・hash・文書だけを保存し、model／binary／raw traceは保存しない。

比較図: 専用8行列（Git管理外: `.local-artifacts/llama-mtp-20260914/plots/phase85-fixed-heatmap.png`） / ファイル全体（Git管理外: `.local-artifacts/llama-mtp-20260914/plots/phase85-full-sidecar-heatmap.png`）。
詳細表: [CSV](llama-mtp-quantization-results.csv) / [集約JSON](../../../../../ci/matrix/llama-mtp-quantization-results-v1.json)。

[計画](../../../../plans/archive/2026/09/11-20/llama-mtp-quantization-benchmark.md) / [メイン計画](../../../../plans/main-plan.md)
