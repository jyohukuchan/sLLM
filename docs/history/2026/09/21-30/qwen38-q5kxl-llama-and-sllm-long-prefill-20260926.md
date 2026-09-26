# Qwen3.8-27B llama.cpp Q5_K_XLとsLLM長文prefill（2026-09-26）

ユーザー指定の独立性能測定。Phase 87全体の確認待ち状態は変更しない。

## llama.cpp

計測開始時の公式HEAD `fcc891545b0f06de346d8f67d1e6c61f9bf0e777`を独立worktreeに固定し、Vulkan版と`gfx1030`／`gfx1201`向けHIP版の`llama-bench`をビルドした。モデルはダウンロード済みの`Qwen3.8-27B-UD-Q5_K_XL.gguf`（20,218,178,624 B）。GGUF名は`Qwen3.8-27B`、architectureは`qwen35`。`Q5_K_XL`はファイル名上の量子化レシピで、実際のtensor型はF32、Q4_K、Q5_K、Q6_K、Q8_0の混在。`llama-bench`の`model_type`欄は`qwen35 27B Q4_K - Small`と表示されたため、識別にはファイルパスとサイズを使う。

長文測定中に公式HEADは`4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0`へ1 commit進んだ。差分は`common/common.cpp`と`tools/rpc/rpc-server.cpp`のディレクトリ作成utility整理のみで、`llama-bench`、モデル演算、HIP／Vulkan backendには変更がない。表の実測版は引き続き固定`fcc8915`とする。

単一GPU、`-ngl 999 -ncmoe 0 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512`。prefillは`-p 8192 -n 0 -d 0`、decodeは`-p 0 -n 128 -d 8192`。各条件はwarmup後に2回測定し、`avg_ts`を採用した。モデルロード、tokenize、samplingは`llama-bench`の測定区間外。

| GPU | Backend | Prefill 8192 (token/s) | Decode 128、depth 8192 (token/s) |
| --- | --- | ---: | ---: |
| R9700 | HIP／ROCm | 962.048 | 25.354 |
| R9700 | Vulkan／RADV | 825.796 | 26.291 |
| V620 | HIP／ROCm | 366.850 | 19.934 |
| V620 | Vulkan／RADV | 212.968 | 20.883 |

全行で対象デバイスを1台だけ公開し、JSONに単一の`ROCm0`または`Vulkan0`と`n_gpu_layers=999`を記録した。モデル常駐中のVRAM使用量は約20 GBで、GPU実行を確認した。生成結果の数値oracleはこの性能測定に含まない。

## sLLM

Qwen3.8-27B-NVFP4 safetensorsを使い、専用のopt-in長文診断で測定した。入力は既存の8192-token coding fixtureを先頭に置き、決定的な有効token IDで指定長へ延長。単一request、MTPなし、paged MXFP8 E4 KV、greedy GPU argmax、出力予算1 token、decode遷移0。warmup 0回＋測定1回。時間は`request.prefill`を囲むhost intervalで、モデルロードとrequest生成は除外した。両GPUの成功行はHIP dispatchのみ、fallbackなし、cleanup zero。

| GPU | 入力token | Chunk | Prefill (s) | Prefill (token/s) | 結果 |
| --- | ---: | ---: | ---: | ---: | --- |
| R9700 | 32768 | 2048 | 146.876 | 223.099 | 成功 |
| V620 | 32768 | 2048 | 262.362 | 124.896 | 成功 |
| R9700 | 65536 | 2048 | 566.469 | 115.692 | 成功 |
| V620 | 65536 | 2048 | 913.294 | 71.758 | 成功 |
| R9700 | 131072 | 512 | 1692.954 | 77.422 | 成功 |
| V620 | 131072 | 512 | 3507.306 | 37.371 | 成功 |

R9700の32768入力はchunk 8192だとprojection-pack workspaceの`hipMalloc`がOOMで失敗した。失敗時のrequest後cleanupはzero。chunk 2048へ下げて再試行した成功値を表に載せた。131072入力は残VRAMからchunk 512を選んだ。prefill速度はchunkが異なる行を直接の長さスケーリングとして読まない。

V620のVRAM総量は34,342,961,152 B＝34.34 GB＝31.98 GiB、R9700は34,208,743,424 B＝34.21 GB＝31.86 GiB。`rocm-smi`の使用バイトを10進GBに換算して表示した。例えばV620の33.2 GB使用は総量内であり、32 GiBを超えた意味ではない。

llama.cppのGGUF Q5_K_XLとsLLMのNVFP4は異なる重み形式で、KV形式と入力長も異なる。表は各エンジンの実行速度であり、同一条件の優劣比較ではない。sLLMの長文測定は単発で、長文出力の独立数値oracleは含まない。

## 追補：R9700、8192入力のchunk幅比較

前回の長文prefill専用診断を8192入力にも対応させ、同じ8192-token coding fixtureを使った。モデル・exact `gfx1201` target・MXFP8 E4 KV・MTP無効・出力1 token・state capacity 8193を固定。各幅を別processでwarmup 1回＋測定3回実行し、prefillの中央値とMADを比較した。実行順は8192、2048、4096、1024。モデルロードとrequest生成はprefill計測区間外。

| Chunk | Prefill中央値 | MAD | Prefill速度 | 結果 |
| ---: | ---: | ---: | ---: | --- |
| 1024 | 17.148秒 | 0.006秒 | 477.732 token/s | 成功 |
| 2048 | 16.616秒 | 0.005秒 | 493.023 token/s | 成功 |
| 4096 | — | — | — | `layer.57.mlp_gate_matmul`の作業領域`hipMalloc`でOOM |
| 8192 | — | — | — | `layer.16.linear.qkv_matmul`のprojection-pack作業領域`hipMalloc`でOOM |

成功した2048は1024よりprefill時間が`3.10%`短い。両条件は入力fixture SHA-256と出力token SHA-256が一致し、全反復がHIPのみ・fallbackなし・cleanup zeroだった。失敗した両条件もpost-error cleanupはzero。1024と2048のtracked workspace high-waterはそれぞれ0.881／1.762 GBで、幅増加に伴う作業領域負担が見える。4096／8192の速度は取得できないため推定しない。この比較は各幅1 processの3反復に限定する。

## 追補：1か月前のllama.cppとのdecode比較

2026-08-26 23:59 JST以前の最終commit `bf942164697d2d62c2237a17b677dc2c017ea8e7`（commit時刻2026-08-26 23:49 JST）を旧版とした。比較対象は前回固定した`fcc891545b0f06de346d8f67d1e6c61f9bf0e777`であり、移動するHEADを追わない。両版を同じhost compiler、ROCm 7.14、Vulkan環境、Release設定、HIP exact targetでビルドし、sourceの変更・patch・モデル変換は加えていない。

同じQ5_K_XL GGUF、文脈8192、decode 128、F16 KV、Flash Attention on、batch/ubatch 2048/512、単一GPU全層offloadで比較した。旧版も`nextn_predict_layers=1`を認識し、通常実行はMTPを除く64層。`llama-bench`のdepth事前充填は両版とも計測外で、tokenizer／samplingも計測外。各backendの旧版直後に固定現行版を測り、同じGPU内で実行を重ねない。独立したV620とR9700の行は並列に実行した。

| GPU | Backend | 旧版 token/s | 現行固定版 token/s | 速度変化 |
| --- | --- | ---: | ---: | ---: |
| R9700 | HIP | 24.853 | 25.366 | +2.06% |
| R9700 | Vulkan | 26.463 | 26.382 | −0.31% |
| V620 | HIP | 19.900 | 20.048 | +0.74% |
| V620 | Vulkan | 21.126 | 20.948 | −0.84% |

基本はwarmup後2回測定の`avg_ts`。V620 HIPは初回の旧19.148±0.834／現行19.327±1.046 token/sで変動が大きかったため、この条件だけ現行→旧の逆順でwarmup後3回測定した表の値を使った（標準偏差は旧0.107／現行0.102 token/s）。両processの全5 sampleを集約しても旧19.599／現行19.760 token/s、+0.82%であり、大きな改善を示さない。追加のGPU測定は行わない。

4条件ともdecodeの大幅な高速化はなく、最大はR9700 HIPの約2.1%。他は1%以内でほぼ同等。前回得た高いdecode速度は1か月前にも出ていた。これは同じGGUF／GPU／software環境での小規模測定であり、model品質の一致や別artifactの性能を主張しない。旧版と現行版にはQKV、GDNや各backendの更新があるが、この測定だけで個別変更の寄与は特定しない。

旧Vulkan／gfx1030 HIP／gfx1201 HIP binaryのSHA-256は順に`31f6d57051cb7504746b09e2ac595312c1ce88bead2535c5ec2cbeec2886dad6`、`4a45519cf0881c3205f973578916d05732f9acc678e5b704186481e1dac4a5e5`、`abc639d0c977355bbf2cac06b7770cc33c2a3c4096e207130518334e282006c2`。現行binaryは最初の測定からhash不変。raw出力はignored結果ディレクトリの`month-*` JSON／log、実行引数・終了状態・UUID・binary hashは`month-manifest-*.json`、集約値は`month-decode-comparison-summary.json`に保存した。8行と追加2行は正常終了し、入力長／出力長／KV／device／offload／sample数を照合した。

旧版worktreeの取込みは[`THIRD_PARTY_NOTICES.md`](../../../../../THIRD_PARTY_NOTICES.md#llama-cpp-month-comparison-20260926)に記録した。

## 追補：R9700 HIPのdecode graph kernel数

「llama.cppのdecodeがsLLMより速いのはkernel起動数が少ないためか」を確認した。既存のStage6読み取り専用HIP Graph instantiate interposerを使い、固定`fcc8915`の`llama-bench`と現行sLLM binaryでgraph nodeを列挙した。これは起動時の診断であり、interposer付きの時間を速度比較には使わない。

| R9700 `gfx1201`、MTPなし | 1 token decode graphのkernel node | その他のnode |
| --- | ---: | --- |
| llama.cpp、Q5_K_XL／F16 KV | 1828 | なし |
| sLLM、NVFP4／MXFP8 E4 KV | 986 | memcpy 1、memset 33（全1020 node） |

llama.cppは`-p 0 -n 4 -d 8192 -r 1 --no-warmup`、sLLMは8192入力／4出力、chunk 2048、固定GPU sampling、MTPなし、warmup 0／測定1で実行した。両方とも単一R9700。llama-benchは1 tokenずつdecodeする。取得graphのGDN本体48個とfull-attention本体16個も通常64層の1 token構成と一致する。sLLMは3 decode遷移のうちwhole graph再生2回をauditに記録し、`native_kernel_nodes_per_replay=986`がinterposerの列挙数と一致した。HIPのみ、fallbackなし、cleanup zeroを確認した。sLLMにはsampling・出力制御も含む一方、llama-benchのsamplingは計測外なので、機能範囲が完全に同じとは主張しない。

llama.cpp側は別kernelの`quantize_q8_1`が433個、量子化matvecが計433個、GDN本体48個、Flash Attention tile／combineが各16個。sLLM側は残る別FP8 quantizerが97個であり、producer融合による削減が既に入っている。少なくともこのHIP経路ではllama.cppの総kernel数は少なくなく、sLLMの方が約46%少ない。両者ともGraph再生を使うため、hostからのGraph launch削減もllama.cpp固有の優位とはしない。

llama.cppにはQKV、gate/up＋GLU、norm、GDN等の融合があるが、総kernel数だけで速度差は説明できない。量子化matvec／Attention等のkernel内処理と読み出し効率、数値形式の違いが候補であり、時間内訳の比較なしに寄与を断定しない。Vulkanのdispatch数はこの診断では未測定。

raw DAG、実行manifest、kernel名一覧、sLLM auditはignoredの`.local-artifacts/benchmark-20260926/launch-count/`に保存した。使用interposerは既存`.local-artifacts/phase87/stage6/dag/libhip_graph_dag_dump.so`。外部sourceの新規コピー・変更やproductionコード変更は行っていない。

## 追補：R9700 HIPの定常decode時間内訳

同じR9700で両engineを順に`rocprofv3` 1.3.2（git `2b22ab0195cc1461cd9abf3b969e9dd7c10af350`、ROCm 7.14）へ通した。kernel／HIP runtime／ROCTX／memory-copy traceをCSV取得し、両方で`ROCPROFILER_QUEUE_INTERPOSITION=0`を設定した。DAG取得interposerは付けていない。固定llama.cppとsLLMのbinary hashは前節と不変。

llama.cppは文脈8192から16 token、sLLMは8192入力から17出力（16 decode遷移）、MTPなし。llama.cppはQ5_K_XL＋F16 KV、sLLMはNVFP4＋MXFP8 E4 KV＋固定GPU samplingであり、数値形式とsampling範囲は異なる。両方に15回の`hipGraphLaunch`があり、capture直後を除いた最後の8 correlation IDに結び付くdispatchだけを選んだ。llama.cppは毎回1828 dispatch、sLLMは毎回1020 dispatchで、GDN／Attention／行列積等のfamily数も全8回で一致した。前節のsLLM明示kernel node 986に対し、profilerでは33 memsetと1 memcpyがROCm内部のfill／copy kernelとして現れるため34 dispatch増える。起動数に関する結論は変わらない。

以下は1 tokenあたり8回平均、単位ms。categoryは各kernelを一度だけ割り当てる。融合されたGLU、norm、量子化等は所属kernelのcategoryへ含み、別計上しない。両engineとも選択範囲でkernel実行の重なりは0だったため、category時間の和はGPU busy時間と一致する。

| 区分 | llama.cpp | sLLM | sLLM−llama.cpp |
| --- | ---: | ---: | ---: |
| 投影／MLP／headの行列積 | 32.006 | 36.048 | +4.042 |
| Full Attention＋KV append | 1.057 | 4.179 | +3.122 |
| GDN状態更新＋conv | 0.425 | 2.061 | +1.636 |
| Norm／elementwise／RoPE（producer融合分を含む） | 1.164 | 1.774 | +0.610 |
| 別kernelのactivation量子化 | 0.537 | 0.648 | +0.110 |
| copy／gather／packing／fill | 0.596 | 0.041 | −0.555 |
| sampling／decode制御 | 0 | 0.145 | +0.145 |
| **kernel本体合計** | **35.784** | **44.896** | **+9.111** |
| graph内のkernel外区間 | 8.279 | 4.702 | −3.576 |
| **GPU graph span** | **44.063** | **49.598** | **+5.535** |

Graph間の空白はllama.cpp 0.632／sLLM 0.028 ms、定常token開始間隔は44.687／49.643 msだった（こちらは選択8回の間にある7区間の平均）。Graph launchのhost API時間は0.988／0.724 msで、GPU処理と重なるため上の表へ加算しない。graph内のkernel外区間にはdispatch待ち等が含まれ、純粋なlaunch固定費とは断定しない。

主なkernel本体:

- llama.cppの量子化matvec全433起動は計32.006 ms。融合ありQ5_K 64起動が12.794 ms、融合ありQ6_K 80起動が8.235 ms、融合なしQ6_K 81起動が5.487 ms、融合なしQ5_K 144起動が5.079 ms。
- sLLMのFP8 dot4 233起動が18.490 ms、NVFP4 decode 168起動が17.194 ms、小さいBF16投影96起動が0.363 ms。
- Attentionの主kernelはllama.cpp Flash Attention tile 16起動が0.953 ms、sLLMの128-way paged split stage1 16起動が3.929 ms。
- GDN本体はllama.cpp 48起動が0.346 ms、sLLM 48起動が1.939 ms。ただしsLLM側はgated normも同じkernel内に含むため、純粋な同一演算kernel比較としては扱わない。
- 別量子化はllama.cpp 433起動でも計0.537 ms、sLLM 97起動で計0.648 ms。起動数だけから処理時間を予測できない。

このprofileでは、sLLMは少ない起動数でgraph内外の空白を約4.2 ms短縮する一方、kernel本体は約9.1 ms長い。投影、Attention、GDN＋convの3 familyで約8.8 msの差を占める。したがって、今回の速度差の主な候補は重いkernel本体の効率と形式／演算契約であり、総kernel数の削減不足とは解釈しない。形式差を分離した単体比較や最適化採用の証拠ではない。

profilerの観測負担があるため、この時間を通常測定のTPOTへ直接代入したり、通常速度差の割合へ換算したりしない。例えばllama.cppの通常decode約39.4 ms/tokenに対して、選択範囲のprofile token間隔は約44.7 msだった。単一profile内の8回のgraph span標準偏差はllama.cpp 0.041／sLLM 0.067 msであり、process間の変動を示すものではない。

raw CSV／report／実行manifest、SHA-256付き集約結果はignoredの`.local-artifacts/benchmark-20260926/decode-breakdown/`。解析scriptは同じignored task directoryの`analyze-decode-breakdown.py`。全kernelを分類でき、両process正常終了、sLLMのHIP-only／fallbackなし／15 completed replay／cleanup zeroを照合した。sourceやproduction kernelの変更、新しい外部source importは行っていない。

## 証跡と検証

3つの差が大きいfamilyを直接コード比較し、GDNのfusion境界を揃えた追補は[コード比較記録](qwen38-llama-sllm-decode-code-comparison-20260926.md)を参照する。上のfamily表は取得時のkernel名分類を保持し、GDN関連準備処理の再分類による合計変更は追補に記録した。

- raw JSONとlog: ignoredの`.local-artifacts/benchmark-20260926/results/`。モデル・build・生traceはGitへ追加しない。
- llama.cpp HEAD: `fcc891545b0f06de346d8f67d1e6c61f9bf0e777`。Vulkan／gfx1030 HIP／gfx1201 HIPの`llama-bench` SHA-256は順に`7d0b0db4feb225a54e800ce583f2bc525cf5a9ba7f465cd620b272b37499d1b7`、`dd552efe8892421f69ef0415a13f66e661c8946e8a63d7d5a62d73c13011db49`、`757f12636f64a98e6371a3054ed1ffec6e41736c381e7cc1d04aeb144bdce51b`。
- sLLM作業treeの基底commit: `f3a55c581f52bb28f17a99410a1b2deb7306181c`。長文診断は`crates/sllm-hip/src/bin/sllm-phase78-qwen38-benchmark.rs`に追加。gfx1030／gfx1201実行binaryのSHA-256は順に`11fe10a202660b93a1d21e3f0296e6deaf7d335309fc0a17fece0f79b8ece69a`、`0fd0ce0da434dbccee18462d8037ba25c5e68a428f8de80d35ad0d6f3899d230`。
- `cargo check -p sllm-hip --bin sllm-phase78-qwen38-benchmark`成功、同binaryのhost unit test 20件成功。両exact HIP target向けrelease build成功。性能行のGPU実行auditは個別JSONに記録した。
- 8192入力への診断拡張後はfocused parser test成功、`gfx1201`向けrelease build成功。比較用binary SHA-256は`1cdc2452cbade7cc2d2f7344cb5739eb6d8eec54cc7e4cc9625baa5f3d5a6575`。比較のraw JSONと失敗logも上記ignored結果ディレクトリに置いた。
- 外部sourceの独立worktreeは[`THIRD_PARTY_NOTICES.md`](../../../../../THIRD_PARTY_NOTICES.md#llama-cpp-qwen38-q5kxl-benchmark-20260926)に記録した。
