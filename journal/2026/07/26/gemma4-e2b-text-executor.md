# Gemma4 E2B `Gemma4TextExecutor`

## 前回の要点

依頼BFのconfig駆動化により、`google/gemma-4-E2B` は実configから
`Gemma4Text` descriptorまで組み立てられるが、`Gemma4TextExecutor`未実装として停止していた。
`config.json` SHA-256は`e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8`、
単一の`model.safetensors`は10,246,621,918 bytesである。量子化経路を先に混ぜず、HF
Transformersの実装を唯一の仕様根拠として、source BF16/F32 activationで層ごとに確認する方針だった。

## 今回の変更点

- `crates/ullm-engine/src/gemma4_text_executor.rs` を追加した。source safetensorsを直接読んで既存のBF16×F32 matvecへstreamし、Gemma4のembedding scale、local/full RoPE、Q/K/V norm、shared K/V、4 residual norm、double-wide MLP、PLE、tied head、final soft-capをF32 activationで実装した。CPU staging fallbackは`ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1`で拒否し、R9700 (`AMD Radeon Graphics` / `gfx1201` / compute 12.0) だけを選択する。V620は選択しない。
- `crates/ullm-engine/src/bin/ullm-gemma4-text-trace.rs` を追加した。embedding、全35 layer output、final norm、soft-capped logitsを`ullm.architecture_trace.v1`のNPZ/metadataに出力する。traceはdiagnostic-onlyで、campaign、FP32 corpus、bit一致gate、serving sessionを使わない。
- `model_config.rs` は executorが実configをfail-closedに検証するため、Gemma4のmax position、bidirectional attention、global KV fieldと`Gemma4TextNonquantized` statusを保持するようにした。Gemma4 sourceではtext-only causal、attention bias/dropoutなし、direct RMSNorm、`gelu_pytorch_tanh`、non-MoE、default/proportional RoPE、tied headだけを受理する。
- HF Transformers 5.12.1を読んだ。`modeling_gemma4.py`のattention L1180--1289、decoder layer L1369--1455、PLE L1612--1630 / L1737--1815、conditional head L2445--2535と`modeling_rope_utils.py`のproportional RoPE L187--254を根拠にした。attention scaleは1.0でattention soft-capは渡されず、final logitsだけが30でsoft-capされる。KV共有はlayer 15以降で、E2Bのshared local/full sourceはlayer 13/14である。
- 非量子化のHF CPU F32対R9700 BF16 source/F32 activation traceを`benchmarks/results/2026-07-26/gemma4-e2b-nonquantized-v0.1/`に保存した。token 2の1 stepは38 tensor / generated `184`、2 decode stepsは76 tensor / `184,3910`、`The capital of France is`と`Once upon a time,`の各4 stepsは152 tensorずつで、各stepのgreedy token列はHFと同一だった。最大abs差はstory case final normの`1.1825562e-4`、最大relative L2は`2.6786e-6`だった。これは数値gateではなく最初の構造的乖離を探す局在化記録であり、見つからなかった。
- 実生成はHF/uLLMともに`The capital of France is Paris.\n\nThe`、および`Once upon a time, in a world where`となった。capital promptを8 tokenまで伸ばすと両者とも入力を反復したため、base checkpointのgreedy特性として記録した。candidateだけが壊れる現象ではない。
- Phase 4をread-onlyで確認した。既存SQ8_0 runtimeはQwen3-14Bのfixed width/head/intermediate/norm/RoPE/layer arrayを前提にしており、Gemma4のmixed-width local/full attention、PLE、shared K/V、tied head、soft-capを表せない。既存AQ4_0/SQ8_0 production codeとBH/BKの保護ファイルには触れていない。新カーネルも追加していない。

## 次の行動

1. Phase 4を進めるなら、既存SQ8_0を流用せず、Gemma4専用のquantized artifact descriptorとresident executorを設計する。nonquantized traceを先に再利用し、量子化誤差とarchitecture誤りを混ぜない。
2. diagnostic executorはsource matrixを毎projection streamするため、性能・serving pathとして扱わない。resident BF16またはquantized memory plan、prefill/decode cacheの設計は別タスクに分離する。
3. multimodal vision/audio、processor/chat template、MoE/bidirectional branchesはこのtext-only causal scope外のままとする。新しいsource configで分岐が変わった場合はfail-closed rejectionを保つ。
