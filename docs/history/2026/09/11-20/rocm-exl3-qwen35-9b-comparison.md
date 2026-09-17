# Qwen3.5-9B: EXL3とllama.cpp Q4_K_Mの比較

2026-09-17、5B以上のモデルで比較するユーザー依頼により実施。
前回の2Bと同じ系列のQwen3.5-9Bを使い、両artifactを同じBF16 sourceから生成する。

## 条件

- Source: `Qwen/Qwen3.5-9B`、revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
  [model lock](../../../../models/locks/qwen3.5-9b-bf16.json) の15ファイルをSHA-256照合し一致。
  BF16 safetensors合計19,306,310,880 bytes。text hidden4096、32層、GDN key16/value32 head、head dim128。
- R9700 exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`。
- rocm_exl3: `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`＋[既検証patch](../../../../references/patches/rocm-exl3-gfx1201.patch)。
  前回のsource/binary hashと一致を再確認。PyTorch2.12 ROCm7.2 wheel、host ROCr1.21.0明示preload。
- llama.cpp: `bc52a12b38941b0a690ade65fbc5749715224e30`、既存gfx1201 HIP Release buildを再使用。
- EXL3: decoder4bpw/head6/vision16、calibration16×256。前回と同じ動作比較用の縮小校正で、標準校正品質ではない。
- llama.cpp: Q4_K aliasの標準Q4_K_M、imatrixなし、CPU量子化32threads。
- 単一要求、MTPなし、KV FP16、context4096、pp128/512/2048、decode入力128・出力64指定。
  各条件warmup1＋measured3、中央値とspreadを記録する。
- 元のEXL3 Tokenizer/乱数生成関数でtoken ID列を固定し、両engineへ同じ列を渡す。
  native timerのphase境界、EXL3の64指定→63出力、llamaの64出力→63 timed stepsは
  [2B比較](rocm-exl3-investigation.md)と同じ定義差として残す。

## 結果

同一token ID列、1 warmup＋3 measuredの中央値。主表はEXL3 `max_chunk_size=2048` と
llama.cpp `batch=2048,ubatch=2048` を揃えた。速度はnative engine timingで、HTTP転送時間を含まない。

| 測定 | EXL3 4bpw tok/s | llama.cpp Q4_K_M tok/s | 観測した差 |
| --- | ---: | ---: | --- |
| prefill 128 | 398.3 | 1609.2 | llama 4.04倍 |
| prefill 512 | 2938.5 | 2986.5 | llama +1.6%、ほぼ同程度 |
| prefill 2048 | 4917.6 | 3441.2 | EXL3 +42.9% |
| decode・128入力/64指定 | 75.10 | 78.48 | llama +4.5% |

spread（max−min／中央値）はEXL3が順に1.05%/1.12%/9.39%/0.50%、llamaが2.82%/1.65%/0.88%/0.24%。
EXL3 pp2048の3値は4917.577/4939.145/4477.422 tok/sで、低い1値も除外せず保持した。
3サンプルの短い観測であり、広い条件での性能順位や統計的有意性を主張しない。

前回2BではEXL3のdecodeが約10.5%速かったが、今回9Bのこの構成ではllamaが約4.5%速かった。
一方9Bの長いprefillはEXL3が速い。サイズだけでなく各engineのshape別経路も変わり、
どちらかが常に速いという結果ではない。

### Chunk条件の追加確認

最初は前回と同じllama ubatch512で測定し、pp128/512/2048/decodeは
1629.6/2967.6/3351.5/80.87 tok/sだった。pp2048ではllamaだけ4分割、EXL3は一括だったため、
ubatch2048で16 request全体を追加測定し、主表にはその一式を採用した。
行ごとの最良値選択や異なる設定の混合は行っていない。ubatch512の16 requestも集約JSONに保持する。
両条件のserverは同じmodel/source/binary、全34/34層GPU、FA有効、KV f16、context4096、parallel1、
threads8/batch threads8、fit off、cache RAM0、各request cache_prompt=falseである。
EXL3は当該fork既定のeager、llamaは既定HIP Graph有効。これもengine実装差に含む。

### 生成・終了と数値検査

EXL3は空が青い理由をRayleigh scatteringと青/紫への感度差で説明する2文を生成し、
59行の観測logitsはすべて有限値だった。日本語の計算質問には `5` と答え、観測1行のlogitsも有限。
両probe、EXL3 benchmark、llamaの2 benchmark、各serverのSIGTERM終了はいずれもexit0。
llamaも英語の説明文と日本語 `5` を生成してEOS停止し、最終ubatch2048でも日本語を再確認した。
これは動作smokeであり、perplexityや同等品質の認定ではない。
9B固有のGDN形状（key16/value32、128×128）を既存の独立torch参照と比較し、
長さ1/15/17・履歴なし/ありの6ケースで一致、process exit0を確認した。

## Artifactと量子化コスト

| 項目 | EXL3 | llama.cpp Q4_K_M |
| --- | ---: | ---: |
| 重み／GGUFファイル合計 | 7,307,469,398 bytes | 5,780,090,688 bytes |
| metadata等を含むEXL3出力directory | 7,330,786,076 bytes | — |
| 量子化時間 | 約34分39秒 | 70.766秒 |

EXL3時間はログ作成15:19:14.536〜最終書込15:53:53.529 JSTから求めた概算で、準備・hash検証を含まない。
通常32層は各約53〜55秒、6bit output headは251.82秒だった。Q4_K_Mはquantizer自身の報告値で、
前段のBF16 GGUF変換時間は含まない。EXL3の校正・Hessian・trellis searchと、imatrixなしQ4_K_Mでは
処理内容自体が異なる。

Q4_K_Mは442 tensor中Q4_K 223、Q6_K 35、F32 184、payload5,769,121,792 bytes、表示5.02 BPW。
outputはQ6_K（834,355,200 bytes）、embeddingはQ4_K（572,129,280 bytes）。
EXL3はdecoder4bpw/head6に加えBF16 embedding・未量子化vision等を保持するため、
保存サイズ差を量子化方式だけの圧縮率差と解釈しない。BF16 GGUFは18,407,321,408 bytes。

英語smokeのEXL3 peak Torch allocatedは5,529,328,640 bytes。
llamaの最終ubatch2048起動時native bufferはmodel4812.25 MiB、KV128.00 MiB、recurrent50.25 MiB、compute368.06 MiB。
これらは異なるmemory counterで、driverを含む同一基準のpeak VRAMとして順位付けしない。

## 測定上の限界

- EXL3のprefillは最後の入力tokenをgenerationへ回し、N−1を処理してNを分子にする。llamaはN全体をprompt時間に含める。
- 出力64指定でEXL3は実出力63・63 forward、llamaは実出力64・初回を除く63 timed steps。位置が1 tokenずれる。
- EXL3 DefaultSamplerとllamaはmin_p0.08→temperature0.8で合わせたが、前者GPU sampler、後者CPU samplerである。
  llamaのignore_eosはEOGをmaskし、EXL3のstopconditionsなしとは完全一致しない。生成token列の一致は要求していない。
- EXL3校正16×256はworkflow smoke用で、標準250×2048の品質を保証しない。Q4_K_Mとのbit配分も異なる。
- 同一入力16本を用い、EXL3 16＋llama 2構成各16＝全48 requestでprompt/cache/output countとnative timingの正値を検証。
  throughput用のloop内でTensorのJSON変換やfile書込みを行わず、計測終了後へ分離した。

## 再現

[runner](../../../../../ci/tools/benchmark_rocm_exl3_large.py) は `generate-prompts` / `exl3` / `llama` の3 modeを持つ。
Torch等は実行modeで遅延importする。manifestは元の `rocm_tools.bench_model.rand_prompt` と実際のTokenizerから生成し、
語彙数248077を保存して実行時にも照合する。

container内の主要コマンド:

```sh
python /src/convert.py -i /model-source -o /work/models/qwen35-9b-exl3-4bpw \
  -w /work/exl3-quant-work -b 4 -hb 6 -vb 16 -cr 16 -cc 256
python /llama-src/convert_hf_to_gguf.py /model-source --outtype bf16 --outfile /work/models/qwen35-9b-bf16.gguf
/llama-build/bin/llama-quantize /work/models/qwen35-9b-bf16.gguf /work/models/qwen35-9b-Q4_K_M.gguf Q4_K 32
python /work/benchmark.py generate-prompts --model-dir /model-source --exl3-repo /src --prompts /work/prompts.json
python /work/benchmark.py exl3 --model-dir /work/models/qwen35-9b-exl3-4bpw \
  --exl3-repo /src --prompts /work/prompts.json --output-dir /work/bench-exl3-rerun
/llama-build/bin/llama-server -m /work/models/qwen35-9b-Q4_K_M.gguf \
  -ngl 99 -c 4096 -np 1 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 \
  --fit off --cache-ram 0 --host 127.0.0.1 --port 18089 -t 8 -tb 8 --no-webui -lv 4
# 別shellから
python /work/benchmark.py llama --prompts /work/prompts.json \
  --output-dir /work/bench-llama-rerun --url http://127.0.0.1:18089
```

前回と同じhost ROCr preloadが必要。`LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1`、
`PYTHONPATH=/exl3-runtime/lib-autotune:/src`、GPU UUID可視化をcontainerへ設定済み。
変換コマンドは今回実行した記録であり、再生成時は既存output/workを上書きしない別pathを使う。

## 保存先

`/home/homelab1/datapool/rocm-exl3-experiment/large-9b/` にmodel、log、raw result、hashを保存する。
専用containerは `rocm-exl3-9b-comparison`。source/model/buildはread-only mount、出力は`/work`。
完了時にcontainerを停止し、全GPUの使用率0と検証前のVRAM使用量への復帰を確認した。
ホストhomeやcredential、Docker socketは外部実行環境へ渡さない。
既存のsource checkoutを再使用し、追加の外部source copy、production実装変更、commit/pushは行わない。

[集約値・全sample・hash](../../../../../ci/matrix/rocm-exl3-qwen35-9b-v1.json) /
[計画](../../../../plans/archive/2026/09/11-20/rocm-exl3-qwen35-9b-comparison.md)
