# rocm_exl3 実装・R9700動作調査（2026-09-17）

## 依頼と対象

ユーザー指定の [CarouselAether/rocm_exl3](https://github.com/CarouselAether/rocm_exl3) を
`reference/rocm_exl3/` にcloneし、実装確認、GPU動作、モデル量子化を調査した。
途中でユーザーから軽微なバグ修正とGPUの自由利用が明示的に許可された。
既存のsLLM本体の未コミット変更を保持し、本調査からproductionへ外部コードを移植していない。

- 固定HEAD: `550dcfed786ad7bffa08b7a6b2a216fc474cbbb5`、upstream version 1.4.4。
- ライセンス: MIT、Copyright (c) 2025 Turboderp。clone内の原文を保持。
- 対象GPU: Radeon AI PRO R9700、exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`。
- V620 `gfx1030` はupstream対象外。WMMAを持たないため今回full-runtime移植は行わない。
- モデル: 既存 `Qwen/Qwen3.5-2B` BF16、revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。
  [model lock](../../../../models/locks/qwen3.5-2b-bf16.json) の全12ファイルをSHA-256照合し一致。
- 環境: host ROCm compiler 7.14.60850／LLVM 23、専用Docker内PyTorch `2.12.0+rocm7.2`、
  HIP wheel build `7.2.53211`、Triton 3.7.0。実行時のHIP共有libraryはtorch同梱版。
  host compilerとtorch wheelのversionを同一と表記しない。
- 最終実行では `LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1` を指定する。
  bundled HIP 7.2とhost ROCr 1.21.0の組合せを検証した。bundled ROCrもmapped library一覧には残るため、
  全runtimeが7.14へ統一されたとは表記しない。
- host home、credential、Docker socketを外部実行環境へmountせず、source read-only、対象model read-only、
  GPU deviceと専用work領域だけを渡した。依存導入はcontainer内のみ。
- 専用成果物: `/home/homelab1/datapool/rocm-exl3-experiment/`。モデル、binary、raw logはGit管理外。
- GPU解放のため、ユーザーの追加許可後に既存 `sllm-server` PID 3640782へSIGTERMを送り、終了を確認。
  Qwen subagent serviceは開始時から停止中で変更なし。

## 上流の不具合と局所修正

[再適用用patch](../../../../references/patches/rocm-exl3-gfx1201.patch) は固定HEADへの差分。
`reference/rocm_exl3/` の変更はcommitしていない。

1. **RDNA4 WMMA**: 上流はgfx1201をbuild対象に列挙するが、gfx11の256-bit WMMA intrinsicを呼び、
   `wmma-256b-insts`不足でcompile失敗。gfx12の128-bit intrinsicへ分岐し、A/B fragmentを変換。
   gfx11とgfx12でoperand順・C fragment配置が異なる。`c_row/c_col`でstore/loadを統一し、
   GEMM innerに残る直接的なrow/column計算も修正した。gfx11分岐の数式は維持。
2. **2bit encoderのLDS超過**: K=2で65536-byte cost＋704-byte scratchを要求し、
   R9700の`shared_memory_per_block=65536`を超え、`quantize_rdna.hip:90`でinvalid argument。
   K=1で既に使っているGPU global scratchをK=2にも使用し、launcherとkernelを一致させた。
   CPU fallbackではない。
3. **autotuneの固定LDS超過**: model load時のGEMM/MGEMM autotuneだけがcompile-time `SMEM_MAX=92160`を
   共有autotunerへ渡し、`coop_autotune.cu:303`でinvalid argumentとなった。`EXL3_ROCM_MGEMM=0`でも
   通常GEMMが同じ要求をするため回避できない。ROCm sibling内の4呼出しを既存の
   `exl3_rdna_smem_budget(device)`へ変更し、common CUDA sourceは編集しなかった。
   最終linkではこのtranslation unitだけ同一記録flagsで再compileし、残る108 objectを再使用した。

失敗した初期WMMA adapterはcompileだけ通り数値検査に失敗し、採用しなかった。
wrapper修正後もGEMM直接consumerの配置が古いままで8/8失敗したため、その箇所を修正した。
最終のPASSと、これらの失敗候補・build logを分けて保存する。

## 検証結果

| 対象 | 結果 | 範囲 |
| --- | --- | --- |
| gfx1201 extension | 109 source compile/link成功 | 固定HEAD＋局所patch（最終5ファイル） |
| WMMA | 16/16成功 | CPU参照、FP16/BF16/INT8、境界、蓄積、signedness |
| EXL3 GEMM | 8/8成功 | bits2/4/8、3 codebooks、split-K、N128/256/384 |
| EXL3 GEMV | 54/54成功 | bits1..8、direct/LDS、wave/tail/split-K、FP16/FP32出力 |
| paged attention | 18/18成功 | FP32参照、MHA/GQA、decode/prefill、head64/128/256 |
| GDN | 6/6成功 | Qwen2B形状、長さ1/15/17、履歴あり/なし |
| tile encoder | 12/12成功 | bits2/4/6/8 × tiles1/17/65、finite、decode一致、trellis closure |
| Qwen3.5-2B非量子化 | 生成・終了成功 | 37入力、観測63 logit rowsが有限値 |
| full model EXL3 | 変換・生成・通常終了成功 | 4bpw decoder、6bit head、vision16、calibration16×256 |

GEMM/GEMVのCPU参照は演算を独立に計算するが、重み復元にはupstreamのcodecを共有する。
encoder検査もcodebookの独立oracleではなく構造roundtripである。最終model実行には最終buildを使用し、
ROCr preload構成でGDNを含む全114ケースを再実行して成功した。
各scopeをGPU全体や未実行の形式へ一般化しない。

最終のsource/patch/binary identity、log hash、集約結果は
[機械可読記録](../../../../../ci/matrix/rocm-exl3-investigation-v1.json)へ保持する。



## モデル生成・サイズ・性能

入力weightは4,548,221,488 bytes、変換後weightは2,783,714,651 bytes、metadata等を含むoutput directoryは
2,806,937,754 bytes。embeddingとvisionを量子化せず、head6bit、decoder4bpw、MTP4bitを含むため、
ファイル全体が一律4bitになるわけではない。校正は動作確認用の16×256 tokenで、標準250×2048とは異なる。

最終の英語promptは37 tokenで、非量子化/EXL3とも観測63 logit rowsが有限値、Rayleigh scatteringを説明する
自然な文章を生成した。日本語41-token prompt「2 + 3 の計算結果を、数字だけで答えてください。」は両者が `5` と回答し、
EOS停止・通常終了exit0を確認。英語・日本語の最初の出力位置だけを比べたKLD（非量子化→EXL3）は
`0.01345191117364558` / `0.012240581609087385`、top1は2/2一致。
これは小さいsmokeでありperplexityやタスク品質の認定ではない。

同じ最終binary/runtime、単独GPU、上流 `bench_model.py -p 128 512 -n 64 -r 3` で測定。
各行は初回を捨てた2回の中央値。decodeは128-token promptから64-token指定、prefix cache hitなし。
llama.cpp列は後続のユーザー依頼で追加した別engineの測定で、以下の条件・時間定義差を伴う。

| 計測 | rocm_exl3 非量子化 tok/s（spread） | rocm_exl3 EXL3 tok/s（spread） | llama.cpp Q4_K_M tok/s（spread） | EXL3/非量子化 |
| --- | --- | --- | --- | --- |
| prefill 128 | 4693.5（3.2%） | 1515.4（0.9%） | 4057.8（3.40%） | 0.323x |
| prefill 512 | 7453.3（2.8%） | 6647.3（0.7%） | 9553.3（0.37%） | 0.892x |
| decode・64 token指定 | 76.8（0.5%） | 181.9（0.2%） | 164.5（0.72%） | 2.368x |

この条件ではdecodeが速い一方、prefillは遅い。EXL3はrows<=144のquant GEMMと長いprefillの
reconstruct+FP16 GEMMを切り替えるため、単一の「EXL3は何倍」という数値へまとめない。
自作finite-logit probeは全logitsを観測し別のoverheadを加えるので、速度は上記専用benchmarkから取る。
英語probeのpeak torch allocatedは非量子化5,147,271,168 bytes、EXL3 1,481,394,176 bytes。
これはTorch allocatorの値でありdriverを含む全VRAM peakではない。

## llama.cpp Q4_K追加比較（2026-09-17）

同じ入力model revisionの全12ファイルを再hash照合し、保存済みclean llama.cpp
`bc52a12b38941b0a690ade65fbc5749715224e30`（2026-09-14取得）とgfx1201 HIP Release binaryを再使用した。
server/bench/quantizeのbinary hashは既存build identityと一致。追加cloneや外部source copyはなく、
既存の[import notice](../../../../../THIRD_PARTY_NOTICES.md#llama-cpp-mtp-quant-benchmark-20260914)の範囲で実行した。

`llama-quantize ... Q4_K 32` のQ4_Kは **Q4_K_Mへのalias**。imatrixなしの標準量子化で、CPU量子化時間は16.532秒。
BF16 GGUFは3,897,387,904 bytes、Q4_K_M GGUFは1,312,164,736 bytes（tensor payload 1,301,202,176 bytes）。
335 tensorの内訳はQ4_K 168、Q6_K 27、F32 140、quantizer表示は5.36 BPW。
token embedding（出力headと共有）、一部qkv/FFN down等はQ6_Kなので、全行列が4bitという意味ではない。
llama側GGUFはvisionを含まず、EXL3側はBF16 embeddingと別の6bit head等を保存するため、
保存サイズの差を量子化方式だけの圧縮率差へ読み替えない。

測定は`llama-bench`ではなく **llama-server `/completion`** のnative timingsを使った。
前回EXL3の生成器測定にsampling/生成制御を含める点を近づけるためであり、HTTP転送時間は主指標に含めない。
条件はR9700単独、context4096、parallel1、全26/26層GPU offload、FA on、KV f16/f16、
batch2048/ubatch512、CPU threads8、MTP off、fit off、cache RAM0、各request cache_prompt=false。
HIP Graphはllama既定有効、EXL3側は当該fork既定のeagerであり、これも実装差に含む。

EXL3の`rand_prompt`と同じtorch CPU RNG seed1234・語彙範囲で9本のtoken ID列を再生成し、
元の`rand_prompt`関数でも全9本が完全一致することを確認した。llamaには整数token配列で渡し、
BOSの自動挿入や文字列の再tokenizeを避けた。samplingはmin_p0.08→temperature0.8、
top_k0、top_p1、repeat_penalty1。llamaはseed1234、CPU sampler、ignore_eos=trueで長さを固定した。
EXL3の元benchはEOS maskなし・stopconditionsなしであり、出力token列の一致や同等品質は主張しない。

**時間境界の違い**: EXL3のprefillは最後の入力tokenをgeneration側に回すため、実処理127/511 tokenに対して
128/512を速度の分子としている。llamaは128/512全体をprompt時間へ含める。
出力64指定では、EXL3は当該Job実装の`max_new_tokens - 1`により実出力63、generation時間は最初のforward前から。
llamaは実出力64で、初回はprompt logitsから選び、native decode速度は残り63 steps／predicted_msである。
両decode値は63 forward相当だが、最初の位置が1 token異なる。厳密な同一演算量のkernel比較ではなく、
近い条件のengine比較として読む。

全9requestでcache_n0、入力128/512、指定出力数、raw token数、limit停止、timingsの分子/時間を確認した。
最初の測定はtokenizer.jsonから語彙範囲248070を仮定したが、実行時のEXL3 Tokenizerは248077だった。
語彙範囲の差により乱数token列が変わるため、その4010.4/9556.5/164.0 tok/sとpp128追加確認4064.2は主表から除外した。
元のEXL3 `rand_prompt`関数から入力を生成し、全9配列を別呼出しで再照合して完全一致を確認した。
最初の一致済みrunはpp128spread10.08%で、照合用runtimeのimportも重なっていたため、
照合用containerを停止したうえで9request全体を1回測り直し、その一式を最終表とした。
各runは保存し、行ごとの最良値の選択や平均への混合は行わない。
最終の2 measured値はpp128が3988.782/4126.769、pp512が9570.809/9535.694、decodeが163.956/165.144 tok/s。
今回の主表ではQ4_K_MはEXL3比でprefill128が2.678倍、prefill512が1.437倍、decodeが0.905倍。
逆にEXL3のdecodeはQ4_K_M比1.105倍。2 measured samplesの短い観測であり一般的な性能順位ではない。

Q4_K MUL_MATをCPU oracleへ比較する6ケース（K256、M16、N1/3/7/8/9、N1は2ケース）をgfx1201でPASS。
日本語の計算smokeは回答`5`・EOS停止・正常終了。文字列smokeはllama側27 tokenであり、
前回EXL3側41 tokenとtokenizeが異なるため、ここを速度比較には使っていない。
serverはSIGTERMによりexit0。詳細logでR9700 BDF `0000:07:00.0`、26/26 offload、FA有効を確認した。
起動時model buffer1204.91 MiB、KV48.00 MiB、recurrent state19.27 MiB、compute48.16 MiB。
これらはTorchのpeak allocatorとは別指標であり、同じpeak VRAM欄には混ぜない。

再現物は `/home/homelab1/datapool/rocm-exl3-experiment/llama-q4k/`、専用containerは
`llama-q4k-exl3-comparison`。[測定runner](../../../../../ci/tools/benchmark_llama_q4k_exl3.py)と
[集約JSON](../../../../../ci/matrix/rocm-exl3-investigation-v1.json)に条件・全sample・hashを保存した。
container内の再現コマンドは次のとおり。source/model/buildはread-only mountで、生成物は`/work`へ置く。

```sh
python /llama-src/convert_hf_to_gguf.py /model-source --outtype bf16 --outfile /work/models/qwen35-2b-bf16.gguf
/llama-build/bin/llama-quantize /work/models/qwen35-2b-bf16.gguf /work/models/qwen35-2b-Q4_K_M.gguf Q4_K 32
/llama-build/bin/llama-server -m /work/models/qwen35-2b-Q4_K_M.gguf \
  -ngl 99 -c 4096 -np 1 -b 2048 -ub 512 -fa on -ctk f16 -ctv f16 \
  --fit off --cache-ram 0 --host 127.0.0.1 --port 18089 -t 8 -tb 8 --no-webui
# 別のcontainer内shellから、既存modelを使って再測定する例
python /work/benchmark.py --prompts /work/prompts-matched.json --output-dir /work/server-benchmark-rerun
```

[追加比較計画](../../../../plans/archive/2026/09/11-20/rocm-exl3-llama-q4k-comparison.md)

## このホストでの再実行

専用container `rocm-exl3-investigation` とcheckout外の `/work` bind mountを使用する。
最終extensionは `/work/lib-autotune/`。`/work/lib/`、`/work/lib-final/`、`/work/lib-validated/` は途中候補なので使わない。
検証済み依存を `rocm-exl3-investigation:tested` imageにも保存した。終了時はcontainerを停止し、GPUを解放した。
container PID1は待機用sleepのため停止時statusは137だが、GPU workload各processの最終終了コードは0である。
container内のROCm compilerは `/opt/rocm/core-7.14`、R9700のみ可視化する。

```sh
docker start rocm-exl3-investigation
exl3_exec() {
  docker exec \
    -e LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1 \
    -e PYTHONPATH=/work/lib-autotune:/src \
    -e LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    rocm-exl3-investigation "$@"
}
exl3_exec python /work/probe_rocm_exl3_quantizer.py --output /work/tile-rerun.json
exl3_exec python /work/probe_rocm_exl3.py --model /models/qwen35-2b --output /work/unquantized-rerun.json
exl3_exec python /work/probe_rocm_exl3.py --model /work/models/qwen35-2b-exl3-4bpw --output /work/exl3-rerun.json
```

新しい出力先への変換（既存output/workは上書きしない）:

```sh
exl3_exec python /src/convert.py -i /models/qwen35-2b \
  -o /work/models/qwen35-2b-exl3-4bpw-rerun -w /work/quant-work-rerun \
  -b 4 -hb 6 -vb 16 -cr 16 -cc 256
```

sourceはread-only mountで、build先はcheckout外へ指定する。再build時は、モデル実行を停止してから
`MAX_JOBS=24 PYTORCH_ROCM_ARCH=gfx1201 ROCM_PATH=/opt/rocm/core-7.14` を指定し、container内 `/src` から
`python setup.py build_ext --build-lib /work/lib-rerun --build-temp /work/build-rerun` を実行する。
標準の `pip install --no-build-isolation .` と同じbuilderを呼ぶが、source mirrorへ生成物を置かない構成である。
新規checkoutには固定HEADへ [patch](../../../../references/patches/rocm-exl3-gfx1201.patch) を適用する。

`python-environment.txt` にpip freeze、`final-build-identity.json` に最終binary/source/ROCr hash、
`model-identity.json` に入力model hashを保存した。host toolchainの変更や別torch環境へ証拠を流用しない。
初期setupで不足していたlibatomic、C compiler、Python headersは専用container内へ導入した。
初期containerではread-onlyなsystem file mountがapt更新と衝突した。hostへの導入で回避せず、
container内に限定して環境を整備した。
最終containerはread-only system file mountを外し、`dpkg --audit`が空、`pip check`成功まで修復した。
source read-only mountへJIT出力しようとするimportは失敗するため、extensionのlink完了後に上記PYTHONPATHで読み込む。

## 終了時segfaultとAPI停止条件

最初のEXL3生成はfinite logitsと自然な文章を返したが、`model.unload()`後のprocess exitが139だった。
「unload完了」logだけでは成功と判定できない。非量子化は同条件でexit0。
GDBではtorch同梱ROCrの `hsa_signal_store_screlease → AqlQueue::~AqlQueue → GpuAgent::~GpuAgent →
Runtime::Unload → HIP RuntimeTearDown` にSIGSEGVを確認した。
原因をPython破棄順や特定source bugへ断定せず、上流が `os._exit()`で迂回している既知症状と区別して記録する。

host ROCr 1.21.0を上記LD_PRELOADで明示すると、英語EXL3生成と日本語EOS停止の両方が通常exit0になった。
同構成で全114件の数値・構造検査も再成功した。これはこの環境の検証済み回避であり、他GPU/torch版の保証ではない。
`runtime-loaded-libraries.json` とROCr hashを保持する。benchmark script自体は上流どおり `os._exit()`を使うので、
通常終了の証拠は別の自作probeのexit codeで確認した。

また `Job` の既定値はEOSでの停止を自動指定しない。最初の日本語probeは「5」の後も特殊tokenを出力した。
probeで `stop_conditions=config.eos_token_id_list` を指定した最終runは、非量子化/EXL3とも回答が正確に `5` で
EOS停止した。これはcaller側の指定修正であり、fork本体のbug修正には数えない。

## 実装の読み取り

[実装調査ノート](../../../../references/rocm-exl3-implementation.md) を参照。
単独kernelの検証はfull model品質、別GPU、tensor parallel、visionを証明しない。
生成テキストとfinite logitsは動作smokeであり、独立reference engineとの品質同等性の証明ではない。

## 再現用入口

- [生成・finite logit probe](../../../../../ci/tools/probe_rocm_exl3.py)
- [encoder構造検査](../../../../../ci/tools/probe_rocm_exl3_quantizer.py)
- upstream `rocm_tools/wmma_check.hip`、`gemm_check.hip`、`gemv_check.hip`、`attn_check.py`。
  WMMA harnessのbannerはgfx1151固定文字列のため、実行targetはcompile flagとUUIDで識別する。

計画: [調査計画](../../../../plans/archive/2026/09/11-20/rocm-exl3-investigation.md)

後続: [Qwen3.5-9Bでの比較](rocm-exl3-qwen35-9b-comparison.md)（5B以上の追加依頼、2026-09-17）。
