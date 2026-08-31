# Phase 55: Gemma 4 26B-A4B MoE text production path

> 状態: 完了（2026-08-31）
> 作成日: 2026-08-31

## 目的

計画済みのGemma 4 MoEを、既存Gemma full/sliding attention、Qwen sparse MoE、NVFP4、GGUF、共通frontend／serverへ統合する。
architecture文字列のacceptだけでは完了とせず、固定した26B-A4B artifactを単一32 GiB AMD GPUの通常CLI／API／WebUIから
text generationに使用できるproduction pathまで実装する。

## 固定対象

- semantic source: `google/gemma-4-26B-A4B-it` revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`。
- primary artifact: `nvidia/Gemma-4-26B-A4B-NVFP4` revision
  `a19cfe00be84568a6867111c9a68c9c44fdcffe6`。
- text topology: hidden 2,816、30 layer、16 Q head、sliding 8 KV head × 256、full 2 KV head × 512、
  sliding window 1,024、context 262,144。
- MoE topology: 128 routed expert、token当たりtop-8、expert intermediate 704。dense MLPをshared branchとして実行し、
  routed expert branchと規定のnorm／residual順で結合する。
- artifact recipe: routed expert weight/activation NVFP4 block-16、attention／router／shared MLP等はartifactに固定された
  BF16/F32、KVはModelOpt 0.43 `fp8_cast`のconstant amax 448（scale bufferを持たない暗黙unit scale）に従う。
  保存されていないper-layer KV scaleを捏造せず、別dtypeへの暗黙fallbackを行わない。
- scope: text-only、single GPU、single active request、CLI／OpenAI-compatible API／dynamic model lifecycle／WebUI。
  image、video、audio、MTP、multi-GPU、expert parallel、tensor parallelは本Phaseに含めない。
- primary targetはR9700 exact `gfx1201`、secondary targetはV620 exact `gfx1030`とする。full residentがsecondaryへ
  収まらない場合は未実行をPASSにせず、operator／slice証拠へ限定する。

## 受入条件

1. 両sourceの完全revision、license、support file、shard size/hash、safetensors catalog、quantization recipe、tokenizer、
   chat templateをreviewed identityへ固定し、missing/extra/duplicate/range/shape/dtype/scale不一致をresident allocation前に拒否する。
2. Gemma routerのscaleなしRMSNorm、hidden-size root scale、softmax、stable top-8、top-k再正規化、per-expert scaleと、
   dense shared MLP＋routed expertのnorm／residual順を独立oracleへ一致させる。
3. token `1/3/7/8/17/31/32/33`、expert `0/127`、tie、skew、非finite、NVFP4 K境界`15/16/17`と
   `31/32/33`、attention window `1023/1024/1025`をhost contractと影響するexact GPU oracleへ含める。
4. container-neutral load plan、Gemma MoE graph、resident owner、request stateを共通prepared executionへ接続し、
   CPU numerical fallback、Qwen topology代用、全expert dense計算、requestごとのweight展開を行わない。
5. canonical GGUF architectureを`gemma4moe`としてreader/writer/converter/derived lock/model libraryへ追加し、
   sourceとGGUFのdescriptor、tensor bytes、生成結果を照合する。
6. tokenizer、BOS/EOS/stop、固定chat templateをfrontendへ接続し、raw prompt、Chat Completions、SSE、benchmark、
   model load/unload/cache、WebUIの既存操作を追加のarchitecture専用ユーザーflagなしで使えるようにする。
7. primary GPUでfixed/Unicode/code/stop、prefill 17、decode 17、連続要求、cancel/recovery、shutdownをHIP-only、
   fallbackなし、nonfiniteなし、cleanup 0でPASSし、reference runtimeとの固定token/logitまたはtask比較を別証拠として記録する。
8. affected host/GPU checks、1回のintegration review、互換性・runtime・model lock・GGUF・provenance・main plan・history同期を完了する。

性能最適化はarchitecture成立後にprofile上の支配箇所へ限定する。正しい基準providerが遅いことだけを理由に機能を未対応へ
戻さず、性能値と改善余地を履歴へ記録する。

## 実装順序

1. immutable source/artifact identity、config、catalog、NVFP4 recipeを実装する。
2. Gemma固有router／MoE semantic contractと独立oracleを追加する。
3. tensor mapping、load plan、Gemma MoE graph、resident executionを接続する。
4. `gemma4moe` GGUF converter／reader／derived lockを実装する。
5. tokenizer/template、CLI、server、dynamic model library、WebUIへ接続する。
6. actual artifact、GPU、service smoke、計画・履歴同期とintegration reviewを行う。

## 非対象

- Gemma 4 MTP／Diffusion／multimodal。
- NVIDIA/CUDA kernelの移植、vLLM／Transformers sourceのcopy。
- INT GGUF量子化、CPU offload、partial expert residency。
- 26B BF16 artifactを32 GiB targetへ無理に収容すること。

## 完了結果

- 固定sourceとcanonical `gemma4moe` GGUFをR9700 exact `gfx1201`で全resident実行し、17-token prefill、17-token decode、
  cancel/replayを含む35個の出力token列SHA-256が完全一致した。
- 通常CLI、raw Completions、Chat Completions非stream／SSE、Unicode／code／stop、prefix再利用、client cancel／recovery、
  dynamic model libraryのfolder選択／load／unload、metrics、Hugging Face検索／file command、統合WebUI起動／終了をactual modelでPASSした。
- source／GGUFの詳細identity、実測、integration defectと修正、cleanup監査、local AMD hostで実行不能なNVIDIA reference runtimeの
  証拠範囲は対応履歴へ記録した。
- host workspace test／clippy、WebUI test／lint／build、exact HIP evidence、1回のintegration review、関連正本文書の同期を完了した。

[対応する履歴](../../../../../history/2026/08/21-31/phase55-gemma4-moe.md)
