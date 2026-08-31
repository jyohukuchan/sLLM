# MiniMax M3 Phase 58 reader record

## 境界

この記録はMiniMax M3 Phase 58へ渡すcode表現を含まないsemantic／artifact要点である。公式CUDA MSA、
Transformers、vLLM、SGLang等のsourceをcopy、adapt、portしない。llama.cppも固定sourceのGGUF naming、shape、
layer scheduleを独立cross-checkするためだけに読み、source expressionやcontrol flowをsLLMへ移植しない。

## 固定source

| source | identity | 用途 |
| --- | --- | --- |
| [MiniMax M3 official artifact](https://huggingface.co/MiniMaxAI/MiniMax-M3/tree/f0e1c1e04d40177e4673a22097036854f536e9c0) | `f0e1c1e04d40177e4673a22097036854f536e9c0` | primary config、artifact、tokenizer、generation、processor、Community License |
| [MiniMax M3 official repository](https://github.com/MiniMax-AI/MiniMax-M3/tree/79882a353ea7d8b3b52ecaf6523ba7ab2a6fb6e5) | `79882a353ea7d8b3b52ecaf6523ba7ab2a6fb6e5` | model card／architecture summary |
| [MSA paper](https://arxiv.org/abs/2606.13392) | arXiv `2606.13392v2` | Multi-head Sparse Attention semantic |
| [official MSA repository](https://github.com/MiniMax-AI/MSA/tree/80434d7f67877c6570ca19cac444b84bc9855dac) | `80434d7f67877c6570ca19cac444b84bc9855dac` | operator境界のconcept確認だけ |
| [Transformers MiniMax M3 implementation](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py) | `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c` | sigmoid MoE演算順とreleased MTP境界のsemantic cross-check |
| local llama.cpp | `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` | canonical GGUF naming、shape、released MTP tensor absenceのcross-checkだけ |

model artifactのMiniMax Community LicenseとsLLM runtimeのMIT licenseを混同しない。固定`LICENSE`は3,339 bytes、
SHA-256 `b53f2fdda3049b0e9013207be51efc2d372cda1fcfdd8bb4bb8b22658ca5db9c`である。非商用利用、商用表示、
年間売上高が20 million USDを超える主体の事前書面許可、禁止用途等の条件はartifact取得／配布時に別途維持する。

## Artifact identityとmanifest不整合

- root architectureは`MiniMaxM3SparseForConditionalGeneration`、text architectureは
  `MiniMaxM3SparseForCausalLM`、`model_type=minimax_m3_vl`。
- exact `config.json`は5,254 bytes／SHA-256
  `c9c97ce1e4eece60012d5a10ea87717458bfb1f19c2c7a615a3dbff83d090c6b`。
- exact `model.safetensors.index.json`は2,706,437 bytes／SHA-256
  `54dbde502126d07f6999077437a06b5df1f71e317518956d0aad1c8197df524e`、59 shard、23,416 tensor。
- shard file合計は854,176,398,808 bytes、headerを除くdtype由来tensor bytesは854,172,958,720 bytesである。
  一方、index `metadata.total_size=869,157,697,024`はshard file合計より14,981,298,216 bytes大きい。
  loaderはこの差を丸めたり修正した値で置換せず、manifest mismatchを保持し、容量admissionには大きい方を使う。
- Hub APIのdtype別parameter countはBF16 426,993,800,960、F32 46,339,200である。LFS OIDはremote artifact
  identityであり、local full payload hashの証拠ではない。
- 23,416 tensorはtext 22,893、vision／projector 523へ分かれる。textはroot 3、先頭dense 3 layer×11、
  sparse MoE 57 layer×401である。routed expert sourceは57×128×3=21,888 tensor。
- configは`num_mtp_modules=7`、`num_nextn_predict_layers=1`を持つが、indexにはMTP／next-token tensor名が存在しない。
  runtimeはconfigだけから未公開MTP weightを補わず、released artifactのMTP productionをunsupportedとして扱う。

## Text topology

- hidden 6,144、vocab 200,064、60 layer、context 1,048,576、RMSNorm epsilon `1e-6`、
  Gemma-style `(1+w)` norm、SwiGLU-OAI alpha 1.702／limit 7.0。
- attentionは64 query head、4 KV head、head dim 128、partial RoPE dim 64、theta 5,000,000、per-head QK norm。
- layer 0..2はdense FFN intermediate 12,288とdense GQA。layer 3..59はMSAとMoEを使う。
- MoEは128 routed expert、stable top-4、shared expert 1、expert／shared intermediate 3,072、sigmoid score、
  routing bias、routed scaling factor 2.0。selectionはunbiased sigmoid score＋bias、mix weightは選択後にbiasなしscoreを
  top-4内で正規化する。routed weighted sumだけを2.0倍し、shared expert出力はscaleせず加算する。
- fixed Transformers実装もMTP generationを実装せず`mtp.*`をunexpected-load対象外としている。config名だけから
  speculative control flowを推測しない。

## Multi-head Sparse Attention

- index branchはGQA groupごとに1 query head、全groupで共有するindex key表現を使う。M3 configは4 index head、
  index dim 128である。
- token scoreはscaled index Q・K dot product、causal block scoreはvisible token scoreの最大値である。
- block sizeは128、top-kは16。current local blockを必ず含め、その1 blockもtop-k slotを消費する。
  top-k未満しかvisible blockがない場合は存在するblockだけを返す。
- block選択はGQA groupごとに独立する。main branchは選択されたblock内のcausally visible tokenへ通常のscaled
  dot-product／exact softmax attentionを適用する。block scoreだけをattention weightとして使わない。
- layer 0..2はsparse index tensorを持たずdense attentionである。layer 3..59だけにQ/K index projectionとnormが存在する。
- official MSA repositoryはMITだがNVIDIA SM100／CUDA向けである。Phase 58はそのkernelをHIPへportせず、
  127／128／129境界、partial current block、stable tie、per-group選択を独立FP32 oracleへ固定する。

## Vision／special token

- vision towerはhidden 1,280、32 layer、16 head、intermediate 5,120、patch 14、image size 2,016、3D RoPE。
- patch mergeはspatial 2、temporal 2。text projectorは6,144、image token 200025、video token 200026、
  BOS 200019、EOS 200020。
- Phase 58はvision config、tensor family、processor identityをcatalogへ保持するだけで、multimodal executionを
  production CLI／API／WebUIへ接続しない。

## Quantized artifact availability

- official `MiniMaxAI/MiniMax-M3-MXFP8` revision
  `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`は31 shard、file合計443,749,077,256 bytes。
- AMD `amd/MiniMax-M3-MXFP4` revision `b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a`は59 shard、
  file合計242,666,026,728 bytes。
- NVIDIA `nvidia/MiniMax-M3-NVFP4` revision `901464083161bf8612a29ff7ad29914cd4ab4a85`は88 shard、
  file合計250,103,762,320 bytes。
- いずれもlocal 32 GiB単一GPUにも3台合計にも収まらない。未reviewed community pruning／quantizationを
  production artifactへ無断で採用しない。

## Canonical GGUF cross-check

fixed llama.cppのarchitecture名は`minimax-m3`である。source 21,888 expert別tensorは57 layer×3 projectionの
171 expert-axis tensorへlosslessにstackできるが、source payloadを読まないPhase 58はmapping dry-runに留める。
Gemma-style normの`+1` bake、vision patch temporal slice、dtype変換、量子化をmetadata-only catalogが実行したとは扱わない。

## Phase 58の証拠範囲

- official BF16 artifactはweightだけでlocal R9700 34,208,743,424 bytesの約25倍である。KV／workspace前に容量を超える。
- Phase 58はidentity、typed config／catalog、manifest mismatch、checked capacity、MSA／MoE oracle、model-free／verified slice GPU、
  GGUF dry-run、model library gray表示をfoundation証拠とする。
- full resident、通常generation、multimodal、MTP、CLI／API／WebUI production対応、性能を主張しない。

[対応する計画](../plans/archive/2026/08/21-31/phase58-minimax-m3-foundation.md)
