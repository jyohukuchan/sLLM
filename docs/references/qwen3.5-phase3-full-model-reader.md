# Qwen3.5 Phase 3 Stage B-D full-model reader記録

## 範囲と固定provenance

この文書は、固定したQwen3.5-4BのStage B-D実装へ渡す技術的reader結論だけを記録する。loader、kernel、runtime、tokenizer library、APIの恒久手順はここへ複製しない。外部sourceのcodeはcopy、adapt、port、直接reuseしていない。

| source | local path | 固定commit SHA | 用途 |
| --- | --- | --- | --- |
| llama.cpp | `reference/llama.cpp` | `f5919bf458ef190468b5c329bb293f8a54a1e69c` | semantic/reference reader only |
| vLLM | `reference/vLLM` | `568afb3a13806beb53bb2e6bd518269357b237c0` | semantic reader only |
| SGLang | `reference/SGLang` | `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`（tag object `d21f3c3a10606ba3c7bf43f981496da0a7d620cd`） | independent semantic cross-check only |

今回の確認対象pathは、vLLMの `vllm/model_executor/models/qwen3_next.py`、`qwen3_5.py`、`qwen3_5_mtp.py`、`qwen3_vl.py`、`layers/mamba/gdn/qwen_gdn_linear_attn.py`、`layers/mamba/mamba_utils.py`、`layers/layernorm.py`、SGLangの対応する `python/sglang/srt/models/qwen3_5.py`、`qwen3_5_mtp.py`、`qwen3_vl.py`、config/layernorm、およびllama.cppの `src/models/qwen3next.cpp` である。vision/MTPのHF safetensors orientationはvLLMをreader、SGLangを独立cross-checkとして概念だけを抽出し、codeはcopy、adapt、portしていない。固定sourceのidentityとlicense境界は[source-lock manifest](source-lock.md)と[provenance方針](../provenance/README.md)を正とする。

## fixed cache、model identity、read boundary

前readerの「fixed cache不存在」は撤回する。固定cacheは次に存在する。

```text
/home/homelab1/.cache/sllm/models/Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

| 項目 | 固定値 |
| --- | --- |
| repo / revision | `Qwen/Qwen3.5-4B` / `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| reader実施時lock fingerprint（停止policy導入前） | `sha256:89ba8a6b2e1b7c0324090ddf15ce0e673ff4c3dc242c4127690d490056d8efd1` |
| 現行lock fingerprint | `sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935` |
| cache files | 13（lock対象。`.gitattributes`はruntime/evidence対象外） |
| tensor index | 738 = text 426 + vision 297 + MTP 15 |
| index metadata total | `9319737856` bytes |

13 filesは `LICENSE`、`README.md`、`chat_template.jinja`、`config.json`、`merges.txt`、2つのsafetensors shard、`model.safetensors.index.json`、`preprocessor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`video_preprocessor_config.json`、`vocab.json` である。

このreader記録は、固定lock、config、index、safetensors header、bounded byte-range evidenceを根拠にする。現行lockは同じresolved revision、file identity、architecture、tensor catalogへversioned generated-token停止policyを追加したためfingerprintが更新されており、runtimeと後続evidenceは現行fingerprintを使用する。weight shard payloadをこのreaderが全read、全tensor decode、mmap、保存することはしていない。従って「13 files / 738 tensors」のcatalog、shape、dtype、offset、分類を記録するが、payload全体の意味的正しさやfull-model数値正しさをこの文書だけで証明しない。raw weight、slice、binary、traceをGitやreportへ保存しない。

## text configとexplicit schedule

実cacheの `config.json#/text_config` と固定lockから、次を契約とする。既定値への黙ったfallbackはしない。

```text
model_type                         = qwen3_5_text
dtype                              = bfloat16
hidden_size                       = 2560
num_hidden_layers                 = 32
intermediate_size                 = 9216
vocab_size                        = 248320
num_attention_heads               = 16
num_key_value_heads               = 4
head_dim                          = 256
hidden_act                        = silu
rms_norm_eps                      = 1e-6
tie_word_embeddings              = true
attention_bias                    = false
attention_dropout                 = 0.0
attn_output_gate                  = true
use_cache                         = true
max_position_embeddings           = 262144
linear_conv_kernel_dim            = 4
linear_key_head_dim               = 128
linear_num_key_heads              = 16
linear_num_value_heads            = 32
linear_value_head_dim             = 128
mamba_ssm_dtype                   = float32
full_attention_interval           = 4
mtp_num_hidden_layers             = 1
mtp_use_dedicated_embeddings      = false
mlp_only_layers                   = []
```

`layer_types`は長さ32の明示listで、0-basedのfull-attention layerは `{3, 7, 11, 15, 19, 23, 27, 31}`、残り24層はGDN/linear attentionである。

```text
[L,L,L,F, L,L,L,F, L,L,L,F, L,L,L,F,
 L,L,L,F, L,L,L,F, L,L,L,F, L,L,L,F]
```

`full_attention_interval=4`は整合性cross-checkであり、scheduleをlistから再推測する代用品ではない。top-levelは `Qwen3_5ForConditionalGeneration` / `qwen3_5`、text実行対象は `model.language_model` だけである。vision 297 tensorとMTP 15 tensorはlock済みknown-unconsumedであり、unknown、global ignore、実行済みとして扱わない。固定modelはdenseで、独立 `lm_head.weight` は要求しない（embedding tying）。

## main text tensor family

index/headerのtext catalogは426 tensorで、bias tensorはない。以下のshapeはsafetensorsのlogical shape、dtypeは固定実headerのcontractである。

| family | count | shape | dtype |
| --- | ---: | --- | --- |
| `model.language_model.embed_tokens.weight` | 1 | `[248320,2560]` | BF16 |
| `layers.*.input_layernorm.weight` | 32 | `[2560]` | BF16 |
| `layers.*.post_attention_layernorm.weight` | 32 | `[2560]` | BF16 |
| `layers.*.mlp.gate_proj.weight` | 32 | `[9216,2560]` | BF16 |
| `layers.*.mlp.up_proj.weight` | 32 | `[9216,2560]` | BF16 |
| `layers.*.mlp.down_proj.weight` | 32 | `[2560,9216]` | BF16 |
| full `self_attn.q_proj.weight` | 8 | `[8192,2560]` | BF16 |
| full `self_attn.k_proj.weight` | 8 | `[1024,2560]` | BF16 |
| full `self_attn.v_proj.weight` | 8 | `[1024,2560]` | BF16 |
| full `self_attn.o_proj.weight` | 8 | `[2560,4096]` | BF16 |
| full `self_attn.q_norm.weight` | 8 | `[256]` | BF16 |
| full `self_attn.k_norm.weight` | 8 | `[256]` | BF16 |
| GDN `linear_attn.in_proj_qkv.weight` | 24 | `[8192,2560]` | BF16 |
| GDN `linear_attn.in_proj_z.weight` | 24 | `[4096,2560]` | BF16 |
| GDN `linear_attn.in_proj_b.weight` | 24 | `[32,2560]` | BF16 |
| GDN `linear_attn.in_proj_a.weight` | 24 | `[32,2560]` | BF16 |
| GDN `linear_attn.conv1d.weight` | 24 | `[8192,1,4]` | BF16 |
| GDN `linear_attn.A_log` | 24 | `[32]` | F32 |
| GDN `linear_attn.dt_bias` | 24 | `[32]` | BF16 |
| GDN `linear_attn.norm.weight` | 24 | `[128]` | F32 |
| GDN `linear_attn.out_proj.weight` | 24 | `[2560,4096]` | BF16 |
| `model.language_model.norm.weight` | 1 | `[2560]` | BF16 |

required text、config-conditional、known-unconsumed、rejectedの分類を分ける。missing/duplicate/overlap/out-of-range、wrong dtype/shape、unexpected bias、wrong layer-class tensor、unknown main prefix、tie矛盾、quantized/converted checkpointはrejectする。`generation_config.json`は固定revisionに存在しないためplaceholderを作らない。

## visionとMTPのexpected shape導出

vision/MTPはPhase 3で実行しないが、known-unconsumedという分類はshapeやdtypeの検証省略を意味しない。固定sourceのdefault値は使わず、lock済み`config.json`から明示fieldを型付きで抽出して次の式へ代入する。visionは`N=depth`、`V=hidden_size`、`I=intermediate_size`、`C=in_channels`、`T=temporal_patch_size`、`P=patch_size`、`M=spatial_merge_size`、`S=M*M`、`O=out_hidden_size`、`E=num_position_embeddings`とする。`deepstack_visual_indexes`は空でなければ、現行297-name namespaceにない追加mergerを要求するためrejectする。

| vision family | count | expected shape | dtype |
| --- | ---: | --- | --- |
| `model.visual.blocks.*.attn.proj.weight` / `.bias` | `N` / `N` | `[V,V]` / `[V]` | BF16 |
| `model.visual.blocks.*.attn.qkv.weight` / `.bias` | `N` / `N` | `[3V,V]` / `[3V]` | BF16 |
| `model.visual.blocks.*.mlp.linear_fc1.weight` / `.bias` | `N` / `N` | `[I,V]` / `[I]` | BF16 |
| `model.visual.blocks.*.mlp.linear_fc2.weight` / `.bias` | `N` / `N` | `[V,I]` / `[V]` | BF16 |
| `model.visual.blocks.*.norm1.weight` / `.bias` | `N` / `N` | `[V]` / `[V]` | BF16 |
| `model.visual.blocks.*.norm2.weight` / `.bias` | `N` / `N` | `[V]` / `[V]` | BF16 |
| `model.visual.merger.linear_fc1.weight` / `.bias` | 1 / 1 | `[V*S,V*S]` / `[V*S]` | BF16 |
| `model.visual.merger.linear_fc2.weight` / `.bias` | 1 / 1 | `[O,V*S]` / `[O]` | BF16 |
| `model.visual.merger.norm.weight` / `.bias` | 1 / 1 | `[V]` / `[V]` | BF16 |
| `model.visual.patch_embed.proj.weight` / `.bias` | 1 / 1 | `[V,C,T,P,P]` / `[V]` | BF16 |
| `model.visual.pos_embed.weight` | 1 | `[E,V]` | BF16 |

現行namespaceのcountは`12N+9=297`なので`N=24`を必須とする。fused `attn.qkv.*`だけを受理し、外部engineがloaderで扱えるsplit `attn.q/k/v.*`を暗黙にaliasしない。mergerの第1linearはspatial shuffle後の`V*M^2`から同じ幅へ、第2linearは`O`へ写す。patch projectionはConv3dの`[out,in,kT,kH,kW]`である。全積・和はchecked arithmeticとし、zero、overflow、rank/dimension不一致をrejectする。

MTPはmain text configの`H=hidden_size`、`I=intermediate_size`、`A=num_attention_heads`、`K=num_key_value_heads`、`D=head_dim`を使う。`mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=false`、`tie_word_embeddings=true`を必須とし、独立embedding/lm-headを受理しない。

| MTP family | count | expected shape | dtype |
| --- | ---: | --- | --- |
| `mtp.fc.weight` | 1 | `[H,2H]` | BF16 |
| `mtp.layers.0.input_layernorm.weight` / `post_attention_layernorm.weight` | 1 / 1 | `[H]` / `[H]` | BF16 |
| `mtp.layers.0.mlp.gate_proj.weight` / `up_proj.weight` / `down_proj.weight` | 1 / 1 / 1 | `[I,H]` / `[I,H]` / `[H,I]` | BF16 |
| `mtp.layers.0.self_attn.q_proj.weight` / `k_proj.weight` / `v_proj.weight` / `o_proj.weight` | 1 / 1 / 1 / 1 | `[2A*D,H]` / `[K*D,H]` / `[K*D,H]` / `[H,A*D]` | BF16 |
| `mtp.layers.0.self_attn.q_norm.weight` / `k_norm.weight` | 1 / 1 | `[D]` / `[D]` | BF16 |
| `mtp.norm.weight` / `pre_fc_norm_embedding.weight` / `pre_fc_norm_hidden.weight` | 1 / 1 / 1 | `[H]` / `[H]` / `[H]` | BF16 |

固定text値では`mtp.fc.weight=[2560,5120]`、Q/K/V/Oはそれぞれ`[8192,2560]`、`[1024,2560]`、`[1024,2560]`、`[2560,4096]`となる。GDN storage dtypeは外部runtime実装から推定せず、固定safetensors headerを正とする。`linear_attn.dt_bias`はBF16 `[32]`、`linear_attn.norm.weight`はF32 `[128]`であり、B1開始時readerが両者を逆に記録した判断は固定cache照合で撤回した。

## full attention: Q/gate、GQA、RoPE

### head-wise Q/gate packing

full layerの `q_proj` は、各query headについて query 256要素の直後にoutput gate 256要素を置く。projection出力を `Y = X W_q^T`、`Y ∈ BF16[M,8192]` とすると、次のhead-wise reshapeだけが正しい。

```text
QG = reshape(Y, [M,16,512])
Q[m,h,j]    = Y[m,512*h + j]
gate[m,h,j] = Y[m,512*h + 256 + j]
0 <= h < 16, 0 <= j < 256
```

`Y[:,0:4096]`をQ、`Y[:,4096:8192]`をgateとするflat-half splitは禁止する。Q/Kはhead-wise RMSNorm（固定epsilon、Gemma系のeffective scaleは `1 + raw_weight`）後にRoPEを適用し、attention出力に `sigmoid(gate)` を乗じてから `o_proj` へ渡す。

Qは16 heads、K/Vは4 KV heads、head dimensionは256である。GQAではKV head `h` をquery headへ4回repeatする。

```text
Kq[m,4*h+r,j] = K[m,h,j]
Vq[m,4*h+r,j] = V[m,h,j]
0 <= h < 4, 0 <= r < 4, 0 <= j < 256
```

これはGDNのQ/K 16 headsからV 32 headsへのrepeatとは別の契約である。

### text-only MRoPE

`rope_theta=10000000`、`partial_rotary_factor=0.25`、`rotary_dim=64`、`mrope_interleaved=true`、`mrope_section=[11,11,10]`を使う。sectionのscalar幅は `[22,22,20]`で、rotary dimensionの軸は `[0,22)`がaxis 0、`[22,44)`がaxis 1、`[44,64)`がaxis 2である。残り192 dimensionにはRoPEを適用しない。

text-onlyでは3軸に同じabsolute positionを渡す。

```text
P[0,t] = P[1,t] = P[2,t] = t
axis(d) = 0 if 0 <= d < 22
          1 if 22 <= d < 44
          2 if 44 <= d < 64
```

NeoX pair `j=0..31`について `inv_freq[j] = rope_theta^(-2*j/64)`、`p = P[axis(j),t]` とし、

```text
c = cos(inv_freq[j] * p)
s = sin(inv_freq[j] * p)
rotated[j]    = x[j]    * c - x[j+32] * s
rotated[j+32] = x[j+32] * c + x[j]    * s
```

Q/Kのprefill positionsは `0..T-1`、decode positionsはprefix lengthを含む `T,T+1,...` であり、decodeでpositionをresetしない。vision/videoの異なるaxis positionを扱うmultimodal RoPEはPhase 3対象外である。

### full-attention cache

8つのfull layerだけがKV cacheを持ち、runtime dtype/layoutはFP16、logical shapeは `[4,T,256]`（batch 1、KV heads、sequence、head dimension）である。append、stride、capacity、prefill/decodeの軸順、request-local lifetime、alias禁止をKV descriptorへ含める。BF16 weight/activationやGDN stateへ暗黙変換しない。

## GDN / linear attention contract

GDN projectionは次のphysical channel orderである。

```text
in_proj_qkv: [q:16*128=2048][k:16*128=2048][v:32*128=4096] = 8192
in_proj_z:   z:32*128 = 4096
in_proj_b:   b:32
in_proj_a:   a:32
```

### convolution

`conv1d.weight`の固定storage shapeはBF16 `[8192,1,4]`であり、middleのsingleton input-channel次元もheader契約に含める。biasなし、depthwise channel 8192、causal kernel length 4である。request-local `conv_state`はBF16 row-major `[3,8192]`（stride `[8192,1]`）で、過去3 tokenを `[x[t-3],x[t-2],x[t-1]]` のoldest-to-newest順に保持する。current inputを加えた4 tapをoldest-to-current順に畳み込み、SiLUを適用する。prefill scanとdecode stepは同じstate transitionで、request間共有やposition resetをしない。

### L2、GDN repeat、gate/decay

convolution後の各128-vectorについて、Q/KのL2 normalizationはmain RMSNormと別opで、FP32で次を計算する。

```text
qhat = q / sqrt(sum_i(q[i]^2) + 1e-6)
khat = k / sqrt(sum_i(k[i]^2) + 1e-6)
```

Q/Kは16 heads、Vは32 headsで、Q/KをVへrepeat factor 2でinterleaveする。Full attentionのGQA repeat4と混同しない。

各value headのgateはFP32で、

```text
beta_t = sigmoid(float(b_t))
g_t    = -exp(float(A_log)) * softplus(float(a_t) + float(dt_bias))
```

`A_log`のstorageはF32 `[32]`、`dt_bias`のstorageはBF16 `[32]`で、式の評価時に`dt_bias`をFP32へcastする。`a`と`b`のprojection出力は32 channelsである。`g`はlog-space decayであり、低精度のままexp/softplusを評価しない。

### recurrent update

各value headのrequest-local recurrent stateを `S[h] ∈ F32[128,128]`（row-major、`[value_dim,key_dim]`）とする。prefill/decodeともzero stateからtoken順に同じ式を使う。

```text
r_t = v_t - S_(t-1) @ khat_t
S_t = exp(g_t) * S_(t-1) + beta_t * outer(r_t, khat_t)
o_t = S_t @ qhat_t
```

全stateは `[32,128,128]`、F32で保持する。GDNの `mamba_ssm_dtype=float32` はこのstate/arithmeticの根拠である。request completion前にstateをreuse/freeせず、request間で共有しない。

### gated normとprojection

GDN normはmain/full RMSNormのoffset-one scaleを流用しない。raw `norm.weight`をmultiplierとして、FP32 accumulationで、

```text
u = o / sqrt(mean(o^2) + 1e-6) * norm_weight
u = u * SiLU(z)
```

を行う。出力は `[T,32,128] -> [T,4096]` にflattenし、`out_proj [2560,4096]`へ渡す。

## dtype、accumulation、matmul shape、boundary

初期baselineのdtype契約は、BF16 input/weight、FP32 multiply/accumulate、BF16 outputである。GEMVは `M=1` の同一GEMM oracleと意味を共有し、FP16 accumulation、TF32、CPU fallback、全op一律の緩いtoleranceは禁止する。RMSNorm/L2、softmax、GDN beta/decay、recurrent updateもFP32 accumulationとする。KVはFP16、GDN recurrentはF32、conv stateはBF16であり、これらをBF16 activationへ一括同一視しない。

matrix notationは `[M,K] x [K,N] -> [M,N]` とする。

| operator | shape |
| --- | --- |
| MLP gate/up | `[M,2560] x [2560,9216] -> [M,9216]` |
| MLP down | `[M,9216] x [9216,2560] -> [M,2560]` |
| full q | `[M,2560] x [2560,8192] -> [M,8192]` |
| full k/v | `[M,2560] x [2560,1024] -> [M,1024]` |
| full o | `[M,4096] x [4096,2560] -> [M,2560]` |
| GDN qkv | `[M,2560] x [2560,8192] -> [M,8192]` |
| GDN z | `[M,2560] x [2560,4096] -> [M,4096]` |
| GDN b/a | `[M,2560] x [2560,32] -> [M,32]` |
| GDN out | `[M,4096] x [4096,2560] -> [M,2560]` |
| tied lm projection | `[M,2560] x [2560,248320] -> [M,248320]` |

必須caseは `M=1,3,17,255,256,257`、sequence length `511,512,513`、および各kernel tile/alignment境界の `B-1/B/B+1` とする。非整列M/K/N、zero、subnormal、大値、NaN/Inf境界も含める。

GPU toleranceはreader段階では未校正・未PASSである。実装後、同じimmutable candidateをcanonical `gfx1030` と `gfx1201`で実行し、op・shape・入力range・accumulation/output dtype別にFP32 host oracleとの差（max abs/relative、outlier、NaN/Inf）を記録して事前登録する。GPU不在、timeout、crash、CPU fallback、別SHA、unregistered toleranceはPASSにしない。

## tokenizer、chat template、EOS report境界

text-only frontendはQwen2Tokenizer系BPE、vocab 248320、model max length 262144、BOS自動追加なしを前提とする。system/user/assistantとgeneration promptのtemplate branchだけを対象にし、image/video/tool branchは明示unsupportedとする。

固定metadataには次の差異がある。

```text
config.json#/text_config/eos_token_id = 248044 = <|endoftext|>
tokenizer_config.json#/eos_token        = <|im_end|> = 248046
```

Phase 3 text-only stop policyは解決済みで、停止集合と判定順は `[248046,248044]`（`<|im_end|>`を先に判定）である。

- prompt tokenではstop判定しない。判定対象はgenerated-only tokenである。
- 新規生成tokenをargmaxした直後にstop判定し、stop token自身で即停止する。
- stop token自身およびstop後tokenをvisible decoded outputへ含めない。
- reportにはinput token IDs、generated token IDs、stop token ID、stop reason、max-tokenか明示stopかを記録する。
- `max_new_tokens=0`は生成前にmax-token停止とする。
- `generation_config.json`がないことを理由にplaceholderを作らない。

chat templateとtokenizerは同じ固定revisionから検証し、EOSの片方だけを暗黙採用しない。例えばtext-only `hello` のgeneration promptは、`<|im_start|>user`、本文、`<|im_end|>`、`<|im_start|>assistant`、thinking branchを含む固定templateの結果として扱う。

## Stage B-D sequenceとhandoff

これはcandidate境界とreader上の依存順を記録するもので、恒久的な実行手順ではない。

| candidate | readerで固定する責務 | handoff |
| --- | --- | --- |
| B0 | lock/config/index/header parser、typed text config、738-tensor catalog、verified range | `TextConfig`、`TensorCatalog`、immutable byte-range descriptors |
| B1 | tokenizer/decoder/chat-template frontend、generated-only stop policy | input/output token IDs、rendered text、stop metadata |
| B2 | bounded shard range、weight registry、required/known-unconsumed/rejected分類 | immutable weight descriptors |
| C0 | BF16 host oracle、shape/order、accumulation contract | deterministic oracle vectors/descriptors |
| C1 | copy/gather/RMSNorm/residual | normalized hidden/residual buffers |
| C2 | matmul/GEMV/GEMM、SiLU gated MLP | MLP output buffers |
| C3 | full attention、head-wise Q/gate、RoPE、GQA、KV | full-attention output/KV handle |
| C4 | GDN projection、conv、L2、beta/decay、recurrent、gated norm | linear-attention output/state handle |
| D0 | explicit 32-layer graph、residual、MLP wiring | execution plan |
| D1 | KV、conv/recurrent state lifetime、stride、prefill/decode state continuity | request-owned cache/state handles |
| D2 | prefill/decode、tied logits、greedy、stop/report | CLI generation result and G3 evidence |

API handoffの境界は、Rustのtyped execution plan/runtimeから、versioned public C ABIのdescriptor、buffer、cache/state handleへ渡す形とする。backendはexact target、dtype、layout、shape、alignment、lifetime capabilityを検証し、合わなければrejectする。Rustがrequest-local stateとcompletion lifetimeを所有し、GPU backendはopaque handleを扱う。candidate APIの正確なABI version、descriptor fields、ownership/error surfaceは未確定であり、実装候補として残す。

## 解決済み事項と残り

このStage B-D reader記録で、fixed cacheの存在・identity・payload全readなし境界、text config/schedule/tensor family、full/GDNのhead・state/cache、text-only MRoPE、EOS/template/stop/reportは解決済みとする。

残りは次の5点に限定する。

- tokenizer libraryの選定とtext-only template normalization boundary。
- candidate public API（ABI version、descriptor、ownership/error contract）の確定。
- 実装後のcanonical GPU tolerance校正。
- 独立したfull-model G3 golden token sequenceの確定。
- 両canonical hostのfull-model VRAM budget、allocation plan、peak VRAM。

G3 goldenはstop tokenから逆算せず、固定model lock、同一candidate、exact target、全dispatch、fallbackなし、token IDs、stop reason、health、cleanupを含む独立evidenceとして確定する。

[Stage A reader記録](qwen3.5-phase3-reader.md) / [Phase 3全体計画](../plans/active/2026/08/1-10/phase3-qwen35-4b-bf16.md)
