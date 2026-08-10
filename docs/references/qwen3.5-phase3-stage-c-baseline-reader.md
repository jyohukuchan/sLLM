# Qwen3.5 Phase 3 Stage C baseline op reader記録

## 範囲とprovenance

Stage C最初のbaseline opを、既存RMSNormを再実装せず、C1a contiguous BF16 copy/residual addとC1b embedding gatherへ分割するための技術的事実を記録する。外部codeのcopy、adapt、port、直接reuseは行わない。

| source | fixed commit | inspected path | 抽出した事実 |
| --- | --- | --- | --- |
| llama.cpp | `f5919bf458ef190468b5c329bb293f8a54a1e69c` | `ggml/src/ggml-cuda/getrows.cu`、`cpy.cu`、`ggml-cuda.cu` | gatherはinteger row indexごとにsource rowを選び、row内要素を独立に出力する。copy/addは別semantic opであり、backendはdtype/layout適合をdispatch前に判定する。 |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` | `vllm/model_executor/layers/vocab_parallel_embedding.py`、`linear.py`、`utils.py`、`activation.py`、`models/qwen3_next.py`、`layers/layernorm.py` | tensor parallel size 1ではtoken IDを直接embedding lookupへ渡し、TP用mask/reduceを行わない。unquantized linearはactivationとweightを渡すbias optionalのlinearで、weight storageは`[N,K]`、outputは末尾dimension `N`となる。SwiGLUはgate側へSiLUを適用してup側と要素積を取る。Qwen layerは最初のresidual capture後、attention/MLP境界でresidual addとRMSNormを繰り返す。 |

固定source identityとlicense境界は[source-lock manifest](source-lock.md)と[provenance方針](../provenance/README.md)を正とする。vLLMは意味・制約のreaderだけに使い、実装表現を参照しない。llama.cppも今回は直接流用せずclean implementationとする。

## C1a: BF16 copyとresidual add

- 既存`SemanticOpKind::Copy`と`Add`を使用し、新しい意味名を増やさない。
- input/outputはunquantized BF16、rank 1以上、zero extentなし、row-major contiguous、同一shapeに限定する。
- broadcast、dtype conversion、strided view、zero-length、input/output alias、2 input間のpartial overlapを受理しない。
- copyはBF16 storage bitを変えないbit-exact operationとする。
- residual addは各要素をFP32へ変換して加算し、outputだけをBF16 round-to-nearest-evenへ丸める。NaN/Infは演算結果として保持し、finite値へclampしない。
- baseline kernelは1 elementを1 work-itemで処理し、exact targetの単一HIP dispatchとする。最適化済み、fused add-RMSNorm、broadcast対応とは主張しない。
- copy/addを一つのversioned elementwise public C ABI familyで表し、operation kindだけをdescriptorへ固定する。prepareはmetadataをcopyし、executeは既存queue/completion ownershipへ接続する。

## C1b: embedding gather

- weightはunquantized BF16 `[vocab, hidden]`、token IDsはI32 rank 1、outputはBF16 `[tokens, hidden]`とする。
- Phase 3はsingle GPU/batch 1なのでTP mask、padding、all-reduceを実装しない。
- token IDは`0 <= id < vocab`をdevice dispatch前またはfail-closed device resultで検査し、負値/out-of-rangeをrow 0へclampしない。
- output row `t`はweight row `token_ids[t]`のbit-exact copyとする。duplicate IDはduplicate rowとして合法、zero token countはrejectする。
- C1aとC1bは別candidateとし、C1aのelementwise ABI/kernelをembedding固有契約へ拡張しない。

## C2: BF16 linearとSiLU gated MLP

- baseline linearはactivation BF16 `[M,K]`、checkpoint向きのweight BF16 `[N,K]`、output BF16 `[M,N]`とする。bias、batch rank 3以上、transposed/strided view、quantized weight、broadcastは受理しない。
- `M=1`のGEMVと`M>1`のGEMMは同じsemantic descriptorと数値oracleを共有する。各outputは`K`個のBF16 operandをFP32へ変換し、固定した`k=0..K-1`順にFP32 multiply/addして、最後にBF16 round-to-nearest-evenへ変換する。FP16 accumulation、TF32、split-K順序変更、CPU fallbackはbaseline契約外とする。
- weightはmodel storage `[N,K]`をそのまま読み、暗黙transposeした複製を要求しない。数式上は`A[M,K] x W^T[K,N] -> O[M,N]`である。
- SiLU gated multiplyは独立した2-input/1-output opとし、同一shapeのBF16 gate/upから各要素をFP32へ変換して`gate / (1 + exp(-gate)) * up`を計算し、outputだけをBF16へ丸める。gate/upを連結したfused storageやclamp、in-place aliasは受理しない。
- Qwen dense MLPは`gate=linear(x, gate_weight)`、`up=linear(x, up_weight)`、`mixed=silu_mul(gate, up)`、`output=linear(mixed, down_weight)`のtyped graphとして後続D0で接続する。C2はこのgraph全体を単一opaque kernelへ融合せず、linearとSiLU multiplyの再利用可能なbaseline ABIを提供する。
- baseline HIP kernelはcorrectness優先とし、linearは1 output elementを1 work-item、SiLU multiplyは1 elementを1 work-itemで処理する。性能最適化、hipBLAS/hipBLASLt選択、tile/shared-memory、fused MLP、一般shape最適化は主張しない。

## C3: text-only full attention

primary workspaceの固定checkoutを再確認し、llama.cppは`f5919bf458ef190468b5c329bb293f8a54a1e69c`、vLLMは`568afb3a13806beb53bb2e6bd518269357b237c0`と一致した。vLLM `qwen3_next.py:240-320,333-400`とllama.cpp `qwen3next.cpp:208-285`から次の意味と順序だけを抽出し、code表現はcopy、adapt、portしない。

- full-attention projectionはhidden BF16 `[M,2560]`からQ/gate BF16 `[M,8192]`、K/V BF16 `[M,1024]`を得る。Q/gateはheadごとの`[Q 256, gate 256]`で、`[M,16,512]`へreshapeして最後の軸を二分する。flat `[Q 4096][gate 4096]` splitは禁止する。
- Qは`[M,16,256]`、K/Vは`[M,4,256]`である。Q/Kはheadごとにepsilon `1e-6`、checkpoint raw scaleへ`1 + raw_scale`を適用するRMSNormを行い、その後にRoPEを適用する。VへRMSNorm/RoPEを適用しない。
- attention scaleは`head_dim^-0.5`、固定head dimension 256ではexact FP32 `1/16`とする。generationはcausal decoderで、query absolute position `q`から見えるkeyは`k <= q`だけである。GQAはquery head `h`がKV head `h/4`を参照し、K/Vのmaterialized repeat copyを要求しない。
- output gateはattention resultへFP32 `sigmoid(gate) * value`を要素ごとに適用してBF16へ丸め、その後に既存matmulで`o_proj [2560,4096]`へ渡す。既存`SiluMul`は意味が異なるため流用しない。
- text-only MRoPEは`rope_theta=10000000`、`rotary_dim=64`、NeoX pair、3軸同一absolute position、prefill `0..T-1`、decodeは既存prefix長から継続する。固定configの`mrope_interleaved=true`はmultimodalのfrequency-axis assignmentをinterleaveするが、text-onlyは全軸positionが同一なので最初の64 dimensionへの通常のpartial NeoX RoPEと数値的に同一である。将来multimodalへ拡張するときにsections `[11,11,10]`をcontiguous `[22,22,20]`軸区間と誤解して流用しない。
- full-attention KVは8 full layerごとのrequest-local opaque stateとし、K/Vを別の連続FP16 `[4,capacity,256]` row-major bufferとして所有する。論理length、capacity、layer、session、in-flight transitionを型に含め、appendは指定positionへK/Vを書いたcompletionが成功してからlengthをpublishする。失敗、timeout、早期drop、stale length、concurrent appendではlengthを進めず、resource ownershipを既存completion/reaper契約へ渡す。

baseline causal attentionの独立数値契約は、各query/headについてhead dimensionを`d=0..255`順にFP32 multiply/addし、exact scale `1/16`を乗じる。future keyはscore配列へsentinelを書かず候補集合から除外する。visible keyをabsolute position昇順に走査してFP32最大値を求め、同じ順で`exp(score-max)`と和を計算し、同じ順で正規化probabilityとVのFP32 weighted sumを計算して、最後だけBF16 RNEへ丸める。valid requestでは各query自身がvisibleなのでall-masked rowは生じず、空visible setはdescriptor/state errorとしてrejectする。この順序はsLLM baselineとして新規に固定するclean contractであり、外部engine codeの直接流用ではない。

C3は依存順に次へ分割する。

1. C3a0: lock済みtext configへRoPE、cache、attention gate、max positionをtyped fieldとして保持し、raw config validationだけに閉じ込めない。
2. C3a1: head-wise Q/gate split、Q/K head RMSNorm、text-only partial NeoX RoPE。KV stateとcausal attentionは含めない。
3. C3a2: request-local FP16 KV descriptor/handleとtransactional append。
4. C3b: GQA causal score、stable softmax、V reduction。
5. C3c: distinct sigmoid multiplyと既存`o_proj` matmulへのcontiguous handoff。

C3a0/C3a1のhost negativeはmissing/wrong typed config、8192/1024/256以外の幅、flat-half split、wrong head count、zero/overflow/noncontiguous/alias、position reset、position上限超過、wrong theta/rotary dimension/sections/interleaved flagを拒否する。C3a1 G1は`M=1/3/17`とposition `0/1/3/255/256/257`を交差させすぎないbounded setにし、signed zero/subnormal/finite大値/NaN/Infと独立scalar oracleを使う。C3a2/C3bはprefill `1/3/17/255/256/257`とdecode prefix `0/3/255/256/257`、capacity `B-1/B/B+1`を別caseで覆う。実weight G2は固定full layer 1層のQ/K/V/QNorm/KNorm sliceから始め、全8 layerとfull G3は後続integrationへ残す。

## 検証境界

C1a/C1bともhost negative contract、NumPyまたは独立scalar oracle、1/3/17/255/256/257要素、model hidden 2560境界、exact `gfx1030`/`gfx1201`のsynthetic semantic G1を持つ。実weight G2はC1bで固定embedding rowsをbounded readして追加し、全embedding matrixのhost複製を作らない。C2は`M/K/N`それぞれに1/3/17とtile境界前後を交差させすぎないbounded case set、`M=1`、非整列shape、signed zero/subnormal/finite大値/NaN/Infを含むscalar FP32 oracleを持つ。full `[2560,9216]` weightのhost複製は作らず、model実shapeはdescriptor/build boundaryと後続G3で確認する。Stage C unitのfocused結果はfinal G3または性能根拠へ昇格しない。
