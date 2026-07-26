# 追加アーキテクチャ対応: 調査・軽量検証・取得計画 v0.1

作成日: 2026-07-26

## 結論

Qwen3.5-9B の **AQ4_0** は既に専用実装で稼働対象になっている。一方で
Qwen3.5 dense の **SQ8_0** は未対応である。Gemma4（最小の
google/gemma-4-E2B）と Qwen3.5 MoE は、現行の Qwen3-14B SQ8_0
ローダーにそのまま載せられる差分ではない。特に MoE は expert routing と
grouped GEMM を新設する必要があり、残りの共有作業時間で完了を約束できる
規模ではない。

本書の検証は、新規アーキテクチャの解釈を独立実装と照合するための軽量な
HF CPU trace に限定する。既存の FP32 reference corpus、campaign、
authorization、candidate、release のいずれも使用しない。

## 調査の根拠と対象

設定はローカルの実体だけでなく、2026-07-26 に Hugging Face の同一 revision
から取得した config.json と SHA-256 を照合した。以下の設定 hash はローカルと
リモートで一致した。

| 対象 | Hub ID / revision | ローカル設定 SHA-256 | 判定 |
| --- | --- | --- | --- |
| Qwen3-14B 基準 | Qwen/Qwen3-14B-FP8 | c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793 | 既存 SQ8_0 |
| Gemma4 最小 | google/gemma-4-E2B / d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f | e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8 | 取得済み、gated ではない |
| Qwen3.5 dense | Qwen/Qwen3.5-9B / c202236235762e1c871ad0ccb60c8ee5ba337b9a | d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05 | 取得済み、gated ではない |
| Qwen3.5 MoE | Qwen/Qwen3.5-35B-A3B / 59d61f3ce65a6d9863b86d2e96597125219dc754 | 5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944 | 取得済み、gated ではない |

以下で「Gemma4」は特記しない限り上記最小 E2B の **text decoder** を指す。
モデル全体には vision/audio 構成もあるが、それを text-only の初回対応に暗黙に
含めない。

## Phase 1: 現行ローダーの境界

### qwen3_loader.rs が汎用である部分と固有である部分

crates/ullm-engine/src/qwen3_loader.rs は config.json を読まない。従って
architectures、model_type、head 数、RoPE 設定、正規化方式などの config field
を選択・検証する分岐はなく、architectures の値を固定決め打ちする処理も存在しない。
これは「汎用的に任意の architecture を読み込む」のではなく、呼出し元が既に
Qwen3 package であることを保証している設計である。

相対的に汎用な部分は package reader、RuntimeContext/RuntimeStream、テンソル形状
から得る hidden/head width、SQ overlay の受渡しである。主な入口は
Qwen3PackageModelRuntime::load と
Qwen3PackageModelRuntime::load_with_sq_overlay、layer 生成は
qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay である。

ただし decoder block は同関数内で次の名前と構成に固定されている。

- input_layernorm → self attention → post_attention_layernorm → MLP の residual 配置
- attention の q_proj / k_proj / v_proj / o_proj、q_norm / k_norm
- MLP の gate_proj / up_proj / down_proj と SiLU gated MLP
- attention scale は 1 / sqrt(head_dim)、MLP epsilon は 1e-5、既定 rotary 次元は
  head_dim / 4

crates/ullm-engine/src/qwen3_names.rs の alias も model. と
model.language_model. の layers / embed_tokens / norm を扱うのみで、任意の text
model 名や module tree を解釈しない。つまり、層が linear attention であること、
attention 出力 gate、追加の post norm、MoE、layer-specific head width、logit
soft-cap、embedding scale をこのローダーに渡しても表現できない。

### Qwen3.5 dense の既対応範囲

Qwen3.5 は Qwen3 generic loader の枝ではない。次の **AQ4_0 専用**実装である。

- crates/ullm-engine/src/qwen35_aq4_model_runtime.rs (Qwen35Aq4ModelRuntime)
- crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs
- crates/ullm-engine/src/qwen35_aq4_head_runtime.rs
- crates/ullm-engine/src/qwen35_aq4_session.rs
- crates/ullm-engine/src/qwen35_package_contract.rs

qwen35_package_contract.rs は self attention と linear attention を判別し、AQ4_0
実行時に hybrid layer と dense MLP を扱う。Qwen3.5 の RMSNorm は
loader.rs::effective_qwen35_rmsnorm_weight_values で 1 + weight を適用する明示的な
アーキテクチャ固有実装である（linear-attention の gated norm は別扱い）。従って
本番稼働中の Qwen3.5-9B は **AQ4_0 では対応済み**である。

対して Qwen3-14B の SQ8_0 は sq8_layer_runtime.rs、sq8_model_head_runtime.rs、
sq8_embedding_runtime.rs、sq8_stack_runtime.rs、sq8_worker_* が Qwen3-14B の
層数・幅・attention 構成を前提にしている。tools/sq8_canonical_artifact.py の
load_source_contract と tools/build-sq-fp8-w8a16-artifact.py も FP8 E4M3、
dynamic、128x128 block の 2-D weight を要求する。BF16 source で hybrid attention
の Qwen3.5-9B はこの経路に通らない。結論として **Qwen3.5 dense の SQ8_0 対応は
未実装**である。

### アーキテクチャを追加する際に触る箇所

以下は将来の実装範囲であり、この調査では変更していない。

| 領域 | 追加・変更が必要になるファイル / 関数 | 理由 |
| --- | --- | --- |
| config/loader | qwen3_loader.rs::{Qwen3PackageModelRuntime::load_with_sq_overlay,qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay}、qwen3_names.rs | config を contract として読み、architecture、layer type、RoPE、norm、head width、weight name を表現する loader に分離する必要がある。 |
| Qwen3.5 package | qwen35_package_contract.rs、qwen35_aq4_model_runtime.rs | dense 以外の package contract、MoE tensor layout、text-only/multimodal 境界を定義する必要がある。 |
| 量子化 | tools/sq8_canonical_artifact.py::load_source_contract、pair_fp8_weights、tools/build-sq-fp8-w8a16-artifact.py、crates/ullm-quant/src/main.rs::{run_one_direct_package_convert,run_direct_prototype_package} | BF16 input、非 2-D expert tensor、shared expert、layer-dependent width を package/format が表せるようにする。現行の SQ8_0 source contract は Qwen3 FP8 専用である。 |
| kernel dispatch | aq4_package_runtime.rs、backend_operation_registry.rs、qwen35_aq4_layer_runtime.rs、sq8_layer_runtime.rs、sq8_stack_runtime.rs | architecture-aware dispatch、hybrid linear/full attention、Gemma の local/global attention、追加 norm/PLE、又は MoE routing を結ぶ。runtime/src/kernels/ は新規実装時にだけ別途扱う。 |
| tokenizer / serving | services/openai-gateway/src/ullm_openai_gateway/tokenizer.py、deploy/served-models/*.profile.json | tokenizer hash/vocabulary、chat template、text-only と multimodal processor の serving contract を追加する。 |
| 軽量検証 | tools/architecture_hf_trace.py | HF の独立 trace と uLLM debug trace を同一 schema で比較する。既存 corpus/campaign を呼ばない。 |

## Phase 2: Qwen3-14B からの実デルタ

### 基準: Qwen3-14B-FP8

実際の config は architectures=[Qwen3ForCausalLM]、model_type=qwen3、hidden size
5120、40 layers、40 Q heads / 8 KV heads、head dimension 128、intermediate 17408、
SiLU、RMS epsilon 1e-6、RoPE theta 1,000,000 である。sliding window と RoPE scaling
は無く、tied embedding は false、vocabulary は 151936 である。Q/K projection は
5120/1024 wide、MLP gate/up は 17408 wide であり、weight source は FP8 E4M3
dynamic 128x128 block である。現行 loader の attention scale は 1/sqrt(128) である。
tokenizer は Qwen2Tokenizer で chat template を持つ。

### 1. Gemma4: google/gemma-4-E2B

| 項目 | config / checkpoint で確認した値 | Qwen3-14B からの意味のある差分 |
| --- | --- | --- |
| architecture | Gemma4ForConditionalGeneration / gemma4、text config は gemma4_text | conditional-generation wrapper（vision/audio を含む）。初回は text decoder のみに範囲を固定する必要がある。 |
| hidden / layers | 1536 / 35 | 幅・層数とも異なる。 |
| attention | 8 Q / 1 KV、local head dim 256、global head dim 512、HF 実装の scale は 1.0 | GQA 比・head dim が layer type により変化する。Qwen3 固定 head_dim の前提は使えない。 |
| attention 配置 | sliding が連続 4 層、次が full（full: 4, 9, 14, 19, 24, 29, 34） | window 512 の local attention と full attention が混在する。 |
| RoPE | sliding: theta 10,000。full: proportional theta 1,000,000、partial rotary factor 0.25 | layer type ごとに RoPE contract が異なる。 |
| normalization | RMSNorm epsilon 1e-6、HF 実装は weight を直接 scale（+1 ではない）。Q/K norm、V RMS norm、input/post-attn/pre-FF/post-FF/PLE post norm | Qwen3 の二つの residual norm より多い。正規化の位置を落とすと layer trace で即座に不一致になる。 |
| MLP | intermediate 6144、gelu_pytorch_tanh、use_double_wide_mlp=true | KV-shared layer は checkpoint 上 12288 wide MLP で、通常 layer の 6144 と異なる。 |
| PLE / residual | per-layer input size 256、embedding [262144, 8960]、projection [8960, 1536]、layer scalar | 各 decoder layer に PLE residual があり、embedding scale sqrt(1536) も必要。Qwen3 loader には対応する state がない。 |
| KV sharing | config num_kv_shared_layers=20 | Transformers 5.12.1 source は layer 15 以降を shared と扱う一方、checkpoint は layer 15/19/34 の physical K/V tensor も含む。この release の「どの K/V を実際に使うか」は HF trace で確認してから実装する必要があり、physical tensor の存在だけから推測してはならない。 |
| head / embedding | tied embedding true、final logit soft-cap 30 | lm head を別 weight として要求する Qwen3 path と異なる。最後に 30 * tanh(logits / 30) が必要。 |
| tokenizer/template | GemmaTokenizer。tokenizer config に chat template は無い。現環境の AutoProcessor は Gemma4Processor の optional dependency 不足で読めなかった。 | text tokenizer は確認済み、multimodal processor と chat prompt は未確認のため初回 support の範囲に含めない。 |

実テンソルも確認した。local layer 0 の Q/K は [2048,1536] / [256,1536]、global
layer 4 は [4096,1536] / [512,1536]、layer 15 の MLP gate/up は [12288,1536]
である。embedding は [262144,1536] で lm head は physical に別置きされない（tied）。

**新カーネル評価:** MoE のような grouped GEMM/routing primitive は不要である。ただし
現行の Qwen3 SQ8_0 kernels/dispatch が mixed head width、local mask、追加 norm、PLE、
soft-cap を表せる事実は確認できない。新しい汎用 primitive が不要でも、新しい runtime
composition と dispatch（場合によっては専用 kernel）が必要になる。

### 2. Qwen3.5 dense: Qwen/Qwen3.5-9B

| 項目 | config / checkpoint で確認した値 | Qwen3-14B からの意味のある差分 |
| --- | --- | --- |
| architecture | Qwen3_5ForConditionalGeneration / qwen3_5、text config qwen3_5_text | Qwen3 generic loader の architecture 名ではない。AQ4_0 専用 path が既にある。 |
| hidden / layers | 4096 / 32 | 基準より小さい。 |
| attention | 16 Q / 4 KV、head dim 256、softmax scale 1/sqrt(256) | GQA 比と head dim が異なる。Q projection は gate を含むため [8192,4096]。 |
| attention output gate | HF Qwen3_5Attention は Q projection を Q と gate に分割し、sigmoid(gate) を attention output に掛けてから O projection | Qwen3 にはない演算。 |
| hybrid layer | layer_types=[linear,linear,linear,full] * 8、full_attention_interval=4 | 4 層のうち 3 層は linear attention。linear key: 16 heads × 128、value: 32 heads × 128、conv kernel 4。 |
| RoPE | theta 10,000,000、partial rotary factor 0.25、mrope_interleaved=true、sections [11,11,10] | full attention でも Qwen3 の full-head RoPE とは違う。 |
| normalization | RMS epsilon 1e-6、Q/K/input/post/final は 1 + weight、linear-attention gated norm は raw weight | 既存 AQ4_0 が明示的に実装済みの重要差分。 |
| MLP | intermediate 12288、SiLU gated dense MLP | dense MLP 自体は既存 AQ4_0 と合う。 |
| embedding/head | vocab 248320、tied false、embedding/lm head [248320,4096] | vocabulary と token ID contract が異なる。 |
| logit soft-cap / embedding scale | text config に logit soft-cap 又は embedding-scale field は無い | config に無いことまでは確認済み。実装時は HF trace/source で text embedding の runtime 処理を固定する。 |
| wrapper | MTP one layer、vision components、multimodal Qwen template | 本番 AQ4_0 の確認範囲は language model。vision/multimodal と MTP は対応済みと確認できていない。 |

**新カーネル評価:** 既存の **AQ4_0** dense path に対しては不要である。SQ8_0 を
新たに要求する場合、MoE kernel は不要だが、BF16 source からの artifact 化、hybrid
linear attention、Q output gate、mRoPE、1+weight norm、head/tokenizer contract を
SQ8 dispatch に導入する必要がある。現行 AQ4_0 operation を SQ8_0 に自動再利用できる
根拠は無い。

### 3. Qwen3.5 MoE: Qwen/Qwen3.5-35B-A3B

| 項目 | config / checkpoint で確認した値 | Qwen3-14B からの意味のある差分 |
| --- | --- | --- |
| architecture | Qwen3_5MoeForConditionalGeneration / qwen3_5_moe、text config qwen3_5_moe_text | 観測済み architecture 値と一致する。現行 uLLM にこの executor はない。 |
| hidden / layers | 2048 / 40 | 基準より狭いが、層数は同じ。 |
| attention / hybrid | 16 Q / 2 KV、head dim 256、scale 1/sqrt(256)、[linear,linear,linear,full] * 10、Q output gate、mRoPE/partial factor 0.25/theta 10,000,000 | Qwen3.5 dense と同じ hybrid attention 系を持つ。 |
| router | 256 experts、top-k 8、router auxiliary loss 0.001 | HF Qwen3_5MoeTopKRouter は router logits を FP32 softmax、top-k 後に selected weight を和 1 に正規化して元 dtype に戻す。 |
| expert layout | experts.gate_up_proj [256,1024,2048]、experts.down_proj [256,2048,512]、router gate.weight [256,2048] | 現行 2-D dense matrix package と別の 3-D tensor contract。 |
| shared expert | shared gate/up [512,2048]、down 相当 [2048,512]、shared-expert gate [1,2048] | routed output に sigmoid(shared_expert_gate) * shared_mlp を足す。routed experts だけでは一致しない。 |
| normalization / soft-cap | Qwen3.5 dense と同じ 1+weight RMSNorm 系。text config に logit soft-cap field は無い | router と expert だけでなく Qwen3.5 hybrid block の norm contract も必要である。 |
| embedding/head | vocab 248320、tied false、embedding/lm head [248320,2048] | Qwen3.5 dense と同じ tokenizer family、head width は異なる。 |
| tokenizer/template | Qwen2Tokenizer、Qwen3.5 multimodal template | text tokenization は確認済み。vision/video processor の serving support は未確認。 |

**新カーネル評価:** 必要である。top-k routing、selected expert の gather/scatter、
expert-wise grouped GEMM、weighted reduction、shared expert/gate を一体として実行する
新しい MoE runtime path が必要である。検索した現行 package/dispatch には 3-D expert
weight または MoE executor は無かった。この点が三対象で最も支配的な工数リスクである。

## Phase 3: 独立 HF CPU trace ハーネス

### 成果物と設計

tools/architecture_hf_trace.py を追加した。これは既存の corpus/campaign 関連を
import せず、次の二境界だけを持つ。

1. capture-hf: local-only Hugging Face checkpoint を CPU で実行し、embedding、各 text
   decoder layer の出力、final norm、last-token logits を保存する。
2. compare: uLLM debug runner が保存した同一 schema の trace と、入力 token/step/config
   hash/shape を含めて比較する。失敗時は最初の step-NNNN__layer-NNNN を返す。

trace は metadata.json と tensors.npz（C-contiguous F32）で構成する。各 step の
input token、greedy next token、tensor 名、shape、config hash を記録するため、異なる
prompt、異なる config、異なる decode path を「一致」と誤認しない。最大 prompt は 64
トークン、最大 decode は 4 token、thread は 1--32（default 8）に制限した。

既存の global Python には Torch 2.12.0+cpu と Transformers 5.12.1 があったが、FP8
checkpoint の CPU load に必要な accelerate は無かった。そのため global environment を
変更せず、/home/homelab1/.venvs/ullm-architecture-hf を system-site-packages 付きで作り、
accelerate 1.14.0 のみを追加した。この venv でも torch.cuda.is_available() は false を
確認した。

実行例（新規 architecture の strict FP32 reference）:

~~~bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/homelab1/.venvs/ullm-architecture-hf/bin/python tools/architecture_hf_trace.py capture-hf --model-dir /path/to/model --token-ids 1,2,3 --output /tmp/reference --new-tokens 2 --threads 8 --dtype float32
tools/architecture_hf_trace.py compare --reference /tmp/reference --candidate /tmp/ullm-trace --report /tmp/trace-comparison.json
~~~

candidate writer は ullm.architecture_trace.v1 を出せばよく、この Python tool と link
する必要はない。現在の Qwen3 SQ8_0 trace は GPU RuntimeBuffer readback に依存し、
本タスクの GPU 禁止下では candidate を採取していない。将来の architecture 実装には、
この schema を出す **diagnostic-only** writer（production/campaign とは別）を先に付ける。

### 先に固定した判定基準

strict FP32 comparison の既定値は全 tensor に対して次である。

- element-wise: abs(candidate - reference) <= 5e-5 + 5e-4 * abs(reference)
- relative-error の分母 floor: 1e-4（ゼロ近傍で相対値を暴走させない）
- tensor L2 relative error: <= 1e-4
- NaN/Inf、schema/token/config hash の不一致、shape の不一致は即時失敗

この値は同じ FP32 weight と短い decode を別実装で実行する architecture bring-up の
診断として十分に小さく、単一要素の大きな実装差を隠さない値である。一方で GEMM の
演算順差を「bitwise でなければ失敗」とはしない。これは release gate ではない。量子化
candidate は本閾値を正しさの合格証に流用せず、まず unquantized/FP32 diagnostic path
で解釈を確定してから、別途明記した量子化誤差の診断として用いる。

### Qwen3-14B での自己検証結果

既対応基準で以下を CPU のみで実行した。Torch threads は 8、GPU は使用していない。

~~~bash
... architecture_hf_trace.py capture-hf --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8 --token-ids 151643,151644,198 --new-tokens 1 --threads 8 --dtype bfloat16 --allow-quantized-reference
~~~

- Transformers 5.12.1 / Torch 2.12.0+cpu は FP8 checkpoint を CPU で BF16 に展開した。
  そのためこれは strict FP32 semantic validation ではなく、trace plumbing の実行確認で
  ある。
- 40 layer、embedding、final norm、logits を含む 43 tensor を取得した。HF の
  load + 1 step は 58.9 秒、greedy next token は 654 だった。
- 同一 trace の serialize/deserialize compare は 43/43 pass（すべて誤差 0）だった。
- synthetic candidate の step-0000/layer-0003 の 1 要素を +1.0 すると、期待通り
  step-0000__layer-0003 を最初の失敗として reject した。

**未達を明記する:** Qwen3-14B の実際の uLLM SQ8_0 candidate trace は、GPU を使用
しないという本タスクの制約のため採取していない。よって「HF と uLLM の数値が一致した」
という主張はしていない。確認できたのは HF reference capture、trace schema、比較器、
layer-level failure localization である。これは重い FP32 corpus を代わりに走らせても
埋められない独立実装照合の欠落であり、GPU 許可時又は CPU diagnostic writer の完成後に
最初に実施する。

## Phase 4: 重み取得状況

保存先の慣習は /home/homelab1/datapool/ai_models/safetensors/ だった。確認時の
/home/homelab1/datapool 空き容量は 9,026,977,660,928 bytes（約 9.03 TB）である。

| 優先 | 対象 | ローカル path | safetensors | 状態 |
| --- | --- | --- | ---: | --- |
| 1 | Gemma4 E2B（最小） | /home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B | 1 shard / 10,246,621,918 bytes | 完全、remote config と照合済み |
| 2 | Qwen3.5-9B dense | /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B | 4 shards / 19,306,310,880 bytes | 完全、remote config と照合済み |
| 3 | Qwen3.5-35B-A3B MoE | /home/homelab1/datapool/ai_models/safetensors/Qwen3.5-35B-A3B-BF16 | 14 shards / 71,903,878,016 bytes | 完全、remote config と照合済み |

いずれも gated repo ではなかった。従って「早めに download を開始する」という待ち時間を
作る必要はなく、完全なローカル shard の存在とリモート config 一致を確認した時点で新規の
大容量 download は行わなかった。未完了 download を重ねる方が共有 I/O に悪影響である。

過去に試行された Qwen/Qwen3-Coder-Next-FP8 も
/home/homelab1/datapool/ai_models/safetensors/Qwen3-Coder-Next-FP8 に残っている。
40 shards、80,381,394,600 bytes を確認した。このモデルは本三対象の download 判断には
含めていない。

## Phase 5: 工数と着手順

これは implementation estimate であり、既に完了した調査・HF reference capture の
時間ではない。共有 60 時間のうち BA の decode 最適化と BB の CDNA3 作業にも時間が割かれる
前提で、意図的に楽観値を置かない。

| 対象 / 範囲 | 必要な主作業 | 新カーネル | 保守的見積り | 根拠 |
| --- | --- | --- | ---: | --- |
| Qwen3.5-9B **AQ4_0**（既存） | diagnostic trace exporter との接続、HF-uLLM layer 比較、既存 text-only 範囲の確認 | 不要 | 2--6 h | architecture/runtime は既に専用 AQ4_0 path があり、残るのは独立照合である。multimodal/MTP は含まない。 |
| Qwen3.5-9B **SQ8_0**（新規） | BF16→SQ8_0 artifact contract、Qwen3.5 package/loader、hybrid linear/full attention、Q gate、mRoPE、1+weight norm、head/tokenizer contract、trace | grouped GEMM は不要。ただし SQ8_0 dispatch/path の追加が必要 | 28--48 h | 現在の SQ8_0 は Qwen3 FP8 2-D source と Qwen3-14B 固定形状に限られ、AQ4_0 の演算をそのまま利用できない。 |
| Gemma4 E2B text-only | text package/loader、local/full mixed attention、mixed head width、Q/K/V norm、4 residual norm、PLE、tied head、soft-cap、tokenizer contract、trace | MoE primitive は不要。既存 dispatch で十分かは未確認で、新しい composition/dispatch は必要 | 48--72 h | 35 layers の複数 residual/norm と PLE/KV-sharing semantics が Qwen3 block と大きく異なる。vision/audio は含めない。 |
| Qwen3.5-35B-A3B MoE text-only | Qwen3.5 hybrid base に加え、3-D expert package、FP32 router softmax/top-k/renormalization、gather/scatter、grouped GEMM、weighted reduce、shared expert、trace | **必要** | 72--120 h | 256 experts/top-8 と shared expert があり、現行 executor/package に MoE 表現がない。性能ではなく正しい routing path だけでも新規面積が大きい。 |

### 推奨する順序

1. **trace writer を先に最小化する。** 新 architecture の loader を書く前に、
   architecture_hf_trace.py schema で embedding/layer/final/logits を出せる
   diagnostic-only endpoint を用意する。既存 corpus/campaign には接続しない。
2. **Qwen3.5-9B AQ4_0 を独立照合する。** これは既対応 dense architecture の解釈を
   最短で確認し、hybrid layer/mRoPE/1+weight norm の trace contract を検証できる。
3. **新規実装が必要なら scope を一つだけ選ぶ。** SQ8_0 が必須なら Qwen3.5-9B
   SQ8_0 の方が Gemma より狭い（ただし 28--48 h）。format を増やさないなら、
   Gemma4 E2B を text-only に強く限定して開始するが、48--72 h なので共有残時間では
   発表までの完了はリスクが高い。
4. **Qwen3.5 MoE は最後に置く。** grouped GEMM/routing を含むため、上記いずれかと
   並行に完遂するのは現実的ではない。残り時間の一部しか architecture work に使えない
   条件では、三対象すべての実装完了を約束する判断材料はない。

## BF: config 駆動ローダー設計と進捗

更新日: 2026-07-26

### 現行 Qwen3 前提の棚卸し

`crates/ullm-engine/src/qwen3_loader.rs` には、config を読まずに Qwen3 block を
組み立てる前提が **15 個**ある。ここでの個数は単なる型名やエラー文字列ではなく、重み
名・形状・実行構成を Qwen3 として固定する独立した契約を数えたものである。

| # | 箇所 | 固定されている前提 |
| ---: | --- | --- |
| 1 | `Qwen3PackageModelRuntime::{load,load_with_sq_overlay}` | 全 layer が同一 hidden/Q/KV head/head dim/value dim である。 |
| 2 | 同上 | package の選択 layer を decoder layer とみなし、architecture を確認しない。 |
| 3 | 同上 | attention scale は `1 / sqrt(head_dim)` である。 |
| 4 | 同上 | decoder MLP 用 epsilon は `1e-5` である（既存互換値）。 |
| 5 | `default_rotary_dim` | rotary dim は `head_dim / 4` を偶数化した値である。 |
| 6 | `qwen3_package_decoder_layer_runtime_from_package_with_sq_overlay` | layer namespace は `model.language_model.layers.{i}` である。 |
| 7 | 同上 | `input_layernorm → attention → post_attention_layernorm → MLP` の 2 residual-norm block である。 |
| 8 | 同上 | attention projection は `q_proj/k_proj/v_proj/o_proj` の 4 本である。 |
| 9 | 同上 | Q/K RMSNorm (`q_norm/k_norm`) が必ず存在する。 |
| 10 | 同上 | input/Q/K/post norm は Qwen3 型 RMSNorm weight として materialize する。 |
| 11 | 同上 | MLP は `gate_proj/up_proj/down_proj` の dense gated MLP である。 |
| 12 | `qwen3_self_attn_runtime_weights_from_package_with_sq_overlay` | Q/K head dim は norm length から得る。 |
| 13 | 同上 | KV head 数は K rows/head dim、value dim は V rows/KV head から得る。 |
| 14 | 同上と `qwen3_self_attn_runtime_shape` | Q projection は plain Qwen3 又は Qwen3.5 output-gate の 2-D layout に限る。 |
| 15 | `Qwen3PackageModelDecodePlan` と `decoder.rs` 呼出し | paged KV は全 layer 共通の full causal-attention layout で、local/linear/MoE state を持たない。 |

隣接する Qwen3-only 実装もある。`qwen3_names.rs` は `model.` と
`model.language_model.` の namespace alias に限り、`sq8_embedding_runtime.rs` と
`sq8_model_head_runtime.rs` は Qwen3-14B の vocab/hidden/tensor 名を固定する。
従って BF ではこれらの実行器を「汎用 executor」とは呼ばず、まず loader の入口を
architecture contract で閉じる。

### 読み取る config field（実 config に基づく）

loader は top-level の `architectures` を必須かつ単一要素として読み、wrapper model では
`text_config` を text decoder contract として読む。今回読み取る field は以下に限定する。

| architecture | 必須 field |
| --- | --- |
| `Qwen3ForCausalLM` | `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `hidden_act`, `rms_norm_eps`, `rope_theta`, `vocab_size`, `tie_word_embeddings`, `attention_bias`, `attention_dropout` |
| `Gemma4ForConditionalGeneration` | top-level `model_type` / `text_config`、text の `model_type`, width/layer/head/KV/head dim/intermediate/norm/activation/vocab/tied fields、`layer_types`, `sliding_window`, `rope_parameters`, `global_head_dim`, `num_kv_shared_layers`, `use_double_wide_mlp`, `hidden_size_per_layer_input`, `final_logit_softcapping` |
| `Qwen3_5ForConditionalGeneration` | top-level `model_type` / `text_config`、text の width/layer/head/KV/head dim/intermediate/norm/activation/vocab fields、`layer_types`, `full_attention_interval`, `attn_output_gate`, linear-attention width/head/conv fields、`rope_parameters.{mrope_interleaved,mrope_section,rope_theta,partial_rotary_factor}`、`mlp_only_layers` |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5 dense の共通 field に加え、`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`, `router_aux_loss_coef` |

実 config に無い値を既定値として発明しない。たとえば Qwen3.5 text config には
`tie_word_embeddings` が無く、wrapper top-level の `false` を読む。Gemma4 の full/local
RoPE は `rope_parameters` の layer kind ごとの object から読み、root の値へ平坦化しない。

### 実装境界

1. `model_config` module が package manifest の `source_model_dir/config.json`（又は明示的な
   model directory）から config を読み、single-architecture contract に解決する。source
   model directory/config が無い、JSON が壊れている、architecture が複数/未知、観測済み
   architecture の必須 field が欠ける場合は fail-closed とする。
2. 解決結果は `Qwen3`、`Gemma4Text`、`Qwen35DenseText`、`Qwen35MoeText` の明示的 enum
   で表す。layer type、RoPE、norm convention、MLP/MoE/embedding/head の差は enum 内の
   descriptor として保持する。文字列ベースの後段 switch は作らない。
3. 既存 `Qwen3PackageModelRuntime` は Qwen3 contract だけを受け入れ、config の幅・head
   数・layer count・activation・norm/embedding contract と package shape を照合する。
   既存 Qwen3 の runtime epsilon/rotary default は出力回帰を避けるため互換値のままとし、
   config の宣言値は contract として露出・検証する。実行セマンティクスを黙って変更しない。
4. 既存 AQ4_0 `Qwen35Aq4ModelRuntime` は Qwen3.5 dense contract と package layer pattern
   を照合する。これは新しい SQ8_0 executor を意味しない。
5. Gemma4 text と Qwen3.5 MoE は config/layer descriptor まで作るが、構成後にそれぞれ
   `Gemma4TextExecutor`、`Qwen35MoeExecutor` 未実装として明示的に停止する。MoE は
   grouped GEMM/routing/gather-scatter が無いため、dense executor にフォールバックしない。

この境界では kernel、SQ8_0 format、package conversion、multimodal processor を変更しない。
特に config が Gemma4/MoE だからといって Qwen3 executor で重みを読む経路は一切作らない。

この順序は「どれを入れるか」を決定するものではない。残り時間と発表で必要な demo 範囲を
踏まえて、人間が scope を選ぶための事実とリスクの整理である。

### 実装・検証結果（2026-07-26）

`crates/ullm-engine/src/model_config.rs` を追加し、package manifest の
`source_model_dir/config.json` から source config を SHA-256 とともに読み込むようにした。
`architectures` は単一要素でなければならず、未知値、source model directory の欠落、必須
field の欠落、未対応の Qwen3 `rope_scaling` はすべて fail-closed になる。既知の四系統は
`Qwen3` / `Gemma4Text` / `Qwen35DenseText` / `Qwen35MoeText` の型付き contract に解決する。
文字列を後段の既存 Qwen3 executor へ黙って流す fallback はない。

- Qwen3 package loader と Qwen3-14B `SQ8_0` serving/generation loader は Qwen3 contract
  と package/static geometry を device allocation より前に照合する。従来の Qwen3 runtime
  rotary dim と MLP epsilon は出力回帰を避けるため変更していない。
- Qwen3.5-9B `AQ4_0` の `Qwen35Aq4ModelRuntime` は Qwen3.5 dense config と package の
  embedding shape / 32 layer の linear/full pattern を load 前に照合する。演算・量子化・
  sampling path は変更していない。
- Gemma4 E2B と Qwen3.5-35B-A3B MoE は実 config を完全に descriptor へ組み立てるが、
  それぞれ `Gemma4TextExecutor` / `Qwen35MoeExecutor` が未実装として明示的に停止する。
  Gemma は local/full mixed attention、追加 norm、PLE、tied head、soft-cap が、MoE は
  routing / gather-scatter / grouped GEMM / weighted reduction / shared expert が停止理由である。

実 package/config を使った inspect の結果は次の通りだった。

| package / source config | architecture | 結果 |
| --- | --- | --- |
| Qwen3-14B `SQ8_0` | `Qwen3ForCausalLM`, config SHA `c5d7d0e8…233793` | existing Qwen3 full-attention executor を許可。5120 / 40 / 40Q / 8KV / 128 / 151936 を再現。 |
| Qwen3.5-9B `AQ4_0` | `Qwen3_5ForConditionalGeneration`, config SHA `d0883072…932b05` | existing AQ4_0 text executor を許可。32 層 `[linear,linear,linear,full] * 8`、Q gate、mRoPE、`1 + weight` norm を再現。 |
| Gemma4 E2B | `Gemma4ForConditionalGeneration`, config SHA `e5faef0d…ae73b8` | descriptor を組立て後、`Gemma4TextExecutor` 未実装で exit 2。 |
| Qwen3.5-35B-A3B MoE | `Qwen3_5MoeForConditionalGeneration`, config SHA `5e4d7f74…bc7944` | descriptor を組立て後、`Qwen35MoeExecutor` 未実装で exit 2。 |

unit / loader test は `model_config` 7件、`qwen3_loader` 9件、SQ8 trace writer 2件が
pass した。前者には unknown architecture rejection、source model directory 欠落 rejection、
Qwen3 rope-scaling rejection、Gemma/MoE の explicit unimplemented status を含む。

#### 既存モデルの実行回帰

- Qwen3.5-9B `AQ4_0` は `HIP_VISIBLE_DEVICES=1` の isolated R9700 上で既存
  `ullm-aq4-decode-step-profile 2 --warmup 0 --measured 1` を実行した。config contract を
  model load 前に通過し、M=2 prefill + M=1 decode は成功、次 token は **491**（local
  tokenizer で `2 produce` → ` new`）、elapsed 13.429 ms / 74.466 tok/s だった。
- Qwen3-14B `SQ8_0` は同じ R9700（HIP `gfx1201`、runtime device ID 0）で
  `ullm-sq8-architecture-trace` を実行した。config contract を通過し、token 198 の 1-step
  serving forward を正常に完了した。隔離を確認してから artifact を読む順序に直したため、
  `HIP_VISIBLE_DEVICES` を指定しない誤実行は重量 payload の展開前に失敗する。

#### HF trace との照合

`tools/architecture_hf_trace.py capture-hf` で local Qwen3-14B-FP8 を CPU BF16 reference として
token 198 / 1 step で採取し、43 tensor（embedding、40 layer、final norm、logits）を得た。
HF と SQ8_0 candidate は config SHA、input、shape、greedy next token **262** が一致した。
candidate は production/campaign/corpus を使わない diagnostic-only writer から採取し、GPU
load + one step は 945.2 秒だった。

strict compare（`atol=5e-5`, `rtol=5e-4`, `l2_relative_max=1e-4`）は **fail** だった。
embedding は bit-exact だが、SQ8_0 candidate と FP8/BF16 reference の差は最初の因果的 decoder
境界 `step-0000__layer-0000` から現れる（relative L2 `0.008560965`、max abs `0.05371094`、
4904/5120 element）。最終 norm は L2 `0.01470678`、logits は L2 `0.00815878`、
144625/151936 element が tolerance 外で、42/43 tensor が strict tolerance 外だった。
compare report の `first_failure` は lexical sort の `final-norm` だが、layer number 順で
局在化した最初の実行境界は layer 0 である。

これは strict numerical equality の達成ではない。現行 comparison は SQ8_0 と FP8/BF16
checkpoint を比較しており、unquantized uLLM Qwen3 diagnostic path は未実装のため、この差を
量子化誤差と既存 executor 差へさらに分解することは今回確認できなかった。一方、config
駆動化で新規に追加した経路は load-time validation のみであり、既存の数学演算を変更して
いない。top-1 が同一であることはこの一入力での回帰確認であって、strict tensor match の
代替合格条件ではない。

保存した artifact は
`benchmarks/results/2026-07-26/config-driven-loader-v0.1/` にある。ここには HF trace、SQ8
candidate trace、comparison report を残した（FP32 corpus / numerical gate / campaign は使わない）。

### 工数見積りへの影響

Gemma4 E2B **48--72 h**、Qwen3.5 MoE **72--120 h** は据え置く。今回取り除けたのは
architecture/config dispatch の共通前提だけであり、Gemma の layer composition と MoE の新規
kernel/routing primitive の実装面積は変わらない。特に MoE の 3-D expert payload、top-8 routing、
grouped GEMM、shared expert は config descriptor を持てても実行できないままである。

## BI: Qwen3.5-35B-A3B MoE ランタイム基盤（2026-07-26）

loader の変更とは独立に、MoE の実行プリミティブを runtime ABI として追加した。対象の
実 checkpoint を直接読んだ結果、text decoder は 40 層すべて MoE、`256 experts / top-8 /
I=512 / shared I=512` であり、routed `gate_up_proj` は BF16 `[256,1024,2048]`、`down_proj`
は BF16 `[256,2048,512]`、router は BF16 `[256,2048]` だった。shared expert とその
sigmoid gate も全層に存在する。詳細な source contract、decode/prefill を別経路とする設計、
workspace と VRAM 計算は
`docs/plans/qwen35-moe-runtime-foundation-v0.1.md` に記録した。

- CPU reference と public ABI は routing / gather / raw-BF16-or-F32 grouped GEMM /
  gated-SiLU / weighted scatter / shared sigmoid gate を段階ごとに保持する。gfx1201 用の
  correctness-first HIP 実装も同じ ABI を実装し、非 gfx1201 と feature 未指定を
  fail-closed にする。
- HF `Qwen3_5MoeTopKRouter` で実 layer-0 BF16 router fixture を生成した。3 token × top-8
  の expert ID と normalized score は CPU と R9700 の双方で **完全一致**（最大絶対誤差 0）。
  exact-tie は PyTorch の不安定 ordering を契約にせず、boundary tie として明示的に検出する。
- 実 layer-0 3-D `gate_up_proj` から source expert `[52,148]` の raw BF16 `[2,37,71]` slice
  を取り、local assignment `[1,0,1]` を通した。HF F32 expected、CPU reference、CPU ABI、
  R9700 grouped GEMM は **完全一致**（最大絶対誤差 0）であり、expert-axis/row/column layout
  も直接照合した。
- synthetic full MoE block では CPU C ABI は F32/BF16 とも全段 0 差、R9700 は最終出力
  `2.384185791e-7` 以内（BF16 の全段最大は shared gate/up `3.576278687e-7`）だった。これは
  timing ではなく correctness check である。
- R9700 は 31.859 GiB、text decoder の raw BF16 は 63.613 GiB（不足 31.754 GiB）、
  complete checkpoint は 66.965 GiB（不足 35.106 GiB）であり、full resident inference は
  実施不能である。量子化や CPU offload を暗黙に導入して代替しない。

この時点では hybrid linear/full attention、mRoPE、KV state、loader 結線、weight residency
policy が未実装なので、35B の end-to-end 生成は未到達である。従って `Qwen35MoeExecutor`
を実行可能とする変更ではなく、後続が利用する正しさ優先の MoE execution substrate までを
完了した。72--120 h は full text-only runtime と実推論まで含む見積りとして据え置く。一方、
ここで実装した substrate 単体はおよそ 16--28 h 規模に分解でき、残る工数の主因は attention
integration、streaming/quantization residency、prefill specialization、trace writer である。
