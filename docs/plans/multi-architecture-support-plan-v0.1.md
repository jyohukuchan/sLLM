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

この順序は「どれを入れるか」を決定するものではない。残り時間と発表で必要な demo 範囲を
踏まえて、人間が scope を選ぶための事実とリスクの整理である。
