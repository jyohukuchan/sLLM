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
| KV sharing | config num_kv_shared_layers=20 | Transformers 5.12.1 source は layer 15 以降を shared と扱う。各 attention kind の最後の非共有層（この E2B では sliding=13、full=14）が full-length K/V を保持し、layer 15 以降はそれを再利用する。checkpoint に残る shared 層の physical K/V tensor は HF module がロードしない。 |
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
| Gemma4 E2B text-only | source-BF16/F32 diagnostic executor に加え、complete-checkpoint resident BF16 weight、device K/V cache、prefill/decode、trace と throughput evidence は実装済み。package/loader、継続的 serving、SQ8_0 量子化は残る | MoE primitive は不要。既存 SQ8_0 dispatch は Qwen3-14B 固定のため、そのままは使えない | diagnostic 16--28 h / package・量子化・serving 48--72 h | 35 layers の複数 residual/norm と PLE/KV-sharing semantics は実装確認済み。streaming diagnostic path と resident BF16 path は分離した。vision/audio は resident payload に含めるが実行対象には含めない。 |
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
5. Gemma4 text は source BF16 weight / F32 activation の diagnostic-only
   `Gemma4TextExecutor` を持つ。これは package、SQ8_0/AQ4_0、multimodal、serving の
   fallback ではなく、HF layer trace と生成の architecture bring-up 専用である。
   Qwen3.5 MoE は引き続き `Qwen35MoeExecutor` 未実装として明示停止する。MoE は
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
- Gemma4 E2B は実 config を完全に descriptor へ組み立て、source BF16 weight / F32
  activation の diagnostic-only `Gemma4TextExecutor` を許可する。これは local/full mixed
  attention、追加 norm、PLE、tied head、soft-cap をHF実装どおり合成する独立 path であり、
  既存の量子化 serving path には接続しない。Qwen3.5-35B-A3B MoE は引き続き
  `Qwen35MoeExecutor` 未実装として停止し、routing / gather-scatter / grouped GEMM /
  weighted reduction / shared expert が停止理由である。

実 package/config を使った inspect の結果は次の通りだった。

| package / source config | architecture | 結果 |
| --- | --- | --- |
| Qwen3-14B `SQ8_0` | `Qwen3ForCausalLM`, config SHA `c5d7d0e8…233793` | existing Qwen3 full-attention executor を許可。5120 / 40 / 40Q / 8KV / 128 / 151936 を再現。 |
| Qwen3.5-9B `AQ4_0` | `Qwen3_5ForConditionalGeneration`, config SHA `d0883072…932b05` | existing AQ4_0 text executor を許可。32 層 `[linear,linear,linear,full] * 8`、Q gate、mRoPE、`1 + weight` norm を再現。 |
| Gemma4 E2B | `Gemma4ForConditionalGeneration`, config SHA `e5faef0d…ae73b8` | diagnostic-only `Gemma4TextExecutor` を許可。35層、`[sliding×4, full]×7`、PLE/KV共有/tied head/soft-cap の source-BF16/F32 path を選択。 |
| Qwen3.5-35B-A3B MoE | `Qwen3_5MoeForConditionalGeneration`, config SHA `5e4d7f74…bc7944` | descriptor を組立て後、`Qwen35MoeExecutor` 未実装で exit 2。 |

unit / loader test は、unknown architecture rejection、source model directory 欠落 rejection、
Qwen3 rope-scaling rejection、Gemma diagnostic contract、MoE の explicit unimplemented
status を含めて実行した。Gemma executor 側では BF16 展開、direct-weight RMSNorm、proportional
RoPE の非回転 channel、sliding window を個別に検証した。

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

#### BL: Gemma4 E2B text-only non-quantized executor（2026-07-26）

`google/gemma-4-E2B` の local source directory
`/home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B` を直接読んだ。`config.json` は
SHA-256 `e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8`、単一 shard
`model.safetensors` は 10,246,621,918 bytes / SHA-256
`76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57` だった。
`ullm-model-config-inspect --require-executor` は `Gemma4ForConditionalGeneration` を
diagnostic-only の `Gemma4TextExecutor` として受理し、35層、hidden 1536、8Q/1KV、
local/global head 256/512、`[sliding×4, full]×7`、window 512、KV共有20、PLE 256、tied
embedding、final soft-cap 30 を実configどおり出力した。

HF の唯一の基準は local environment の Transformers 5.12.1、
`transformers/models/gemma4/modeling_gemma4.py` とした。以下は名前からの推測ではなく、
実際に読んだコード位置とその適用内容である。

| 項目 | HF code location | 実装した意味論 |
| --- | --- | --- |
| local/full attention | `Gemma4TextAttention.__init__` L1180--1204、`forward` L1243--1289 | `layer_types` から sliding を判定し、sliding は window 512 / head 256、full は head 512。attention scale は L1194 の **1.0**。local/full の各最後の非共有層だけが L1201--1204 の規則で shared K/V を保存し、L1248--1270 の規則で共有層が再利用する。E2B では source layers 13/14 がそれぞれ local/full K/V を供給する。 |
| local/full RoPE | `Gemma4TextRotaryEmbedding` L1087--1174、`modeling_rope_utils.py::_compute_proportional_rope_parameters` L187--254 | sliding は default theta 10,000 の full-width RoPE。full は `global_head_dim` を明示して proportional theta 1,000,000 / partial 0.25 とし、512-wide head の前半64 pairだけを回す。HF の float32 frequency construction と `rotate_half` と同じ half-split 回転を実装した。 |
| attention soft-cap | `eager_attention_forward` L821--852、`Gemma4TextAttention.forward` L1272--1285 | eager attention 自体は optional `softcap` を持つが、Gemma4 text attention は `scaling` と `sliding_window` だけを渡し softcap を渡さない。従って E2B text attention logits には soft-cap を掛けない。 |
| RMSNorm と residual 配置 | `Gemma4RMSNorm` L193--211、`Gemma4TextDecoderLayer.forward` L1409--1455 | norm は F32 variance / eps `1e-6` の後に weight を**直接**掛ける（`1 + weight` ではない）。input norm → attention → post-attention norm → residual、pre-FF norm → MLP → post-FF norm → residual、PLE residual → `layer_scalar` の順である。Q/K は weight付き head RMSNorm、V は `with_scale=False` の RMSNorm。 |
| MLP | `Gemma4TextMLP` L1068--1084、`activations.py::GELUTanh` L110--114 | `gelu_pytorch_tanh(gate) * up` を down projection する。`num_hidden_layers - num_kv_shared_layers = 15` 以降は L1071--1079 の double-wide 条件で 12,288、前段は6,144。GELU literal / 演算順もHFに合わせた。 |
| PLE | text model init L1612--1630、`get_per_layer_inputs` L1737--1779、`project_per_layer_inputs` L1781--1815、layer apply L1445--1452 | token PLE `[vocab, layers×256]` を sqrt(256) でscaleし、main embeddingからの projection を 1/sqrt(1536) でscale、per-layer direct RMSNorm後に `(projection + token_identity) / sqrt(2)`。各layerで gate → GELU → PLE multiply → projection → norm → residual を適用する。 |
| embedding/head/final cap | `Gemma4TextScaledWordEmbedding` L1458--1470、text init L1600--1603、conditional head L2445--2454 / L2528--2535 | input embedding は sqrt(1536) でscaleする。conditional wrapper の `lm_head.weight` は `model.language_model.embed_tokens.weight` と tied。final logits のみ `30 * tanh(logits / 30)` を掛ける。 |

checkpoint 形状も上の分岐を検証した。layer 0 local Q/K/V/O は
`[2048,1536] / [256,1536] / [256,1536] / [1536,2048]`、layer 4 full は
`[4096,1536] / [512,1536] / [512,1536] / [1536,4096]`、layer 15以降の MLP
gate/up は `[12288,1536]`、PLE embedding/projection は `[262144,8960]` /
`[8960,1536]` である。HF text model が L1632--1638 で shared-layer K/V projection
weight を unexpected keys として無視することも確認し、executor は physical tensor の存在を
実行根拠にしていない。

実装は `crates/ullm-engine/src/gemma4_text_executor.rs` と
`ullm-gemma4-text-trace` である。safetensors source BF16を直接読み、既存の
`ullm_runtime_matvec_bf16_f32`（`runtime/src/ullm_runtime_api_primitives.inc` L122--214）へ
streamし、activation/attention/softmax/norm/PLE/logitsはF32で保持する。CPU staging fallbackは
`ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1` を要求して拒否し、runtime identity は HIP
`AMD Radeon Graphics` / `gfx1201` / compute 12.0 / 30--34 GiB のR9700だけを受理する。
V620 (`gfx1030`) は明示的に選択しない。新カーネルは不要だったため、BHが編集中の
`runtime/src/ullm_runtime_parts/part_01.inc` と
`runtime/src/ullm_runtime_hiprtc_sources.inc` は変更していない。BKの
`sq8_serving_runtime.rs` と既存 `AQ4_0` / `SQ8_0` production codeにも変更はない。

`model_config.rs` は executor が source semantics を fail-closed に確認するため、
`max_position_embeddings`、`use_bidirectional_attention`、
`num_global_key_value_heads` と `Gemma4TextNonquantized` status を追加した。これはBFの
config descriptorを変更する必要があった唯一の理由であり、non-null bidirectional mode、MoE、
attention bias/dropout、alternate K=V attention、異なる norm/RoPE/activation は受理しない。

非量子化の層比較は CPU F32 HF reference（Torch 2.12.0+cpu、threads=8）対 R9700の
BF16 source/F32 activation candidate で行った。これは promotion / bit-exact gate ではなく、
architecture解釈の局在化専用である。結果 artifact は
`benchmarks/results/2026-07-26/gemma4-e2b-nonquantized-v0.1/` にある。

| 入力 / scope | 生成 token | 比較した tensor | 観測された最大差 | 局在化結果 |
| --- | --- | ---: | --- | --- |
| token `2` / 1 step | `184` | 38 | final norm abs `3.9100647e-5`、relative L2 `1.4548e-6` | layer 0--34すべてF32丸め範囲で連続し、構造的な最初の乖離なし。 |
| token `2` / 2 decode steps | `184, 3910` | 76 | step1 final norm abs `3.2901764e-5`、relative L2 `1.2634e-6` | shared local/full K/Vを含むdecodeも同じ。 |
| `The capital of France is` / 4 steps | `9079, 236761, 108, 818` | 152 | abs max `1.0681152e-4`、最大 relative L2 `2.0131e-6` | 同じ各step/layerを通過。両者のdecodeは `The capital of France is Paris.\n\nThe`。 |
| `Once upon a time,` / 4 steps | `528, 496, 1902, 1298` | 152 | abs max `1.1825562e-4`、最大 relative L2 `2.6786e-6` | 同じ各step/layerを通過。両者のdecodeは `Once upon a time, in a world where`。 |

comparison JSONの `pass` は tool の既定表示であり、ここでは採否閾値として使用していない。
すべての layer output、final norm、soft-capped logits の実測値を確認し、最初の構造的乖離層が
無いことを示す診断に限定した。`The capital ...` を8 tokenまで伸ばすとHF/uLLMともに入力を
反復したが、これは base checkpoint のgreedy continuationそのもので、uLLM固有の壊れ方では
ない。文章品質の確認には上記の一致した物語導入も併用した。

Phase 4は、Phase 3を通過した後に read-only で既存SQ8_0の境界を確認した。
`sq8_layer_runtime.rs` L115--175 / L349--385 と `sq8_stack_runtime.rs` L267--275 は
`Qwen3Sq8LayerConfig::qwen3_14b`、Qwen3-14Bのfixed hidden/KV/head/intermediate、
fixed norm/RoPE、固定layer arrayを要求する。Gemmaのlocal/full mixed width、scale=1.0、
PLE、shared K/V、tied source head、final capを表せないため、既存SQ8_0 production codeを
変更せず流用することはできない。したがってこの変更では量子化artifactや serving を作らず、
Phase 4は「専用 artifact/descriptor + resident executor が別途必要」と記録した段階に留めた。
非量子化の原因切り分けを量子化誤差で汚さないという順序は守られている。

### 工数見積りへの影響

Gemma4 E2Bの見積りは scopeを分けて更新する。既存BF16×F32 matvecを使う
diagnostic-only source executor / trace / short greedy generationは **16--28 h** が妥当だった。
一方、BCの **48--72 h** は source-to-quantized package、resident weight管理、mixed-width
prefill/decode、KV cache、tokenizer/serving integration、実用速度を含むtext-only production
scopeとしては据え置く。今回の実装は後者を完了したという主張ではない。Qwen3.5 MoE
**72--120 h** も据え置く。MoEの3-D expert payload、top-8 routing、grouped GEMM、shared
expert は config descriptorを持てても実行できないままである。

#### BO: Gemma4 E2B resident BF16 text execution（2026-07-26）

上の BL 時点の「resident executor が別途必要」という状態を更新した。量子化 artifact を
挟まず、`model.safetensors` の全 2,011 BF16 tensor / 10,246,357,958 B payload を R9700
へ一度だけ upload する `Gemma4TextExecutor::load_resident` と
`ullm-gemma4-resident` を追加した。vision/audio tensor も allocation には含めるが、本段階の
forward は text decoder のみであり、serving 統合はしない。

実 config からの resident plan は R9700 reported 31.859 GiB に対し、config maximum の
131,072 token で weight 10,246,357,958 B、shared-aware K/V 1,624,817,664 B、temporary
1,170,432 B、合計 11,872,346,054 B (11.056984 GiB) だった。local source 12層は
window 512 の固定 F32 ring、full source 3層だけは最大 context まで確保する。layer 15--34
には重複 K/V allocation を置かず、HF と同じ local source 13 / full source 14 を参照する。
source file header も含む conservative total 11,872,610,014 B も fit する。算術的には残り
VRAM がさらに full-KV 約 1.95M token 分あるが、実 config の max position が 131,072 なので
support する最大文脈は 131,072 token と記録する。

resident path は existing BF16×F32 matvec と既存 paged F32 K/V write/decode-attention を
使い、host fallback を environment gate で拒否する。新 kernel は追加していない。そのため
BH が編集中の `runtime/src/ullm_runtime_parts/part_01.inc` と
`runtime/src/ullm_runtime_hiprtc_sources.inc`、BK が編集中の
`crates/ullm-engine/src/sq8_serving_runtime.rs`、既存 `AQ4_0` / `SQ8_0` production path に
触れていない。prefill API は M=N、decode API は M=1 だが、既存 primitive は matvec のため
prefill projection は因果 token 順に M=1 launch を N 回発行する。これは semantic/KV
transition の fallback ではなく、batch GEMM 未実装という明示した性能上の残項である。

resident validation は BL の4-step greedy trace を正確に再現した。`The capital of France is`
は `9079,236761,108,818` (`Paris.\n\nThe`)、`Once upon a time,` は
`528,496,1902,1298` (`in a world where`) であり、cached M=1 decode と毎 step token 0 からの
full re-prefill が双方で一致した。window 512 を越える 513 token boundary では両 route が
top-1 `184` / logit `14.404961585998535` となり、12 local cache は各々
`capacity/cache_len/absolute_len=512/512/513` を示した。20 shared layer の snapshot は
13/14 mapping を全て示し、対照として shared layer の physical K/V を誤って再投影する
diagnostic は別 token 列 `506,236789,500,236772` となった。

R9700 only、service 非稼働、cooldown 後の wall-clock measurement は six-token prefill
18.296336 tok/s、four-token decode 15.613216 tok/s だった。load/warmup/profiler range は
除外し、logical BF16 weight + F32 K/V read/write 下限も artifact に保存した。llama.cpp
ROCm commit `68a5592` は BF16 GGUF を `gemma4 E2B BF16` と認識し、F32 K/V、flash attention
off、同じ 6/4 token/3 repeat 条件で prefill 218.955938 tok/s、decode 69.959983 tok/s を
示した。GGUF は text-only export なので complete source checkpoint との VRAM footprint
比較には用いない。温度、clock、aggregate throttle state と live VRAM を各 measurement
artifact に保存した。aggregate throttle state の原因別 field は `N/A` のため原因は未確認とする。

証跡は `benchmarks/results/2026-07-26/gemma4-e2b-resident-v0.1/` にある。ここでは軽量昇格
policy の文章品質原則を参照したが、serving は task scope 外なので promotion / manifest / service
操作は行っていない。FP32 reference corpus、bitwise gate、campaign も使っていない。

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

追補: decode を prefill の可変 group kernel に流用しないため、`moe_decode_gemm` の独立 ABI と
gfx1201 kernel を追加した。synthetic `M=1` decode と `M=5` prefill はそれぞれ CPU reference と
CPU ABI で全段 0 差、R9700 では final output が各々最大 `2.384185791e-7` 差だった。さらに実
layer-0 top-8 `[52,148,101,178,151,128,116,166]` の BF16 `[8,37,71]` slice は HF F32 expected/
CPU/R9700 decode GEMM で 0 差である。実 weight slab の物理 gather と prefill
histogram/prefix-sum compaction は residency/最適化段階として未実装のまま明示している。

## BN: Qwen3.5-35B-A3B AQ4_0 text package (2026-07-26)

BI の raw-residency audit を safetensors header から追認した。R9700 は 31.859375 GiB に
対し、text decoder は 63.613162 GiB、routed + shared experts は 60.234528 GiB、complete
checkpoint は 66.965497 GiB であり、expert 圧縮なしの architecture support は成立しない。
対象の40 text layer はすべて MoE で、routed tensors は rank-3
`[256,1024,2048]` gate/up と `[256,2048,512]` down である。dense-layer fallback は検出されず、
rank-3 を単純な dense plan に落とす実装は採らなかった。

新規の text-only candidate package を
`/home/homelab1/datapool/ullm/product/qwen35-35b-a3b-aq4_0-g8-moe-v0.2/` に製造した。
既存 `AQ4_0` の higher-fidelity `aq4_e4m3_g8_ts_flloyd16` を使い、80 routed expert
tensors のみ量子化し、router/shared expert/attention/embedding/norm/`lm_head` は raw
passthrough とした。per-expert codebook は held-out test で quality benefit がなく不安定な
tail を持ったため、routed down と gate/up に各一つ、全 40×256 expert で共有する global
codebook を採用した。`SQ8_0` は experts だけで約30 GiBを消費して KV と非expertを残せず、既存
pipeline もこの BF16 rank-3 source contract を扱わないため採らない。

streaming/resume conversion は tensor ごとの staging/再読込検証を行い、完成 package を全量
再検証した。80 tensor の relative MSE は `0.003603673..0.004363885`、max-absolute outlier は
layer 39 `down_proj` の `0.043730080` である。router は全40 tensor SHA一致、1,280 条件付き
router input で top-8 0変化だった。batch 1 の artifact/KV/workspace byte ledger は 262,144
token でも `30,858,010,436 B`、headroom `3,350,732,988 B` と算出する。ただし loader が未結線
なのでこれは R9700 実 allocation ではなく、empirical residency は未確認である。

重要な quality boundary も確認した。CPU で one-layer-at-a-time に全40層・8 token を実行する
source-vs-source control は 320/320 の selected expert set と final hidden state が完全一致した。
同じ入力の `AQ4_0` candidate は router weight が raw でも upstream expert error により
selected set を 105/320、ordered top-k を 238/320 変えた。従って package metadata は
`not_passed` であり、architecture descriptor/runtime ABI の正しさを量子化 serving 品質に
読み替えない。軽量昇格 policy に基づく promotion、service 操作、FP32 corpus/campaign/bitwise
gate は行っていない。次段には、top-k stability requirement を満たす容量内の既存 format policy
又は requirement 自体の再判断と、MoE loader/residency integration が必要である。

## BP: `SQ8_0` resident runtime の descriptor 境界（2026-07-26）

### Phase 1: Qwen3-14B 固定契約の棚卸し

BF が数えた `qwen3_loader.rs` の 15 契約とは別に、実際に resident `SQ8_0` を
load/execute/serve する経路を読んだ。ここでは単なる定数参照ではなく、別の
architecture を通すと誤った重み・workspace・KV state・出力契約を選ぶ **15 個の
独立した固定契約**を数える。

| # | file / function | 現在固定されているもの | 一般化時に必要な descriptor 情報 |
| ---: | --- | --- | --- |
| 1 | `sq8_layer_runtime.rs::Qwen3Sq8LayerConfig::{qwen3_14b,validate}` | hidden=5120、Q/KV=40/8、head/value dim=128、I=17408、eps=1e-6、theta=1,000,000、Qwen3 の position/sequence 制限。 | decoder shape、層ごとの attention geometry、norm convention、RoPE と context contract。 |
| 2 | `load_qwen3_14b_sq8_layer_weights` と `qwen3_sq8_layer_tensor_names` | Q/K/V/O + gate/up/down の seven-projection、固定 tensor namespace と 2-D shape。 | layer ごとの projection set、tensor name mapping、dense/MoE、physical K/V の有無。Gemma shared layer は K/V を要求してはならない。 |
| 3 | `Qwen3Sq8LayerWorkspace::{allocate,validate_synchronized_preconditions}` | hidden/KV/intermediate activation と 4 個の activation quantizer が全層同じ幅。 | per-layer workspace plan。mixed local/full width、double-wide MLP、PLE scratch、linear state を別に会計する。 |
| 4 | `Qwen3Sq8LayerWorkspace::enqueue_with_attention` | input/post residual norm、Q/K norm、SiLU MLP、`1/sqrt(128)` full causal attention、single RoPE の順序。 | 明示的な architecture/layer composer。residual norm の位置と数、Q/K/V norm、activation、attention scale、output gate、RoPE kind を表す。 |
| 5 | `enqueue_{prefill_with_paged_kv,paged_decode,cached_prefix_chunk_with_paged_kv}` と `validate_paged_cache_contract` | 全 layer が独立した full causal paged K/V、identity block table、40Q/8KV/128 の同一 cache shape。 | `Own` / `SharedFrom` / `LinearState` の state mode、shared source layer、sliding window と cache retention policy。 |
| 6 | `Sq8LayerRuntimeTrace` と layer execution report | trace tensor の hidden/KV/I width と seven projection/4 quantization の counter。 | layer-dependent tensor shape と operation list。比較 writer は dynamic layer count/shape を記録する。 |
| 7 | `sq8_stack_runtime.rs::Qwen3Sq8StackRuntime::load` | `QWEN3_14B_SQ8_STACK_LAYERS=40` の boxed array、40 個の norm、同一 artifact layout。 | descriptor の layer vector を dispatch 前に検証し、architecture 固有 backend を選ぶ。既存 `SQ8_0` は legacy backend として明示保持する。 |
| 8 | `Qwen3Sq8PagedDecodeRuntime` と `Sq8PagedStackExecutionReport` | M=1 decode workspace、40 cache lengths、全層一様な K/V write / attention count。 | cache owner 層だけを数える state-aware report。shared/local/linear 層には別の counter/shape が必要。 |
| 9 | `sq8_embedding_runtime.rs::Qwen3Sq8EmbeddingRuntime::load` | `model.embed_tokens.weight`、151936×5120、独立 embedding、scale なし。 | embedding tensor mapping、vocab/hidden、tied flag、embedding scale。Gemma4 は tied source weight と `sqrt(hidden)` scale を使う。 |
| 10 | `sq8_model_head_runtime.rs::Qwen3Sq8ModelHeadRuntime::{load,run_*}` | `model.norm.weight` と独立 `lm_head.weight`、151936×5120、fixed final norm、logit cap なし。 | output tie、final norm convention、vocab/hidden、logit soft-cap。Gemma4 は embedding から投影し最後に cap=30 を掛ける。 |
| 11 | `sq8_generation_runtime.rs::Qwen3Sq8GenerationRuntime::{load,generation_cache_shape}` | 40-element cache array、8-token fixed prompt/16-token context、Qwen3 vocab/EOS、full attention only。 | architecture-specific request plan、tokenizer/profile、dynamic layer cache set、context and EOS contract。 |
| 12 | `sq8_serving_runtime.rs::Qwen3Sq8ServingSession::{load_with_prefill_mode,qwen3_14b_sq8_serving_cache_shape,load_qwen3_14b_sq8_serving_norms}` | 40 layers、4096 context、16-token blocks、fixed artifact/package SHA、Qwen3 norms/cache/sampler。 | served architecture profile と descriptor-bound package/artifact identity。multi-architecture serving はこの profile dispatch を別 task で持つ。 |
| 13 | `sq8_worker_backend.rs` と `sq8_worker_protocol.rs::Sq8WorkerProfile` | `Qwen3Sq8*` backend、R9700 kernel guard、Qwen3 vocab/EOS/reasoning default、artifact/package identity。 | backend selection と per-model tokenizer/reasoning profile。worker protocol を Qwen3 default から黙って流用しない。 |
| 14 | `sq_canonical.rs::{validate_manifest,validate_source_contract}` | `SQ8_0`、FP8 E4M3 dynamic source、BF16 128×128 2-D block scale、`fp8_checkpoint` import。 | artifact format id と source model/config SHA、tensor rank/layout、per-layer quantization metadata、tied aliases、MoE expert axis を schema に追加する。 |
| 15 | `tools/sq8_canonical_artifact.py::{load_source_contract,pair_fp8_weights}` と SQ8 build tools | Qwen3 FP8 shard、2-D paired weight/scale、fixed block assumptionsを conversion 時にも強制。 | descriptor-aware source importer。BF16 Gemma4、rank-3 MoE、shared/tied tensorsを別 format path で明示対応する。 |

したがって、ここで「全固定値を一つの汎用 kernel parameter struct にする」設計は採らない。
`SQ8_0` の seven-projection kernel/artifact は Qwen3-14B の production contract として残し、
descriptor は *どの resident composition が選べるか* と *その composition に必要な state* を
選ぶ境界にする。新しい kernel が必要なら BH が編集中の
`runtime/src/ullm_runtime_parts/part_01.inc` と
`runtime/src/ullm_runtime_hiprtc_sources.inc` には追加せず、新規 source に分ける判断を維持する。
この BP 実装では新 kernel を必要としなかった。

### Phase 2: 実装した境界と優先順位

`model_config.rs` に `ResidentModelDescriptor` を追加した。これは config SHA-256、decoder、
embedding/output、layer vector を持つ closed typed descriptor である。layer は attention
kind/heads/head dim/value dim/scale/RoPE/window/KV mode/norm、dense または MoE MLP、PLE を
明示する。`SharedFrom { source_layer_index }` は Gemma4 E2B の layer 15 以降を local source
13 / full source 14 に結ぶ。Qwen3.5 は hybrid linear/full、mRoPE、MoE expert/top-k/shared
expert metadata まで表すが、未実装の executor を実行可能とは表示しない。

- Qwen3-14B `SQ8_0` は `require_qwen3_14b_sq8_0` を通る exact legacy backend にした。
  serving/generation/architecture trace の入口と stack load は descriptor と artifact source
  config SHA の一致を device allocation 前に確認する。既存の projection/KV/kernel math は
  変更しない。
- Gemma4 `Gemma4TextExecutor` は descriptor から checkpoint contract、resident memory plan、
  device K/V allocation、shared-source snapshot、input/PLE/layer/attention/MLP/head を選ぶ。
  local/full の width、window、RoPE、four residual norms、double-wide MLP、PLE、tied head/
  embedding scale/soft-cap の個別分岐を保持した。
- `SQ8_0` artifact schema 自体を Gemma4/MoE まで拡張した、という主張はしない。現行 schema
  は #14--15 の制約のままであり、MoE の実 quantized end-to-end は AQ4_0 package/residency
  integration 待ちである。

artifact の一般化境界も同じである。現行 `sq-fp8-artifact-v0.2` / `SQ8_0` は
`source.config_sha256` を既に持つが、Qwen3 の FP8 E4M3 2-D block weight/scale と
seven-projection import を前提にする。Gemma4 又は MoE を量子化 resident に載せる時は
この format id を緩めない。別の strict format に、少なくとも architecture、source config
SHA、layer descriptor binding、tensor role/alias（tied embedding と shared K/V を含む）、
rank/axis-aware quantization layout、MoE expert axis と router/shared-expert passthrough を
記録させる。これにより legacy `SQ8_0` loader が未知の tensor layout を解釈することはない。

最初に通す新 architecture は **Gemma4 E2B resident BF16** と決めた。raw source BF16 weight は
約 9.54 GiB（complete resident allocation は既存 evidence で約 11.06 GiB）で R9700 に載り、
既に greedy/KV/window の実行器がある。対して Qwen3.5-35B-A3B text BF16 は 63.613 GiB で
R9700 に載らず、量子化が前提であり、AQ4_0 package を進める BN の作業とも依存する。
Gemma4 を先に descriptor-connected execution で確認し、MoE は descriptor で state を表現して
も quantized residency/executor が揃うまで fail-closed にする順序が最小の曖昧さである。

### Phase 3: 検証方針

Qwen3-14B `SQ8_0` は config SHA
`c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793` と canonical artifact の
`source.config_sha256` が同じであることを read-only で確認した。descriptor unit test は legacy
5120/40/40Q/8KV/128/I=17408/151936/eps/RoPE/seven-projection contract を通過する。GPU の
greedy regression と Gemma4 resident greedy re-run は、共有 R9700 lock が空いてから
`tools/architecture_hf_trace.py` の既存 trace と BL/BO の token/text 証跡を使って追記する。
数値閾値、FP32 corpus、bitwise gate、campaign は合否に使用しない。

## CB: Gemma4 E2B BF16 served-model integration（2026-07-27）

### Phase 1: serving に不足していた閉包

BL/BO の `Gemma4TextExecutor` は raw BF16 source と resident K/V を正しく実行できたが、
served-model として起動するには次の境界が未接続だった。

| 領域 | 不足していたもの | 今回の閉じ方 |
| --- | --- | --- |
| format / worker | `SQ8_0` と `AQ4_0` の worker しかなく、Gemma が generic loader に暗黙に流れることを拒否する入口が無かった。 | 新しい **`BF16_0`** と `ullm-gemma4-worker` を追加。worker は `gemma4_e2b_bf16_rdna4_v1`、`gfx1201`、resident profile、greedy top-k=1、EOS=1、v1/no-reasoning を exact match で要求する。既存 SQ8/AQ4 の許可集合は変更していない。 |
| product / package | source directory を worker に直接渡す strict product/package closure が無かった。 | `ullm.gemma4_e2b_bf16_package.v1` が `Gemma4ForConditionalGeneration` / `gemma4` / `gemma4_text` / vocab 262144 と `config.json` / `model.safetensors` の byte SHA-256 を bind する。worker は package の hash と executor の config/tensor contract の双方を確認する。 |
| worker guard | BO の BF16 fallback rejection を serving manifest に表す contract が無かった。 | `ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL`、`ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL`、`ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL` の完全一致を必須にした。余分な guard、execution selector、artifact、別 architecture は拒否する。 |
| tokenizer | base `google/gemma-4-E2B` の local `tokenizer_config.json` は `GemmaTokenizer` / BOS=2 / EOS=1 / PAD=0 だが **`chat_template` を持たない**。従ってそのまま gateway に載せると `apply_chat_template` が失敗する。 | base tokenizer を上書きせず、Google `gemma-4-E2B-it` revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7` の `chat_template.jinja`（SHA-256 `0a2c8073…c5b5`）を明示入力にした overlay を作る。base tokenizer JSON/config の SHA と template source/revision を `provenance.json` に bind し、native template が既にある source への上書きは拒否する。 |
| manifest / gateway | v2 は reasoning dialect を必須にするが、Gemma base E2B の reasoning token protocol は確認していない。 | `ullm.served_model.v1` / `ullm.worker.v1` を選んだ。BQ の v2 execution selector は使わず、環境変数での selector 混入を既存 gateway/worker loader が拒否する。Python gateway の tokenizer contract は class/hash/template を generic に検証するため、`GemmaTokenizer` を新たな profile shortcut に偽装しない。 |

`tools/generate-served-model.py` には、任意 BF16 を許可せず
`BF16_0 × ullm.worker.v1 × ullm.gemma4_e2b_serving_receipt.v1` だけを明示的に受理する
dispatch を追加した。receipt は source commit、worker binary、package manifest、template hash を
相互に bind する。

### 配置と host-side 検証

candidate は active manifest と独立して次へ配置した。

- product: `/opt/ullm/gemma4-e2b-serving-v0.1/products/gemma4-e2b-bf16-e5faef0d`
- worker: `/opt/ullm/gemma4-e2b-serving-v0.1/releases/gemma4-e2b-bf16/ullm-gemma4-worker`
- immutable manifest: `/opt/ullm/gemma4-e2b-serving-v0.1/manifests/gemma4-e2b-bf16.manifest.json`
  (`e01fa275a8e682c44606df2f1549cb0676df04d7b55b29e7f238ec7ec43fc8c9`)

全 product / worker / manifest / receipt は `root:root`、ディレクトリ 0555、内容 0444
（worker は 0555）にした。`tools/validate-served-model.py` はこの snapshot を受理し、
Transformers 5.12.1 は overlay を `GemmaTokenizer` として読み、基本 user prompt を
`<bos><|turn>user ... <|turn>model` へ render できた。なお IT tokenizer JSON 自体の SHA は
base E2B と異なるので、template と base vocabulary が同一であるとは仮定せず、実際の template
render と gateway generation をこの後の runtime evidence で確認する。

runtime / gateway / raw greedy evidence は
`benchmarks/results/2026-07-27/gemma4-serving/` に追記する。候補検証では
`/etc/ullm/served-models/active.json` を一切切り替えず、R9700 lock が空いた時だけ隔離 port の
gateway を起動する。

### Phase 3: isolated worker / gateway evidence

candidate は `127.0.0.1:18080` の手動 gateway として起動し、active manifest を Gemma4
へ切り替えなかった。起動直後の固定 sleep で成功と見なさず、最初の probe を 3.25 s 後にして
bounded exponential backoff を行った。`/readyz` は 1 回目、実測 3.276 s で `200
{"status":"ready"}`、`/v1/models` は candidate model ID を返した。英日 2 request と
lightweight policy の 10-case suite は全件 HTTP 200・nonempty completion まで到達した。生の
request/response は `gateway-*.json` に保存している。

worker wire protocol は一要求ずつの single-active contract なので、raw comparison も逐次に
行った。Gemma4 resident worker の token は、BL/BO の read-only trace と以下の通り完全一致した。

| input label | generated token IDs | BL/BO との比較 |
| --- | --- | --- |
| `gemma-capital-france` | `[9079, 236761, 108, 818]` | exact |
| `gemma-once-upon` | `[528, 496, 1902, 1298]` | exact |

各 request は `released.outcome=length` と `reset_complete=true` で終了した。これは単に raw
executor が動くことではなく、manifest-bound `BF16_0` worker、wire protocol、resident reset が
BL/BO の greedy path を変えていないことの確認である。

### Chat quality decision

**この E2B candidate は active promotion の対象にしない。** service/gateway transport は動作したが、
`docs/plans/lightweight-promotion-policy-v0.1.md` の「実際の文章」基準では明白な崩壊を確認した。

- `ja_explanation` は user prompt を繰り返した。
- `ja_multiturn` は `1.` の反復ループになった。
- `en_multiturn` は同一文を繰り返した。
- `ja_long_summary` には `<unused56>` が現れ、translation と structured-reasoning は空になった。

これは数値しきい値ではなく、保存済み応答を読んだ品質判定である。一方、France prompt は
`The capital of France is Paris.` を含む応答を返しており、HTTP/wire/tokenizer path の失敗とは
区別する必要がある。

tokenizer contract 自体は mechanical に通った。overlay は Transformers 5.12.1 で
`GemmaTokenizer` として読み込まれ、BOS/EOS/PAD (`2/1/0`)、template hash、rendered
`<|turn>user ... <|turn>model`、vocabulary range を検証し、gateway の実 request にも使用された。
しかし base `google/gemma-4-E2B` には upstream chat template がなく、E2B-it の template を
明示 provenance 付き overlay として試しただけである。上記の文章品質により、**base E2B に対して
この chat template が semantic に正しいとは確認できなかった**。次に必要なのは、base checkpoint
用に upstream が根拠を与える chat interface、または instruction-tuned E2B checkpoint を対象にした
別の source/package/manifest であり、推測による template の置換は行わない。

candidate の immutable manifest (`e01fa275…c8c9`) と product/worker は保存したまま、AQ4_0
active manifest は `3507102…b3e7` へ戻す運用を別証跡に記録する。速度改善や Gemma4 の量子化は
この task の対象外である。

### AQ4_0 restoration boundary

isolated run の間に別セッションが active manifest を `d3d9c454…6038c2` へ切り替えたことを
観測した。この task はその manifest を採用せず、trusted protected source
`/opt/ullm/aq4-gqa-grouped-deployment-v0.1/manifests/aq4-gqa-grouped-protected-c8074928-7e34eed1.json`
を SHA-256 検証後に root-owned temporary file 経由で atomic rename した。最終 snapshot は
`3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7`、`root:root 0644`、
`ullm-openai.service` は `active/running` である。`aq4-final-restore-stability.txt` は 4, 5,
8, 12 s の全観測で同じ SHA と `is-active=active` を記録する。

service の start-limit は共有操作による lock-conflict で二度 fail 状態になった。明示的な
`Start request repeated too quickly` を確認した時だけ `reset-failed` と一回の `start` を使い、
Gemma candidate を service に向けた start/restart は一度も行っていない。
