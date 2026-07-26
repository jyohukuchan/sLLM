# llama.cpp R9700 decode attention の KV split 調査

## 前回の要点

SQ8_0 の R9700 direct paged decode は、40 Q head に一つずつの 256-thread
workgroup を置くため、attention layer dispatch あたり 40 WG / 320 wave32、
64 CU × 32 waves/CU に対する supply envelope は 15.625% だった。multi-tile
source split は wave supply を戻して性能を上げたが、partial online-softmax merge
の association 差が逐次 SQ8 activation quantization で feedback 発散し、frozen
bitwise full-model gate は NO-GO だった。

## 調査結果

- llama.cpp `68a5592` の gfx1201 build で、Qwen3-14B Q8_0 / F16 KV、depth 1028
  の後に16 token生成を rocprofv3 で採取した。decode の active body は
  `flash_attn_ext_vec<128,1,...>` であり、WMMA body ではなかった。
- 各 token は40 layerで、layerごとに vector main と
  `flash_attn_combine_results<128>` を一回ずつ実行する。16 step 合計は main 640、
  combine 640 launch、すなわち **80 attention launch/token** だった。
- main の raw grid/block は `(32,40,40)/(32,4,1)`。したがって
  `P=40/4=10`、400 WG/layer dispatch、1,600 wave32/layer dispatch、supply proxy
  **78.125%** である。combine は40 WG / 160 wave32 / **7.8125%** である。
  llama.cpp attention は17,600 WG/token（main 16,000 + combine 1,600）、uLLM
  direct は1,600 WG/tokenなので、この局面の幾何学的な並列 work は11.0倍である。
- llama.cpp の main は `blockIdx.y` で KV partial を選び、128-token単位を
  `gridDim.y * 128` stride で走査する。今回の internally padded KV length=1280では
  各 P=10 partial が一つの連続128-token tileを担当する。partial は
  `(max, sum, weighted-V)` を作り、combine が global max で rescale/mergeする。
  よって、これは uLLM が数値上 reject した source-tile split と同型の
  flash-decoding / split-KV 方式である。
- 40 Q / 8 KV の GQA は llama.cpp が `head / 5`、uLLM が
  `q_head / q_per_kv` で処理する。llama.cpp の KV cache は layerごとの連続
  tensor、uLLM は `block_table` を使う paged K/V である。
- selected 16 decode interval の llama.cpp attention は 12.219677 ms、
  **0.763730 ms/token**、sum-of-dispatch-duration 全 kernel 442.300522 ms中の
  **2.762754%** だった。既存 uLLM phase1 direct は30.773224 ms/token、51.05%で
  ある。weight/cache format と独立 capture が異なるため、絶対時間を公正な
  end-to-end 比較とは扱わないが、uLLM attention が支配的な理由の構造証拠にはなる。
- wave supply は静的 launch geometry であり、実 residency / achieved occupancy /
  physical HBM bandwidth は **未確認**である。llama.cpp の86.5% engine logical
  GB/s 指標を attention physical bandwidth と同一視していない。

## 数値契約と方針

llama.cpp は P=10 merge を通常経路として実行するが、P を指定する公開 knob と
fixed input output comparator がない。Pを変えた llama.cpp の実際の出力差、その
許容基準は **未確認**である。source は direct と同じ reduction orderを保証せず、
partial max/sum/weighted-V の再結合を明示している。

従って現行 bitwise gate の下で llama.cpp 型 P>1 split/merge を uLLMへ導入する
ことはできない。数値契約を保って取り込める候補は、direct token-orderを変えない
page-address、load/coalescing、GQA reuse、launch overhead の改善だけであり、これらは
40→400 WG/layer の主効果を作らない。splitを再評価するなら、凍結済み v0.2
artifact-FP32相対 gate（JSON SHA-256
`64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`）で、matched
direct control と同等以上の全layer/hidden/logits/top-k/feedback品質を先に示す
明示的な非-bitwise candidate に限る。pass可能性は **未確認**である。

## 非干渉とサービス窓

- `ullm-openai.service` は15:04:17+09:00に一回だけ停止し、15:04:26に
  isolation check後 profile開始、15:04:37に exit=0後ただちに一回の startで復旧した。
  final は active/running/enabled、`NRestarts=0`。`StartLimitBurst=3` の窓で
  intentional stopは一回だけである。
- `llama-qwen35-udq4.service` は前後とも inactive/dead/disabled、`NRestarts=0`で、
  起動していない。profile subprocess は `HIP_VISIBLE_DEVICES=1` で only ROCm0
  gfx1201を列挙し、R9700のみで実行した。CPUは `nice -n19`、`ionice -c3`、`-t8`に
  制限した。
- `lsof` は Docker overlay を statできない warning を出したが、R9700 renderD129 /
  kfd holder行はなかった。よって同checkの可視範囲での isolation は確認済みだが、
  Docker namespace を含む絶対的無holderは **未確認**である。
- 準備中に visibility mask前の `llama-bench --help` が backend初期化で全adapterを
  列挙した。model load、GPU dispatch、profile、service操作はなく、V620 workloadは
  実行していない。以後の model/profile command はすべて R9700-only maskを通した。
- `/etc/ullm/served-models/active.json`、systemd unit、`/opt/ullm`、activation、
  campaign、uLLM production kernel、llama.cpp build成果物は変更していない。

raw profile、source read、selection script、CSV/JSON summary は
`benchmarks/results/2026-07-26/llamacpp-attention-analysis/` に保存した。

## 次の行動

1. 現行 bitwise contractのままでは direct-order-preserving な paged attention
   micro-optimizationだけを別候補として評価する。P>1 merge を既定候補にしない。
2. v0.2 FP32 reference corpus 完成後にのみ、split candidate を control と同一入力・
   feedback decodeで全指標比較する。性能測定は数値 pass の後に別R9700窓で行う。
3. exact-state merge を主張するなら、direct recurrenceと同一になる証明と
   full-model bitwise gateを先に用意する。llama.cpp の merge をその根拠にはしない。
