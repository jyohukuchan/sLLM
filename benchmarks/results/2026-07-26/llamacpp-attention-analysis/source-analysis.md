# llama.cpp decode attention のソース読解

対象は `/home/homelab1/llama.cpp-src` の commit
`68a5592c10666d4d89b8480b5b9e8f8068b2f64c` と、そこに既に存在した
`build-rdna4/` である。checkout は `tools/llama-bench/llama-bench.cpp` に
事前から未コミット変更があったため、読み取りだけを行い、再ビルドしなかった。
`CMAKE_HIP_ARCHITECTURES=gfx1201`、`GGML_HIP=ON`、
`GGML_CUDA_FA=ON`、`GGML_HIP_ROCWMMA_FATTN=ON` を cache から確認した。

## 実際に選ばれる decode body

`-fa on`、Q=1、head dimension=128 の条件では、WMMA を利用可能な build でも
chooser は vector FATTN を優先する。

| 根拠 | 内容 |
| --- | --- |
| `ggml/src/ggml-cuda/fattn.cu:452-503` | 小さい Q batch は vector kernel を選ぶ。WMMA 分岐内でも `Q->ne[1] <= 2` なら vector を返す。 |
| `fattn-vec.cuh:533-574` | Q=1 は `flash_attn_ext_vec<D=128, cols_per_block=1>` を launch helper へ渡す。 |
| 本プロファイル | `flash_attn_ext_vec<128,1,...>` を 640 回、`flash_attn_combine_results<128>` を 640 回観測した。 |

従って、ここでいう split は `stream_k=true` の別経路ではない。vector body が
`launch_fattn(..., stream_k=false)` を呼び、そこで `parallel_blocks` を用いる
非-stream-K の KV split である。

## KV split と GQA

vector main body は `fattn-vec.cuh:250-256` で
`blockIdx.y * nthreads` から開始し、`gridDim.y * nthreads` stride で K/V を
走査する。本 capture では `nthreads=128`、`parallel_blocks=P=10`、KV length は
内部 pad により 1280 なので、各 partial workgroup はちょうど一つの連続 128-token
領域を処理する。一般には同じ split が 128-token stride で複数領域を巡回する。

GQA は `fattn-vec.cuh:104-112` の `head / gqa_ratio` で扱う。Qwen3-14B の
40 Q heads / 8 KV heads では ratio=5 であり、Q head 0--4 が KV head 0、…、
35--39 が KV head 7 を読む。profile の main grid Z=40 は、この 40 Q head を
直接示す。

`fattn-common.cuh:1107-1180` は occupancy query と KV tile 数から
`parallel_blocks` を選び、tail efficiency を探索する。`P>1` なら workspace を
確保し、`fattn-common.cuh:1260-1267` で別 launch の combine を実行する。
この benchmark に P を直接指定する公開 CLI/environment knob は見つからなかった。

## partial の統合

main は partial ごとに online softmax の max、denominator、未正規化 weighted-V を
保持する（`fattn-vec.cuh:258-515`）。combine は `fattn-common.cuh:911-966` で
global max を取り、各 partial を `exp(partial_max-global_max)` で再重み付けし、
numerator/denominator を加算して最後に除算する。

これは uLLM の source-tile partial + merge と同じ種類の有限精度変換である。
direct の token 順 online state と同じ加算順を維持する実装ではない。

## KV cache layout

llama.cpp は layer ごとに `ggml_new_tensor_3d` で連続 K/V tensor を確保する
（`src/llama-kv-cache.cpp:245-256`）。FATTN が見る view は
`[head_dim=128, kv_heads=8, n_kv=1280, sequence=1]`
（同:1223-1289）であり、uLLM の `block_table` をたどる paged layout ではない。
graph は FATTN 前に view/permute を作る（`src/llama-graph.cpp:2092-2127`）。

重要な差として、F32 KV cache を指定しても FATTN 直前で K/V を F16 に cast する
（同:2113-2120）。したがって今回の F16 profile は 86.5% の llama.cpp F16 KV
baseline の経路を測ったものであり、llama.cpp の F32 cache baseline の attention
body も F16 K/V を読む。ただし F32 cache row には別途 cast の費用がある。

## uLLM direct との対照

| 観点 | llama.cpp | uLLM phase1 direct |
| --- | --- | --- |
| KV storage | layer ごとの連続 tensor | paged K/V + `block_table` |
| decode main grid | 40 Q heads × P=10 partials | 40 Q heads |
| workgroup | 128 threads (wave32 ×4) | 256 threads (wave32 ×8) |
| partial state | max/sum/weighted-V、別 combine | 一 workgroup 内の単一 online state |
| GQA | `head / 5` | `q_head / q_per_kv` |
| current default | P=10 split + merge | multi-tile split は direct fallback |

uLLM の launcher は `runtime/src/ullm_runtime_parts/part_01.inc:3749-3829` で
Q head ごとに 256-thread workgroup 一つを launch する。kernel は
`ullm_runtime_hiprtc_sources.inc:7733-7867` で GQA と paged physical timestep を
解決している。split API 自体は `part_01.inc:3831-3949` にあり、partial の grid は
`q_heads * split_count`、merge は `q_heads` である。

現行 `decoder.rs:2495-2546` は multi-tile merge の numerical feedback を理由に、
one-tile 以外を direct kernel へ戻す。この current-source 読解は phase1 raw trace
の source object と完全同一とは確認できない。phase1 の記録済み HIPRTC/launcher
SHA-256 は `ad050…032b57` / `dcc883…723927`、現在の whole-file SHA-256 は
`1600d2…798f3` / `daee4e…a0bbe` である。phase1 の実測 geometry/timing は raw
trace に依拠し、current source は構造説明にのみ用いた。

## 数値差の確認範囲

P を変える public knob がなく、`llama-bench` は logits/tensor A/B comparator でも
ない。そのため「P を変えると llama.cpp の特定 prompt の出力が何だけ変わるか」は
**未確認**である。source 上は direct と bitwise 同一な reduction order を保証せず、
llama.cpp は P=10 の arithmetic を通常経路として実行している、までは確認できる。
それを「分割数を変えても数値差を許容するという実測済み品質方針」と読むことはできない。
