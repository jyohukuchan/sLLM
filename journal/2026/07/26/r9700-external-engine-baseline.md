# R9700 external-engine decode baseline

Date: 2026-07-26

## 前回の要点

- R9700 gfx1201 で Qwen3-14B-FP8 由来 SQ8_0 の handwritten decode
  Phase 0 は、F32 KV、cache 1028 -> 1044、16 M=1 step x 5 回で
  15.294955751 tok/s だった。
- 外部エンジンの通常ベンチマークは短い prompt / generation を測ることがあるため、
  context 約 1036 の steady decode として揃えずに比較してはいけなかった。

## 今回の変更点

- 公式 Qwen/Qwen3-14B-GGUF の Qwen3-14B-Q8_0.gguf を固定 revision
  530227a7d994db8eca5ab5ced2fb692b614357fd で取得した。SHA-256 は
  a0dfe649137410b7d82f06a209240508e218f32f5b6fd81b69d6932160cfcd9d、
  サイズは 15,698,533,728 B である。GGUF Hub metadata と FP8 source
  revision 9a283b4a5efbc09ce247e0ae5b02b744739e525a はともに
  base_model:Qwen/Qwen3-14B を宣言するため、base model は同一と確認した。
- llama.cpp 68a5592 の README と source を確認した。-p は prompt-only、
  -n は generation-only、-pg は combined prompt+generation、-r は repeat、
  -d は timer 前に KV depth を prefill/restore する。従って
  -p 0 -n 16 -d 1028 -r 5 を使い、depth 1028 に対する 16 回の synchronized
  M=1 decode だけを計測した。depth prefill、tokenization、sampling、model load は
  時間に含まれない。flash attention は on、全 layer は GPU 要求とした。
- llama-bench の argument parser は f32 cache type を受け取らなかったため、
  外部 llama.cpp checkout に f32 -> GGML_TYPE_F32 の最小 parser mapping を加えた。
  patch / rebuild evidence は result environment に残した。この repository、uLLM runtime、
  production artifact は変更していない。
- 5 回の総時間から求めた steady decode は次の通りである。

| row | KV | mean tok/s | median tok/s | sample variance (tok/s)^2 | uLLM 比 |
| --- | --- | ---: | ---: | ---: | ---: |
| uLLM SQ8_0 reference | F32 | 15.294956 | 15.308831 | 0.002970 | 1.000000x |
| llama.cpp Q8_0 | F32 | 30.468075 | 31.089355 | 1.264832 | 1.992034x |
| llama.cpp Q8_0 | F16 | 34.885347 | 35.053291 | 0.250263 | 2.280840x |
| vLLM FP8 SSE steady output | auto; resolved dtype 未確認 | 15.455471 | 15.443856 | 0.035319 | 1.010495x |

- vLLM 0.21.0+rocm722 は R9700-only container で起動し、同一 FP8 checkpoint、
  TP=1、context 1044、single sequence で動いた。warmup 1 回後の 5 request は全て
  server usage として 1028 prompt + 16 completion = 1044 total を返した。
  15.455471 tok/s は first content event 後の SSE interval を 75 gap 合算で測った
  client-visible 値であり、uLLM / llama.cpp の kernel/decode-loop timer と同一視しない。
  prefix caching は server default で enabled、auto KV の最終 dtype は log から未確認である。
- SGLang v0.5.15.post1-rocm720-mi30x は同じ FP8 checkpoint の load と 1,044 token
  BF16 KV allocation まで成功した。default AITER path が gfx1201 用に JIT rebuild
  した後、decode CUDA graph capture の sgl_kernel.elementwise.rotary_embedding で
  Segmentation fault、Rank 0 scheduler exit code -11 となった。標準構成の時間制限付き
  attempt としてここで停止し、AITER/graph fallback を強制しなかった。
- raw AMD SMI telemetry は llama F32 が 36--61 C / 4--3391 MHz / 8--288 W、F16 が
  39--61 C / 41--3460 MHz / 12--263 W、vLLM process envelope が
  40--59 C / 6--3404 MHz / 11--329 W だった。uLLM reference の sampled load
  73 C / 3298 MHz / 250 W より低温だが、外部側 envelope は load/warmup/idle を含み
  timed decode window と同期していない。開始 hotspot は F32 37 C、F16 40 C、vLLM 40 C
  であり、cool な状態だが完全に同値ではない。AMD SMI の THROTTLED 文字列は観測値としてのみ
  保存し、熱的 throttle の因果は結論していない。
- ullm-openai.service は 13:47:11 から 14:09:58 JST までの 1 回だけ stop した
  (22 min 47 s)。restore 後は active/enabled/NRestarts=0、llama-qwen35-udq4.service は
  inactive/disabled のままで起動していない。active manifest、systemd unit、/opt/ullm は
  変更していない。

## 保存状態と次の行動

- 生データ、5 回の sample、temperature/clock/power history、full command、image identity、
  vLLM success / SGLang failure の log は
  benchmarks/results/2026-07-26/r9700-external-engine-baseline/ に保存した。
  summary.json と JSONL normalized rows、README に結果と比較上の注意点を固定した。
- Q8_0 は int8 + FP16 scale/32 で約 8.5 bpp、SQ8_0 は OCP E4M3FN +
  [128,128] BF16 scale で約 8.0 bpp であり、これは品質比較ではなく memory-bound
  8-bit 級の speed positioning である。
- MI300X を借りる際は同じ cache 1028 -> 1044 / 16 M=1 / five-repeat contract を再使用し、
  vLLM は kernel-only timer も取れる場合に SSE result と別 row にする。SGLang は gfx1201
  failure を workaround 済みと扱わず、対象 hardware で標準 image を再試行するまで未稼働とする。
