# Phase 61: OCP MXFP8／MXFP6 weight-activation

## 2026-08-31: 数値形式とresident契約

- OCP MX v1.0に従い、MXFP8 E4M3 W8A8とMXFP6 E3M2 W6A6を追加した。どちらもK-axis block 32、
  1 byteのE8M0 shared scale、roundTiesToEven、有限値へのsaturationを使う。
- MXFP8 valueは1 byte／element、MXFP6 E3M2 valueはlittle-endian 6-bit streamの4 value／3 byteとした。
  OCPは物理配置を規定しないため、MXFP6 packingとvalue plane直後へscale planeを置くresident順はsLLM契約として固定した。
- E8M0 `0xff`はblock全体へNaNを伝播する。入力NaNを含むblockはこのscaleを生成し、Infはscale 1でelement最大有限値へ
  saturationする。E3M2のsubnormal、最大値、RNE tie、31／32／33境界をCPU oracleで検証した。

## 2026-08-31: GGUF／Qwen実行統合

- 現行GGUF registryにMXFP8／MXFP6の標準type IDがないため、新しいIDは作らずI8 carrierのvalue tensorとE8M0 scale tensorを
  versioned `sllm.tensor_recipe`で結合した。rank 2 `[N,K]`、Kの32整列、value／scaleの厳密byte数、1対1 bindingを検証する。
- reviewed Qwen3.5 denseのexact text-linear setを、uniformなMXFP8 W8A8またはMXFP6 W6A6 graphへlowerする。
  embedding、normalization、activation／output graph境界はBF16のままで、各matmulがactivationを同形式へ動的量子化する。
- CLI、server、embedding、resident upload、監査identityを同じ経路へ接続した。MX形式の混在、scale欠落、K非32倍、
  exact `gfx1030`／`gfx1201`以外はfallbackせず拒否する。

## 2026-08-31: public HIP runtimeと実GPU evidence

- public C ABI／Rust loweringへ専用op version、tensor encoding、decode／prefill kernel IDを追加した。1 dispatch目でBF16 activationを
  MXへ変換し、2 dispatch目でresident weightと積を取り、FP32 accumulate／BF16 RNE outputを生成する。
- exact V620 `gfx1030`とR9700 `gfx1201`のtarget別release code objectで、各形式のdecode `M=1,K=32,N=7`と
  prefill `M=3,K=64,N=5`を実行した。全8 caseが独立CPU oracleに相対誤差0.02以内で一致し、観測最大は
  `0.003403014`だった。kernel ID／symbol、dispatch count 2、HIP-only、fallback 0、cleanup 0も一致した。
- 形式・shapeごとの4 output SHA-256は両GPUで完全一致した。R9700は`HIP_VISIBLE_DEVICES=2`で単独可視化し、
  論理device 0へ接続した。
- `cargo check --workspace`、core 546 test中該当を含む全lib test、HIP 141 test、CLI 83 test、server 123 test
  （既知fixture 1件ignored）、native host CTest 5件をPASSした。

## Evidence境界

当初の完了結果はoperator、ABI、GGUF loader／graph loweringの実装とmodel-free correctnessを示した。後続の実モデル測定は
下記のexact `gfx1030`固定scopeだけを追加し、長時間安定性、gfx1201／gfx942 full-model、別software tupleへ一般化しない。

## 2026-08-31: Qwen3.5-4B実モデル品質・VRAM・速度follow-up

- 固定BF16 GGUF `sha256:c571c54e...c675`から、MXFP8 GGUF `sha256:f253d9f4...076f`とMXFP6 GGUF
  `sha256:d0ff2e1d...264e`を実変換した。exact V620 `gfx1030`、FP16 KV、同じ10 case datasetでBF16 residentを完全解放してから
  candidateを常駐させ、全実行をHIP-only、fallback 0、cleanup 0で完了した。
- prefill／teacher-forced decode合計20 rowに対し、MXFP8はtop-1 16/20=`0.80`、KLD mean／p99
  `0.0149036／0.0696269`、perplexity相対差`-0.00179966`だった。MXFP6は15/20=`0.75`、KLD
  `0.0371302／0.146398`、perplexity相対差`+0.0296547`だった。品質runnerのreport SHA-256はMXFP8
  `ca475551a3b7ccbb2b97f8d46c0dbf85acb557c936b27a231da15e3c710566f4`、MXFP6
  `8a42118a7f2b0e034b1f26815d0f484d0c8ee86548b3129c7ede728203509ac9`である。
- resident VRAMはBF16 `8,411,592,192` bytes、MXFP8 `4,954,035,712` bytes（`-41.10%`）、MXFP6
  `4,061,763,072` bytes（`-51.71%`）。17 input／4 output、1 warmup＋3 measuredの中央値は、BF16 prefill／decode
  `284.03／45.68 tok/s`、MXFP8 `48.10／20.17 tok/s`、MXFP6 `100.16／20.06 tok/s`だった。
- dynamic activationはblockごとの単一thread処理からwave32協調scale reduction／encodeへ変更し、MXFP8 short-prefillは
  row-8 v2、MXFP6はtiled-16 v3 providerまで試した。それでもfull-modelはBF16より遅い。現段階のMX形式はformat／correctness／
  memory foundationであり、速度または品質を根拠とするproduction defaultにはしない。
- benchmark report SHA-256はBF16 `0d4869b7a152d88e880a0081cc4f5e5b46d842ea05dc527a9ea113fad45e154e`、MXFP8
  `d00c91131255d201670cfa5b0226c6783ee16990563f98e8fd7a85353295ec2a`、MXFP6
  `dae76f6921247ce017b4f0c8711bd41152a01985b94222827c62beaf4b809774`である。

## 2026-08-31: completion auditでの両RDNA再確認

- 現在のwave32 providerをexact `gfx1030`／`gfx1201`のtarget別release code objectで再確認した。各targetでMXFP8／MXFP6の
  decode `M=1,K=32,N=7`、短prefill `M=3,K=64,N=5`、Qwen3.5-4B GDN shape由来の非整列prefill
  `M=17,K=2560,N=32`を実行し、全12 caseが独立CPU oracleに相対誤差0.02以内でPASSした。観測最大は
  `0.0038314175`で、kernel ID／symbol、dispatch count 2、HIP-only、fallback 0、cleanup 0も一致した。
- 形式・shapeごとの6 output SHA-256は両GPUで完全一致した。operator report SHA-256はgfx1030
  `91a04761b600fee47efcd18ccbd23cc09424be3def7c5d9241e42ed7fb233bea`、gfx1201
  `2d701aaf4e8b2e8b410a948514fc9ed64481479458c353995874b46c733dad10`である。gfx1201は`HIP_VISIBLE_DEVICES=2`で
  R9700だけを可視化し、論理device 0へ接続した。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase61-ocp-mxfp-weight-activation.md)
