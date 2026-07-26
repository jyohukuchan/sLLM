# BI: Qwen3.5-35B-A3B MoE runtime foundation v0.1

Date: 2026-07-26

## 前回の要点

- Qwen3.5-35B-A3B は `Qwen3_5MoeForConditionalGeneration` であり、既存 Qwen3
  dense executor に 3-D expert weights を渡して動かすことはできなかった。
- config 駆動 loader は architecture contract を読めるようになったが、MoE executor は
  routing / gather-scatter / grouped GEMM / shared expert が未実装として fail-closed だった。
- 35B raw BF16 の resident inference が R9700 に載るかは、実重みの byte audit を先に
  行う必要があった。

## 今回の変更点

- 実 config と safetensors header を直接確認した。text decoder は 40 層すべて MoE、
  `H=2048, E=256, K=8, I=512, shared I=512`。routed expert は BF16
  `[256,1024,2048]` gate/up と `[256,2048,512]` down、router は BF16 `[256,2048]`、
  shared gate/up/down と shared sigmoid gate は別 tensor だった。dense replacement layer
  は観測されなかった。
- loader 非依存の public MoE ABI と CPU reference を追加した。stage は route、gather、
  raw F32/BF16 grouped GEMM、gated-SiLU、weighted scatter、shared sigmoid gate であり、
  assignment-major layout を明示している。CPU reference は raw BF16 matrix bytes も
  safetensors と同じ形で読める。
- gfx1201/R9700 専用の correctness-first HIP kernels を追加した。decode は
  `moe_decode_gemm` ABI/専用 kernel、prefill は `moe_grouped_gemm` ABI/可変 group kernel
  として実装上も分離した。decode の物理的な selected-weight slab gather は residency layer
  の責務として残し、現段階では提供された 3-D buffer を selected ID で参照する。prefill の
  histogram/prefix-sum/compaction は後続の専門化対象である。
- HF `Qwen3_5MoeTopKRouter` を直接呼ぶ fixture generator を追加した。実 BF16 layer-0
  router の 3 token × top-8 は CPU reference / CPU C ABI / R9700 C ABI で ID と score が
  完全一致した。F32 dot のみでは一つの expert 順が入れ替わることを検出し、HF の BF16
  linear activation/logit/selected-score 境界を明示的に再現して修正した。
- 実 layer-0 `gate_up_proj` の source expert 52/148 から raw BF16 `[2,37,71]` slice を
  採取し、local assignment ID `[1,0,1]` の grouped GEMM を実行した。HF F32 expected、
  CPU reference、CPU C ABI、R9700 はすべて 0 差であり、3-D expert axis/row/column の
  layout を小さい実 weight slice で直接検証した。
- 実 decode の first-token top-8 source expert
  `[52,148,101,178,151,128,116,166]` から raw BF16 `[8,37,71]` slice も採取した。HF F32
  expected、CPU reference、CPU C ABI、R9700 decode GEMM はすべて 0 差だった。
- synthetic full MoE block は prefill (`M=5`) と decode (`M=1`) を別 ABI で通した。CPU C ABI
  は F32/raw BF16 とも全 stage 0 差、R9700 の各 path の final output は最大
  `2.384185791e-7` 差、prefill BF16 stage 全体最大は shared gate/up の
  `3.576278687e-7` だった。これは timing ではない。
- R9700 は 31.859 GiB。text decoder raw BF16 63.613 GiB（31.754 GiB 不足）、complete
  checkpoint 66.965 GiB（35.106 GiB 不足）であり、full resident inference は実施不能と
  判定した。量子化/offload を暗黙に導入して生成成功と見なすことはしていない。
- `tools/architecture_hf_trace.py` は変更せず self-test を実行し、意図的な layer-3破壊を
  `step-0000__layer-0003` に局在して reject した。full 35B HF trace は raw model だけで
  66.965 GiB 必要な一方、shared host の利用可能 RAM が約43 GiBだったため安全に起動して
  いない。MoE full-model candidate も未結線である。
- 運用逸脱: 一度だけ `cargo test -p ullm-runtime-sys --lib` を HIP 可視化指定なしで実行し、
  既存の `first_hip_*` test が ROCm の先頭 HIP device（列挙順から V620 の可能性が高い）を
  使用した。MoE kernel や timing ではないが V620 は使用禁止であり、この実行は不適切だった。
  以後の GPU correctness run はすべて `HIP_VISIBLE_DEVICES=1` /
  `ULLM_HIP_VISIBLE_DEVICES=1` を明示して R9700 のみを使った。

## 次の行動

- BF の config descriptor とこの ABI を結線する `Qwen35MoeExecutor` を作る。ただし
  hybrid linear/full attention、mRoPE、KV state、Q output gate は MoE substrate の外側
  なので別々に conformance trace を追加する。
- resident policy を明示して選ぶ。raw BF16 full resident は不可能なので、expert
  streaming/offload 又は明示的な quantized format のどちらかを設計・検証してから
  end-to-end generation に進む。
- prefill は expert histogram/prefix-sum で group-compaction する grouped GEMM、decode は
  8 expert slab gather を専用化し、正しさ fixture を維持したまま GPU performance window
  でのみ測定する。
- 72--120 h の full text-only runtime estimate は据え置く。今回の substrate は
  およそ16--28 hとして切り出せたが、remaining critical path は loader/attention integration、
  residency、prefill specialization、full architecture trace である。
