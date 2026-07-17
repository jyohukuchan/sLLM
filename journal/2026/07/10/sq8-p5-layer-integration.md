# SQ8 P5-C/D one-layer integration

日付: 2026-07-10

## 前回の要点

P5-Aで全4 projection classの7つのMを測定し、P5-Bでgfx1201 CKをproduction C ABIとRust APIへ統合した。残っていた受入条件は、Qwen3-14B decoder layer一層の中間・最終tensor比較とreferenceより速いoptimized layerの証明だった。

## 今回の変更点

- `b047fdd Add SQ8 decoder layer runtime and oracle`
  - `hidden → input RMSNorm → QKV → q/k headwise RMSNorm → full RoPE → causal 40:8 GQA → o → residual → post RMSNorm → gate/up → SiLU → down → residual`を実装した。
  - QKVとgate/upはそれぞれ量子化を一回だけ共有する。oとdownを含め、一層は4 quantization・7 projectionである。
  - 独立CPU oracleはF64 accumulation、pure CPU RMSNorm/RoPE/attention/SiLU/residual、各7 projection直後のBF16 RNE→F32契約を実行する。
- `a8446d9 Add auditable SQ8 layer validation`
  - 実artifact/thin package、reference/optimized latency、4量子化のbit一致、17中間・最終tensorを一つの型付きJSONで検査するexampleを追加した。
  - past KVがない独立runnerでは`position_offset>0`を拒否する。公開層APIはstream完了後にだけ返り、異なるstreamとの競合を起こさない。
  - RMSNorm、RoPE、causal attention、add、SiLUは5つの`ULLM_REQUIRE_HIP_*`を必須にし、host stagingのfallbackを失敗にする。
- `e3bd34e Add SQ8 layer result validator`で、hash、tensor/activation gate、timing再計算、dispatch、guardを独立に再検査する。
- `63bf38a Record SQ8 one-layer evidence`で正式結果を保存した。

## 実測結果

- R9700/gfx1201、layer 0、M=8、position offset 0。
- optimized p50 `0.777319 ms`、reference W8A16 p50 `16.530962 ms`、speedup `21.266652x`。
- 最終出力: relative L2 `0.003996148`、cosine `0.999992019301`、非有限値`0`。
- 17 tensor checkは全合格。最大relative L2はdown projectionの`0.011561786`、そのcosineは`0.999933386518`。
- 4つのGPU activationは、GPU直前F32をCPUで再量子化したbyteとscale bitに全一致した。
- M=8一層でimplementation ID 1、2、4を実行した。ID 3はgate M=128実fixtureをproduction ABIに通し、relative L2 `0.001658324`、cosine `0.999998624980`で合格した。
- validator: tensor 17件、activation 4件、dispatch 4件を再検査して合格した。

P5 acceptanceはgreenとする。

## 次の行動

P6では、P5の同期一層runnerをそのまま40回呼ぶのではなく、一つのstreamとresident bufferを所有するfull-stack runnerを作る。まずoffset 0のprefillを実トークン列で全40層へ展開し、次にpast KV付きchunk/decode、lm_head、source-correct logitsへ進む。
