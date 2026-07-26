# SQ8 gfx1201 GPU F32 control の CPU 突合

Date: 2026-07-26

## 前回の要点

- canonical Qwen3-14B `SQ8_0` artifact の CPU strict-F32 full-model reference は、raw-p0001 の
  9 position で決定的に capture 済みだったが、1 token 8.742 秒で full coverage は数日規模だった。
- GPU control は CPU reference の raw artifact/package admission だけを共有し、数値 decode と
  F32 forward は独立に実装して CPU 真値と突合する方針だった。

## 今回の変更点

- R9700/gfx1201 専用の standalone F32 control を追加した。raw OCP E4M3FN + raw BF16 scale を
  GPU で F32 decode し、projection は standard hipBLAS SGEMM、KV cache は F32、attention は
  serial-head の三段 causal softmax とした。CK、WMMA、HIPRTC、production SQ8 dispatch、候補
  kernel は参照しない。
- device guard は `HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1`、single visible exact
  gfx1201 を fail-closed で要求する。実行 device は R9700 `0000:47:00.0` であり、V620/gfx1030、
  service、active manifest、activation、campaign に触れていない。
- CPU 9 position と logits/final hidden/全 40 layer hidden、計 378 tensor を比較した。測定前に
  固定した契約は token exact、nonfinite 0、max-abs `<=2e-5`、relative-L2 `<=1e-5`、cosine
  `>=0.999999` だった。
- token は全 position で exact、GPU の同一 session replay hash も exact、nonfinite は 0 件だった。
  しかし 9 position 全てが比較不合格となった。worst max-abs は position 0/layer 19 の
  `0.0048828125`、worst relative-L2 は position 2/layer 23 の `1.4888965980100416e-05`、
  minimum cosine は `0.9999999998906846` だった。
- layer 0 の max-abs は `1.192092896e-7`、relative-L2 は `2.963378064e-8` と小さいが、後段の
  residual で差が増幅する。CPU の K-order FMA と hipBLAS reduction order の差とは整合するが、
  CPU/GPU libm 差も残るため reduction order だけが原因であることは未確認である。SGEMV 置換の
  診断はより悪化したが、control の適格化根拠にはしていない。
- 結果を見て閾値を変更して通すことはしなかった。GPU control は v0.2 control として非適格であり、
  CPU 100 position の追加・GPU full coverage reference は開始していない。
- performance-only の p=0..99 診断では、p0 `0.614119774` s、p1--99 fit
  `0.374549014 + 0.000497507391 × position` s（R² `0.996947573`）だった。28,853 forward は
  context layout を含む compute-only model で `4.922--11.118` h、capture I/O は未測定である。

## 保存物と判定

- CPU/GPU comparison receipt の SHA-256 は
  `aea6640282e5ca78b16f7e145f77598ee4456142c5e46cd6298ead556db51d43`、GPU 9-position run receipt は
  `8bf3fd4996a9859a9f15a03195d9ef2303ef4eeed43767557fdccbff53ecf8e0`、p100 performance receipt は
  `45f1868e42e13f8be46d1a3cb5a44b5be4ce1aef0b2453bc116510f5645e42ea` である。
- `14b-gpu-fp32-control-SHA256SUMS` は 9 metadata receipt と各 payload hash chain を束縛する。
- これは失格した GPU diagnostic の保全であり、frozen v0.2 JSON や CPU strict-F32 reference の
  status を変更しない。v0.2 は `blocked_reference_or_capture` のままである。

## 次の行動

1. GPU を control にするには、測定結果から独立した新しい比較契約を review/freeze し、その契約で
   新しい CPU/GPU capture を実行する必要がある。既存 9-position 結果を事後的に適格化してはならない。
2. その前に、reduction order と CPU/GPU libm の寄与を component-level で分離する。現時点では
   原因の単独断定はしない。
3. GPU control が適格化されるまで、full coverage capture と候補 kernel の v0.2 評価には使わない。
