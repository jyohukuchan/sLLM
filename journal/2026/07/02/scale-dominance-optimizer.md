# scale dominance optimizer

## 前回の要点

- 直前の診断では、同じ36 tensorでE4M3がUE5M3よりわずかに低いBF16-errorを出した。
- ただし、E4M3もUE5M3も現行実装ではunsigned local-scaleであり、E4M3だけがlocal-scaleの符号を使ったわけではない。
- UE5M3はUE4M3/E4M3のscale値を数学的に含むため、理論上は同じblock-size、同じcodebook-index bit数、同じcodebook制約ならUE5M3がUE4M3より悪くなるべきではない。

## 今回の変更点

原因を3つに分けて修正した。

1. scale形式ごとに別sampleを使っていた。
   - `run-aq-codebook-opt-experiment.py`ではcandidate indexをseedへ混ぜていたため、同じtensorでもE4M3とUE5M3でsampled blockが一致しなかった。
   - 同じblock-sizeでは同じsampleを使うようにした。

2. 上位形式が下位形式の解を引き継いでいなかった。
   - 以前はUE5M3を独立した初期codebook/global-scaleから最適化していたため、局所解の差でUE4M3を下回れないことがあった。
   - `monotonic floor`を追加し、UE5M3評価時にUE4M3のlocal-scale indexをUE5M3 scale tableへ写像してbest-so-farへ入れるようにした。

3. 任意のUEaMb候補を固定リスト外から指定できなかった。
   - `tools/aq_scale_formats.py`を追加し、`ue4m3`、`ue5m3`、`ue5m4`などのunsigned E/M scale tableを共通生成できるようにした。
   - `aq4_ue4m3_g16_ts_flloyd16`のようなcandidate IDを動的に解釈できるようにした。

4. first-pass samplerにも同じ比較保証がなかった。
   - `run-aq-tensor-sample.py`も同じblock-sizeではsampleを共有し、上位UEaMb候補が下位UEaMb候補の解をmonotonic floorとして受け取るようにした。
   - JSONLには内部stateを出さず、`lifted_floor_count`と比較前提だけを記録する。

## 検証

scale tableの入れ子性:

- `UE5M3`は`UE4M3`/`E4M3`を完全に含む。
- `UE5M4`は`UE5M3`を完全に含む。
- `UE6M4`は`UE5M3`を完全に含む。

small smoke:

- 出力: `benchmarks/results/2026-07-02/aq/2026-07-02-aq-scale-dominance-smoke-ue4m3-ue5m3-flloyd16.json`
- 条件: Qwen3.5-9B、2 tensor、65536要素/tensor、4 iteration、aq4 g16 free Lloyd
- 違反: `0`
- UE5M3/UE4M3 ratio:
  - `lm_head.weight`: `0.991180`
  - `model.language_model.embed_tokens.weight`: `0.993951`

36 tensor check:

- 出力: `benchmarks/results/2026-07-02/aq/2026-07-02-aq-scale-dominance-ue4m3-ue5m3-flloyd16-36t.json`
- 条件: Qwen3.5-9B、36 tensor、262144要素/tensor、8 iteration、aq4 g16 free Lloyd、FP16 global-scale
- 違反: `0/36`
- UE5M3/UE4M3 relative-MSE ratio:
  - min: `0.993651`
  - mean: `0.995209`
  - max: `0.996761`
- UE5M3は全36件でUE4M3のfloorを利用し、その後のfloor init最適化が選ばれた。

UEaMb chain smoke:

- 出力: `benchmarks/results/2026-07-02/aq/2026-07-02-aq-scale-dominance-chain-smoke-ue4m2-ue5m4.json`
- 条件: Qwen3.5-9B、4 tensor、65536要素/tensor、4 iteration、aq4 g16 free Lloyd
- 候補: `UE4M2 -> UE4M3 -> UE5M3 -> UE5M4`
- dominance pair: `24`
- 違反: `0`
- 例: `lm_head.weight`
  - UE4M2: `0.005795105`
  - UE4M3: `0.005069191`
  - UE5M3: `0.005040730`
  - UE5M4: `0.004756701`

first-pass sampler smoke:

- 条件: Qwen3.5-9B、1 tensor、4096要素、aq4 g16 free Lloyd
- `UE4M3`: `0.005119672`
- `UE5M3`: `0.005119672`
- UE5M3/UE4M3 ratio: `1.0`
- UE5M3はUE4M3のfloorを再現した。

## 解釈

前回のUE5M3<E4M3は、scale形式の表現力そのものではなく、比較手順とoptimizerの問題だった。

- 別sampleを比較していたため、0.4%程度の差は測定ノイズとして混ざり得た。
- UE5M3はUE4M3のscale集合を含んでいても、独立最適化ではUE4M3の解へ必ず到達するとは限らない。
- `scale-window`探索は局所探索なので、上位形式の探索空間が広いほど、下位形式の良い解を自動で再発見する保証はない。

`monotonic floor`により、同じblock-size・同じcodebook条件のunsigned E/M scale形式では、上位形式が下位形式の解を再現できる限り、上位形式の結果が下位形式より悪くならない。

## 次の行動

- 同じ仕組みをweighted scale/codebook実験にも適用する。
- UE5M4、UE6M4など、mantissa bitを増やした形式でも36 tensor checkを追加する。
- Rust/C++側の本変換器にも、候補比較時のmonotonic floorとUEaMb scale tableの入れ子性テストを移植する。
