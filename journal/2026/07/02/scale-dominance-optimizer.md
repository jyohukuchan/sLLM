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
- u8 scale-indexを超えるUEaMb形式を、仕様上どのbppとして扱うか整理する。

## Rust本変換経路への追記

Rust側の`ullm-quant`にもUEaMb scale table生成を移植した。

- `aq4_ue4m3_g16_ts_flloyd16`のようなcandidate IDから`ue4m3`を解釈できるようにした。
- `ue4m2`、`ue4m3`、`ue5m3`、`ue5m4`、`ue6m4`のscale table生成をunit testで固定した。
- 入れ子性は`scale_table_contains_all`で検証する。
- OOM防止のため、Rust parserはexp bit/mantissa bitともに8bit以下へ制限した。

Rust prototype output smoke:

- codebook artifact: `benchmarks/results/2026-07-02/aq/2026-07-02-aq-scale-dominance-rust-codebooks-qwen35-9b-g16.json`
- tensor: `model.language_model.layers.3.self_attn.k_proj.weight`
- family: `attn_k`
- candidates: `UE4M3`, `UE5M3`
- UE4M3 relative MSE: `0.005399506075`
- UE5M3 relative MSE: `0.005399506075`
- 違反: `0`

このtensorではUE4M3でclampが発生していないため、rangeだけ広いUE5M3はUE4M3と同値になった。これは「UE5M3がUE4M3より悪くならない」ことの検証であり、strict改善が出ないこと自体は自然である。

Rust dry-run chain:

- 出力: `benchmarks/results/2026-07-02/aq/2026-07-02-ullm-quant-scale-dominance-chain-dry-run-attn-k.json`
- tensor: `model.language_model.layers.3.self_attn.k_proj.weight`
- family: `attn_k`
- candidates: `UE4M2 -> UE4M3 -> UE5M3 -> UE5M4`
- 違反: `0`
- relative MSE:
  - UE4M2: `0.006104047419`
  - UE4M3: `0.005399506075`
  - UE5M3: `0.005399506075`
  - UE5M4: `0.005150528234`

`UE5M4`はscale候補数が`495`で、現行prototype packageのu8 scale-index保存上限を超える。そのため、今回は書き出しではなくdry-run探索の検証対象として扱った。

## scale-windowに関する注意

`center +/- scale_window`だけを探索する場合、上位scale tableが下位scale tableを含んでいても、下位形式で選ばれたscaleが上位形式の探索窓外へ出る可能性がある。実際に合成データでは`scale_window=4`でUE5M3がUE4M3よりわずかに悪くなるblockが作れた。

対応:

- Rust CLIに`--scale-window all`/`--scale-window exhaustive`を追加した。
- C++ kernelのwindow終端計算をoverflowしない形に修正した。
- unit testで`scale_window=usize::MAX`時のUE4M2 -> UE4M3 -> UE5M3 -> UE5M4のrelative MSE単調非増加を固定した。
- C++ kernelが`scale_window=usize::MAX`を受けても正常に完了することをunit testで確認した。

結論として、理論的なUEaMb支配性を検証する場合は`--scale-window all`を使う。速度重視の通常変換で小さいwindowを使う場合、その結果は近似探索であり、数学的な支配性保証とは分けて扱う。

## Python補助ツールの追記

`export-aq-family-codebooks.py`と`verify-aq-one-tensor.py`も、固定候補リストだけでなく`candidate_from_id`経由の動的UEaMb candidateを解決できるようにした。

smoke:

- `export-aq-family-codebooks.py`で`aq4_ue4m3_g16_ts_flloyd16`の`attn_k` codebook exportが成功。
- `verify-aq-one-tensor.py`で`model.language_model.layers.3.self_attn.k_proj.weight`、`aq4_ue4m3_g16_ts_flloyd16`の検証が成功。
- Python verify relative MSE: `0.005399505821193474`

## weighted条件の追記

activation weightedのscale search + codebook条件でもUEaMb支配性を確認した。

- 出力: `benchmarks/results/2026-07-02/aq/2026-07-02-aq-weighted-scale-dominance-ue4m2-ue5m4-mlp3.jsonl`
- activation stats: `benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512`
- families: `mlp_down`, `mlp_gate`, `mlp_up`
- tensor数: `12`
- candidates: `UE4M2 -> UE4M3 -> UE5M3 -> UE5M4`
- dominance pair: `72`
- 違反: `0`
- UE5M3/UE4M3 weighted relative MSE ratio:
  - min: `1.0`
  - mean: `1.0`
  - max: `1.0`

UE5M3はUE4M3と同じmantissa bitでrangeだけが広いため、今回のMLP weighted sampleではUE4M3をfloorとして完全再現した。UE5M4は一部tensorでさらに改善した。
