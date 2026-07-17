# paged decode wave softmax

## 前回の要点

generic paged decode attention の 256-thread head-parallel online softmax は、value lane ごとに同じ `max_score`、rescale、weight、denominator を計算していた。score 自体は既存の block-wide reduction で共有されている。

## 今回の変更点

- `runtime/src/ullm_runtime_hiprtc_sources.inc` の online path で、各 wave の lane 0 だけが `new_max`、rescale、`exp(score - new_max)`、更新後 denominator を計算するようにした。
- `__shfl(..., 0, warpSize)` で scalar を wave 内へ broadcast し、全 value lane は broadcast 値で numerator を更新するようにした。wave32/wave64 とも `warpSize` を使うため、ABI・dispatch・gate・block table・barrier は変更していない。
- runtime-sys に長文・断片化 block table・plain/gated・有限な極端値を含む HIP/CPU differential test を追加した。利用可能な全 HIP device を順に検証する。

## 次の行動

- gfx1201 を含む HIP kernel の静的 compile/runtime test、runtime-sys 全テスト、cargo fmt/check を逐次実行する。
- git diff を確認して、この作業だけを一つの commit にまとめる。

## 製品実測による不採用

### 前回の要点

commit `cd60eb2` では、online softmax の scalar 計算を各 wave の lane 0 に集約した。長文・plain/gated・断片化 block table の differential test は通過した。

### 今回の変更点

- 旧 p1339/g64 profile は paged decode 506 calls、533,991.609 us、平均 1,055.319 us、decode 42.845 tok/s だった。
- `cd60eb2` 後は同じ 506 calls が 577,539.613 us、平均 1,141.382 us となり、kernel 時間が 8.16% 悪化した。profile decode も 41.549 tok/s へ低下した。
- unprofiled は 43.10/43.80/43.85 tok/s で、旧 44.77 tok/s のばらつき内以下だった。製品性能の改善根拠がないため、wave softmax source 変更は不採用として `cd60eb2^` の thread-local online softmax へ戻した。
- 長文 CPU/HIP differential test は paged decode の数値回帰を検出できるため残した。

### 次の行動

- lane 0 集約は再採用しない。次の paged decode 最適化候補は、同じ p1339/g64 製品条件で kernel profile と unprofiled throughput の両方を比較する。
