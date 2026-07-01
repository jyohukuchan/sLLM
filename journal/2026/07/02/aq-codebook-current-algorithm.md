# AQ codebook current algorithm

## 要点

- 今後しばらくは BF16-error を主要な判断指標として扱う。
- 現行の codebook 決定は block-size を考慮している。候補ごとの `group_size` で raw-value を block に分け、block 内の最大絶対値で正規化した値から codebook を作る。
- 現行の codebook 決定は local-scale が取れる離散値を含めた同時最適化ではない。codebook を決めた後に、scale format から得た local-scale 候補と tensor-scale を使って近傍探索している。
- したがって現行方式は「block-size-aware codebook + discrete local-scale search」であり、「scale-aware codebook optimization」と呼ぶにはまだ弱い。

## 次に試す改善

- codebook、tensor-scale、local-scale、codebook-index の割り当てを交互に更新する方式を検証する。
- block ごとに許される local-scale の離散値を前提に、BF16-error が直接小さくなるように codebook を更新する。
- activation second moment を使う場合は、codebook 更新と local-scale 探索の両方で同じ重みを使う。
