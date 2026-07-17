# paged decode split-source runtime

## 前回の要点

generic paged decode attention は1 block/q headで全source tokenを逐次走査する。wave単位のsoftmax scalar共有は製品実測でkernel時間が8.16%悪化したため不採用になった。

## 今回の変更点

- 既存APIとkernelを維持したまま、caller-owned workspaceを使う明示的split-source plain/gated ABIを追加した。
- partial kernelは`q_heads * split_count` blockでsource tileごとのonline softmax state `{max, denominator, numerator[value_dim]}` を書く。workspace strideは`value_dim + 2` floatsとする。
- merge kernelはsplit id昇順でpartial stateを数値安定に統合し、plainまたはsigmoid gate付きoutputを書く。partialとmergeは同じstreamへ連続launchし、host同期は追加しない。
- invalid block idはpartial denominatorの`-1` sentinelで伝え、mergeが対象q headのoutputを全ゼロにする。
- Rust workspace byte helperとC++側の独立したoverflow、grid、backend、buffer size検査を追加した。explicit split APIはlegacy kernelへfallbackしない。HIPRTC unavailable時だけguard無しでhost stagingを許可する。
- CPU plain/gated、workspace境界/overflow、undersized/source tile zero、HIP backend mismatch、native invalid table zero、必須context群とsource tile 128/256のHIP differentialを追加した。

## 次の行動

- runtime primitiveの正しさを確定後、engine/product dispatchとは別の性能実測でsource tile 128/256とlegacyを比較する。
