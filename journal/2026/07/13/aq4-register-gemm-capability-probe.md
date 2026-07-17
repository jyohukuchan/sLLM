# AQ4 register GEMM capability probe

## 前回の要点

AQ4 batch の capability probe は `rows=2, cols=3, group_size=2` で、`gfx1201` の register BM8 判定が要求する形状に一致せず、最初の実リクエストで kernel compile/setup が発生していた。

## 今回の変更点

- probe geometry を `rows=32, cols=128, group_size=16, batch=128` に変更した。
- packed index、scale index、16-entry codebook、2-entry scale table を ABI の必要バイト数に合わせて生成した。
- matrix/group/input/output の要素・バイト数を checked arithmetic で算出し、row scales は未使用のままにした。
- binding 寸法、必要バイト数、nibble packed layout、scale_count と batch_count の分離、classifier が利用可能な場合の BM8 選択をテストした。
- 既存の probe fault checkpoint、single synchronize、cache publication 順序は変更していない。

## 次の行動

親エージェントで diff と targeted/full validation を統合し、HIP 実機で `ULLM_EXPERIMENTAL_HIP_AQ4_REGISTER_BM=8` を指定した load probe が register kernel を選択することを確認する。

