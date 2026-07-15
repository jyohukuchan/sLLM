# P2 candidate1 AQ4 layer-0 matvec oracle

## 前回の要点

attempt-3 の GPU differential trace では decoder layer-0 の量子化経路に
stage sample が記録されたが、`model.language_model.layers.0.linear_attn.in_proj_qkv.weight`
のテンソル単位出力は記録されていない。したがって、stage sample だけから
カーネル不具合を分類しないことが必要だった。

## 今回の変更点

- クリーンな専用 worktree/branch で、layer-0 QKV の AQ4（group size 16、低ニブル順）を
  行単位でストリーミング復号する CPU-only oracle を追加した。
- source safetensors の embedding 3 行と layer-0 input RMSNorm ベクトルだけを読み、
  attempt-3 と同じ 3 固定ケース（prompt-0 step 0/1、prompt-1 step 0）の入力を
  コンテキスト結合と epsilon `1e-6_f32` 付きで再構成した。Qwen3.5 の raw BF16
  RMSNorm weight は additive delta のため、runtime と同じ `raw_f32 + 1.0_f32` を
  適用し、raw payload SHA と effective f32 値の SHA を report に固定した。
- package AQ4 CPU matvec と BF16 source QKV matvec を比較した。各ケースの
  `max_abs` は 0.7104448--0.8944544、relative L2 は 0.02345423--0.02735736、
  cosine は 0.9996409--0.9997432 だった。これは量子化差分の観測であり、GPU 不具合の
  判定ではない。
- Python 参照式と host runtime の `aq4_matvec_f32` 契約を模した明示的な f32
  積・加算モデル（runtime API 自体は呼び出していない）を比較した。3 ケースとも
  固定 numeric bound（max abs/relative L2 `1e-4`）を通過したが、bit-exact 一致を
  意味しない。最大絶対差は epsilon f32 適用後で `5.60e-5` 以下、bit mismatch は
  `7924/7934/7960` だった。行スケールは 0、row-scale override は存在しない。
- 係数テーブル、index、scale、codebook、source tensor、source safetensor、cases、trace
  manifest/payload の実ファイル SHA-256を、capture 前後の stat とともに report に固定した。
  package/source/cases/trace のcanonical containmentとTOCTOUを検証し、形状、BF16/f32 と
  ニブル順、非有限値数（全 0）も固定した。GPU tensor output がないため、
  status/classification は `blocked_missing_gpu_tensor_output` /
  `inconclusive_missing_gpu_tensor_output` のままとした。
- GPU、常駐サービス、holdout、フルモデルの再実行は行っていない。CPU での読み取りと
  bounded-memory oracle 実行だけを行った。

## 次の行動

別の後続作業で qkv テンソル単位の GPU capture（`ullm.aq4_layer0_matvec_tensor_output.v1`）
を追加し、同じ入力・同じ payload identity と比較する。今回の結果では閾値、昇格、No-Go
判定を変更しない。
