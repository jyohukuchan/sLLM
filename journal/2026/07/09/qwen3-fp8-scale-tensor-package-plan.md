# Qwen3 FP8 scale tensor package plan

## 前回の要点

- SQ8_0とvLLM FP8の同一モデル比較には、Qwen3-14B-FP8のuLLM側実行行が必要。
- Qwen3 `model.*` と既存runtime `model.language_model.*` のnamespace差はruntime lookup側で吸収済み。
- 比較計画の後半には、vLLM + Qwen3-14B-FP8 smoke / representativeとの比較を置いている。

## 今回の変更点

- `ullm-quant`のdirect package計画で、Qwen FP8補助テンソル `*.weight_scale_inv` がAQ4量子化対象になる誤分類を修正した。
- 修正前のQwen3-14B-FP8 dry-runは、total `723`、supported `280`、passthrough `443` だった。
- 修正後のdry-runは、total `723`、supported `0`、passthrough `723` になった。
- `*.weight_scale_inv` は `family=other` / `action=passthrough` になり、FP8本体の `F8_E4M3` matrixもpassthroughのままになる。
- 同一モデル比較の次ステップは、AQ4 direct package変換ではなく、FP8/SQ8_0 package import経路を作ることだと整理した。

## 次の行動

- Qwen3-14B-FP8の `F8_E4M3` payload、BF16 `weight_scale_inv`、BF16 passthrough tensorをAQ再量子化なしで扱うpackage import設計を決める。
- そのpackage/importを使って40-layer `manifest-all` のuLLM SQ8_0 smoke rowを作る。
- vLLM + FP8の既存smoke/representative行と同じshapeで再測定し、same-model throughputとして扱えるか判定する。
