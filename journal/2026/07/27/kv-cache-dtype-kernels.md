# KV cache FP16 / FP8 kernel completion and measurement

## 前回の要点

- BT は generic paged-KV の F16 / FP8 E4M3FN storage contract、CPU reference、
  scale validation、capacity accounting を完成させたが、AQ4 resident native
  writer/reader と GPU 実測は未接続だった。
- BR/BH の F32 prefill staging と GQA-grouped decode split は既存 production
  behavior として保持する必要がある。

## 今回の変更点

- F32 legacy symbols/bodiesを保持したまま、非 F32 K/V ordered specialization を
  HIPRTC/launcher/C ABI/Rust FFI/AQ4 resident routingへ追加した。FP8 は K/V 独立の
  per-token/KV-head FP16 scale と gfx1201 native E4M3FN conversion を使う。
- AQ4 full model の native required path で F32/F16/FP8 capacity、M=128 prefill、
  3,968-token prefix decode、3,968-token natural-language generationを測定した。
  FP8 16,256 logical context の model load は成功した。
- F32 regression は SQ8 serial-GQA oracle の hidden/logit 10/10 byte match と、
  BH decode control 27.576901 tok/s で確認した。
- 結果は
  `benchmarks/results/2026-07-27/kv-cache-dtype-kernels/run-20260727T021656+0900/`
  と同ディレクトリの README / summary に保存した。
- F16/FP8 を served model には昇格していない。manifest execution contract に
  `ULLM_KV_CACHE_DTYPE` の認可された selector がなく、typed prefill も F32
  WMMA path より遅い。長文 text は F32/F16/FP8 で 64 token 完全一致だった。

## 次の行動

- typed causal-prefill reader を F32 WMMA path と同等以上の GQA/WMMA staging に
  最適化し、FP8の容量利得を prefill の実用性能へつなげる。
- served-model schema が KV dtype を fail-closed に表現できるようになった時点で、
  F16/FP8 candidate を同一モデルとして lightweight promotion suite で評価する。
- 64-token quality trace は最終回答前に切れているため、必要なら思考出力を含む
  より長い生成予算で同じ長文 factual prompt を追加評価する。
