# `SQ8_0` decode-attention redesign: shipping audit

## 前回の要点

BH の `SQ8_0` Qwen3-14B decode redesign は、direct 15.228021 tok/s から
GQA-grouped split-tile-20 27.378731 tok/s（1.790050x）まで到達した。一方、
served-model manifest は tile 20 のような typed execution setting を表現できず、
同一モデルでの service-candidate 文章品質比較が未実施だった。現行本番は別モデルの
`AQ4_0` Qwen3.5-9B P3 であり、`SQ8_0` の昇格は本番モデル置換になるため対象外である。

## 今回の変更点

- `ullm.served_model.v2` に、fail-closed な
  `worker.execution.paged_decode_attention` contract を追加した。現在許可する値は
  `gqa_grouped_split` と tile `20|128|256|512` のみで、対象を `SQ8_0` / gfx1201 /
  split-HIP guard に限定した。gateway は継承した selector を scrub し、typed contract
  の selector だけを注入する。pipeline は表現できず、worker 側も混入を fail-closed にする。
- 既存 active `AQ4_0` P3 manifest
  `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` は
  `worker.execution: null` のまま validator を通過した。promotion / rollback は raw
  manifest bytes を swap する既存方式で typed field を保持し、往復 test も追加した。
- `AQ4_0` の既存 C=1339 ROCprof trace を marker と `hipModuleLaunchKernel`
  correlation で再集計した。`ullm_paged_decode_attn_f32_kernel` は decode marker 内で
  0 回、split partial/merge の合計は inclusive kernel time の 8.97854% だった。
  Qwen3.5-9B config は 32 層（linear 24、full 8）、Q/KV=16/4、head/value dim=256
  の GQA 4:1 である。
- BH grouped body の 5:1 / 128 shape と異なるため、`AQ4_0` への直接適用は不可と
  判定した。4:1 / 256 用の新 kernel は prefill-attention 作業が所有する runtime source
  を変更する必要があり、本タスクでは実装・昇格しない。
- 同一 source commit と worker binary を固定した direct / grouped-tile-20 `SQ8_0`
  manifest と、固定 ten-prompt suite capture の隔離実行を準備した。比較は exact match
  を閾値にせず、実生成文と blocking failure を読む。

## 次の行動

- R9700 lock の current owner が完了した後、non-blocking `flock` で P3-compatible
  C=1339 ROCprof trace を一度だけ取得し、historical trace の route/割合を current
  package binding でも確認する。lock が held のときは実行しない。短い release/reacquire
  gap で三回試みたが、実行直前の lock check が busy を返したため GPU work は開始していない。
- active `AQ4_0` P3 manifest を loopback gateway で smoke し、direct と grouped
  `SQ8_0` を別々の isolated gateway で fixed suite 実行後に並置比較する。systemd、
  active manifest、`/opt/ullm` は変更しない。`SQ8_0` は昇格しない。
- trace / capture / postflight をこの日付の evidence directory に追加してから、plan と
  本 journal を結果で更新する。
