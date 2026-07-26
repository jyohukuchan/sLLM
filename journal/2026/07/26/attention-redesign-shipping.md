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
  判定した。ただし GQA 協調の原理は適用可能なので、共有 source を上書きせず隔離
  worktree で 4:1 / 256 専用 body を実装した。これは BH の literal tile-20 body の流用
  ではなく、新規の shape-closed specialization である。
- 同一 source commit と worker binary を固定した direct / grouped-tile-20 `SQ8_0`
  manifest と、固定 ten-prompt suite capture の隔離実行を準備した。比較は exact match
  を閾値にせず、実生成文と blocking failure を読む。
- 2026-07-27 01:06--01:09 JST の一つの owned window で、service を一度 stop/start
  して current `AQ4_0` P3 C=1339 ROCprof、active P3 の隔離 10-prompt smoke、`SQ8_0`
  direct と grouped tile-20 の各 10-prompt capture を連続実行した。service は
  `NRestarts=0` のまま active に復帰し、active manifest SHA-256 は開始前後で
  `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` だった。
- current trace は direct paged-attention kernel 0 回、split partial/merge 512 launches、
  37.378910 ms / 411.411732 ms = 9.08552% を確認した。BH body の 5:1/128 条件とは
  `AQ4_0` の 4:1/256 が異なるため、literal 直接適用不可という結論を維持する。
- `AQ4_0` P3 は unchanged manifest で全 10 request を完走した。`SQ8_0` two-arm
  capture も全 request 成功かつ自動 blocking なしだったが、grouped 側の Python code
  response はコードを出さず、JavaScript 説明には誤り、Japanese multiturn は不完全だった。
  したがって exact-match 0% を閾値にせず、実文章を読んだ結果として quality approval は hold
  とした。`SQ8_0` はいずれにせよ active `AQ4_0` を置換するため昇格しない。
- 隔離 source commit `c8074928` の `AQ4_0` full-model A/B（C=1339、warmup 6、各 32
  decode step ×2）は direct 74.110977 tok/s、4:1×256 grouped 74.509830 tok/s、
  **1.005382×**（+0.398854 tok/s）だった。token `4445` の 32-token 列は各 run で同一
  だが、これは狭い診断であり文章品質 gate には使わない。
- typed `aq4_gqa_grouped_split` / tile 128 manifest `69a5…ec2e` を
  `tools/promote-served-model.py --yes` で昇格した。active/candidate の固定 10 prompt は
  全件成功、blocking なし、生成文も全件同一だった。exact match 1.000 は記録値であり、
  合否閾値ではない。service restart は 1 回で成功し、`NRestarts=0`。active は現在この
  `AQ4_0` candidate であり、`SQ8_0` は昇格していない。

## 次の行動

- active `AQ4_0` candidate の source commit `c8074928` は local branch
  `bq-aq4-grouped-c807` に固定した。共有 runtime source の owner が未確定差分を commit
  した後、同じ shape-gated patch を main に統合する。service を再起動して取り直す必要は
  ない限り、新たな service window は使わない。
- `SQ8_0` grouped tile-20 は service-candidate evidence として保持するが、今回の fixed
  suite の実文章品質は hold のままとする。将来再評価するなら、同じ model/control で
  code/multiturn completion を十分に観察できる prompt contract を別証跡で設計する。
- active manifest は `AQ4_0` grouped candidate
  `69a5e1eb2e7713a1d017332539a587b9a13cf925cbfb28d7c89719ba6709ec2e`。`SQ8_0`
  promotion は実行しない。
