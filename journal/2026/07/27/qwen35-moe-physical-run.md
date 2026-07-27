# Qwen3.5-35B-A3B AQ4_0 MoE physical-window result

## 前回の要点

- BW は Qwen3.5-35B-A3B `AQ4_0` MoE の resident loader、shared decode workspace、F16 KV と
  36 個の AQ4 HIP guard を結線済みとしていたが、本番 worker が R9700 を保持していたため
  full runtime は未実行だった。
- 262,144-token ledger は `30,858,010,436 B`、package payload は `25,029,380,864 B` で、
  既存 9B worker と同居できない。CE は active manifest を変えず、一回だけの service window を
  許可された。

## 今回の変更点

- 10:22 JST に `ullm-openai.service` を一回だけ停止した。停止前後の active manifest SHA-256 は
  `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd` で同一、停止後の R9700
  process list は空、edge は 36 C だった。
- `HIP_VISIBLE_DEVICES=1` / runtime device 1、36 AQ4 guard、F16 typed-KV 3 guard で isolated
  runner を実行した。9B `AQ4_0` baseline は HIP backend で top-1 token 220 に一致した。
- 35B runner は 262,144-token F16 KV load の allocation 前、full layer 3 において
  `Qwen3.5 MoE full layer 3 does not match the inspected mRoPE/Q-gate/KV contract` を返して停止した。
  そのため VRAM load、生成文、40 layer raw-BF16 route 照合、MoE prefill/decode throughput は
  いずれも未到達である。ledger 超過・OOM の測定結果ではない。
- source を再確認すると、descriptor builder は mRoPE に `rotary_dim: None` と
  `partial_rotary_factor: Some(0.25)` を出すのに対し、MoE runtime validator は
  `rotary_dim == Some(64)` を要求していた。これは layer 3 の fail-closed が示した contract の
  矛盾であり、今回の停止点である。
- 同じ manifest で service を一回だけ起動して復旧した。OpenWebUI bridge 内から `/readyz`、
  `/v1/models`、completion `service restored` を確認し、`ActiveState=active` / `NRestarts=0` だった。
  MoE は昇格していない。
- evidence は `benchmarks/results/2026-07-27/qwen35-moe-physical-run/` に保存した。gateway の短文
  response timings（prefill 70.942962 tok/s、decode 122.082175 tok/s）は recovery probe の実測であり、
  参照 9B benchmark と workload が違うため性能回帰の判定には使用しない。9B baseline の functional
  top-1 は一致、同条件 throughput regression は未確認である。

## 次の行動

- `resident_rope_from_qwen35()` と MoE runtime validator の mRoPE rotary-dimension contract を一つに
  揃え、descriptor-level regression test を加える。
- 新しい service window が明示的に許可された時だけ、修正済み release binary で 262,144-token
  F16-KV load、短い generation、全 40 layer の route read-back、VRAM telemetry、同条件 9B
  prefill/decode baseline を同じ一回の window で再実行する。
