## 前回の要点

R9700 と借用予定 MI300X の比較は、FP8 形式差を補償せず raw byte を流すと
無意味になる。GPU は並行する CE/CF と共有であり、lock 保持時に奪えない。

## 今回の変更点

- 独立 HIP microbenchmark と build/run/ISA-audit scripts を追加した。
- OCP E4M3FN と FNUZ の二オペランド `×4` 補償を kernel と CPU oracle に固定した。
- gfx1201 WMMA と gfx942 MFMA の静的 ISA を両方確認できるようにした。
- rental runner に P0 完走を prerequisite とする任意 `hw_microbench` stage を追加した。

## 次の行動

- R9700 lock が解放され、service/measurement process が不在になった後だけ実測する。
- 実測 JSONL、telemetry、実測 wall time を comparison table に転記する。
- MI300X では P0 完走後にのみ任意 stage を実行し、host の peak source を残す。
