# R9700 FP8単独並列化の検証

ユーザーの「FP8単独でR9700に適用する検証を行い、問題無ければ採用」に対応した。
結論は**既定採用を見送り、今回の候補変更を復元**。正しさの不一致はないが、M1の速度改善は
実wholegraphの同一process AB/BAで安定せず、M3は通常MTPありで明確に退行した。
V620の既存採用範囲とR9700のscheduler=1を維持する。commit/pushは実施していない。

## 条件と候補

- R9700 `gfx1201`、GPU `GPU-a8e9ddefa2d60f55`、ROCm 7.14、wave32/co_v6。
- Qwen3.8-27B-NVFP4 revision `57926baca9a82b4d6906b43f2750d55315f5b10f`、
  NVFP4本体＋FP8 GDN、MXFP8 E4 KV、prefill chunk 2048、context capacity 10240。
- 従来coding chat-template 8192入力／128出力、seed 123、temperature 1、top-k 20、top-p 0.95、
  fresh request、通常計測は1 warmup＋3 measured。MTP幅2。
- R9700のNVFP4は直列のまま、FP8 QKV/Zだけを候補とした。最終候補はM1に限定。
- M1実providerは `launch_fp8_outer_gfx1201_dot4`、M3はhipBLASLt。
  readonly入力、別出力、M3のworkspace 0とgraph lifetimeを確認。M3はwhole MTP graphから到達する。

生成物root: `.local-artifacts/phase87/r9700-fp8-fork/`（非追跡）。

| identity | SHA-256 |
| --- | --- |
| baseline binary | `dc11e7555a0facd3d8d4e332298c2bcd3e0a25513ed7c494a528595974b00f2c` |
| M1＋M3候補 binary | `a620073672d850eab28cc536bbce0e98feb30d16f4758d9e108cfa5be23b1e50` |
| M1候補 binary | `c656140331e743873809836163e9342a2ba3137b0c811afea18cc805a443cc3f` |
| M1候補 source | `d33e7482d6d882a85011f5a5dbe536810cf2703a2f55b27dc708d9b82e422dd1` |
| 復元するbaseline source | `7cb87884b44959f815934e6f1fbf4552988af645fb26a4157ba554a1022b6180` |

## 通常モデル

中央値、token/s。全warmup/measured runはbaselineと生成token列が一致した。

| 実行 | MTPなし | MTPあり |
| --- | ---: | ---: |
| A1 baseline | 21.476801 | 35.622923 |
| B1 FP8 M1＋M3 | 21.809116 | 22.005197 |
| B2 FP8 M1のみ | 21.843773 | 35.622255 |
| A2 baseline | 21.473607 | 未追加計測 |

通常計測のM1は約1.7%改善し、MTPありは同等。一方M3は−38.23%のため除外した。
B1/B2はMTPなしでは同じM1 forkだが、binaryが異なるので通常表だけを厳密な同一binary ABBAとは呼ばない。
詳細は `normal-comparison.json` と各 `a1/b1/b2/a2` report/execution。

## 実wholegraphの同一process AB/BA

単体graphと通常モデルの傾向が異なったため、同じ作業単位で方法を追加した。
M1候補binaryに診断専用shimを使い、capture時の96回のdependency更新をAだけno-opとした。
これで自然な直列frontierを維持する。Bは元のHIP APIを実行する。追加replay/event/同期はない。
同じ入力のfresh requestでwarmup B、続いてA/B/B/Aを2回。9 captureすべて構造検査PASS、全run N0一致。
各graphは1172 nodes（kernel 1170、memcpy 1、memset 1）、Aは1171 edges、Bは1219 edges。
別の実DAG比較でもFP8 48組だけのforkを確認し、NVFP4 56組と他の辺は不変だった。

| 対比較 | 速度差 | TPOT短縮率 |
| --- | ---: | ---: |
| AB 1 | +1.5240% | +1.5011% |
| BA 1 | −0.0853% | −0.0854% |
| AB 2 | −1.7024% | −1.7319% |
| BA 2 | +1.6938% | +1.6656% |

A中央値21.477501、B中央値21.635088 token/s（+0.7337%）。
Aは21.4727〜21.4809、Bは21.1071〜21.8385で、Bのばらつきが大きい。
全round改善・通常TPOT寄与1%以上という既存の採用目安を満たさず、安定した利得を確認できなかった。
原因を物理的なGPU競合やROCmの特定不具合に断定しない。
証拠は `fullgraph-ab/summary.json`、`capture-1220723.jsonl`、`model/mtp-off/report.json`。

## 単体と品質

実providerの1pair graph（3 process×6 AB/BA）はM1が17/18 round退行、M3が12/18退行。
48pair graphでもM1は18/18退行（median −102.09 µs/pair）。数値oracle、finite、N0、cleanupは通過した。
単純なpair数では実モデルとの差を説明できず、合成graphの値を本番TPOTへ直接換算しない。
48pair runnerはJSON配列終端のlogger不備でFAILになったため、その記録を保持し、値を変えず正規化した
`unit/bundle/runs/run-1-normalized.json` と修正前source/binaryを保存した。GPU処理の完了とrunner PASSを混同しない。

固定128位置（8191〜8318）の全vocab logitsはbaseline/M1候補でbit一致。
FP32 export SHAは両方 `1de799171184ce565f271fb6bf50827ea6e9c517bad8995716d5fcd35fe45a53`。
同じ入力の既存llama.cpp BF16 reference（FP16 KV、batch/ubatch 64）に対し、
mean KLDは**0.023446019680224913 nats、変更差0**。BF16 top1一致率は94.53125%。
referenceは `llama-bf16-coding8192-fp8-reevaluation-v1`、比較は `quality/kld-vs-bf16.json`。
この品質評価は固定128位置の範囲であり、追加の広域corpus評価ではない。

## 復元・検証

候補source、binary、rawは非追跡rootに保持し、本番sourceとCI inventory hashを試行前へ戻した。
候補はgfx1201/gfx1030 build、C++ format、JSON manifest、H3 contracts、focused H3 host testsを通過。
限定レビューでnative分岐とAB/BA shimにcorrectness blockerなし。
最終復元後もgfx1201/gfx1030 build、JSON manifest、H3 contracts、Markdown link、diff検査がPASS。
R9700の再build binaryはbaselineのSHAと完全一致した。結果は同rootの `restored/verification.json` に保存した。
GPU service/performance levelはcanonical runnerのlease開始状態へ復元し、service file hashを維持した。

計画: [R9700 FP8単独並列化](../../../../plans/archive/2026/09/21-30/phase87-r9700-fp8-fork.md)
