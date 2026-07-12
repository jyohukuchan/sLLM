# Qwen3.5 9B AQ4 full-native prefill smoke

## 方法

`resident-evidence.json` に保存された実機 `gfx1201` / `rdna4_aq4_resident` の7ケース（prompt 1, 8, 127, 128, 129, 255, 256、各 `max_new_tokens=4`）を原データとした。比較元は既存の linear-native resident と M1 baseline であり、変更していない。prompt chunk は `min(remaining, 128)` とし、各 chunk の成功後に累積 progress を確定する。出力 token 列、progress、length 終了、reset、clean shutdown、子プロセス状態を機械比較した。

operation audit は resident の `stderr_tail` にある各 `request_released.operation_execution_audit` から抽出した。full-native の物理呼び出しは prefill が `64 * chunks`、decode が `64 * 3 = 192`、token-equivalent は prefill が `64 * prompt_tokens`、decode が192である。

## 結果

| prompt | M1 tok/s | linear-native tok/s | full-native tok/s | full/linear | full/M1 |
|---:|---:|---:|---:|---:|---:|
| 127 | 78.964 | 102.579 | **116.609** | 1.137x | 1.477x |
| 128 | 78.849 | 102.149 | **116.561** | 1.141x | 1.478x |
| 129 | 78.728 | 101.792 | **115.972** | 1.139x | 1.473x |
| 255 | 76.556 | 98.214 | **115.847** | 1.180x | 1.513x |
| 256 | 76.406 | 98.028 | **115.590** | 1.179x | 1.513x |

M=128 の audit は total physical 256（prefill 64 + decode 192）、token-equivalent 8384（8192 + 192）で、`hip.paged-kv-write-chunk-f32.m2-m128` と `hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128` は各8回だった。M=256 は total physical 320（prefill 128 + decode 192）、token-equivalent 16576（16384 + 192）で、coverage は全ケース `true` だった。

最大 full-native prompt TPS は 116.61 tok/s（p127）、generation TPS の最大は 95.42 tok/s（p1）であり、数千 tok/s には達していない。初回 native p8 の cold-launch overhead（linear-native 45.900 tok/s、full-native 47.475 tok/s）は定常境界比較から分離した。

## 検証と限界

- 全7ケースの prompt/generated token 列は M1・linear-native・full-native で一致し、progress は chunk 式と一致した。
- full-native は `verified=true`、`clean_shutdown=true`、67件の子プロセス検査で descendants 空・sibling engine 0だった。
- JSON parse、独自 audit/assert、source hash、`git diff --check` を実施した。比較元JSONと `.rocprofv3/` は変更していない。
- これは1台の gfx1201 上の7ケース resident smoke であり、広い同時実行スループット測定ではない。stderr は bounded tail のため、永続化された完全stderrログの代替ではない。
