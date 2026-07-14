# AQ4 P2 calibration evidence binding v0.1

## 前回の要点

P2 resultはsampled source oracle、same-state all-M1 result、独立validationをhashで結んでいた。しかし、full-vector `source_gate` / `path_gate` comparison artifactはresultと結ばれておらず、5 correctness指標が同じcase、identity、事前bound policyから得られたことを最終validatorが再構築できなかった。

## 今回の変更点

`ullm.aq4_p2_calibration_evidence.v1`をP2専用の隔離binding schemaとする。source calibration側の内部schemaをP2 resultへ直接展開せず、次の値をexact fieldとして束縛する。

- compare kind、comparison manifestの実pathとSHA-256
- expanded case全体、canonical case-set SHA-256、prefillは1 step、decodeはgenerated token数と同じstep count
- model identity、sampled source oracle SHA-256、package content/manifest SHA-256、worker binary SHA-256、bound policy SHA-256
- case内のdevice、prompt/context/prefix、requested/resolved M、phase、mode、sampling/control

全resultは`source_gate` bindingを必須とする。`cold_batched`と`cached_prefix_chunked`は、別SHA-256の`path_gate` bindingとsame-state all-M1 resultを必須とする。`all_m1`とdecodeはpath bindingを持たない。最終matrix validatorは、別caseで同じcomparison SHA-256を使うreuse、source/path swap、case swap、partial matrixを拒否する。

comparison manifestは`ullm.qwen35_aq4_calibration_comparison.v1`のexact fieldsだけを受け付ける。`source_gate`は`independent_source_full -> aq4_target`、`path_gate`は`same_artifact_all_m1 -> aq4_optimized`である。statusは`valid`、promotionはfalse、observed-values-onlyはtrue、nonfinite rowsとgreedy mismatch rowsは0でなければならない。

次の5指標は、calibration実行前からhash-boundされたpolicy値だけと比較する。null、bool、NaN、Infinity、負値、未知field、policy超過を拒否する。

1. hidden relative L2
2. hidden max absolute difference
3. logits relative L2
4. logits max absolute difference
5. top-10 overlap

calibration timingは`performance`へ入れない。performanceは従来どおりraw-v2 measurementの2 warmup + 10 measuredだけから再計算する。

入力ファイルはrun root内に限定し、全path componentのsymlink、leaf symlink、hardlink、非regular fileを拒否する。読取とhashは`O_NOFOLLOW`で開いたfile descriptorを固定し、device/inode/size/mtime/ctime/mode/link countの変化を拒否する。result/report publishはhard-linkによるatomic no-replaceとする。

builder CLIは`--source-calibration-evidence`を必須とし、optimized caseだけ`--path-calibration-evidence`を追加する。source schemaの変更点はこの隔離validator内だけで追随する。

## 次の行動

実R9700 runではcaseごとに別のsource/path comparisonを生成し、このbinding artifactをraw timingとは別rootで保存する。完全6,214件matrix、production trace、power evidenceが揃うまでpromotionはfalseのままとする。
