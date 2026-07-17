# SQ format evaluation execution order plan

## 前回の要点

- R9700/RDNA4のcached-prefix/cold-prefill componentは、SQ候補評価を始める前提速度としては十分と判断する。
- `kup6_gate5_down5` は6層prompt bundleのstrict top1 regression subsetであり、full SQ policyではない。
- T1はpackage-backed component real-batch JSONLまでは進んだが、full package total throughputはまだ未完了である。

## 今回の変更点

- 次の主線を、追加のattention kernel開発ではなくSQ format evaluationへ戻した。
- self-attention componentのrequest-boundary検証は有用だが、最終性能比較ではfull package request-batch runnerを優先する。
- SQ候補1は `sq-fp8-w8a16-r9700-v0` のまま、R9700だけで実装・計測する。
- fixed batch v0.1の評価軸を、`batch=1/4/8`、cold prefill、cached prefix、decode、end-to-end、VRAM、resident bytes、working-set bytes、quality guardに限定した。
- vLLM比較は、uLLM側のAQ4/FP8行が同じschemaで揃ってから行う。

## 次の行動

1. full package request-batch runnerを作り、prefill/decode/end-to-end total throughputをJSONLへ保存する。
2. AQ4 packageで `batch=1/4/8` のsmall gridを通す。
3. `sq-fp8-kup6-gate5-down5-policy-v0.1.json` をruntime pathへ接続する。
4. SQ pathのquality guardをbatch pathにも接続する。
5. FP8 SQ候補1とAQ4 baselineを同じworkload gridで比較する。
6. 結果が固まった後にvLLMを同じgridで測る。
