# Qwen3.5-9B AQ4 P2 source/path oracle v0.1

## 前回の要点

P2では、AQ4 packageの同一artifact pathだけでなく、独立したBF16/F32 source oracleと比較する必要がある。全logit行列を保持する既存の簡易比較は、9B modelの語彙サイズとM gridを扱うにはOOMリスクがある。

## 今回の変更点

この仕様は `capture-qwen35-aq4-p2-oracle.py` と `validate-qwen35-aq4-p2-oracle.py` が扱う三つのhash-bound manifestを固定する。

- `ullm.qwen35_aq4_source_oracle.v1`: 独立sourceモデル（BF16/F32）のbounded hidden/logit sample、exact greedy token、top-k。
- `ullm.qwen35_aq4_path_oracle.v1`: 同一AQ4 artifactのall-M=1 path。source oracleとは別root・別manifestで保存する。
- `ullm.qwen35_aq4_oracle_link.v1`: 両manifest、payload、artifact/package、tokenizer identityをhashで結ぶ比較結果。

PayloadはJSONLを逐次読み取りし、4 MiB、128 cases、128 steps、256 sample values、top-k 32を上限とする。validatorは重複キー、非有限数、symlink/path escape、順序、coverage、payload hashを再計算する。hidden/logitの完全な行列は保存しない。

`same_artifact_all_m1` は path oracle の意味であり、source oracleの代替ではない。linkの `promotion_eligible` は必ず falseから始まり、synthetic fixtureは production evidenceへ昇格できない。
source/path manifestにも `promotion_eligible` を持たせ、`evidence_class=synthetic_fixture` では常に falseを要求する。

## 次の行動

`export-qwen35-aq4-source-oracle.py` は、installed official `transformers` によるCPU-only BF16 source forwardを実行する。`local_files_only`、1 process、`inference_mode`、caseごとのcache解放、final tokenの1行logitだけを使う。CPU preflightはcheckpoint bytesの1.5倍をMemAvailableから要求し、失敗時は実行しない。`accelerate` がないため現環境では `low_cpu_mem_usage=False` を記録するが、full logit matrixを保存しない。

実行済みsource oracleは `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v1/` にあり、独立validatorとSHA256SUMSを通過している。Qwen3.5-9B BF16 source、revision、tokenizer、3 bounded rows、exact greedy/top-kを記録する。AQ4 path oracleは同じcaseを同一artifact identityで供給する。両方の独立payloadが揃うまでP2 candidateはpromotion不可とする。

### CLI bridge

```text
python3 tools/capture-qwen35-aq4-p2-oracle.py capture --kind source \
  --source-root SOURCE --cases CASES.json --payload SOURCE.jsonl --output SOURCE_ORACLE
python3 tools/capture-qwen35-aq4-p2-oracle.py capture --kind path \
  --tokenizer-root TOKENIZER --artifact-manifest ARTIFACT.json \
  --package-manifest PACKAGE.json --model-id Qwen/Qwen3.5-9B \
  --cases CASES.json --payload PATH.jsonl --output PATH_ORACLE
python3 tools/capture-qwen35-aq4-p2-oracle.py link \
  --source-oracle SOURCE_ORACLE --path-oracle PATH_ORACLE --output LINK
python3 tools/validate-qwen35-aq4-p2-oracle.py link LINK \
  --source-oracle SOURCE_ORACLE --path-oracle PATH_ORACLE
```

No model runtime is invoked by these commands. `validate-... probe` distinguishes a present checkpoint from the missing independent forward artifact and reports the exact blocker.
