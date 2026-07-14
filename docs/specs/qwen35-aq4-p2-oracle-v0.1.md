# Qwen3.5-9B AQ4 P2 source/path oracle v0.1

## 前回の要点

P2では、AQ4 packageの同一artifact pathだけでなく、独立したBF16/F32 source oracleと比較する必要がある。全logit行列を保持する既存の簡易比較は、9B modelの語彙サイズとM gridを扱うにはOOMリスクがある。

## 今回の変更点

この仕様は `capture-qwen35-aq4-p2-oracle.py` と `validate-qwen35-aq4-p2-oracle.py` が扱う三つのhash-bound manifestを固定する。

- `ullm.qwen35_aq4_source_oracle.v1`: 独立sourceモデル（BF16/F32）のbounded hidden/logit sample、exact greedy token、top-k。
- `ullm.qwen35_aq4_path_oracle.v1`: 同一AQ4 artifactのall-M=1 path。source oracleとは別root・別manifestで保存する。
- `ullm.qwen35_aq4_oracle_link.v1`: 両manifest、payload、artifact/package、tokenizer identityをhashで結ぶ比較結果。

現行 product が独立した artifact manifest を公開していない場合は、path
manifest の `identity.artifact.artifact_manifest_sha256` を `null` とし、実在する
package manifest の SHA-256 (`package_manifest_sha256`) だけを束縛してよい。production
でこのpackage-only identityを使う場合は、`artifact_binding_kind=package_manifest`、
active served-model manifestの実path/SHA-256、同manifestが宣言するpackage SHA-256を
追加で必須とする。validatorはactive manifestを再読し、`product.artifact=null`、
product root内のpackage manifest、worker binary、gfx1201/
`rdna4_aq4_resident` identityまで一致した場合だけproduction path evidenceとして扱う。
legacyまたはfixtureのpackage-only identityはvalidな診断用artifactとして読めるが、
`usable_as_path_evidence` と `usable_as_p2_oracle_link` はfalseのままとする。

PayloadはJSONLを逐次読み取りし、4 MiB、128 cases、128 steps、256 sample values、top-k 32を上限とする。validatorは重複キー、非有限数、symlink/path escape、順序、coverage、payload hashを再計算する。oracle内外のmanifest、payload、runtime、SHA256SUMS、package、worker、binary、cases入力はsingle-link regular fileだけを受け付け、全path componentのsymlinkを拒否する。`O_NOFOLLOW`で開いたfile descriptorとpathのdevice/inode/size/mode/mtime/ctime/nlinkをread前後に照合し、hardlinkと読み取り中の置換を拒否する。

validatorはmanifest、payload、runtime、SHA256SUMSを最初のpinned file descriptorから得たbytes、file identity、SHA-256のsnapshotとしてvalidation contextへ保持する。schema/semantic、replay、source/path comparison、checksum、最終reportは同じsnapshotを再利用し、検証段階ごとのpath reopenを行わない。link検証ではsource/path/linkのcontextをvalidation完了まで共有する。全semantic/checksum検証後に全snapshotのpath identityを再確認するため、semantic後の別SHA版への差替え、checksum後のrename、同一size rewriteとmtime復元も拒否する。

runtimeが参照するpath binary、served worker、package/artifact manifestはJSONの4 MiB上限を適用せず、single-linkのpinned file descriptorからchunk単位でSHA-256を計算する。外部product/package directoryはroot identityに加え、全entryの相対path、種別、size、identity、regular fileのsame-fd digestをsnapshotへ固定する。validation完了時にfile identityとdirectory treeを再走査するため、path semantics検証後の同一size binary rewrite/rename、worker/package/artifactの置換、directory entryの追加・削除も拒否する。

sourceではconfig、index、全4 shard、tokenizer 5 filesを実pathからstreaming SHA-256で再読し、canonical aggregateを照合する。source manifest runtime、runtime.json、CPU/BF16、package version、thread数、preflight、row count、SHA256SUMSも相互照合する。pathでも`runtime.json`と`SHA256SUMS`を必須とし、rootのfile coverageとchecksum順をexactに検証する。hidden/logitの完全な行列は保存しない。

path runtimeは`ullm.qwen35_aq4_path_oracle_runtime.v1`のexact schemaである。productionでは`evidence_scope=production_gpu`、`device_kind=gpu`、runtime device index `1`、`HIP_VISIBLE_DEVICES=1`、`ULLM_HIP_VISIBLE_DEVICES=1`を実際の子process環境へ設定し、そのexact mappingをruntimeへ記録する。served workerはgfx1201と`rdna4_aq4_resident`でなければならない。CPU実行はsynthetic fixtureに限り、device index `0`、visible mappingなし、`evidence_scope=fixture_only`、promotion falseとする。

path runtimeはsource oracle root、manifest/payload SHA-256、元cases JSONのpath/SHA-256、caseごとのsource greedy列のdomain-separated SHA-256、各stepが使うprompt+直前までのsource token context hashを保持する。validatorはsource payloadとcases入力から全値を再構築する。package/artifact、path binary、served manifest、worker、実行環境、replayのunknown field、duplicate key、hash/path差し替えはfail-closedとする。

greedy tokenは全語彙の最大logitとし、同値では最小token IDを選ぶ。top-kは全語彙をlogit降順、同値token ID昇順に並べた先頭kである。実行中に保持できるlogitはfinal tokenの1 vocabulary rowまでで、sequence×vocabulary matrixは保持しない。

`same_artifact_all_m1` は path oracle の意味であり、source oracleの代替ではない。source/path単体は `usable_as_source_evidence` / `usable_as_path_evidence` のみを判定し、candidate promotionを判定しない。linkも `usable_as_p2_oracle_link` と `promotion_eligible=false` を分け、最終promotionはP2 validation側だけが判断する。synthetic fixtureはproduction evidenceへ昇格できない。

## 次の行動

`export-qwen35-aq4-source-oracle.py` は、installed official `transformers` によるCPU-only BF16 source forwardを実行する。`local_files_only`、1 process、`inference_mode`、caseごとのcache解放、final tokenの1行logitだけを使う。CPU preflightはcheckpoint bytesの1.5倍をMemAvailableから要求し、失敗時は実行しない。`accelerate` がないため現環境では `low_cpu_mem_usage=False` を記録するが、full logit matrixを保存しない。

QA修正版source oracleは `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2/` にあり、独立validatorとSHA256SUMSを通過する。Qwen3.5-9B BF16 source、revision、checkpoint/tokenizer aggregate、3 bounded rows、全語彙tie規則に従うexact greedy/top-kを記録する。旧v1は上書きせず履歴として残す。AQ4 path oracleは同じcaseを同一artifact identityで供給する。両方の独立payloadが揃うまでP2 candidateはpromotion不可とする。

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
python3 tools/export-qwen35-aq4-path-oracle.py \
  --allow-package-only --package-dir PACKAGE --package-manifest PACKAGE/manifest.json \
  --served-model-manifest ACTIVE.json --source-oracle SOURCE_ORACLE \
  --cases CASES.json --tokenizer-root TOKENIZER --output PATH_ORACLE \
  --device-kind gpu --device-index 1 --visible-devices 1
```

`capture`、`link`、`validate`はmodel runtimeを起動しない。`export-qwen35-aq4-path-oracle.py`だけが専用Rust path binaryを1回起動し、production GPU mappingとserved identityを上記契約へ固定する。`validate-... probe`はpresent checkpointとmissing independent forward artifactを区別し、exact blockerを報告する。
