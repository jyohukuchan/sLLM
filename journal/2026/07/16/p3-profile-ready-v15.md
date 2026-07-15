# P3 profile ready v15

## 前回の要点

- canonical profile diagnostic は launcher-v10／execute-binding-v10へ更新され、実行予定先は `aq4-p3-diagnostic-rocprof-capture-v10` になった。
- sealed actual-v12 の raw tracesを使うoffline再構成は、canonical actual captureと同一視しない方針になった。

## 今回の変更点

- maintenance sourceをcommit `2167c33fe56c0efcbd3745055e6de8604aafd456`、tree `b76cdd6937d3f5f63565049596d8192ed6f87cd2`、blob `cf4fedca1912cc6cbe54ffbd63456c3ff1dbba53`、raw SHA-256 `f86f5be10968eab00f1fabae7827cd557514437098545049ac82def2ddbf2f0c`へ固定した。
- offline再構成先をdistinct namespace `aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v10`へ分離した。canonical `aq4-p3-diagnostic-rocprof-capture-v10` はready生成前後ともabsentである。
- profile-ready-v15を1回生成し、profile dry-run-v15を1回実行した。readyは`ready_for_one_case`、`actual_eligible=true`、maximum invocation 1、canonical output no reuseを維持した。
- ready binding、harness trust、QA attestation、ready SUMS SHA-256は順に`4c2c2079fd428c8db156e36d0513726ae49e372927770d4d9aba0a0172b4497b`、`1e480401e736310bd0efb02090ddf61f22b11623dc724de269297163ccbcc404`、`40f946ee08af0d77d5a6279d25bd88bfe7170091f8216292608d817e57c52f17`、`9ac4097022bad03258494c7b24b40aedc280d38ff3086135242d1a354f9dadbb`である。
- dry-run JSON／SUMS SHA-256は`743941cfa6c580d9f6fc786a37b9e270f5ee0f8764bb8ffcbceefb0c79f535fd`／`86ab1e7714e05951a17e6a7584bf6183f68a1e009f289751810025f36329ec67`である。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、formal ready loaderと各`SHA256SUMS`がpassedした。dry-runは全process count 0で、service、GPU、model、capture、actualを操作していない。
- QA manifestは13 test files、685/685 passedで、maintenance 165/165、正規driver環境付きcapture 80/80を含む。

## 次の行動

- ready-v15／dry-run-v15のartifact authorityを下流operator-v13へ渡す。
- offline再構成rootとmaintenance-evidence-v11は別authorityとして検証し、canonical capture-v10のfreshnessには混入させない。
