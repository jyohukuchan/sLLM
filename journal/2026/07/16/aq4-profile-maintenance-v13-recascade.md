# AQ4 profile maintenance v13 recascade

## 前回の要点

- actual-v11 は commit `854e5a348bd3c0f442f2371a0d3619308bce3b95` に封印され、maintenance-evidence-v9、operator-result-v11、actual-audit-v11 だけが存在する。
- actual-v11 は invocation 1/1、return code 1、retry false の pre-stop failure evidence であり、execute-evidence-v9、runtime-v9、capture-v9 は生成されていない。
- launcher と execute-binding-v9 は有効な immutable authority であり、変更する必要がない。

## 今回の変更点

- profile maintenance evidence だけを v9 から fresh v10 へ更新した。launcher が所有する execute-evidence-v9、runtime-v9、capture-v9 は不変である。
- profile ready と profile dry-run を v12 から fresh v13 へ更新した。maintenance-v10、ready-v13、dry-run-v13 は全て未生成である。
- actual-v11 の3 sealed rootsについて、全 `SHA256SUMS`、commit `854e5a34` の全 Git blob、failure stage/reason、service stop/start 0、launcher/capture/rocprof 0、service untouched、invocation 1/1、retry false、no-op restore、failure-evidence-only を readbackするテストに更新した。
- historical ready-v12 artifact tests は current maintenance sourceのmoving constantsから分離し、固定v12 rootsを直接検証するようにした。
- namespace/test変更は commit `72089db8806dd7c2123061356d498cc247f2daf2` に保存し、maintenance testのcurrent blob `2f406df88e278195c2212a706911a78d06f05dd2` を QA exact test manifestへ反映した。test countは156のままである。
- maintenance testsとhistorical ready-v12 artifact testsは合計158件全てpassedした。QA exact aggregateもresident trust chain 382、resident driver 22、ROCTX 5、diagnostic capture 60、selection raw 105、profile family exclusion 39、candidate selector 26の合計639件、failed 0で通過した。GPU command、service操作、actual実行、ready/dry artifact生成は行っていない。

## 次の行動

- final maintenance source commit/blob/SHA-256をprofile-ready-v13 harness authorityとして独立確認する。
- fresh profile-ready-v13とdry-run-v13を別作業で生成・封印する。
- 次のoperator recascadeではmaintenance-v10だけを更新し、launcher/execute-binding-v9とruntime/capture-v9を維持する。
