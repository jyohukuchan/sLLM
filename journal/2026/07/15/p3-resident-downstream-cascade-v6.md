# P3 resident downstream cascade v6

## 前回の要点

- binding-v5 downstream cascadeはlauncher `0ad6d0e6`、maintenance `bf2ef47a`、ready artifact `3a330dce`で固定されていた。
- profile actualのv3/v4失敗証拠に加え、v5 namespaceを使ったactual v6失敗証拠がcommit `3ae0e34e`に保存された。
- generic runnerのhistorical synthetic READYはfixture-onlyに隔離され、live READYはRust driver由来のexact 6-field eventだけを受理するauthority `76c48aa2`になった。

## 今回の変更点

- binding-v6 authorityをdownstreamへcascadeした。
  - binding artifact commit/tree: `31eb65a644eae20a3be6cbeb36b04aaaabf69429` / `e1cc9f2f53e711ea9b5c01d3de9d5379407bf787`
  - binding manifest SHA-256: `b206df5a49b9e97146f9f0488b262814e7f6ec3071a9dbb3c017cfe8dbb2f7b7`
  - runner commit/tree/blob/raw SHA-256: `76c48aa27c08f8cd5115a15e6be25b83d679d8fa` / `e79865753bcbba1a9134670fa2ea57327ab84ea4` / `1929ca23d50c85d3464f9a2c87f1e062d0dc665a` / `bbe978ede0e4662c33d0d12eee4194531f340b9c06001f37d619019197fd5138`
  - validator commit/tree/blob/raw SHA-256: `fb0a5afe86763c95c7bef99ae19ac864c2f56bd5` / `798db857d239b75165f85c2e540e07afa13d3574` / `38e0979ff9dc33fc31f76e4dc91f3421a851d660` / `72cb128fa4290e468d2e482749138a17d8a8ef2e05d1ad456fa1bcc9cc5ad789`
- launcherをcommit `32d7a855a5dee03a82ec86670ffd902380084874`で更新した。binding/runner/validatorを上記authorityへ固定し、execute-bindingとprofile namespaceをv6へ進めた。
- 過去のbase actualがexecute-v6 / maintenance-v9を使用済みなので、fresh base runtime outputはexecute-v7 / execute-evidence-v7 / maintenance-evidence-v10へ進めた。profile runtime outputは未使用のv6へ進めた。
- execute-binding-v6を生成し、commit `28ff5121`へ保存した。
- v3/v4に加え、v5 profile failureのexecute evidence、maintenance evidence、capture failure、operator result、actual auditをhistorical immutable evidenceとしてtestに固定した。test commitは`9a233075`である。
- maintenanceをcommit `0477fddf6827868c5258ad5b36f48c1eb9692855`で更新した。launcher、runner、validator、binding artifact、capture authority、strict QA provenanceをcurrent commit/path/blobへ固定した。
- base/profile ready-v6と両dry-run-v6を生成し、artifact commit `ff15b75ceed5e7b7eabe376e27859106694c285f`へ保存した。
  - 全rootはmode `0555`、全memberはmode `0444` / nlink 1である。
  - 両dry-runは`status=passed`で、sudo、systemctl、launcher、rocprof、capture、docker系を含む全process countが0である。
  - `service_touched=false`、`gpu_command_executed=false`、`model_load_executed=false`である。
- actual execution、GPU workload command、service操作、model loadは行っていない。

## 検証

- resident trust chain: `379 passed`
  - prepare validator: 66
  - generic runner: 54
  - live preflight: 27
  - launcher: 8 + 69
  - maintenance: 155
- resident driver CPU unit: `22 passed`
- diagnostic capture: `49 passed`
- ROCTX / selection raw / profile family / candidate selector: `84 passed`
- QA aggregate: `534 collected / 534 passed / 0 failed / 0 deselected`
- binding-v6、execute-binding-v6、base/profile ready-v6、両dry-run-v6の`SHA256SUMS`はすべてPASSした。
- 生成した5 rootの全memberはarchive bytesとGit blob readbackが一致した。
- historical profile v3/v4/v5の10 rootは`SHA256SUMS`がすべてPASSした。
- fresh runtime outputはbase 3件、profile 4件の計7件がすべてabsentである。

## 次の行動

- actual profile diagnosticへ進む場合は、current profile-ready-v6を入力とするfresh quiet-windowとoperator authorizationを別作業で作成する。
- actual直前にv6 trust chain、fresh 7 output、service epoch、GPU/KFD owner、lock、formal healthを同一境界で再検証し、差異があれば実行しない。
