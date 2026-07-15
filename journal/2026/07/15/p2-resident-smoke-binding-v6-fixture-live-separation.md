# P2 resident smoke binding v6 fixture/live separation

## 前回の要点

- `resident-one-case-smoke-binding-v5`はREADY candidate failure audit対応runnerを束縛しているが、actual-modeのpre-spawn検証でも歴史的5-field `fake-ready.json`へlive 6-field契約を適用して停止した。
- `resident-one-case-smoke-prepared-v1`と既存binding-v3/v4/v5は歴史成果物として変更しない。

## 今回の変更点

- fresh `resident-one-case-smoke-binding-v6`を生成し、fixture-only検証とlive READY検証を分離したrunnerを束縛した。
  - runner commit: `76c48aa27c08f8cd5115a15e6be25b83d679d8fa`
  - tree: `e79865753bcbba1a9134670fa2ea57327ab84ea4`
  - Git blob: `1929ca23d50c85d3464f9a2c87f1e062d0dc665a`
  - raw SHA-256: `bbe978ede0e4662c33d0d12eee4194531f340b9c06001f37d619019197fd5138`
- historical synthetic READYは`stage=pre_spawn_fixture_only`、`runtime_proof=false`、`ready_proof=false`、`model_load_proof=false`として記録し、runtime証明に昇格させない。live `validate_ready`は互換flagを持たず、`served_model_binding`を含むexact 6-fieldだけを受理する。
- generator/validator authorityをcommit `fb0a5afe86763c95c7bef99ae19ac864c2f56bd5`へ更新した。
  - tree: `798db857d239b75165f85c2e540e07afa13d3574`
  - Git blob: `38e0979ff9dc33fc31f76e4dc91f3421a851d660`
  - raw SHA-256: `72cb128fa4290e468d2e482749138a17d8a8ef2e05d1ad456fa1bcc9cc5ad789`
- binding manifestはschema `ullm.aq4_p2_resident_smoke_binding.v6`、status `prepared_not_executed`、promotion/launch eligibilityともfalseである。
  - manifest raw SHA-256: `b206df5a49b9e97146f9f0488b262814e7f6ec3071a9dbb3c017cfe8dbb2f7b7`
  - manifest canonical semantic SHA-256: `7dbb876eeffd83f0e515d8933a4489ef3f0fcdeda16415677a2f88414ac31538`
  - `SHA256SUMS` raw SHA-256: `684e3be7a50393b3b8c7b045c3719727b4ea6f1ceaabfd3476c3158215076e50`
- rootはexact `0555`、`SHA256SUMS`を含む全7 filesはexact `0444` / `nlink=1`である。archived runner/validatorは各Git objectとbyte一致した。
- resident driver pinはbinary SHA-256 `458b8603d6823a1c20ea93e7c0d757c8910f3c36c9a2a34ab536853c0c9e7d34`、source blob `7e37119cc8b66dc0e0f7abcf49b896fcdad8315f`のまま変更していない。
- official生成前の相対`--output`指定はabsolute-path契約でfail-closedした。途中生成されたv6 directoryだけを削除し、canonical absolute pathでfresh生成した。既存bindingやprepared rootは変更していない。
- CPU-only検証はgenerator `66 passed`、runner `54 passed`、formal `validate-binding`、`sha256sum -c SHA256SUMS`がすべてPASSした。runner subprocess 1、trusted validator subprocess 1、synthetic fake-ready subprocess 1であり、model load、GPU command、service操作はすべてfalseである。

## 次の行動

- launcher/downstream laneはbinding-v6 root、manifest raw/semantic SHA-256、runner authority、validator authorityを明示的にpinし、既存failure evidenceを変更せずfresh v6 downstream成果物を生成する。
- actual、GPU、service実行は新しいsingle-use authorizationと全downstream pin更新が完了するまで行わない。
