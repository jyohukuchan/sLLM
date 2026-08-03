# Phase 2 H3・G0・model-free GPU path履歴

## 2026-08-03

- Phase 1完了後の次作業を、ROCm 7.14.0固定toolchain、exact `gfx1030`/`gfx1201` H3、trusted local GPU evidence、G0、model-free最小GPU実行までに限定した。
- H3の20回以上・7日以上の観測はrequired昇格だけの条件であり、G0とmodel-free pathの開発を停止しないと決定した。
- 現行GPU hard gateが未構築のG0/G1/G2/G4/P0をH3自身へ要求するbootstrap循環を記録し、実装前にscope別gateへ修正する作業単位を計画の先頭に置いた。
- model-free最小経路を`Cargo -> ullm-hip -> versioned C ABI -> native HIP -> GPU`とし、allocation、copy、diagnostic kernel、completion、copy-back、解放をcanonical `gfx1030`/`gfx1201`で検証する到達点を定めた。
- 数値op、model load・推論、性能、generic target、互換性昇格を計画範囲外とした。
- この時点では計画文書だけを作成しており、H3、G0、GPU runtimeの実装evidenceはまだない。
- 作業単位0としてCI hard gateを変更scope別へ分割し、H3、G0 runner、model-free runtimeに適用する同一candidate evidenceを明確化した。
- H3 required昇格観測とG0/model-free実装を並行するとCI正本へ同期した。
- `gpu.md`の「実機検証結果なし」を、exact `gfx1030`/`gfx1201`の限定smokeだけが存在しformal G0/G1以降は未検証という表現へ修正し、AMD/software文書と整合させた。
- 公式`docker.io/rocm/dev-ubuntu-24.04:7.14.0-full`をsingle `linux/amd64` manifest digest `sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7`とconfig digestで固定し、ROCm 7.14.0、LLVM 23、`/opt/rocm`同一rootを静的contractへ記録した。
- H3 matrixをexact `gfx1030`/`gfx1201`の2 row、Code Object V6、wave32、`xnack`/`sramecc=unsupported`、non-required、compile-onlyとして固定した。
- HIP artifact metadataをhost側のx86-64 offload bundleと抽出後のAMDGPU device code objectへ分離し、bundle identity、target別ELF ABI/e_flags、candidate SHA/tree、manifest hash、artifact size/hash、row-private build path、非実行scopeをfail-closed検証するschemaとvalidatorを追加した。
- tag-only/`latest`、digest/platform/root/version/LLVM不一致、missing/duplicate/unknown/generic/multiple/wrong target、required化、codegen不一致、target差し替え、stale identity/hash、source/shared build出力、誇張した実行scopeを拒否するnegative testを追加し、255/256/257 byte境界も確認した。この作業単位は静的contractだけであり、H3 compile、GPU実行、数値・model・性能evidenceはまだ生成していない。
- 明示optionでだけ有効になるHIP CMake OBJECT/link pathとCargo build接続を追加し、host-only既定経路を維持した。
- exact `gfx1030`/`gfx1201`を独立compileするH3 runnerを追加した。固定ROCm imageにはCMakeが含まれないため、H3 evidenceではhostのCMakeとlibraryを持ち込まず、image内のpinned `amdclang++`によるcanonical 2 commandのdirect compile/linkを使う。bundle保持host objectの`.hip_fatbin`からdevice ELFを抽出して、Code Object V6、target別e_flags、wave32、定義symbolを同一ROCm rootのLLVM toolsで検査する。明示HIP CMake/Cargo接続は別のbuild pathとして維持し、local systemで両targetがcompile-only PASSしており、生成executableとGPUは実行していない。
- report、metadata、device artifactと全sidecar、candidate SHA/tree、run identity、toolchain/matrix hashをexact 2 rowで照合するfail-closed aggregate contractとnegative testを追加した。
- H3を`host-required`から分離したnon-required workflowとして追加した。workflowはmanifest/config digestを検査したROCm imageをsource read-only、row-private output、`--network none`、GPU device/socketなしで起動し、runner自身もloopback以外のinterfaceと外部到達可能なdefault routeの不在を確認する。imageにGitが含まれないため、candidate identity検査専用にUbuntu 24.04 hostの`/usr/bin/git`だけをexact pathへread-only mountし、実測versionをreportへ記録するがROCm toolchain identityには含めない。
- H3 contract/runner/aggregateの20 test、JSON/schema/workflow validator、matrix validator、Cargo workspace test、local exact 2 target compile-onlyがPASSした。正式なH3 evidenceは同一immutable candidateをdigest固定imageで再検証してから記録する。
- H3実装をcommit `03f90be1ad85145e3abee86e67615c1e17f552b4`（tree `87d034951191f1702817d27a4b16c8dd055f2259`）として公開した。GitHub run `30793742848`ではexact 2 compile rowがPASSし、aggregateがrowの既定run identityとGitHub run identityの不一致を拒否した。workflowからcontainerへrun ID/attemptを明示伝播する修正を次candidateへ追加した。
- canonical G0 rowをV620 `gfx1030`（BDF `0000:03:00.0`、UUID `GPU-76a08c022586fed6`）とR9700 `gfx1201`（BDF `0000:47:00.0`、UUID `GPU-a8e9ddefa2d60f55`）へ固定し、BDF `0000:43:00.0`の2台目V620をspareとして必須evidenceから除外した。
- pinned ROCm 7.14.0のidentity APIだけを呼ぶprivate native observer、exact BDFからのHIP visibility routingと実観測による再検証、read-only AMD-SMI/sysfs health・process前後確認、外部observation injection拒否、temporary binary cleanup、exact H3 artifact rebinding、G0 2 row fail-closed aggregateを実装した。静的contractとhost negative testはPASSしたが、同一immutable candidateのcanonical G0実機evidenceは未取得である。
- 実機の`ras/umc_err_count`が`ue`/`ce`/`de`の3行形式であることを受け、uint64範囲のcanonical keyed parserへ修正し、uncorrectable countには`ue`だけを使用した。missing、duplicate、unknown、signed、whitespace、leading-zero、overflowをhost negative testで拒否する。
- commit `e91ff35caac8247fc056eb14a1d6cee2a2319cc5`（tree `75b229791cd3cf7c6ed38c25264b0cd09a9cde33`）に対し、Python 3.12.10固定環境のH0/H1/H2、digest固定ROCm imageのexact 2 target H3とaggregateがimmutable PASSした。
- 同じcandidateとH3 artifactでcanonical V620 `gfx1030`（BDF `0000:03:00.0`、UUID `GPU-76a08c022586fed6`）とR9700 `gfx1201`（BDF `0000:47:00.0`、UUID `GPU-a8e9ddefa2d60f55`）のG0を直列実行し、両rowとaggregateがPASSした。pre/post healthは不変、GPU processと残留childは0、allocation/copy/kernel/dispatch countは0であり、G0を実行・数値・性能evidenceへ昇格させない。
- G0 aggregateはrun `local-e91ff35c` attempt 1、report SHA-256は`gfx1030=408e95b9b6ccc661a5bab661b0be9da2d9c096425492633d25d6337a3fa22341`、`gfx1201=cb94cffccba48ab908d1ef41a3573bfb3e645a40fb157d458171f36c3a90aa67`である。次はH3 required昇格の7日観測を待たず作業単位5のmodel-free G1へ進む。
- public inference ABIから分離したprivate evidence C ABI、Rustのone-shot completion ownership、native HIP stream/event/device・pinned host buffer lifetime、専用`ullm-hip-evidence` binaryを実装した。各caseは2 device allocation、2 HIP transfer、1 diagnostic XOR dispatchを行い、Rust oracleへbyte exactに照合する。
- G1 builder/runner/aggregateはexact target、Code Object V6、wave32、target別ELF flags、kernel symbol、candidate identity、実際にloadしたHIP/ROCr library path、timeout/output/process cleanup、artifact/sidecar hashをfail-closedに検査する。artifact検査中の`llvm-objcopy`が入力binaryをin-place更新する問題を実buildで検出し、private一時出力の明示と検査前後のsize/SHA-256 bindingへ修正した。
- 最終immutable candidateをcommit `f393d688a051d2b73c8773d8a930a711592609bc`、tree `2ccda6e7c0614d585f26babc6b7c68ca51220bbe`に固定した。H0 106/106、H1 42/42、H2 9/9、digest固定containerのH3 2/2、canonical GPUのG0 2/2、G1 2/2と全aggregateがPASSした。
- G1はV620 `gfx1030`（BDF `0000:03:00.0`、UUID `GPU-76a08c022586fed6`）とR9700 `gfx1201`（BDF `0000:47:00.0`、UUID `GPU-a8e9ddefa2d60f55`）で1、3、17、255、256、257 byteを直列実行した。全caseでbyte exact、fallback/model/semantic opなし、pre/post health正常、GPU process・残留child 0を確認した。
- G1 artifact SHA-256は`gfx1030=40a55e8028355dd1b27b26886ccfef6d0b4085569d2656f90e7ebdc2be1a852c`、`gfx1201=69207b19c1146f73258db848fd5da74a25dd0a8e980b090ee09037da0dd2b1f5`。G1 report SHA-256は`gfx1030=053555c411cd821fb1876ffe505c7144197a5f0ab1bd83d561e91eb390bbbb90`、`gfx1201=b42be61d454a773b3b25515bfdfe23d5c117c0ffd5d71075536ed76204c2c1b2`、aggregate SHA-256は`7e04072f7ba69aeae03767adc3c842088a9b1781e5194e1a543aa385f380bd8d`。
- host/H3/G0 aggregate SHA-256は順に`789b2ec80e87aa2b77d8e2bc2207116b500f61d0ffc03faa41e78b0f30b44424`、`6de2c45fb756d4b8d72e968c2e6860a09d085011140d41025d318d4bb27749c2`、`71c5201116d74fdd2749e1ec31c9ee017d52d5024d0c4192f77381c55e1d2b78`。
- Phase 2前半を完了し、H3 required昇格の20回・7日観測は後続開発を止めないfollow-upとして残した。次はQwen3.5-4Bの完全revision/model lock、最初のsemantic op、G2 model sliceを独立計画化する。

[対応する計画](../../../../plans/archive/2026/08/1-10/phase2-h3-g0-model-free-gpu.md)
