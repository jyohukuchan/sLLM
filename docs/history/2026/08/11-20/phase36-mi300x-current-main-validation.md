# Phase 36 MI300X latest-main実機再検証履歴

## 2026-08-21: Phase 36 COMPLETE

- ユーザー決定により、当初計画したGemma/MoE/長時間安定性のconditional extensionをPhase 36のscopeから削除した。
  Sessions A〜Dの全受入条件とfocused integration reviewがPASS済みのため、Phase 36を完了とする。
- 9B、Gemma/MoE、長時間安定性はPhase 36のPASS claimへ含めず、必要なら将来の独立work unitで扱う。
- Hot Aisle VMはユーザーが削除済みである。A〜Dのraw evidenceとartifactはrepository外へ保持し、Phase 36専用SSH key、
  public key、known-hostsはlocal hostに保持したままとする。

## 2026-08-21: Session D PASS

- Qwen3.5-4B BF16/FNUZ FP8をshort-odd、32/32、prefill-long、decode-long、10,001/2の各5ケース、direct token、
  FP16 KV、greedy、3 warmup＋10 measuredで実行した。全10 rowはexact gfx942/HIP-only、fallbackなし、cleanupをPASSし、
  10,001/2 E2E中央値はBF16 `22.556130816`秒、FP8 `22.556528472`秒、両方`[23066,23066]`だった。
- fixed llama.cpp `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`のexact gfx942 wrapperを同一VMで5ケース実行した。
  10,001/2は`0.8512540725`秒でsLLM/llama比`26.4975`。同じupstream revisionだがGGUF bytes/tensor setが異なるため
  E1に限定した。最初のfocused runでbackend解放前のHIP allocator cacheをcleanup failureと誤判定したため、wrapperは
  backend release完了、親runnerはprocess終了後sysfs baseline一致を権威とするよう修正し、全5 rowを再実行した。
- BF16 10,001/2 rocprofv3のdevice totalは`22.539747157`秒で、GDN `73.95%`、Full Attention `25.12%`、projection
  `0.70%`、other `0.23%`だった。host wallは`35.096660866`秒、kernel外は`12.556934578`秒。明白なfallback、provider
  誤選択、resource defectはなく、性能最適化はSession D blockerにしなかった。
- 全raw、診断run、binary/model/source identityはrepository外へ退避した。最終summary/schema SHA-256は
  `5d05db578fc6466c4dfcf355efde9cd04b0b07567300f882a24703b31bb19214` /
  `1ce037012e128750021f7323735d752f03e57b66fc6be1f3ff86799838867cbb`である。GPU process 0、HBM/GTT baseline、
  RAS CE/UE 0を確認し、provider ROCm 7.2.4へ復元した。このPASSによりA〜Dの実行範囲を完了した。

## 2026-08-21: Sessions B/C PASS

- Session BはFP16/dynamic FP8/static FP8/NVFP4 Full Attentionを各29 case（116/116）、FP16 KV state 19/19、
  独立NumPy low-bit oracleをexact gfx942でPASSした。canonical 4B BF16 GGUFのFP16 KVとdynamic FP8 KVをautoおよび
  512/2,048/4,096/8,192/16,384 token chunkで合計12 row実行し、全て10,001 input / 2 output、入力ID
  `23066`×10,001、生成ID`[23066,23066]`、HIP-only、fallbackなし、cleanup 0だった。後続MTP修正後の最終CLIでも
  FP16/FP8 autoをfocused rerunした。
- Bのauto/16,384指定arenaは`5,278,049,280` bytes、512指定は`270,209,024` bytesだった。request stateはFP16
  `379,289,600` bytes、dynamic FP8 `217,961,216` bytes。gfx942の`contiguous-resident`を維持してHBM/GTTを
  物理観測し、全row終了後はHBM `299,687,936` / GTT `22,695,936` bytesへ復帰した。VMM provider変更は行っていない。
- Session CはBF16 target＋FP16 KVのMTP target-only/width 2/3/4/7/8と、FP8 target＋dynamic FP8 target KVの
  target-only/width 3をPASSした。BF16 MTPのproposalは14/21/28/49/56、全rowでaccepted+rejected=proposed、
  visible 16 tokenは対応target-onlyへ一致した。FP8 targetのMTP side pathはBF16 weights＋FP16 KVとして明示した。
- 公開MTP経路のgfx1201/width 1固定、terminal state capacity不足、quantized GGUF plan schema拒否を修正した。
  forced width 1〜8、exact gfx942、bounded slack、MTP counter/reportをhost testと実機focused rerunで固定した。
- PNG/JPEG/WebPは各64 image-pad tokenと生成ID`[760,1156,6587,264]`を一致させ、serverのfirst-image lazy
  residency、second-image reuse、shutdown後baselineを確認した。OpenAI profile v1はraw/official clientの
  non-stream/SSE、reasoning、stop、seed、1023/1024/1025、HIP dispatch後cancel/recovery、二並行queue、graceful
  shutdownをPASSした。metricはprovider `partition`例外のため`unavailable`のまま保持した。
- B/C rawはrepository外の
  `/home/homelab1/.local/share/sllm-evidence/phase36/session-{b,c}/enc1-gpuvm015-2026-08-21/raw`へ退避した。
  最後にGPU/sLLM process 0を確認し、ROCm 7.14 bindを解除してprovider既定の`/opt/rocm-7.2.4`へ復元した。
  Session D、9B、repeated performance、llama.cpp/rocprofv3比較、Gemma/MoE/安定性はこの時点では未実行である。
- 最終candidate source digestは`f07b31c9a83aee326c62de3c2f0d1d2da8ff189a66085526ddf79edad2bdf94a`である。
  [Session B summary](../../../../../ci/matrix/phase36-mi300x-session-b-summary-v1.json) /
  [schema](../../../../../ci/schema/phase36-mi300x-session-b-summary-v1.schema.json)のSHA-256は
  `13e4d86859191dbadae66e940bd3adfd8e1ec598fa8dba627de8f3581f6bf274` /
  `f3f0f6204655b646805b33155cd347243699367084936e4b37f8298d91dcbfce`、
  [Session C summary](../../../../../ci/matrix/phase36-mi300x-session-c-summary-v1.json) /
  [schema](../../../../../ci/schema/phase36-mi300x-session-c-summary-v1.schema.json)は
  `4fdc5e4f029e097721b2bc1dfb40b0f51282c268dc55fcbbeb4a7c66073c42f5` /
  `4ebb1d0f76a570d7b2d624a4d9f0c05aabe00e05d951dd4c2bf1533b3db20fc0`である。

## 2026-08-21: Session A final auditとfeature-pinned再実行

- closeout監査で、初回artifactのdevice bundleがbare `gfx942`、ELF flags `0x54C`（SRAM ECC/XNACK any）であり、
  A1のfeature固定を満たさないことを検出した。logical runtime target `gfx942`とcodegen targetを分離し、CDNA3だけ
  `gfx942:sramecc+:xnack-`へ固定した。標準的な`/opt/rocm` symlink配置でもCMakeへlogical rootを渡すようbuild scriptも
  修復した。
- 最終CLI/server/9 evidence artifactの全11本は、唯一のdevice bundle
  `hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`、Code Object V6 contractのABI version 4、ELF flags
  `0xE4C`（SRAM ECC on / XNACK off）、全kernel wave64を満たした。generic/別targetはなく、`gfx1201`指定は
  device arch mismatchとしてdispatch前にexit 1で拒否した。
- 最終artifactでtiny `41→42`、FNUZ hipBLASLt 8 solutions、rocprofv3 kernel/allocation trace、99/99 operatorを
  再実行した。operatorのfamily別case数は`2/17/21/8/19/16/6/7/3`、producerに基づく実dispatch数は
  `4/17/21/8/19/16/6/7/6`、summary SHA-256は
  `5daa5869932513490c50cbb9ff330cf47fb581aa333fc1133fc0261a1192222d`である。
- BF16/FP8のverify、Hello、Unicode、stopも再実行し、既存token oracle、HIP-only、fallback/partial offloadなし、
  cleanup 0を維持した。3 warmup + 10 measuredにcorrectness controlを加えた各14 requestではmodel load 1回、reuse true、
  resident/peakはBF16 `8,411,592,192`/`8,477,011,968` bytes、FP8 `4,847,029,760`/`4,912,449,536` bytes、
  model drop後0だった。
- postはsLLM/GPU process 0、model handle 0、GPU/VRAM use 0%、全sysfs RAS block CE/UE 0を確認した。
  ROCm 7.14 bind mountを解除しprovider `/opt/rocm-7.2.4`へ復元した。最終raw 80 fileとmodel lock 2 fileは
  `/home/homelab1/.local/share/sllm-evidence/phase36/session-a/enc1-gpuvm015-2026-08-21/final`へhash一致で退避した。
  VM/keyはSession B用に保持し、ownerをHot Aisle account/local operator、review期限を2026-08-28 23:59:59 JSTとした。
  commit/pushは行っていない。
- 下記の初回A0〜A5記録は問題発見経緯として保持するが、artifact/operator/raw digestはこのfinal auditを正とする。
  最終summary SHA-256は`9e39c0aba7bd1a11725b95df0e15f6a5728cbde2e57ec250d07bc0432ca27dd4`、strict schemaは
  `b00dc2494f4aa7fe21cd27c2ab6f1e2627a5b13e1875fa1e14a2ed5d052c8def`である。

## 2026-08-21: Session A A4〜A5 PASS

- canonical Qwen3.5-4B GGUFをVMへ転送し、BF16
  `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675` / lock
  `425151d06832347a01b946b27336ceffac074eb7f6932af61e8c9821edc1e318`、FP8
  `cf143f6c138f0e4a6372959bf348568159278202eca6081ce29346fdef1cfe0d` / lock
  `21b4fed31b6cf00e79e74b464f7ff8422d02292872c43d84924fa47e228e68d1`へ一致させ、転送後はmode `0400`にした。
  両modelの`verify-model`は426 loadable / 738 weight entryでPASSした。
- BF16 `Hello` greedy 5は`[11,353,2688,4313,310]`、FP8は`[11,353,1044,4313,310]`だった。
  両方ともexact `gfx942`、HIP-only、fallbackなし、partial offloadなし、cleanup 0で、FP8は`native-fnuz`と
  `e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32`をreportした。Unicode chatは両dtypeとも
  `[90700,8340,25,271,16]`、stop commaはgenerated `[11]`、visible `[]`、matched `,`で一致した。
- BF16/FP8の3 token目差をloader corruptionや非決定として扱わず、同一BF16 repeat、scalar matmul、wave32 matmul、
  wave32 RMSNorm、Phase 28 GDN順序のbounded A/Bで切り分けた。gfx942 BF16は意図したwave64 BF16 reduction、FP8は
  hipBLASLt FNUZを使うcross-provider N1差であり、同一real-number式と既存oracle toleranceを満たす。bit-exactな
  cross-dtype token gateへは昇格しない。
- 切り分け中に、Phase 29でexact gfx1030/gfx1201だけへ承認したGDN wave32 treeが共通sourceからgfx942へも
  適用されていたtarget-scope漏れを検出した。wave64 buildだけPhase 28の128項逐次norm和を維持し、RDNAのPhase 29/35
  provider、kernel symbol、ABI、dispatch/resourceを変えない修復を行った。gfx942 GDN token 1/3/17のfocused rerunは
  3/3 PASS、最大絶対/相対誤差`0.00390625`/`0.014705882`、state一致、fallback/cleanup 0だった。
- A5 post観測はsLLM/GPU process 0、model file handle 0、GPU use 0%、VRAM used 299,687,936 bytesでbaselineへ復帰し、
  ECC correctable/uncorrectableは0だった。request/workspace cleanup、retryable/durable quarantineも全実行で0である。
  post process/static/metric/rocm-smi raw SHA-256は順に
  `103b5abc1190fd28e82bbd5b037a8061cd6a4b4be101f70b0d8141db2878b9d5`、
  `74226562c56ae314c1135280270d4a46048062417591d081660980eab2d6ba6e`、
  `9b1333808b792ec27c31f726383e3209580df1b32b11e9604e1d8c30b27ddd7a`、
  `24114eb48d78fa2550f699e4d0ee979a9237641a1016db708f596ae76caf7699`。
- GPU実行終了後にROCm 7.14 bind mountを解除し、provider既定の
  `/opt/rocm -> /etc/alternatives/rocm -> /opt/rocm-7.2.4`へ復元した。VMとPhase 36専用SSH keyはSession Bへ
  継続できる状態で保持し、Sessions B〜Dはまだ実行していない。Session AをA0〜A5 PASSとして完了した。
- 最終結果は[Session A final summary](../../../../../ci/matrix/phase36-mi300x-session-a-final-v1.json)へ固定した。
  summary SHA-256は`9b2b6a6d1b1a8f53a3d1b80753640117dfb2e65d3936229199ed8484da0e412e`、
  [strict schema](../../../../../ci/schema/phase36-mi300x-session-a-final-v1.schema.json) SHA-256は
  `8e18a61d70e39dc2ddb2116c3de1b3a5e379bc9c7a3017deb2b9e7f80a54b88d`、最終native source SHA-256は
  `360a9e3330104bae4bed7164f105c8c51ea644a1ad779a7a03746f59669623ef`である。

## 2026-08-21: Session A A0〜A3 PASS

- Hot Aisle VM `enc1-gpuvm015`へhost-key fingerprint
  `SHA256:1CqzHeymzgO+N4ot01vQ5lTVDuZi+fiB/OEcKDwPvvg`を照合して接続した。Ubuntu 24.04.4、kernel
  `6.8.0-124-generic`、amdgpu `6.16.13`、MI300X VF x1、BDF `0000:ff:00.0`、UUID
  `GPU-1228c84fe776f2f4`、`gfx942:sramecc+:xnack-`、wave64、304 CU、205,822,885,888 bytes HBM、NPS1/SPX、
  VMM=trueを確認した。foreign GPU workloadとECC uncorrectableはなかった。`amd-smi metric`はprovider側
  `Namespace.partition`例外のため`unavailable`とし、0へ置換していない。
- provider driverとROCm 7.2.4を削除せず、ROCm 7.14.0/LLVM 23 user-spaceを追加した。production build中は
  `/opt/rocm`を追加rootへbind mountし、HIP runtime `7.14.60850-0000000`、rocprofv3 1.3.2、hipBLAS/hipBLASLtを
  同じrootへ閉じた。tiny `41→42`は1 dispatch、2 copy、1 allocation、FNUZ hipBLASLt solution queryは8件、
  rocprofv3 kernel/allocation traceもPASSした。driver交換、reboot、package削除はない。
- source baseはcommit `faf39339d42c837c1ff899f90b03632ac5fe57af`である。Phase 36の9 semantic source/runner fileを
  内容hashへ結合したcandidate digestは`fa2c82c936f61c897c87cee82cb92b0aa100cb0b0c766734e42d24c8df2bc892`で、
  exact `gfx942`、Code Object V6、wave64 release buildを作成した。CLI、server、9 evidence binaryのoffload bundleは
  すべて`hipv4-amdgcn-amd-amdhsa--gfx942`だけを含み、別target/generic bundleはなかった。wrong-target
  `gfx1201`起動はdevice arch exact mismatchでdispatch前に拒否した。
- current public FP8 GGUF経路がgfx942を拒否していたため、exact gfx942を`native-fnuz`へrouteした。レビューでGGUF内の
  OCP E4M3FN byteをdtype labelだけFNUZへ変える欠陥を検出し、BF16/F32 scaleをFP32へ正規化して全有限OCP値を
  E4M3FNUZへrebaseするresident uploadへ修正した。gfx1201 OCP pathは維持し、unsupported target/dtypeはfail closedにした。
  core 186 unitと全integration、CLI/server全testをPASSした。
- Phase 12相当の99 operatorを一つのbounded runnerへ固定した。FNUZ FP8 2、BF16 17、elementwise 21、attention
  preprocess 8、KV state 19、Full Attention 16、output gate 6、RMSNorm 7、GDN 3の全99 caseが独立oracle、
  native HIP dispatch、fallbackなし、cleanup zeroでPASSした。RMSNormはwidth 1/3/255/256/257/2560/4096を
  独立BF16 byte oracleへ一致させ、wave64 kernel id 2とresource count 3 allocation/3 copy/1 dispatchを確認した。
  bounded summary SHA-256は`245761fae98488a98151bcbf72d49a7bc20ad8bb5c8acbdab3b3f8ade19a6cfc`である。
- runner初回はproducerの肯定field `no_fallback=true`をfallback使用と誤分類して64/99集約となった。raw producerは
  KV/Full Attentionを含めPASSしており、validatorを修正してhost test 9/9後に全99を再実行したため、GPU numerical retryや
  candidate failureには数えていない。A0〜A3をPASSとし、固定4B BF16/FP8 GGUFのhash照合とA4短生成へ進む。

## 2026-08-21: Phase 36開始とaccess準備

- ユーザーの明示指示によりPhase 36を開始し、Session Aのlocal/access準備へ移った。この項目は開始時点の記録であり、
  その後のVM/GPU実行とsource修正は上のA0〜A3記録へ継続した。
- repository外にPhase 36専用の短命ED25519 SSH keyを作成した。public fingerprintは
  `SHA256:RCRizUoQSKknYNhx69EI2pjbDPohqLSGLjU8f9ZqM4U`で、private keyはlocal hostだけにmode `0600`で保持する。
  VM作成時はこのpublic keyだけを登録し、endpoint、remote user、VM側Ed25519 host-key fingerprintを照合してから接続する。

## 2026-08-20: 計画作成

- ユーザー指示により、Phase 35後のlatest mainを単一MI300Xで再検証し、問題があれば修正するPhase 36を計画した。
- 課金GPU sessionをA〜Dへ分割した。Session Aはidentity、ROCm/artifact、Phase 12相当99 operator、
  Qwen3.5-4B BF16/FNUZ FP8短生成までを2〜3時間、上限4時間で実行する詳細計画とした。
- Session Bはlow-bit KV/chunked prefill/10k+、CはMTP/vision/OpenAI service、Dはperformance/llama.cpp/profileとした。
- この時点では計画のみであり、VM作成、credential作成、GPU実行、production source修正は開始していない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase36-mi300x-current-main-validation.md)
