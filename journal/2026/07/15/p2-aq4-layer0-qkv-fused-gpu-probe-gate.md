# P2 AQ4 layer-0 QKV/Z/gate/beta fused GPU probe gate

## 前回の要点

fused probe report schema v2とCPU診断は既存実装で完了していた。HIP GPU、service停止、holdout、数値Go/No-Go、promotionは未実施である。

## 今回の変更点

- clean detached worktree `6082df4966190ae4977b699460a5ecb93fee8e34`から作成したprobe artifact（binary SHA-256 `42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840`、receipt SHA-256 `90e9ef6d383f7ef25e9526659f035e40291ba1a5efa7f8ba36340c8b245d9504`）は前回のまま固定した。
- 今回はgateの観測と復旧を強化した。lock取得後、物理card 2かつgfx1201を検証するamd-smi JSON observerを別プロセスで開始し、初回sample/failure markerを待ってからprobeを実行する。終了時はobserver停止、failure marker不在、sample存在を検証する。observer子プロセスへ親の終了シグナルtrapを継承させない。
- 失敗trapはobserver停止、lock cleanup、service startを独立した戻り値で保持し、lock cleanupが失敗してもservice startを必ず試行する。start後にhealth/GPU/worker操作は実行しない。
- 出力契約はHIP standalone QKV reference（operation `standalone_aq4_matvec_f32`、raw RPB unset、effective 32、source `architecture_default:gfx1201`）、Q/K/V row segments、全sidecarのshape/bytes/rows/case offset/SHA/finite、output layout、report/sidecarのnlink=1までfail closedで検証する。promotionは常にfalseである。
- execute準備ではfresh BASEの`attempts` parentを作成した後、attemptディレクトリをcreate-newで確保する。source checkoutのmodeを信頼せず、probeをattempt内へinstall/chmod `0555`し、runtime copyのregular/nlink=1/SHA-256を検証して、runtime binaryのstat/hashを`runtime-probe-stat.json`へarchiveする。実行対象はこのruntime copyだけである。

## 検証

- 実行: `bash -n`、binary/receipt SHA256SUMS検証、fresh BASEの`MOCK_ARCHIVE_SETUP=1`（source checkout mode `0775`からruntime mode `0555`へ変換するケースを含む）、`PREFLIGHT_ONLY=1`、`MOCK_PREFLIGHT=1`、6 tests（observer mock、runtime copy archive、read-only mock、validatorのwrong-reference/layout negativeを含む）。
- 未実施: GPU kernel probe、fused report/output、health実測、holdout、数値閾値判定、promotion。service stop/startは下記の実行で復旧確認済みである。

## 実行結果（2026-07-15）

- 通常EXECUTEは2回試行した。attempt1はsudo非対話認証不足でservice確認前に終了し、`attempts/attempt1/`へruntime copyと`runtime-probe-stat.json`（runtime SHA-256 `42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840`、stat SHA-256 `9d1855b3f5f8b1375708b25e50a9fd3f90471a883046b06dedcaa96ffc97e9c6`）だけを保存した。
- PTY `sudo -v`後のfresh `attempt2/attempts/attempt1/`はservice stop→stable2停止→lock acquireまで進んだが、observer初回sampleで停止した。実行対象のprobe kernelは未起動で、`run.log`は`observer failed before first sample`、observer failure markerが残った。`amd-smi --showmeminfo vram --showuse --showpower --json`は、実機AMD-SMI 26.2.2/ROCm 7.2.1のsubcommand式CLIと不一致で`AmdSmiInvalidSubcommandException`になった。
- 失敗trapの復旧は完了した。serviceはactive/running、旧MainPID `1834403`から新MainPID `2381944`、workerは`1834494`から`2382329`、`NRestarts=0`である。`/run/ullm/r9700.lock`はuid/gid `1000:1000`、mode `0600`、nlink `1`の新inode `770421`をservice workerが保持している。GPU kernel、numeric threshold、promotionは実施していない（promotion=false/unclassified固定）。
- attempt2の証跡はruntime stat SHA-256 `aeae99d8b4c06d16570f3b119d836a7995fe6f27e7e7c18c291c6244d77ab9a9`、prestate SHA-256 `fab7cebdcbb8304528fa53fec9727fc6612bfe7e8409d77e353be3b44ed88fb5`、run log SHA-256 `eb00c14a402386228ca5dbf656ffaf0271e78e47dcde2e0f16bfa4edf21b04a8`である。
- attempt2の初回sample失敗を根拠に、実機のread-only CLI capabilityを再確認した。`amd-smi version`はAMD-SMI `26.2.2+e1a6bc5663`/ROCm `7.2.1`、metricは`amd-smi metric -g 2 -m -u -p -t --json`、gfx/ASIC identityは`amd-smi static -g 2 -a --json`で取得できる。単発metric JSONはgpu=2のusage/power/mem_usageを、static JSONはgpu=2・`target_graphics_version=gfx1201`を返した。
- observerを現行subcommand式へ最小修正し、metric/static両JSONの単一gpu=2、usage/power/mem_usage、gfx1201をfail closedで検証してmonitorへ保存する。mockでは初回成功、初回失敗伝播、observer停止を検証した。既存attemptは保全し、GPU kernel/serviceの再実行は行っていない。既定attemptが既存のため、fresh BASEでPREFLIGHT_ONLY/MOCK_PREFLIGHTを再確認した。

## attempt3 実行結果（2026-07-15）

- fresh `attempt3/attempts/attempt1/`で、PTY `sudo-v`→`PREFLIGHT_ONLY`成功→通常EXECUTEを1回行った。modern observerは初回sampleを取得し、card2/gfx1201、metric usage/power/mem_usage、static ASIC identityを保存した（monitor SHA-256 `ed24965e05b6da58503fe2372eb5407e68e01725b0565ea523d0325f725ac83a`）。
- probeはHIP device1/gfx1201、visibility `1/1`、両kernel guard、fused RPB4、standalone effective RPB32/default、Q/K/V segments、5 sidecarsのshape/rows/case/finite/SHA/input consumed SHAを満たし、`fused_report=valid ... promotion=false`を出力した。report SHA-256は`458f998116be75b1c363ec49965d55fbfbf286d7c6ecbf3e1158df3ccabe547c`、sidecar SHA-256はqkv `d6a238abc5cd8ac7dd687b7c86dba48e25327a754425bb39ec671f3d01e33a03`、qkv-standalone `24248fd1c4b4b7186f9b048a7fa4c69925904a04b265a273390089df7312545e`、z `0785b49c69c8a6ead41c7905e6ef1f07f5708cad3dbd4f82f81f2617b3ec502e`、gate `49e42d6d6ea71af7915a13cd81b462c6882064912cb048ae755338fe798863e6`、beta `47844e8b1fb2b3dfefa2df8dcf1b489ed2d2a98c86ce0b32f7ef2e27d63d2302`である。
- probe/output validator後のlock cleanupとservice startは実行されたが、Gate全体の終了コードは`1`になった。失敗箇所は最終post-start checkである。直後のread-only確認ではservice active/running、health HTTP 200、MainPID `2442053`がlock inode `771003`を保持、worker `2442481`、`NRestarts=0`、lock mode `0600`/uid/gid `1000:1000`/nlink `1`を確認した。numeric threshold、holdout、promotionは実施していない（classification unclassified、promotion=false）。runtime stat SHA-256は`d09bcdf35a2cf8e8054ecd074f7516b598ae541e687410ae31ab7e271b6b1863`、prestate SHA-256は`135ba31a8eec0f4d34acbdb344922ac2ac54087a0b9c61cd9a79a8c912ec90e4`である。

## attempt3 閾値なし数値比較（2026-07-15）

- attempt3 GPU reportとCPU formal reportはschema v2、dtype `float32`、shape `[4096]`、rows `3`、Q/K/V row segments、output layout、input sidecar SHA/consumed SHAが一致することを比較ツールで確認した。canonical pathとinodeは実行環境が異なるため一致しないが、semantic input identityは一致している。GPU reportは`458f998116be75b1c363ec49965d55fbfbf286d7c6ecbf3e1158df3ccabe547c`、GPU sidecarはqkv `d6a238abc5cd8ac7dd687b7c86dba48e25327a754425bb39ec671f3d01e33a03`、qkv-standalone `24248fd1c4b4b7186f9b048a7fa4c69925904a04b265a273390089df7312545e`、z `0785b49c69c8a6ead41c7905e6ef1f07f5708cad3dbd4f82f81f2617b3ec502e`、gate `49e42d6d6ea71af7915a13cd81b462c6882064912cb048ae755338fe798863e6`、beta `47844e8b1fb2b3dfefa2df8dcf1b489ed2d2a98c86ce0b32f7ef2e27d63d2302`である。CPU formal sourceは`/tmp/ullm-fused-cpu-followup6/`、report SHA-256は`7876ee835c25c2b312c4f288e40a559883bdc875ff5026363f019bd339a5085c`、sidecar SHA-256はqkv/qkv-standalone `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473`、z `7ed98f1c7f8988958377b548f44afe3a2ddc5180150d1e3191c7d0e2a408b286`、gate `dbf470352abb0bbe31e23018d5770608a424048f83191f3e063360f6ba857857`、beta `ed4a3a57629fddf561f4f115f5b598a59a10984e579d5a8bff23dbaf0478bf64`である。
- 比較成果物は`benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-fused-gpu-probe-v0.1/attempt3/comparisons/attempt3-vs-cpu-formal-v1/comparison.json`（SHA-256 `6e171794f95e3327a6a546a5d50e91dd34718c84a9894ebcfceacbb93b2088f5`）であり、比較ツールは`tools/compare-aq4-layer0-fused-attempt3.py`（SHA-256 `982db13090732fc26093026db18953a854d963a888035f4d2c035308b854a236`）である。比較は固定閾値を持たない記述的な測定で、`thresholds=null`、`policy_decision=not_evaluated`、holdout/promotionは未実施である。
- 集約値（全ペアでnonfinite=0）は次のとおりである。GPU fused QKV対GPU standalone QKVはbyte mismatch `16285/98304`、max abs `3.814697265625e-06`、relative L2 `7.107261355045151e-08`、cosine `0.9999999999999886`。GPU fused QKV対CPU formal QKVは`25934/98304`、`5.7220458984375e-05`、`8.815654484874577e-07`、`0.9999999999995889`。GPU standalone QKV対CPU formal QKVは`25939/98304`、`5.91278076171875e-05`、`8.828352279004388e-07`、`0.9999999999995917`。GPU z対CPU formal zは`12421/49152`、`2.86102294921875e-05`、`9.195597723437779e-07`、`0.9999999999995598`。GPU gate対CPU formal gateは`98/384`、`1.5795230865478516e-06`、`8.724489477997521e-07`、`0.9999999999996194`。GPU beta対CPU formal betaは`67/384`、`4.76837158203125e-07`、`1.9279977243837735e-07`、`0.9999999999999825`である。これらは数値傾向の記録であり、合否・promotionの閾値判断には使用していない。

## post-start readiness の期限付き再試行（2026-07-15）

- `run-fused-gpu-probe-gate.sh`のpost-start判定を、固定120回ループから絶対monotonic deadline（既定120秒）内の全predicate再評価へ変更した。各attemptはservice active、SubState running、新MainPID（旧PIDとの差分を含む）、NRestarts不変、active/package/worker SHA tuple不変、lockのregular/nlink=1/mode=0600/uid=1000/gid=1000、flock保持、MainPIDのlock fd owner、health HTTP 200を順番どおり評価し、途中の一条件の失敗で残りを省略しない。
- `$ATTEMPT_ROOT/post-start-readiness.jsonl`へattemptごとの条件結果とfailure reasonを追記し、成功またはdeadline timeout時に`post-start-readiness.json`（schema `ullm.aq4_layer0_qkv_fused_gpu_post_start_readiness.v1`）をcreate-new保存する。成功artifactはattempt数・elapsed、timeout artifactは期限・attempt数・最終failed subcondition理由を保持し、追加GPU/probeを実行しないことを`safety`へ記録する。health predicateはcurlのHTTP statusを厳密に`200`へ固定した。
- GPU/serviceに触れない`MOCK_POST_START_READINESS=1`経路を追加し、ownerとhealthの初回失敗後の成功（2 attempts）、期限切れrc1、timeout後の追加GPU/probeなし、epoch/hash/lock predicate保持をテストで確認した。`bash -n`、fused/standalone gate tests（12 tests）、post-start test、`git diff --check`を実行した。GPU kernel実行、service stop/start、systemctl実測、promotionは行っていない。

## post-start readiness 独立監査修正（2026-07-15）

- JSONLはshellのtruncate作成を廃止し、Pythonの`os.open(O_WRONLY|O_CREAT|O_EXCL, 0644)`と`fchmod(0644)`でcreate-newする。各attemptは`O_WRONLY|O_APPEND|O_NOFOLLOW`で開き、regular file、nlink=1、mode=0644をfd上で再検証してから追記する。既存pathはpreflightとcreate-newの両方で拒否し、既存内容を変更しない。
- `POST_START_POLL_SECONDS`はfiniteかつ正数だけを受理する。timeoutは1〜3600秒、attempt上限は1〜512（既定512）へfail closedで制限し、上限到達時は`attempt_limit` artifactを保存する。各attemptは開始前と完了時にabsolute deadlineを検査し、health curlのconnect/max timeoutは実行直前のdeadline残時間と5秒の小さい方へ拘束する。
- 既存JSONL拒否・非正数/nonfinite poll拒否・attempt 2件上限をmock testsへ追加した。fused/standalone gate testsは15件すべて成功し、`bash -n`、`py_compile`、fresh mock preflight、`git diff --check`を実行した。GPU kernel、service、systemctlは実行していない。

## 次の行動

attempt3のGPU/output/observer/復旧証跡、閾値なし比較成果物、監査修正済みの期限付きpost-start readiness診断を限定commitとして保存する。GPU/serviceの再実行・promotionは行わない。
