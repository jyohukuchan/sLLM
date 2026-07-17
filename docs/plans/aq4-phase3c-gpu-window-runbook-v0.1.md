# AQ4 Phase 3c GPU window runbook v0.1

## 目的と承認境界

これはAQ4 fidelityのPhase 3cを、**ユーザーがGPU windowを明示承認した後だけ**一回実行するためのrunbookである。Phase 3c-prep中にこの手順、GPU binary、GPU probeは実行しない。

目的は、07/14のdirect M=1/cold diagnosticと同じ3 rowについて、CPU layer 0 referenceとproduction HIP M=1 `dispatch_token_for_phase -> run_device_step`のlayer 0 device bufferをstage単位で比較し、最初の有意差を特定することに限る。kernel修正、service操作、manifest変更、P3 harnessの再開は含めない。

この手順はserviceを停止・起動・再起動せず、systemd、active manifest、P3 harnessを変更しない。trace binaryは既存の`/etc/ullm/served-models/active.json`を**read-onlyでhashしてidentityに記録するだけ**である。lockが利用できない場合、GPUを使わず失敗として終了する。GPU windowの外部調整はこのrunbookの権限外である。

追加の承認条件として、traceの前にR9700対象だけを機械的に確認する。`tools/query-hip-device-identity.cpp`をhost-onlyでbuildし、`HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1`のfiltered HIP ordinal 0について、可視GPU数、architecture、name、PCI BDFを読み取り専用で取得する。このtoolはdevice memoryの確保、stream作成、kernel launchを行わない。`gfx1201`でない、可視GPU数が1台でない、name/BDFが欠ける場合は終了し、traceを起動しない。続けて、そのHIP BDFを明示した`amd-smi --gpu "$R9700_BDF"`だけでASIC identityを取得し、同じBDF・`gfx1201`・R9700のPCI device ID `0x7551`をassertする。`amd-smi list`、対象指定なしの`amd-smi`、V620を対象とする問い合わせは使用しない。

H9（ハードウェア固有要因）のため、同じBDFだけを対象にECC/error block、bad page、clock、power、temperature、DPM performance level、driver/IFWI、firmwareをread-onlyで実行前・実行後に保存する。利用不可のmetricはexit codeとstderrを保存して「未取得」と記録するが、設定変更やmonitoring daemonの導入は行わない。単発windowで決定性や他GPU比較を検証しない。

## 固定するidentityとfixture

| 項目 | 固定値 |
| --- | --- |
| trace tooling source | `5a0fb4c50476d5153ced22bd6847c2729bfdb975`（stage tooling、failure evidence保持、manifest guard set記録を含む） |
| package | `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package` |
| cases | `tests/fixtures/qwen35-aq4-p2-oracle/cases.json` |
| replay | `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/differential-trace-gpu-v1-input/replay.json` |
| CPU hybrid input | `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-hybrid-diagnostic-v0.1/input/hybrid-input.jsonl` |
| R9700 mapping | physical card 2 → `HIP_VISIBLE_DEVICES=1` → filtered HIP ordinal 0 → global runtime device index `1` → `gfx1201` |
| R9700 HIP guard source | `tools/query-hip-device-identity.cpp`（host-only `g++` build、filtered ordinal 0だけをquery） |
| R9700 ASIC cross-check | HIP guardが返すPCI BDFを`amd-smi static --gpu`へ渡し、`gfx1201` / `0x7551` / non-empty nameをassertする |
| H9 telemetry | 同じBDFだけに`amd-smi metric`（ECC/clock/power/temperature/DPM）、`bad-pages`、`static`（driver/IFWI）、`firmware`をread-onlyで実行前・後に保存する |
| lock | `/run/ullm/r9700.lock`（既存regular fileだけをnonblockingで取得） |
| result root | `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1` |

比較するcontextは、CPU/GPUとも同一の次の3 rowである。GPU traceは各rowの最終tokenだけを保持するため、CPU streamも同じtimestepだけを比較器が選ぶ。

| case / step | token context | context length | `context_token_ids_sha256` |
| --- | --- | ---: | --- |
| `fixture-prompt-0` / 0 | `[11,12,13]` | 3 | `42ea52c728680a54afafd1c1e1e45f13300c3ceb962f320f3900196a0c46215c` |
| `fixture-prompt-0` / 1 | `[11,12,13,220]` | 4 | `6af1601b9bf35d095b24c5bac3a95a01bf77d047b576441d0a5f9510eec66249` |
| `fixture-prompt-1` / 0 | `[21,22]` | 2 | `3bca9e21e3b6f741ed412f91d7696146c254ff68bd9be9ca41b1d172eb3549e6` |

stage順は`qkv_dequant_row_scale`、`z_dequant_row_scale`、`recurrent_gate`、`recurrent_beta`、`recurrent_state_after`、`recurrent_output`、`attention_residual`、`post_norm`、`mlp_activation`、`layer_output`である。full f32 tensorを比較し、summaryには固定座標`0,1,31,127,1024,2048,4095,last`だけを残す。

## 実行前の確認

- ユーザーがこのPhase 3c GPU windowを明示承認していること。
- active service、systemd unit、active manifestに対する変更操作をこの手順に追加しないこと。`systemctl`、`service`、`kill`、manifest writeは実行しない。
- `/run/ullm/r9700.lock` がregular fileとして存在し、実行時にnonblocking取得できること。busyまたは欠損なら、その時点で終了する。lockを作成・修復・待機・再試行しない。
- 07/16停止中P3 harnessのpath、script、output、環境変数、`rocprof`を使用しない。出力は上表の独立した`p2/aq4-phase3c-gpu-stage-trace-v0.1`だけに書く。
- 既存output root、`gpu-trace`、`cpu-reference`、comparison outputが存在しないこと。既存evidenceを削除・上書きしない。
- RPBはprocess起動前に固定する。`ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4`、`ULLM_AQ4_MATVEC_SILU_MUL_RPB=8`、`ULLM_AQ4_MATVEC_ADD_RPB=8`をtrace childだけに与え、実行中に変えない。これはcompile-time RPBとlaunch-time RPBがずれる既知の条件付きcache bugを除外するためである。
- `tools/query-hip-device-identity.cpp`がtrackedかつHEADに対してcleanであり、host-only `g++`と`amd-smi`が利用可能であること。guardは`HIP_VISIBLE_DEVICES=1`で可視化されたordinal 0だけを問い合わせ、返ったPCI BDF以外を`amd-smi`へ渡さない。
- HIP guardまたはASIC cross-checkが失敗した場合、CPU reference、lock取得、trace binary、比較器へ進まない。guard evidenceだけを保存して終了する。health telemetryはguard成功後だけに採取する。

## 追加承認済みの service 一時停止 window（2026-07-17、今回だけ）

この節は、2026-07-17にユーザーから得た「このマシンに関する全権限」と、AQ4本番 gateway を一時停止して Phase 3c を一回だけ行うという明示承認にだけ適用する。上の「service を停止・起動・再起動しない」という記述は、最初の lock-contention 試行を含む通常の GPU window の安全境界として残す。本節はその境界に対する狭い例外であり、`ullm-openai.service` 以外の unit、active manifest、P3 harness を変更する権限にはならない。

- 操作する service は `ullm-openai.service` だけである。`systemctl restart`、`kill`、`rm`、lock の強制解放、manifest write、V620 を対象にする command は使わない。
- 07/16 に停止した P3 harness の lock/root/artifact/environment/`rocprof` は参照・変更しない。P3 の `prepare_lock_substrate`、recovery、finalize を今回の lock 問題の回避手段として使わない。
- 1回目の evidence root `.../aq4-phase3c-gpu-stage-trace-v0.1/` と、service 操作前に終了した準備用 leaf `service-stop-window-v0.1` は削除・上書きしない。今回の実service windowの `OUT` はその配下の新規 leaf `service-stop-window-v0.2` とし、開始時にその leaf が不存在であることを確認する。
- この unit は `RuntimeDirectory=ullm` かつ `RuntimeDirectoryPreserve=no` であることを、service 停止直前に読み取り専用で記録する。したがって systemd が停止時に `/run/ullm` と lock leaf を削除した場合、これは「free lock」ではなく **lock取得失敗** である。既存regular fileだけを nonblocking 取得する契約は維持し、`mkdir`、`touch`、`install`、symlink 作成その他の lock substrate 作成・修復は行わない。この場合は trace を起動せず、直ちに一度だけ service 復旧へ進む。

### 停止前の read-only snapshot と非GPU準備

GPU を使わない source/fixture 検査、host-only HIP guard build、`cargo build`、CPU input identity、CPU stage stream は service 停止前に完了させる。これにより停止時間を、guard・telemetry・GPU trace・復旧に限る。これらの準備が失敗した場合は service を停止しない。

`OUT` を次に固定する。既存の最初の試行の root は parent としてだけ存在してよく、`OUT` 自体は存在してはならない。

```bash
OUT="$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.2"
test ! -e "$OUT"
install -d -m 700 "$OUT"
```

既存の実行blockにある trace tooling commit/diff、fixture、RPB、fusion guard の全検査をそのまま実行し、`query-hip-device-identity` と2つの trace binary を build する。ただしこの時点では HIP guard を実行せず、GPU health telemetry も採らない。CPU reference の `cpu-input-identity.json`、`cpu-stages.f32le`、`cpu-reference/` をこの `OUT` に生成してから、直後に以下を `service-window-pre-stop.json` として保存する。

- `systemctl show ullm-openai.service` の `ActiveState`、`SubState`、`MainPID`、`NRestarts`、`ExecMainStartTimestamp`、`ControlGroup`、`RuntimeDirectory`、`RuntimeDirectoryPreserve`。
- cgroup 内 PID、各 PID の `exe`、親 PID、`/dev/kfd` と `/dev/dri/` に限った FD target。これを service/worker の read-only snapshot とする。
- `/etc/ullm/served-models/active.json` の SHA-256。
- `/run/ullm/r9700.lock` の `lstat`（regular file / device / inode / mode / uid / gid）と、その device/inode に一致する `/proc/locks` の holder PID。

停止直前の期待baselineは、`MainPID=1218698`、`NRestarts=0`、`active/running`、`ExecMainStartTimestamp=Thu 2026-07-16 18:14:24 JST`、manifest SHA-256=`feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`である。値が変わっている場合は snapshot の実測値を優先して記録する。`active/running` でない、MainPID が 0、manifest が読めない場合は stop しない。

### 単回 stop → lock → trace → restore

`systemctl stop ullm-openai.service` は一回だけ実行し、開始・終了時刻と stop 後の unit state を `service-window-stop.json` に保存する。stop が返った後、**作成を伴わない** lock probe を一回だけ行う。probe は `lstat` で regular file/non-symlink を確認してから `O_RDWR|O_NOFOLLOW`（`O_CREAT` なし）で開き、`LOCK_EX|LOCK_NB` を取得・解放する。probe は lock の内容を書き換えず、結果・errno・device/inode・`/proc/locks` holder を `service-window-lock-after-stop.json` に保存する。

lock probe が regular file でない、path/親component がない、open/flock が失敗する、または holder が残る場合は、待機・再probe・trace を行わない。この一回を lock acquisition failure として記録し、以下の restore 手順へ直行する。特に `RuntimeDirectoryPreserve=no` による ENOENT を free と見なしてはならない。

lock probe が成功した場合だけ、次の順番を厳守する。

1. 既存blockと同一の `HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1` で、事前build済みの HIP guard を一回実行する。同じ HIP BDF だけを `amd-smi static --gpu "$R9700_BDF" --asic --bus --json` に渡し、`gfx1201` / `0x7551` / BDF 一致を assert する。guard failure なら telemetry/trace は実行せず restore する。
2. 既存の `capture_r9700_health before` と同じ、BDF指定済み4 commandを一回採る。対象外GPUへの query はしない。
3. 既存blockの `flock -n` 内 trace command を、この新しい `OUT/gpu-trace` に対して一回だけ実行する。固定fixture、3 context、RPB、7 fusion guard、`HIP_VISIBLE_DEVICES=1` は変更しない。lock probe 成功後であっても、この trace 内 `flock -n` が失敗した場合は trace command を起動せず、retry せず restore する。
4. trace の exit code にかかわらず、既存の `capture_r9700_health after` を一回採り、直ちに restore する。trace が terminal frame/manifest/SHA256SUMS を満たす成功時だけ、既存の checksum、manifest assert、CPU/GPU 30 record 比較を実行する。

`systemctl stop` が成功した時点から EXIT trap を armed にし、guard、telemetry、trace、comparison のいずれで失敗しても `systemctl start ullm-openai.service` を一回だけ呼ぶ。明示的な restore と trap の二重 start を防ぐため、restore 済み flag を持つ。start の失敗時に restart/stop/start を重ねない。

### restore 後の必須確認

`systemctl start ullm-openai.service` の一回の結果と開始・終了時刻を保存し、その後は読み取り専用で確認する。gateway は host localhost ではなく Docker bridge `172.20.0.1:8000` に bind しているため、実際の healthz は既存の `open-webui` container network namespace から `http://172.20.0.1:8000/healthz` を GET する。最大90回、1秒間隔の health poll は service start の再試行ではない。`/healthz` が HTTP 200 かつ本文 `{"status":"ok"}` を返した時点で終了し、90回で成功しなければ失敗として記録する。`/readyz` の HTTP 200 / `{"status":"ready"}` は worker ready の補助確認として同じ namespace から一回だけ取得する。

次を `service-window-post-restore.json` として保存・判定する。

- `ullm-openai.service` が `active/running`、MainPID が正、`NRestarts` が停止前値から増えていないこと。明示的 stop/start により MainPID/worker PID が変わることは期待どおりである。
- 新しい worker が service cgroup に存在し、対象 worker の `/dev/kfd` FD を確認できること。HIP guard が確認した R9700 BDF だけに `amd-smi process --gpu "$R9700_BDF" --general --json` を実行し、worker PID を owner として記録する。KFD はその worker の `/sys/class/kfd/kfd/proc/<worker-pid>/vram_51545` を読むだけに留め、全GPU/全PID scan はしない。
- active manifest SHA-256 が停止前 snapshot と完全一致すること。
- container namespace の `/healthz` が正常応答すること。`/readyz` または worker/GPU/KFD owner が失敗した場合も復旧確認失敗として扱う。

この復旧確認のどれかが失敗したら、それが最優先の問題である。追加の service 操作、設定変更、trace retry、GPU再実行をせず、収集済み evidence と実際の状態を journal に記録して直ちに報告する。

## 承認後に一回だけ実行するコマンド

次のblockをそのまま一回実行する。`cargo build`、HIP guardのhost-only build、CPU referenceはGPU kernelを起動しない。GPUに対するread-onlyのidentity/health queryはtrace前後にR9700だけへ行い、device memory確保・stream作成・kernel実行は最後の`flock`内のtrace binaryだけである。

```bash
set -euo pipefail
umask 077

REPO=/home/homelab1/coding-local/ultimateLLM/uLLM-project
TRACE_TOOLING_COMMIT=5a0fb4c50476d5153ced22bd6847c2729bfdb975
PACKAGE=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package
CASES="$REPO/tests/fixtures/qwen35-aq4-p2-oracle/cases.json"
REPLAY="$REPO/benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/differential-trace-gpu-v1-input/replay.json"
HYBRID_INPUT="$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-hybrid-diagnostic-v0.1/input/hybrid-input.jsonl"
OUT="$REPO/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1"
LOCK=/run/ullm/r9700.lock
R9700_HIP_GUARD_SOURCE="$REPO/tools/query-hip-device-identity.cpp"
R9700_HIP_GUARD_BIN="$OUT/query-hip-device-identity"

test -d "$REPO"
test -d "$PACKAGE"
test -f "$CASES"
test -f "$REPLAY"
test -f "$HYBRID_INPUT"
test -f "$LOCK"
test ! -L "$LOCK"
test ! -e "$OUT"
test -f "$R9700_HIP_GUARD_SOURCE"
command -v g++ >/dev/null
command -v amd-smi >/dev/null
git -C "$REPO" ls-files --error-unmatch tools/query-hip-device-identity.cpp >/dev/null
git -C "$REPO" diff --quiet HEAD -- tools/query-hip-device-identity.cpp
git -C "$REPO" diff --cached --quiet HEAD -- tools/query-hip-device-identity.cpp
test "$(git -C "$REPO" rev-parse "$TRACE_TOOLING_COMMIT")" = "$TRACE_TOOLING_COMMIT"
git -C "$REPO" diff --quiet "$TRACE_TOOLING_COMMIT" -- \
  crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs \
  crates/ullm-engine/src/qwen35_aq4_model_runtime.rs \
  crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs \
  tools/verify-aq4-layer0-package-embedding-fixture.py \
  tools/compare-aq4-layer0-cpu-gpu-stage-stream.py
git -C "$REPO" diff --cached --quiet "$TRACE_TOOLING_COMMIT" -- \
  crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs \
  crates/ullm-engine/src/qwen35_aq4_model_runtime.rs \
  crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs \
  tools/verify-aq4-layer0-package-embedding-fixture.py \
  tools/compare-aq4-layer0-cpu-gpu-stage-stream.py

install -d -m 700 "$OUT"
R9700_HIP_GUARD_SOURCE_SHA256="$(sha256sum "$R9700_HIP_GUARD_SOURCE" | awk '{print $1}')"
printf '%s\n' \
  "trace_tooling_commit=$TRACE_TOOLING_COMMIT" \
  "package=$PACKAGE" \
  "cases=$CASES" \
  "replay=$REPLAY" \
  "hybrid_input=$HYBRID_INPUT" \
  "r9700_lock=$LOCK" \
  "r9700_hip_guard_source=$R9700_HIP_GUARD_SOURCE" \
  "r9700_hip_guard_source_sha256=$R9700_HIP_GUARD_SOURCE_SHA256" \
  "HIP_VISIBLE_DEVICES=1" \
  "ULLM_HIP_VISIBLE_DEVICES=1" \
  "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4" \
  "ULLM_AQ4_MATVEC_SILU_MUL_RPB=8" \
  "ULLM_AQ4_MATVEC_ADD_RPB=8" \
  > "$OUT/phase3c-preflight.txt"

g++ -std=c++20 -Wall -Wextra -Werror \
  -D__HIP_PLATFORM_AMD__ \
  -I/opt/rocm/include \
  "$R9700_HIP_GUARD_SOURCE" \
  -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib \
  -o "$R9700_HIP_GUARD_BIN"
test -x "$R9700_HIP_GUARD_BIN"
sha256sum "$R9700_HIP_GUARD_BIN" > "$OUT/query-hip-device-identity.sha256"

env \
  HIP_VISIBLE_DEVICES=1 \
  ULLM_HIP_VISIBLE_DEVICES=1 \
  "$R9700_HIP_GUARD_BIN" \
  > "$OUT/r9700-hip-device-guard.json" \
  2> "$OUT/r9700-hip-device-guard.stderr"

R9700_BDF="$(python3 - "$OUT/r9700-hip-device-guard.json" <<'PY'
import json
import re
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["schema_version"] == "ullm.r9700_hip_device_guard.v1"
assert payload["status"] == "valid"
required = payload["required"]
actual = payload["actual"]
assert required["hip_visible_devices"] == "1"
assert required["ullm_hip_visible_devices"] == "1"
assert required["visible_hip_device_count"] == 1
assert required["architecture"] == "gfx1201"
assert actual["hip_visible_devices"] == "1"
assert actual["ullm_hip_visible_devices"] == "1"
assert actual["visible_hip_device_count"] == 1
assert actual["filtered_hip_ordinal"] == 0
assert actual["architecture"] == "gfx1201"
assert isinstance(actual["name"], str) and actual["name"].strip()
bdf = actual["pci_bdf"].lower()
assert re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bdf)
print(bdf)
PY
)"

amd-smi static --gpu "$R9700_BDF" --asic --bus --json \
  > "$OUT/r9700-amd-smi-identity.json" \
  2> "$OUT/r9700-amd-smi-identity.stderr"

python3 - \
  "$OUT/r9700-hip-device-guard.json" \
  "$OUT/r9700-amd-smi-identity.json" \
  "$OUT/r9700-architecture-guard.json" <<'PY'
import json
import re
import sys

hip_path, smi_path, output_path = sys.argv[1:]
hip = json.load(open(hip_path, encoding="utf-8"))
smi = json.load(open(smi_path, encoding="utf-8"))
actual = hip["actual"]
bdf = actual["pci_bdf"].lower()

records = []
def visit(value):
    if isinstance(value, dict):
        if isinstance(value.get("asic"), dict) and isinstance(value.get("bus"), dict):
            records.append(value)
        for child in value.values():
            visit(child)
    elif isinstance(value, list):
        for child in value:
            visit(child)

visit(smi)
assert len(records) == 1, f"expected exactly one targeted amd-smi record, got {len(records)}"
record = records[0]
asic = record["asic"]
bus = record["bus"]
architecture = str(asic["target_graphics_version"]).split(":", 1)[0].lower()
device_id = str(asic["device_id"]).lower()
market_name = str(asic["market_name"]).strip()
smi_bdf = str(bus["bdf"]).lower()
assert architecture == "gfx1201", architecture
assert device_id == "0x7551", device_id
assert market_name and market_name != "N/A"
assert smi_bdf == bdf, (smi_bdf, bdf)
assert hip["status"] == "valid"
assert actual["architecture"] == "gfx1201"
assert isinstance(actual["name"], str) and actual["name"].strip()
assert re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bdf)

report = {
    "schema_version": "ullm.aq4_phase3c_r9700_architecture_guard.v1",
    "status": "valid",
    "r9700_identity_basis": {
        "hip_visible_devices": "1",
        "filtered_hip_ordinal": 0,
        "architecture": "gfx1201",
        "pci_device_id": "0x7551",
    },
    "hip": actual,
    "amd_smi": {
        "pci_bdf": smi_bdf,
        "architecture": architecture,
        "pci_device_id": device_id,
        "market_name": market_name,
    },
}
json.dump(report, open(output_path, "w", encoding="utf-8"), indent=2, sort_keys=True)
print(json.dumps(report, sort_keys=True))
PY

capture_readonly_telemetry() {
  local stem=$1
  shift
  local status=0
  if "$@" > "$OUT/${stem}.json" 2> "$OUT/${stem}.stderr"; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" > "$OUT/${stem}.exit-code"
  return 0
}

capture_r9700_health() {
  local phase=$1
  capture_readonly_telemetry "gpu-health-${phase}-metrics" \
    amd-smi metric --gpu "$R9700_BDF" --ecc --ecc-blocks --clock --power --temperature --perf-level --json
  capture_readonly_telemetry "gpu-health-${phase}-bad-pages" \
    amd-smi bad-pages --gpu "$R9700_BDF" --pending --retired --un-res --json
  capture_readonly_telemetry "gpu-health-${phase}-static" \
    amd-smi static --gpu "$R9700_BDF" --driver --ifwi --limit --json
  capture_readonly_telemetry "gpu-health-${phase}-firmware" \
    amd-smi firmware --gpu "$R9700_BDF" --ucode-list --json
  python3 - "$OUT" "$phase" "$R9700_BDF" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
phase = sys.argv[2]
bdf = sys.argv[3]
names = ("metrics", "bad-pages", "static", "firmware")
records = {}
for name in names:
    stem = root / f"gpu-health-{phase}-{name}"
    exit_code = int(stem.with_suffix(".exit-code").read_text(encoding="utf-8").strip())
    stdout = stem.with_suffix(".json").read_bytes()
    stderr = stem.with_suffix(".stderr").read_bytes()
    try:
        parsed = json.loads(stdout)
        json_status = "parsed"
    except json.JSONDecodeError:
        parsed = None
        json_status = "unparsed"
    records[name] = {
        "exit_code": exit_code,
        "json_status": json_status,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "data": parsed,
    }
summary = {
    "schema_version": "ullm.aq4_phase3c_gpu_health.v1",
    "status": "complete" if all(item["exit_code"] == 0 for item in records.values()) else "partial",
    "phase": phase,
    "target_pci_bdf": bdf,
    "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "records": records,
}
json.dump(summary, open(root / f"gpu-health-{phase}-summary.json", "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
}

capture_r9700_health before

python3 "$REPO/tools/verify-aq4-layer0-package-embedding-fixture.py" \
  --package "$PACKAGE" \
  --hybrid-input "$HYBRID_INPUT" \
  --output "$OUT/cpu-input-identity.json"

(
  cd "$REPO"
  ULLM_BUILD_GIT_COMMIT="$TRACE_TOOLING_COMMIT" \
    cargo build --release -p ullm-engine \
      --bin ullm-aq4-differential-trace \
      --bin ullm-aq4-layer0-family-isolation
)

TRACE_BIN="$REPO/target/release/ullm-aq4-differential-trace"
CPU_BIN="$REPO/target/release/ullm-aq4-layer0-family-isolation"
test -x "$TRACE_BIN"
test -x "$CPU_BIN"

env \
  -u HIP_VISIBLE_DEVICES \
  -u ULLM_HIP_VISIBLE_DEVICES \
  -u ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL \
  -u ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL \
  -u ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL \
  -u ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL \
  -u ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL \
  -u ULLM_REQUIRE_HIP_RMSNORM_KERNEL \
  -u ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL \
  "$CPU_BIN" \
    --package "$PACKAGE" \
    --hybrid-input "$HYBRID_INPUT" \
    --output "$OUT/cpu-reference" \
    --stage-stream-stdout \
    > "$OUT/cpu-stages.f32le" \
    2> "$OUT/cpu-reference.stderr"

TRACE_STARTED_AT_UTC="$(date --iso-8601=seconds --utc)"
set +e
env \
  -u ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING \
  -u ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA \
  HIP_VISIBLE_DEVICES=1 \
  ULLM_HIP_VISIBLE_DEVICES=1 \
  ULLM_SERVED_MODEL_MANIFEST=/etc/ullm/served-models/active.json \
  ULLM_BUILD_GIT_COMMIT="$TRACE_TOOLING_COMMIT" \
  ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB=4 \
  ULLM_AQ4_MATVEC_SILU_MUL_RPB=8 \
  ULLM_AQ4_MATVEC_ADD_RPB=8 \
  ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 \
  ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL=1 \
  ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1 \
  ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL=1 \
  ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL=1 \
  ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1 \
  ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL=1 \
  bash -ceu '
    lock=$1
    trace_bin=$2
    package=$3
    cases=$4
    replay=$5
    trace_root=$6
    out=$7
    (
      flock -n 9
      exec "$trace_bin" "$package" "$cases" "$replay" "$trace_root" 1 \
        --enable-intermediate-trace \
        --enable-linear-stage-trace \
        > "$out/gpu-trace.stdout" \
        2> "$out/gpu-trace.stderr"
    ) 9<>"$lock"
  ' bash "$LOCK" "$TRACE_BIN" "$PACKAGE" "$CASES" "$REPLAY" "$OUT/gpu-trace" "$OUT"
TRACE_EXIT_CODE=$?
set -e
TRACE_FINISHED_AT_UTC="$(date --iso-8601=seconds --utc)"
printf '%s\n' \
  "started_at_utc=$TRACE_STARTED_AT_UTC" \
  "finished_at_utc=$TRACE_FINISHED_AT_UTC" \
  "exit_code=$TRACE_EXIT_CODE" \
  > "$OUT/gpu-trace.exit-status.txt"
capture_r9700_health after
if [ "$TRACE_EXIT_CODE" -ne 0 ]; then
  exit "$TRACE_EXIT_CODE"
fi

(
  cd "$OUT/gpu-trace"
  sha256sum -c SHA256SUMS
)
python3 - "$OUT/gpu-trace/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["mode"] == "aq4_gpu_intermediate_diagnostic"
assert manifest["production_worker_unchanged"] is True
assert manifest["device_index"] == 1
assert manifest["identity"]["device"]["backend"].lower() == "hip"
kernel = manifest["stage_contract"]["kernel_stage_trace"]
assert kernel["enabled"] is True
assert kernel["layer_index"] == 0
assert kernel["f32le_stream_file"] == "kernel-stages.f32le"
assert len(kernel["stage_order"]) == 10
guard = manifest["guard_set"]["linear_stage_guard"]
assert guard["expected_architecture"] == "gfx1201"
assert len(guard["required_environment"]) == 7
PY

python3 "$REPO/tools/compare-aq4-layer0-cpu-gpu-stage-stream.py" \
  --cpu-stream "$OUT/cpu-stages.f32le" \
  --gpu-stream "$OUT/gpu-trace/kernel-stages.f32le" \
  --output "$OUT/cpu-gpu-stage-compare/comparison.json"
```

`flock -n` が失敗した場合、inner trace commandは起動しない。lockを待たず、serviceを操作せず、その一回を失敗として扱う。CPU commandはglobal runtime device 0を明示する既存binaryであり、HIP deviceを使わない。

## 成功・失敗の判定

まず操作上の成功を次のすべてと定義する。

- `r9700-architecture-guard.json`が`status=valid`である。filtered HIP ordinal 0は可視GPU数1、`gfx1201`、non-empty name、PCI BDFを記録し、その同一BDFのtargeted `amd-smi` recordは`gfx1201`、PCI device ID `0x7551`、non-empty market nameである。
- `gpu-health-before-summary.json`と`gpu-health-after-summary.json`、および各raw stdout/stderr/exit-codeが保存されている。metric未対応などでsummaryが`partial`ならH9 telemetryは未取得箇所を明示し、H9を否定しない。
- `cpu-input-identity.json` が`status=valid`で、全hybrid embedding rowがpackage BF16 passthroughとbit-exactである。
- `cpu-stages.f32le`と`kernel-stages.f32le`がterminal frameまで完全であり、比較器が3 context × 10 stage = 30 recordを受理する。
- `SHA256SUMS`、GPU trace manifest、`HIP`/`gfx1201`/global device `1`、7つのfusion guard、RPB固定、`production_worker_unchanged=true`を確認できる。
- `cpu-gpu-stage-compare/comparison.json`が`status=valid`かつ全値finiteである。

数値の判定は修正可否ではなく、H5の局所化のためのevidence分類である。07/15の限定QKV/Z/gate/beta probeが相対L2概ね`1e-6`以下だった事実を基準にし、次を使う。

| stageのfull-tensor relative L2 | 判定 | 次の扱い |
| --- | --- | --- |
| `<= 1e-5` | 既知のf32 reduction/数学関数の丸めと両立する帯域 | そのstage単独ではroot候補にしない。 |
| `> 1e-5` かつ `<= 1e-3` | 要記録だが静的レビューだけでは有意と断定しない | 最初の境界候補として残すが、fixしない。 |
| `> 1e-3` かつ `<= 1e-2` | 限定probeの帯域を二桁以上上回る有意差 | 最初にこの条件を満たすstageをH5局所化候補として記録する。 |
| `> 1e-2` またはnon-finite/shape不一致 | 強い実装差またはtrace contract failure | evidenceを保存し、Phase 4承認まで修正しない。 |

`max_abs`と`cosine`は補助指標であり、relative L2を置き換えない。最初の`> 1e-3` stageの直前までが`<=1e-5`なら、そのstageを担当するfusion/device operationが最優先のレビュー対象になる。QKV/Zが先ならAQ4 dequant/fused projection、gate/betaならA/B式、`recurrent_state_after`ならrecurrent kernel、`attention_residual`以降ならout projection/norm/MLPを調べる。これはroot causeの確定ではなく、Phase 4に持ち込む根拠の分類である。

## 所要時間と失敗時

予定windowは10分を確保する。目安はCPU input identity + CPU stage streamが2--5分、exclusive GPU trace（package load、3 row、D2H含む）が2--4分、checksum/比較が1分未満である。これは上限保証ではなく、途中でerrorが出たら待機・再試行しないための運用枠である。

失敗時は次を守る。

- 一回のrun内でも、別windowでも自動・手動の再試行をしない。
- `OUT`、`phase3c-preflight.txt`、R9700 architecture guard、GPU health telemetry、CPU receipt/stream/report、`gpu-trace.stdout`、`gpu-trace.stderr`を削除・上書きしない。
- traceがpublication前に失敗した場合、toolが残す`gpu-trace.incomplete-PID`も保存する。ただしterminal frame、manifest、`SHA256SUMS`を満たさないrootは比較・promotionに使わない。
- lock failure、R9700 architecture guard failure、architecture/backend mismatch、input identity mismatch、checksum failure、comparison parser failureはすべて「測定無効」であり、数値の結論を出さない。guard failure時はHIP/ASIC evidenceだけを残し、CPU referenceやtraceを起動しない。
- health telemetryの一部が取得不能でも設定を変更して補完しない。traceが成功しても、その項目についてH9は判定不能として記録する。単発runの結果だけで決定性・熱相関・他GPU比較を結論づけない。
- service/systemd/active manifest/P3 harnessを復旧操作の名目で変更しない。failure evidenceとexit codeをjournalに追記して、次の判断をユーザーに委ねる。
