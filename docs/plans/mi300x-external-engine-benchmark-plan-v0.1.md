# MI300X×1 外部エンジン推論速度比較 実行計画 v0.1

- Status: 実行準備のみ。GPU を使用しない予約前作業と、MI300X×1 借用中の実行を分離する。
- Date: 2026-07-26
- Parent: [existing-engine-benchmark-plan-v0.1.md](existing-engine-benchmark-plan-v0.1.md) の `Future MI300X grid` を、単一 MI300X 用に具体化する付属計画である。
- uLLM gate: [sq8-cdna3-mi300x-validation-checklist-v0.1.md](sq8-cdna3-mi300x-validation-checklist-v0.1.md) を先に実行する。この文書は同チェックリストを置換しない。

## 1. 目的、比較境界、禁止操作

MI300X×1 上で vLLM、SGLang、llama.cpp の実サーバー速度を、同一 prompt/context/generation 条件、同一の同時実行数スイープ、同一の HTTP 計時定義で採取する。14B は MI300X の HBM 192 GB に対して小さいため、単発 stream の値を主結果にしない。主結果は `C=1,2,4,8,16,32,64,128` の飽和曲線である。

| 項目 | 方針 |
| --- | --- |
| vLLM / SGLang | 同一の `Qwen/Qwen3-14B-FP8`、revision `9a283b4a5efbc09ce247e0ae5b02b744739e525a` を読む。重み、tokenizer、revision を固定するので uLLM `SQ8_0` の元重みとの format 比較になる。 |
| llama.cpp | FP8 safetensors を直接読めないため、公式 `Qwen/Qwen3-14B-GGUF` の `Qwen3-14B-Q8_0.gguf` を使う。Q8_0 は約 8.5 bpp であり、FP8 行との同一 format 比較ではない。この注記を result に必ず残す。 |
| ROCm/ATOM | 親計画には残すがこの借用枠には追加しない。三 engine、48 HTTP trial/engine、uLLM gate を先に終える方が比較価値と課金時間に対して優先される。 |
| uLLM A′ の失敗 | 外部 engine を中止する理由ではない。fragment/lane が失敗したら uLLM の後続段は止めるが、三 engine は MI300X の達成可能 tok/s を示す hardware target として続行する。 |
| 禁止操作 | `ullm-openai.service`、`/etc/ullm/served-models/active.json`、activation、campaign、`git push`、履歴書換え、`/opt/ullm`、既存 kernel、既存 result に触れない。GPU を使うのは借用時のこの runbook だけである。 |

R9700 の既知値は decode で paged attention 51.05%、CK projection 40.13%、`C=1036` の KV 込み論理帯域効率 36.1088%（640 GB/s 基準、15.294955751 tok/s、231.096063 GB/s）、論理 KV 帯域 decode 55.157770 GB/s / prefill 391.459814 GB/s である。これらは MI300X の予測値ではない。MI300X 行は同じ式で再計算し、hardware/partition が異なれば別行とする。

## 2. 予約前に固定する比較 contract

### 2.1 モデル artifact

ローカルに確認済みの FP8 source は 4 safetensors shard と tokenizer を含み、directory 合計は 16,342,198,967 byte である。

~~~bash
export FP8_REPO=Qwen/Qwen3-14B-FP8
export FP8_REV=9a283b4a5efbc09ce247e0ae5b02b744739e525a
export FP8_SOURCE=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8
export PRELEASE=/srv/mi300x-prelease-20260726

mkdir -p "$PRELEASE/manifests"
(
  cd "$FP8_SOURCE"
  find . -maxdepth 1 -type f -printf '%P\0' | sort -z | xargs -0 sha256sum
) >"$PRELEASE/manifests/qwen3-14b-fp8.sha256"
du -sb "$FP8_SOURCE" | tee "$PRELEASE/manifests/qwen3-14b-fp8.du.txt"
~~~

Q8_0 GGUF は公開済みであることを予約前に確認した。計画時点の Hugging Face revision は `530227a7d994db8eca5ab5ced2fb692b614357fd`、file size は 15,698,533,728 byte である。実行時にも revision とダウンロード後 SHA-256 を record する。

~~~bash
export GGUF_REPO=Qwen/Qwen3-14B-GGUF
export GGUF_REV=530227a7d994db8eca5ab5ced2fb692b614357fd
export GGUF_FILE=Qwen3-14B-Q8_0.gguf
~~~

FP8 artifact の revision は upstream model repository の revision であり、ローカル directory を Git repository とみなさない。転送元・転送先で上の SHA-256 manifest を照合する。

### 2.2 共通 prompt と token contract

`C=1036` の既存 decode 解釈に合わせ、比較 workload は **1024 prompt token、20 output token** とする。最初の 4 output token は decode warm-up、残る 16 token の wall-clock throughput と ITL を主 decode 値にする。decode cache-length window は `1028 -> 1044`、中点は `1036` である。prefill 行は同じ 1024-token prompt に output 1 token を要求するため、first decode token を含む effective prefill 値として明記する。

committed fixture には `raw-p1024` が存在しない。既存 512-token fixture を推測で代用しない。予約前に下の command で FP8 tokenizer から一つの UTF-8 prompt と raw ID companion を作り、uLLM の比較用 batch harness にも同じ raw ID を渡す。uLLM が同じ raw ID を読む準備が間に合わない場合、外部 benchmark は実行してよいが、uLLM との直接速度比較欄は `not-comparable: shared-prompt harness unavailable` とする。

~~~bash
export WORKLOAD=$PRELEASE/workload-qwen3-14b-p1024-g20-v1
mkdir -p "$WORKLOAD"
python3 - "$FP8_SOURCE" "$WORKLOAD" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

from transformers import AutoTokenizer

model_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
target = 1024
tok = AutoTokenizer.from_pretrained(
    model_dir, revision="9a283b4a5efbc09ce247e0ae5b02b744739e525a"
)
seed = "A fixed inference benchmark checks the same token sequence on every engine. " * 400
raw = tok.encode(seed, add_special_tokens=False)
special = len(tok("", add_special_tokens=True).input_ids)
prompt = None
ids = None
for count in range(min(len(raw), target - special), 0, -1):
    candidate = tok.decode(
        raw[:count], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    candidate_ids = tok(candidate, add_special_tokens=True).input_ids
    if len(candidate_ids) == target:
        prompt, ids = candidate, candidate_ids
        break
if prompt is None:
    raise SystemExit("could not construct an exactly 1024-token prompt")
packed = struct.pack("<" + "I" * len(ids), *ids)
(out / "prompt-1024.txt").write_text(prompt, encoding="utf-8")
(out / "prompt-1024.u32le").write_bytes(packed)
(out / "prompt-1024.json").write_text(json.dumps({
    "model": "Qwen/Qwen3-14B-FP8",
    "revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
    "add_special_tokens": True,
    "token_count": len(ids),
    "token_ids": ids,
    "u32le_sha256": hashlib.sha256(packed).hexdigest(),
}, indent=2) + "\n", encoding="utf-8")
with (out / "common-1024.jsonl").open("w", encoding="utf-8") as stream:
    for _ in range(512):
        stream.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
PY
(
  cd "$WORKLOAD"
  sha256sum prompt-1024.txt prompt-1024.u32le prompt-1024.json common-1024.jsonl
) | tee "$WORKLOAD/SHA256SUMS"
~~~

`common-1024.jsonl` は 512 行の同一 UTF-8 prompt である。各 server の prefix/prompt cache を明示的に無効化するので、同一 prompt の再利用で prefill を省略してはならない。512 行にする理由は最大 `4 × C=512` request を bench client の oversampling に依存せず送るためである。

全 engine の smoke 後、次を実行して raw response を保存する。`usage.prompt_tokens == 1024` でない engine は HTTP sweep に進めない。GGUF tokenizer 差による failure は速度 failure ではなく、format comparison の不成立である。

~~~bash
export PROMPT_JSON=$(jq -Rn --rawfile prompt "$WORKLOAD/prompt-1024.txt" \
  '{model:$ENV.BENCH_MODEL,prompt:$prompt,max_tokens:1,temperature:0,ignore_eos:true}')
curl -fsS "http://127.0.0.1:$BENCH_PORT/v1/completions" \
  -H 'Content-Type: application/json' --data "$PROMPT_JSON" \
  | tee "$RESULTS/$BENCH_ENGINE/prompt-count.json"
jq -e '.usage.prompt_tokens == 1024' "$RESULTS/$BENCH_ENGINE/prompt-count.json"
~~~

### 2.3 固定する測定 matrix

| leg | prompt / output | concurrency | trial | primary result |
| --- | --- | --- | --- | --- |
| admission smoke | 1024 / 20 | 1 | 1 | `/v1/models`、stream、token count、cache 無効化の確認 |
| prefill | 1024 / 1 | `1,2,4,8,16,32,64,128` | 各 C 3 回 | `sum(prompt tokens) / wall time`、TTFT |
| decode / E2E | 1024 / 20 | 同上 | 各 C 3 回 | 16-token decode throughput、TTFT、ITL p50/p95、総 output throughput |

各 measured call は `4 × C` request、arrival rate `inf`、client-side maximum concurrency `C` である。各 call の前に official client の endpoint validation と `--num-warmups 2` を実行し、それらは raw result に含めない。順序は C の昇順、repetition `1..3`。3 trial の median を主値、全 raw trial を併記する。median から 10% 以上離れた trial は捨てず `outlier=true` とし、同一 C/条件で 1 回だけ追加 trial を採る。

## 3. モデル配置：推奨、所要時間、fallback

### 3.1 推奨経路

推奨は、GPU lease より前に persistent volume/object storage へ **ローカルの FP8 artifact を upload** し、同じ storage に Q8_0 GGUF も事前配置する方法である。FP8 はすでに local にあり local manifest との byte-for-byte 照合ができる。instance 上の Hugging Face download は revision 固定の有効な fallback だが、network、rate limit、HF/Xet、disk の待ちが課金時間に入るため第一選択ではない。

| 方法 | FP8 の確実性 | GPU lease への影響 | 所要時間の扱い |
| --- | --- | --- | --- |
| local upload + manifest verify（推奨） | local 4 shard/tokenizer と SHA-256 を直接照合できる | persistent storage を GPU attach 前に準備できれば 0 分 | 16,342,197,548 byte の payload 下限を予約前の同一路 1 GiB transfer 実測で算出する。 |
| instance で HF revision-fixed download | commit ID に固定でき、`hf` が file integrity を検査する | persistent volume に GPU 前 download できれば許容、GPU 中なら fallback のみ | provider/HF の実測 throughput は未確認。download 中は benchmark を開始しない。 |

payload だけの理想下限は `bytes × 8 / sustained_bits_per_second` である。FP8 16.34 GB の下限は 100 Mbit/s で 21.8 分、500 Mbit/s で 4.4 分、1 Gbit/s で 2.2 分である。FP8 + Q8_0 の計 32.04 GB はそれぞれ 42.7 / 8.5 / 4.3 分である。これは protocol、disk、checksum、retry を含まない。予約枠の見積りは同一路の実測だけで決め、実測がなければ GPU lease 中に transfer しない。

upload の例は以下である。`REMOTE_VOLUME` は GPU instance に attach する persistent path へ予約前に置換する。転送先の `sha256sum -c` が通らなければ artifact gate は fail である。

~~~bash
export REMOTE_HOST=cpu-staging.example.invalid  # reservation-specific hostname
export REMOTE_VOLUME=/mnt/persistent/mi300x-qwen3
rsync -aH --info=progress2 --partial --append-verify \
  -e 'ssh -o ServerAliveInterval=30' \
  "$FP8_SOURCE/" "$REMOTE_HOST:$REMOTE_VOLUME/Qwen3-14B-FP8/"
rsync -aH --info=progress2 --partial --append-verify \
  -e 'ssh -o ServerAliveInterval=30' \
  "$PRELEASE/manifests/qwen3-14b-fp8.sha256" \
  "$REMOTE_HOST:$REMOTE_VOLUME/"
rsync -aH --info=progress2 --partial --append-verify \
  -e 'ssh -o ServerAliveInterval=30' \
  "$WORKLOAD/" "$REMOTE_HOST:$REMOTE_VOLUME/workload-qwen3-14b-p1024-g20-v1/"
ssh "$REMOTE_HOST" "cd '$REMOTE_VOLUME/Qwen3-14B-FP8' && sha256sum -c '$REMOTE_VOLUME/qwen3-14b-fp8.sha256'"
ssh "$REMOTE_HOST" "cd '$REMOTE_VOLUME/workload-qwen3-14b-p1024-g20-v1' && sha256sum -c SHA256SUMS"
~~~

### 3.2 HF download fallback

Hugging Face CLI と十分な disk がある staging/instance では、必ず revision を指定する。

~~~bash
export MODEL_ROOT=/mnt/persistent/mi300x-qwen3
export FP8_MODEL=$MODEL_ROOT/Qwen3-14B-FP8
export GGUF_MODEL=$MODEL_ROOT/Qwen3-14B-GGUF/$GGUF_FILE
mkdir -p "$MODEL_ROOT/Qwen3-14B-GGUF"

hf download "$FP8_REPO" \
  --revision "$FP8_REV" \
  --local-dir "$FP8_MODEL"

hf download "$GGUF_REPO" \
  --revision "$GGUF_REV" \
  --include "$GGUF_FILE" \
  --local-dir "$MODEL_ROOT/Qwen3-14B-GGUF"

sha256sum "$FP8_MODEL"/*.safetensors "$GGUF_MODEL" \
  | tee "$MODEL_ROOT/model-artifacts.sha256"
~~~

公開 Q8_0 GGUF が使えなくなった場合だけ BF16 原本から変換する。FP8 safetensors を Q8_0 source と偽装して変換してはならない。この fallback は CPU/disk-heavy で、所要時間は source path、CPU、storage が未確認であるため予約前に一度計時する。GPU lease 中には実行しない。

~~~bash
export BF16_MODEL=/absolute/path/to/Qwen3-14B-BF16
export LLAMA_SRC=/srv/src/llama.cpp
python3 "$LLAMA_SRC/convert_hf_to_gguf.py" "$BF16_MODEL" \
  --outfile "$MODEL_ROOT/Qwen3-14B-BF16.gguf" --outtype bf16
"$LLAMA_SRC/build/bin/llama-quantize" \
  "$MODEL_ROOT/Qwen3-14B-BF16.gguf" "$GGUF_MODEL" Q8_0
sha256sum "$GGUF_MODEL" | tee "$MODEL_ROOT/Qwen3-14B-Q8_0.gguf.sha256"
~~~

## 4. instance admission と version pin

以下の command は **借用した MI300X instance 上だけ**で実行する。partition、clock、power の変更 command は含めない。

~~~bash
export RUN_ID=mi300x-external-$(date -u +%Y%m%dT%H%M%SZ)
export RESULTS=/mnt/persistent/mi300x-results/$RUN_ID
mkdir -p "$RESULTS"/{environment,images,hardware,telemetry,vllm,sglang,llama.cpp,normalized}

uname -a | tee "$RESULTS/environment/uname.txt"
docker version | tee "$RESULTS/environment/docker-version.txt"
rocminfo | tee "$RESULTS/hardware/rocminfo.txt"
hipcc --version | tee "$RESULTS/environment/hipcc-version.txt"
amd-smi version | tee "$RESULTS/hardware/amd-smi-version.txt"
amd-smi --rocm-smi | tee "$RESULTS/hardware/amd-smi-rocm-smi.txt"
amd-smi list --json | tee "$RESULTS/hardware/amd-smi-list.json"
amd-smi static --gpu 0 --json | tee "$RESULTS/hardware/amd-smi-static-gpu0.json"
amd-smi partition --gpu 0 --current --json | tee "$RESULTS/hardware/partition-current-gpu0.json"
amd-smi partition --gpu 0 --memory --json | tee "$RESULTS/hardware/partition-memory-gpu0.json"
amd-smi partition --gpu 0 --accelerator --json | tee "$RESULTS/hardware/partition-accelerator-gpu0.json"
amd-smi topology --gpu 0 --json | tee "$RESULTS/hardware/topology-gpu0.json"
amd-smi process --gpu 0 --json | tee "$RESULTS/hardware/process-before.json"
~~~

Admission pass は、(a) visible device が 1 台の MI300X/gfx942、(b) 他 process が GPU を使用していない、(c) FP8/Q8 artifact と workload hash が一致、(d) Docker/ROCm/HIP output が保存済み、である。SKU、XCD、NPS/memory/accelerator partition は変更せず record する。full-card MI300X ではない、または HBM path の partition mapping を一次資料で確認できない場合も benchmark 自体は行えるが、5.3 TB/s を partition theoretical cap として使ってはならない。

planning snapshot として 2026-07-26 に存在を確認した値は次である。これは latest という曖昧な語を result に残さないための既知 fallback であり、借用開始時に再確認する。

| engine | planning snapshot | confirmed image/source digest/commit |
| --- | --- | --- |
| vLLM | release `v0.26.0` | `vllm/vllm-openai-rocm:v0.26.0` / `sha256:5709fafe47123becb2f5e61c32d0b97beff1a629bb40bb753c15464f69a97a18` |
| SGLang | release `v0.5.16` | `lmsysorg/sglang:v0.5.16-rocm720-mi30x` / `sha256:80d04638deb64fac000fa565cb46e5d2f692173dc125a32a956014a6383ecaee` |
| llama.cpp | release `b10107` | source commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9` |

借用時に latest stable release を確認し、対応する公式 ROCm image、CLI option、digest を 10 分以内に admission する。新しい release に公式 MI300X/ROCm image がなければ tag 名を推測して作らない。planning snapshot を使う場合は `latest-at-run=false` と明記する。

~~~bash
curl -fsSL https://api.github.com/repos/vllm-project/vllm/releases/latest \
  | tee "$RESULTS/images/vllm-release-latest.json"
curl -fsSL https://api.github.com/repos/sgl-project/sglang/releases/latest \
  | tee "$RESULTS/images/sglang-release-latest.json"
curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
  | tee "$RESULTS/images/llama-cpp-release-latest.json"

export VLLM_TAG=v0.26.0
export VLLM_IMAGE=vllm/vllm-openai-rocm:$VLLM_TAG
export SGLANG_TAG=v0.5.16-rocm720-mi30x
export SGLANG_IMAGE=lmsysorg/sglang:$SGLANG_TAG
export LLAMA_REF=b10107

docker pull "$VLLM_IMAGE"
docker pull "$SGLANG_IMAGE"
docker image inspect "$VLLM_IMAGE" | tee "$RESULTS/images/vllm-image-inspect.json"
docker image inspect "$SGLANG_IMAGE" | tee "$RESULTS/images/sglang-image-inspect.json"
docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$VLLM_IMAGE" \
  | tee "$RESULTS/images/vllm-repodigests.txt"
docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$SGLANG_IMAGE" \
  | tee "$RESULTS/images/sglang-repodigests.txt"
docker run --rm --entrypoint /bin/bash "$VLLM_IMAGE" -lc \
  'python3 -c "import vllm; print(vllm.__version__)"; vllm serve --help' \
  >"$RESULTS/images/vllm-version-and-help.txt"
docker run --rm --entrypoint /bin/bash "$SGLANG_IMAGE" -lc \
  'python3 -c "import sglang; print(sglang.__version__)"; python3 -m sglang.launch_server --help' \
  >"$RESULTS/images/sglang-version-and-help.txt"
~~~

`rocm/vllm` と `rocm/vllm-dev` は deprecated であり使わない。SGLang の generic `latest-rocm` tag は計画時点で確認できなかったため使わない。image pull、digest、help のいずれかが planning snapshot と食い違う場合、その engine を `image-or-cli-admission-failed` として止め、他 engine へ進む。

Docker engine 用に公式 ROCm container の device/security arguments を共通化する。`--network host` を使うため `-p` は不要である。

~~~bash
ROCM_DOCKER=(
  --device=/dev/kfd
  --device=/dev/dri
  --group-add video
  --cap-add SYS_PTRACE
  --security-opt seccomp=unconfined
  --ipc=host
)
export FP8_MODEL=/mnt/persistent/mi300x-qwen3/Qwen3-14B-FP8
export GGUF_MODEL=/mnt/persistent/mi300x-qwen3/Qwen3-14B-GGUF/Qwen3-14B-Q8_0.gguf
export WORKLOAD=/mnt/persistent/mi300x-qwen3/workload-qwen3-14b-p1024-g20-v1
(
  cd "$FP8_MODEL"
  sha256sum -c /mnt/persistent/mi300x-qwen3/qwen3-14b-fp8.sha256
)
sha256sum "$GGUF_MODEL" | tee "$RESULTS/environment/gguf-sha256.txt"
(
  cd "$WORKLOAD"
  sha256sum -c SHA256SUMS
) | tee "$RESULTS/environment/workload-sha256.txt"
~~~

## 5. 各 engine の起動 command

各 engine は一つずつ起動・測定・停止する。同時に二つの model server を動かさない。health function は model load 完了を確認するだけであり、timeout まで ready にならなければ当該 engine を止める。

~~~bash
wait_openai_ready() {
  local name=$1 port=$2
  local output="$RESULTS/$name/models.json"
  for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >"$output"; then
      return 0
    fi
    sleep 1
  done
  echo "$name did not become ready within 600 seconds" >&2
  return 1
}
~~~

### 5.1 vLLM: FP8 safetensors

official ROCm image は vllm/vllm-openai-rocm である。FP8 checkpoint なので quantization=fp8、model dtype=bfloat16、KV=bfloat16 を明示する。prefix caching は必ず無効にする。同一 prompt の duplicate row が prefill KV reuse されると prefill sweep が無効になるためである。

~~~bash
docker run -d --name mi300x-vllm \
  "${ROCM_DOCKER[@]}" \
  --network host \
  -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -v "$FP8_MODEL:/models/Qwen3-14B-FP8:ro" \
  -v "$RESULTS/vllm:/run/results:rw" \
  --entrypoint /bin/bash "$VLLM_IMAGE" -lc '
    exec vllm serve /models/Qwen3-14B-FP8 \
      --revision 9a283b4a5efbc09ce247e0ae5b02b744739e525a \
      --tokenizer-revision 9a283b4a5efbc09ce247e0ae5b02b744739e525a \
      --dtype bfloat16 --quantization fp8 --kv-cache-dtype bfloat16 \
      --max-model-len 4096 --gpu-memory-utilization 0.90 \
      --max-num-seqs 128 --max-num-batched-tokens 131072 \
      --tensor-parallel-size 1 --pipeline-parallel-size 1 \
      --no-enable-prefix-caching \
      --served-model-name qwen3-14b-fp8 --host 0.0.0.0 --port 8000
  ' >"$RESULTS/vllm/container-id.txt" 2>"$RESULTS/vllm/docker-run.stderr.txt"

wait_openai_ready vllm 8000
docker inspect mi300x-vllm | tee "$RESULTS/vllm/container-inspect.json"
docker logs mi300x-vllm >"$RESULTS/vllm/container.log.txt" 2>&1
~~~

Admission requires the log to show the intended model path/revision, FP8 quantization, one tensor-parallel rank, max sequence count 128, and prefix caching disabled. Any model auto-conversion, unsupported quantization fallback, or load OOM is `vllm-server-admission-failed`; preserve log/inspect and proceed to SGLang rather than changing parameters in the paid slot.

### 5.2 SGLang: FP8 safetensors

official MI300X-oriented planning image は lmsysorg/sglang:v0.5.16-rocm720-mi30x である。attention backend は image/runtime によって変わり得るため hard-code せず、startup log の selected backend を record する。radix cache は無効化して prefill reuse を禁止する。

~~~bash
docker run -d --name mi300x-sglang \
  "${ROCM_DOCKER[@]}" \
  --network host \
  -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -v "$FP8_MODEL:/models/Qwen3-14B-FP8:ro" \
  -v "$RESULTS/sglang:/run/results:rw" \
  --entrypoint /bin/bash "$SGLANG_IMAGE" -lc '
    exec python3 -m sglang.launch_server \
      --model-path /models/Qwen3-14B-FP8 \
      --revision 9a283b4a5efbc09ce247e0ae5b02b744739e525a \
      --dtype bfloat16 --quantization fp8 --kv-cache-dtype bfloat16 \
      --context-length 4096 --mem-fraction-static 0.90 \
      --max-running-requests 128 --max-total-tokens 524288 \
      --max-prefill-tokens 131072 --prefill-max-requests 128 \
      --tp-size 1 --pp-size 1 --disable-radix-cache --stream-interval 1 \
      --served-model-name qwen3-14b-fp8 --enable-metrics \
      --host 0.0.0.0 --port 30000
  ' >"$RESULTS/sglang/container-id.txt" 2>"$RESULTS/sglang/docker-run.stderr.txt"

wait_openai_ready sglang 30000
docker inspect mi300x-sglang | tee "$RESULTS/sglang/container-inspect.json"
docker logs mi300x-sglang >"$RESULTS/sglang/container.log.txt" 2>&1
~~~

524288 は 128 request × 4096 token capacity である。SGLang がこれより小さく clamp、または max-running-requests を 128 未満へ下げるなら requested-C contract を満たさない。設定を下げて見かけ上成功にせず `sglang-capacity-admission-failed` とする。stream-interval 1 が効かず一 event が複数 token なら ITL は算出しない。

### 5.3 llama.cpp: Q8_0 GGUF / HIP build

llama.cpp は native HIP build にする。build は GPU lease 前の同じ ROCm image/host で済ませることが望ましい。できない場合でも source download/build を benchmark timebox 内に隠さず、engine admission の 12 分上限に含める。planning snapshot の b10107 と commit を実行時に再照合する。

~~~bash
export LLAMA_SRC=/srv/src/llama.cpp-b10107
git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
git -C "$LLAMA_SRC" rev-parse HEAD | tee "$RESULTS/llama.cpp/source-commit.txt"
git -C "$LLAMA_SRC" status --short | tee "$RESULTS/llama.cpp/source-status.txt"

cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_SRC/build" --target llama-server llama-bench llama-quantize \
  -j "$(nproc)"
"$LLAMA_SRC/build/bin/llama-server" --version \
  | tee "$RESULTS/llama.cpp/llama-server-version.txt"
"$LLAMA_SRC/build/bin/llama-bench" --help \
  >"$RESULTS/llama.cpp/llama-bench-help.txt"
~~~

server の総 context 524288 を 128 parallel slot で割るので、startup log で one slot あたり 4096 context であることを確認する。Q8_0 weight と F16 KV cache を明示し、prompt cache と KV reuse を無効化する。F16/BF16 の format 名は FP8 engine と異なるが、KV はどちらも 2 byte/element であり、result に各々を記録する。

~~~bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  "$LLAMA_SRC/build/bin/llama-server" \
    --model "$GGUF_MODEL" --alias qwen3-14b-q8_0 \
    --host 0.0.0.0 --port 8080 \
    --gpu-layers all --ctx-size 524288 --parallel 128 \
    --batch-size 131072 --ubatch-size 1024 \
    --cache-type-k f16 --cache-type-v f16 --flash-attn auto \
    --cont-batching --no-cache-prompt --cache-reuse 0 --metrics \
    >"$RESULTS/llama.cpp/server.stdout.txt" \
    2>"$RESULTS/llama.cpp/server.stderr.txt" &
export LLAMA_SERVER_PID=$!

wait_openai_ready llama.cpp 8080
ps -fp "$LLAMA_SERVER_PID" | tee "$RESULTS/llama.cpp/server-ps.txt"
~~~

llama-server の OpenAI-compatible /v1/completions route が common client を受け、usage.prompt_tokens=1024 と stream token boundary を返すことが admission で必要である。受けない場合、llama.cpp の HTTP sweep は `openai-contract-failed` とし、native llama-bench だけを supplementary microbenchmark として残す。llama-bench は tokenizer、HTTP、queue、TTFT、concurrency を測らないので、三 engine の主比較値に混ぜない。

次の llama-bench command は llama-server の HTTP sweep を完了して LLAMA_SERVER_PID を停止した**後だけ**に実行する。server と benchmark を同時に GPU へ載せない。

~~~bash
"$LLAMA_SRC/build/bin/llama-bench" \
  -m "$GGUF_MODEL" -ngl all -c 4096 \
  -ctk f16 -ctv f16 -fa auto \
  -b 1024 -ub 1024 -r 5 -p 1024 -n 0 -o json \
  >"$RESULTS/llama.cpp/llama-bench-prefill.json"

"$LLAMA_SRC/build/bin/llama-bench" \
  -m "$GGUF_MODEL" -ngl all -c 4096 \
  -ctk f16 -ctv f16 -fa auto \
  -b 1024 -ub 1024 -r 5 -p 0 -n 16 -d 1028 -o json \
  >"$RESULTS/llama.cpp/llama-bench-decode.json"
~~~

### 5.4 共通 smoke と停止

各 server の ready 後、まず次を一度だけ実行する。20 token が一 token ごとの SSE event で返ること、early EOS を無視して 20 token を完走すること、usage が返ることを確認する。

~~~bash
run_stream_smoke() {
  local engine=$1 port=$2 model=$3
  jq -Rn --rawfile prompt "$WORKLOAD/prompt-1024.txt" \
    --arg model "$model" \
    '{model:$model,prompt:$prompt,max_tokens:20,temperature:0,ignore_eos:true,
      stream:true,stream_options:{include_usage:true}}' \
    | curl -fsS -N "http://127.0.0.1:$port/v1/completions" \
        -H 'Content-Type: application/json' --data-binary @- \
    | tee "$RESULTS/$engine/stream-smoke.sse"
}
~~~

異常 server は自 engine の log を保存して停止する。Docker server は自分が作成した名前だけを対象にする。

~~~bash
docker logs mi300x-vllm >"$RESULTS/vllm/container-final.log.txt" 2>&1 || true
docker rm -f mi300x-vllm || true
docker logs mi300x-sglang >"$RESULTS/sglang/container-final.log.txt" 2>&1 || true
docker rm -f mi300x-sglang || true
kill "$LLAMA_SERVER_PID" 2>/dev/null || true
wait "$LLAMA_SERVER_PID" 2>/dev/null || true
~~~

## 6. ベンチマーク client、telemetry、正規化

### 6.1 採用する client

主 client は planning snapshot の vLLM image 内の **official `vllm bench serve`** である。これは GPU device を mount しない別 container として、各 engine の OpenAI-compatible `/v1/completions` に接続する。従って HTTP request、arrival、warm-up、TTFT、ITL、wall-clock の定義が三 engine で一つになる。

旧 `benchmarks/benchmark_serving.py` は vLLM v0.26.0 では deprecated wrapper であり使わない。SGLang の `python -m sglang.bench_serving` も deprecated wrapper なので主値には使わない。後者の現行 module 名は `python3 -m sglang.benchmark.serving` であり、version/help を保存して native tool の存在を記録するが、metric definition が common client と異なるため主値には混ぜない。

~~~bash
docker run --rm --entrypoint /bin/bash "$SGLANG_IMAGE" -lc \
  'python3 -m sglang.benchmark.serving --help' \
  >"$RESULTS/images/sglang-benchmark-help.txt"
~~~

llama.cpp の official `llama-bench` は 5.3 節の supplementary microbenchmark として実行する。一方、三 engine の TTFT、ITL、同時実行総 throughput は HTTP server でしか揃わないため common client を優先する。

### 6.2 common client command

この function は raw detailed JSON を engine/phase/C/repetition ごとに一つずつ保存する。client container には GPU device argument を渡さない。model directory の mount は tokenizer 読み込みだけに用いる。

~~~bash
run_common_bench() {
  local engine=$1 port=$2 model=$3 phase=$4 c=$5 repetition=$6 output=$7
  local result_dir="/results/$engine/$phase"
  mkdir -p "$RESULTS/$engine/$phase"

  docker run --rm --network host \
    -e BENCH_BASE_URL="http://127.0.0.1:$port" \
    -e BENCH_MODEL="$model" \
    -e BENCH_C="$c" \
    -e BENCH_OUTPUT="$output" \
    -e BENCH_RESULT_DIR="$result_dir" \
    -e BENCH_RESULT_FILE="c${c}-r${repetition}.json" \
    -v "$FP8_MODEL:/models/Qwen3-14B-FP8:ro" \
    -v "$WORKLOAD:/workload:ro" \
    -v "$RESULTS:/results:rw" \
    --entrypoint /bin/bash "$VLLM_IMAGE" -lc '
      set -euo pipefail
      vllm bench serve \
        --backend openai \
        --base-url "$BENCH_BASE_URL" \
        --endpoint /v1/completions \
        --model "$BENCH_MODEL" \
        --tokenizer /models/Qwen3-14B-FP8 \
        --dataset-name custom \
        --dataset-path /workload/common-1024.jsonl \
        --skip-chat-template \
        --disable-shuffle \
        --custom-output-len "$BENCH_OUTPUT" \
        --num-prompts "$((4 * BENCH_C))" \
        --max-concurrency "$BENCH_C" \
        --request-rate inf \
        --num-warmups 2 \
        --temperature 0 \
        --ignore-eos \
        --save-result \
        --save-detailed \
        --result-dir "$BENCH_RESULT_DIR" \
        --result-filename "$BENCH_RESULT_FILE" \
        --disable-tqdm
    '
}
~~~

SGLang と llama.cpp に対しても `--tokenizer` は FP8 model directory を使う。これは client-side workload selection を完全に同じにするためであり、各 server の actual tokenizer token count は 2.2 節の smoke response で別に検証する。FFP8 server の output model name は qwen3-14b-fp8、llama.cpp の output model name は qwen3-14b-q8_0 である。

### 6.3 raw JSON の contract と正規化

次の helper は raw detailed JSON を検査し、primary metric JSON を出す。pre-fill の output length は 1、decode の output length は 20、input length はすべて 1024、decode の各 request は ITL を正確に 19 個持つことが必須である。stream chunk が複数 token なら ITL を均等割りしない。その run を `stream-boundary-contract-failed` とする。

~~~bash
normalize_common_result() {
  local engine=$1 phase=$2 c=$3 repetition=$4 raw=$5 out=$6
  python3 - "$engine" "$phase" "$c" "$repetition" "$raw" "$out" <<'PY'
import json
import math
import statistics
import sys

engine, phase, c_s, rep_s, raw_path, out_path = sys.argv[1:]
c = int(c_s)
repetition = int(rep_s)
with open(raw_path, encoding="utf-8") as f:
    data = json.load(f)

expected_output = 1 if phase == "prefill" else 20
expected_requests = 4 * c
required = ("duration", "completed", "failed", "input_lens", "output_lens",
            "ttfts", "itls", "start_times")
missing = [k for k in required if k not in data]
if missing:
    raise SystemExit(f"missing detailed fields: {missing}")
if data["completed"] != expected_requests or data["failed"] != 0:
    raise SystemExit("request completion contract failed")
if len(data["input_lens"]) != expected_requests or any(x != 1024 for x in data["input_lens"]):
    raise SystemExit("input-token contract failed")
if len(data["output_lens"]) != expected_requests or any(x != expected_output for x in data["output_lens"]):
    raise SystemExit("output-token contract failed")
if len(data["ttfts"]) != expected_requests or len(data["start_times"]) != expected_requests:
    raise SystemExit("timing-array length mismatch")
max_seen = int(data.get("max_concurrent_requests", 0))
if max_seen < c:
    raise SystemExit(f"requested concurrency {c}, observed only {max_seen}")
duration = float(data["duration"])
if not math.isfinite(duration) or duration <= 0:
    raise SystemExit("invalid benchmark duration")

def percentile(values, p):
    if not values:
        return None
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    lo, hi = int(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)

result = {
    "engine": engine,
    "phase": phase,
    "requested_concurrency": c,
    "repetition": repetition,
    "format": "fp8-safetensors" if engine in ("vllm", "sglang") else "q8_0-gguf",
    "prompt_tokens_per_request": 1024,
    "output_tokens_per_request": expected_output,
    "request_count": expected_requests,
    "wall_seconds_official_client": duration,
    "ttft_seconds_p50": percentile(data["ttfts"], 0.50),
    "ttft_seconds_p95": percentile(data["ttfts"], 0.95),
    "observed_max_concurrency": max_seen,
}
if phase == "prefill":
    result["effective_prefill_tokens_per_second"] = sum(data["input_lens"]) / duration
else:
    if len(data["itls"]) != expected_requests:
        raise SystemExit("missing ITL arrays")
    timed_starts = []
    timed_ends = []
    all_itls = []
    for start, ttft, itls in zip(data["start_times"], data["ttfts"], data["itls"]):
        if len(itls) != 19:
            raise SystemExit("stream-boundary-contract-failed: expected 19 ITLs")
        itls = [float(x) for x in itls]
        timed_starts.append(float(start) + float(ttft) + sum(itls[:3]))
        timed_ends.append(float(start) + float(ttft) + sum(itls))
        all_itls.extend(itls[3:])  # transitions 4->5 through 19->20
    steady_wall = max(timed_ends) - min(timed_starts)
    if not math.isfinite(steady_wall) or steady_wall <= 0:
        raise SystemExit("invalid steady decode wall time")
    result.update({
        "steady_decode_tokens_per_second": (16 * expected_requests) / steady_wall,
        "steady_decode_wall_seconds": steady_wall,
        "itl_seconds_p50": percentile(all_itls, 0.50),
        "itl_seconds_p95": percentile(all_itls, 0.95),
        "total_output_tokens_per_second": sum(data["output_lens"]) / duration,
    })
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
PY
}
~~~

ここで steady decode の開始は各 request の第 4 token 到着時、終了は第 20 token 到着時である。つまり first 4 generated tokens を warm-up として除外しつつ、queue と batch scheduling の影響は wall time に残す。raw JSON、normalized JSON、error JSON のいずれも残し、既存 `tools/run-external-benchmark.py` にはこの vLLM v0.26 detailed schema の parser が確認できないため、この lease では強制的に流用しない。

### 6.4 telemetry と実行 loop

engine ごとに sweep の前後を AMD SMI で採取する。monitor は自分が起動した PID だけを停止する。thermal/power violation は MI300+ の violation field を raw JSON のまま保存する。

~~~bash
start_telemetry() {
  local engine=$1
  amd-smi process --gpu 0 --json >"$RESULTS/$engine/process-before-sweep.json"
  amd-smi monitor --gpu 0 --power-usage --temperature --gfx --mem \
    --vram-usage --violation --watch 1 --json \
    >"$RESULTS/telemetry/$engine-amd-smi-monitor.json" 2>&1 &
  export TELEMETRY_PID=$!
}
stop_telemetry() {
  local engine=$1
  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
  amd-smi process --gpu 0 --json >"$RESULTS/$engine/process-after-sweep.json"
  amd-smi static --gpu 0 --json >"$RESULTS/$engine/static-after-sweep.json"
}
run_engine_sweep() {
  local engine=$1 port=$2 model=$3
  local c repetition raw normalized
  start_telemetry "$engine"
  for c in 1 2 4 8 16 32 64 128; do
    for repetition in 1 2 3; do
      run_common_bench "$engine" "$port" "$model" prefill "$c" "$repetition" 1
      raw="$RESULTS/$engine/prefill/c${c}-r${repetition}.json"
      normalized="$RESULTS/normalized/${engine}-prefill-c${c}-r${repetition}.json"
      normalize_common_result "$engine" prefill "$c" "$repetition" "$raw" "$normalized"

      run_common_bench "$engine" "$port" "$model" decode "$c" "$repetition" 20
      raw="$RESULTS/$engine/decode/c${c}-r${repetition}.json"
      normalized="$RESULTS/normalized/${engine}-decode-c${c}-r${repetition}.json"
      normalize_common_result "$engine" decode "$c" "$repetition" "$raw" "$normalized"
    done
  done
  stop_telemetry "$engine"
}
~~~

server 起動直後に次の順で呼ぶ。各 engine の run が終わってから 5.4 節の停止 command を実行する。

~~~bash
run_stream_smoke vllm 8000 qwen3-14b-fp8
export BENCH_ENGINE=vllm BENCH_PORT=8000 BENCH_MODEL=qwen3-14b-fp8
# 2.2 節の prompt-count command をここで実行する。
run_engine_sweep vllm 8000 qwen3-14b-fp8

run_stream_smoke sglang 30000 qwen3-14b-fp8
export BENCH_ENGINE=sglang BENCH_PORT=30000 BENCH_MODEL=qwen3-14b-fp8
# 2.2 節の prompt-count command をここで実行する。
run_engine_sweep sglang 30000 qwen3-14b-fp8

run_stream_smoke llama.cpp 8080 qwen3-14b-q8_0
export BENCH_ENGINE=llama.cpp BENCH_PORT=8080 BENCH_MODEL=qwen3-14b-q8_0
# 2.2 節の prompt-count command をここで実行する。
run_engine_sweep llama.cpp 8080 qwen3-14b-q8_0
~~~

VRAM baseline/peak、power、temperature、gfx/memory clock、violation/throttle status は telemetry JSON から report する。monitor option が provider の AMD SMI version で受理されない場合、`amd-smi monitor --help` と error を保存し、VRAM/power/thermal/throttle 欄を `unconfirmed: monitor option unavailable` とする。別の flag 名を推測して置き換えない。

## 7. KV 込み論理帯域指標と記録 schema

この節の値は physical HBM counter ではない。全 weight/KV byte が毎 token に HBM を渡ると仮定した **KV 込み logical streaming lower-bound** であり、L2 reuse、page table、activation、workspace、copy、launch overhead は含まない。R9700 の 36.1088% と同じ分類の指標にはなるが、uLLM 側は F32 KV、外部 FP8 server は BF16 KV、llama.cpp は F16 KV なので、KV element size を併記せず eta だけを比較してはならない。

記号の衝突を避ける。以下では Q を同時実行 request 数、L を decode context length とする。既存記録で C=1036 と書かれている cache midpoint はこの計画では L=1036 である。

Qwen3-14B の確認済み config は 40 layer、40 query head、8 KV head、head dimension 128 である。外部 engine の 2 byte KV に対する per generated token の declared denominator は次である。

~~~text
W = sum(.safetensors shard file bytes)        # vLLM / SGLang
  = Q8_0 GGUF file bytes                      # llama.cpp

KV_read(L)  = 40 layers * 8 KV heads * (128 K + 128 V) * 2 B * L
            = 163,840 * L bytes
KV_write    = 163,840 bytes
B_engine(L) = W + KV_read(L) + KV_write
logical_rate_B_per_s = steady_decode_tok_per_s * B_engine(1036)
eta_logical_percent = 100 * logical_rate_B_per_s / B_peak_B_per_s
~~~

実行時に W、L、KV type/byte count、B_peak source、partition、TPS を one JSON に保存する。AMD が公表する full MI300X peak 5.3 TB/s は、AMD SMI で full MI300X SKU と full-HBM path が確認でき、一次資料 URL/date を evidence に保存した場合だけ `B_peak_B_per_s=5300000000000` として使う。NPS/XCD partition の HBM mapping が未確認なら `eta_logical_percent=null`、理由を `partition-bandwidth-unconfirmed` とする。XCD 数だけから 5.3 TB/s を等分しない。

結果の minimum record は以下である。

| group | 必須 field |
| --- | --- |
| identity | run ID、UTC start/end、engine、server/client version、image repo digest または llama.cpp commit、argv、environment allow-list |
| model | repo、revision、local file SHA-256、format、weight byte W、tokenizer source/revision、KV dtype/element bytes |
| hardware | AMD SMI SKU、BDF、gfx arch、XCD/NPS/memory/accelerator partition、ROCm/HIP/docker version |
| workload | prompt text/u32le/SHA-256、prompt 1024、output 1 or 20、warm tokens 4、steady tokens 16、request count 4Q、Q |
| measured | effective prefill tok/s、steady decode tok/s、total output tok/s、TTFT p50/p95、ITL p50/p95、trial raw JSON、median、outlier flag |
| resource/health | VRAM baseline/peak、power、temperature、gfx/mem clock、violation/throttle sample、OOM/failure reason |
| logical metric | L=1036、W、KV read/write bytes、B_engine、B_peak source/value、logical rate、eta または null reason |

hardware counter による physical HBM/L2 efficiency は、この lease の必須 deliverable にしない。counter permission、name、unit、XCD attribution が事前に固定できた場合だけ native profiler raw output と metadata を保存して別欄に載せる。counter を推測で選び、logical byte を physical byte と書き換えない。

## 8. 課金時間内の順序、timebox、中止規則

### 8.1 固定順序

1. GPU lease 前に model/workload/image/build を準備し、transfer/hash を終える。GPU lease 中に download、GGUF conversion、source 修正、調査 build を始めない。
2. lease 開始後、既存 uLLM checklist の入口 preflight を 5--10 分、Stage 1 fragment/lane を 2--5 分で実行する。最短 go/no-go は 10--20 分で出す。
3. A′ が pass した場合だけ、既存 checklist の 5-shape differential、occupancy、HBM/L2 を続ける。A′ が fail した場合は uLLM の後続を停止し、evidence を保存する。
4. A′ pass/fail のどちらでも、この文書 4 節の外部 engine admission を開始する。
5. vLLM -> SGLang -> llama.cpp の順に、server load/smoke/prompt count/sweep/stop を一 engine ずつ終える。
6. llama-bench supplementary result、telemetry、hash manifest を保存して終了する。結果の読み替えや source edit は lease 外で行う。

### 8.2 hard timebox

| phase | 計画時間 | hard cap | pass / 次へ進む条件 |
| --- | ---: | ---: | --- |
| uLLM preflight | 5--10 分 | 10 分 | existing checklist の device/toolchain/hash admission |
| uLLM fragment/lane | 2--5 分 | 10 分 | Stage 1 pass。A′ fail なら uLLM 後続のみ終了 |
| uLLM full continuation（A′ pass 時） | 45--95 分 | 110 分 | differential、occupancy、HBM/L2 を existing checklist の規則で実施 |
| external admission/image/hash | 5--10 分 | 10 分 | section 4 の device/artifact/image contract |
| vLLM load/smoke + sweep | 25--40 分 | 52 分 | HTTP/token/stream/capacity contract と 48 raw trials |
| SGLang load/smoke + sweep | 25--40 分 | 52 分 | 同上 |
| llama.cpp load/smoke + sweep + llama-bench | 32--52 分 | 62 分 | 同上。HIP build が未済なら build の時間もこの cap に含む |
| manifest/archive | 10--15 分 | 15 分 | raw/normalized/log/telemetry の hash 完了 |

最短の有用シナリオは、uLLM go/no-go 10--20 分の後、外部三 engine を全 sweep する **107--177 分（1 時間 47 分--2 時間 57 分）** である。A′ fail ならこれがそのまま採るべき経路になる。A′ pass 時に existing uLLM full continuation も行う想定シナリオは、表の下限/上限を足した **152--287 分（2 時間 32 分--4 時間 47 分）**。予約は transfer/build 済みを前提に **5 時間**を確保し、各 hard cap に達したら未完 engine を明示して終了する。金額はこの計画に含めない。

### 8.3 中止・continue の規則

| phase / event | 行動 |
| --- | --- |
| uLLM Stage 1 fragment/lane fail | 同一 binary/input で一度だけ再現を採取し、uLLM 後続を停止。外部 engine phase は続行する。 |
| external device/artifact/image admission fail | その engine を実行しない。原因、digest、help、hash、AMD SMI raw output を保存し、他 engine へ進む。 |
| server load fail / unsupported FP8 / OOM | command parameter をその場で下げない。その engine を status=failed または oom として保存し、次 engine へ進む。 |
| prompt count !=1024、output count 不一致、SSE 1-token boundary 不一致 | status=failed_harness_contract。TTFT/ITL/throughput を作らず、その engine の sweep を停止する。 |
| Q request 未達、C=Q における OOM、server が capacity を clamp | その Q を oom/capacity-failed として保存し、より大きい Q は skip。max-num-seqs 等を下げて別条件の値を混ぜない。 |
| thermal/power violation が連続 10 sample | traffic を止め、5 分 cool-down、同一 Q を一度だけ再試行。再発なら thermal-or-power-limited として engine sweep を停止し、clock/power 設定を変更しない。 |
| median から 10% 超の outlier | raw を保持し、同一条件で一度だけ追加 trial。置換前後とも report し、恣意的に削除しない。 |
| profiler/counter/monitor metadata が不足 | logical metric と HTTP result は続行できる。physical HBM/L2 または telemetry 該当欄を unconfirmed とする。 |
| hard cap 到達 | source edit、タグ探索、再構成に延長しない。status=timed_out と evidence を残して次の優先 phase または終了へ進む。 |

最後に result directory 自体を hash 化し、GPU process が残っていないことを確認する。

~~~bash
amd-smi process --gpu 0 --json | tee "$RESULTS/hardware/process-after-all.json"
find "$RESULTS" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"$RESULTS/SHA256SUMS"
du -sh "$RESULTS" | tee "$RESULTS/result-size.txt"
~~~

## 9. 実行前 checklist

- [ ] uLLM の existing checklist 用 bundle と、この external workload artifact を分離して hash 固定した。
- [ ] local FP8 transfer または persistent-volume HF download を GPU lease 前に完了し、FP8 4 shard/tokenizer と Q8_0 GGUF を照合した。
- [ ] 1 GiB 以上の同一路 transfer を計時し、network estimate を実測で更新した。
- [ ] vLLM/SGLang/llama.cpp の release、image digest、source commit、help output を lease 時点で保存する。
- [ ] AMD SMI で SKU、XCD、NPS、memory/accelerator partition、空 GPU process を確認する。設定変更はしない。
- [ ] uLLM preflight -> fragment/lane を先に実行し、A′ fail でも external phase を捨てない。
- [ ] 各 engine で cache 無効化、1024 prompt token、20 output token、stream boundary、C=128 capacity を admission する。
- [ ] C=1..128、prefill/decode、3 trial、48 raw HTTP trial/engine を残す。
- [ ] format 差、KV dtype、logical-vs-physical bandwidth の境界、unconfirmed 項目を結果に明記する。
