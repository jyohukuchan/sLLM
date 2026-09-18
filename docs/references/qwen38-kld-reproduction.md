# Qwen3.8-27B KLD 再現手順
この資料は、[完了計画](../plans/archive/2026/09/11-20/qwen38-cross-engine-kld.md) と
[測定履歴](../history/2026/09/11-20/qwen38-cross-engine-kld.md) に記録した比較を、固定済みの
local artifact から再現するための最小手順である。全 engine は同じ teacher-forced token 列から
raw FP32 full-vocabulary logits を保存し、`qwen38_kld_compare.py` が llama.cpp BF16 を基準に
FP64 で `KL(reference || candidate)` を計算する。sampling は行わない。
## 固定入力と identity

```bash
export REPO=/home/homelab1/coding-local/sLLM
export ROOT=/home/homelab1/datapool/qwen38-kld-20260918
export SOURCE=/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-BF16
export LOCK=$REPO/docs/models/locks/qwen3.8-27b-bf16.json
export MANIFEST=$ROOT/cases-v1.json
export LONG_MANIFEST=$ROOT/cases-long-v1.json
export VOCAB=$ROOT/vocab-v1.json
```
`cases-v1.json` は8本の主 case と英語 repeat の9 case（主集計2632位置）、
`cases-long-v1.json` は4097-tokenの自然文/code各末尾257位置、`vocab-v1.json` は248077 valid
token IDsである。manifestのsha256、各出力行の入力token hash、raw file hashは各manifestと
`$ROOT/capture-audit.json` に残る。
平均KLDは評価位置ごとの平均であり、case別結果とcaseを等重みとした平均は個別の比較JSONにも保存する。
モデルは `Qwen/Qwen3.8-27B` の resolved revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` に固定する。lock本体とsource file hashesは
`$LOCK` と `$ROOT/bf16-source-identity.json`、FP8 sourceの全file/LFS検証は
`$ROOT/fp8-source-identity.json`、EXL3 artifact revisionは
`$ROOT/exl3-model-identities.json` を正とする。runner/binary hashは
`$ROOT/initial-runner-identities.json`、NV実行引数は
`$ROOT/nv-phase2-execution.json`、EXL3 KV実行引数は
`$ROOT/exl3-kv-matrix-execution.json` にある。
container image identityを再取得する場合は次を実行する。
```bash
docker inspect qwen38-kld-reference qwen38-kld-exl3 qwen38-kld-vllm \
  --format '{{.Name}} image={{.Config.Image}} digest={{.Image}}'
```

作業終了時はこれらの専用containerを停止する。再実行前に`docker start`で必要なcontainerを起動する。
測定時のimage ID・mount・関連環境変数は`$ROOT/runtime-final-identity.json`にも保存している。
測定時の reference/EXL3 image は `rocm-exl3-investigation:tested`、digestは
`sha256:43d75307b3045d3146014b2751bb097cd3df58829a0a5280b173a6930eea398e`、vLLM imageは
`vllm/vllm-openai-rocm:latest`、digestは
`sha256:950cac5145672c33bfee958f8b3d93457c5963a6ba77df5ab7a5905f92967d6b` である。
## BF16 llama.cpp reference

V620二台を同時に見せ、BF16 GGUF、layer split、FP16 KV、batch/ubatch 64、Flash Attentionを
固定する。これは `llama-kv-stage-status.json` の実行形に合わせたコマンドである。
```bash
docker exec qwen38-kld-reference /work/qwen38_kld_llama \
  --model /baseline-model/Qwen3.8-27B-BF16.gguf \
  --manifest /work/cases-v1.json --output-dir /work/results/replay-llama-bf16-fp16 \
  --device ROCm0,ROCm1 --tensor-split 1,1 --split-mode layer \
  --n-gpu-layers 999 --n-ctx 2048 --n-batch 64 --n-ubatch 64 \
  --cache-k f16 --cache-v f16 --flash-attn on --threads 8 --threads-batch 8
```
long caseはmanifest/outputを `cases-long-v1.json`/`replay-llama-bf16-fp16-long` に変え、
`--n-ctx 5120` とする。chunk1 controlは短いmanifest/outputを変え、`--n-batch 1 --n-ubatch 1`
を指定する。Q8_0/Q4_0 KVは同じコマンドで `--cache-k` と `--cache-v`、outputを変更する。
## EXL3

EXL3は `qwen38-kld-exl3` 内の `exllamav3` と単一可視のR9700
`GPU-a8e9ddefa2d60f55` を使う。plain 3/4/5bpwの実行例は次の通りで、各 bpw の
model/output directoryだけを対応する既存名へ変える。
```bash
docker exec --env ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
  --env LD_PRELOAD=/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1 \
  --env LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  --env PYTHONPATH=/exl3-runtime/lib-autotune:/src --env PYTHONUNBUFFERED=1 \
  qwen38-kld-exl3 python3 /work/exl3_dump.py \
  --model /work/models/exl3-4.00bpw --manifest /work/cases-v1.json \
  --output-dir /work/results/replay-exl3-4-fp16 --chunk 64 --context 2048 \
  --k-bits 16 --v-bits 16
```
KV行は同じ wrapper に `--k-bits 8 --v-bits 8`、`4/4`、4bpwの追加 `6/6` または `4/8` を渡す。
long行は `cases-long-v1.json`、`--context 5120`、freshな `replay-*` output名を使う。output directory
は毎回新規にし、既存 `manifest.json` を上書きしない。
## sLLM NVFP4 と MX

ROCm 7.14.0のhost buildでは、対象GPUごとにexact targetを固定する。build identityとhashは
`$ROOT/initial-runner-identities.json` および build logを併読する。
```bash
cd "$REPO"
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
SLLM_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
SLLM_HIP_CODEGEN_FEATURES='co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0' \
SLLM_HIP_TARGET=gfx1201 CMAKE_HIP_ARCHITECTURES=gfx1201 \
SLLM_ENABLE_HIP_RUNTIME=1 SLLM_ENABLE_PUBLIC_HIP_RUNTIME=1 \
CARGO_TARGET_DIR=$REPO/.local-artifacts/lowp-boundary/build/sLLM/gfx1201 \
CARGO_BUILD_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 LD_LIBRARY_PATH=/opt/rocm/lib \
cargo build --release -p sllm-hip \
  --bin sllm-qwen38-kld-dump --bin sllm-qwen38-mx-kld-dump
```
MXのgfx1030 buildは同じ環境で `SLLM_HIP_TARGET`/`CMAKE_HIP_ARCHITECTURES` と
`CARGO_TARGET_DIR` を `gfx1030` に変える。NVFP4実行は次の形で、対象binary・target・UUIDを
一致させる。
```bash
ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 LD_LIBRARY_PATH=/opt/rocm/lib \
python3 $REPO/ci/tools/qwen38_kld_sllm.py \
  --binary $REPO/.local-artifacts/lowp-boundary/build/sLLM/gfx1201/release/sllm-qwen38-kld-dump \
  --model-root /home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4 \
  --manifest $MANIFEST --output-dir $ROOT/results/replay-sllm-nvfp4-fp16-gfx1201-chunk32 \
  --target gfx1201 --device-index 0 --kv fp16 --chunk-size 32
```
MX GGUFは実際の3.8 BF16 cacheから、kindごとに次のhost-only converterで作成する。

```bash
cd "$REPO"
cargo run --release -p sllm-cli --bin sllm-convert-qwen38-mx -- \
  --kind mxfp6 --lock $LOCK --cache $ROOT/qwen38-bf16-cache \
  --output $ROOT/sllm-mxfp6/replay-Qwen3.8-27B-MXFP6.gguf \
  --derived-lock $ROOT/sllm-mxfp6/replay-Qwen3.8-27B-MXFP6.derived-lock.json
# --kind mxfp8、sllm-mxfp8/replay-Qwen3.8-27B-MXFP8.gguf と対応するderived-lockへ置換する
```
`$ROOT/qwen38-bf16-cache` はlockに列挙されたファイルだけを置くclean cacheである。raw HF
cacheをそのまま渡してextra filesを含めると、`verify_cache` の exact file-set検査で拒否される。
converterはQwen3.8 lock/fingerprintを必須にし、derived lockは `qwen38:<fingerprint>` を持つ。
GGUFのlegacy `qwen35` metadata prefixまたは `qwen35-compatible-qwen38` は共有実装上のrecipe名であり、
Qwen3.5 artifactへ読み替えない。
MX runは `qwen38_kld_sllm_mx.py` を介し、測定binaryが wrapper default（target/release）外の `.local-artifacts` 配下のため `--worker` を明示する。
MXFP6は `GPU-08b2ddcbd6e6b36c`、MXFP8は
`GPU-76a08c022586fed6`、target `gfx1030`、logical device 0を使う。
```bash
ROCR_VISIBLE_DEVICES=GPU-08b2ddcbd6e6b36c LD_LIBRARY_PATH=/opt/rocm/lib \
python3 $REPO/ci/tools/qwen38_kld_sllm_mx.py \
  --source-root $SOURCE --manifest $MANIFEST --worker $REPO/.local-artifacts/lowp-boundary/build/sLLM/gfx1030/release/sllm-qwen38-mx-kld-dump \
  --output-dir $ROOT/results/replay-sllm-mxfp6-fp16-gfx1030-chunk32 \
  --kind mxfp6 --gguf $ROOT/sllm-mxfp6/replay-Qwen3.8-27B-MXFP6.gguf \
  --derived-lock $ROOT/sllm-mxfp6/replay-Qwen3.8-27B-MXFP6.derived-lock.json \
  --model-lock $LOCK --target gfx1030 --device-index 0 --kv fp16 --chunk-size 32
```
`mxfp8`、E4/E5 KV、chunk64は `--kind`、`--gguf`、UUID、`--kv`、`--chunk-size` と outputを
対応する既存名へ揃える。longは `--manifest $LONG_MANIFEST` と専用outputを使う。

## vLLM FP8

working profileは `qwen38-kld-vllm`、R9700単一可視、vLLM `0.21.0`、TP1、BF16 model dtype、
CPU weight offload 8 GiB、GPU memory utilization 0.8、eager、chunk32、FLA autotune singleである。
KVのCLI要求はBF16でもadapterがengine引数を `auto` に変換し、manifestでresolved BF16を記録する。

```bash
docker exec qwen38-kld-vllm python3 /work/qwen38_kld_vllm.py \
  --model-root /model --manifest /work/cases-v1.json \
  --output-dir /work/results/replay-vllm-fp8-bf16 --tensor-parallel-size 1 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.8 --cpu-offload-gb 8 \
  --max-num-batched-tokens 32 --fla-autotune single --max-num-seqs 1 \
  --enforce-eager
```
containerの `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`、`VLLM_USE_V2_MODEL_RUNNER=0`、
`PYTHONPATH=/work` 等はimage環境とmanifestを確認する。FP8 E4M3 KVは `--kv-cache-dtype fp8_e4m3`
へ変え、longはmanifest/outputと `--max-model-len 4099`を使う。

E5M2はexperimentalである。現在のR9700 NVFP4 gfx1201はE5 KVを
`OcpE5M2 is incompatible with target gfx1201`として拒否し、成功数へ入れない。vLLMのFP8 checkpoint
でもfull-model E5 KVはcheckpoint guardが拒否する。writerのE4/E5独立CPU oracle
`$ROOT/vllm-kv-writer-independent-oracle.json` がPASSしても、full checkpoint実行の証拠にはしない。
拒否ログは `$ROOT/results/sllm-nvfp4-kv-mxfp8-e5-gfx1201-chunk32/adapter.stderr` と
`$ROOT/logs/vllm-fp8-e5m2-patched-full.log` を保持する。
NVFP4のE5結果は、binary・targetをgfx1030、UUIDを`GPU-08b2ddcbd6e6b36c`へ変更し、
同じV620のFP16 controlと組にして取得した。詳細は`$ROOT/results/sllm-nvfp4-e5-gfx1030-matrix.json`にある。
## 比較、audit、summary

各比較は同一scopeのmanifestとvocabを渡し、outputを新規名にする。
```bash
python3 $REPO/ci/tools/qwen38_kld_compare.py \
  --reference $ROOT/results/llama-bf16-fp16 \
  --candidate $ROOT/results/replay-sllm-mxfp6-fp16-gfx1030-chunk32 \
  --inputs $MANIFEST --vocab $VOCAB \
  --output $ROOT/results/replay-kld-sllm-mxfp6-fp16-gfx1030-chunk32-vs-llama-bf16.json \
  --rows-per-block 8
```
同一形式のKV差はcandidateをKV output、referenceをFP16 controlにする。MX chunk差はserial1
または同形式controlをreferenceにする。long比較は両方を `LONG_MANIFEST` と
`llama-bf16-fp16-long` に置き換える。

full raw bytes/input mappingのauditは次で行う。現在のcapture setにpending variantがある場合は
`pending`を成功へ読み替えない。
```bash
python3 $REPO/ci/tools/qwen38_kld_audit.py --root $ROOT
python3 $REPO/ci/tools/qwen38_kld_summary.py --root $ROOT
jq '{expected_captures,validated:(.validated|length),pending,unavailable}' $ROOT/capture-audit.json
```

最後に `$ROOT/nv-phase2-execution.json`、`$ROOT/exl3-kv-matrix-execution.json`、
`$ROOT/logs/status/phase2-ledger.json` のstate/returncode/引数と、各 output manifest の
nonfinite count、HIP dispatch/fallback欄を確認する。`SESSION.md` は再開メモであり、実プロセスと
execution ledgerを優先する。
