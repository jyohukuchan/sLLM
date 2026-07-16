# Qwen3.5 AQ4 P2 path oracle

## 前回の要点

Qwen3.5-9B AQ4 P2 source oracle v2 は、fixture 3-row（2 cases、step 2+1）の
独立 BF16 CPU forward として strict validator を通過している。path oracle は同じ
payloadを流用せず、AQ4 packageを1回ロードして bounded observation を取得する必要が
ある。現 product profile は `artifact=null` で、実在する package manifest だけがある。

## 今回の変更点

- `ullm-aq4-p2-path-oracle` 専用 Rust binary を追加した。既存の read-only calibration
  observer で final hidden 5点、logit先頭32点、全語彙top-k 10件だけを逐次保持し、
  all-M=1 の3-row JSONLをstdoutへ出す。
- `export-qwen35-aq4-path-oracle.py` を追加した。source manifestをstrict検証し、source
  payloadからgreedy token列だけをreplay入力として抽出し、path stdoutを厳密に再検証して
  package manifest SHA、実行binary SHA、source replay SHA、all-M=1/model_loads=1を
  runtime sidecarへ束縛する。source payloadのpath流用はしない。
- artifact manifestがないproductを、package SHAだけを束縛する明示的 package-only
  identity としてschema/validator/capture/linkへ追加した。artifact hashは `null` のまま、
  validatorは `valid` として再検証するが、`usable_as_path_evidence`、linkのusable、
  promotionはfalseに維持する。
- fake-binary testでreplay、bounded payload、package-only identity、link blockerを
  検証した。

CPU path実行は以下で試行した。

```text
python3 tools/export-qwen35-aq4-path-oracle.py \
  --package-dir /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package \
  --package-manifest /home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package/manifest.json \
  --allow-package-only --cases tests/fixtures/qwen35-aq4-p2-oracle/cases.json \
  --source-oracle benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2 \
  --tokenizer-root /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --output /tmp/qwen35-path-oracle-attempt --binary target/debug/ullm-aq4-p2-path-oracle \
  --timeout-seconds 120
```

CPU device 0 は model load 前に明確に拒否された。

```text
Qwen3.5 AQ4 model workspace admission failed:
workspace capacity is insufficient: required=8668919584 capacity=0
```

CPU fallbackの `total_global_mem=0` と既存production workspace admissionの組み合わせで
あり、production runtimeへcapacity overrideを入れずにbinary側だけで回避できない。
これはunsupported CPU path evidenceとして扱い、実行中のモデル常駐やRSS増加は無く、
`/tmp/qwen35-path-oracle-attempt` は作成されていない。

## 次の行動

same-artifact all-M=1 は、排他取得できる R9700 (`gfx1201`) で実行する。runtimeの
device indexはCPU=0、HIP V620=1/2、R9700=3である。推定package residentは約7.2 GiB
（tensors 4.4 GiB、passthrough 2.9 GiB）で、R9700 VRAMは約34.2 GB、実行前に既存使用
約7.4 GBを解放し、モデルとworkspaceのために少なくとも約10 GB（安全側に20 GB以上）を
確保する。出力先は以下とし、GPU processが完了してからstrict validator/linkを実行する。

```text
benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/path-oracle-v1/
benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/oracle-link-v1/
```

## GPU run and detached attestation

The single sanctioned GPU attempt ran from 11:54:00Z to 11:54:31Z with the
service stopped and the physical R9700 (`GPU[2]`, `gfx1201`) exposed as runtime
device index 1 (`HIP_VISIBLE_DEVICES=1`). VRAM was 87,384,064 bytes at baseline
and 7,343,022,080 bytes at peak, with a recorded power sample of 21 W; no OOM
occurred. Service recovery finished active with health true and `NRestarts=0`.
The raw command, monitor, device, baseline, and recovery evidence is hash-bound
under `path-oracle-gpu-run-v1/`.

The original `path-oracle-v1` output is immutable and metadata-invalid because
its runtime sidecar hard-coded `device=cpu`. A detached `path-oracle-v2` copy
records `device=gpu`, runtime index 1, visible-device mapping, and observed
`gfx1201`, while preserving the v1 manifest and payload. Its package-only
identity is explicitly bound to the active served-model manifest SHA and package
manifest SHA; the path validator is valid and usable as path evidence. The
detached attestation validator reconstructs raw SHA, GPU mapping, resources,
service recovery, prompt-plus-greedy context hashes, and replay hashes without
retaining token IDs.

Source/path bounded diagnostics are separate from the same-artifact P3 regression
gate. Replay alignment is exact (`step_i` uses prompt plus greedy tokens before
`i`). Bounded intersection metrics are hidden relative-L2 0.5452883336, hidden
cosine 0.9833240686, hidden max-abs 1.0846476555; logits relative-L2 0.6151289249,
logits cosine 0.9446401707, logits max-abs 8.3477816582; top-10 overlap mean
0.70/minimum 0.10. Greedy and top-k are not exact. The AQ4 threshold template is
unbound with null values, so `policy_missing` remains a blocker; no threshold is
inferred from this run. No further GPU run is permitted.
