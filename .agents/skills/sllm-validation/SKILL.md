---
name: sllm-validation
description: sLLM の変更範囲に応じて、既存の host、HIP compile-only、GPU correctness、performance 検証の入口を選ぶ。
---

# sLLM 検証

変更の影響範囲に合わせて、必要な行だけを選ぶ。全 suite の実行や GPU の再実行を既定の gate にしない。

以下のコマンドはrepository rootから実行する例。最初に現在の作業計画、対象model、変更したcrate／kernelを確認する。
Phase 7／Qwen3.5-4B用の行を、Qwen3.8や別modelの検証の代わりに使わない。
一般的なRust変更は該当crateのfocused test、文書変更はMarkdown・link検査から始める。

## Host

環境・shell・C++ host probeを確認するときは、GPUを起動しない次の入口を使う。

```bash
scripts/dev/check-environment.sh host
```

登録済みの CPU-only H0/H1/H2 を実行するときは、まず一覧を確認し、影響する行だけを選ぶ。

```bash
python3 ci/tools/run_local_verification.py --list
python3 ci/tools/run_local_verification.py \
  --rows h0 \
  --output-root .local-artifacts/ci/local-draft \
  --allow-dirty-local
```

影響に応じて `--rows h1` または `--rows h2` を追加する。dirty checkoutでは `--allow-dirty-local` を付ける。clean checkoutの厳密な証拠だけで `--strict` を使い、local draftをimmutable evidenceと呼ばない。

## Compile-only

HIP/C++またはtarget選択に触れる変更では、影響するtargetのcompile-onlyを選ぶ。次はgfx1030向けの例。

```bash
python3 ci/tools/run_phase7_compatibility_compile.py \
  --target gfx1030 \
  --output-dir /tmp/sllm-compile-gfx1030 \
  --allow-dirty-local
```

compile成功はGPU実行、数値正しさ、model動作、性能、互換性の証拠へ昇格させない。公開runtimeのH3 rowが対象なら、対応する `ci/tools/run_h3_compile.py` または `ci/tools/run_h3_public_runtime_compile.py` の `--help` とmatrixから影響する行を選ぶ。

## GPU correctness

GPUを使う前に、V620をQwenローカルsubagentが占有していないか確認する。

```bash
/home/homelab1/.local/bin/qwen38-subagent-server status
```

V620が必要でQwen serviceが動いていれば、idleであることを確認してから停止する。1台のV620へ縮退して続行しない。

```bash
/home/homelab1/.local/bin/qwen38-subagent-server stop
```

model-free G1の既存artifactを検証する入口は、canonical rowとstaged metadataを明示する。

```bash
python3 ci/tools/run_g1_evidence.py --help
```

identity引数はartifactの実際のbuild情報から埋める。dirty buildや未reviewのartifactに、便宜的なHEADをreviewed/tested SHAとして付けない。
draftでは対象kernelの既存testを使い、immutable identityを得るためだけのcommitや全面再buildを追加しない。

数値kernelは、対象GPUのexact targetに対応する独立NumPy oracleと照合する。`selected_backend=hip`、fallback未使用、zero selectionなし、timeout/crashなしを満たさない結果を `PASS` と扱わない。CPU fallbackやcompile-onlyをGPU correctnessへ読み替えない。

## GPU performance / full-model smoke

以下は既存Phase 7／Qwen3.5の観測例であり、全model共通の入口ではない。現在のtaskに対応するbinary／runnerを優先し、Phase 7が対象の場合だけ選択を生成して実行する。出力先はrunごとに分ける。

```bash
python3 ci/tools/phase7_lifecycle.py resolve \
  --event workflow_dispatch --requested-profile daily \
  --output /tmp/sllm-phase7-daily-selection.json
python3 ci/tools/run_phase7_gpu_observation.py \
  --selection /tmp/sllm-phase7-daily-selection.json \
  --output-dir /tmp/sllm-phase7-daily-observation \
  --model-cache /absolute/path/to/locked/model-cache \
  --allow-dirty-local
```

固定direct performance rowを個別に測る場合は、binary、build manifest、model lock、model cache、rowを全て明示する。

```bash
python3 ci/tools/run_engine_performance.py \
  --row engine-performance-direct-4b-gfx1030-short-odd \
  --binary /absolute/path/to/sllm \
  --build-manifest /absolute/path/to/build-identity.json \
  --model-lock docs/models/locks/qwen3.5-4b-bf16.json \
  --model-cache /absolute/path/to/locked/model-cache \
  --output-dir /tmp/sllm-performance-gfx1030
```

性能比較はmodel revision、GPU target、入力／出力長、数値型、binary、warmup／反復条件を合わせる。実測した差とばらつきを報告し、新しい必達倍率やhard gateを追加しない。
GPU runのraw trace、profile、生成binary、model payloadはrepositoryへ追加せず、既存runnerのcompact reportとdigestだけを必要な場所へ残す。

参照: `docs/development/environment.md`、`docs/development/testing.md`、`docs/development/local-qwen-subagent.md`、`docs/compatibility/software.md`。
