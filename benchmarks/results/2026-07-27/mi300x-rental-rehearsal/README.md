# SQ8_0 CDNA3 MI300X レンタル手順リハーサル（2026-07-27）

## 結論

gfx942 非搭載の開発機で実行できる runner の全段階を、実際の
`tools/run-sq8-cdna3-mi300x-validation.sh` で通した。CPU、generic `SQ8_0`
HIPRTC 27 kernel compile、gfx942 feature build、ISA audit は pass した。physical
は gfx942 がないため **非成功** として明示的に停止した。gfx942 実機での A′/B
physical pass は、当然ながらまだ主張しない。

最終 runner revision は `3f0243fa3b40e570810baa971563c54195bf025e`。この
revision の clean worktree で中断・再開も実地確認した。

## 実行条件と GPU 安全性

- 開発機の `rocminfo` 観測 arch は `gfx10,gfx1030,gfx12,gfx1201` であり、
  `gfx942` はない。
- CPU/HIPRTC/build/ISA は `HIP_VISIBLE_DEVICES=-1` で実行した。
- R9700 の lock は他作業が保持していたため、physical binary を起動して R9700
  を触ることはしなかった。`--rehearsal-no-gfx942` は preflight の観測を stamp に
  保存し、physical を binary 起動前に expected failure とする。
- `ullm-openai.service` は起動していない。

## 実測結果

`patched-all/stage-timings.tsv` は、jobs=32・新規 target directory での full
rehearsal である。physical の 0 秒は GPU を使わず marker で止めた時間である。

| stage | 結果 | 実測 | 証跡 |
| --- | --- | ---: | --- |
| preflight | pass（rehearsal-only） | 0 s | `patched-all/logs/preflight.log` |
| CPU | pass | 79 s | `patched-all/logs/cpu.log` |
| HIPRTC | pass、27/27 | 18 s | `patched-all/logs/hiprtc.log` |
| build | pass | 54 s | `patched-all/logs/build.log` |
| ISA | pass、2 CCOB / 912 MFMA | 6 s | `patched-all/logs/isa.log` |
| physical | expected fail、P0 非成功 | 0 s | `patched-all/logs/physical.log` |

offline pass 部分の合計は **157 s (2 min 37 s)**。これは 64-core Threadripper の
ローカル測定であり、13 vCPU だった rental host へそのまま外挿しない。一方、初回
rental の successful release build は 52.57 s で、今回の 54 s と整合する。

ISA の static evidence は `patched-all/isa/summary.txt` と
`patched-all/isa/gfx942-ck-resource-metadata.tsv` に残す。検出値は 2 gfx942 code
object、`v_mfma_f32_16x16x32_fp8_fp8` 合計 912（456 + 456）、CK GEMM 120 entry、
最大 VGPR 454 / SGPR 62 / AGPR 198 / LDS 49,152 B、private/spill 0 である。
実効 occupancy/residency は gfx942 実機なしには未確認である。

## 期待した failure の実行確認

通常 mode で `normal-no-gfx/` に preflight と physical を実行した。

- preflight: `rocminfo did not report required gfx942 (observed:
  gfx10,gfx1030,gfx12,gfx1201); this host cannot run the physical P0 gate.`
- physical: `physical is blocked because preflight found no required gfx942 device ...
  This is a fail-closed non-success.`

`--rehearsal-no-gfx942 --stage all` では offline stages のためだけに preflight を
完了扱いにし、physical は
`physical is deliberately blocked: --rehearsal-no-gfx942 observed no gfx942 ...`
で止める。この mode は local rehearsal 専用であり P0 pass にはならない。

`missing-linker/` では `ULLM_RENTAL_LINKER=/definitely/not/a/linker` を指定し、
preflight が build 前に `required rental linker is unavailable` と fail することを
確認した。

## Cargo linker/mold の実測

Rental runner の CPU/build log は Rust invocation が `-C linker=cc` であり、
`-fuse-ld=mold` を含まない。`.cargo/config.toml` は変更していない。

対照として `local-default-config/` に同じ clean feature build を runner override
なしで実行した。53 s で pass し、verbose log に `-C linker=clang` と
`-C link-arg=-fuse-ld=mold` の両方が残っている。従って rental override は
process-local で、ローカルの clang+mold 設定を落としていない。

preflight は `cargo`、`rustc`、選択された `ULLM_RENTAL_LINKER`、C++/ROCm/LLVM/
`rocminfo`/`zstd` を必須検査する。`clang`、`mold`、`rustup` は optional status
として記録する。これらは runner が使わないため、`cc` と `rustc` があれば
不在でも P0 を妨げない。

`optional-tools-absent-clean/` では、`clang`、`mold`、`rustup` を PATH から外し、
必須 `cc`/`cargo`/`rustc`/ROCm tools だけを渡した preflight を実行した。stage は
pass し、三つすべてを `unavailable (not required by the rental runner)` と明示した。

## 冪等性と中断再開

- 同一 `patched-all/` directory に full command を再実行すると、preflight、CPU、
  HIPRTC、build、ISA はすべて `SKIP (already complete)` となり、未完の physical
  だけを再試行した。`--stage cpu` 単独の再実行も `SKIP cpu` だった。
- `interrupt-resume/` では clean worktree の CPU compile 中に runner process group
  へ SIGTERM を送り、exit 143 を得た。この時点では `preflight.done` のみが残り、
  `cpu.done` と CPU success timing は存在しなかった。
- 同じ command を再実行すると `SKIP preflight` の後に CPU 73 s、HIPRTC 18 s、
  build 53 s、ISA 6 s が pass し、physical は expected failure になった。
  `resume-command.summary` の wall time は 150 s である。

## 取得物・ネットワーク前提

P0 runner 自体は model download、container pull、`pip`、`rustup`、server 起動を
行わない。デフォルトは Cargo `--offline` で、`--allow-network` を明示した場合だけ
Cargo の stage command が network を使える。

新規 runner target directory は debug + release で最大 4.8 GiB（local default
config の release-only target は 2.0 GiB）まで成長した。これは rental host の
disk reservation に含める。rehearsal 後は logs、timings、environment、ISA summary/
resource TSV を残し、再生成可能な Cargo target と extracted code-object files は
整理した。

| 区分 | P0 で必要か | size / 測定 | 扱い |
| --- | --- | --- | --- |
| tracked source checkout | 必須 | local tracked tree 24,217,770 B | lease 前に clean checkout を作る。clone transfer size は未確認。 |
| Cargo lock registry archives | 必須 | 29 archive、2,590,666 B (2.471 MiB) | `cargo fetch --locked` を lease 前に実行し、直後に `cargo fetch --locked --offline` を pass させる。local fetch は 1 s 未満（整数秒記録 0 s）。 |
| Rust/ROCm/C++ tools | 必須 | provider image size は未確認 | runner は install しない。`cargo`/`rustc`/selected `cc` と ROCm tools を preflight で止める。 |
| Qwen3-14B-Q8_0.gguf | 不要 | 15,698,533,728 B | P0 から除外。 |
| Qwen3-Coder-Next-FP8 | 不要 | 80.4 GB / 52 files | P0 から除外。 |
| Qwen3-30B-A3B-FP8 | 不要 | 32.5 GB / 17 files | P0 から除外。 |
| Qwen3.6-35B-A3B-FP8 | 不要 | 37.5 GB / 56 files | P0 から除外。 |
| vLLM/SGLang/llama.cpp images | 不要 | saved logs に byte size・download time はない | P0 から除外。 |

初回 evidence にある Qwen3-14B-FP8 と container image の正確な bytes は未確認で
あり、推測値は書かない。上の非 P0 model 群だけで約 166.1 GB（記録された十進 GB
表記の合計）なので、借用後の取得対象にしてはならない。

## 借用時間の判断

cache/toolchain/source を lease 前に準備し、offline fetch verification まで済ませた
場合、P0 の runner 本体は **2 時間で判定可能** と判断する。1 時間は、preflight が
数分で通り physical が初回または 1 回の同条件再現で結論を出せる場合のみ可能で、
保証しない。source/Cargo cache/toolchain のいずれかを借用後に作る cold start は
1 時間では不可、2 時間でも保証しない。

時間が足りない場合でも preflight、CPU、HIPRTC、build、ISA、B を skip しない
physical A′/B は捨てない。捨てる順は profiler/occupancy query、full-model、
artifact transfer、model/container download、external engine、timing sweep、
hand-written A である。P0 failure では source を借用中に編集せず、同一条件の
再現を 1 回だけ残して終了する。
