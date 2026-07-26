# SQ8_0 CDNA3 MI300X A′/B 再検証チェックリスト v0.2

- Date: 2026-07-26（2026-07-27 rehearsal 更新）
- Status: gfx942 非搭載 host で runner の offline stages・fail-closed physical・冪等性・中断再開を実行確認済み。B control の根因は CPU oracle で特定・修正済みだが、**修正後の gfx942 実機確認は未実施**である。
- Scope: `SQ8_0` の gfx942 A′ bring-up と、その独立 B control の再検証を、次の MI300X 借用で 1--2 時間以内に判定するための手順。
- Out of scope: hand-written MFMA の経路 A、本番 dispatch、full-model enable、serving、release、campaign、authorization、`/etc/ullm/served-models/active.json`、systemd 操作。activation は本手順の範囲外であり、候補ができた場合は lightweight promotion policy に従う。

v0.1 は初回レンタルの設計・実測記録として残す。本書はその未解決点を
再試験するための実行可能な v0.2 runbook である。A′ は gfx942 bring-up
用の CK XDL 再利用経路であり、CDNA3 本番目標の経路 A を承認するものではない。

## 0. 2026-07-27 の実行 rehearsal

実行 receipts は
[`mi300x-rental-rehearsal`](../../benchmarks/results/2026-07-27/mi300x-rental-rehearsal/)
にある。gfx942 はない host なので、これは physical success の証明ではない。
ただし runner 自身を実行し、次を確認した。

| stage | local 実測（jobs=32、新規 target） | 結果 |
| --- | ---: | --- |
| preflight | 0 s | rehearsal mode では gfx942 不在を記録して offline continuation。normal mode は fail-closed。 |
| CPU | 79 s | pass |
| generic `SQ8_0` HIPRTC | 18 s | 27/27 pass |
| gfx942 feature release build | 54 s | pass |
| ISA/resource audit | 6 s | 2 CCOB、MFMA 912、pass |
| physical | 0 s | gfx942 不在を理由に expected fail、P0 非成功 |

offline pass 部分は 157 s だった。これは Threadripper の local measurement であり、
13 vCPU の rental host の wall-clock をそのまま表さない。一方、初回 rental の
successful release build は 52.57 s で、今回の 54 s と矛盾しない。

同じ results directory の rerun は complete stage を `SKIP` し、physical だけを
再試行した。さらに clean worktree で CPU compile 中に SIGTERM を送り、
`preflight.done` だけが残った後に再実行した。resume run は preflight を skip し、
CPU 73 s、HIPRTC 18 s、build 53 s、ISA 6 s を通過して expected physical failure
まで到達した。従って resume は記述だけでなく実地確認済みである。

## 1. 初回レンタルから確定していること

初回の evidence は
[`mi300x-rental-v1`](../../benchmarks/results/2026-07-26/mi300x-rental-v1/)
にある。MI300X VF 1 台、`gfx942:sramecc+:xnack-`、ROCm 7.2.4、NPS1/SPX
で次を観測した。

| 項目 | 結果 | 扱い |
| --- | --- | --- |
| A′ fragment/lane probe | 256 lane/register coordinate の全単射、logical matrix pass | 実機 sub-gate pass |
| A′ の実形状 5 case | CPU expectation と全 case `max_abs=0` | 実機 sub-gate pass |
| B control | `k_or_v_tail_id1`: expected `0.53125`、observed `0.03125` | 未解決だった。A′ pass ではない |
| A′ timing | M=128 gate/up full: 249.415 TFLOPS、M=1 gate/up tail: 3,019.8 GB/s | projection-only。full-model/HBM 効率ではない |
| occupancy/residency | retained evidence なし | 未確認 |
| full model | 実行なし | 未確認 |

失敗時 B の独立した raw stderr は保存済みファイル群からは見つからず、値は
`README.md` と `env.txt` の記録で確認した。したがって、初回の `id0` との
比較を後から捏造しない。`id1` は smoke case/instance のラベルであり、保存済み
evidence は「id0 が正しい」ことを示していない。

## 2. B control の根因とオフライン再現

### 2.1 根因: hipBLAS の row-major weight view が誤っていた

B は OCP E4M3FN を BF16 に dequant して hipBLAS F32 GEMM を行う control
であり、FNUZ prepack を通らない。従って OCP/FNUZ の bias 差や x2/x4 scale
補償は今回の B failure の原因ではない。

物理配置は row-major `W[N,K]` と row-major `A[M,K]` である。hipBLAS を
column-major として使うと、出力は `C^T = W * A^T`、すなわち column-major
`C[N,M]` になる。この buffer は求める row-major `C[M,N]` と同じ memory
layout である。ここで `W` の buffer は column-major には `W^T[K,N]` と見える
ため、正しい first operand は `OP_T`、`lda=K` である。GEMM contract 全体は
`m=N, n=M, k=K, lda=K, ldb=K, ldc=N` になる。

旧呼び出しは `OP_N, lda=N` だった。これは `W[column + k*N]` を読み、row-major
`W[column*K + k]` ではない strided permutation を使う。そのため tail fixture
の `k=4992` にある寄与を読み落とした。

`k_or_v_tail_id1` の output `[0,0]` では、K=0 の寄与が
`(0.5 * 0.25) * (0.5 * 0.5) = 0.03125`、final K128 block の正しい寄与が
`(0.5 * 2.0) * (0.5 * 1.0) = 0.5` である。旧 view の final-K address
`4992 * 1024 = 5,111,808` は fixture 内の `W[998,2048]`（zero）を指すため、
前者だけが残る。従って観測 `1/32` と期待 `17/32` の差が正確に `16/32 = 0.5`
になる。これは丸め、tail mask、FNUZ scale の推測ではなく address calculation
で説明できる決定的な差である。

修正は
[`sq8_ck_gfx942_control.hip.cpp`](../../runtime/src/sq8_ck_gfx942_control.hip.cpp)
で `HIPBLAS_OP_T` と `lda=K` を使い、上記の全 GEMM dimension/leading-dimension
contract を named variables に固定したもの。

### 2.2 CPU oracle の証跡

[`sq8_gfx942_aprime.rs`](../../crates/ullm-engine/src/sq8_gfx942_aprime.rs)
に、実機 smoke の完全な M=1/N=1024/K=5120 fixture と hipBLAS の column-major
address calculation を再現する oracle を追加した。旧 call は `0.03125`、正しい
control reference と修正 call はともに `0.53125`、差は厳密に `0.5` を assert
する。さらに corrected GEMM contract (`m=N,n=M,k=K,lda=ldb=K,ldc=N`) も test
で固定する。

ローカル実行済み:

```bash
CARGO_BUILD_JOBS=8 \
  cargo test -p ullm-engine \
  b_control_hipblas_layout_oracle_reproduces_the_mi300x_tail_delta \
  --lib -- --nocapture
# 1 passed; 0 failed
```

これは CPU oracle の pass であり、hipBLAS を gfx942 上で実行した確認ではない。
次のレンタルで B を skip せず 5 shape の B/CPU comparison を通すまで、B は
**実機未確認**である。

## 3. オフラインで閉じた gfx942 範囲

### 3.1 native A′/B build

次の feature build を `GPU_ARCH=gfx942`、`HIP_VISIBLE_DEVICES=-1`、jobs=8 で
完走した。A′ CK wrapper と修正後 B hipBLAS wrapper の双方が hipcc/link を通る。

```bash
GPU_ARCH=gfx942 ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=-1 \
  CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR="$PWD/target/cdna3-gfx942-offline" \
  cargo build --release -p ullm-engine \
    --features rocm-ck-gfx942-aprime \
    --example sq8_gfx942_aprime_physical_smoke
```

この feature は native A′/B wrapper を追加するだけである。full-model の gfx942
profile を有効化した、という意味ではない。

### 3.2 generic SQ8_0 HIPRTC compile audit

`tools/sq8-cdna3-hiprtc-audit.cpp` は runtime の `HipRtcRuntime` を直接使い、
GPU を enumerate/open/launch せず exact HIPRTC source/options で gfx942 compile
を行う。次の **27/27** をローカルで pass した。

| 分類 | pass した runtime kernels |
| --- | --- |
| basic | `matvec_bf16_f32`, `top1_f32`, `rmsnorm_f32`, `add_f32`, `rope_f32` |
| norm/activation | `segmented_rmsnorm_f32`, `segmented_rmsnorm_silu_mul_f32`, `silu_mul_f32`, `sigmoid_mul_f32` |
| attention | `causal_attn_f32`, `causal_attn_f32_flash2`, `causal_attn_batch_f32`, `causal_attn_batch_f32_flash2`, `cached_prefix_attn_f32`, `cached_prefix_attn_f32_flash2` |
| SQ8_0 matvec | `sq_fp8_matvec_f32`, `sq_fp8_matvec_batch_f32`, `sq_fp8_matvec_pair_f32`, `sq_fp8_matvec_triple_f32` |
| paged/Qwen3.5 | `paged_decode_attn_f32`, `paged_kv_write_f32`, `paged_chunk_f32`, `qwen35_split_q_gate_f32`, `qwen35_qk_norm_rope_f32`, `qwen35_qk_norm_rope_batch_f32`, `qwen35_qk_norm_rope_paged_kv_write_f32`, `depthwise_conv1d_f32` |

これは初回の A′ projection-only static check から、attention、normalization、
activation、matvec、paged path の generic HIPRTC source compile まで範囲を広げた。
ただし HIPRTC compile pass は launch、numerics、full-model dispatch を証明しない。
`rmsnorm_shuffle_prototype`、`segmented_rmsnorm_silu_mul_shuffle_prototype`、
`paged_causal_gqa_chunk_wmma_kernel` は architecture-specialized prototype であり、
この generic current path の合格対象から意図的に除外した。

現行 model head は R9700/gfx1201 identity を fail-closed で要求し、layer profile
にも gfx942 A′ full-model profile はない。この integration gate は本タスクで
は変更していない。従って「generic kernels compile」と「SQ8_0 full model が
gfx942 で動く」は別であり、後者は未実装・未検証である。

### 3.3 ISA と static resource audit

`tools/audit-sq8-cdna3-gfx942-isa.sh` は linked physical-smoke binary の
`.hip_fatbin` から CCOB を取り出し、zstd 展開・gfx942 HSACO unbundle・disassemble
を行う。`GetTypeString` の selected A′ 4 instance contract を binary 内で assert
し、`v_mfma_f32_16x16x32_fp8_fp8` を必須にする。ローカル run では 2 gfx942 code
object に計 **912** 個（各 456 個）を検出した。

同じ audit は AMDGPU metadata の CK GEMM 120 entry を TSV で保存する。今回の
範囲は最大 VGPR **454**、SGPR **62**、AGPR **198**、LDS **49,152 B**、private
segment 0、VGPR/SGPR spill 0、workgroup 256 だった。ROCm の
`rocwmma/internal/constants.hpp` が示す LDS max 65,536 B 以下なので、static
metadata 上の single-workgroup LDS fit は pass した。

ここで言えるのは「spill/private allocation と一 workgroup の LDS 超過で即座に
launch/occupancy がゼロになる状態ではない」までである。VGPR/SGPR allocation、
XCD/NPS partition、runtime loader が決める active blocks/CU は static metadata
だけからは導けない。実効 occupancy/residency は引き続き **未確認**であり、次回
レンタルで実 function を対象に HIP occupancy API を取る必要がある。

## 4. Cargo linker/mold の恒久対策

ローカルの [`.cargo/config.toml`](../../.cargo/config.toml) は変更しない。
開発機の `linker = "clang"` と `-fuse-ld=mold` による速度を保つためである。

レンタル runner は Cargo environment override をその process だけに適用する。

```text
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=cc
CARGO_ENCODED_RUSTFLAGS=
```

`ULLM_RENTAL_LINKER` / `ULLM_RENTAL_ENCODED_RUSTFLAGS` を明示すれば provider の
正しい toolchain に差し替えられる。これにより `.cargo/config.toml` を退避・編集
する手作業は不要である。`--locked` と default `--offline` も runner が付ける。

この override を使う runner の local `build` stage は gfx942 feature build を
54 s で完走した（新規 target directory、jobs=32、GPU 実行なし）。verbose Rust
invocation は `-C linker=cc` であり、`-fuse-ld=mold` は含まなかった。対照として
override なしの local build も 53 s で pass し、`-C linker=clang` と
`-C link-arg=-fuse-ld=mold` の両方を確認した。従って rental process は local の
clang+mold 設定を変更しない。

preflight は `cargo`、`rustc`、選択された `ULLM_RENTAL_LINKER`、C++/ROCm/LLVM/
`rocminfo`/`zstd` を必須検査する。`clang`、mold、`rustup` は optional status として
environment receipt に記録する。runner は `cc` と既存 Rust toolchain を使うため、
この三つの不在自体は P0 の blocker ではない。`ULLM_RENTAL_LINKER` が存在しない
simulation は build 前に `required rental linker is unavailable` で fail した。
さらに `clang`、mold、`rustup` を PATH から外した preflight は pass し、三つを
`unavailable (not required by the rental runner)` と receipt に出力した。

## 5. 次回レンタルの 1 本 runner

実行ファイルは
[`tools/run-sq8-cdna3-mi300x-validation.sh`](../../tools/run-sq8-cdna3-mi300x-validation.sh)
である。次のように repository root から実行する。

```bash
bash tools/run-sq8-cdna3-mi300x-validation.sh \
  --repo "$PWD" \
  --results-dir "$PWD/benchmarks/results/2026-07-29/mi300x-rental-v2" \
  --jobs 8 \
  --hip-visible-devices 0
```

### 5.1 runner contract

1. `preflight -> cpu -> hiprtc -> build -> isa -> physical` の P0 順で実行する。
   physical 単独実行は、前 5 stage の `.done` stamp がなければ fail-closed する。
   normal preflight が gfx942 不在で拒否した場合、physical もその device admission
   を明記して非成功にする。
2. stage ごとに `logs/<stage>.log`、`state/<stage>.done`、`stage-timings.tsv` を
   保存する。失敗後に同じ command を再実行すると pass 済み stage は skip し、
   failure stage から再開する。
3. `state/revision.txt` は HEAD と tracked diff SHA-256 を結ぶ。source が変わった
   state directory の再利用は拒否する。
4. preflight/CPU/HIPRTC/build/ISA には `HIP_VISIBLE_DEVICES=-1` を渡す。GPU を
   触るのは physical stage だけである。
5. physical stage は `ULLM_SMOKE_SKIP_B_CONTROL` を `env -u` で消す。B failure
   を A′ only pass に読み替える経路はない。
6. runner は model download、container pull、pip/rustup、server 起動、`/etc`、
   service、activation を行わない。ネットワークは既定で禁止し、必要時だけ
   `--allow-network` を人間が明示する。
7. `--rehearsal-no-gfx942` は local rehearsal 専用である。gfx942 不在を state に
   記録して offline stage を通すが、physical は GPU binary を起動せず expected
   failure とする。この option を使った run は P0 pass ではない。

normal mode と rehearsal mode の preflight、CPU、HIPRTC、build、ISA、physical stage
は gfx942 非搭載の local host で実行済みである。physical は実機がないため成功を
試しておらず、明示的な fail-closed behavior だけを確認した。

### 5.2 GPU lease 前に行うこと

- fixed commit の clean checkout を作り、`cargo fetch --locked` の直後に
  `cargo fetch --locked --offline` を pass させ、feature build も GPU lease 外で
  済ませる。2026-07-27 の lockfile は 29 registry archive、合計 2,590,666 B
  (2.471 MiB) だった。local fetch は 1 s 未満だったが remote transfer time は
  未確認である。cache/registry が無ければ runner は default offline で早く失敗する。
- Rust/ROCm/hipBLAS/CK、`rocminfo`、LLVM tools、zstd を provider image 上で
  CPU-only staging できるなら先に確認する。`cargo`/`rustc` と `cc` または
  `ULLM_RENTAL_LINKER` の値もここで決める。clang、mold、rustup を install する
  必要はない。
- model、GGUF、Docker image、external engine、full 13.2 GB artifact は P0 に
  不要である。P0 results を決めるために借用中に取得しない。
- source/lockfile/binary と fixture の hash を保存する。dirty 開発 worktree を
  そのまま rental host へ持ち込まない。

### 5.3 借用中に取得しない artifact

P0 runner の network contract は Cargo dependency だけであり、default は
`--offline` である。source checkout 自体は runner が clone しないため、lease 前に
用意する。local tracked source tree は 24,217,770 B だったが、remote clone transfer
size は未確認である。

初回 evidence に残る下記は P0 に不要で、download time も保存されていない。Qwen3-
14B-Q8_0.gguf は 15,698,533,728 B、Qwen3-Coder-Next-FP8 は 80.4 GB、Qwen3-
30B-A3B-FP8 は 32.5 GB、Qwen3.6-35B-A3B-FP8 は 37.5 GB である。container image と
Qwen3-14B-FP8 の exact byte size は保存 evidence からは**未確認**であり、推測で
埋めない。これらを借用開始後に取得すると 1--2 時間 P0 の時間を侵食する。

## 6. 優先順位、過去の時間、次回見積り

初回では preserved README は約 2 時間の lease を記録する一方、運用上は環境構築
を含め実質約 5 時間を消費した。stage 別 wall-clock は回収されていないため、
後から精密な内訳を作らない。残る直接 evidence は以下だけである。

| 初回の摩擦 | 保存済み evidence | 今回の対策 |
| --- | --- | --- |
| linker | `build.log` は `clang` 不在、`build2.log`/`build3.log` は mold 不在で失敗。`build4.log` の successful release build は 52.57 s。 | runner の Cargo environment override。`.cargo/config.toml` を触らない。 |
| Rust/Python 依存 | `rustup.log`、PEP 668 で失敗した `pip.log`。 | lease 前 staging。P0 runner は rustup/pip をしない。 |
| model/image/download | pull/download logs はあるが stage 時間はない。 | P0 から完全に外す。必要なら別承認・別 timebox。 |
| B failure | B skip により A′だけが通った。 | CPU oracle と B non-skip physical gate を最優先にする。 |

cache を温めた前提の P0 forecast は、local rehearsal と初回 remote build receipt
を反映して次へ更新する。local offline baseline は 2 min 37 s だが、physical と
remote vCPU/ROCm 差は含まないため、その数字だけで booking を短縮しない。

| P0 stage | 目安 | 根拠 |
| --- | ---: | --- |
| preflight | 1--3 min | device/toolchain/evidence admission。local は 0--1 s。 |
| CPU oracle/tests | 2--5 min | local 73--79 s、remote vCPU の余裕。 |
| generic HIPRTC audit | 1--3 min | local 18 s、27 programs、GPU launch なし。 |
| clean gfx942 release build | 1--5 min | local 53--54 s、保存済み remote warm build 52.57 s。 |
| ISA/resource audit | 1--3 min | local 6 s、fatbin extract/unbundle/disassemble。 |
| physical A′/B 5 case | 5--15 min | fragment + five-shape differential |
| evidence/retry reserve | 10--20 min | 同一条件の再現を 1 回まで |

pre-provision 済みなら stage budget は **30--60 分**、一回の同条件再現を含む
conservative booking は従来どおり **45--90 分**とする。従って **2 時間**は P0
判定に十分と判断する。**1 時間**は toolchain admission が即時に通り、physical が
初回または一回の再現で結論を出す場合だけ可能で、保証しない。source/Cargo cache/
toolchain を借用後に作る cold start は 1 時間では不可、2 時間でも保証しない。

時間切れでは P0 の preflight、CPU、HIPRTC、build、ISA、B を skip しない physical
A′/B を捨てない。profiler/occupancy query、full-model、artifact transfer、model/
container download、external engine、timing sweep、hand-written A の順に捨てる。P0 が
失敗したら source を借用中に書き換えず、同一条件の evidence を一回だけ取り lease
を止める。P0 が全 pass して初めて、occupancy query と限定的な additional evidence
を P1 として検討する。

## 7. レンタル時の exit gate

P0 pass は以下をすべて満たすこととする。

1. exact gfx942 が `rocminfo` で確認でき、toolchain/feature/ISA audit が pass。
2. B CPU oracle が pass し、physical B が skip されずに全 5 shape で CPU tolerance
   を満たす。
3. A′ が同じ 5 shape で CPU/A′-B differential を満たす。
4. physical log、source fingerprint、stage timing、ISA/resource TSV が results
   directory に残る。

P0 pass でも full-model、production dispatch、経路 A、実効 occupancy、HBM/L2、
partition performance は pass にならない。B または A′ が失敗したら source を
借用中に書き換えず、同一条件の evidence 再現を 1 回だけ取り、オフライン解析へ
戻る。

## 8. Phase 4 の判断

Phase 1--3 の offline completion を優先したため、手書き MFMA の経路 A には
**着手していない**。gfx1201 hand-written WMMA が component pass 後にも full-model
step 1 で落ちた前例があるため、実機なしの新規 A kernel はこの rental-decision
critical path に載せない。A′/B の実機 P0 と full-model integration gate が閉じた
後に、別 scope で skeleton/ISA/resource audit を検討する。
