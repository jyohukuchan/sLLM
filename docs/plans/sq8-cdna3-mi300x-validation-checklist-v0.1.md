# SQ8_0 CDNA3 MI300X A′ 実機検証チェックリスト v0.1

- Status: 2026-07-26 に MI300X×1 で Stage 1 を pass、Stage 2 の A′ 対 CPU は 5 形状を pass した。B control は failure のため skip しており、occupancy/residency と performance gate は未確認である。A′ bring-up 全体の完了ではない。第 9 節を正とする。
- Date: 2026-07-26
- Scope: 独立した `SQ8_0` の CDNA3/gfx942 A′ bring-up。A′ は installed CK gfx942 XDL instance を使う立ち上げ用の足場であり、raw OCP E4M3FN を直接 MFMA に渡さない。
- Out of scope: hand-written MFMA の案 A、本番 dispatch/serving、release、campaign、authorization、`/etc/ullm/served-models/active.json`、サービス操作。案 A の実機検証は、R9700 側 `SQ8_0` 手書き最適化の成果が固まり次第、別の計画として追加する。

この文書の目的は、課金中の MI300X をその場の実装・調査に使わず、短時間で判定価値の高い検証を順に終えることである。A′ が通っても案 A や本番経路を承認するものではない。

## 1. 既知の事実と実行境界

以下は予約前に確認済みのローカル事実であり、クラウド環境でも同じ事実だと仮定してはならない。

| 項目 | 確認済みのローカル事実 | クラウドで再確認するもの |
|---|---|---|
| OCP→FNUZ prepack | commit `71dcf25a`。全 256 byte 中 254 を受理し、OCP `0x7f`/`0xff` は拒否、OCP negative zero `0x80` は FNUZ `0x00`。受理値では `OCP(raw) = 2 * FNUZ(mapped)`。各変換 operand の scale は x2、両 operand の積は x4。 | 実装を再発明しない。使用する source/binary の commit、Cargo.lock、SHA-256 を照合する。 |
| canonical `SQ8_0` artifact | hash-checked scan は 280 tensor、weight payload 13,212,057,600 byte（約 13.2 GB）、BF16 scale 806,400 個。`0x80` は 207,515、`0x7f`/`0xff` は 0、BF16 scale x2/x4 の overflow/underflow/non-exact は 0。 | full-artifact test を予約に入れる場合だけ、コピー先の manifest/hash と容量を照合する。 |
| A′ | commit `8e8f6d02`。Default `16x128x128` main-K-loop に `v_mfma_f32_16x16x32_fp8_fp8` が 24 個。全 instance の静的 VGPR/SGPR/AGPR/LDS は記録済み。実効 occupancy は未確認。 | 実行時の HIP module/function、block size、partition ごとの occupancy/residency を実測する。 |
| B control | raw OCP → BF16 dequant → hipBLAS GEMM、F32 accumulation。CPU 照合 17 test は通過済み。 | 実 MI300X で CPU reference との 5 形状 differential を通す。 |
| physical smoke | `crates/ullm-engine/examples/sq8_gfx942_aprime_physical_smoke.rs`。one-wave fragment 診断の後、5 形状の A′/B/CPU comparison を行う。artifact path は持たない。 | exact gfx942、visible device 1 台、実行結果、raw evidence recorder を確認する。 |
| local ROCm | ROCm 7.2.1、HIP/hipcc `7.2.53211-e1a6bc5663`、AMD clang 22.0.0git。 | cloud image の HIP runtime/driver、hipcc、CK archive、hipBLAS、profiler を採取して比較する。 |

`rocm-ck-gfx942-aprime` は `GPU_ARCH=gfx942` を要求し、`rocm-ck-gfx1201` と相互排他である。実行時 selector は `gcnArchName` の厳密 `gfx942`、または既知の `:xnack+/-` と `:sramecc+/-` modifier のみを受理する。`gfx940`、`gfx9420`、未知 modifier、複数 visible device は pass ではない。

### 1.1 共通の中止規則

各段階で「中止」と書かれたら、次を行って次段階へ進まない。

1. stdout/stderr、raw output、device/ROCm manifest、実行した argv と allow-list 化した環境を保存する。
2. ハッシュ、時刻、partition、device identity を evidence manifest に追記する。
3. 同一 binary・同一入力・同一 partition での再実行は、環境起因の一過性 error を除外する **1 回だけ** 許す。入力、source、compiler option、partition、kernel layout は変えない。
4. 再現したらインスタンスを停止し、原因解析と修正はローカルへ持ち帰る。課金時間中に source を書き換えない。

「再試行」は新しい仮説を試すことではない。故障した入出力を採取するための一度の同一再現である。

## 2. レンタル前に完了させる準備

この節が未完のまま GPU instance を開始してはならない。特に build、artifact 転送、profiler の counter 名調査を課金時間へ持ち込まない。

### 2.1 source、binary、evidence recorder を固定する

- [ ] A′ の source commit、`Cargo.lock` hash、`git diff --no-ext-diff --exit-code` の結果、`GPU_ARCH=gfx942` で build した binary の SHA-256 を、予約用 manifest に記録する。共同作業中の dirty worktree をそのままクラウドへ複製しない。
- [ ] 新しい専用 target directory で、GPU を見せずに feature build と CPU-only test を完走する。既存 build/release tree は使わない。ローカルでは例えば `CARGO_TARGET_DIR` を新規一意ディレクトリにし、`HIP_VISIBLE_DEVICES=-1` で build する。
- [ ] `cargo test --offline -p ullm-engine --lib sq8_gfx942_aprime`、`cargo test --offline -p ullm-engine --test sq8_fnuz_prepack`、`cargo test --offline -p ullm-runtime-sys sq8_ck_gfx942_aprime_tests` の結果を保存する。クラウドで `--offline` を使うのは、registry/cache または vendor tree を同梱済みの場合だけである。そうでなければ lockfile を固定して `--locked` を使う。
- [ ] physical smoke の実行結果だけでは raw matrix、64 lane × 4 register dump、各形状の A′/B/CPU output をファイルに保存しないことを確認する。現在の smoke source はそれらを stdout に要約しているだけである。予約前に、以下の binary format を出力する instrumented recorder または等価な別 executable を source review・build・hash 固定する。これが未完なら借りない。
  - fragment input A/B: `.u8`、logical CPU expectation/matrix output/lane dump: little-endian `.f32le`。
  - 5 形状の input payload: `.u8`、scale: `.f32le`、A′/B/CPU output: `.f32le`、比較結果: JSON。
  - `f32le` は element count、shape、row/column order、endianness、SHA-256 を companion JSON に明記する。raw file を後から CSV へ丸めて置換しない。
- [ ] occupancy/residency probe と performance harness を、A′の physical smoke とは別に準備する。現在この文書で確認できる source は occupancy query も profiler capture も実装していない。実際に選択された HIP module/function に対する `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` 等の query を行えること、counter discovery と raw capture を保存できることを予約前に確認する。
- [ ] 予約用 bundle に、source archive、binary、recorder、occupancy probe、performance harness、入力 fixture、hash manifest、実行手順だけを含める。credential、cloud token、artifact の秘密情報を evidence や command line に含めない。

### 2.2 持ち込む資材と artifact 転送の判断

`SQ8_0` artifact 全量が必要な段階と、形状だけを使う段階を混同しない。

| 段階 | 必要な資材 | full 13.2 GB / 280 tensor artifact | 小さい fixture/subset で足りるか |
|---|---|---|---|
| 入口 preflight | source/binary bundle、ROCm/identity collector、hash manifest | 不要 | 不要。 |
| 第一段 fragment/lane | physical-smoke fixture、CPU expectation、raw recorder | **不要** | fixture のみで足りる。artifact path を持たない test である。 |
| 第二段 5 形状 differential | 同 smoke の 5 deterministic fixture、A′/B/CPU recorder | **不要** | fixture のみで足りる。GPU は実モデル M/N/K を通るが、重みは synthetic input である。 |
| 第三段 occupancy/residency | occupancy probe、選択 kernel identity、static-resource manifest | **不要** | 不要。 |
| 第四段 kernel timing / HBM / L2 | prebuilt timing harness、同一の synthetic full-shape input、profiler config/counter metadata | 原則不要 | microbenchmark だけなら足りる。ただし実 artifact の prepack/caching/load を測る根拠にはならない。 |
| optional artifact-prepack/cache/full-model test | canonical artifact、manifest、hash、artifact-aware harness | **必要** | subset は full 280 tensor の完全性、全 weight byte、全 cache residency を証明できない。 |

最初の A′ bring-up reservation は、optional artifact-prepack/cache/full-model test を明示的に追加しない限り、full artifact を持ち込まない。第一〜第三段では full artifact は判定価値を増やさず、転送と容量確認だけが課金リスクになる。

実 byte を使う補助試験が必要なとき、subset は「各 A′ instance family を少なくとも 1 tensor、`0x80` を含む payload、対応する BF16 scales、抽出元 manifest と SHA-256」を満たす明示的な抽出物に限る。現在、そのような転送用 subset が存在することは**未確認**である。作るなら予約前に抽出規則・対象 tensor list・hash を固定する。subset pass を full artifact pass と呼んではならない。

full artifact を使う場合は、GPU allocation 前に cloud の persistent volume または object storage へ転送し、転送先で manifest/hash を照合する。転送時間の見積もりには、予約前に同じ経路で 1 GiB 以上を送って得た持続 throughput を使う。payload だけの下限は `13,212,057,600 × 8 / sustained_bits_per_second` 秒であり、protocol、checksum、disk I/O、再送の余裕は別に取る。ネットワーク帯域は現時点で**未確認**なので、推測値で予約時間を決めない。

artifact-aware test では raw OCP payload、derived FNUZ payload、B の BF16 weight staging、output、raw profiler evidence が同居し得る。13.2 GB の空きだけで十分とは判断しない。実際に使う harness の allocation plan と `df`/quota を予約前に検証し、必要容量を manifest に記録する。

### 2.3 ROCm/driver 互換性 gate

ローカル baseline は ROCm 7.2.1 / hipcc `7.2.53211-e1a6bc5663` である。cloud が別 version なら「同じ gfx942 だから動く」とは言えない。予約前または GPU 課金前の image 起動時に、少なくとも次を raw text/JSON として採取する。

- [ ] OS/image ID、kernel、AMD GPU driver、firmware/VBIOS（取得できる範囲）。
- [ ] `hipcc --version`、`ROCM_PATH`、HIP runtime/loader と hipBLAS の version、`libdevice_gemm_operations.a` の path と SHA-256。
- [ ] `rocminfo` と provider/AMD SMI の raw output。GPU name、PCI BDF、visible device count、`gcnArchName`、VRAM、XCD/NPS/topology が読める箇所を保存する。
- [ ] feature build の compiler/linker command、dynamic library resolution、A′ selected code object の hash。`GPU_ARCH=gfx942` 以外で通っていないことも確認する。
- [ ] profiler executable/version、利用可能 counter の raw discovery output、profiling permission の有無。

| 差分 | 壊れ得るもの | 予約前の扱い |
|---|---|---|
| hipcc/CK header/archive が 7.2.1 と異なる | `ck::f8_t` macro、OCP-named opaque ABI instance、link、selected HSACO、MFMA codegen/resource metadata | remote image で offline build/link/ISA audit をやり直す。A′ binary の持込だけで pass にしない。 |
| HIP runtime/driver が異なる | module load、`gcnArchName` modifier、occupancy API、launch error、numerical/performance behavior | exact device selector と Stage 1 を最初に実行する。結果が違えばその reservation では先へ進まない。 |
| hipBLAS が異なる | B control の BF16 GEMM API/algorithm/数値・runtime link | B が CPU tolerance を満たすまで A′の結果を解釈しない。 |
| profiler/counter schema が異なる | HBM/L2 counter 名、単位、permission、XCD attribution | counter name を推測で置換しない。公式または tool の metadata を保存し、対応が取れなければ performance gate は未確認として止める。 |
| container の library path/CPU toolchain が異なる | binary が起動しない、local build と別の shared library が load される | exact image で build するか、`ldd`/loader manifest が一致する build artifact に限定する。 |

ROCm mismatch は失敗ではないが、local binary の互換性と static ISA evidence を無条件に引き継げないという意味である。remote re-audit を済ませるまでは未確認である。

### 2.4 local build と remote build の決定

| 方法 | 課金時間への影響 | 主な利点 | 主なリスク | 採用条件 |
|---|---|---|---|---|
| local で build して bundle を持込 | GPU 課金前に完了可能 | 最短で Stage 1 に入れる。ローカルの offline ISA evidence と同じ source を封印できる。 | cloud ROCm/driver/CK archive と ABI が違うと link/load/HSACO が一致しない。 | remote image が local baseline と一致し、library manifest と小さな dry-run が照合済み。 |
| cloud の CPU-only staging/image build | GPU 課金を避けられるなら推奨 | remote toolchain/CK/hipBLAS を使った binary を先に固定できる。 | Cargo cache、network、disk quota、image の寿命が未確認。 | provider が GPU 無しの staging、persistent volume、または image build を許す。 |
| GPU lease 中に remote build | 原則避ける | 最後の手段として device と同じ image で build できる。 | build/cache/download の待ちが全て課金時間となり、失敗時に Stage 1 へ到達できない。 | 前二者が不可能で、build 時間を別枠として予約済み。 |

build 所要時間は cloud の CPU、cache、network、disk に依存し、現時点で**未確認**である。予約前に同一 image で fresh `CARGO_TARGET_DIR` の build を 1 回計時し、その実測だけを予約計画へ使う。実測がないなら、GPU lease 中の build を計画に入れない。

### 2.5 feature、環境変数、依存関係

実機実行用の最小 contract は次である。値が provider image に依存するものは placeholder のまま予約前 checklist で確定する。

```bash
export GPU_ARCH=gfx942
export HIP_VISIBLE_DEVICES=<single-visible-device-token>
export ROCM_PATH=<remote-ROCm-root>       # /opt/rocm 以外なら必須
export CARGO_TARGET_DIR=<new-rental-only-target-directory>

cargo build --locked -p ullm-engine \
  --features rocm-ck-gfx942-aprime \
  --example sq8_gfx942_aprime_physical_smoke
```

- `HIP_VISIBLE_DEVICES` は空文字列でも comma 区切りでもない 1 token でなければならない。runtime は CPU device を index 0 に常設するため、physical smoke は全 runtime device から fail-closed gfx942 selector が受理する唯一の候補を選ぶ。visible GPU 1 台を runtime index 0 と仮定しない。
- `rocm-ck-gfx1201` を同時に指定しない。`GPU_ARCH=gfx1201`、`gfx940`、`gfx950` は A′ feature では失敗が期待値である。
- 依存する runtime component は HIP runtime (`amdhip64`)、hipcc/header、CK `libdevice_gemm_operations.a`、hipBLAS、`libdl`、Rust/Cargo と lockfile の依存物である。各 path/version/hash を evidence に残す。
- `LD_LIBRARY_PATH`、provider 固有 profiler 設定、clock/power 設定は現時点で**未確認**である。cloud の公式 image が要求する値だけを使い、値を推測して export しない。
- `CARGO_TARGET_DIR` は既存 release/build tree を指してはならない。cloud でも予約専用の新しい path を使う。

## 3. 課金中の実行順序

低コストで全体の可否を決めるものを前に置く。各段階の pass は次段階へ進む条件であり、A′ を production candidate に昇格させる条件ではない。

### 3.0 入口 preflight（検証段階の前、計画 5〜10 分）

最初の validation stage は次節の fragment/lane 診断である。この入口はその前に、誤った instance や toolchain に課金を使わないための admission だけを行う。

1. [ ] evidence directory を新規作成し、source/binary/fixture/hash manifest を copy して SHA-256 を再照合する。
2. [ ] `HIP_VISIBLE_DEVICES` の token 数、HIP device count、`gcnArchName`、GPU name、PCI BDF、ROCm/driver/CK/hipBLAS version を採取する。
3. [ ] exact selector が `gfx942` または許可済み modifier 付き `gfx942` を受理することを確認する。GPU 名だけ、compute major/minor だけ、cloud 商品名だけでは代用しない。
4. [ ] A′ feature が有効な binary、`GPU_ARCH=gfx942`、`rocm-ck-gfx1201` 非選択を照合する。remote build を選んだ場合は、ここまでに build/link/ISA audit を終える。
5. [ ] raw evidence recorder、occupancy probe、performance harness、counter metadata が全て hash manifest にあることを確認する。未準備のツールをここで書き始めない。

以下のいずれかなら Stage 1 を起動せず中止する: exact gfx942 でない、visible device が 1 台でない、feature/build/link が不一致、ROCm mismatch の re-audit 未完、raw recorder 未準備、容量不足、または source/binary hash 不一致。

### 3.1 第一段 — one-wave FNUZ `16x16x32` fragment/lane 診断（計画 2〜5 分、上限 10 分）

これは全体の可否を決める段階である。artifact は使わない。64 lane の 1 wave で FNUZ A `16x32` row-major と B `32x16` column-major を使い、logical C `16x16` と lane ごとの 4 FP32 accumulator slot を読む。

1. [ ] prebuilt physical-smoke/recorder を、入口 preflight で記録した唯一の device だけへ実行する。
2. [ ] logical matrix output と CPU expectation を各 element で比較する。現在の smoke contract の acceptance は `abs <= 0.002` **または** `rel <= 1e-5` である（両方を超えた element があれば失敗）。NaN/Inf は即失敗である。
3. [ ] 64 × 4 の raw accumulator dump を `.f32le` で保存し、fixture の一意な logical value から `(lane, register) -> (row, column)` を推論する。256 logical coordinate がちょうど 1 回ずつ現れる全単射でなければならない。
4. [ ] host-side expected matrix、actual logical matrix、raw lane dump、inferred map、shape/endianness/fixture hash、stdout/stderr を同じ run ID に保存する。

現行 physical smoke 自体を起動する最小 command は次である。これは raw recorder を置き換えない。実行前に binary を build 済みにし、recording executable の argv も同じ source/binary manifest に固定する。

```bash
(
  cd "$REPO"
  GPU_ARCH=gfx942 \
  HIP_VISIBLE_DEVICES="$ONE_VISIBLE_GPU" \
  "$SMOKE_BIN"
) >"$EVIDENCE/stage-1-fragment/smoke.stdout" \
  2>"$EVIDENCE/stage-1-fragment/smoke.stderr"
```

`REPO`、`ONE_VISIBLE_GPU`、`SMOKE_BIN`、`EVIDENCE` は予約前に absolute path/value を manifest へ固定する。未定義の placeholder のまま実行しない。provider が runtime loader 用の環境を要求する場合は、2.5 節で確認した値を allow-list に加え、その値も記録する。

| 結果 | 判定 | 切り分け | 次の行動 |
|---|---|---|---|
| logical matrix が tolerance 内、raw mapping が 256 coordinate の全単射 | **Pass** | FNUZ operand/matrix semantics と diagnostic fragment dump の双方がこの device/ROCm で整合する。 | 第二段へ進む。 |
| logical matrix 不一致、NaN/Inf、launch/HIP error | **Stop** | operand format、OCP→FNUZ compensation、A/B row/column semantics、rocWMMA/CK matrix semantics、または runtime/toolchain の問題。raw lane map を根拠に先へ進めない。 | 同一条件で 1 回だけ再現採取し、止める。 |
| logical matrix は一致するが raw mapping が非全単射 | **Stop** | logical operation は通っても、fragment/lane 仮定が誤っている。lane/register layout を production assumption にしてはいけない。 | 同一条件で 1 回だけ再現採取し、止める。第 4 節の線引きに従う。 |
| device identity/selector が不一致 | **Stop** | `gfx942` でないか、visible-device isolation が破れている。 | kernel を実行せず instance を止める。 |

この段階の成功出力は「logical matrix の pass」と「256 element の raw mapping 全単射」の両方である。片方だけの成功を A′ pass と記録しない。

### 3.2 第二段 — 実モデル寸法 5 形状の A′/B/CPU differential（計画 5〜15 分、上限 20 分）

第一段 pass 後にだけ実行する。各 case は sparse deterministic fixture だが GPU は実モデルの M/N/K を通る。first/final K128 block が nonzero で、残りに OCP negative zero を含むため、K-block stride、tail、`0x80 -> 0x00` 正規化を隠せない。

| case | M | N | K | 選択される A′ instance |
|---|---:|---:|---:|---|
| `k_or_v_tail_id1` | 1 | 1,024 | 5,120 | Default `16x128x128` |
| `q_or_o_full_id1` | 16 | 5,120 | 5,120 | Default `16x128x128` |
| `gate_or_up_tail_id2` | 1 | 17,408 | 5,120 | KPadding `16x128x256` |
| `gate_or_up_full_id3` | 128 | 17,408 | 5,120 | Default `16x256x128` |
| `down_tail_id4` | 1 | 5,120 | 17,408 | Default `16x128x256` |

各 case で次を必ず保存・判定する。

1. [ ] dispatch が表の instance ID と一致すること。shape outside dispatch table、buffer alias、異なる device/backend は failure である。
2. [ ] B（raw OCP → BF16 → hipBLAS F32 accumulation）対 CPU を各 element で比較する。acceptance は `abs <= 1e-5` **または** `rel <= 1e-5`。B は correctness control であり、B が失敗した状態で A′を採点しない。
3. [ ] A′（FNUZ prepacked、operand ごとの x2 scale）対 CPU、A′ 対 B を各 element で比較する。A′ の documented BF16-output allowance は `abs <= 0.125` **または** `rel <= 0.008`。NaN/Inf、length mismatch、tolerance 超過は failure。
4. [ ] fixture input、A′/B/CPU raw output、max abs/rel と failing index、選択 instance、run time、stdout/stderr を保存する。全 output の hash を比較し、同一入力の一度だけの再実行で nondeterminism も確認する。

中止条件は次である。

- B 対 CPU が失敗した: B control または remote hipBLAS/ROCm まで含めて未検証である。A′ failure と分類せず、ここで停止する。
- A′ 対 CPU または A′ 対 B が失敗した: A′ の OCP/FNUZ scale、layout、CK opaque ABI、shape/tail のいずれかが未検証である。性能測定へ進まない。
- 5 case のどれかで selection/allocator/launch error、非有限値、再現しない output が起きた: 1 回の同一再現採取後に停止する。

### 3.3 第三段 — 実効 occupancy / residency 実測（計画 10〜20 分、上限 30 分）

第二段の全 5 case が pass して初めて行う。静的 resource metadata は occupancy の代用ではない。

| A′ instance | VGPR | SGPR | AGPR | LDS | workgroup |
|---|---:|---:|---:|---:|---|
| Default `16x128x128` | 83 | 50 | 0 | 18,432 B | wave64 / 256 threads |
| KPadding `16x128x256` | 250 | 50 | 30 | 36,864 B | wave64 / 256 threads |
| Default `16x256x128` | 158 | 50 | 26 | 34,816 B | wave64 / 256 threads |
| Default `16x128x256` | 166 | 50 | 30 | 36,864 B | wave64 / 256 threads |

1. [ ] A′の各選択 instance について、実際に launch される HIP module/function を hash で同定する。wrapper や別 kernel への proxy query は使わない。
2. [ ] `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`、または同じ function を対象とする HIP occupancy API で、workgroup size 256 と実際の dynamic shared-memory argument を指定して query する。API 名、return status、input、active blocks/CU、取得可能な wave/CU 関連値を JSON に保存する。
3. [ ] device property、XCD/NPS topology、clock/power state、kernel module hash、static resource table と実測値を一つの record に結ぶ。static metadata から active blocks を逆算して「実測」と書かない。
4. [ ] profiler/trace が利用できるなら、同じ function の launch trace を保存し、実行された grid/block と instance selection が query 対象と一致することを確認する。

pass は「各選択 instance の実 function に対し query が成功し、module/hash/block/partition が evidence に結び付くこと」である。具体的な最低 active-block 数は現時点で**未確認**なので、都合のよい閾値を当日に追加しない。数値は測定値として持ち帰り、性能判断は第四段で同一 partition 上の A′/B 比較と併せて行う。

query が actual function を指せない、API が error を返す、trace と function identity が食い違う、または partition/topology を採取できない場合は、occupancy/residency は未確認である。その状態で性能の説明を作らず、課金を停止する。

### 3.4 第四段 — partition 別性能、HBM/L2 counter（1 partition あたり計画 30〜60 分、上限 90 分）

第三段 pass と、予約前に hash 固定した timing harness / counter mapping が前提である。physical smoke は correctness test であり、これを反復して性能 benchmark の代わりにしてはならない。

1. [ ] 利用可能な XCD/NPS configuration を provider/driver の raw topology として保存する。現在の configuration を変更できない場合は、その 1 configuration だけを測る。異なる configuration を比較する場合は、provider が事前に用意した別 instance/profile を使い、動的な partition 変更を試みない。
2. [ ] configuration ごとに、A′ と B を同じ shape、同じ data placement、同じ stream/launch policy、同じ warm-up/iteration policy で測る。測定回数、warm-up 回数、timing clock、sample order は予約前に固定し、実行時に変えない。
3. [ ] native profiler の counter discovery output から、HBM read/write、L2 request/hit/miss または等価な counter と単位を version ごとに確定する。counter 名を他の ROCm version から推測しない。raw profiler result は tool の native format のまま保存する。
4. [ ] normalized result には kernel/module hash、A′またはB、M/N/K、instance ID、XCD/NPS/topology、clock/power state、wall time、counter 値と単位、counter metadata source を含める。HBM effective bandwidth を出すなら `(HBM read bytes + HBM write bytes) / kernel elapsed seconds` とし、logical tensor size だけから実測帯域を名乗らない。
5. [ ] L2 hit rate を出すのは、採取した counter metadata が分子・分母と scope を定義している場合だけである。異なる XCD/NPS row を平均して 1 つの値にしない。

CDNA3 の理論 HBM 帯域上限は、SKU 名や XCD/NPS 数から暗算しない。configuration ごとに次の手順で決める。

1. exact SKU、HBM stack count、stack あたり bus width、data rate、または vendor が公表する device peak HBM bandwidth を、version/date 付きの一次資料とともに evidence manifest に記録する。
2. vendor/provider の topology が active HBM stack/channel の partition への割当を明示している場合だけ、`partition theoretical cap = full-device peak × active HBM data path / full-device HBM data path` を計算し、分子・分母を記録する。
3. XCD partition 数や NPS mode だけから peak bandwidth を等分しない。HBM mapping が資料で確認できない場合、partition cap は **未確認** とし、full-device peak を partition peak と取り違えない。
4. profiler の byte counter が複数 XCD/agent をどう帰属させるかも metadata で確認する。scope が不明なら HBM/L2 の比較は未確認である。

以下なら第四段を中止する: correct A′/B の同一条件比較ができない、counter permission がない、counter definition/units が不明、XCD/NPS topology が取れない、thermal/clock state が記録できない、または raw profiler output を回収できない。別の counter をその場で探索したり、実装を変えたりして延長しない。

## 4. fragment/lane 仮定が誤っていた場合の線引き

第一段の raw dump は production fragment-layout contract ではなく、物理 device で contract を発見する診断である。論理 matrix pass と raw mapping pass のどちらも必要である。

| 観測 | 課金中に許す範囲 | 持ち帰って offline で直す範囲 |
|---|---|---|
| logical matrix mismatch | hash/device/ROCm/fixture/endianness を確認し、同一 binary・同一 input を 1 回だけ再現する。 | FNUZ operand format、x2/x4 compensation、A/B row/column tag、rocWMMA fragment template、CK ABI semantics、K128 scale placement。 |
| logical matrix pass、raw lane/register mapping が非全単射 | raw `.f32le` と fixture を保存し、同一 run を 1 回だけ再現する。 | lane/register→coordinate map、fragment layout、store/load packing、lane shuffle、LDS layout。hand-written A の lane map に流用してはならない。 |
| raw dump の host-side decoder にだけ明白な 1:1 serialization bug がある | 予約前に hash 固定済みで、kernel input/output/fragment template を変えない decoder-only variant がある場合だけ、その variant を 1 回実行できる。 | recorder/decoder が未準備、または kernel code/fragment declaration に触れる修正。現在、当該 prebuilt variant の存在は**未確認**である。 |
| row-major/column-major を変えた kernel variant を試したい | 実行しない。 | CPU oracle、unit test、offline compile/ISA audit、新しい source/binary hash を作ってから次回 reservation で試す。 |

したがって現地で許される「layout 修正」は、事前固定済みの host-side evidence decoder の明白な serialization 修正までである。kernel の fragment type、operand layout、lane packing、LDS、scale application、CK wrapper、または手書き MFMA を書き換えることは現地デバッグであり、この runbook の範囲外である。

## 5. 時間とコストの見積もり

以下は cloud 実測値ではなく、予約枠を管理するための timebox である。金額・時間単価は記載しない。build、network、storage、partition 数、profiler permission は未確認なので、予約前の dry-run で得た実測があればそちらを優先する。

| 作業 | 課金 GPU を使うか | 計画時間 | 延長しない上限 | 備考 |
|---|---|---:|---:|---|
| source/binary/recorder/profiler を固定、artifact の事前転送 | 使わない | 予約前に完了 | GPU lease に持ち込まない | full artifact を使う optional test だけ事前転送する。 |
| 入口 preflight | 使う | 5〜10 分 | 10 分 | identity、toolchain、hash、visible device を確認。 |
| 第一段 fragment/lane | 使う | 2〜5 分 | 10 分 | 最も高い判定価値。再現は 1 回だけ。 |
| 第二段 5 形状 differential | 使う | 5〜15 分 | 20 分 | 5 case の raw output を保存。 |
| 第三段 occupancy/residency | 使う | 10〜20 分 | 30 分 | actual module/function query を必須とする。 |
| 第四段 timing + HBM/L2 | 使う | 30〜60 分 / configuration | 90 分 / configuration | 追加 XCD/NPS configuration ごとに同じ枠を追加する。 |
| evidence hash/upload/recovery | 使う | 5〜10 分 | 10 分 | instance 停止前に raw data を回収する。 |

- 最短の go/no-go scenario は、入口 preflight と第一段だけで **約 10〜20 分** である。第一段 failure はこれで終了し、第二段以降を実行しない。
- 1 configuration で第四段まで通す想定 scenario は **約 1〜2 時間** の予約枠である。これは全ツールが事前に build され、counter mapping が有効で、同一再現以外のデバッグをしない場合の timebox である。
- 追加 configuration は **30〜60 分ずつ** 加算する。GPU lease 中の build、artifact 転送、layout 修正はこの計画に含まれないため、発生した時点で中止して別予約に分ける。

## 6. 持ち帰る証跡

evidence はクラウドの一時 root disk だけに置かない。run ごとに新規 directory を作り、native raw output を不変で回収し、最後に SHA-256 manifest を作る。推奨する論理構成は次である。

```text
sq8_0-mi300x-aprime-<UTC-run-id>/
  manifest.json
  hashes.sha256
  source-and-build/
  environment/
  stage-1-fragment/
  stage-2-differential/<case>/
  stage-3-occupancy/
  stage-4-performance/<partition-id>/
```

| 区分 | 必ず保存するもの | 保存形式 |
|---|---|---|
| `manifest.json` | run ID、UTC timestamp、source commit/tree/Cargo.lock hash、binary/module hash、operator、stage status、stop reason、artifact を使ったか | UTF-8 JSON。schema version を明記。 |
| `environment/` | OS/image/kernel、ROCm/hipcc/driver/firmware、hipBLAS/CK archive hash、`rocminfo`、SMI raw output、GPU name/BDF/`gcnArchName`/visible count、XCD/NPS/topology、disk/network dry-run | tool 固有の raw text/JSON をそのまま保存し、要約 JSON は別 file。 |
| command provenance | argv の JSON array、working directory、allow-list 環境、exit code、start/end monotonic/UTC time | JSON。credential、access token、secret path の内容は保存しない。 |
| 第一段 | A/B `.u8`、CPU expected/actual matrix/lane dump `.f32le`、inferred lane map JSON、fixture schema、stdout/stderr | raw binary + companion JSON。64 × 4 layout と endianness を明記。 |
| 第二段 | 各 5 case の OCP/FNUZ input `.u8`、scale `.f32le`、A′/B/CPU output `.f32le`、tolerance/max error/failing index/selected instance JSON、stdout/stderr | raw binary + JSON。shape と row order を明記。 |
| 第三段 | occupancy API input/output/status、actual HIP module/function identity、grid/block/shared-memory、device property、trace | JSON + native trace/raw tool output。 |
| 第四段 | native profiler result、counter discovery/metadata、timing samples、normalized per-kernel table、partition/topology、clock/power record、theoretical-cap source/calculation | native format を保持し、正規化は CSV と JSON の両方。 |
| artifact optional test | source artifact manifest/hash、subset selection manifest、transfer verification、容量計画 | JSON/text。canonical payload を不要に複製しない。 |

raw `f32le` は IEEE-754 binary32 little-endian、`.u8` は byte stream とし、JSON の数値丸め版で代替しない。すべての raw file に shape、element count、layout、byte length、SHA-256 を結び付ける。failure evidence も削除せず、`manifest.json` の `status` と `stop_reason` に残す。

## 7. 借りる前に解消すべき未確認事項

以下は現在の cloud rental environment について確認できていない。すべてを予約前 checklist の質問・回答・証跡として埋める。回答が得られない項目を推測で埋めない。

| 未確認事項 | 予約前に確認する内容 | 未確認のままの場合 |
|---|---|---|
| MI300X exact SKU | provider の商品名ではなく device/SKU、HBM capacity/stack、PCI/device identity、firmware | theoretical HBM cap を決めず、租借しない。 |
| XCD/NPS/partition | current topology、変更可能性、configuration ごとの HBM/XCD 帰属、別 profile の可用性 | partition 別 performance claim をしない。利用可能な 1 configuration に限定するか中止。 |
| ROCm/driver/hipcc | HIP runtime、driver、hipcc、CK archive、hipBLAS、profiler version と local 7.2.1 との差 | local binary/ISA evidence を流用せず remote re-audit。完了できなければ中止。 |
| storage | persistent volume/object storage、free space/quota、I/O throughput、instance stop 後の保持 | full artifact/large evidence を持ち込まない。raw evidence を安全に回収できなければ中止。 |
| network | artifact/upload の持続 throughput、egress policy、checksum と retry の可否 | full artifact transfer を GPU lease に入れない。 |
| GPU visibility/isolation | exact `gcnArchName`、visible device count、`HIP_VISIBLE_DEVICES` behavior、他 tenancy の干渉 | one-device contract を満たさなければ Stage 1 を起動しない。 |
| profiler access | counter permission、counter metadata、native output format、XCD attribution | 第四段を予約しない。 |
| clock/power/thermal policy | fixed/dynamic clock、power cap、thermal telemetry、provider throttling policy | cross-run performance comparison をしない。 |
| CPU/RAM/build cache | source build 時間、Cargo cache/vendor、disk I/O、container lifecycle | build を GPU lease に持ち込まない。 |
| artifact security | checkpoint を置ける storage、access control、消去/retention policy | optional artifact test を外す。 |

## 8. 最終判定の記録

run を終えたら、次のいずれかを `manifest.json` の明示的な status として残す。

1. `fragment_rejected`: 第一段の logical matrix または raw mapping が失敗。A′/B/性能の結論は出さない。
2. `differential_rejected`: 第一段は pass したが、B または A′ の 5 形状 differential が失敗。性能測定をしていないことを明記する。
3. `occupancy_unconfirmed`: correctness は通ったが actual function の residency/occupancy evidence を取れなかった。静的 register/LDS 表で補完しない。
4. `performance_incomplete`: correctness/occupancy は通ったが counter/topology/thermal evidence が欠ける。timing を architecture-wide conclusion にしない。
5. `aprime_bringup_complete`: 第一〜第四段の準備済み gate を通過し、partition ごとの raw evidence を回収済み。これは A′ の bring-up record であり、案 A、本番 `SQ8_0` dispatch、release、activation の承認ではない。

このチェックリストの終点は「次のオフライン判断に必要な生データを持ち帰ること」である。課金中に新しい実装・campaign・authorization・activation を実行しない。

## 9. 2026-07-26 MI300X 実施結果

生データは
[mi300x-rental-v1](../../benchmarks/results/2026-07-26/mi300x-rental-v1/README.md)
に保全した。対象は AMD Instinct MI300X VF、gfx942:sramecc+:xnack-、
ROCm 7.2.4、NPS1/SPX、VRAM 196,288 MB である。

### 9.1 Stage ごとの結果

| stage | 結果 | 根拠と残る境界 |
| --- | --- | --- |
| 3.0 preflight | 部分確認 | exact gfx942 modifier、GPU 名、ROCm、NPS1/SPX、VRAM は記録された。firmware、PCI BDF、visible CU、CK/hipBLAS hash、process isolation、raw recorder manifest は未確認。 |
| 3.1 fragment/lane | pass | logical max_abs=0.007812、max_rel=0.000000、256 lane/register coordinate の全単射を確認。 |
| 3.2 A′ 5 shape | A′ 対 CPU は pass | k_or_v_tail_id1、q_or_o_full_id1、gate_or_up_tail_id2、gate_or_up_full_id3、down_tail_id4 の A′ 対 CPU はすべて max_abs=0.000000。 |
| 3.2 B control | failure、未修正 | k_or_v_tail_id1 で期待 0.53125、観測 0.03125、差 0.5。成功 log は ULLM_SMOKE_SKIP_B_CONTROL による B skip を使った。 |
| 3.3 occupancy/residency | 未確認 | HIP occupancy query、active wave/block、clock、resource residency を回収していない。 |
| 3.4 partition / HBM / L2 | 未完 | A′ projection の 200 repeat timing はあるが、counter、empirical HBM peak、thermal、他 partition、A/B 比較はない。 |

fragment/lane と A′ 対 CPU の pass は、この MI300X VF / NPS1-SPX の
deterministic fixture に限定する。full-model logits、prefill/decode、
artifact prepack/cache、B との正常な differential、production dispatch は
検証していない。

### 9.2 device guard の修正

実機で smoke の旧 device_count()==1 guard が構造的に通らないことが分かった。
uLLM runtime は CPU device を index 0 に常設するので、GPU が 1 枚だけでも
runtime count は 2 になる。

本体の physical smoke は、保存済みの rental patch と一致する修正を取り込む。
HIP_VISIBLE_DEVICES の 1 token を確認した後、全 runtime device を列挙し、
fail-closed gfx942 selector が受理する device がちょうど 1 台であることを
要求し、その index を使う。複数候補・候補なしは fail closed のままである。

### 9.3 B control の扱い

B の tail 処理取りこぼしが疑われるが、根因は**未確認**である。B は未修正で
あり、ULLM_SMOKE_SKIP_B_CONTROL は A′ 対 CPU を観測するためだけの escape
hatch である。B=0 や A′-B=0 と表示された skip run の値は self-comparison
であり、B pass ではない。

このため、この checklist 全体の status は aprime_bringup_complete ではなく
differential_rejected である。ただし A′ の fragment/lane と A′ 対 CPU の
物理 sub-gate は pass した。B を直して skip なしで 5 形状を再実行し、
occupancy/residency と partition-specific performance を回収するまで、次の
Phase や production integration へ進めない。

### 9.4 時間見積もりとの対比

計画の最短 go/no-go は preflight 5--10 分と fragment/lane 2--5 分を合わせて
約 10--20 分だった。借用全体は約 2 時間だが、stage 開始・終了時刻は
回収ログからは**未確認**である。したがって、最短 go/no-go の見積もりを
実測で満たしたとは記録しない。
