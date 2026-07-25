# Independent SQ8_0 CDNA3 CK gfx942 XDL reuse evaluation

Date: 2026-07-26

## 前回の要点

- 独立 SQ8_0 の高速 projection は、手書き WMMA ではなく
  runtime/src/sq8_ck_gfx1201.hip.cpp の CK
  DeviceGemmMultipleD_ABScale / DeviceGemmXdlUniversal instance だった。
- canonical SQ8_0 は OCP E4M3FN raw payload と [128,128] scale である一方、
  gfx942 の FP8 MFMA operand は FNUZ である。raw OCP payload の直接投入は不可で、
  derived FNUZ prepack、0x80 -> 0x00、変換 operand ごとの scale 二倍が必要である。
- このホストには gfx942 実機がない。オフライン compile/ISA はできるが、数値、
  actual occupancy、residency、timing は確認できない。

## 今回の変更点

- ROCm 7.2.1 CK header と /opt/rocm/lib/libdevice_gemm_operations.a を実査した。
  現行 gfx1201 と同じ F8/F8/BF16、float A/B scale、RowMajor/ColumnMajor/RowMajor、
  [1,128,128] の DeviceGemmMultipleD_ABScale registry が存在する。現在の
  default/K-padding の 16x256x128、16x128x128、16x128x256 XDL instance は、OCP
  ABI で link できる。
- ck::f8_t は target architecture で自動選択されない。
  amd_ck_fp8.hpp は CK_USE_OCP_FP8=1 なら f8_ocp_t、それ以外なら f8_fnuz_t に
  alias する。gfx942 hardware が FNUZ operand を要求することとは別である。
  archive symbol は ABScale f8_ocp_t のみで、f8_fnuz_t は 0 件だった。
- 隔離ディレクトリ /tmp/ullm-sq8-cdna3-ck-gfx942.kwLrwP で
  hipcc --offload-arch=gfx942 の exact registry probe を実施した。FNUZ alias の
  compile-only は通るが、link は f8_fnuz_t specialization の
  add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_1_128_128_mem_v1_default_instances
  が未定義で失敗した。OCP alias は同じ archive への link に成功した。
- OCP ABI で link した fat binary から gfx942 HSACO を抽出した。正確な
  16x128x128 main-loop symbol は v_mfma_f32_16x16x32_fp8_fp8 を 24 個含み、
  FP8 conversion instruction は 0 個だった。これは native MFMA の実証であると
  同時に、CK が OCP-to-FNUZ conversion を挿入しない実証でもある。

| 16x128x128 code-object variant | VGPR | SGPR | AGPR | LDS | spill | wave / max workgroup |
|---|---:|---:|---:|---:|---:|---|
| main K-loop tail | 83 | 50 | 0 | 18,432 B | 0 / 0 | wave64 / 256 |
| no-main-loop tail | 65 | 38 | 0 | 18,432 B | 0 / 0 | wave64 / 256 |
| no-main-loop no-tail | 39 | 34 | 0 | 18,432 B | 0 / 0 | wave64 / 256 |

- この archive metadata は compiler Occupancy: annotation を出さないため、
  occupancy は **未確認** と記録した。静的 resource から実機 occupancy を推測しては
  いない。hipModuleOccupancyMaxActiveBlocksPerMultiprocessor を gfx942 実機で
  実行して初めて確認できる。
- 結論は条件付きで A′（CK gfx942 XDL instance reuse）を隔離 prototype の先行候補に
  する。新規 gfx942 body は OCP macro で prebuilt C++ ABI に合わせるが、渡す byte
  buffer は derived FNUZ のみとし、weight/activation scale をそれぞれ二倍する。
  これは f8_ocp_t の semantic reuse ではなく opaque ABI reuse である。A の手書き
  native MFMA は fallback/tuning route、B の dequant-to-FP16/BF16 は正当性 control
  として維持する。
- 本番 build/release tree、/opt/ullm、service、artifact、既存 gfx1201 source/body
  には変更を加えていない。final activation は行っていない。

## 次の行動

1. Phase 1 で exact gfx942 arch selector、OCP-to-FNUZ byte/scale oracle、artifact
   scan、scale-range gate を実装し、OCP-ABI CK link/type/ISA を version-locked test
   として固定する。
2. Phase 2 で既存 gfx1201 source を編集せず、FNUZ-only derived buffer を受ける
   separate A′ prototype を作る。選ぶ全 CK instance と M/N/K/tail を gfx942 HSACO
   まで監査する。
3. gfx942 実機でのみ A′/A/B の kernel/end-to-end differential、actual occupancy、
   residency、partition 固有の timing を比較する。raw OCP byte を CK MFMA に渡す
   route はいかなる結果でも追加しない。
