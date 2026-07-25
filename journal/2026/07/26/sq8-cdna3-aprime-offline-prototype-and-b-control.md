# `SQ8_0` CDNA3 A′オフライン試作とB対照経路

## 前回の要点

- gfx942ではCKの`f8_fnuz_t`再生成はリンクできず、既存
  `f8_ocp_t` ABIのXDL code objectだけが再利用可能だった。
- canonical `SQ8_0`のOCP-to-FNUZ oracleは、`0x80 -> 0x00`、`0x7f`/
  `0xff`拒否、片側scale x2・両側積x4、実artifactの範囲gate通過まで
  CPUのみで確認済みだった。
- CDNA3の本番目標は手書きMFMAであり、A′はMI300X初回借用時の
  format/fragment検証用の足場として残す方針である。

## 今回の変更点

- gfx1201本体を変更せず、`rocm-ck-gfx942-aprime` feature配下にA′、
  direct-OCP-to-BF16 hipBLAS B対照、exact gfx942 selector、内部専用C ABI、
  CPU参照、物理スモークを追加した。
- A′はFNUZ prepack済みopaque byteだけを受ける。`f8_ocp_t`は既存CK
  archiveへ結線する型名だけであり、raw OCPをCKへ渡す入口は作っていない。
  activation F32 scaleとartifact BF16 weight scaleはいずれも既存oracleを
  通して片側x2にし、CKの積でx4になる。
- BはFNUZ/CKを経由せず、OCPをBF16へdequantしてhipBLAS F32 GEMMを行う。
  CPUではA′のFNUZ参照、BのBF16/FP16参照、artifact BF16 scale prepack、
  FNUZ fragment fixtureを検証した。
- 隔離`--offload-arch=gfx942`ビルドが成功した。抽出したgfx942 code objectの
  Default 16x128x128 main-K-loopには
  `v_mfma_f32_16x16x32_fp8_fp8`が24個あり、static metadataは
  VGPR 83 / SGPR 50 / AGPR 0 / LDS 18,432 B / spill 0 / wave64 / 256 threads
  だった。実機occupancyは未確認である。
- selectorは`gfx942`と既知HIP modifierだけを受理し、wrong `GPU_ARCH`と
  gfx1201 feature同時有効化をビルド時に拒否する。既存gfx1201 public headerと
  dispatch/bodyにgfx942 routingがないこともCPU testで固定した。
- GPU実行、service、release、`/opt/ullm`、activationは一切行っていない。

## 次の行動

- 明示承認後、単一のMI300X/gfx942可視化環境で
  `sq8_gfx942_aprime_physical_smoke`を一度だけ実行する。
- 先に16x16x32 FNUZ fragmentのlogical matrixとlane/register dumpを判定し、
  次に5つの実M/N/K形状でA′、B、CPU期待値を比較する。
- 同じ実機でactive-block/wave occupancyを取得する。fragment、数値、occupancyの
  いずれかが不成立ならA′を本番経路に進めず、手書きMFMA AとBを維持する。
