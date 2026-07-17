# SQ8 P4 one-projection optimized component

日付: 2026-07-10

## 前回の要点

P0からP2でsource-correctなcanonical artifact、CPU oracle、R9700 native HIP referenceを確立した。P3では必要な128x128 block scaleをhipBLASLtが受理しないことを確認し、gfx1201上でnative FP8 WMMAを使うComposable Kernel ABScaleを選択した。

## 今回の変更点

- CPUの動的activation量子化oracleと、M=`1,2,4,8,16,32,128`の再現可能な実weight fixtureを追加した。
- GPUのrow×K128動的OCP E4M3量子化とCK ABScale GEMMを結合し、量子化単独、GEMM単独、量子化込みをHIP eventで分けて測定した。
- 実行可能なCK候補をすべてCPU oracleで検証してから最速候補を選ぶようにした。全Mで38候補中6候補が対応し、6候補すべてが数値gateに合格した。
- R9700の単独visibility、gfx1201、メモリ上限、空きVRAM、入力byte数、activation FP8 byte、scale bit、非有限値、fallbackをfail closedで検証した。
- 26 MiBのweightが64 MiB L3へ残る影響を分離するため、各GEMM前に別の256 MiB GPU bufferを読み切る`--cache-mode evicted`を追加した。bufferはGPUでindex hash初期化し、67,108,864 word全体のGPU checksumが独立CPU checksumと一致することを必須にした。退避kernelは同一stream上でstart eventより前に置き、性能値には含めていない。

正式結果は`uLLM-project/benchmarks/results/2026-07-10/sq8-optimized-component-v0.1/`へ保存した。主要値は次のとおり。

- M=8 warm: 量子化0.010800 ms、GEMM 0.028300 ms、量子化込み0.033460 ms、12.535 TFLOP/s。
- M=8 evicted: 量子化込み0.063001 ms、6.658 TFLOP/s。
- M=8 reference: 1.154280 ms。warm optimizedは34.50倍。evicted optimizedとwarm referenceの保守的比較でも18.32倍。
- M=2からM=8の集約スループット比: warm 3.878倍、evicted 3.900倍。推奨値2.5倍を超えた。
- 全Mの最大relative L2は0.001683、最小cosineは0.99999858。固定gateの0.005以下、0.9999以上を満たした。
- activation FP8とscaleは全MでCPU fixtureと完全一致し、NaN/Infとfallbackは0だった。

rocprofv3 traceでは、最終測定部分がcache eviction、activation quantization、CK OCP FP8 GEMMの順で実行された。CK code objectにはP3で確認済みの`v_wmma_f32_16x16x16_fp8_fp8`が含まれる。ROCm profiler SDK 1.1はgfx1201でderived occupancy、`FETCH_SIZE`、raw L2 readを実dispatchでも0と返すため、この0は実測値に使っていない。代わりに非ゼロのwave数、dispatch属性、checksum検証済みの256 MiB読出し、warm/evicted差、逆アセンブルを証拠として残した。

関連commit:

- `c741918`: fixtureとreference benchmarkのfail-closed化。
- `97b1350`: CK optimized component benchmark。
- `0de350c`: cache-evicted measurementと検証。
- `287a372`: 正式M grid、profiler、environment、gate判定の保存。

P4判定はgreenとする。これはproduction runtimeへの統合完了ではなく、P5のentry gateを満たしたという判定である。

## 次の行動

P5としてk/v、gate/up、downの三つの未検証shapeへ同じcomponent gateを展開する。shapeとMごとの実測dispatch表を固定し、QKVまたはgate/upで共有できる入力量子化を一度だけ行う。その後、decoder layer一層を統合し、中間tensorと最終layer出力を独立oracleと比較する。dominant projectionのいずれかが未検証またはflatであれば、40層へのoptimized path展開は停止する。
