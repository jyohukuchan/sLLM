# GPU アーキテクチャ機能リファレンス（ROCm 7.2.1〜7.2.4 検証済み）

## 目的とスコープ

uLLMのHIPカーネル実装（Codex/Claude含む）が、CDNA(MI300系)向けの前提やwave64を無意識に持ち込んで、RDNA(コンシューマ/ワークステーション)向けの制約を見落とすことを防ぐためのリファレンス。2026-07-15時点、ROCm 7.2.1を基準に作成し、同日付で最新版7.2.4(2026-05-29公開)までの変更を追跡調査して反映した。**7.2.1→7.2.4の間、RDNA4/gfx1201固有の致命的バグ修正・大幅な性能改善は、公式のバージョン付きリリースには確認できなかった**(詳細は8節)。

**このドキュメントの使い方**: HIPカーネル(特にAQ4_0/SQ8_0のGEMV/GEMM/attention)を新規実装・変更する前に、対象GPUの節と「実装前チェックリスト」を確認すること。AGENTS.mdからリンクし、実装エージェントが自動的に参照できるようにする。

**信頼度の凡例**:
- `[公式]` — AMD公式ドキュメント(GPUOpen、rocm.docs.amd.com)、AMD公式GitHubリポジトリのissue/README
- `[公式Issue]` — AMD公式リポジトリ(ROCm/ROCm、ROCm/aiter等)のissueで報告された内容。issueは今後変わりうるため状態を要再確認
- `[コミュニティ・要検証]` — 個人ブログ・非公式リポジトリ由来。数値・主張は未検証、参考情報として扱う

---

## 対象GPU一覧

| GPU | アーキテクチャ | gfx ID | 用途 | CU数 | VRAM | メモリ帯域(理論値) | Matrix命令 |
|---|---|---|---|---|---|---|---|
| Radeon AI PRO R9700 | RDNA4 | **gfx1201** | メインターゲット | 64 CU(4096 SP) | 32GB GDDR6 | 640〜644 GB/s(256bit) | WMMA(wave32) |
| Radeon Pro V620 | RDNA2 | **gfx1030** | サブターゲット | — | — | — | **なし**(WMMA非搭載) |

`[公式]` R9700仕様: https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html

---

## 1. Wave実行モデル: wave32 vs wave64（最重要・最頻出の見落とし源）

- **RDNA系(RDNA1〜4)はwave32がデフォルト・主要な実行幅**。一方、**AMD公式ROCmドキュメント/サンプルの大半はCDNA(MI200/MI300、wave64)を前提に書かれている**。CK(Composable Kernel)やAITERのデフォルト実装もCDNA/wave64向けに重度に最適化されているという指摘が複数ソースで一致している `[コミュニティ・要検証]`(一次ソースでの明示確認はできていないため要検証)。
- **decodeとprefillで最適な実行幅が異なる**という原則がある `[コミュニティ・要検証、AMD公式のFLOPS数値とは整合]`:
  - **prefill(compute-bound、大きな行列)**: WMMA(wave32)が真価を発揮。RDNA4はRDNA3比でFP16スループットが2倍(後述)
  - **decode(memory-bound、n=1のGEMV)**: WMMAの大タイルを効率的に埋められず、wave64のDOT命令+ビットフィールド展開の方が実用的という報告がある
- **見落としがちな罠**: SGLangをgfx1151(RDNA3.5、同じくwave32)へ移植した際、`moe_topk`カーネルの`WARP_SIZE`不整合(wave64前提のハードコード)がMoE推論の破綻要因になった実例がある(`JeremiahM37/strix-halo-sglang`)`[コミュニティ・要検証]`。**HIPコードで`WARP_SIZE`や`__AMDGCN_WAVEFRONT_SIZE__`をハードコードせず、コンパイル時定数として扱うこと**。

**実装前チェック**: 新規HIPカーネルを書く/移植する際、`WARP_SIZE`/wave幅を32か64かハードコードしていないか確認する。CK/AITER由来のコードを参考にする場合は特に要注意(wave64前提の可能性が高い)。

---

## 2. 行列演算命令(WMMA)

### 2-1. 基本仕様 `[公式]`

出典: https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ , https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/ , https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/ , https://github.com/ROCm/amd_matrix_instruction_calculator

- **RDNA3/RDNA4のWMMAは`__builtin_amdgcn_wmma_<C,D format>_16x16x16_<A,B format>_w32_gfx12`という組み込み関数群**。命名規則通り**全て16×16×16タイル固定・wave32実行必須**。より大きな行列は複数回呼び出して分解する。
- **RDNA2(gfx1030、V620)にはWMMA/Matrix命令が存在しない**。標準ALU/DOT命令のみで演算する必要がある。
- ピークFLOPS/clock/CU比較:

| dtype | RDNA2 | RDNA3 | RDNA4(R9700) |
|---|---|---|---|
| FP16 | — (WMMA非搭載) | 512 | **1024** |
| BF16 | — | — | **1024** |
| INT8 | — | 512 | **2048** |

### 2-2. Wide-K WMMA(狭いdtypeでの帯域改善技法) `[公式]`

出典: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/

- グローバル/共有メモリのload/storeは最大128bit(16byte)幅だが、素のWMMA命令はFP16では128bit幅を飽和させる一方、**FP8では64bit幅、INT4では32bit幅しか使わずメモリ帯域を無駄にする**。
- 解決策: **Kを2倍に拡張した2つのWMMA命令を融合し、メインループで128bitベクトルロードを行う**(Wide-K WMMA)。行列積の結合則により通常WMMAとビット完全一致の結果を保証しつつ、狭いdtypeでのメモリスループットを実質倍増できる。llama.cppでも採用実績あり。
- **AQ4_0(4bit/INT4相当)は現状帯域の1/4、SQ8_0(FP8)は1/2しか使えていない可能性がある** — uLLM独自kernelでの改善余地が大きい箇所。

### 2-3. レーンマッピングの罠 `[公式Issue]`

出典: https://github.com/ROCm/ROCm/issues/6025

- gfx12(RDNA4) wave32でのWMMA出力は「lane i (0..31)がcolumn i%16、row (i/16)*8〜(i/16)*8+7を保持」という非直感的なマッピング。「lane%16=row」と誤認すると16×16出力タイルが**サイレントに転置される**。AMD公式ドキュメントの記載不足がissueとして提起されている。
- rocWMMA fragment API経由なら直接この罠には嵌りにくいが、生のWMMA組み込み関数を直接叩く自前kernelでは`amd_matrix_instruction_calculator`ツールで検証すること。

### 2-4. 融合GEMMのレイアウト不整合 `[公式]`

出典: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/

- 「第1GEMMの出力DをそのままA行列として次のGEMMに使う」際(FlashAttention的な連鎖)、D行列がM-major、次段の入力AがN-major(転置)を要求するというレイアウト不整合が起きやすい。**A・B行列を入れ替えて呼び出すことで転置を回避する**手法が推奨されている。QK^T→softmax→×Vの連鎖で同様の問題が起きていないか要確認。

### 2-5. rocWMMAライブラリの移行 `[公式]`

出典: https://github.com/ROCm/rocWMMA

- `ROCm/rocWMMA`リポジトリは`[DEPRECATED] Moved to ROCm/rocm-libraries repo`と明記。新機能を追う場合は`ROCm/rocm-libraries`モノレポを参照すること。

**実装前チェック**: 狭いdtype(FP8/INT4)のkernelはWide-K WMMA適用を検討したか。生WMMA使用箇所はレーンマッピングを`amd_matrix_instruction_calculator`で検証したか。融合GEMM連鎖でレイアウト転置コストが発生していないか。

---

## 3. FP8サポート

- **FP8 E4M3変換のbuiltin(`__builtin_amdgcn_cvt_f32_fp8`等)はgfx1200/gfx1201固有で提供される** `[公式、7/08 journalで実機確認済み]`。全条件で性能改善が確認されている。
- `__builtin_amdgcn_cvt_scalef32_f32_fp8`はROCm 7.2時点でgfx1200ではfeature不足により使用不可(project内実装記録より、不採用リスト参照)。
- V620(RDNA2)はBF16を含め低精度演算のハードウェアサポートが弱く、FP8は不採用としている(project既存方針)。

---

## 4. メモリ階層・リソース予算

`[コミュニティ・要検証、AMD公式仕様と概ね整合]` 出典: https://zolotukhin.ai/zinc/docs/amd-gpu-reference/

- **LDS(Local Data Share)**: RDNA3/RDNA4は**64KB/CU**(Work Group Processorあたり128KB)、32バンク×4byte/バンク構成。
- **VGPR**: CU内は2 SIMD構成、各SIMD **192KB VGPR**、16〜32レジスタ単位で動的割当、**最大16 wavefront同時実行**。
- VGPR圧迫はコンパイラのwave occupancy低下に直結する。本家llama.cppのRDNA4チューニング事例(`rm_kq`定数を2→1に変更しVGPR使用量削減)でdense decode +13%の実測あり(下記6節参照)。**uLLM独自kernelでもVGPR使用量とoccupancyのトレードオフを`rocprof`で確認し、意図的にVGPR上限を絞る/緩める実験をする価値がある**。

## 5. メモリ帯域の実効値

`[コミュニティ・要検証]` — 理論帯域の**67〜93%程度**が典型的な実効利用率とされる。R9700の理論640〜644GB/sに対し、実効値は環境・kernel設計に大きく依存する。本家llama.cpp RDNA4チューニング実測ではdense modelがメモリ帯域の79〜81%まで到達、MoEはディスパッチオーバーヘッドで56〜61%止まりという報告がある(6節参照)。

---

## 6. 既知のドライバ/ランタイム/ライブラリの落とし穴(gfx1201固有)

**2026-07-15追記**: 以下7件をROCm 7.2.2/7.2.3/7.2.4の公式リリースノート・GitHub issueで再調査した。**公式のバージョン付きリリースで「修正済み」と裏付けられたものはゼロ**だった。詳細は8節参照。

| 問題 | 内容 | 状態(7.2.4時点) | 出典 | 信頼度 |
|---|---|---|---|---|
| AITER arch table未登録 | `aiter/ops/triton/utils/arch_info.py`の`_ARCH_TO_DEVICE`にgfx1201が未登録。FP8 WMMA処理が**警告なくFP32へフォールバック**。想定35-40 tok/sに対し実測18-22 tok/sまで劣化。ROCm/RDNA4調査・SGLangフォーク調査・vLLMフォーク調査の3方向で独立に確認済み | **未修正**。TransformerEngineは「R9700サポート計画なし」と明言(2026-07-01)。aiter側はunified attentionのgfx12対応PRがマージされ部分改善あるが、AMD開発者自身が「RDNA4とCDNAのギャップは大きく大半のカーネルを作り直す必要がある」と言及、根本ギャップは未解消 | `ROCm/TransformerEngine#520`, `#359`, `ROCm/aiter#3294`(Open), `#900`(Open) | `[公式Issue]` |
| hipBLASLt FP32性能不整合 | gfx120X向けFP32カーネルの性能アラインメント問題(未チューニングカーネルで最大10倍の性能差、~2TFLOPS→~19TFLOPS) | **公式7.2.4パッケージへの反映は未確認**。issue自体は2026-07-07にクローズされたが(7.2.4リリースの約1ヶ月後)、検証はTheRock nightlyビルド(`torch+rocm7.13.0a20260501`)経由のみ。hipBLASLt公式CHANGELOGに1.2.1/1.2.2の変更履歴エントリなし | `ROCm/ROCm#5674` | `[公式Issue]` |
| ISA検出バグ(誤検出だった) | HIPランタイムがgfx1201を「2つのISAを持つが単一ISAしかサポートできない」として拒否するケースがある、と報告 | **ROCm自体のバグではなかった**。原因はUbuntuパッケージ版の古い`libamdhip64-dev`/`libhsa-runtime64-1`(5.7.1)がAMD公式リポジトリ版と競合していたこと。該当パッケージ削除で解消、報告者本人がクローズ | `ROCm/ROCm#6110`(Closed, 2026-04-13) | `[公式Issue]` |
| eGPU誤認識(誤検出だった) | ROCm 7.1.0がeGPU構成で8060SをGFX1201と誤認識、と報告 | **ROCm自体のバグではなかった**。原因は`.bashrc`に設定していた`HSA_OVERRIDE_GFX_VERSION=12.0.1`環境変数(ユーザー設定)。削除で解消 | `ROCm/ROCm#5696`(Closed, 2025-12-04、7.2.1より前) | `[公式Issue]` |
| 電力/クロック高止まり | HIP初期化後、R9700がプロセス終了までクロック・電力が高止まりしアイドルに戻らない(Vulkanでは非発生)。ROCm 7.1.1/HIP 7.1.52802で確認 | **未修正**。AMD側はMESファームウェアのバグと特定しパッチ済みと主張(2026-03-18クローズ)だが、**クローズ後もROCm 7.2.1環境での再現報告が2026-04まで継続**(ユーザー報告)。ROCmユーザースペースのバージョンとは別のファームウェア/カーネルドライバのリリースサイクルに依存するため、7.2.2〜7.2.4のリリースノートには言及なし | `ROCm/ROCm#5706` | `[公式Issue]` |
| コンテナ起動失敗 | vLLMがRDNA4(gfx1201)コンテナ内で起動失敗する報告 | **未修正・Open継続**。2026-04-22を最後に進展なし | `vllm-project/vllm#40081`(Open) | `[公式Issue]` |
| CK/AITERのwave64偏重 | CK・AITER含む多くのROCmライブラリのデフォルトバックエンドがCDNA(wave64)向けに最適化されており、RDNA3/4(wave32)では最適でない。回避策としてTritonバックエンドへの切替が推奨される場合がある | **段階的改善中、根本ギャップは未解消**。wave32対応PR(`composable_kernel#2594`, `#2722`, `#2723`, `#3421`)が複数マージ済みだが、CKライブラリのバージョン自体は7.2.1〜7.2.4を通じて1.2.0のまま変化なし | 複数ソース(1節参照) + `ROCm/composable_kernel`のPR群 | `[コミュニティ・要検証]`+`[公式Issue]` |

**AITERギャップは「要監視」項目**([uLLM-project.mdの検討中](../../uLLM-project.md)参照)。公式の是正時期は不透明(TransformerEngineは明示的にwon't-fix)。uLLMはAITERに依存していないため直撃はしないが、**同種のarch-keyed dispatchテーブルを自前実装する際はgfx1201を明示的にマッピングしないと同じ罠にはまる**。

**上記2件(ISA検出バグ、eGPU誤認識)はROCm自体のバグではなく環境要因だったため、実装上の回避策は不要**。ただし「gfx1201関連のissueは環境設定・パッケージ競合が原因のことも多い」という教訓として記録している。

---

## 7. AMD公式ライブラリのRDNA4対応成熟度

| ライブラリ | RDNA4(gfx1201)対応 | 備考 |
|---|---|---|
| Composable Kernel(CK) | 動作するが最適化はCDNA(wave64)寄りとの指摘 | `[コミュニティ・要検証]`、uLLMはCK ABScaleを採用済み。CKのGEMMインスタンスがWave32インスタンスを正しく選択しているか要検証 |
| AITER | **arch table未登録、FP8がFP32へサイレントフォールバック** | `[公式Issue]`、上記6節参照 |
| hipBLASLt | 128×128 block scale非対応(project内で実機検証済み・不採用)、FP32性能アラインメント問題あり | `[公式Issue]` + project実測 |
| rocWMMA | fragment API自体はRDNA4で動作するが、ライブラリはdeprecated(rocm-librariesへ統合) | `[公式]` |
| vLLM公式ROCmバックエンド | AITER最適化パス(`ROCM_AITER_FA`)はMI300X/MI325X/MI355X専用。RDNA/Radeonは`TRITON_ATTN`/`ROCM_ATTN`へフォールバック | `[公式]` https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html |
| SGLang公式 | 公式サポート対象はMI300X/MI250/MI35X(gfx942/gfx950)のみ。gfx1201への言及なし | `[公式]` https://docs.sglang.io/docs/hardware-platforms/amd_gpu |

**結論**: RDNA4/gfx1201向けの最適化は、AMD公式スタック全体を通じてまだ未成熟で、公式ドキュメント・サンプルの大半はCDNA前提。**「公式ライブラリを頼れば速くなる」という前提を疑い、自前kernel実装で攻める現在のuLLMの方針は、この観点からも妥当**。

---

## 8. ROCm 7.2.1→7.2.4 バージョン別変更点(2026-07-15調査)

`[公式]` 出典: https://rocm.docs.amd.com/en/docs-7.2.2/about/release-notes.html , https://rocm.docs.amd.com/en/docs-7.2.3/about/release-notes.html , https://rocm.docs.amd.com/en/latest/about/release-notes.html , https://github.com/ROCm/ROCm/releases

| バージョン | 公開日 | 主な内容 |
|---|---|---|
| 7.2.1(基準) | 2026-03-25 | — |
| 7.2.2 | 2026-04-14 | ROCTracerのカーネル操作イベント欠落の修正。HIPのストリームキャプチャ検証・イベント戻り値の是正。hipBLASLt 1.2.2でMXFP8/MXFP4 GEMM性能改善。**rocSHMEM 3.2.0でgfx1201のメモリコヒーレンシ問題を修正**(7.2.1〜7.2.4を通じて唯一の明示的なgfx1201向け修正)。`AMD_DIRECT_DISPATCH`環境変数を非推奨化 |
| 7.2.3 | 2026-05-04 | vLLMワークロードのROCprofiler-SDKトレースで発生していた大きな不規則アイドルギャップを削減。MIGraphX 2.15.0で埋め込み処理性能改善 |
| 7.2.4 | 2026-05-29 | AI推論ワークロード向け品質リリース。`hipGraphLaunch`のマルチリストグラフディスパッチを最適化しレイテンシ低減(**uLLMのtoken-by-token処理のようなkernel launch回数が多いパターンに間接的に効く可能性**)。ROCprofiler-SDKによるvLLMプロファイリング安定性向上 |

**結論**: 7.2.2〜7.2.4いずれの公式リリースノートも、サポートGPU表はMI355X/MI350X/MI325X/MI300X等**Instinct(データセンター)GPUのみ**で構成され、RDNA4/R9700への言及はrocSHMEM 3.2.0の1件を除き皆無だった。6節で挙げた7つの既知問題のうち、公式のバージョン付きリリースで「修正済み」と裏付けられたものはゼロ。実際の修正の多く(hipBLASLt FP32チューニング等)は**TheRock nightlyビルド経由で先行提供されており、安定版パッケージにはまだ降りてきていない**。7.2.4時点でも「RDNA4/gfx1201向けの致命的バグ修正・大幅な性能改善」と言えるものは見当たらない。gfx1201固有の改善を急ぎたい場合は、公式7.2.xパッケージではなくTheRock nightlyビルドを検討する余地がある(ただし安定性はトレードオフ)。

その他の非推奨化・breaking change(7.2.1→7.2.4通期): ROCTracer/ROCProfiler/`rocprof`/`rocprofv2`の非推奨化(`rocprofv3`への移行推奨、**uLLMは既にrocprofv3採用済みのため対応不要**)、ROCm SMIの将来的なメンテナンスモード移行(amd-smiへの移行推奨、**uLLMは既にamd-smi採用済みのため対応不要**)。

---

## 9. RDNA4向け実測チューニング値(参考、Vulkanバックエンドでの計測だが原理はHIPにも転用可)

出典: 本家llama.cpp GitHub Discussion #21043「RDNA4 Llama Experiments」`[コミュニティ・要検証、ただし本家公式リポジトリ内discussionで複数実測がクロス確認されている]`

| チューニング | 効果 | 適用コスト |
|---|---|---|
| PCIe ASPMを`performance`ポリシーに固定 | dense decode +10.8% | コード変更不要、`echo performance > /sys/module/pcie_aspm/parameters/policy`。運用ドキュメントに追記すべき |
| prefillのmicro-batch/batchサイズ拡大(`-ub 2048 -b 16384`相当) | MoEモデルのprefill +29% | batched prefill実装時のデフォルト値設計に活用 |
| VGPR圧力を絞る(`rm_kq`定数を2→1に相当する調整) | dense decode最大+13%(AMDVLKドライバ) | 自前kernelのVGPR使用量チューニング |
| `HIP_FORCE_DEV_KERNARG=1`環境変数 | カーネル起動オーバーヘッド削減 | uLLMのtoken-by-token処理(kernel launch回数が多い)に直接効く可能性が高い。未設定なら試す価値あり |

---

## 10. ROCm 7.2.1 既知バグ一覧(2026-07-15調査)

uLLMはROCm 7.2.1を標準バージョンとして固定運用する方針(今後CDNA対応する場合も含む)。ここでは7.2.1自体の既知バグを、公式Known Issues・GitHub issue双方から網羅した。

### 10-1. 公式リリースノートのKnown Issues `[公式]`

出典: https://rocm.docs.amd.com/en/docs-7.2.1/about/release-notes.html

公式が明記する既知問題は3件のみで、**いずれもCDNA3/CDNA4(MI300X/MI325X/MI350系)に集中**しており、RDNA関連の記載は本体リリースノートには存在しない(RDNA向けは別ページ、10-2参照)。

| # | 問題 | 影響範囲 | 回避策 |
|---|---|---|---|
| 1 | hipBLASLtで特定GEMM構成の性能低下(MI300X/MI325Xで16384×16384×6656等、MI350系で4096×4096×1×8192/16384等) | MI300X/MI325X/MI350系(gfx942/gfx950) | developブランチで修正済み、将来リリースへ反映予定 |
| 2 | MI300XのCPX/NPS4パーティションモード(38CU構成)でhipBLASLtのGEMM実行時間が延長 | MI300X(gfx942、パーティション化構成) | `ROCm/ROCm#6066`(Open)で追跡中 |
| 3 | ROCTracerが一部/全てのカーネル操作イベント受信に失敗する可能性 | 全アプリ(アーキ非依存) | ROCprofiler-SDKへ移行推奨(**uLLMは既にrocprofv3採用済みのため対応不要**) |

### 10-2. RDNA(R9700/gfx1201含む)の既知バグ `[公式Issue]`

6節で既出の4件(AITER arch table未登録、hipBLASLt FP32性能問題、R9700電力/クロック高止まり、vLLMコンテナ起動失敗)に加え、以下を新たに確認した。**特に上位2件は、将来マルチGPU構成(TP)を検討する際の重大な障害になりうる**。

| 問題 | 影響GPU | 状態 | 出典 |
|---|---|---|---|
| **RCCLデッドロック**: vLLM TP=2でデュアルR9700が最初のマルチGPU操作でデッドロック、両GPU使用率100%で無応答。RCCL 2.27.3では正常だった回帰 | R9700×2(gfx1201) | Open(2026-04-27) | `ROCm/rocm-systems#5480`, `vllm-project/vllm#40980`(重複報告) |
| **NCCLエラー**: vLLM TP=2でROCm 7.2.0は動作するが7.2.1で失敗(Qwen3-0.6B、Qwen3.5-9Bで再現) | R9700×2(gfx1201) | Closed(回避策: 7.2.0へダウングレード) | `ROCm/ROCm#6148`(2026-04-14) |
| RCCLデュアルGPU集約演算失敗(単体では正常、collective参加時に"HIP failure") | RX 7900 XTX×2(gfx1100/RDNA3) | Open(2026-03-28) | `ROCm/ROCm#6074` |
| **rocBLASLtがgfx1201用ではなく誤って"gfx1200.dat"のTensileファイルを検索し失敗**。モデル読み込みが2分後にSIGKILLされる(Ollama v0.7.22.1で100%再現) | R9700(gfx1201) | Open(2026-05-08) | `ROCm/rocm-libraries#7192` |
| **ファン制御不能**: SMUファームウェアv50にドライバがv46までしか対応せずファン制御レジスタへアクセス不可。負荷時に**109°Cでサーマルスロットリング**発生(ASUS Turbo版) | R9700(gfx1201) | Closed | `ROCm/ROCm#6101`(2026-03-31) |
| PyTorch常駐プロセスがアイドル時もGPU使用率100%・高電力を維持(既出#5706=llama.cpp固有問題とは別事象。`GPU_MAX_HW_QUEUES=1`は無効) | R9700(gfx1201) | Closed | `ROCm/ROCm#6298`(2026-05-25、ROCm 7.2.0.70200/PyTorch 2.10.0+rocm7.0) |
| hipMallocAsync/hipMallocFromPoolAsyncがOOM/segfaultで失敗(同期版hipMallocは正常、6.4.xからの回帰) | RX 7800 XT/7900 XTX(RDNA3) | Closed(7.2.2で言及、7.2.1への直接言及なし) | `ROCm/ROCm#6178` |

**Radeon/Ryzen公式限定事項ページ記載の追加項目**(`[公式]`、出典: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html):
- ONNX Runtime + MIGraphX実行プロバイダーの組み合わせでアプリ実行失敗の可能性(R9700)
- Janus Pro推論でtransformers==5.1.0使用時に障害。`transformers==4.57.5`固定が推奨(R9700)
- RHEL環境のR9700でUnsloth QLoRA実行時にGPUメモリアクセス障害
- RX 7000シリーズでAO Triton + PyTorch 2.9の組み合わせがデフォルト無効化

> **重要**: RCCL/NCCLの2件(デッドロック、TP=2回帰)と、rocBLASLtのTensileファイル誤検索、ファン制御不能によるサーマルスロットリングは、**いずれもR9700という現行ターゲットGPU自体に対する7.2.1固有の実害**。特にファン制御問題は連続稼働のhomelab運用にとってハードウェア寿命に関わるリスクであり、対象GPU(ASUS Turbo版かどうか等)・ファームウェアバージョンを確認すべき。マルチGPU(TP)は現状R9700×2構成で7.2.1のまま安定動作する保証がない(NCCL側は7.2.0へのダウングレードが唯一確認された回避策、RCCL側は回避策未確認のままOpen)。

### 10-3. CDNA3/CDNA4(gfx942/gfx950)の既知バグ `[公式Issue]`(将来対応時の参考)

10-1の公式Known Issues 3件に加え、GitHub issueベースで以下を確認した。**MXFP4/FP8量子化の数値的正しさに関わる問題が複数報告されている**点は、uLLM自身が量子化フォーマット(AQ4_0/SQ8_0)を持つプロジェクトとして特に注視すべき。

| 問題 | 影響GPU | 状態 | 出典 |
|---|---|---|---|
| Kimi-K2.5-MXFP4がMI350Xで意味不明な出力を生成(多言語/記号混在、非一貫)。7.1→7.2の回帰、またはMXFP4逆量子化kernelの疑い | MI350X(gfx950) | Closed(2026-03-07) | `vllm-project/vllm#36337`(ROCm 7.2.53150/vLLM v0.17.0) |
| GLM-5 MXFP4 sparse MLAデコードでクラッシュ。原因はTP時のアテンションヘッド数がAITERカーネルの最小要件(16)を下回ること | MI355X(gfx950)、8GPU構成 | Open(2026-04-03) | `vllm-project/vllm#38924` |
| **GLM-5.2-FP8のaiter `gemm_a8w8_blockscale_bpreshuffle`カーネルが数値的に不正確**。GSM8K精度が0.925→0.000に崩壊、72%が解析不能な応答 | MI350X/MI355X(gfx950) | Open(2026-06-18) | `sgl-project/sglang#28685` |
| fmha_v3のMI300向けカーネルバイナリが、大規模プリフィル(≥20,480トークン)でMI325Xをハングさせる(KIQ fence timeout、全8GPU接続喪失・冷再起動要)。原因はMI325XとMI300が同一CU数(304)・同一gfx942名を報告し、AITERディスパッチャがMI300用の壊れたkernelを誤選択すること | MI325X(gfx942) | Open(2026-05-12) | `ROCm/aiter#3139` |
| Ray分散実行でPipeline Parallelism(PP)>2使用時、raylet C++プロセスがクラッシュ | MI350X(gfx950)、多ノード | Open(2026-03-30) | `ray-project/ray#62190` |

**所見**: `aiter#3139`(MI325X/MI300のgfx942同一視によるカーネル誤選択)は、6節で述べた「arch-keyed dispatchでgfx1201を明示マッピングしないと同じ罠にはまる」という教訓の**CDNA版の実例**。将来CDNA対応する場合、gfx942の中でもMI300とMI325Xを区別するarch検出ロジックが必要になる可能性がある。

### 10-4. インストール・アップグレード関連 `[公式]`

- ROCm 7.2.1はUbuntu 24.04.4(kernel 6.8 GA/6.17 HWE)をサポート追加、Ubuntu 24.04.3はEoS
- Radeon/Ryzen版7.2.1はUbuntu 22.04.4、RHEL 10.1(Radeon系限定)に対応
- 7.2.1固有の新規インストール障害は確認できず。一般的なAPT依存関係の注意点(バージョンピン、クリーンインストール推奨)のみ

---

## 実装前チェックリスト(要約)

新規HIPカーネルを書く/既存カーネルを変更する前に確認すること:

1. **wave幅を32か64かハードコードしていないか**。CK/AITER由来のコード片は特にwave64前提の可能性が高い
2. **狭いdtype(FP8/4bit)のGEMM/GEMVはWide-K WMMA(K方向2命令融合)を適用できないか**。素のWMMAは帯域を1/2〜1/4しか使わない
3. **生WMMA組み込み関数を直接叩く場合、出力のレーンマッピングを`amd_matrix_instruction_calculator`で検証したか**(gfx12は「lane%16=row」と誤認しやすい転置バグの温床)
4. **decode(M=1、memory-bound)とprefill(M>>1、compute-bound)で同じkernel戦略を使い回していないか**。decodeはwave64 DOT寄り、prefillはWMMA寄りが定石
5. **AITER/CK/hipBLASLt等の公式ライブラリにgfx1201が正しくマッピングされているか**。サイレントなFP32フォールバックは性能劣化はするが動作はしてしまうため気づきにくい
6. **VGPR使用量とoccupancyのトレードオフを`rocprof`で確認したか**
7. **PCIe ASPM設定、`HIP_FORCE_DEV_KERNARG=1`等の運用レベルの改善を見落としていないか**(コード変更不要の改善)
8. **マルチGPU(TP)を検討する場合、R9700×2でのRCCLデッドロック(10-2節)を踏まえているか**。7.2.1では未解決、NCCL側は7.2.0ダウングレードのみ確認済みの回避策
9. **R9700のファン制御・温度を運用時に監視しているか**(10-2節、モデル・ファームウェアによってはサーマルスロットリングのリスク)

---

## 更新履歴

- 2026-07-15: 初版作成。ROCm 7.2.1基準。今回のuLLM-project.md技術スタック調査(AMD ROCm/RDNA4最適化技術、vLLM/SGLangフォーク調査)の結果を整理して作成
- 2026-07-15: ROCm 7.2.1→7.2.4(最新版)の変更を追跡調査し反映。6節の既知問題7件のステータスを更新(2件は環境要因の誤検出と判明、5件は未修正)。8節「ROCm 7.2.1→7.2.4 バージョン別変更点」を新設
- 2026-07-15: ROCm 7.2.1を標準バージョンとして固定運用する方針決定に伴い、10節「ROCm 7.2.1 既知バグ一覧」を新設(公式Known Issues、RDNA/R9700固有バグ、CDNA3/CDNA4固有バグ、インストール関連)。R9700デュアル構成のRCCLデッドロック、ファン制御によるサーマルスロットリングの2件を実装前チェックリストに追加
