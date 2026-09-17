# MXFP8 GPU差の調査記録

> 2026-09-15、調査完了。Stage A〜Dの数値結果を記録した。
> codecの原因仮説を棄却したが、残るGPU間の演算経路差の原因は未特定。
> 本番kernel・既定の変更、commit／pushは対象外。

## 対象と証拠

Qwen3.8-27Bのteacher-forced MTP、同じ8条件×256強制位置を比較する。
各GPUのMXFP8／MXFP6はそのGPU自身のBF16 draftを基準とする。
modelと強制prefix identityを照合し、48 logitsファイル（12,205,424,640 bytes）の
SHA256とサイズをすべて検査してから配列を読んだ。NumPyで再計算したtop-1は
6系列×2,048位置すべてで既存reportと一致した。

- 再実行: `python3 ci/tools/analyze_mxfp8_gpu_divergence.py`
- 集約: [mxfp8-gpu-divergence-v1.json](../../../../../ci/matrix/mxfp8-gpu-divergence-v1.json)
- 個別行・入力hash・診断artifact: `.local-artifacts/mxfp8-gpu-divergence/`
- 解析コードのfocused確認: `python3 ci/tools/analyze_mxfp8_gpu_divergence.py --self-test`

## Stage A: 閾値化されたtop-1と連続誤差

以下は2,048行の平均。相対L2は各行の `||量子化logits−BF16 logits||₂ / ||BF16 logits||₂`。

| 指標 | gfx1030 MXFP8 | gfx1201 MXFP8 | gfx1030 MXFP6 | gfx1201 MXFP6 |
| --- | ---: | ---: | ---: | ---: |
| 相対L2 | 0.074509 | 0.073947 | 0.115723 | 0.115050 |
| 最大絶対差 | 1.00991 | 0.99261 | 1.54618 | 1.52952 |
| BF16 top-1位置での絶対差 | 0.34427 | 0.35571 | 0.51678 | 0.52664 |
| BF16 top-1からのflip数 | 55 | 36 | 62 | 67 |

MXFP8のgfx1201側の平均相対L2は約0.76%低いが、BF16 top-1位置の誤差は逆に大きい。
全語彙での平均距離とtop-1保持率を同一視しない。異なるtarget hiddenを持つので、
この差だけでnative命令やcompanion固有の精度差へ帰属しない。

BF16 marginの平均はgfx1030 `5.30014`／gfx1201 `5.23074`、中央値は`4.375`／`4.28125`。
gfx1201が系統的に大きいという説明は支持されなかった。

両GPUに共通するflip位置はMXFP8で12、gfx1030のみ43、gfx1201のみ24。
MXFP6では共通17、gfx1030のみ45、gfx1201のみ50だった。Jaccard係数はいずれも約0.152で、
同じ少数の「難しい位置」だけで説明できない。ただし位置の非共有自体は原因を特定しない。

BF16 top-1が両GPUで一致する1,950位置ではMXFP8のflipは27／24、不一致の98位置では28／12だった。
MXFP8の19件の差のうち16件がBF16基準自体の不一致位置にある。
top-1が一致する場合もtarget hiddenの同一性を意味しない。

MXFP8対MXFP6の保持率差について、GPU間の差の差は固定8 promptの平均で+1.171875ポイント。
prompt単位の探索的sign-flip検定は両側p=0.0625であり、位置を独立標本とみなす推論や、
一方だけ有意であることから相互作用を断定する記述を避ける。

Stage Aだけでは原因を特定できなかったため、Stage B〜Dへ進んだ。

## Stage B: selectorと復号検査の帰属

実際のheaderを使ったhost probeで、6形状のMXFP8は両targetともID99、
`matmul.mxfp8.w8a8.m1.col2.v1`を選択した。
MXFP6はo投影 `K=6144,N=5120` のみgfx1030 ID20／gfx1201 ID100で、残りは両target ID100だった。
これはhost selectorの実行証拠であり、保存済みモデルrunの全dispatch traceではない。

旧codec検査はgfx1201 native FP8復号を実行していたが、比較相手も同じbuiltinだった。
全256 codeの一致を、独立したソフト復号oracleとの等価性へ読み替えない。
またcol2の`#else`へ切り替えるだけでは`decode_scaled`の例外側にnative復号が残り、
scale共有も同時に変わる。native寄与の診断では復号だけを変更する必要がある。

## Stage C2: 独立したE4M3FN復号oracle

元のcodec検査とは独立したhost数式を用い、全256 codeをGPUから復号した。
gfx1030 `GPU-76a08c022586fed6`とgfx1201 `GPU-a8e9ddefa2d60f55`の両方で不一致0。
有限値254 code（signed zero 2、subnormal 14を含む）はFP32 bit一致、NaN 2 codeはclass一致だった。
各targetはHIPで1 block×256 threadを実行し、fallbackなし、cleanup failure 0。
gfx1201のdevice assemblyで`v_cvt_f32_fp8_e32`を確認した。

この全code範囲ではnative／software scalar復号値の相違を棄却できる。
NaN payload／sign、activationのencode、モデル全体の加算順やcompilerによる変化までを証明する検査ではない。
モデル対照の結果は次節に示す。

R9700 user serviceは測定時に停止し、終了後にunit／run.sh／binary hash不変、
healthz／readyzとも200、performance level `auto`復帰を確認した。
本番sourceの着手時336ファイルのhashは検査後もすべて一致した。

## Stage C1／C3: 同一GPUでのモデル対照

再buildした基準binaryは既存gfx1201測定とSHA256
`2fdcb03a7e0640120cf67dab6982bc087647b74126658f11920f7d4fee5b7ffb`が一致し、
既存のBF16／MXFP8 reportを再利用した。

| 診断変更 | target hidden一致 | logits hash一致 | top-1一致 | 同一GPUのBF16からのflip |
| --- | ---: | ---: | ---: | ---: |
| col2のvalue復号だけをソフト化 | 8/8 | 8/8 | 2,048/2,048 | 36/2,048 |
| MXFP8 activation encodeだけをソフト化 | 8/8 | 8/8 | 2,048/2,048 | 36/2,048 |
| MXFP8 E4 KV append encodeだけをソフト化 | 8/8 | 8/8 | 2,048/2,048 | 36/2,048 |

3対照の最大絶対差と相対L2はすべて0。col2のlane割当・scale共有は維持し、
共通decoder全体を変えてtarget側へ影響を広げる操作は避けた。
activationとKVのscale決定も維持した。対象device symbolのISAでは、
col2のnative decode命令が3→0、activation encodeが1→0、KV encodeが2→0となった。
この測定条件では、これらのnative／software codec差を原因仮説から除外できる。

## Stage D: attentionとtarget hidden

Phase84.5 r5と同じnative／Rustのwave選択無効化を適用し、BF16とMXFP8を測定した。
長いM1で使う共通staged32は維持され、waveが選択されるprefill側などが変更対象になる。

- 通常gfx1201からtarget hiddenが変化したのは3/8条件
  （`code-python-review-zh`、`code-rust-bugfix-zh`、`trans-zh-en-tech`）。
- gfx1030とのtarget hidden一致は依然**0/8**。Q1をこのattention対照だけでは解消できなかった。
- 診断内のBF16／MXFP8のtarget hiddenは**8/8一致**。
- 同じ診断BF16を基準としたMXFP8のflipは**38/2,048**（通常経路では36/2,048）。
- 経路変更前後のtop-1はBF16で26位置、MXFP8で27位置が変化した。
  最大logit絶対差はそれぞれ11.71875／11.125、平均行相対L2は0.033584／0.039464。

旧BF16を基準にすると診断MXFP8は58 flipになるが、target hiddenが異なる比較を含む。
品質の記録には同一診断内の38 flipを使う。GPU間の入力が一致していないため、
この結果からcompanion単独のハードウェア優劣は認定しない。

## 実行・artifact identity・検証

5 model run（10,240強制位置）はすべてexact gfx1201
`GPU-a8e9ddefa2d60f55`、HIP-only、nonzero dispatch、fallbackなし、cleanup 0で完走した。
診断変更は各build後に復元し、before／after hash一致を記録した。
終了時も着手時の本番source 336ファイルのhashが一致した。
R9700 serviceはunit／run.sh／binary hash不変、healthz／readyz 200、performance level `auto`復帰を確認した。

ISA抽出時に`llvm-objcopy`が生成ELFを再直列化したため、activation encode対照の実行ELF hashは
build直後と異なる。実行中の`/proc/PID/exe`で実際のhashを記録し、実行artifactを保存した。
entry pointと全ALLOC sectionのaddress／flags／bytesは元buildと一致することを照合した。
他の実行対照はbuild時hashと一致する。抽出コマンドは別出力ファイルを指定する形へ修正した。
この対応は生成artifactに限り、本番sourceや常駐service binaryの変更を含まない。
詳細は集約の`executed_binary_identity`／`elf_inspection_identity_note`を参照する。

比較は`ci/tools/compare_mxfp8_diagnostic_reports.py`を使い、各reportのmodel／prefix／shape、
logitsのSHA256・サイズを検査してからNumPy比較を行った。等しいhashの場合もtop-1整合を検査し、
異なるpayloadでは行ごとのargmaxもreportと照合した。hash／prefix不一致、argmax不整合、
hidden欠落・不一致を含むfocused fixture、Python syntax、Markdown local links、差分の空白検査を確認した。

## 訂正した前提

- 対象は64ファイルではなく48ファイル。
- margin≥1でも少数のflipがあり、gfx1201 MXFP6ではmargin≥4に1件あった。
- Phase84.5のr5はgeneric対照、r7はM1／prefillを維持してM2〜M4をwaveに揃える別対照。
- gfx1030でのwave-block対照はgfx1201のnative FP8命令を検証していない。

## 結論と未解明範囲

当初の受入基準に従い、原因を特定できなかった部分も数値付きで記録して調査を完了する。
codecのnative変換は、この測定で観測した差を説明しない。attention対照でもtarget入力差は残り、
残るtarget／companionの演算経路への帰属は未確定である。
near-tieのtop-1保持率だけを量子化品質全体と同一視しない。

本番kernel・既定・常駐serviceの設定は維持した。新しい外部コードの取込み、commit／pushは行っていない。

## 追記: WMMA帰属の確認（2026-09-15、ユーザー指示）

「WMMA有無による差かどうかだけ確認し、結果に関わらずこれ以上追わない」という指示で追加測定した。

既定のgfx1201 BF16系列を`rocprofv3 --kernel-trace`で1条件（`code-rust-bugfix-zh`、
8,284 prompt token・256強制行）採取した。314,644 dispatch・43 kernelのうち、
**WMMA kernelが840回dispatchされていた**。

| kernel | 回数 | 形状の役割 |
| --- | ---: | --- |
| `sllm_nvfp4_gfx1201_wmma128x64_pad68_k5120n17408_v1` | 448 | 27B target MLP gate/up |
| `sllm_nvfp4_gfx1201_wmma128x64_pad68_k17408n5120_v1` | 224 | 同 down |
| `sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_v1` | 168 | Kahan lookahead |

`matmul_kernel_internal.hpp:3285-3313` のとおり、同じMLP prefill行列積に対し
gfx1201は`Nvfp4W4A4PrefillGfx1201WmmaKahan`（WMMA 128x64＋Kahan補償）、
gfx1030は`sllm_nvfp4_w4a4_prefill_compensated128x64_v1`／dp4a index32を選ぶ。
gfx1030はRDNA2で行列コアを持たないため、**同一の行列積が必然的に別の累積方式**になる。
これらはhidden stateを直接生成する演算である。

分離のため両targetで`SLLM_NVFP4_W4A4_FORCE_BASELINE=1`を指定して`target_hidden`を比較しようとしたが、
**両targetとも同一エラーで失敗した**。

```
Qwen node layer.0.mlp_gate_matmul.qwen38_projection_pack2 failed:
backend status 260: matmul kernel launch: invalid configuration argument
```

cleanupは両target `current_bytes=0 retryable_cleanup=0 durable_quarantine=0`。
NVFP4 W4A4のbaseline rollback経路がQwen3.8のpacked projection nodeに対応していない。
**別の欠陥として記録し、本作業では修正しない。**

### 結論

WMMAはgfx1201で実際に使われており、gfx1030では構造上使えない。
その差はhidden stateを生成するMLP prefillに直接乗るので、
**WMMAは乖離の確認された寄与要因である。** 
一方、唯一の整合手段がこのモデル経路でcrashするため、
**単独原因かどうかは検証できなかった。** 指示に従いこの線の調査はここで終了する。

以下は主張しない: WMMAが8/8すべての`target_hidden`差を説明すること、
WMMAを外せば両GPUがbit一致すること、MXFP8のtop-1優位（cluster検定 p=0.0625で不支持）について何か。

証拠: [WMMA帰属](../../../../../ci/matrix/gfx1201-wmma-attribution-v1.json)、
raw traceは`.local-artifacts/mtp-bench/wmma-check/`。
R9700 serviceはunit／run.sh／binary hash不変、healthz／readyz 200で復帰した。

[保存済み計画](../../../../plans/archive/2026/09/11-20/mxfp8-gpu-divergence-investigation.md)
