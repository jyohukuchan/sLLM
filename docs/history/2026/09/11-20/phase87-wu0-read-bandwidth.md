# Phase 87 WU0: 読み出し専用の帯域上限

## 状態と受入範囲

2026-09-19完了。対象は[Phase 87計画のWU0](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu0-読み出し専用の帯域上限計測のみ)。
WU1のattention最適化は別の作業単位とする。

- 段階0と同じ73 payloadをexact `gfx1030`／`gfx1201`で測る。
- `uint4`読み出し、register累積、block当たり1値の出力によって、全payloadの書き戻しを省いた帯域を得る。
- 予熱と512 MiB以上の巡回条件を維持する。読み出しmodeの巡回容量は実際に読むsourceだけで数える。
- 各blockの結果を独立したhost整数oracleで確認し、非整列tailも含める。
  このprobeは浮動小数点行列積ではないため、checksumを正確に比較する。
- payloadごとのread GB/sを段階0の同じ行へ並べ、論理byte数からWU1／WU2等の短縮余地とその半分を再計算する。

## 基準と計測方法

段階0のread-request counterとcopyの混合帯域によるscoreは、履歴のレビュー補正に従って
今回の上限計算には使用しない。行列積は論理重み量、attentionはK/Vの論理payloadを用いる。
kernel時間と通常の8192/128速度は、推論実装・binary identityを照合した段階0の基準を使う。

copy／readを交互順で各3回起動し、各payloadは3 warmup＋9 measuredで測る。
9回の中央値を各起動の値とし、その3起動の中央値と範囲を記録する。
readの分子は読み出したpayload bytesだけとし、block checksumの少量の書き込みは別のfieldへ記録する。

測定前に算定方法を次へ固定する。行列積の重み量はFP8 `K*N+4*N`（resident channel scaleを含む）、
NVFP4 `N*ceil(K/2)+N*ceil(K/16)+4`とする。別に保持するactivation量は分けて示す。
形状ごとの時間下限モデルは`論理重みbytes / 対応payloadのread帯域中央値`、余地は基準kernel時間との差。
形状別のsigned差と、既存の速い形状を遅くしない場合の正の差の合計を両方残す。
採用候補の打ち切り線は後者の半分とし、算術・復号等を省いた帯域だけの見積りであることを明記する。
attentionの代表はdecode context平均8256のK/V payloadを16層分とし、両端8193／8319の測定範囲も併記する。

元のcopy probeは`.local-artifacts/phase87/wu0/original-phase87-copy-bandwidth.hip.cpp`へ保存し、
段階0に記録したsource SHA-256との一致を確認した。既存の結果は上書きしない。

## 結果

この節と次節の値は、後述の[再計測](#再計測-クロック状態の補正2026-09-20)で置き換えた。

両GPUで同じ73 payload、copy／read各3起動、各payload 3 warmup＋9 measuredを完了した。
最終runは計876行、うちread 438行で、非整列境界を含め全行PASS。
各blockの結果を確認し、sourceの内容はbufferごと・アドレスごとに異なる32-bit整数列とした。
同じcache lineを誤って繰り返し読む実装が定数fillのchecksumを通ることを避けた。

下表はM=1、read帯域の3起動中央値、decimal GB/s。73 payloadのcopy／read帯域と範囲、
段階0の各kernel行に対応する値は[結果JSON](phase87-wu0-read-bandwidth-results.json)を正とする。

| payload／形式 | V620 read GB/s | R9700 read GB/s |
| --- | ---: | ---: |
| FP8 K5120,N1024 | 142.10 | 220.31 |
| FP8 K5120,N6144 | 362.58 | 480.57 |
| FP8 K5120,N10240 | 402.78 | 513.65 |
| FP8 K5120,N12288 | 415.25 | 522.48 |
| FP8 K5120,N17408 | 427.54 | 533.77 |
| FP8 K6144,N5120 | 360.88 | 481.41 |
| FP8 K17408,N5120 | 421.63 | 534.19 |
| FP8 lm_head K5120,N248320 | 450.93 | 634.54 |
| NVFP4 K5120,N17408 | 407.75 | 512.66 |
| NVFP4 K17408,N5120 | 400.64 | 511.26 |
| attention用17,448,960 byte | 300.02 | 415.85 |

read kernelは16 KiB単位で、各threadの4本の`uint4` loadを先に発行してから加算する。
wave内shuffleと1回のblock同期で集約し、blockごとに4 byteだけを書く。
ISAでも4本の128-bit loadの発行とscalar storeを確認した。
1 MiB以上のsource巡回容量の最小値は540,917,760 byteで、512 MiBを上回る。
小payloadの値にはkernel起動・集約処理・cacheの影響が大きく、物理DRAM帯域とは呼ばない。

### 計測器の修正と不採用の試行

- 初期のtile／LDS集約、wave集約とCU数によるgrid制限、flat grid-stride版を比較した。
  いずれも数値は通ったが、既存matmulより低い帯域が観測された。これらを最終値へ混ぜない。
- 最終版は連続した4本のvector loadを先に発行する形にし、full tileとtailの処理を分離した。
  16 byte未満の末尾は、残る32-bit wordと最後の1〜3 byteに分け、host oracleと同じ値を加算する。
- read側だけ各サンプルの間にD2H・host検査が入っていたため、9回分の結果領域を分け、
  全計測後に全blockを検査する方式へ修正した。copyと同じ計測間隔にし、検査を省略していない。
- V620の`profile_standard`診断でも大きな改善はなかったため、最終測定は元の`auto`を維持した。
  診断終了後の設定復元も記録した。
- 試行とsource snapshotは`.local-artifacts/phase87/wu0/`、最終runは`pipeline-gfx1030`／`pipeline-gfx1201`。
  推論kernelの最適化候補の採用ではなく、WU0計測器を成立させるための修正・比較である。

## 余地と打ち切り線

MTPなし、8192/128の段階0 kernel時間を基準にする。行列積のbyteは重みとresident scale、
attentionはGQA共有が成立した場合のunique K/Vで、GL2C/EA要求量は使わない。

| 対象 | 基準 ms/token | read参照での時間 ms/token | signed余地 ms/token | 正の形状別余地の合計 |
| --- | ---: | ---: | ---: | ---: |
| V620 attention stage1（context 8256） | 13.1805 | 0.9299 | 12.2506 | 12.2506 |
| R9700 attention stage1（context 8256） | 3.3740 | 0.6709 | 2.7031 | 2.7031 |
| V620 FP8 projection | 25.0463 | 27.4618 | −2.4155 | 0（下記の制限） |
| R9700 FP8 projection | 23.1025 | 20.9368 | 2.1657 | 2.3594 |
| V620 NVFP4 projection | 19.7225 | 20.7785 | −1.0560 | 0（下記の制限） |
| R9700 NVFP4 projection | 16.2195 | 16.4444 | −0.2249 | 0（下記の制限） |

WU1のV620では、context 8193／8256／8319による余地は12.2512／12.2506／12.2525 ms/token。
代表8256を採り、**WU1の打ち切り線は6.1253 ms/token（表示上6.13）**とする。
R9700で同じscopeを評価する際の参照は余地2.7031、その半分1.3516 ms/token。

WU2のR9700 FP8は、既存の速い形状を遅くしない前提で正の差を合計し、
**余地2.3594 ms/token、打ち切り線1.1797 ms/token（表示上1.18）**へ置き換える。
主な正の差は`K5120,N17408` 0.6995、`K5120,N10240` 0.5443、`K6144,N5120` 0.4664、
`K5120,N12288` 0.3307、lm_head 0.3185 ms/token。
活性値量子化の融合による追加利益は、この重み読み出しの見積りに含めていない。

### 上限という呼称の制限

**これは指定したread probeの達成帯域による参照モデルであり、全kernelに対する物理的な上限ではない。**
V620 lm_headのread probeは約451 GB/s（公称512の88%）で、既存matmulの論理帯域約487 GB/sを下回った。
R9700では大きいlm_head payloadで約635 GB/sまで達したが、payloadごとに同じ割合になるわけではない。
したがって「全形状でピークの90%」という一律仮定は維持しない。

小さいpayloadの起動・集約の固定費、アクセス順、cache、GPUの状態が実kernelと異なり、
特定の差の原因を一つに断定していない。signed差が負の形状を、最適化余地が存在しない証明にしない。
V620 FP8と両GPUのNVFP4にはこのprobeから正の余地を確定できなかったため、
0を新しい打ち切りgateにせず、結果JSONでも`cutoff_applicable=false`とした。
MTPの混在shapeは候補範囲のまま残し、今回の主要打ち切り線をMTPありへ無条件に適用しない。

WU1→WU2の順序は維持する。WU1でV620を主対象、R9700を同時比較した後、WU2のFP8へ進む。
R9700でも今回の参照上はattentionの余地がFP8の正差合計よりわずかに大きくなった。

## 再計測: クロック状態の補正（2026-09-20）

**上の「結果」「余地と打ち切り線」の値は、この節の再計測で置き換える。**

R9700のFP8 lm_head（1.27 GB）だけが634.5 GB/sと、公称比で突出していたため原因を調べた。
メモリクロック最大1258 MHzからGDDR6 20.1 Gbps×256 bit ≈ 644 GB/sであり、634.5 GB/sはその約98.6%である。

- 同じlm_head payloadでもM=1は634.5、M=2／3は563.7 GB/sで、各3起動の範囲は狭かった。
  測る順番を入れ替えると速い条件と遅い条件が入れ替わり、同じアドレスのbufferでも両方の値が出た。
  約330 msの連続読み出しの後では、配置・順番によらずすべて約635 GB/sに揃った。
  差の原因はallocationの配置ではなく、計測時点のGPUの自動クロック状態である
  （メモリ／fabricのどちらのクロックかは特定していない）。
- 元の計測は各payloadの計測前にsynchronizeを挟む3回のwarmupだけで、中規模payloadの多くを
  クロックが上がり切る前に測っていた。連続動作中のdecoderとは条件が異なり、上限を低く見積もっていた。
- tile型の読み出しkernelと、grid全体で連続16 byteを読むinterleaved型を300 msの連続warmup後に比べると、
  R9700の52.5／89.2 MB／1.27 GBで567.2／593.6／634.4と570.6／593.9／634.4 GB/sとなり、差はなかった。
  kernel構造は原因ではない。

計測器に`--warmup-ms`（counted warmupの後、synchronizeを挟まない連続起動を指定時間続ける）と、
host閉形式oracle付きの`--read-kernel interleaved`を追加した。再計測はtile型（WU0と同じkernel）に
`--warmup-ms 300`を加え、その他の条件（73 payload、copy／read各3起動、3 warmup＋9 measured、
512 MiB以上の巡回、全blockのchecksum照合）は同じにした。両GPUで全行PASSした。
V620はGPU-76a08c022586fed6、ローカルQwen停止中。R9700はservice停止中で、performance levelは`auto`のまま。

| payload／形式 | V620 旧 | V620 再計測 | R9700 旧 | R9700 再計測 |
| --- | ---: | ---: | ---: | ---: |
| FP8 K5120,N1024 | 142.1 | 259.0 | 220.3 | 285.4 |
| FP8 K5120,N6144 | 362.6 | 413.0 | 480.6 | 527.6 |
| FP8 K5120,N10240 | 402.8 | 445.9 | 513.7 | 567.4 |
| FP8 K5120,N12288 | 415.2 | 456.8 | 522.5 | 580.9 |
| FP8 K5120,N17408 | 427.5 | 470.1 | 533.8 | 596.0 |
| FP8 K6144,N5120 | 360.9 | 413.6 | 481.4 | 530.0 |
| FP8 K17408,N5120 | 421.6 | 470.8 | 534.2 | 596.8 |
| FP8 lm_head K5120,N248320 | 450.9 | 505.1 | 634.5 | 634.6 |
| NVFP4 K5120,N17408 | 407.8 | 446.4 | 512.7 | 569.7 |
| NVFP4 K17408,N5120 | 400.6 | 444.9 | 511.3 | 567.0 |
| attention用17,448,960 byte | 300.0 | 364.1 | 415.8 | 475.2 |

単位はdecimal GB/s、M=1、3起動の中央値。V620 lm_headの505.1 GB/sは公称512の約99%で、
R9700と同様に大きいpayloadでは公称値近くまで読める。

### 再計測後の余地と打ち切り線

| 対象 | 基準 ms/token | read参照 ms/token | signed余地 | 正の形状別余地 | 打ち切り線（半分） |
| --- | ---: | ---: | ---: | ---: | ---: |
| V620 attention stage1（context 8256） | 13.1805 | 0.7662 | 12.4143 | 12.4143 | **6.2072** |
| R9700 attention stage1（context 8256） | 3.3740 | 0.5871 | 2.7869 | 2.7869 | 1.3935 |
| V620 FP8 projection | 25.046 | 24.102 | 0.945 | 1.052 | 0.526 |
| R9700 FP8 projection | 23.103 | 19.021 | 4.081 | 4.081 | **2.041** |
| V620 NVFP4 projection | 19.722 | 18.890 | 0.832 | 0.832 | 0.416 |
| R9700 NVFP4 projection | 16.220 | 14.807 | 1.412 | 1.412 | 0.706 |

- WU1の結論は変わらない。V620 attentionはread参照の約17倍の時間がかかっている。
- WU2（R9700 FP8）の余地は2.36から4.08 ms/tokenへ増え、打ち切り線は2.04 ms/tokenになる。
  形状別の余地は`K5120,N10240` 1.009、`K5120,N17408` 0.978、`K6144,N5120` 0.850、`K5120,N12288` 0.525、
  lm_head 0.319、`K5120,N6144` 0.200 ms/token。現行hipBLASLtはread参照の約74〜86%である。
- 補正後は両GPUのFP8／NVFP4とも正の余地を持つ。旧値で負になっていたのは計測側のクロック状態による。
  V620 FP8は`K6144,N5120`（0.681 ms/token、参照の約88%）以外はほぼ参照どおり（96〜99%）。
  NVFP4はV620で約0.83、R9700で約1.41 ms/tokenの余地がある。
- これらも読み出し専用kernelによる参照値であり、演算・復号・reductionの費用を含まない。

診断用の配置・順番・kernel比較（単独のHIP program）は記録用の数値だけを残し、sourceは保存しない。
再計測の生出力は`.local-artifacts/phase87/wu0/warm-gfx1030`／`warm-gfx1201`、binaryは
`.local-artifacts/phase87/wu0b/`。集計は[再計測結果JSON](phase87-wu0-read-bandwidth-warm-results.json)を正とする。

## 検証と保存先

- 両targetのHIP compile（`-O3 -Wall -Wextra -Werror`）、最終read/copy 876行、全block checksum／copy全byte検査をPASS。
- 独立Python整数oracleでも両GPU×3 read round×73 payloadのchecksumを照合し、不一致なし。
- build/source/ISA identity、実行command、binary/output hash、設定復元は最終JSONとraw evidenceへ記録。
- 推論binaryとgraph／matmul／lowp route sourceのhashは段階0と一致した。単独probeの変更なので、
  既存の8192/128・1 warmup＋3 measured、KLD基準を再利用した。
- 変更範囲は単独probe、測定・集計器、計画・履歴。WU1の実装は未着手。

結果: [WU0の73 payloadと段階0行への結合](phase87-wu0-read-bandwidth-results.json)

計画: [Phase 87](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)
