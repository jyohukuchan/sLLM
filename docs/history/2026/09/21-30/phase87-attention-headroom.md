# Phase 87: attention kernelの改善余地の分析

2026-09-24。WU-P1の計測で、R9700のpaged decode候補が現行kernelより26〜32%速かったことから、
両GPUのattention kernelにどれだけ余地があるかを既存の計測から調べた。新しいGPU計測はしていない。
結果は[Phase 87計画の段階11](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)の根拠とする。

## 使ったデータ

- kernel時間: [WU-P1](phase87-wu-p1-paged-attention.md)の最終report（`.local-artifacts/phase87/wup1/run-gfx{1030,1201}-final-r1/report.json`）。
  同一processのAB/BA/AB、各round中央値の中央値。
- 本番で使われるkernelとgrid: [段階6](phase87-stage6.md)のMTPあり最終profile（kernel trace）。
- 読み出しの参照値: [WU0](../11-20/phase87-wu0-read-bandwidth.md)の「attention用17,448,960 byte」のread probe
  （V620 364.1、R9700 475.2 GB/s）。物理的な上限ではない。
- KVの論理byte数: MXFP8 E4、KV head 4、head dim 256で、token当たり `4 × (256 code＋8 E8M0 scale) × 2（K・V）＝ 2,112 byte`。

## decode

| GPU | KV長 | M | 現行 ms | 実効 GB/s | paged候補 ms | 実効 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 | 8192 | 1 | 0.2559 | 67.6 | 0.2540 | 68.1 |
| V620 | 8192 | 3 | 0.6029 | 28.7 | 0.5957 | 29.0 |
| V620 | 65536 | 1 | 2.1358 | 64.8 | 2.1116 | 65.5 |
| R9700 | 8192 | 1 | 0.1515 | 114.2 | 0.1115 | 155.2 |
| R9700 | 8192 | 3 | 0.3772 | 45.9 | 0.2708 | 63.9 |
| R9700 | 65536 | 1 | 1.2114 | 114.3 | 0.8828 | 156.8 |

実効帯域は、read参照値に対してV620で約19%、R9700で約24%（paged候補で約33%）にとどまる。
KVが長くても同じ割合なので、起動の固定費ではなく処理の中身が律速している。

段階6のprofileのgridから、本番kernelの構造上の原因が2つ読み取れる。

1. **query行ごとにKVを読み直している（両GPU）。** M=3のgridはM=1のちょうど3倍
   （V620 `causal_attention_decode_gqa6_staged32_split_stage1_kernel` 1,536／512 workgroup、
   R9700 `causal_attention_decode_wave_split_staged_stage1_kernel` 9,216／3,072 wave）で、
   MTP verifyの各行が同じKVを別々に読む。M=3の時間はM=1の2.4〜2.5倍になる。
2. **R9700ではGQAの共有がない。** R9700の本番kernelはquery head 24本 × 分割128で3,072 waveを起動し、
   1本のKV headを6つのquery headが別々に読む。重複分はcacheに当たるとみられるが、実効帯域を下げている。
   V620はWU1でGQAを共有しているが、それでも参照値の約19%であり、1 keyずつの逆量子化とonline softmaxの更新が
   律速していると推定する（profilerでは未確認）。

## prefill

| GPU | KV長 | M | provider | 現行 ms | 実効 TFLOP/s |
| --- | ---: | ---: | --- | ---: | ---: |
| V620 | 8192 | 128 | `gqa6_qtile8_w16` | 16.935 | 1.52 |
| V620 | 65536 | 128 | `gqa6_qtile8_w16` | 174.928 | 1.18 |
| R9700 | 8192 | 128 | `gqa6_qtile8_w16` | 12.438 | 2.07 |
| R9700 | 65536 | 128 | `gqa6_qtile8_w16` | 152.940 | 1.35 |

実効TFLOP/sは `2 × 2 × 24 head × M × KV長 × 256` をkernel時間で割った値（chunk内のcausal maskの半減は無視）。
どちらのGPUもscalarとwave内の加算で計算しており、R9700のWMMA、V620のpacked FP16 dot命令を使っていない。
段階6のprofileでは、8192入力のprefillのうちattentionがV620で約8.5秒（約24%）、R9700で約4.5秒（約30%）を占め、
KV長の2乗で増える。

## 余地の上限（目安）

いずれも参照値からの上限で、MXFP8の逆量子化とsoftmaxの費用があるため到達の保証はない。

| 項目 | V620 | R9700 | 前提 |
| --- | --- | --- | --- |
| decode M=1（MTPなし、16層） | 約3.4 ms/token（TPOTの約6%） | 約1.9 ms/token（約4%） | read参照値の80% |
| MTP verify M=3（MTPあり） | 約3.7 ms/token（約11%） | 約2.8 ms/token（約10%） | 3行でKVを共有しM=1並みの費用、1 blockあたり約2.49 token |
| prefill（8192入力） | TTFTの約2割 | TTFTの約3割 | 行列演算命令で数倍〜十数倍 |

R9700の数字は現行kernelに対する値で、paged候補（段階10で本番になる読み方）に対しては小さくなる。
段階11の着手時に、段階10後のkernelを基準に上限を計算し直す。

計画: [Phase 87](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)
