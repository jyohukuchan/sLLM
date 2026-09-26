# Phase 87: GPU空白時間の再計測（WU2後）

2026-09-20。WU1・WU1.1・WU2の採用後に、段階0で測った「GPUが止まっている時間」を同じ方法で測り直した。
対象はQwen3.8-27B NVFP4、8192入力／128出力、MXFP8 E4 KV、固定GPU sampling、MTPなし／あり。
profileは1 warmup＋1 measured、通常速度はWU2の1 warmup＋3 measured（同一binary）の中央値を使う。
binaryは`.local-artifacts/phase87/wu2/candidate-bin/`のWU2最終版で、採用後のsourceとの差は整形のみである。

## GPU時間と空白（profile計測、ms/token）

| 条件 | GPU実行 | 空白 | profile wall |
| --- | ---: | ---: | ---: |
| V620・MTPなし | 58.016 | 9.554 | 67.570 |
| V620・MTPあり | 31.409 | 5.924 | 37.333 |
| R9700・MTPなし | 43.652 | 7.554 | 51.206 |
| R9700・MTPあり | 26.422 | 5.328 | 31.751 |

段階0との比較（空白、ms/token）はV620なし9.634→9.554、V620あり6.048→5.924、
R9700なし7.874→7.554、R9700あり5.321→5.328で、ほぼ変わらない。

## 通常計測での空白の見積り

profilerを使わない通常計測のTPOTから、profileのGPU実行時間を引いた値。

| 条件 | TPOT | GPU実行 | 空白 | 割合 | 段階0の割合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V620・MTPなし | 62.923 | 58.016 | 4.91 | 7.8% | 8.3% |
| V620・MTPあり | 34.561 | 31.409 | 3.15 | 9.1% | 8.7% |
| R9700・MTPなし | 46.671 | 43.652 | 3.02 | 6.5% | 5.1% |
| R9700・MTPあり | 29.016 | 26.422 | 2.59 | 8.9% | 8.6% |

空白の絶対値は2.6〜4.9 ms/tokenで段階0とほぼ同じだが、GPU側が速くなった分、割合は上がった。

## 空白の内訳

kernel数と隙間の分布は段階0から変わっていない。

| 条件 | kernel数/token | 隙間の中央値 | 50 µs未満の合計 | 50 µs以上の合計 |
| --- | ---: | ---: | ---: | ---: |
| V620・MTPなし | 1,171 | 7.9 µs | 7.793 | 1.759 |
| V620・MTPあり | 523 | 4.8 µs | 3.735 | 2.188 |
| R9700・MTPなし | 1,171 | 4.5 µs | 5.799 | 1.754 |
| R9700・MTPあり | 544 | 4.5 µs | 2.773 | 2.555 |

大きな隙間の内訳も同じで、MTPなしはsampler→`__amd_rocclr_copyBuffer`→次tokenのembeddingの往復が約1.0〜1.1 ms/token、
MTPありは`sllm_kv_state_bf16_to_mxfp8_e4_token_major_v1`とattentionの間が約1.4〜1.5 ms/tokenを占める。
WU1〜WU2はkernelの中身だけを変えたので、host側の挙動は変わっていない。段階5の対象は段階0のときと同じである。

## kernel系統別の変化（MTPなし、ms/token）

| 系統 | V620 段階0 | V620 今回 | R9700 段階0 | R9700 今回 |
| --- | ---: | ---: | ---: | ---: |
| FP8行列積 | 25.046 | 25.261 | 23.103 | 18.511 |
| NVFP4行列積 | 19.722 | 21.655 | 16.220 | 17.211 |
| full attention | 13.319 | 4.838 | 3.455 | 2.161 |
| GPU実行の合計 | 64.387 | 58.016 | 49.491 | 43.652 |

R9700のモデル速度向上（19.59→21.43 tok/s）は、FP8行列積−4.59とattention−1.29を含むGPU実行時間の
−5.84 ms/tokenで説明できる。host待ちの減少によるものではない。

## 調査: M=1のNVFP4 decodeが約10%遅くなる条件

`sllm_matmul_nvfp4_w4a4_decode_scale_lut_v1`（grid 139264）の1回あたり時間を指標に、
V620・MTPなし・同一入力で条件を変えて測った。各行は1 warmup＋1 measuredのprofile実行。

| 条件 | decode attention | NVFP4 µs/call |
| --- | --- | ---: |
| 段階0のbinaryを今日実行 | 旧staged32 | 118.2 |
| `015b1513`のbuild | 旧staged32 | 117.6 |
| `d234e58b`のbuild | GQA共有＋split128 | 128.6 |
| HEAD（WU2最終binary） | GQA共有＋split128 | 129.7 |
| HEAD、split128を無効化 | GQA共有＋split32 | 129.3 |
| HEAD、attention sourceだけ`015b1513`へ差し替え | 旧staged32 | 117.8 |
| 上に未使用kernel 24個を追加 | 旧staged32 | 118.2 |
| 上を16 KiB LDS・192 threadの未使用kernelに変更 | 旧staged32 | 117.4 |
| HEAD、`SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1` | baseline wave split | 127.6 |
| **`015b1513` attention build、同じFORCE_BASELINE** | baseline wave split | **127.8** |

最後の対照が決定的である。**遅いbuildを使わなくても、decode attentionをbaselineに変えるだけで
同じ約10%の低下が出る。** したがって原因はbinaryに新kernelが含まれること自体ではなく、
同じtoken内でNVFP4 matmulの前後に実行されるattention kernelの違いである。
旧staged32のときだけ速く、GQA共有でもbaselineでも遅い。

### 排除できた原因

- 機材・環境の変化: 段階0のbinaryを今日実行して19.705 ms/token（段階0は19.722）と再現した。
- クロック・電力: 0.5秒間隔の記録で、busy時平均235 W対235 W、平均sclk 2311対2304 MHz、mclk分布も同等。
  さらに`GRBM_GUI_ACTIVE`は遅い側が1 dispatchあたり308,532→335,319 cycleと約8.7%増えており、
  clock低下ではなくcycle数の増加である。
- kernel自身のcode: 両binaryの逆アセンブルは、PC相対の定数表offsetを除いて完全に一致する。
  lowpのcode objectは`015b1513`とHEADで`__hip_cuid_`以外バイト一致する。
- 起動設定: VGPR 64／SGPR 128／LDS 1536／scratch 0／workgroup 256、queue・streamまで同一。
- attention workspaceの大きさ: split128を無効化すると792,576 byte（旧と同じ）に戻るが、遅いまま。
- 割り当て配置: 確保列は3,720件中workspaceの1件だけが異なり、他は同サイズ・同順。
  別プロセスで512 MiBのVRAMを確保した状態でも変化なし。state capacityを12288へ変えても変化なし。
- codeの量・資源: 未使用kernelを24個（16 KiB LDS・192 thread版を含む）足しても速いまま。
- 計測系: `SLLM_PHASE87_PROFILE=0`でROCTX範囲を止めても遅いまま。kernel同士の重なりは148,717件中18件。
- メモリ量とcache: 1 callあたりのGL2C EA read bytesは60.505 MB対60.610 MB（+0.17%）、
  GL2C hit率15.20%対15.02%、`SQ_WAVES`は4,352で同一。

### 現時点の理解と次の一手

同じ命令列・同じ起動設定・同じ読み出し量・同じwave数で、cycleだけが約9%増える。
L2のhit率が変わらないことから、増えているのはL2 miss後の待ち時間である。
直前に走るattention kernelの種類でこれが変わるため、DRAM側の状態（page/bank、書き戻しの排出、
MALLの内容）がkernelをまたいで影響していると考えられるが、確定していない。

次に行うと切り分けが進む項目:

1. attentionを含まない単体ベンチで同じ形状のNVFP4 M=1を測り、本来の1 call時間を求める。
   旧staged32が「速くしている」のか、他のattentionが「遅くしている」のかを決められる。
2. 旧staged32とGQA共有でKV読み出しの並びがどう違うかを、同一形状のKV読み出しだけのkernelで再現する。
3. `rocprof-compute`等でNVFP4 kernelのmemory stall内訳を両条件で取得する。

この差は採用済み最適化の利益（V620は今日の同一セッション比較で14.287→15.925 tok/s）を上回らないため、
WU1／WU1.1／WU2の採否は変更しない。

## 保存先

profile、traces、クロック記録、旧binaryの対照実行は`.local-artifacts/phase87/idle-recheck/`。
解析は`ci/tools/phase87_decode_profile.py`（ROCTX範囲`sllm_phase87_decode_`）と
`ci/tools/phase87_profile_breakdown.py`を段階0と同じ条件で使用した。
実行前後でローカルQwenサービスとR9700サービスは停止のまま、performance levelは`auto`を維持した。

計画: [Phase 87](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)
