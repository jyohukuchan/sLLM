# SQ8_0 R9700 手書き WMMA projection prototype の実現性調査

Date: 2026-07-26

## 前回の要点

- decode の時間占有は paged decode attention 51.05%、CK projection
  40.13%で、attention 側の安全な即時候補は尽きた。projection は
  gfx1201 固定の CK 実装であり、手書きの余地を調べる価値がある。
- static CK metadata では 128x256 の二形が LDS 36,864 B、VGPR 242/175、
  256x128 が 34,816 B/VGPR 154、128x128 が 18,432 B/VGPR 100 だった。
  64 KiB LDS のみを基準にすると前3形は 8 wave32（32-wave reference の
  25%）であり、LDS が設計上の主な勝ち筋だった。
- source-tile split の経験から、partial reduction を merge する経路は
  standalone の微小差でも multi-step SQ8 decode で増幅し得る。よって
  projection も full-model feedback gate を性能測定より先に置いた。

## 今回の変更点

- runtime/src/sq8_handwritten_gfx1201.hip.cpp に private だけの
  M=1/decode WMMA body を追加した。外部 C ABI、public header、legacy
  dispatch、既存 CK body、通常 serving の CK 選択は変更していない。
- body は gfx1201 の v_wmma_f32_16x16x16_fp8_fp8 を rocWMMA 経由で
  使い、raw OCP E4M3FN payload と canonical K128 scale-block を扱う。
  artifact weight scale は BF16 [128,128]、runtime activation scale は
  既存 quantizer の F32 [M,128] 出力であり、後者を BF16 と誤認していない。
- explicit test profile と component harness、CK control、full-model
  comparator、one-window script を追加した。normal/default 実行は候補を
  選ばない。
- static CCOB audit は wave32、32 threads/workgroup、LDS 1,280 B、VGPR
  47、SGPR 24、private 0、spill 0、WMMA 8本を確認した。LDSだけなら
  32 workgroup は 40,960 Bなので、CK大形の25% ceilingとは違い、LDS が
  32-wave reference を妨げない設計である。ただしこれは実測 occupancy
  ではない。

## CK baseline と数値 gate

- 実モデル M=1 mapping は q/o と k/v が Default 16x128x128、gate/up が
  KPadding 16x128x256、down が Default 16x128x256。全 N/K は128整列、
  M=1だけが MPerBlock=16 の tail だった。
- HIP event の CK control（helper + BF16-to-F32 boundary）は、7 projection
  / layer で 571.8696 us、論理 route traffic 578.36 GB/s、640 GB/s
  nominal reference 比 0.9037だった。q/o の論理 rate は 1,001.72 GB/s
  と nominal を超えるため、これは物理 HBM帯域ではない。PMC byte counter
  が unusable のままで、physical HBM 効率は **未確認** である。
- 測定前に固定した条件は、(a) CK BF16 workspace boundary 後の4実形状で
  finite/F32 bit一致、(b) candidate を実際の M=1 projection 経路として
  少なくとも2 feedback decode step、generated ID/top-1/hidden/logits を
  CKとbit一致、の両方である。
- component は4/4 passした。しかし full-model は token
  [66,198,197,197] が同一でも step 1 から hidden 5,120/5,120 と logits
  151,936/151,936 が不一致だった。step 1--3 hidden max abs は
  0.387939/0.797844/1.287994、logits は 0.189508/0.183819/0.250601。
  finite でも gate 不合格なので candidate event timing は意図的に未実行。
- component raw fixture は当時、有限 payload の小さい cycle と BF16起源
  activation scale だった。後で全有限 payload code と varied F32 scale に
  source fixtureを強化したが、service windowを増やさないため再実行して
  いない。full-model failure が決定的である。

## 判定

この body は **numerical NO-GO**。LDS/VGPR の大幅な削減は、将来の
occupancy 余地を示すが、CKより速いという実測証拠ではない。full-model
不一致の厳密な source-level 根因は **未確認** である。WMMA fragment/lane
layout または K128 partial accumulation association が CK と違うことは
仮説であり、原因として断定していない。

次は first divergent layer の actual-artifact input/output を採取し、K128
partial ごとにCKとの差を局在化する。CKと同じ association/fragment contract
を再現し、強化componentと同じmulti-step gateをpassした場合だけ、新しい
承認済み R9700 windowで性能を測る。default CKはそれまで維持する。

## 非干渉とサービス窓

- R9700 は AMD SMI GPU 2（gfx1201、0000:47:00.0）だけを使用し、V620 は
  使用していない。
- stop/isolate/restore は2回。第1回（08:30:12--08:30:48 JST）は AMD SMI
  の no-process sentinel parser 不備で GPU kernel 前に中断し、serviceを
  復旧した。第2回（08:31:52--08:33:27 JST）が実測で、restore後
  ullm-openai.service=active/running、NRestarts=0 だった。
- llama-qwen35-udq4.service は inactive/disabled、gdm3 は inactiveを
  preflight/finalで記録した。unit 内容、power cap/profile、active manifest、
  campaign、authorization、/opt/ullm、remote は変更していない。
- 第2回93 sample は edge 36--46 C、hotspot 37--60 C、memory 34--48 C、
  gfx 0--3421 MHz、memory 96--1258 MHz、socket power 7--204 W、
  THROTTLED/UNTHROTTLED 22/71 だった。throttleの物理原因は **未確認** で、
  timingには条件付きの扱いが必要である。

raw evidence は
benchmarks/results/2026-07-26/sq8_0-handwritten-projection/ に保存した。
