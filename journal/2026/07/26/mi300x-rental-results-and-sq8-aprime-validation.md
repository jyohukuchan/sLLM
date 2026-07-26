# MI300X レンタル結果と SQ8_0 CDNA3 A′ 実機検証

## 前回の要点

- SQ8_0 CDNA3 A′ は、installed CK gfx942 XDL instance を使う isolated
  bring-up route として、CPU oracle・format gate・静的 ISA audit までを
  完了していた。実機 fragment/lane、数値、occupancy、performance は未確認
  だった。
- MI300X×1 の借用では、最初の uLLM go/no-go を 10--20 分、A′ continuation
  と外部 engine を含む実行はそれ以上として timebox を置いていた。
- ATOM は別作業の commit a646804f で No-go としており、この借用の追加
  評価対象にはしなかった。

## 今回の変更点

- Hot Aisle の MI300X VF 1 台、gfx942:sramecc+:xnack-、ROCm 7.2.4、
  NPS1/SPX、VRAM 196,288 MB で実行した約 2 時間の結果を
  benchmarks/results/2026-07-26/mi300x-rental-v1/README.md に整理した。
  image digest、14B / MoE の全 sweep 行、artifact revision の確認範囲、
  計測限界を明記した。
- A′ の fragment/lane probe は max_abs=0.007812、max_rel=0.000000、
  256 lane/register coordinate の全単射で pass した。実モデル寸法 5 形状の
  A′ 対 CPU expectation はすべて max_abs=0.000000 だった。
- A′ projection の 200 repeat では、M=128 gate/up full が 249.415 TFLOPS、
  M=1 gate/up tail が 3,019.8 GB/s だった。これは full-model decode や
  実測 HBM 効率ではない。
- physical smoke の旧 device_count()==1 guard は、runtime が CPU を
  index 0 に常設するため GPU 1 枚でも通らない構造的不具合だった。保存済み
  rental patch と一致する修正を本体へ取り込み、selector が受理する gfx942
  device を全 runtime device から一意に選ぶようにした。
- B OCP-to-BF16 control は未修正である。k_or_v_tail_id1 で期待 0.53125、
  観測 0.03125、差 0.5 だった。tail 処理取りこぼしは疑いに留まり、根因は
  未確認である。成功した A′ log は ULLM_SMOKE_SKIP_B_CONTROL で B を skip
  しており、B pass を示さない。
- Qwen3-14B の同一クライアント測定では、C=1 clean p50 が llama.cpp
  49.06、vLLM eager 41.16、SGLang 35.45 tok/s。C=128 sweep は順に
  140.09、2,526.96、1,158.50 tok/s だった。各 request は
  1,010 token prefill + 16 token output なので、これを decode 専用性能と
  読み替えない。vLLM は --enforce-eager で torch.compile 無効だった。
- Qwen3-Coder-Next-FP8 は vLLM request が hybrid attention の block
  size=544 / Triton arange 制約で失敗し、SGLang は動作した。
  Qwen3-30B-A3B-FP8 では AITER Flash Attention が既定 Rocm Attention
  より全 C で遅かった。Qwen3.6-35B-A3B-FP8 の vLLM MTP は未検証である。
- 外部 engine 計画、MI300X validation checklist、CDNA3 port plan を更新し、
  A′ の物理 sub-gate pass と、B/occupancy/performance/production gate が
  未完であることを分離した。借用全体は約 2 時間だが stage 別 timestamp は
  回収されていないため、最短 go/no-go 見積もりを実測で達成したとは書かない。

## 次の行動

1. B control を ULLM_SMOKE_SKIP_B_CONTROL なしで k_or_v_tail_id1 から
   再現し、0.53125 と 0.03125 の差の根因を特定・修正する。
2. B を独立対照へ戻して 5 形状を再実機検証し、A′ / B / CPU の differential
   を成立させる。
3. 同一条件で HIP occupancy/residency、clock/thermal、HBM/L2 counter、
   partition 別結果を回収する。現行の 1 partition timing を一般化しない。
4. full-model logits・prefill/decode と prepack/cache の評価は、B と
   residency gate が成立した後の別の明示承認済み実機 window で行う。
