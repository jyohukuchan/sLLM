# 追加アーキテクチャ対応の調査・軽量検証 v0.1

## 前回の要点

- Qwen3-14B の SQ8_0 と Qwen3.5-9B の AQ4_0 は既存本番経路であり、今回の目的はそれらの
  production gate を再実行することではなく、Gemma4 と Qwen3.5 dense/MoE の立ち上げに必要な
  architecture 差分を事実確認することだった。
- CPU 64 core で既存 FP32 reference corpus が進行中であり、GPU と重い CPU 利用を避ける必要が
  あった。

## 今回の変更点

- qwen3_loader.rs が config.json / architectures を一切読まず、Qwen3 型の
  attention + SiLU dense MLP + 固定 residual layout を直接読む loader であることを確認した。
  Qwen3.5-9B は別の AQ4_0 runtime で既対応だが、SQ8_0 は Qwen3-14B 固定 contract のため
  未対応である。
- Hugging Face の実 config.json を local checkpoint と照合した。Gemma4 の最小実体は
  google/gemma-4-E2B、Qwen3.5 MoE の architecture は
  Qwen3_5MoeForConditionalGeneration だった。三対象の checkpoint は全てローカルに完全に
  存在し、gated ではなく、config hash は remote revision と一致したため重複 download は
  行わなかった。
- Gemma4 E2B は local/full attention、mixed head width、複数 norm、PLE、tied embedding、
  logit soft-cap を持つ。Qwen3.5 dense は hybrid linear/full attention、Q output gate、
  mRoPE、1+weight norm を持つ。Qwen3.5 MoE は 256 experts/top-8、shared expert、3-D
  expert weights を持ち、grouped GEMM/routing が新規に必要である。
- tools/architecture_hf_trace.py を追加した。これは corpus/campaign を import しない
  HF CPU reference capture と trace comparator で、embedding、各 layer、final norm、logits を
  F32 schema に保存する。Qwen3-14B-FP8 を CPU 8 threads、1 decode step で実行し、40 layer を
  含む 43 tensor の HF trace を 58.9 秒で取得した。自己比較は 43/43 pass、synthetic の
  layer 3 一要素差は layer 3 に局在して reject した。
- GPU を使わない制約に従ったため、実 uLLM SQ8_0 candidate trace は採取していない。従って
  HF-uLLM 数値一致は未確認であり、既存 corpus を代替として使っていない。

## 次の行動

1. 新 architecture の実装前に、ullm.architecture_trace.v1 を出す diagnostic-only の
   uLLM trace writer を最小化し、既対応 Qwen3.5-9B AQ4_0 から独立 HF comparison を行う。
2. 新規実装を一つだけ選ぶ場合は、SQ8_0 が必須なら Qwen3.5-9B SQ8_0（28--48 h）、
   format を増やさないなら Gemma4 E2B text-only（48--72 h）を明示 scope で選ぶ。
3. Qwen3.5-35B-A3B MoE（72--120 h）は grouped GEMM/routing を要するため最後に置き、
   残り共有時間で完遂すると約束しない。
