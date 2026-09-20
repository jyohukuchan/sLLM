# vllm-mxfp4 参照追加・実機比較

2026-09-20、ユーザー依頼。外部engineの独立試行であり、sLLMの形式採用とは別。

## 受入条件

- `reference/vllm-mxfp4` を固定SHAで取得し、参照manifestに登録する。
- R9700 1枚で専用モデルを取得・必要な変換を行い、文章生成を試す。
- 既存の8入力・2,632位置・248,077語彙のBF16比KLDと、可能な限り同じ条件の速度を取得する。
- GPU、モデル／image identity、変換、KV・投機方式等の比較差分、失敗を明記する。
- モデル・raw logits・container／build生成物はGit管理外へ保存する。

## 条件

- upstream: `31b9a94a7f74eeb3f59e66d16b1b27dfafcd0663`。
- R9700 `gfx1201`、TP=1。既存V620を混在させない。
- AMD MXFP4本体＋上流converterによるFP8 MTP、追加DFlash2 FP8 draft。
- KLDは既存BF16 baselineと固定token列を再利用し、全語彙raw logitsを比較する。
- 速度の基本条件は8192入力／128出力、1 warmup＋3回、temperature=1、top_p=.95、top_k=20。
- sLLM.mdとmain-planの不一致はユーザー回答でW4A6を正として解消。外部engineは上流のW4A8を試す。

## 状況

2026-09-20完了。sourceを固定取得し、AMD MXFP4＋FP8 MTPとDFlash2を用意した。
R9700でchat smoke、同じ8192/128の投機なし／DFlash速度、既存BF16比の短文・長文2設定の全語彙KLDを取得。
GPU設定、model／image identity、比較差分を記録し、API containerを停止した。

[対応する履歴](../../../../../history/2026/09/11-20/vllm-mxfp4-investigation.md)
