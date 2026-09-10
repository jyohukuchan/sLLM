# MTP採用率のGPU・言語・タスク比較

状態: 測定・比較完了。2026-09-11ユーザー依頼。

## 目的と判定

Qwen3.8 27B NVFP4で直前の8192/128 coding fixtureに観測したV620 69.8% < R9700 78.0%が、他の言語・タスクでも一貫するか確認する。量子化・kernel最適化は行わない。

英語・日本語・中国語で同等の依頼内容を用意し、Pythonコード生成、スケジューリング推論、会議要約、創作の4タスクと組み合わせる。12 prompt × seed 123/456/789 × 2 GPU = 72要求。各GPUの同じresident serverで別のwarmup 1要求後に逐次実行する。最大出力128、T=1/P=.95/K20、MXFP8 E4 KV、MTP幅2、batch1、共通化後source c27e346dを使う。自然な入力長を維持し、8192-token速度基準とは分ける。

各GPUで同一prompt/seedを使い、HTTP/usageとshutdown auditを要求順に対応付ける。提案・採用・棄却token数、出力長、実HIP、CPU fallbackなし、非zero dispatch、要求解放と最終解放を確認する。採用率はaccepted/proposed。seed別、promptごとの合算、言語/タスク別と全体の合算を報告し、逆転・同率も記録する。全体はtoken加重値とし、prompt平均とは区別する。

生成は各GPUの自己回帰出力に従う。同じseedでも生成履歴が異なるため、採用率差を特定kernelの誤差・品質差へ因果帰属しない。1 prompt/task/languageと3 seedの探索的比較であり、全言語・全タスクの普遍則やBF16品質を証明しない。短い出力の打ち切りとEOSも記録する。

## 実行・成果物

既存のexact gfx1030/gfx1201 server artifactをsource hash照合後に使用する。V620のlocal Qwen serviceは停止済み。R9700は既存serviceを一時停止し、元のunit/script/binary不変とhealth/ready復帰を確認する。

raw prompt/runner/resultsは `.local-artifacts/mtp-language-task` と `.local-artifacts/phase83/mtp-language-{v620,r9700}-r1` へ保存。結果と再現用prompt・compact evidenceを履歴に記録し、main-planへ結論を追記する。

結果: 36組でV620<R9700が17、逆転16、同率3。両GPUの合算は66.59%/67.06%。全測定要求と解放はPASS、R9700 service復帰を確認した。

履歴: [条件・結果と限界](../../../../../history/2026/09/11-20/mtp-language-task-acceptance.md)。
