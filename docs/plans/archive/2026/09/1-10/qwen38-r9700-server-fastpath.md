# Qwen3.8 R9700 server高速経路の適用

状態: 完了（2026-09-06）。

ユーザーが「FP16 KVでよいので高速経路が適用されるようにして」と指示した。
初回MXFP8配置ではPhase78 opt-in環境変数が未指定だったため、既存FP16 KVとR9700向け
高速経路をサーバーへ接続する。対象モデル・GPU・単一要求・OpenWebUI接続は維持する。

受入条件は、FP16 KVを選択できること、既存Phase78のR9700高速設定を実際の常駐processへ適用すること、
同条件のHTTP decode測定と生成/SSE/再利用を確認すること。未検証の新kernel実装や一般モデルへの
default昇格は含めない。MXFP8配置との比較はKVとopt-inが同時に変わる総合比較として扱う。

同条件greedy decode7.881→19.922 tok/s、temperature0.7は12.822 tok/s。
生成・SSE・キャンセル後再利用・終了時memory0を確認し、最終再起動後のOpenWebUI container経由Chatも成功した。

履歴: [適用・測定記録](../../../../../history/2026/09/1-10/qwen38-r9700-server-fastpath.md)。
