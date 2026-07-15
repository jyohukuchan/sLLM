# AQ4 layer-0 QKV diagnostic probe v6 rebuild

## 前回の要点

attempt1はruntimeの`Aq4MatvecBatch`/`ColdPrefill` admissionでblockedとなり、GPU再実行は禁止している。診断loaderはsingle matvecだけbatch plan解決を省略し、production loaderは従来経路を維持する。

## 今回の変更点

- main HEAD `4a4b0e28eb27fa6710a339e470ee80d21d602680`のclean detached worktreeから、`CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe`を実行した。
- nlink=1、mode0555のimmutable binaryを更新した。binary SHA-256は`f58f0734ec595d9a9cd76161d28d096b2b18fc6e437cf3ad9d526eb710c7cf69`、build receipt SHA-256は`0c94ef027982ff82094053e1fc150294044e25b5c365c6e55009c5cff2cf1637`である。receiptはsource commit、tree_clean、jobs=1、exit_status=0を固定している。helperはCargoの`Compiling`/`Finished`等の進捗行をhash対象から除外し、cold/cached再buildでreceipt内容の完全一致を確認した。
- gateはsource commitとbinary/receipt SHAを固定検証するよう更新した。attempt1 archiveは保持し、BASE直下の実行出力は作成していない。
- GPU、service、holdout、数値Go/No-Go、promotionは実施していない。CPU正式sidecarのbit-exact evidence（output SHA `9683b8c5...82b473`、3 rows、finite、promotion=false）は保持している。

## 次の行動

binary/receipt/gate/tests/docs/journalだけを通常commitする。GPU attempt2は、このrebuildのmock/preflight・CPU再確認・service状態確認・明示的な追加許可が揃うまでNo-Goとする。
