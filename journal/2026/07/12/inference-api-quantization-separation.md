# 推論処理とAPI接続の分離

## 目的

AQ4_0、SQ8_0、今後追加する量子化方式を、OpenAI互換APIやOpenWebUIの接続処理へ個別実装せず追加できる構造にする。Qwen3.5 9B AQ4はモデルをワーカー内へ一度だけロードし、要求間ではKV cacheなどの要求状態だけを初期化する。

## 実装済み

- served-model manifestをAPI、worker、product、tokenizer、promotion evidenceの共通契約にした。
- gatewayはformat固有の分岐を持たず、manifestに指定されたworkerを共通JSONLプロトコルで駆動する。
- `InferenceSession`と`SessionInferenceBackend`を量子化非依存の実行境界にした。
- SQ8 sessionを共通worker境界へ移行した。
- Qwen3.5 AQ4のprojection、decoder layer、lm head、model runtime、request reset、inference sessionをlibraryへ抽出した。
- AQ4 manifest経路を常駐sessionへ接続した。legacy CLI経路だけは移行期間用の互換childを残している。
- AQ4 resident契約は`AQ4_0`、`qwen35_aq4_rdna4_v1`、gfx1201、greedy sampling、20個のHIP必須条件を完全一致で検証する。
- OpenWebUIのモデル移行は明示した旧uLLMモデルだけを対象とし、無関係なモデルを変更しない。

## 主なコミット

- `f6d4e31` Qwen3.5 package契約
- `5481ec8` AQ4常駐model runtime
- `5852369` AQ4常駐inference session
- `a8802dd` sessionとworker runtimeの共通adapter
- `b33b187` AQ4 manifest workerの常駐接続
- `c9daabd` AQ4 resident配備profile

## 2026-07-12 実GPUスモーク

- release workerを並列数1でビルドした。
- 実product `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package`（約7.2 GiB）をR9700/gfx1201へロードし、manifest workerの`ready`を確認した。
- R9700のVRAM使用率はロード後約22%。終了後は0%へ戻った。
- 同一resident workerで2要求を連続実行し、両方が`reset_complete=true`で完了した。
- `[9419]`からの4 tokenはresident/legacyとも`[11,353,1044,264]`。
- `[16,10,16,28]`からの4 tokenはresident/legacyとも`[17,147868,97424,98661]`。
- resident decodeは56.29 / 55.96 token/s、legacy decodeは56.28 / 56.32 token/s。計測基準は共通worker timingの`predicted_n / predicted_ms`で一致する。
- resident workerは正常shutdownし、既存`ullm-openai.service`は停止せずactiveを維持した。

## 完了後の判断

- 実product用のpromotion receiptは、実GPU evidenceへハッシュで結び付けて発行した。仮receiptはproductionへ置いていない。
- 長いprompt、EOS終了、length終了、cancel、連続要求、API streaming、OpenWebUI表示を確認した。
- manifest経路は常駐workerだけを使い、`ullm-engine` childを起動しない。
- legacy child経路は初回移行時のrollbackで実際に役立ったため、現時点では明示的な互換・復旧経路として残す。APIやmanifest経路からは分離されており、新しい量子化方式の通常追加には影響しない。

## Manifest配備とOpenWebUI E2E

- `39f255e`でAQ4 receiptを実GPU evidenceへfail-closedで結び付けた。
- 初回manifest配備の1011-token試験で、共通progress trackerが128-token batchだけを入力として許す不具合を検出した。AQ4 residentは1 tokenずつ進めるため最初のtokenでfatalになっていた。
- 不完全なmanifest配備を外し、legacy環境へ戻してから修正した。
- `2c08757`でtokenwiseと128-token batchの両方を受理し、wire通知を128-token境界＋finalへ統一した。library 411件、AQ4 worker 10件、SQ8 worker 5件が成功した。
- 修正後のrelease worker hashは`a7218f269cf88c1fd85e2893ee2e6632ea179defeecd898ecd5a3efd2fc80650`。evidenceとreceiptを更新し、manifest hash `c2ce3265f2e21fcf8ef3e11ff720c860a43988df764090aee450107282edd61b`を配備した。
- 1011-token promptはprogress `[128,256,384,512,640,768,896,1011]`、prefill 63.64 token/s、HTTP 200、length/max_tokensで完了した。
- EOS応答はstop/eos_token、切断はclient_disconnect/cancelled/reset_complete、直後の要求も成功した。
- OpenWebUI browser smokeは`BROWSER_OK`を表示し、情報欄に`predicted_per_second`、`finish_reason`、`termination_reason`を表示した。page errorは0。
- 修正後配備のsystemd `NRestarts=0`、OpenWebUI restart count 0。
