# Qwen3.5 AQ4 P2 full-model driver v0.1

## 前回の要点

P2 engine bridgeは、AQ4 sessionから要求M、実際の物理幅、操作監査、端末ライフサイクルを
個人情報を含まない形で取得できるようにした。既存workerのJSONLプロトコルは変更しない。

## 今回の変更点

ullm-aq4-p2-full-modelは served-model manifestと公開fixtureのtoken IDを読み込み、常駐
Qwen35Aq4InferenceSessionを直接実行する専用オフラインdriverである。Mは
1,8,16,32,64,128だけを受け付ける。

実行例:

    cargo run -p ullm-engine --bin ullm-aq4-p2-full-model -- \
      --served-model-manifest PATH \
      --fixture tests/fixtures/qwen35-aq4-p2-oracle/cases.json \
      --m 128 --output RESULT.json [--case-id fixture-prompt-0] [--device-index 1]

成功結果は一度だけ指定ファイルへatomicにfsync公開される。既存ファイルは上書きしない。
結果にはserved-model/package/binaryのidentity digest、要求/解決/実測幅、生成時間、
operation audit digest、terminal lifecycle/reset、status、OOM、fallbackを含む。
prompt本文、prompt token ID、生成token ID、生成本文は含まない。

driverの既定値はruntime device 0であり、起動中のworkerやサービスを停止・再設定しない。
GPU実行ではmanifestのdevice/required-environmentと同じHIP guardを呼び出し側で設定し、
manifestのpackageを完全に読み込めるVRAMを用意する必要がある。CPU/synthetic fixtureは
parser、fixture安全性、atomic publicationの契約テストだけを実行する。

## 次の行動

実機GPUでmanifest/packageを指定し、Mごとに別結果ファイルへ実行する。結果validatorを
追加する場合も、既存resultを上書きせず、identity digestとreset factsを再計算する。
