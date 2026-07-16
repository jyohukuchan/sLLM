# P3 qualification-only no-eligible path

## 前回の要点

- Candidate A production activationはP2 qualified GOとselected artifactがある場合だけ有効になり、実P2はrejected NO-GOのままである。

## 今回の変更点

- 権威specを再確認し、7 prompts×10 runs、M=128+other、paired full-model CIはpromotion rawだけの条件と判定した。
- GPU測定値を作らずP2 rejectionを終端化する`qualification_only_diagnostic` raw variantを追加した。
- このvariantはmetrics、capabilities、pairs、CI fieldを持てず、rejected qualification、P3 commit/tree/source archive、build未実行、profile未測定、runtime OFFだけを結合する。
- selector outputへP2 rejection receipt/SHA256SUMS、policy、plan、actual receipt、raw/qualification hashes、P3 implementationを投影し、canonical `no_eligible_candidate`を生成する。
- qualified performance rawのschemaと統計条件は変更していない。
- worker activationはserved-modelのworker binary/package contentとも一致しなければならないようにした。

検証結果: qualification-only/selector/producer mutation testsを含むPython 180 tests、worker 25 testsが通過。GPUとholdoutは実行していない。

## 次の行動

- 実P2 rejection packageからimmutable qualification/raw/selection packageをcreate-newで発行する。
- runtime default OFFとactivation artifact不存在を実行確認し、P3項目別最終監査を固定する。
