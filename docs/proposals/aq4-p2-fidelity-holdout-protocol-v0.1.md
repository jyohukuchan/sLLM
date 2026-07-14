# AQ4 P2 fidelity calibration/holdout protocol v0.1

## 前回の要点

前回案は、独立BF16 sourceとの数値比較と、同一artifactの `all_m1` との厳密な挙動比較を分離した。現時点ではAQ4の受入れ値を測定から逆算せず、候補昇格を保留する。attempt2の2ケース・3行は、校正値や境界値の根拠に使わない。

## 今回の変更点

### 決定的で互いに重ならない分割

既存の `representative / production_server / cold_prefill / r9700-rdna4 / aq4_0_target` 48ケースを入力とする。内訳は、prompt長4種（1011, 1024, 1339, 2048）× baseline mode 2種（`all_m1`, `cold_batched`）× M 6種（1, 8, 16, 32, 64, 128）である。各8層（stratum）を次の固定式で分ける。

```text
digest = SHA256("ullm.aq4_p2_fidelity_split.v1\\0" || case_sha256)
digest順の先頭3件 = calibration
digest順の末尾3件 = holdout
```

したがって calibration は24ケース、holdout は24ケースであり、各stratumを3件ずつ含む。prompt fixture、fixture hash、case hash、全context hash（canonical token JSON + 改行）、`step=0`、`row_count=1` を各行に固定する。attempt2のケースIDとcontext hashが混入した場合は拒否する。重複、欠落、非有限値、symlink、identity/hash不一致も拒否する。

### 測定前に凍結する指標と式

`policy.json` に式、集計、margin、sample minimum、絶対境界を保存する。連続値のcalibration平均を `μ` とする。

```text
higher-is-better: bound = min(ceiling, max(floor (存在時), μ - max(abs_margin, relative_margin*abs(μ))))
lower-is-better:  bound = min(ceiling, μ + max(abs_margin, relative_margin*abs(μ)))
```

指標は token agreement、top-k overlap（k=10）、logit cosine/relative-L2、hidden cosine/relative-L2、raw hidden max-abs（診断専用）、BF16 top1がAQ4 top10に保持された二項率とする。全指標の `sample_minimum=24` である。token agreementとtop1保持率は24件の成功数 `s` に対する95%片側Wilson下側限界を使い、`n=24` を反映する（mean-minus-marginではない）。top-k overlapの絶対floorは1/10、cosineのfloorは0、連続値のmarginは固定abs=.01/rel=.01（relative-L2はabs=.05/rel=.05）とする。relative-L2は1（100%）を超えた行を病理的ドリフトとして集計前に拒否する。この上限は構造的拒否方針であり、raw hidden max-absには自然な無次元上限がないため、同指標はpromotion boundから外し最大値だけをdiagnostic receiptへ記録する。token agreement/top1保持率には既存AQ4受入れ根拠がないため絶対floorを設けず、Wilson下限を使う。これにより、greedy exactを無理に数値floorへ変換せず、連続指標と二項率の非空疎性を固定できる。

品質指標は現行48 fixtureが自然文ではなく固定token列であることから、`BF16 top1 retained in AQ4 top10`（BF16 sourceのgreedy tokenがAQ4 top-10集合に含まれるか、0/1）にする。token hash、step、ordered top-k identityから計算でき、target tokenを持たないfixtureへ架空のteacher-forced scoreを付けない。自然文品質suiteは実体とidentityが別途固定されるまで必須条件にしない。

active-vs-BF16 envelopeはcalibration 24件だけで一度freezeする。holdoutはfreeze後に一回だけ評価し、holdout値をboundの導出へ戻さない。attempt2の観測値、payload、VRAM/power、producer summary はthreshold sourceとして禁止することをJSONで明記した。

candidate-vs-activeは従来どおり厳密挙動ゲートである。context/token hash、greedy token、順序付きtop-k、KV/cache長、position、scheduler ownership/counter/reset、lifecycle/cancelのいずれかの不一致は即時No-Goとする。

### 実行数、時間、Gate式

- 計画行数は48行（calibration 24、holdout 24）、各行1 full-context/step-zero row。attempt2の3行は除外する。
- split/validatorは48 fixtureのハッシュ確認を含むCPU処理で、実行時間はO(48)の小規模ファイル処理（目安1秒未満、未測定の推定値）である。
- 本ツールはモデルを起動せず、calibration/holdoutのCPU/GPU時間は未束縛・未実測のまま残す。実モデル測定後にidentityと時間をreceiptへ追記する。

holdoutの判定は次の論理積である。

```text
GO = exact(candidate, active)
  && identity_and_shape_ok
  && all_rows_present_and_finite
  && for higher metrics: observed >= frozen_bound
  && for lower metrics:  observed <= frozen_bound
  && bf16_top1_retained_in_aq4_top10_no_regression
  && holdout_evaluation_count == 1
```

どれか一つでも偽、または calibration/holdout のidentity・split SHA・receiptが不一致ならNo-Goである。freeze receiptの時点では `holdout_status=not_started` とし、昇格判定はまだ成立しない。

## 次の行動

1. 実体の48-case fixture indexをidentity付きで受け取り、`split` とvalidatorを実行する（まだ測定しない）。
2. 独立BF16 sourceとactive AQ4についてcalibration 24件を一度だけ測定し、metrics JSONを入力してfreeze receiptを作る。
3. identity、finite、exact behavioral gateを再確認した後、holdout 24件を一回だけ評価し、上記の論理積でNo-Go/Goを判定する。

この提案の機械可読定義は `tools/generate-aq4-p2-fidelity-holdout.py`、検証は `tools/validate-aq4-p2-fidelity-holdout.py` が担当する。現時点ではGPU/service/raw evidenceを変更していない。
