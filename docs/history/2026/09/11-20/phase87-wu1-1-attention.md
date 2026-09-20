# Phase87 WU1.1: attentionのsplit数と採用基準の再判定

## 状態と着手時の基準

完了（2026-09-20）。新しい採用基準を読み直し、探索の打ち切り線と採否を分離する。
V620のcontrolはWU1採用済みC1 tile8/split32、R9700は従来のwave split32。
V620はGQA共有tile8×split64/128、R9700は共有なしsplit64/128/256を比べる。
V620の残る参照余地は約4.61 ms/token、探索の打ち切り線は約2.31 ms/token。
候補は計画の範囲内とし、context別選択は必要になった場合のみ扱う。

## 受入条件

- 独立FP32 oracle、finite、repeat、output/workspace guard、cleanupを両GPUで検査する。
- 指定context 1023/1024/1025/8192/8193/8256×M1..3。causal-tailと強いQ/Kのfixtureも継続する。
  1023は既存runtime fast pathの範囲外なので、単体kernelの境界検査として扱う。
- 300 ms連続warmup、同一process AB-BA-AB、3 round×9 samples。
  stage1・stage2・合計を別記し、採用の短縮量はstage1+stage2合計を使う。
  全roundの中央値差が改善の向きで揃うことを確認する。
- 通常TPOTのcontrolはWU1最終ビルドの実測値、V620 64.7196917、R9700 52.1042577 ms/token。
  1%はそれぞれ0.6471969／0.5210426 ms/token。context8256/M1の合計差×16層を主判定とし、
  他のcontext/Mで退行する範囲は採用から除く。数値はN0/N1の根拠を示せる範囲のみ自動採用する。
- split増による誤差boundを別途解析し、示せない場合はN2として実測とともに提示する。
- 採用した場合は、両GPUのMTPなし／あり8192/128、1 warmup＋3 measured、token差、cleanupを記録する。
  WU1最終controlはsource/build/binary対応を確認して再利用し、不要なcontrol全面再測定はしない。

## 単体結果

両GPU各26ケースでoracle／finite／repeat／guard／cleanupがPASSした（V620 3実装、R9700 4実装、計182 oracle行）。
最大1 BF16 ULP、absolute 0.00195312、relative 0.00689655。split数変更によるcontrolとの差は記録し、bitwise一致を要求しない。

context8256/M1、各27 sampleの中央値は次のとおり。全候補でこの形状の3 roundの改善方向は一致した。

| GPU | split | stage1 (ms) | stage2 (ms) | 合計 (ms) | control合計 (ms) | 16層短縮 (ms/token) | TPOT比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 GQA | 64 | 0.325248 | 0.014520 | 0.339768 | 0.347928 | 0.130560 | 0.202% |
| V620 GQA | 128 | 0.253926 | 0.019320 | 0.273406 | 0.348209 | 1.196848 | **1.849%** |
| R9700 wave | 64 | 0.149243 | 0.013200 | 0.162443 | 0.229765 | 1.077152 | 2.067% |
| R9700 wave | 128 | 0.141083 | 0.017801 | 0.158923 | 0.231245 | 1.157152 | **2.221%** |
| R9700 wave | 256 | 0.135883 | 0.029240 | 0.165283 | 0.230445 | 1.042592 | 2.001% |

V620のsplit128は探索の打ち切り線約2.31 ms/tokenには届かないが、採用の性能条件は満たす。
この二つを混同せず、追加探索は止めて既存候補の採否を判断する。
R9700はsplit256のstage1が最速でもmerge費用が増えるため、合計ではsplit128が最速。

両targetのsplit128はcontext8192/8193/8256×M1..3の全27 roundでcontrolより速かった。
V620の短context/M1等には退行があるため、採用した適用範囲は最終KV長8192以上・M1〜3とし、
それ以外はsplit32を維持する。最小可視長は8190以上となり、以下のN1解析に基づいて既定へ反映した。

## 数値分類: 長context限定のN1

採用scopeは最終KV長8192以上、M1〜3（各rowの最小可視長K>=8190）。短contextはsplit32を維持する。
scoreのQK dot、FP32への復号、score scaleはPに依存せず、有限scoreから選ぶglobal maxも同じ。
partition pの最大値m_pと全体最大Mを使う実数式は次のとおり。

```text
sum_p exp(m_p-M) * sum_{i in p} exp(s_i-m_p) * (1,v_i)
  = sum_i exp(s_i-M) * (1,v_i)
```

Pは32→128のrefinement。L_P=ceil(K/P)とすると、元の各keyの係数が通る
exp/subtraction pathは r_P<=L_P+1、FP32 multiply/add pathの保守的な深さは d_P<=3L_P+2P+C。
stage2のexp総呼出し数ではなく、各元keyが通るstage2 expは1回と数える。
K=8190では(P32→P128) r=257→65、d-C=832→448となり、以後の長いKでも両方非増加である。

通常値域では共通のunit roundoff u=2^-24、exp/subtractionの共通相対誤差eps_rを置く。
共通の係数上界 F_N(P)=(1+u)^d_P*(1+eps_r)^r_P-1 は、P128で非増加。
通常のgamma_d=d*u/(1-d*u)表現を使う場合はd*u<1の範囲に限る。
Z=sum w_i、A=sum w_i*v_i、w_i=exp(s_i-M) とし、
|delta Z|<=F_N*Z、|delta A|<=F_N*sum(w_i*|v_i|)と書ける。signed Vのcancellationは絶対値和で扱う。

underflow／subnormal／zeroを相対exp誤差へ押し込まない。
非正引数に対する同じdeterministic/nonnegative expfの共通絶対誤差eps_a（0化を含む）と、
今回のbinaryのFP32演算はFLOAT_DENORM_MODE_32=3（FLUSH_NONE、source/outputともflushなし）である。
probeのP32/P128と両GPUのproduction kernel descriptorをread-onlyで確認し、ROCm/LLVMのenumへ照合した。
DAZを前提にせず、gradual underflowの有限・overflowなし局所算術誤差 |fl(t)-t|<=rho*|t|、rho<=1を使うと、
F_A(P)=(1+rho)^d_P*(1+eps_a)^r_P-1、
|delta Z|<=K*F_A、|delta A|<=(sum |v_i|)*F_A という極めて粗い共通上界でも非増加となる。
ここではexpの絶対誤差をweighted sumだけで評価せず、unweighted sum |v_i|を使う。

前提はfinite、overflowなし、同じexpf、expf(0)=1、数値的なexact zeroの保存。
最大scoreを含むpartitionのdenominatorは1以上で、そのmerge scaleも1なので、合成後denominatorは1以上。
最終divisionとBF16 RNEには両方式共通の丸め上界を置く。
したがって演算順変更の共通forward bound非増加としてN1を支持する。
pointwise誤差改善、具体的なOCML expf ULP、NaN/Inf payload、token一致、全payloadの同等性は主張しない。
短Kではdが増えるため、この根拠を短contextへ拡張しない。入力DAZを使う別compiler modeへも一般化しない。

## 実装と検証の範囲

- V620はGQA共有tile8/split128、R9700は通常wave split128。context8192未満とM4は従来split32。
- 既存ID93の論理identityを保持し、実際のstage1 gridとworkspaceは分割数128へ同期する。
  V620のgridはM*4*128、R9700はM*24*128。workspaceは3,170,304 byte/query（32版の4倍）。
  stage2はglobal workspaceを逐次mergeし、128*256要素をLDSに置かない。
- 比較controlには`SLLM_CAUSAL_ATTENTION_DECODE_SPLIT128=0`を使える。これは比較用の切替であり、
  rollback環境変数の追加を受入条件にはしていない。
- 新規host境界テストがRust metadataの32固定の置換漏れを検出した。128/32へ同期して再実行しPASS。
  GPU性能の証拠へCPUテスト結果を読み替えていない。

## 統合検証と通常モデル

- 最終release build: gfx1030/gfx1201 PASS。source/binary SHAは`production-build-identity.json`に保存。
  新しいP128 stage1／mergeを含め、最終binaryのFP32 denormal mode=FLUSH_NONEを直接確認した。
- Rust metadata 6 tests、native host selector 1 testはPASS。
- 公開API: 両GPU各28ケース（1023/1024/1025/8191/8192/8193/8256×M1〜4）がPASS。
  独立oracle、repeat、workspaceとgrid、state cleanupを確認した。
- 1回のintegration reviewでcorrectness blockerなし。数値解析は専門的なboundとbinary modeの確認へ限定した。

通常モデルはWU1と同じ8192/128、固定device sampling、MXFP8 E4 KV、1 warmup＋3 measured。
controlの再利用は開始時source/build/binaryがWU1最終版と一致することを確認し、元executionのhashと保存binaryへの
対応を`baseline-reuse.json`へ記録した。raw実測reportは変更していない。

| GPU | MTP | control tok/s | candidate tok/s | 最初のtoken分岐（0始まり） |
| --- | --- | ---: | ---: | ---: |
| V620 | なし | 15.4512 | 15.8311 | 18 |
| V620 | あり | 27.1806 | 28.8826 | 2 |
| R9700 | なし | 19.1923 | 19.5890 | 15 |
| R9700 | あり | 35.1302 | 33.9323 | 38 |

V620 MTPの受理数は74/106→77/102、R9700は78/100→75/106。
モデル速度はV620のMTPなし+2.46%／あり+6.26%、R9700のMTPなし+2.07%／あり−3.41%。
R9700 MTPありの低下を隠さず記録する。生成列とproposal数が異なるので、モデル速度の差をすべてkernel単体の効果とは
扱わない。採用判定は同じ入力を使う単体AB/BAの結果による。数値順序変更によるtoken差はN1の観測結果として残し、
同provider内の生成列は全反復で再現可能である。
全構成でHIP実行、fallbackなし、terminal logits finite、ID93実行、cleanup zeroを確認した。
採否は単体短縮量で判断しモデル全体は記録するという新基準に従い、両GPUのlong scopeへ採用する。
この1つの生成trajectoryのMTP受理率差を、品質差や分布全体の受理率変化と一般化しない。

作業開始時のsource、dirty status、control binaryは`.local-artifacts/phase87/wu1-1/initial-state.json`と
同directoryの`before/`、`baseline-bin/`へ保存した。既存の変更は保持し、WU2へは進まない。

## 証拠と再現入口

[集約結果JSON](phase87-wu1-1-attention-results.json)に採用scope、全形状の3 round差・中央値/MAD、
oracle、数値分類、source/binary SHA、FP32 denormal mode、公開API、モデル前後値・最初のtoken分岐を保存した。
raw sample・生成token列・binaryは`.local-artifacts/phase87/wu1-1/`へ置き、Gitへ追加しない。

- probe: `native/hip/tests/phase87_wu1_1_probe.hip.cpp`をROCm 7.14 `amdclang++`、
  `-O3 -ffp-contract=off -std=c++17 -Wall -Wextra -Werror -x hip --offload-arch=<target>`、
  `-I include -I native/hip/src -I native/lowp/include`でbuildする。
- operator: `python3 ci/tools/run_phase87_wu1_1.py --target <target> --binary <probe> --output <new-directory>`。
- モデル: `ci/tools/run_phase86_mtp_catch_up.py`を再利用。exact環境・job list・binaryは各execution JSONに記録する。
- 集約: `python3 ci/tools/summarize_phase87_wu1_1.py --root .local-artifacts/phase87/wu1-1 --output <summary.json>`。

計画: [Phase87 WU1.1](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu11-decode-attentionの組合せと採用基準による再判定)
