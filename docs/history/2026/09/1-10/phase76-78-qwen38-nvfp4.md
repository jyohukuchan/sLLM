# Phase 76〜78: Qwen3.8 NVFP4統合と単一要求最適化

2026-09-05、ユーザーの「目標に妥協・変更があったことは記録しつつ、Phaseを完了扱いにしたあと、pushして」によりPhase 78を完了扱いとした。
旧条件をすべて達成したという完了ではなく、下記の目標変更・未達・未実施を受け入れた終了である。
Phase 79以降を今回実装したという意味ではない。

## 到達点と目標の変更

| 対象 | 到達点（長文9435/128） | 旧目標 | 終了判断 |
| --- | --- | --- | --- |
| V620 prefill | r25 312.115 tok/s | 340.80 tok/s | 約8.4%未達を受容。目標へは約9.2%改善が必要だった |
| V620 decode | r25 15.445 tok/s | 16.86 tok/s | 実artifact量による実効帯域50%超へ基準変更 |
| R9700 prefill | r23 1145.812 tok/s | 779.06 tok/s | 1 warm＋3 measuredの探索値では上回る。ID72 opt-inを含む |
| R9700 decode | r23 18.828 tok/s | 21.07 tok/s | 未達を記録し、Phase終了の妨げにしない |

V620 decodeはcounter量21.655160 GB/transitionと通常速度から約334.45 GB/s（公称512 GB/sの65.32%）、
parameter payload19.051207 GBだけでも約294.24 GB/s（57.47%）の実効指標となる。
counter測定は通常速度測定とは別であり、物理DRAM utilizationの直接実測ではない。
今後、目標を追うためのモデル固有構造・投影組合せに依存する追加最適化は要求しない。
既存のexact shapeに限定したopt-in実装は保存し、汎用的な性能保証へ読み替えない。

## 実装と確認した範囲

Phase 76で混合FP8/NVFP4 artifact、weight/scale upload、固定recipe、actual-model HIP経路を統合した。
Phase 77〜78でNVFP4 DP4A／prefill、FP8 matmul、attention、KV lifetime、deferred completion、HIP Graph、
projection量子化共有を進めた。r25はV620の2つのM1024 FP8 prefill shapeへ64-bit cooperative loadを接続する。
r26 signedpackなど数値的に正しくても速くない候補は非採用とした。

両targetのrelease build、core／native host contracts、対象kernelの独立数値oracle、実モデルのtoken／文章／停止理由、
HIP-only、nonfinite0、request／resident cleanup0を各記録の範囲で確認した。
最新r25のwhole-model counterも両targetで出力とaudit一致、全kernel coverageを確認した。
V620は180701 trace行−runtime copy/fill787＝179914 audit dispatch、R9700は185240−790＝184450。
R9700計測中のprofile_standardは終了時にautoへ復元した。

## 未達・未実施・採用保留を残す事項

- sLLM最終candidateの4行（17/17、512/32、2048/128、9435/128）3 warm＋10 measuredは未実施。
  したがって全行のllama.cpp比prefill／TTFT条件を正式PASSとはしない。llama.cpp側の固定4行測定は保存済み。
- ID72 gfx1201 NVFP4 full-K FP16 stagingはN2の採用判断保留を維持し、明示opt-inの探索経路として保存する。
  Phase完了やpushをdefault採用・一般的な品質承認へ読み替えない。
- 最終性能探索に含まれる各opt-inは測定済みのshape／targetに限定する。公開ソース全体のdefault経路が
  上記速度を無条件に出すという保証ではない。
- counterはGL2C/EA read-request bytesであり、物理DRAM bytesではない。通常runのGTT/offload保証を
  counterだけから導かない。CPU試験・compile-onlyからGPU PASSを主張しない。

これらを隠して旧受入条件へPASSを付けず、ユーザー承認によるPhase終了条件の緩和として記録する。
将来のリリース品質やPhase 79の採用判断は、その作業範囲で別途扱う。

## 証拠

詳細な測定履歴・数値分類・raw artifact名は継続ロードマップに残す。
raw model、binary、traceはGitへ追加しない。r25 source manifest SHA256は
`12d409672170b2d3bbfa5db90a8afaf4d879a7b62463972796f668c3553514c3`。
V620／R9700 r25 binary SHA256はそれぞれ
`4804159b3877790183193c909bac5400b01c006ed65535aa90c45059c65120e7`／
`5c3dcb65012b0af48d539dbba1aac1b5f743ee6366a9afe4d266e4fad6029bdb`。

## 公開前の確認

整形とlint修正に加え、Rust dispatch検査へID82のK5120/N6144専用symbolを追加した。
既存3形状の検査にこの形状とN6143/6145の境界を追加し、正しいGDN z専用経路を誤って拒否する問題を修正した。
workspace host testsは1415 passed／28 ignored／0 failed、修正後のHIP library tests、workspace clippy、
Rust／C++ format、native host test、Markdown links、環境・hygieneをPASSした。
公開用の両release buildでpublic projection G1と9435/128の0 warm＋1 measured smokeを実行し、
全token／文章／停止理由／audit一致、HIP-only／nonfinite0／cleanup0を確認した。
これは正式3+10性能比較の代替ではない。整形後のartifactを以前のcounter binaryとbyte-identicalとは扱わない。
[公開ビルドのhashと集約検証](phase76-78-qwen38-nvfp4-evidence.json)に証拠を固定した。

[完了計画](../../../../plans/archive/2026/09/1-10/phase78-accepted-closeout.md) ·
[Phase 79以降を含む継続計画](../../../../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)
