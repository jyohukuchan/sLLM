# Phase 87 WU-P1: Paged Attention試作

2026-09-24完了。128-token blockのPaged Attentionをprobeだけで試作し、現行の連続KV attentionと比較した。
対象はQwen3.8のMXFP8 E4 KV、Q heads 24、KV heads 4、head dim 256、GQA 6、exact V620 `gfx1030`／R9700 `gfx1201`。
本番source、公開ABI、既定provider、KV memory配置は変更していない。

## 結論

- 両GPUともdecode 18条件とprefill 9条件、計27条件でbitwise一致、独立FP64 oracle、guard、再実行、cleanupがPASS。
  数値分類はN0。CPU fallback・timeout・crash・空のGPU選択はPASSへ数えない。
- attention kernel単体の同一process AB/BA/AB比較で、全roundの増加はdecode／prefillとも10%未満。
  最悪の増加はV620が+1.345%／+8.194%、R9700が−16.346%／+3.290%。
  R9700のdecodeの負値は、全roundで現行連続KVより短いことを示す。
- [2026-09-24のユーザー方針](kv-paged-migration-decision.md)の再考条件には当たらない。
  [本移行の独立計画](../../../../plans/archive/2026/09/21-30/paged-kv-full-migration.md)を作り、Phase 88のバッチ処理より前に進める。
  WU-P1の数値・速度は本移行全体やモデル全体の測定結果ではない。

## 試作と比較条件

- `native/hip/tests/phase87_wup1_paged_decode.hpp`: M=1〜3。32/128 splitをpage境界へ合わせ、各区間のblock tableを1回読み、
  現行のGQA共有／wave stage1とstage2の加算順・丸めを保つ。R9700のwave経路ではE8M0の8-byte scale rowをまとめて読み、
  共有のE4M3FN／E8M0復号演算を使う。
- `native/hip/tests/phase87_wup1_paged_prefill.hpp`: 現行qtile4／qtile8を同じQwen形状でprobe化。
  V620のqtile8は4 keyのLDS staging、R9700のqtile8候補はwave-local K/V読み出しと8-byte scale row読出しを使う。
  オンラインsoftmax、QK、value累積、BF16 RNEの順序は維持する。
- `native/hip/tests/phase87_wup1_paged_attention_probe.hip.cpp`と`ci/tools/run_phase87_wup1.py`: 非identityの逆順block table、
  現行連続KV kernelを同じprocess内で呼ぶ対照、AB/BA/AB各5 sample。decodeは16 replay（KV65536のみ4）、prefillは1 replay。
  KV長は1023／1024／1025／8192／8193／65536。decodeは各長M=1,2,3、prefillは各長M=128に加え、
  8192でM=127/129、8193でM=219を検証した。
- 全outputを現行kernelとbitwise比較し、別実装のFP64 oracleは各caseの3 query/head組×6出力次元をsampleする。
  oracleの最大絶対誤差は両GPUとも0.000993未満（閾値0.03125）。このoracleは全要素を独立再計算したものではない。
  全要素のbitwise対照と非identity page配置で、表の無視・転記誤りも検出する。

## 最終GPU結果

| GPU | decode最悪増加 | prefill最悪増加 | 数値 | binary SHA-256 |
| --- | ---: | ---: | --- | --- |
| V620 `gfx1030` | +1.345%（54 round） | +8.194%（27 round） | 27/27 PASS | `253f5edf5d9e32c5d41e407f9d5e6343b154cfed818e915741b8fcdb9486e90a` |
| R9700 `gfx1201` | −16.346%（54 round） | +3.290%（27 round） | 27/27 PASS | `7b6f164d400e6cf6ebd2f1bd91f2354aefbffbc147c3f7d7316d051a796cdc07` |

| GPU・形状 | 現行kernel中央値 ms（AB／BA／AB） | paged中央値 ms（AB／BA／AB） | 増加率 %（AB／BA／AB） |
| --- | --- | --- | --- |
| V620 decode KV8192 M1 | 0.25593／0.25577／0.25598 | 0.25397／0.25377／0.25397 | −0.765／−0.780／−0.783 |
| R9700 decode KV8192 M1 | 0.16155／0.15114／0.15153 | 0.11027／0.11147／0.11225 | −31.747／−26.245／−25.923 |
| V620 prefill KV8192 M128 | 16.93116／16.93472／16.94544 | 16.94664／16.92960／16.94368 | +0.091／−0.030／−0.010 |
| R9700 prefill KV8192 M128 | 12.95853／12.42328／12.43820 | 7.89304／7.78900／7.49859 | −39.090／−37.303／−39.713 |
| V620 prefill KV8193 M219 | 33.37143／33.50499／33.51299 | 33.43679／33.58383／33.59051 | +0.196／+0.235／+0.231 |
| R9700 prefill KV8193 M219 | 17.40745／15.78892／15.79104 | 14.34602／14.03994／14.05118 | −17.587／−11.077／−11.018 |

最終reportは追跡対象外の`.local-artifacts/phase87/wup1/run-gfx1030-final-r1/report.json`と
`run-gfx1201-final-r1/report.json`。GPU UUIDは順に`GPU-76a08c022586fed6`、`GPU-a8e9ddefa2d60f55`。
両reportがsource SHA-256としてprobe本体`78d87963f36710d4ce601b35151e9d65ecc66beae66d602bb1eb3a8023c489fd`、
decode header`efe86fdb257285acc8e9e2ae8100aa904e5504655a46a17dce5fadff374afdb4`、
prefill header`25d6eedb1b3e2001a4e11a5044a09a752b7a9b8a34a88da47a7b35f1dba41803`を記録する。
現行kernel本体のSHA-256は`d10c03d30a670acce5e1f063d4193c6c7263921b7c6f896137fe1ffacd39cd63`。
suite runnerのSHA-256は`2d04a2a30ee9a12096e7d376d775f96bd8a69495c2c83e2b86cdf1fea62a0a2c`。
reportにはcompile command、compiler version、各sample、target、cleanupなども入っている。

## 調整で分かったこととモデル換算

R9700の最初のqtile8 paged候補は、KV8193 M219で現行より+53〜62%遅かった。
M=192まではほぼ同等で、M=193から急に遅くなることを追加probeで確認した。
初期候補のVGPRは105、現行は95だったが、具体的なoccupancyの物理機構はhardware counterで確定していない。
32-bitのkey loopや`launch_bounds`変更では改善しなかった。wave-local K/V読出しへ替え、
E8M0 scaleを8-byte単位へまとめるとM219を上表の短縮側へ戻せた。
最終比較は**最適化したpaged候補と現行連続KV本番kernel**であり、表参照だけの純粋な費用差ではない。

モデルへの換算は記録用の概算に限る。Qwen3.8のfull attention 16層へKV8192 M1のkernel差を単純に掛けると、
V620は約−0.03 ms/token、R9700は約−0.63〜−0.82 ms/tokenである。
prefill KV8192 M128の1 chunkを16層へ掛けると、V620は約−0.08〜+0.25 ms/chunk、
R9700は約−74〜−81 ms/chunkとなる。実モデルではKV append、graph、他kernel、chunk全体、memory配置の影響が加わる。
WU-P1はモデルのTPOT、TTFT、全prefill時間、VRAM peakを測っていない。本移行の最終比較で測る。

## 検証と残る範囲

`clang-format-18 --dry-run --Werror`、`python3 ci/tools/validate_cpp.py --mode format`、
runnerのPython compile、`git diff --check`はPASS。各GPU 27 case、合計54 caseをfinal sourceからfresh compileして実行した。
このprobeの実行対象はexact gfx1030/gfx1201、Qwen3.8のMXFP8 E4、代表MとKV長に限る。
block pool、prefix共有、graph table更新、COW、公開ABI、1M context、8〜16並列は本移行で扱う。

計画: [Phase 87 WU-P1](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#wu-p1-paged-attentionの試作と軽い検証2026-09-24追加)。
