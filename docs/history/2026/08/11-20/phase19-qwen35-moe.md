# Phase 19 Qwen3.5 MoE text-only production path 履歴

## 2026-08-16: 詳細計画作成

- ユーザー指示によりPhase 19の詳細計画を作成した。Phase 20はGGUF統一だけに限定し、
  request batching、chunked prefill、簡易永続化、残るmodel/KV形式をPhase 20から外した。
- README整備と人間による発表はPhase 21に割り当てず、時期を決めないPhase番号未割当の将来タスクへ変更した。
- Phase 19は既存Qwen3.5 DenseのGDN/attention実行とPhase 16Fのlow-bit descriptorを再利用できる
  Qwen3.5 MoEをprimaryとした。候補は`amd/Qwen3.5-35B-A3B-MXFP4`のtext-only component、
  architecture/lineage controlは`Qwen/Qwen3.5-35B-A3B-FP8`とし、exact revisionとlocal 32 GiB収容性は
  P19-A0でfail-closedに固定する。
- 対応範囲はsingle-GPU text-only target generationに限定し、MoE vision/MTP、request batching、
  expert/tensor parallel、CPU offload、GGUF writer/readerを同時実装しない。
- router softmax/stable top-8、OCP MXFP4 routed expert、shared-expert sigmoid gate、weighted deterministic combine、
  decode/prefill別provider、full-model CLI/API、R9700/V620 GPU evidenceの順に実装・検証する計画とした。
- llama.cppはnotice/provenance付きの直接reuse候補、ROCm/ATOM、vLLM、SGLang、LMDeployはreader-onlyとし、
  no-copy境界を計画に固定した。

## 2026-08-16: implementationとartifact固定

- primary artifactを`amd/Qwen3.5-35B-A3B-MXFP4` revision
  `2e19c6576db91e5d5a93455415619262218bf8a1`、semantic sourceを`Qwen/Qwen3.5-35B-A3B-FP8`
  revision `9d1823d2dee688a6b25e77009dc727688c44936e`へ固定した。text-only inventoryは62,053 tensor、
  22,009,481,856 source byte、model fingerprintは
  `sha256:5bca203f6ec8ab9cab4e340a6c337fff7387f9ca2fa12526c48ce999748e83b0`である。
- strict config/tensor/index/shard/support-file/license検証、expert ID/projection/value/scale plane、40個のimmutable layer blobを
  container-neutral load planへlowerした。execution planは493 entry、digest
  `sha256:f96a3389cfaca4ab947fe060ccd6f048d078946e704464277d87019a13fb7ae4`となった。
- common semantic graphへ`SparseMoe`を追加し、Qwen adapterでBF16 router、stable top-8、OCP MXFP4 routed expert、
  BF16 shared expert/sigmoid gate、weighted combineを接続した。40層の3 GDN + 1 full-attention schedule、GQA 16/2、
  hidden 2048、vocab 248,320をDense固定shapeから分離した。
- HIP route/expert C ABI、安全なRust bridge、decode/prefill kernel、request-local metadata/workspaceを追加した。
  model weightはresident lifetimeで一度だけuploadし、requestごとのexpert upload、host routing、CPU expert executionを正常経路にしない。

## 2026-08-16: correctness defectとoracle matrix

- 最初のfull-model generationが入力を反復したためproduction PASSにせずstageへ戻った。native MoE decoderがOCP E2M1
  code 7/15の±6を0としていた欠陥を特定した。actual layer0 gate expert 0だけでも該当codeは12,938要素あり、
  decoderへ6.0を追加した。修正後、同promptは`Thinking Process:`を生成し、APIは`Hello! How can I help you today`を返した。
- NumPy actual-weight oracleとHIPをlayer 0/19/39、expert 0–7/124–131/248–255、M=1/3/7で両target照合した。
  active pairは8/24/56、最大絶対・相対誤差は`1.86265e-9`、fallback 0である。router matrixは
  M=1/2/3/7/8/31/32/33、tie/nonfinite/skew、expert ID境界を別matrixでPASSした。通常caseはactive expert
  8〜166、最大expert count 1〜3、all-tie caseはactive expert 8、最大count 3であり、両targetで一致した。
- full-model auditへSparseMoe submissionとactive pairを追加し、prefillは40 submission/960 pair、decodeは
  40 submission/320 pairをexactに要求した。これによりtop-8以外または256 expert全件実行への退行をfail closedにする。

## 2026-08-16: full-model、性能、service evidence

| target | physical identity | prefill median / MAD / p10–p90 | decode median / MAD / p10–p90 |
| --- | --- | --- | --- |
| R9700 `gfx1201` | UUID `GPU-a8e9ddefa2d60f55` | 216.258 / 0.501 / 215.715–217.028 ms | 204.198 / 0.514 / 203.669–205.157 ms |
| V620 `gfx1030` | UUID `GPU-08b2ddcbd6e6b36c` | 537.832 / 1.747 / 534.182–541.349 ms | 370.711 / 0.202 / 370.065–371.401 ms |

- 各targetはROCm 7.14.0、LLVM 23、Code Object V6、wave32のtarget別release buildで、2 warmup + 11 measuredを
  同一model/prompt/token oracleのresident上で実行した。両targetのprefill tokenは`[30350,87001,12]`、decode tokenは
  `2972`、replayは一致し、全dispatch HIP、fallback false、cleanup 0だった。
- resident currentは22,009,574,016 byte、request state 129,474,560 byte、workspace 17,982,024 byte、
  high-waterは22,230,758,892 byteだった。full modelのloadとexecutionを分離し、R9700/V620のartifact検証は約14.2秒、
  uploadは約45.8秒だった。
- 通常CLIはmodel directoryだけでMoEを自動検出し、`generate`で`Hello` promptから有効なreasoning prefixを生成した。
  serverは`--cache`省略、MoE/low-bit flagなしで起動し、OpenAI non-stream、Unicode SSE、EOS/stop、usage、連続requestをPASSした。
  1秒で切断した長文SSEは`cancelled`となり、直後のrequestは`recovered`を返した。
- OpenAI `seed`とCLI `--seed`をsamplingへ接続した。同一prompt、temperature 0.8、top-p 0.9、seed 1902の二回は
  `GPUs are specialized processors designed to accelerate`と同じusageを返した。greedyは引き続きrandom sourceを読まない。
- shutdown auditは全requestのrequest-state/workspace cleanup 0、final current/model/request/workspace 0、
  retryable cleanup 0、durable quarantine 0だった。

## 2026-08-16: integration reviewとfocused re-review

- integration reviewで、artifact hash検証後にexecution uploadがshard pathを再openしており、検証後のpath置換を同じ
  verified artifactとして読めるcorrectness blockerを検出した。verified ownerが全shardのopen file descriptorと
  device/inode/size/mtime/ctimeを保持し、uploadは同じdescriptorのpositional readだけを使うよう修正した。
  config/index/support fileも同じdescriptorから上限付きで読み、読み込み前後のdescriptor/path identityを照合する。
  同一inodeの内容変更を拒否するfocused unit testと、現行sourceの24.6 GB actual artifact全identity/inventory testを
  release buildでPASSした（artifact検証14.29秒）。
- additive MoE C ABIのdescriptor/dispatch-info 4構造体がC/Rust ABI layout probeの対象から漏れていたため、checked-in
  C probeとRust expectationへ追加した。host CMake static buildと`sllm-hip-sys` ABI testでsize/alignment/offset一致を確認した。
- OpenAI Chat Completionsの`seed`を当初`u64`で受けていたが、固定OpenAPI commit
  `117ce5680e4269f6656a4fd70d28f9755630d938`のschemaは`int64`である。wire/APIを`i64`へ直し、負値はbit patternを保って
  sampling RNGへ渡し、`i64::MIN`/`MAX`の両端を受理して範囲外の両側をinvalid JSONとして拒否する境界testを追加した。
  CLIの独立した`--seed U64` contractは変更していない。
- host側はworkspace全test、all-target clippy warning-deny、Rust format、Python oracle 7件、C++ static、変更C/C++の
  clang-format、markdown local link、diff checkをPASSした。current source/buildでR9700 `gfx1201` full-modelをfocused再実行し、
  fingerprint/plan digest/token/replayは不変、774 dispatchずつ、SparseMoe 40 submissionずつ、active pair 960/320、
  HIP-only、fallback false、cleanup 0を確認した。2 warmup + 11 measuredはprefill 216.184 ms
  （MAD 0.575、p10/p90 215.557–218.200）、decode 204.047 ms（MAD 0.490、p10/p90 203.557–205.439）だった。

## 2026-08-16: closeout

- Phase 19のsingle-GPU、batch 1、text-only production pathを完了した。MoE vision/MTP、request batching、
  expert/tensor parallel、CPU offload、GGUF writer/readerは範囲へ追加していない。
- Phase 20へcontainer-neutral MoE config、expert-axis inventory、mixed recipe、verified load plan、tokenizer/chat metadataを渡す。
  次Phaseはユーザー決定どおりGGUF統一だけを扱う。
- `cargo fmt --check`、workspace全target clippy（warning deny）、workspace全test、Python oracle 7件をPASSし、
  最終sourceで両GPUのrouter matrixを再build・再実行した。markdown/link/diff checkを含むintegration reviewと、findingを変更した
  artifact binding、ABI、API seed、R9700 full-modelだけのfocused re-review後に計画をarchiveした。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase19-qwen35-moe.md)
