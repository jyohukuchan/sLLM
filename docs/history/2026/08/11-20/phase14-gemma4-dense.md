# Phase 14 Gemma 4 Dense履歴

## 2026-08-15: 詳細計画作成

- Phase 13のmodel-neutral executorを利用する二つ目のproduction adapterとしてGemma 4 12B Dense text-onlyを配置した。
- immutable source/model lock、architecture inventory、frontend/adapter、weight/graph、semantic差分、shared executor、
  real-weight slice、RDNA GPU、service、performance bridgeの順にwork unitを分割した。
- R9700をfull-model primary、V620をbounded operator/slice targetとし、VRAM不足の未実行をPASSとしない。
- Gemma 4 Dense完了後はcross-model RDNA performance bridgeへ自動的に進み、goalを終了しないことをqueueで固定した。

## 2026-08-15: A0 source identityとarchitecture inventory

- official base source `google/gemma-4-12B`をresolved revision
  `023679ed352de9bb66cc873c9009ce3482585c08`、Apache-2.0として固定した。
- base sourceにはchat templateとsafetensors indexがなく、単一23,919,549,408-byte
  `model.safetensors`とraw-text tokenizerだけがある。`google/gemma-4-12B-it`を黙って代用せず、
  chat requestはfail-closedとした。
- additive `model-lock-v2`、closed JSON schema、official config fixtureを追加した。direct header 88,952 byte、
  complete header SHA-256 `e432b3ee11ff7f7d179ccbf3827af9669c03a0a28e603000d89c6e1b6c9d4bb7`、
  677 tensor catalog SHA-256 `24e705586f0bba5e1018951a9ee09aa02b1bfccd73f5c0a82e31e29fb7c2931f`を固定した。
- textは48 layer、5 sliding+1 full反復、hidden 3840、MLP 15360、16 Q headである。slidingは
  head dim 256/KV 8/window 1024/default RoPE 10,000、fullはhead dim 512/KV 1/K=V/
  proportional RoPE 1,000,000/rotary 128と分類した。
- direct RMSNorm scale、4 norm/layer、GELU-tanh、embedding sqrt scale、layer scalar、logit softcap 30を
  Qwenと異なる意味として固定した。vision 10 tensorとaudio 1 tensorはknown-unconsumedである。

## 2026-08-15: A1/A2 frontend、verified cache、weight/graph contract

- v1 Qwen lockを変更せず、reviewed model registryがalias+fingerprintからQwen3.5/Gemma 4を選ぶ
  typed dispatchを追加した。
- Gemma verified cacheは全6 locked fileをstreaming hashし、direct safetensorsのmetadata、gap/overlap、
  name/shape/dtype/range、固定sliceをexact-derived catalogと比較する。header-only確認をfull-file PASSへ
  読み替えない。
- official実cacheをrepository外の
  `/home/homelab1/.cache/sllm/models/google--gemma-4-12B/snapshots/023679ed352de9bb66cc873c9009ce3482585c08`
  へ取得し、独立`sha256sum`で全6 fileがlockと一致した。raw model、slice、artifactは追跡していない。
- tokenizer frontendは262,144 IDと13 special roleをexactに検証し、raw text encodeでは公式
  post-processorのBOSを付ける。CLI `verify-model`/`tokenize`/`decode`はreviewed model dispatchを使い、
  Gemma `render`/messagesと未統合generationはfail-closedである。
- weight planは666 text weightを23,814,700,640-byte loadable destinationへ割り当て、audio/vision 11 tensorを
  known-unconsumedにした。full layerのK projectionは`AttentionKAndV`、独立V weightはinvalidとした。
- structural graphは48 layerのdual RoPE、K=V、unit-scale V norm、4 norm、GELU-tanh、layer scalar、
  tied output、softcapを記録し、共通executorへ渡すstate-publicationとterminal-readbackの2 boundaryだけを宣言した。
- reusable semanticはBF16 matmul/add/embedding/argmax、additive direct-scale RMSNormである。新semantic/providerは
  dual RoPE、sliding/full attention、unit-scale V norm、GELU-tanh multiply、scalar scale、logit softcapであり、
  vision/audio/chatはPhase 14 unsupportedである。

## 2026-08-15: official cache実検証とA3 Direct RMSNorm

- release buildのreviewed model dispatchでofficial cache全6 fileを再検証し、677 tensor、666 loadable、11
  known-unconsumed、23,814,700,640 destination byte、weight plan digest
  `sha256:9b555458af54fcb42e8bc64fb73050cfc6b9ff4641f4c426d255439a9f2a6da3`がPASSした。
- actual tokenizerで`Hello, Gemma 4! こんにちは。`をBOS込み9 tokenへencodeし、decodeで
  `<bos>Hello, Gemma 4! こんにちは。`へ復元した。base sourceのchat renderはraw-text-only契約により
  exit 2でfail-closedした。
- RMSNorm scale modeへ既存Qwenのoffset-oneを変更せずDirectをadditiveに追加し、core descriptor、Rust bridge、
  public C ABI、native validation、HIP kernelへ同じmodeを伝播した。fake-HIP launch signatureもproduction
  signatureと一致させ、明示的な数値実行contractでDirectとoffset-oneを区別した。
- 新しいfocused public-C-ABI GPU testはV620 exact `gfx1030`とR9700 exact `gfx1201`で、幅
  `1/3/17/255/256/257/3839/3840/3841/4095/4096`の11 caseをBF16-FP32 oracleへ通した。両targetとも
  fallbackなし、11/11 PASS、`max_abs=0`、`max_rel=0`である。この証拠範囲はDirect RMSNorm単体に限定する。
- shared elementwise ABIへscalar multiply、GELU-tanh multiply、tanh softcapを既存IDを変えずadditiveに
  追加した。core shape/alias contract、Rust bridge、public C ABI、native validation、HIP registry/kernel、
  fake-HIP数値contractを同じ意味へ揃えた。
- V620 exact `gfx1030`とR9700 exact `gfx1201`で3 opそれぞれを長さ
  `1/3/17/255/256/257/3839/3840/3841/262144`へ通した。両targetとも30/30 operation、fallbackなし、
  cleanup anomalyなし、独立BF16-FP32 oracleと全要素exact一致した。structural graphでshared semantic kindへ
  接続し、A3の未対応nodeをdual RoPE 48個とsliding/full causal attention 48個へ限定した。
- Qwen fused mRoPEとは別にsplit-half RoPE contractとdraft HIP kernelを追加した。sliding
  `16/8/256/256/theta 10000`とfull `16/1/512/128/theta 1000000`の2 variantを、両exact targetで
  M=`1/3/17`、position `0/255/262127`開始へ通した。両targetとも6/6 case、fallbackなし、inactive次元
  bit-exact、CPU oracleの`atol=rtol=0.03125`内、`max_abs=0.0214844`だった。public C ABI registry、owned
  execution bridge、KV/attention接続前のkernel draft evidenceとして記録する。
- BF16 Q/K/V、GQA grouping、score scale `1.0`、FP32 softmax、inclusive sliding windowを明示する
  causal-attention contractとdraft HIP kernelを追加した。V620 `gfx1030`とR9700 `gfx1201`でsliding
  short/window境界とfull M=`3/17`の4 caseを独立two-pass CPU oracleへ比較し、両targetとも4/4 PASS、
  fallbackなし、`atol=0.015625/rtol=0.03125`内、`max_abs=0.000244141`だった。public registry、owned
  BF16 KV state、executor接続前のkernel draft evidenceである。
- structural graphのRoPE/attention nodeを対応するbackend-neutral semantic kindへ接続し、全数値nodeが
  semantic kindを持つ状態にした。HIP provider prepareはまだ明示的unsupportedであり、semantic対応をruntime
  PASSへ読み替えない。
- structural graphからPhase 13のmodel-neutral `PreparedExecutionPlan`と`PreparedTransition`を直接生成し、
  node順、state-publication/terminal-readbackの2 boundary、token/start/expected length、binding/state generationを
  保持するhost contractを追加した。Gemma側へ独自wait/cache loopは追加していない。

## 2026-08-15: A3 public RoPE/attentionとA4 transactional publication

- split-half RoPEをversioned public C ABI、native prepared lifecycle、safe Rust owner、model-neutral owned execution bridgeへ
  接続した。通常のpublic static library経由でV620 exact `gfx1030`とR9700 exact `gfx1201`へ7 caseを通し、両targetとも
  fallbackなし、inactive次元bit-exact、`max_abs=0.0214844`だった。
- sliding/full attentionを別のmodel-neutral public providerとして同じowner/bridgeへ接続した。非整列
  `M=3/Hq=3/Hkv=1/D=6`を含む5 caseを両exact targetへ通し、fallbackなし、exact device symbol、
  `max_abs=0.000244141`だった。kernel launchもpublic descriptorと一致してhead dim `1..=512`を受理するよう修正した。
- request-owned BF16 K/V bufferの未確定tailは書込み後もpublished prefixへ含めず、共通transaction guardで
  state-publicationとterminal-readbackの両boundaryが成功した時だけ`committed_length`とgenerationを進めるownerを
  追加した。非整列更新、capacity/stale start、同時transition、boundary順序、forced drop、cancelをhostで確認した。
- H3 public runtimeの直接コンパイル集合を57 file、公開ABIを60 symbolへ同期した。H3/G2/P0 validatorと、gfx1030/
  gfx1201の実compile/link/extract/inspectがPASSした。これはcompile-only evidenceでありGPU実行claimには用いない。

## 2026-08-15: A5 official real-weight embedding/RMSNorm slice開始

- official cache全6 fileをfull hash検証してから、tied embeddingの3実rowとlocked final norm weight 7,680 byteだけを
  bounded readするfocused runnerを追加した。raw weight、入出力、artifactはrepositoryへ保存していない。
- V620 exact `gfx1030`とR9700 exact `gfx1201`でembedding gather、Direct RMSNorm、layer 0のgate/up/down compact
  matmul、GELU-tanh multiply、tied logits compact matmul、実weight Q/K/V projection、q/k/v norm、split-half RoPE、
  sliding attentionの15 operationをpublic Rust/C/HIP経路へ通した。embeddingは
  bit-exact、RMSNorm 3x3840はBF16-FP32 oracle内（`max_abs=0.0625`、`max_scaled_rel=0.006451613`）、
  compact matmul/qkv norm/RoPE/attentionは`max_abs=0`、fallback/cleanup anomalyなしだった。
- この証拠はbounded real-weight sliceに限定する。full dimensionのsingle layer、複数layer、decode state再利用、full model、
  generation/service/performanceは未完了である。

## 2026-08-15: A4/A5 full execution layoutとR9700 full model

- 全958 structural nodeをexact semantic descriptor、tensor view、buffer backingへmaterializeした。weight、constant、
  transition workspace、token/position、request K/V、aliasを区別し、decode K/V tailはattention prefixと同一backingの
  checked offsetへappendする。
- ordered queue上で1 transitionあたり958 node+96 K/V appendをsubmitし、state-publicationとterminal-readbackの2
  boundaryだけでwaitする。両boundary完了前のdrop/failure/cancelでは`committed_length`を進めない。
- model weightを一つの23.8 GB bufferへ詰める最初の試行は、public runtimeのbounded single-allocation contractが安全に
  拒否した。WeightLoadPlanのpacked destination identityは検証に残し、device allocationだけをtensor-sizedへ分割した。
- R9700 exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`でofficial 23,814,700,640-byte text weightをloadし、
  full 48-layer graphをprefill+7 decodeへ通した。生成8 tokenは固定reference `[258882; 8]`と一致し、8,432
  submission/kernel、16 segment/boundary、fallbackなし、peak 23,843,578,492 byte、最終cleanup 0だった。
- immutable weight/constant/queueを持つ`Gemma4ResidentModel`とrequest-local KV/workspace ownerへ分離した。
  request drop後はmodel-resident 23,814,729,316 byteだけが残り、連続requestでweight uploadを繰り返さない。

## 2026-08-15: A6 CLIとOpenAI service

- repository外のofficial Transformers CPU referenceでraw `Hello`を`[2,9259]`へencodeし、greedy 3 token
  `[236764,108,236777]`を固定した。実R9700 CLIは同じ3 tokenと`,\n\nI`を返し、3,162 submission/kernel、
  6 boundary、fallbackなしだった。この外部cross-checkをruntime dependencyやtracked artifactにはしていない。
- server binaryはreviewed lock kindからQwen/Gemma backendを選ぶ。base Gemmaにはlocked chat templateがないため、
  OpenAI messagesはversioned raw transcript `Role: content\n...Assistant:`へ明示変換し、別modelのtemplateを流用しない。
  reasoning history/modeはfail-closedにした。
- 同一R9700 resident processでnon-stream fixed、Unicode SSE、stop string、短いmulti-turnを実行した。意図的な
  200 ms client timeoutはactive SSE requestをcancelし、500 ms後のrecovery non-stream requestが成功した。
- shutdown auditは5 completed+1 cancelledを保持した。各requestのcleanup request-state/workspaceは0で、
  model dropとsession shutdown後はcurrent/request/workspaceすべて0、retryable/durable cleanup 0だった。
- integration reviewでterminal logits未公開によりOpenAI既定temperatureが失敗するproduction gapを検出した。最終BF16
  vocabulary rowをArgmax成功後・transaction commit前にbounded chunk readbackし、既存shared samplerへ接続した。
  実R9700でtemperature省略server requestとCLI `temperature=1/top_p=0.9`をPASSし、cleanup 0を維持した。

## 2026-08-15: A7 bounded direct-engine profile

- official cache検証とresident uploadを一度だけ行い、R9700 exact `gfx1201`で`3/17`と`32/32`を連続計測した。
- short-odd `3/17`: TTFT 998,810,379 ns、prefill 3.019 tok/s、decode 13.774 tok/s、TPOT
  71.486/72.556/73.545 ms（min/median/max）、E2E 2.160 s、peak 23,867,610,772 byte、17,918 dispatch、34 boundary。
- bounded `32/32`: TTFT 88,082,259 ns、prefill 406.642 tok/s、decode 13.434 tok/s、TPOT
  73.442/74.400/75.506 ms、E2E 2.396 s、peak 24,216,250,864 byte、33,728 dispatch、64 boundary。
- 両profileともsubmission=kernel、fallbackなし、request cleanup 0、resident drop後の全runtime cleanup 0だった。
  raw token列は保存せずSHA-256とfirst/last tokenだけをbounded reportへ出し、Qwen/llama.cppとのparity claimは行わない。

## 2026-08-15: integration reviewとcloseout

- 1回のintegration reviewで、OpenAI既定samplingがterminal logits未公開により失敗するcorrectness blocker、C++ formatter後の
  immutable source hash drift、追加Rust bin 3件のdependency closure未登録を検出した。
- 最終BF16語彙行だけをArgmax成功後・transaction commit前にbounded readbackしてshared samplerへ渡した。実R9700の
  temperature省略server requestとCLI samplingで再確認し、fallback/cleanupなしを維持した。
- C++ sourceをcanonical formatterへ揃えてmanifest hashを再生成し、Gemma evidence/profile binをRust dependency closureへ
  登録した。focused validatorとfindingだけのre-reviewをPASSした。
- 最終host evidenceはH0 `513/513`、H1 `421/421`、H2 `36/36` PASSである。workspace Rust test/clippy、C++
  format/static、manifest/schema/workflow、matrix registration、dependency closureもPASSし、Phase 14を完了した。

## 2026-08-15: post-closeout共通RDNA性能bridge

- Phase 14のclosed identityからQwen/Gemma共通profileを取得した結果、Gemmaのdecode BF16 matvecがdevice timeの
  `84.28%`、attentionが`4.07%`であり、RDNA4 FA3-likeは選ばなかった。
- request workspace/prepared semantic再利用とM=1 BF16 matvec streaming loadを採用した。Gemma R9700のfresh
  baseline比で`3/17` `+3.07%`、`32/32` `+3.89%`、V620でも二つ目のcandidate前後に退行なしを確認した。
  詳細なQwen値、profile分類、oracleは[共通RDNA性能bridge履歴](cross-model-rdna-performance-bridge.md)を正とする。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase14-gemma4-dense.md)
