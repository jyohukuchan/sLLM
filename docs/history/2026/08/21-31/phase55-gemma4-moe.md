# Phase 55: Gemma 4 26B-A4B MoE

> 状態: 完了

## 2026-08-31: 対象選定とsource freeze

- WebUI統合完了後のmodel architecture追加として、計画済みで既存資産の再利用範囲が最も広いGemma 4 MoEを最初に選定した。
- semantic sourceを`google/gemma-4-26B-A4B-it` revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`、primary artifactを
  `nvidia/Gemma-4-26B-A4B-NVFP4` revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`へ固定した。
- metadata取得で、hidden 2,816、30 layer、128 routed expert、top-8、expert intermediate 704、5 sliding＋1 full attention、
  static FP8 KV、routed expert NVFP4 block-16のartifact contractを確認した。indexは47,033 tensor、2 shard、
  advertised payload 18,782,360,732 byteである。
- shard headerだけをHTTP rangeで読み、expertごとにgate/up/downのpacked U8 value、E4M3 block scale、FP32 outer scale、
  FP32 input scaleが存在すること、routerがBF16 projection／scale／per-expert scaleを持つことを確認した。
- primary artifactを固定revisionから全取得し、2 shardをそれぞれ
  `10,001,865,236` byte / `b5df31122600666617b05f9be2015552cd2edff401e86b1d99b9127efdc6d819`、
  `8,786,620,352` byte / `ff11061ebf57327af4f1993ff758b0859d7746b0c03ca1b17ded7dec30410962`
  （SHA-256）として検証した。indexの`total_size`はsafetensors headerを除くtensor payload合計であり、shard file sizeとは
  区別して固定する。
- `hf_quant_config.json`はNVFP4 group size 16とstatic FP8 KVを宣言し、attention／router／dense MLPを除外して
  routed expertだけをNVFP4化するartifact構成であることを確認した。全47,033 tensorに`k_scale`／`v_scale`は存在しない。
  producerのModelOpt 0.43契約では既定の`fp8_cast`がconstant amax 448を使いscale bufferを登録しないため、これは
  calibrated per-layer scaleではなく暗黙unit scaleのFP8 castとして固定する。
- full artifact verifierで両shardの全file hash、header/index mapping、range、dtype、shape、catalog digest、text/vision分類、
  11,520 expert projection × 4 planeを検査し、実cache上で541.18秒・PASSを確認した。通常のidentity/router host testは
  19 PASS、外部artifact依存1件のみ明示ignoredである。
- Transformersの公開実装はsemantic referenceとしてのみ読み、source表現やcontrol flowをcopyしない。Gemma routerは
  scaleなしRMSNorm、`hidden_size^-0.5`、BF16 scale、softmax、top-8、top-k再正規化、per-expert scaleの順である。
  sLLM実装は既存のmodel-neutral semantic contractと独立oracleから構築する。

## 2026-08-31: host contract、GGUF、routed expert GPU baseline

- Gemma固有routerのhost oracleとgraph contractを追加し、token境界`1/3/7/8/17/31/32/33`、expert
  `0/127`、stable tie、skew、NaN/Inf、malformed inputをfail-closedで検査した。graphはshared dense branchと
  routed branchの別norm、router用scaleなしRMSNorm、top-k再正規化後のper-expert scaleを一度だけ適用する順序を固定し、
  Qwen用256-expert MXFP4 `SparseMoe`との互換性を明示的に否定した。
- canonical GGUF architecture `gemma4moe`のconverterとCLI kind `gemma4moe-nvfp4`を追加した。11,520 expert
  projectionのE2M1 valueとE4M3 block scaleを標準NVFP4へlossless repackし、F32 outer/input scale、direct
  BF16/F32 tensor、356 vision known-unconsumed tensor、frontend asset、implicit-unit static FP8 KV recipeを保存する。
  固定実artifactの全内容dry-runは35,513 tensor、payload `18,782,360,732` byteでPASSした。actual変換は
  18,824,179,296 byteのGGUFを生成し、SHA-256
  `714cacabd1487e12d14c285e9ab829bd6ae02fe7c3112c26dccea162d056f92d`、metadata digest
  `f38505a882847faaacf83bef43f9154cb11bdb4c0f4ac338be20485dd5771a84`、catalog digest
  `3c08fd2b68a8059af4aa446b70dd5eb0e1b057f722aa31f4326400ad6322145b`として固定した。
  full verifierはderived lock／GGUF SHA、11 source identity、layer 0／29／final direct tensor、expert
  `(layer 0, expert 0)`／`(layer 29, expert 127)`のgate/up/down全plane、597-entry・17,636,771,900-byte
  resident load planを照合してPASSした。変換manifestも11 source、2 output、tool／recipe／output hashを独立検証した。
  actual経路で判明したtensor recipe metadata keyのverifier誤記をcanonical
  `sllm.tensor_recipe{,.sha256}`へ修正し、Hugging Face snapshot symlinkは同一repository直下`blobs`への解決だけを
  manifest identityで許可して、repository外へのescapeを拒否する。検証後は生成した一時bundleだけを削除し、source
  cacheは保持した。
- HIP expert ABIは既存Qwen v1を不変のまま、Gemma用additive v2を追加した。v2はhidden 2,816、
  intermediate 704、128 expert、top-8、layer blob `428,215,552` byte、workspace `27,104` byte/tokenを
  固定し、選択されたexpertだけをNVFP4 W4A4で実行する。R9700 exact `gfx1201`とV620 exact
  `gfx1030`でtoken 1/3を実行し、max absolute/relative error 0、dispatch count 4、fallback allowed/used 0を
  確認した。これはoperator baselineの証拠であり、full resident generationのPASSとはみなさない。
- Router前処理に必要なBF16 `[M,H] * [H]` `BroadcastMul`を独立semantic/HIP opとして追加した。
  exact `gfx1201`と`gfx1030`で`M=1/3`、`H=2816`、lane 31/32、workgroup 255/256、末尾2815、符号、
  skew、Inf/quiet-NaN伝播をhost BF16-RNE oracleと比較し、finite/nonfinite mismatch 0、max absolute/relative
  error 0、fallback allowed/used 0、test-owned cleanup 0でPASSした。
- Gemma固有のrouting metadata生成を`MoeRoute` semantic/HIP opとして分離し、R9700 exact `gfx1201`と
  canonical V620 exact `gfx1030`で独立GPU oracleを実行した。`M=1/3/7/8/17/31/32/33`、128 expert、top-8、
  metadata size `128*M+1032`、stable tieのlower-ID選択、expert `0/127`、full softmax後のselected-weight再正規化を
  全fieldで照合し、metadata/weight mismatch 0、最大weight絶対誤差`5.82077e-11`、最大相対誤差
  `1.08539e-7`、最大selected-weight合計誤差`1.17405e-7`だった。NaN／+Inf／-Infはstatus 1でfail-closed、
  fallback allowed/used 0、test-owned cleanup 0でPASSした。これはrouter operatorの証拠であり、full resident
  generationのPASSとはみなさない。
- runtime監査で、既存`SparseMoe`はrouterとexpertに同一hiddenを渡すQwen固定境界であり、Gemmaの
  router用hiddenとpre-routed-norm後hiddenを区別できないことを確認した。Qwen contractは変更せず、
  Gemmaはrouting metadata生成とexpert実行を別semantic opとして接続する。また全layer static FP8 KVの
  約束に対し、既存sliding-attentionがBF16 explicit stateのみである不足も特定した。

## 2026-08-31: static FP8 sliding KVと継続実行

- logical capacity 262,144を維持しつつ、sliding layerの物理KVを`min(capacity, window+1)`へ限定するring stateを
  versioned ABIで追加した。window 1,024の飽和後は、同一transition内の先行queryが必要なrowを上書きしないよう
  appendを`M=1`へfail-closed制限する。canceled append用spare row、retained intervalだけのimage export/import、
  fork/COW、resident accountingを接続し、cancel後、fork後、fresh restore後にunit-scale E4M3の実byteを再exportして
  一致を確認した。
- full/sliding causal attentionへexplicit score scaleを追加し、Gemma 4 MoEはgraphで固定した`1.0`を渡す。既存ABIと
  legacy `rsqrt(head_dim)`は既存model用に維持する。exact `gfx1201`と`gfx1030`でfull head dim 512／sliding
  head dim 256を独立CPU online-softmax oracleへ照合し、scale `1.0`とlegacy scaleの差を検出した。
  境界`1023/1024/1025`、wrap 1026、prefill/decode、飽和後`M=2`拒否、fallback 0、cleanup 0でPASSした。
- resident/request実行は30個のKV stateをrequest lifetimeで保持し、workspaceとprepared semanticsだけをatomicに
  rebindする`transition_decode`／`execute_next`を追加した。host sessionでprefill 17→decode 17、position
  1017→1034のwindow境界越え、全30 state identity保持、prepare失敗時の旧layout保持、途中dispatch失敗時の全layer
  rewind試行とrequest poison、fresh request recovery、client公開前transitionのcancelを確認した。失敗したrequestを
  再利用せず、Qwen topology、CPU numerical fallback、requestごとのweight展開を使用しない。
- architecture専用の`Gemma4MoeStateImageV1`／`Gemma4MoePrefixStateV1`を追加し、25 sliding layerのretained
  window imageと5 full layerの完全imageを混在させて30 stateをexport/import/forkする。raw imageとprefix forkは
  same-session、portable checkpointはcross-sessionとし、model/source/config/plan/capacity/KV descriptor/window/
  committed length/frontend identityをimport前に検証する。復元後の最初の`execute_next`がimage lengthから`M=1`で
  appendし、過去tokenを再appendしないこと、prefix parent／owner／childのstate ID分離、30=25+5のCOW audit、
  7th import／11th fork fault時の非公開破棄とfresh recoveryをhost adapterで確認した。core全349 testがPASSした。
- terminal Argmaxは既存のNaN sentinel契約を全nonfiniteへ拡張し、NaN／+Inf／-Infをいずれも`-1`でfail-closedに
  する。host fake/oracleとexact `gfx1201`／`gfx1030` public GPU testを更新し、通常finite/tie/boundary結果を維持した。

## 2026-08-31: actual full-resident GPU evidence

- 固定source artifactを用いるenv-gated actual harnessで、canonical load plan、17-token prefill、同一30 KV stateでの
  17-token decode、全layer committed length 34、最後の未公開transition cancel/replay、全argmaxの語彙内／nonfinite
  sentinel不在、fallback 0、allocation cleanup 0を検査した。最初の実行はtest harnessのstate capacityを34として
  sliding window 1,024を満たさずhost側でfail-closedしたため、PASSへ数えずcapacityを1,024へ修正した。
- R9700 exact `gfx1201`はresident `17,636,771,900` byte、peak accounted `17,861,078,900` byte、
  upload 32.258秒、prefill 17は2.901秒、decode 17は8.474秒、submission 17,689、kernel dispatch 19,969でPASSした。
  終了後はtest process 0、VRAM使用57MBへ戻った。
- canonical V620 exact `gfx1030`のdraft runは同じresident／peak、upload 33.652秒、prefill 17は2.225秒、decode 17は
  10.269秒、submission 17,689、kernel dispatch 19,969でPASSした。終了後はtest process 0、VRAM使用16MBへ戻った。
  いずれもoutput 35個（prefill row、decode、cancel/replay監査を含む）が語彙内で、OOM、CPU numerical fallback、
  nonfinite、cleanup残留をPASS扱いしていない。V620の結果はstate image／prefix API追加前のdraft binaryであり、secondaryの
  operator／full-resident適合証拠として保持する。
- state image／prefix APIとrouted expert label監査を含むintegration candidateをR9700 exact `gfx1201`で再実行した。
  最初の強化runはgraph label `routed_experts_nvfp4`を既存監査suffixが数えずMoE submission 0となったためFAILとし、
  exact labelを監査対象へ追加してhost回帰testをPASSさせた後に再実行した。最終runはresident
  `17,636,771,900` byte、peak `17,861,078,900` byte、upload 33.643秒、prefill 17は2.895秒、decode 17は
  8.452秒、submission 17,689、kernel dispatch 19,969、routed MoE submission 570／active pair 8,400だった。
  output 35個は全て語彙内でnonfinite sentinel 0、全30 KV layerはlength 34／capacity 1,024、cancel/replay一致、
  fallbackなし、cleanup retryable 0／quarantine 0でPASSし、終了後はGPU process 0、VRAM使用57MBへ戻った。

## 2026-08-31: source／GGUF同一性とCLI／API／WebUI統合

- 最終candidateで全35 argmax token IDをlittle-endian I32列としてSHA-256へ集約した。R9700 exact `gfx1201`の固定source runは
  upload 31.069秒、prefill 17は2.882秒、decode 17は8.376秒、GGUF runはupload 42.686秒、prefill 17は2.696秒、
  decode 17は8.250秒だった。source fingerprintは
  `69ed6c3b18fcc944d62a4ac8d6357bd760ef0181263f83f1a7f43d0415cb846f`、GGUF SHA-256は
  `714cacabd1487e12d14c285e9ab829bd6ae02fe7c3112c26dccea162d056f92d`で異なるcontainer identityを持つが、output token列は
  両方とも`57c2f914705c86657a3537810e6ed5ba17972b67857c183135d1d0b8a117ccb1`へ完全一致した。両runともresident
  `17,636,771,900` byte、peak `17,861,078,900` byte、submission 17,689、kernel dispatch 19,969、routed MoE
  submission 570／active pair 8,400、fallback 0、nonfinite 0、cleanup 0だった。
- release converterのatomic bundle runはGGUF `18,824,179,296` byte、35,513 tensor、derived-lock fingerprint
  `50fd86e1343646f87e4c56239213a367d41ee81b16fff65fda1bed936844f150`、metadata digest
  `f38505a882847faaacf83bef43f9154cb11bdb4c0f4ac338be20485dd5771a84`、catalog digest
  `3c08fd2b68a8059af4aa446b70dd5eb0e1b057f722aa31f4326400ad6322145b`でPASSした。debug converterを実行中に同じ
  Cargo targetを再buildした一回は最終tool identityの実行ファイルが置換されてfail-closedしたためPASSに数えず、独立した不変release
  executableから再実行した。未公開partialはconverterが回収し、最終bundleだけを公開した。
- 通常CLIのactual smokeで2件のintegration defectを検出した。canonical GGUF metadataのchat-template digestは
  `sha256:<hex>`なのにfrontend factoryだけがprefixなし値と比較していたため正規形へ統一し、prefixなしを拒否する回帰testを追加した。
  またwide prefillは入力行ごとのArgmaxを返すのにCLIだけがsingletonを要求していたため、exact row countを検査して終端行をgenerationへ
  渡すよう修正した。修正後の`sllm generate`はarchitecture専用flagなしで9 input／4 output、exact HIP、static E4M3 KV、fallback 0、
  cleanup 0をPASSした。`--stop What`はvisible outputから一致文字列を除外し、`finish_reason=stop`、decode 1、cleanup 0をPASSした。
- model sourceなしのloopback dynamic `sllm-server`をAPI `127.0.0.1:18080`、WebUI `localhost:65457`、metrics／prefix cache有効で
  標準起動した。`/props`はAMD Radeon AI PRO R9700、exact `gfx1201`、34,208,743,424-byte VRAMを返し、WebUI runtime config、
  `/healthz`、`/readyz`、`/v1/models`、`/metrics`、HTML、model-folder browseをPASSした。bundle folder選択後はalias `model`を
  `gemma4moe`／supported／compatibleとして登録し、WebUIと同じadmin routeでloadするとresident
  `17,636,771,900` byteの`Ready`へ遷移した。
- WebUI requestが互換profile専用の`max_tokens`を送る一方で統合serverの既定がstrict profileだったため、live smokeの400をPASSに数えず、
  clientをcanonical `max_completion_tokens`へ修正した。request body回帰test、WebUI 12 test、lint後、Unicode非stream chat 2件、
  code promptのSSE 1件、raw Completions 1件をactual modelでPASSした。SSEはrole、5 content delta、length finish、usage、`[DONE]`の順で、
  metrics差分は非stream 2件／prompt 44／completion 8、stream 1件／prompt 19／completion 5と一致した。同一promptのprefix再利用は
  同じ出力を返し、client disconnectはstream cancelled／`client_disconnect=1`、active lease 0へ戻った後のrecovery requestは
  200で`OK.`を返した。
- Hugging Face admin経路は導入済み`hf` CLIのstatus、20件のGGUF検索、選択folderに固定した25件のroot GGUF一覧、完全revisionと
  destinationを含むcopy command生成までlive確認した。download実行は外部artifact書込みになるため、この統合smokeでは開始していない。
  unload後はlifecycle `Unloaded`、resident 0、lease 0となり、Ctrl-C終了は`shutdown_audit.clean=true`、API／WebUI両port閉鎖、
  npm／Vinext子process 0、GPU process 0、VRAM約59MBへの復帰をPASSした。
- NVIDIA artifact cardが指定するB200／vLLM reference runtimeはlocal AMD hostでは同一runtimeとして実行できないため、cross-runtime
  token／logit一致は主張しない。本Phaseの数値証拠は固定sourceとlossless GGUFの同一sLLM HIP出力、独立router／expert／attention oracle、
  exact target監査に限定する。同一artifactのNVIDIA reference比較はrelease evidenceのfollow-upであり、local correctnessを偽ってPASSへ
  読み替えない。
- 最終同期後にworkspace全test、workspace check／all-target clippy `-D warnings`、rustfmt、diff check、markdown local-link検査を
  PASSした。WebUIは12 test、typecheck、lint、production buildをPASSした。actual HIP source／GGUF harness、CLI、dynamic API/WebUI
  smokeは上記のexact target／cleanup条件で別にPASSしており、host testをGPU evidenceへ読み替えていない。
- 累積integration reviewはcorrectness／security blocker 0件だった。server wide-prefill行数の明示assert、CLI context clampのserverとの
  統一、CLI `Drop`時shutdown errorの可視化は現行core contract上の不具合ではなくoptional hardeningへ分類した。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase55-gemma4-moe.md)
