# Phase 57: DeepSeek V4 Flash architecture foundation

## 2026-08-31: scope固定

- WebUI／sLLM起動統合とGemma 4 MTP完了後の継続architecture workとしてDeepSeek v4を開始した。
- previewをsupersedeした公式`deepseek-ai/DeepSeek-V4-Flash-0731` revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`、MIT licenseをprimary sourceへ固定した。
- 公式checkpointは48 shard、72,317 tensor、index advertised payload 166,878,536,440 bytesであり、R9700の
  34,208,743,424 bytesへKV／workspace前から収まらない。Phase 55/56相当のsingle-GPU production PASSへ条件を弱めず、
  identity、semantic、operator／verified slice、GGUF dry-run、容量fail-closeのfoundationとしてPhase 57を分離した。
- model名のFlashとspeculative decodingのDFlashを区別した。0731 checkpoint内蔵は3-stage DSparkであり、sLLM要件の
  DFlashを置換しない。両production経路は後段で別identity／別contractとして扱う。
- 外部engine sourceはcopy／adapt／portせず、公式artifact／文書と固定llama.cppの概念cross-checkを
  [reader記録](../../../../references/deepseek-v4-phase57-reader.md)へ分離した。

## 2026-08-31: identity／semantic／GGUF foundation

- exact official `config.json` 1,888 bytes／SHA-256
  `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`と
  `model.safetensors.index.json` 5,602,871 bytes／SHA-256
  `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`をtyped readerへ通した。48 shard、
  72,317 tensor、166,878,536,440-byte payload、43 main layer＋3 DSpark stage、全tensor family／shard coverageを
  fail-closedに照合した。full shard payloadは取得せず、Hub LFS identityとlocal payload hashを混同していない。
- 48 shardのheader prefixだけをbounded rangeで取得し、合計7,998,896 bytes、各header length／SHA-256、72,317 tensorの
  dtype／shape／relative offset／absolute byte range、range gap／overlapなし、shard/index対応、payload合計を照合した。
  exact official ignored testはPASSし、header catalog SHA-256は
  `6d90aa665f26217f4488809b1fdf87a1459702aa4ec46c8b02b44ce66bd4afcc`となった。このdigestはweight payloadを含まない。
- mHC 4 stream／epsilon `1e-6`／Sinkhorn 20、compression ratio 0／4／128のcompleted-block境界、hash layer 0..2、
  score routerの`sqrt(softplus)`、stable top-6、selection-only bias、unbiased weight、renormalization、routed scale 1.5を
  container-neutral host semanticへ固定した。duplicate／out-of-range hash ID、tie、nonfinite、overflowと非aligned入力を拒否した。
- CSA 4:1のprevious-first／current-second 8候補と先頭synthetic zero／negative-infinity row、HCAの非重複128-token block、
  CSA completed blockだけを対象にするLightning Indexer stable top-k 512、独立raw sliding window 128をFP32 oracleへ固定した。
  3/4/5、127/128/129、511/512/513、tie、skew、非aligned feature、nonfinite、shape、overflowの8 testをPASSした。
- `deepseek4`をGGUF reader／writerのreviewed architecture名として受理し、parser上限を100,000 tensorへ拡張した。
  exact official index全72,317行をtarget 67,612／DSpark 4,705へ分類し、direct 1,661、routed expert source 70,656、
  expert stack 138、physical main 1,693／DSpark 106／合計1,799のlossless mappingを構築した。canonical mapping digestは
  `69302fb84672fbafa9e5280e752ba1370a178853cc775f436cac33739d47db91`である。現行GGUF tensor type／recipeは
  block-128×128 E4M3＋UE8M0とsource I32をまだ表現できないため、header／payloadを出力できないfoundation dry-runとして閉じた。
- model libraryでは`deepseek4`をreviewed architectureとして表示する一方、production登録callbackを一切呼ばずgray表示にした。
  理由にはexact minimum resident 166,878,536,440 bytesと、production loader／execution未対応を含めた。

## 2026-08-31: dedicated route operator

- container-neutral `DeepSeekV4MoeRoute` opへscore／hash mode、E=256、K=6、M=1..65,536、stable metadata layout、
  active入力1個＋typed zero placeholderを固定した。Rust→HIP bridgeはinactive placeholderをnative binding化せず、mode、
  renormalize、positive routed scaleを専用ABIへ渡す。
- C ABIにはversioned desc／query／dispatch、専用plan、status、kernel IDを追加し、C++↔Rust layout probeで全constant、
  sizeof／alignment／field offsetをbyte-exactに照合した。host fake runtime CTestは4/4 PASS、Rust wrapper／bridgeの
  DeepSeek V4 focused testは11/11 PASSだった。
- integration reviewで、kernelがmetadata末尾へ書く異常statusを公開completionが成功として返す問題を検出した。同じqueueで
  status 4 bytesをD2Hし、event完了またはdeferred fence確定後、成功公開前に共通validatorへ通すよう修正した。
  nonfinite／out-of-range／duplicate／zero-normalizerは`InvalidArgument`、未知statusは`InternalError`のterminal Failureとして
  query／wait／cached query／deferred finalizeの全経路へ伝播し、positive device completionがある失敗だけを安全に解放する。
  host fake testはstatus 6種とevent accountingを、実GPUtestは異常completionのFailure／diagnostic／releaseを確認する。
- Cargo build scriptにはDeepSeek routeのnative source 7ファイルを`rerun-if-changed`として列挙した。独立target directoryで
  `deepseek_v4_moe_route_runtime.inc`のmtimeを更新し、`sllm-hip-sys` build scriptとnative archiveの再実行を確認した。
- 最終native sourceから作ったCode Object V6／wave32 binaryは、exact V620 `gfx1030`でSHA-256
  `a3190e16ce3abceb304c19fece79662070ecd1d01bc626eed2a8f5f373e162c2`、exact R9700 `gfx1201`で
  `ba7c0f2ef30c3e3668e97acd2a82984f07dc79acad90e4882b570a80eb629a48`となった。両targetでM=1/3/5/17、
  score／hash、stable tie、selection-only bias、unbiased weight、renormalize有無、expert 0／255、duplicate／out-of-range、
  nonfiniteを独立oracleへ照合し、公開completion fail-close、HIP-only、fallback 0、cleanup 0をPASSした。実行前後ともKFD processはなく、VRAM使用量は
  V620 17,162,240 bytes、R9700 59,912,192 bytesへ復帰した。
- このGPU証拠はmodel-free route operatorだけに限定する。full checkpoint resident、mHC／compressed attention native execution、
  expert実行、DSpark／DFlash、full graph、CLI／API／WebUI generation、性能、multi-GPUを証明しない。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase57-deepseek-v4-foundation.md)
