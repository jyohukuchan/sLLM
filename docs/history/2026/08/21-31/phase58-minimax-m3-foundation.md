# Phase 58: MiniMax M3 architecture foundation

## 2026-08-31: scope固定

- WebUI／sLLM起動統合とPhase 57完了後の継続architecture workとしてMiniMax M3を開始した。
- 公式`MiniMaxAI/MiniMax-M3` revision `f0e1c1e04d40177e4673a22097036854f536e9c0`、
  MiniMax Community Licenseをprimary sourceへ固定した。
- exact official `config.json`は5,254 bytes／SHA-256
  `c9c97ce1e4eece60012d5a10ea87717458bfb1f19c2c7a615a3dbff83d090c6b`、
  `model.safetensors.index.json`は2,706,437 bytes／SHA-256
  `54dbde502126d07f6999077437a06b5df1f71e317518956d0aad1c8197df524e`である。
- 公式59 shardのfile size合計は854,176,398,808 bytesだが、indexの
  `metadata.total_size=869,157,697,024`は14,981,298,216 bytes大きい。これを公式manifestの不整合として保持し、
  allocation admissionでは大きい方を使ってfail-closeする。BF16／公式MXFP8／AMD MXFP4／NVIDIA NVFP4のいずれも
  現行local GPU topologyへ収まらないため、full-model production PASSへ条件を弱めずfoundationとして分離した。
- 公式MSA paper／repositoryはsemantic referenceに限る。NVIDIA SM100／CUDA専用sourceをAMD HIPへportせず、
  block selectionとcausal main attentionを独立oracleで実装する。

## 2026-08-31: foundation実装完了

- strict config／index parser、59 shard identity、23,416 tensor catalog、bounded safetensors header readerを追加した。
  header prefix 3,440,088 bytesから算出したpayloadは854,172,958,720 bytesで、shard file合計854,176,398,808 bytes、
  index宣言869,157,697,024 bytesとの三値を保持する。容量admissionは最大値869,157,697,024 bytesへfail-closeする。
- dense layer 0..2とMSA layer 3..59を分離し、block 128、stable top-16、4 GQA group、current local block、partial block、
  exact causal softmaxをFP32 oracleへ固定した。sigmoid top-4 MoEはselection-only bias、unbiased score正規化、routed scale 2.0、
  unscaled shared branchを独立oracleへ固定した。released indexにMTP tensorがないため7 MTP moduleのfull generationは接続しない。
- canonical `minimax-m3` GGUF dry-runはsource text 22,893、vision／projector 523、routed expert source 21,888を分類し、
  expert-axis stack 171とdirect 1,528を合わせたphysical candidate 1,699を生成する。mapping digestは
  `93ad9f5467bb9a7ba3b77c96db5aa0641e5d9e9801f99dc49bf46a8a4a18dd3f`である。payload変換と書込みは無効のままである。
- model libraryは`minimax-m3`を認識するが、Community License、manifest不整合、最低resident bytes、production loader未対応を
  灰色表示し、dynamic production aliasへ登録しない。
- MiniMax M3専用E=128／top-4 public HIP route operatorを追加した。M=1/3/5/17、stable tie、expert 0／127、selection bias、
  unbiased sigmoid weight、scale 2.0、nonfinite／zero normalizerをhost oracleへ照合し、不正値はdevice statusから公開completionの
  query／wait／deferred finalizeまでfail-closeする。途中監査でGPU runnerのexpected target未設定を検出し、exact target文字列を
  contextへ渡すよう修正した。

## 検証

- `cargo test --locked --offline -p sllm-core minimax_m3 --lib`: 33 PASS、3 ignored。
- fixed-revision official config／index、59 header、GGUF catalogの各ignored exact test: PASS。
- `cargo test --locked --offline -p sllm-server model_library::tests::reviewed_minimax_m3_is_visible_but_never_registered_as_production_ready --lib`: PASS。
- `cargo test --locked --offline -p sllm-hip minimax_m3 --lib`: 5 PASS。
- `cargo clippy --locked --offline -p sllm-core --lib --tests -- -D warnings`: PASS。core／hip-sys／hip focused clippyもPASS。
- Werror public-runtime host build／CTest: PASS。
- canonical V620 UUID `GPU-76a08c022586fed6` exact `gfx1030`: GPU oracle PASS。binary SHA-256
  `b14988e6916286c730720a49b997ec99fed052d5c8f0fba4cda916f619247edc`。
- canonical R9700 UUID `GPU-a8e9ddefa2d60f55` exact `gfx1201`: GPU oracle PASS。binary SHA-256
  `212bfcf6f9dd28d2773d01d0890edc9d6165566b8cdb064f55380fcddcd27bc7`。
- 両binaryは各exact targetだけを含むCode Object V6／wave32 artifactで、fallback 0、KFD process残留なし、VRAM baseline復帰を確認した。
- integration reviewはcorrectness／security blocker、release-evidence不足とも0件だった。証明範囲はmodel-free routing operator、
  metadata、header、GGUF dry-runまでであり、full-model resident／generation、MSA GPU、multimodal、MTP、性能を含まない。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase58-minimax-m3-foundation.md)
