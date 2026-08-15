# Phase 16F first-class FP4 model input履歴

## 2026-08-16: 詳細計画作成

- ユーザー決定に従い、提供元NVFP4 PTQ/QATとMXFP4/MXFP8 QAT/native modelをBF16/FP8と同じ操作で扱う
  official model input phaseを追加した。内部evidence分類を起動mode、許可flag、通常警告へ変換しない。
- primary full-modelを既存cache/lockとGemma adapterを再利用できる`unsloth/gemma-4-12b-it-NVFP4` revision
  `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`とした。公開mixed recipeのW4A4 MLP、W8A8 attention、FP8 KV、
  BF16/ignoreを忠実に実行するため、Phase 16 KV量子化の後へ配置した。
- NVIDIA `Gemma-4-31B-IT-NVFP4` revision `4135a98a9b728a548947683219633b25682223ac`は4 shard合計
  `32,633,477,808` byteでR9700 32 GiBへworkspace込みで収まらないため、secondary schema/model-lock/reference targetとした。
- OCP MX v1.0と`moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`をMXFP4/MXFP8 contractへ固定した。
  Kimi full modelは未実装MoE/architectureかつ2.8T級のため、encoding/import boundaryだけを本Phaseで完成させ、Phase 18以降へ渡す。
- safetensors/compressed-tensorsと将来GGUFが同じcontainer-neutral encoding/recipe descriptorへlowerする計画、same-artifact
  reference、task oracle、AMD operator/full-model、performance/UXの受入条件を固定した。本時点ではsource実装やmodel downloadを行っていない。

## 2026-08-16: Phase 16F完了

- provider artifact lockとfail-closed importerを実装した。primaryはrevision
  `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`、`model.safetensors` SHA-256
  `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`で、7 frontend/model file、
  1,389 physical tensor、677 logical tensorを検証する。logical inventoryはMLP NVFP4 144、attention FP8 184、static FP8 KV
  48 layer、recipe digest `sha256:e64f38576cffd36fac5f55d5e7c47846afdc59ef8ef5aec24b66f090aa8522e2`である。
- `QuantizedTensorEncoding`とscale-plane/role/source-range/mixed-recipe descriptorをcontainerから分離した。Unsloth
  compressed-tensorsは直接verified uploadし、将来GGUFは同じdescriptorへlowerする。OCP MXFP4 E2M1 block-32/E8M0と
  MXFP8をNVFP4から分け、全code、special scale、odd tailのtiny oracleを追加した。
- W4A4はBF16 inputを各linear直前にdynamic block-16 NVFP4へ量子化し、packed activation/weightを直接consumeする
  2-dispatch providerとした。M `1/3/7/17/32/33`とK/N `15/16/17/31/32/33`を組み合わせた12 caseはV620/R9700で
  max relative error `0.00381`未満、kernel ID 11、fallbackなし、cleanup 0をPASSした。completion timingは2 dispatch合計で、
  stage別GPU eventは未取得のためstage別性能値を主張しない。
- artifactのstatic BF16 decode scaleをopaque KV stateへ渡すstatic FP8 encodingを追加した。Gemma sliding
  `[Hkv=8,D=256,GQA=2]`とfull `[Hkv=1,D=512,GQA=16]`へ共通causal attentionを広げた。既存17-case
  static-FP8 KV oracleは両targetで全数PASSした。
- primary full-modelは両targetでresident `9,201,189,600` byte、peak accounted `9,221,491,952` byte、958 node、
  8 transition、8,048 submission、10,672 kernel dispatchを実行した。両targetのtoken列は`[532; 8]`、fallbackなし、
  cleanup 0だった。R9700はupload/prefill/decodeが10.458 s/211 ms/106--108 ms、V620は12.096 s/677 ms/
  628--651 msである。これはsame-artifact NVIDIA referenceを実行していないAMD experimental evidenceであり、reference
  correctness PASSにはしない。
- CLIは同じ`--cache MODEL_DIR`で`verify-model`、Unicode tokenize、gfx1201 generateをPASSした。OpenAI serverは同じ引数で
  non-streamとSSEを実行し、2 request後のshutdownで全current byteとcleanup/quarantineが0へ戻った。専用opt-in、警告、
  confirmationは追加していない。multi-GPU hostは既存contractどおり`HIP_VISIBLE_DEVICES=2`でR9700だけを可視化し論理0を使った。
- secondary metadata lockとしてNVIDIA Gemma 4 31B NVFP4とKimi K3を追加した。31Bはcapacity理由でschema/reference-only、
  KimiはMX encoding/import handoffまでとし、MoE/architectureをPhase 18へ渡した。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase16f-first-class-fp4-model-input.md)
