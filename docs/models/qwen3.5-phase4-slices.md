# Qwen3.5 Phase 4 real-weight slice identities

Phase 4のG2対象として、Qwen3.5-2B/9BのRMSNorm、embedding、final outputから
次のrange identityを2026-08-11に固定した。raw sliceは作成・保存していない。

## 共通recipe

- sourceは各model lockで全fileを検証済みのcheckout外cacheとする。
- extractorは`ci/tools/hash_safetensors_slice.py`、SHA-256
  `38b3d68e959bbc1360e73e335bf71f701f7c6c998c1923bc0c8f87afa059d964`とする。
- extractor repository base commitは`0e2526d8e8efa38deed88929977339d71ea03057`、Phase 4
  integration worktree treeは`16282f9014186042580fc927e47750947216d694`とする。
- environmentはLinux `6.17.0-35-generic` x86_64、Python 3.12.3とする。
- ordered argumentsは各表のtensorに対する
  `--cache-root <exact resolved revision cache> --tensor <tensor> --offset 0 --length <size_bytes>`
  とする。cache path自体はartifact identityに含めず、lock fingerprintとresolved revisionを正とする。
- BF16 embedding/final outputは先頭3 rowを使用する。RMSNormはlayer 0 input norm全体を使用する。

## Qwen3.5-2B

- resolved revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- lock fingerprint: `sha256:304e19f8b8ef78bab1848a6cfb46ac619a8ca5c8fd052cac1c43fc3f4d6dcdb3`
- source shard: `model.safetensors-00001-of-00001.safetensors`
- safetensors header: 76,648 bytes

| role | tensor / shape | tensor data offsets | slice relative / absolute range | bytes | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| RMSNorm | `model.language_model.layers.0.input_layernorm.weight` `[2048]` | `[1017129088,1017133184]` | `[0,4096]` / `[1017205744,1017209840]` | 4,096 | `e3bfb63b03722ec637ed4ea2cc552d68a87e661bbba1e889f721858b34162877` |
| embedding | `model.language_model.embed_tokens.weight` `[248320,2048]` | `[10368,1017129088]` | `[0,12288]` / `[87024,99312]` | 12,288 | `68794b8bf247cd8dc544df6ac5a9bce5138af6726a13bfde9d3a9661c2f9943b` |
| final output | embeddingと同じtied alias | embeddingと同じ | embeddingと同じ | 12,288 | `68794b8bf247cd8dc544df6ac5a9bce5138af6726a13bfde9d3a9661c2f9943b` |

## Qwen3.5-9B

- resolved revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- lock fingerprint: `sha256:2d2bc642540e97d4681f8c66140e09f305f487476bb9fe238ca82a298febf893`

| role | tensor / shape | shard / header bytes | tensor data offsets | slice relative / absolute range | bytes | SHA-256 |
| --- | --- | --- | --- | --- | ---: | --- |
| RMSNorm | `model.language_model.layers.0.input_layernorm.weight` `[4096]` | `model.safetensors-00004-of-00004.safetensors` / 77,528 | `[15360,23552]` | `[0,8192]` / `[92896,101088]` | 8,192 | `7d9dae62cc87d982dc1fa1b476eb3ee105ea096e663d4c2e749a0723f344301a` |
| embedding | `model.language_model.embed_tokens.weight` `[248320,4096]` | `model.safetensors-00001-of-00004.safetensors` / 1,776 | `[2034237440,4068474880]` | `[0,24576]` / `[2034239224,2034263800]` | 24,576 | `a1cbad2483c06f9eb3d6d5016cc498ef9683b05f08e1c4ca0313f4cafcc40556` |
| final output | `lm_head.weight` `[248320,4096]` | `model.safetensors-00001-of-00004.safetensors` / 1,776 | `[0,2034237440]` | `[0,24576]` / `[1784,26360]` | 24,576 | `5058ed3afab9fd139e89cef633c0e1170087123653f7ec747aa431dbf5af08b3` |

9Bのembeddingとfinal outputは異なるtensor/range/hashであり、untied output projectionの
non-alias契約を実weightでも確認する。
