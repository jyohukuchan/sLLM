# 参照source固定マニフェスト

## 目的とスナップショット

- この文書は、sLLMが実装の比較・調査だけに使う外部推論engine sourceの取得元、固定revision、local checkout状態を記録する。参照sourceはsLLMの実装、配布物、正しさの証拠ではない。
- 観測日は **2026-08-02**。表の7件は、その日に公式GitHub releaseで `draft=false` かつ `prerelease=false` として観測できた対象のrevisionである。将来の「latest」を約束する記録ではない。
- version/tagは表示用の識別子であり、lock値は40桁の完全commit SHAである。SGLangだけは、commitを指すannotated tag objectも併記する。
- source treeはGit管理対象にしない。追跡するのはこのmanifestと調査記録だけであり、local `reference/` は既存の `.gitignore` の `/reference/` により引き続き無視・未追跡とする。

## 固定source

表の `release publication (UTC)` は公式GitHub Releaseの公開時刻であり、`commit/tag date(s) (UTC)` は固定したcommitとtagに対応するGitの時刻である。これらは、このmanifestを作成した観測日 **2026-08-02** とは別の事実として記録する。lightweight tagは指し先commitの時刻、SGLangのannotated tagはtag objectの時刻を示す。

| source | official origin | local path | release | release publication (UTC) | commit/tag date(s) (UTC) | lock revision |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `reference/llama.cpp` | `b10227` | 2026-08-02T09:43:15Z | commit/lightweight tag: 2026-08-02T09:13:20Z | `f5919bf458ef190468b5c329bb293f8a54a1e69c` |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `reference/vLLM` | `v0.26.0` | 2026-07-27T01:06:58Z | commit/lightweight tag: 2026-07-27T00:57:50Z | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `reference/SGLang` | `v0.5.16` | 2026-07-25T00:13:18Z | commit: 2026-07-24T20:25:42Z; annotated tag: 2026-07-24T20:27:30Z | commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`; tag object `d21f3c3a10606ba3c7bf43f981496da0a7d620cd` |
| TensorRT-LLM | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | `reference/TensorRT-LLM` | `v1.2.1` | 2026-04-20T11:51:33Z | commit/lightweight tag: 2026-04-16T15:13:33Z | `376f7e1bd8ed543f75014309e3fd4b237e9b0e73` |
| ROCm/ATOM | [ROCm/ATOM](https://github.com/ROCm/ATOM) | `reference/AMD-ATOM` | `v0.1.5` | 2026-06-22T18:03:57Z | commit/lightweight tag: 2026-06-22T16:24:59Z | `b0071c550ba3c99b1e9218debb91a6f81550da9a` |
| LMDeploy | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | `reference/LMDeploy` | `v0.15.0` | 2026-07-31T13:00:46Z | commit/lightweight tag: 2026-07-31T12:51:11Z | `f4b8140ba19cd823c541241cbb113cc32f854e6a` |
| KTransformers | [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | `reference/KTransformers` | `v0.6.4` | 2026-07-23T14:32:53Z | commit/lightweight tag: 2026-07-23T13:23:34Z | `924754a00bd8e5c6a2ad97929065c113f35782cf` |

## Checkoutとlicenseの事実

2026-08-02にlocal treeを確認した結果、7件すべてが `HEAD (no branch)` のdetached checkout、shallow repository、working tree cleanだった。KTransformers以外の6件はrecursive submodule statusが空である。KTransformersは4件のgitlinkを持ち、全て未初期化を示す `-` で、各submodule worktreeは空だった。トップレベルのlicenseは次のとおりである。

| source | local checkout | submodule | top-level license | 注意 |
| --- | --- | --- | --- | --- |
| llama.cpp | shallow / detached / clean | なし | MIT | upstreamのlicense noticeを保持すること |
| vLLM | shallow / detached / clean | なし | Apache-2.0 | ファイルごとのnoticeも確認すること |
| SGLang | shallow / detached / clean | なし | Apache-2.0 | annotated tag objectを別途検証すること |
| TensorRT-LLM | shallow / detached / clean | なし | Apache-2.0 | 同梱third-party部分は個別licenseがあり得る |
| ROCm/ATOM | shallow / detached / clean | なし | MIT | upstreamのcopyright noticeを保持すること |
| LMDeploy | shallow / detached / clean | status空 | Apache-2.0 | LFS pointerなし |
| KTransformers | shallow / detached / clean | gitlink 4件、全て未初期化 | Apache-2.0 | submodule worktreeは全て空 |

特殊な取得事実は次のとおりである。

- TensorRT-LLMはLFS smudgeを無効にして取得した。local treeにはLFS pointerが4,121件あり、pointerが示すpayloadは取得・記録していない。
- llama.cppの `models/` には vocabulary GGUFが19件あり、合計サイズは `77556152` bytesである。これは語彙fixtureであり、model weightではない。
- LMDeployはtracked file 1,646件、checkout全体 `15210661` apparent bytes、`.git` を除くworktree `12208451` bytesで、LFS pointerは0件だった。
- KTransformersはtracked file 1,415件、checkout全体 `124228926` apparent bytes、`.git` を除くworktree `77580448` bytesで、LFS pointerは0件だった。upstream treeのgitlinkは `third_party/custom_flashinfer`、`third_party/llama.cpp`、`third_party/pybind11`、`third_party/sglang` の4件であり、全て未初期化のまま保持する。
- licenseは参照・調査のために記録したものである。llama.cpp以外のengine sourceはreader-onlyで参照し、codeのcopy・adapt・portを行わない。llama.cppからの直接reuseだけは、`docs/provenance/README.md` に従って、対象ファイルのlicense、copyright、upstream URL、完全SHA、source/local path、blob ID、hash、`exact`/`adapted`/`ported`区分、変更内容、import commitを記録し、`THIRD_PARTY_NOTICES.md` とsource-file headerを含むnotice processを完了した場合に限り許可する。

## 再現取得コマンド

新しいworkspaceで実行する場合のコマンドを固定する。既存のlocal checkoutを上書きせず、対象pathが存在しないことを先に確認する。

```sh
mkdir -p reference
test ! -e reference/llama.cpp
git clone --no-recurse-submodules --depth 1 --branch b10227 https://github.com/ggml-org/llama.cpp.git reference/llama.cpp
git -C reference/llama.cpp checkout --detach f5919bf458ef190468b5c329bb293f8a54a1e69c

test ! -e reference/vLLM
git clone --no-recurse-submodules --depth 1 --branch v0.26.0 https://github.com/vllm-project/vllm.git reference/vLLM
git -C reference/vLLM checkout --detach 568afb3a13806beb53bb2e6bd518269357b237c0

test ! -e reference/SGLang
git clone --no-recurse-submodules --depth 1 --branch v0.5.16 https://github.com/sgl-project/sglang.git reference/SGLang
git -C reference/SGLang checkout --detach fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1

test ! -e reference/TensorRT-LLM
GIT_LFS_SKIP_SMUDGE=1 git clone --no-recurse-submodules --depth 1 --branch v1.2.1 https://github.com/NVIDIA/TensorRT-LLM.git reference/TensorRT-LLM
git -C reference/TensorRT-LLM checkout --detach 376f7e1bd8ed543f75014309e3fd4b237e9b0e73

test ! -e reference/AMD-ATOM
git clone --no-recurse-submodules --depth 1 --branch v0.1.5 https://github.com/ROCm/ATOM.git reference/AMD-ATOM
git -C reference/AMD-ATOM checkout --detach b0071c550ba3c99b1e9218debb91a6f81550da9a

test ! -e reference/LMDeploy
GIT_LFS_SKIP_SMUDGE=1 git clone --no-recurse-submodules --depth 1 --branch v0.15.0 https://github.com/InternLM/lmdeploy.git reference/LMDeploy
git -C reference/LMDeploy checkout --detach f4b8140ba19cd823c541241cbb113cc32f854e6a

test ! -e reference/KTransformers
GIT_LFS_SKIP_SMUDGE=1 git clone --no-recurse-submodules --depth 1 --branch v0.6.4 https://github.com/kvcache-ai/ktransformers.git reference/KTransformers
git -C reference/KTransformers checkout --detach 924754a00bd8e5c6a2ad97929065c113f35782cf
```

各clone後に、次の検証を行う。`reference/SGLang` ではannotated tag objectも検証対象にする。

```sh
test "$(git -C reference/llama.cpp rev-parse HEAD)" = f5919bf458ef190468b5c329bb293f8a54a1e69c
test "$(git -C reference/vLLM rev-parse HEAD)" = 568afb3a13806beb53bb2e6bd518269357b237c0
test "$(git -C reference/SGLang rev-parse HEAD)" = fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1
test "$(git -C reference/SGLang rev-parse refs/tags/v0.5.16^{tag})" = d21f3c3a10606ba3c7bf43f981496da0a7d620cd
test "$(git -C reference/TensorRT-LLM rev-parse HEAD)" = 376f7e1bd8ed543f75014309e3fd4b237e9b0e73
test "$(git -C reference/AMD-ATOM rev-parse HEAD)" = b0071c550ba3c99b1e9218debb91a6f81550da9a
test "$(git -C reference/LMDeploy rev-parse HEAD)" = f4b8140ba19cd823c541241cbb113cc32f854e6a
test "$(git -C reference/KTransformers rev-parse HEAD)" = 924754a00bd8e5c6a2ad97929065c113f35782cf

for dir in reference/llama.cpp reference/vLLM reference/SGLang reference/TensorRT-LLM reference/AMD-ATOM reference/LMDeploy; do
  test "$(git -C "$dir" rev-parse --is-shallow-repository)" = true
  test "$(git -C "$dir" rev-parse --abbrev-ref HEAD)" = HEAD
  test -z "$(git -C "$dir" status --porcelain=v1)"
  test -z "$(git -C "$dir" submodule status --recursive)"
done

test "$(git -C reference/KTransformers rev-parse --is-shallow-repository)" = true
test "$(git -C reference/KTransformers rev-parse --abbrev-ref HEAD)" = HEAD
test -z "$(git -C reference/KTransformers status --porcelain=v1)"

kt_status="$(git -C reference/KTransformers submodule status --recursive)"
test "$(printf '%s\n' "$kt_status" | sed '/^$/d' | wc -l)" -eq 4
test -z "$(printf '%s\n' "$kt_status" | awk '$1 !~ /^-/ { print }')"
test "$(printf '%s\n' "$kt_status" | awk '{ print $2 }' | LC_ALL=C sort)" = "$(printf '%s\n' third_party/custom_flashinfer third_party/llama.cpp third_party/pybind11 third_party/sglang | LC_ALL=C sort)"
for path in third_party/custom_flashinfer third_party/llama.cpp third_party/pybind11 third_party/sglang; do
  test -d "reference/KTransformers/$path"
  test -z "$(find "reference/KTransformers/$path" -mindepth 1 -maxdepth 1 -print -quit)"
done

test "$(git -C reference/LMDeploy ls-files | wc -l)" -eq 1646
test "$(git -C reference/KTransformers ls-files | wc -l)" -eq 1415
test "$(git -C reference/LMDeploy grep -Il '^version https://git-lfs.github.com/spec/v1$' HEAD -- | wc -l)" -eq 0
test "$(git -C reference/KTransformers grep -Il '^version https://git-lfs.github.com/spec/v1$' HEAD -- | wc -l)" -eq 0
```

以上のコマンドでversion/tagではなく完全SHA、detached、shallow、clean、6件のrecursive submodule status空、KTransformersの未初期化gitlink 4件と空のsubmodule worktreeを再確認できる。LFS payloadの取得、model weightの追加、`reference/` のtrackはこのmanifestの再現手順に含めない。

## 関連文書

- 固定sourceの参照範囲と今回の採用判断は [推論engine参照](inference-engines.md) を参照する。
- 取得作業の完了記録は [取得計画](../plans/archive/2026/08/1-10/reference-source-acquisition.md) と [取得履歴](../history/2026/08/1-10/reference-source-acquisition.md) を参照する。
- 今回の採用作業は [採用計画](../plans/archive/2026/08/1-10/reference-source-adoption.md) と [採用履歴](../history/2026/08/1-10/reference-source-adoption.md) を参照する。
