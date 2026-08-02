# 推論engine CI・test exact-revision調査

## 位置付け

- 2026-08-02に、[参照source固定マニフェスト](source-lock.md)の7件を完全commit SHAのlocal treeで調査した。
- この文書はCI・test設計上の参考点を記録する。参照engineの実装やtest結果を、uLLMの対応実績、正しさ、性能、互換性の証拠にはしない。
- vLLM等のcodeはreader-onlyで参照し、uLLMへのcopy、adapt、portは行っていない。

## 固定revisionと判断

| source | 完全commit SHA | 採用する設計要点 | 採用しない、または補う点 | 主な一次資料 |
| --- | --- | --- | --- | --- |
| llama.cpp | `f5919bf458ef190468b5c329bb293f8a54a1e69c` | 軽量CTest分類、HIP compile-only、commit・backend・modelを持つ性能比較 | model downloadをhost必須testへ入れない。無効化済みbenchmarkをactive gateと扱わない | `reference/llama.cpp/.github/workflows/build-cpu.yml:32-116`、`reference/llama.cpp/.github/workflows/hip-quality-check.yml:39-86`、`reference/llama.cpp/ci/run.sh:207-276` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` | CPU small、GPU機種別、large modelの分離、明示shard、ROCm preflight、artifact検証 | pytest未収集、`soft_fail`、暗黙skipをrequired成功へ変換しない | `reference/vLLM/.buildkite/hardware_tests/cpu.yaml:4-60`、`reference/vLLM/.buildkite/hardware_tests/amd.yaml:1-84`、`reference/vLLM/.buildkite/scripts/hardware_ci/run-amd-test.sh:435-486` |
| SGLang | `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` | testの明示登録、A/B/C段階、runner別matrix、推定時間による決定的sharding、timing artifact | live外部統計、`continue-on-error`、skipped jobを通す集約、root/強権限containerを採用しない | `reference/SGLang/python/sglang/test/ci/ci_register.py:32-85,353-389`、`reference/SGLang/.github/workflows/pr-test.yml:251-596` |
| TensorRT-LLM | `376f7e1bd8ed543f75014309e3fd4b237e9b0e73` | GPU/backend/stageの明示matrix、isolated test、build artifact再利用、baseline比較 | 過去PASSの再利用、waiveによる暗黙免除、可変外部artifact、debug SSHを採用しない | `reference/TensorRT-LLM/docs/source/developer-guide/ci-overview.md:20-50,89-106`、`reference/TensorRT-LLM/jenkins/L0_Test.groovy:227-350,2543-2562` |
| ROCm/ATOM | `b0071c550ba3c99b1e9218debb91a6f81550da9a` | 全visible GPU allocation、ROCm/GPU metadata、hang/fault検出、TTFT/TPOT等の性能記録 | exact target照合不足、missing artifactの許容、PRからのprivileged runner実行を採用しない | `reference/AMD-ATOM/.github/scripts/gpu_preflight_check.sh:80-115`、`reference/AMD-ATOM/scripts/wait_infer_drain.sh:52-57,206-226` |
| LMDeploy | `f4b8140ba19cd823c541241cbb113cc32f854e6a` | kernel単位test、seed、独立reference、marker、GPU数別分類 | CUDA/PyTorch testをuLLMのNumPy/HIP証拠にしない。長時間job、`continue-on-error`、外部report依存を採用しない | `reference/LMDeploy/tests/pytorch/kernel/test_paged_attention.py:119-153,234-316`、`reference/LMDeploy/.github/workflows/unit_test.yml:38-64` |
| KTransformers | `924754a00bd8e5c6a2ad97929065c113f35782cf` | hardware別登録、per-file timeout、warmup後のmetric収集 | AMD/CUDA placeholder、CPU/GPU混在、未収集・依存欠落のfail-open、artifact不在を採用しない | `reference/KTransformers/test/run_suite.py:9-45`、`reference/KTransformers/test/ci/ci_utils.py:99-150`、`reference/KTransformers/.github/workflows/kt-kernel-tests.yml:47-104` |

KTransformersの4 submoduleは未初期化のままとし、調査対象に含めていない。

## uLLMへ反映する共通方針

- 採用する: host/GPU/model/performanceの段階化、versionedな明示登録、決定的sharding、per-test timeout、GPU preflight、immutable artifactの再利用、isolated test、warmupとmetric記録。
- 採用しない: 暗黙skip、0件収集の成功、required testの`continue-on-error`またはsoft-fail、可変tag・外部live統計・mutable artifact/modelへの依存、root/privileged runner、runner labelだけを根拠にした互換性判定。
- 参照engineにもuLLMが必要とする完全なfail-closed result schema、exact AMD tuple、model lockとの暗号学的結合が揃っているとは限らないため、[uLLMのCI・テスト方針](../plans/active/2026/08/1-10/ci-test-strategy.md)で独自に補う。
