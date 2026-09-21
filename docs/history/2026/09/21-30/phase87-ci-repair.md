# Phase 87 CI修復

2026-09-21完了。未公開のPhase 87変更について、CIと同じhost入口とHIP直接コンパイル経路を確認し、
Rustの警告とCI側の登録漏れを修正した。commit／pushと公開CIの再実行は行っていない。

## 修正

- Rust: 不要な式・テスト用確保を整理し、同等のderive／初期化へ変更した。
  graphが保持するownerやcleanupの再試行・quarantine構造は維持した。
  寿命保持だけのfield、固定ABIに対応する多引数関数、確保を増やさず所有権を返すcleanup型などは、
  個別に意図を示してlintを適用除外とした。workspace全体の警告検査は維持する。
- HIP直接コンパイル: `decode_control_kernel.hip.cpp`と二つの内部headerを登録し、入力集合を120→123へ同期。
  generic public-runtimeとRMSNorm H3のcommand／source hash、関連schemaを更新した。
- ELF検査: 新しいkernel、private C bridge、attention templateのdevice stub、
  pinned memory用の`hipHostMalloc`／`hipHostFree`を明示的な許可リストへ追加した。
  public C ABI、未知のsymbol・stubを拒否する検査は維持した。
- CI回帰テスト: build.rsのliteral配列による有限のsource登録loopを認識し、
  新規sourceの登録欠落・重複も従来どおり検出するようにした。

数値kernelの実装と対応GPU／toolchain方針は変更していない。

## 検証

| 検証 | 結果 |
| --- | --- |
| H0: format、Clippy、MSRV、manifest/schema、CI自身のテスト等 | 628件PASS |
| H1: workspace build/test、host契約等 | 1,642件PASS |
| H2: CPUの小さな数値oracle | 38件PASS |
| public HIP直接compile/link: gfx1030／gfx1201 | 各5 commandが成功 |
| 上記ELFのhost symbol、bundle、device target／COv6／wave32検査 | 両target PASS |

host検証は固定Python 3.12.10と`ci/requirements-host.txt`を使い、ネットワークを隔離した
既存runnerの`--allow-dirty-local`で実行した。Rustは開発pin 1.97.1とMSRV 1.85.0を確認した。
H1の75件、H2の9件は対象外として非選択であり、選択済みtestのskipは全rowで0件。

HIP検証には更新後のCI commandそのものを使った。ローカルでのcompile-only確認であり、
公式固定containerでの公開CI証拠やGPU実行成功へ読み替えない。生成ELFは実行していない。
通常のGPUベンチマークは再実行していない。

[結果・証拠digest](phase87-ci-repair-results.json)。raw logとbinaryは
`.local-artifacts/phase87/ci-repair/`へ保存した。

計画: [CI修復計画](../../../../plans/archive/2026/09/21-30/phase87-ci-repair.md)。
