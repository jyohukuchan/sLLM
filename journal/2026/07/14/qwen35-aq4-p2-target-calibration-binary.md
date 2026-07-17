# Qwen3.5 AQ4 P2 target calibration binary

## 前回の要点

engine coreへprepared token直後のchunk observer、source greedy teacher forcing、generation epoch/full-logits validityを追加した。direct-top1、prewarm、未生成、別stepのstale logitsは観測前に拒否される。

## 今回の変更点

- 新binary `ullm-aq4-p2-calibration` を追加した。
- P2 case/model/served/package/content/worker/device/preflightとsource full calibrationをstrict bindした。
- hidden/logitsをf32le sidecarへchunk writeし、compare互換vector rowsとhash-bound execution rowsを分離した。
- source greedy列をcanonical replay hashでteacher forcingし、predicted/committed/divergence/epoch/lifecycleを保存する。
- atomic non-overwrite publishとimmutable blocked artifactを追加した。performance timingとraw-v2 evidenceには使用しない。
- CPU mock test 11件は通過した。OOM/identity/direct-top1/nonfiniteのfail-close分類とmanifest-only blocked rootも含む。GPU/liveは実行していない。
- `CARGO_BUILD_JOBS=1`でengine check、calibration bin build/test、lib testを実行した。binは11/11、libは717 passed/1 ignoredである。
- workspace全体の`cargo fmt --all -- --check`は、担当外の並行差分`ullm-aq4-p2-full-model.rs`と`ullm-aq4-p2-path-oracle.rs`が未formatのため停止した。calibration bin自体は直接rustfmt済みである。
- commitは`676db88 Add AQ4 P2 calibration capture binary`である。
- 独立QAのfollow-upとして、全入力fileへ`st_nlink=1`、`O_NOFOLLOW` fd固定、read前後のfd/path `dev/ino/size/mode/mtime/ctime/nlink`照合、bounded readを追加した。
- hardlink、bounded overflow、same-size rewriteとmtime復元、rename replacement、appendを独立負例で拒否した。source rowsはnewline終端も必須にした。
- outputはlexicalな`.`/`..`/重複separatorを拒否し、既存の非symlink parentをcanonicalizeした非存在leafへ限定した。成功・blockedの両経路でsource artifact/checkpoint/tokenizer rootとのcanonical overlapを拒否し、`RENAME_NOREPLACE`のatomic publishは維持した。
- follow-up後のcalibration bin testは15/15、engine lib testは717 passed/1 ignored、bin check、owned rustfmt check、owned diff checkは通過した。GPU/liveは実行していない。follow-up commitは`8d18d72 Harden AQ4 P2 calibration file inputs`である。
- 独立re-QAのcross-open follow-upでは、source SHA256SUMS検証時のmanifest/rows/hidden/logits各identityとdigestを保持し、実利用fdを開いた直後に再照合して同じfdをparse/scanへ渡した。sum検証後からreopen前のrename replacementとsame-size rewriteを拒否する。
- package treeはdirectory列挙時のfile identityをhash fdへ照合し、aggregate byte countを同じpinned hash fdのsizeから加算するようにした。hash後のpath metadata再読は廃止した。
- thread-localの決定的hook負例でsource 4 payloadのcross-open replacement、source rowsのsame-size rewriteとmtime復元、同一内容package replacementを拒否した。bin testは18/18、libは717 passed/1 ignored、bin check、owned rustfmt/diff checkは通過した。GPU/liveは実行していない。cross-open follow-up commitは`2edc6b6 Close calibration cross-open input races`である。

## 次の行動

親agentへcross-open follow-up commitと検証結果を引き渡す。全体fmtは上記2 fileのownerがformatした後に再実行する。
