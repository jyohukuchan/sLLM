# SQ8_0 CDNA3 MI300X レンタル手順リハーサル

## 前回の要点

- 初回 MI300X 借用では `.cargo/config.toml` の `clang` と mold が provider に
  なく、手作業で config を退避して build を通した。
- B control の hipBLAS row-major view は CPU oracle で修正済みだが、修正後の
  gfx942 A′/B physical confirmation は未実施だった。
- runner は構文・引数レベルでしか確認されておらず、冪等性と中断再開には実行証跡が
  なかった。

## 今回の変更点

- runner を実行して CPU (79 s)、generic `SQ8_0` HIPRTC 27/27 (18 s)、gfx942
  feature build (54 s)、ISA audit (6 s、912 MFMA) を pass させた。gfx942 がない
  normal preflight と physical は fail-closed で止まることも実行確認した。
- `cc`/`rustc`/ROCm tools を preflight の必須条件にし、rental linker が存在しない
  場合を build 前に止めるようにした。`clang`/mold/`rustup` は optional として
  記録し、runner はそれらを必要としないことを明示した。
- `--rehearsal-no-gfx942` を追加した。local では offline stages を通し、physical
  を GPU binary 起動前に expected failure として残す。P0 pass にはならない。
- 同一 results directory の rerun で完了 stage が skip されること、CPU compile
  中の SIGTERM 後に `preflight.done` から CPU を再実行して最後まで続くことを確認
  した。
- rental runner の `cc`/空 Rust flags override と、未変更の local `clang`+mold
  config の双方を clean build で確認した。

## 次の行動

- GPU lease の前に exact clean checkout、`cargo fetch --locked`、
  `cargo fetch --locked --offline`、provider image の `cargo`/`rustc`/`cc`/ROCm
  preflight を済ませる。
- lease 中は normal mode の P0 order を守り、A′/B physical まで通す。失敗時は
  source を変えず同一条件で 1 回だけ再現し、logs/timings/ISA evidence を回収する。
- model、container、full-model、profiler、occupancy、external engine、hand-written
  A は P0 の時間を侵食させない。
