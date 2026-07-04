# 12h golden layer validation plan

## Context

君から、推論エンジン土台作りを始めて約24時間経った現在の進捗をもとに、次の`/goal`で進められる12時間程度のタスクに絞った計画を求められた。

現在の土台は、Qwen3.5-9B由来の`.ullm.d` packageからselected decoder layersを読み、CPU/R9700/V620でself-attention + MLP blockのmulti-request、multi-layer smokeを通せる段階にある。ただし、token idからのend-to-end generationや参照実装とのlayer/logits一致はまだない。

## Decision

12時間タスクは、full prompt generationではなく、golden tensor fixtureを使った1 layer単位の参照比較に絞る。

作成した計画:

- `docs/plans/12h-golden-layer-validation-plan-v0.1.md`

追加した用語:

- `golden tensor fixture`
- `package layer golden smoke`

## Next Action

次の`/goal`では、計画文書の`Next Goal Candidate`を目的にして進めるのがよい。

