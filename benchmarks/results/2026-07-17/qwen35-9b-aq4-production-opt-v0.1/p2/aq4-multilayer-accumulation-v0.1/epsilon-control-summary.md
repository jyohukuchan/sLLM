# Post-norm epsilon diagnostic control

Both runs use the identical CPU fixture and AQ4 package.  The runtime-default probe uses post-norm epsilon `1e-5`; the diagnostic-only control passes `1e-6`, matching the BF16 source.  The production runtime configuration is not modified.

| stage | AQ4 1e-5 relative L2 | AQ4 1e-6 control relative L2 | control - default | relative change |
| --- | ---: | ---: | ---: | ---: |
| post_norm | 0.178437633 | 0.178529637 | +0.000092004 | +0.0516% |
| mlp_gate_projection | 0.156694713 | 0.156727562 | +0.000032848 | +0.0210% |
| mlp_up_projection | 0.171037744 | 0.171103018 | +0.000065274 | +0.0382% |
| mlp_gate_silu | 0.152804856 | 0.152826803 | +0.000021946 | +0.0144% |
| mlp_activation | 0.124305173 | 0.123656226 | -0.000648947 | -0.5221% |
| mlp_output | 0.104565046 | 0.103788976 | -0.000776070 | -0.7422% |
| layer_output | 0.042451384 | 0.042349396 | -0.000101987 | -0.2402% |
| diagnostic_lm_head_readout_logits | 0.026798801 | 0.026853115 | +0.000054314 | +0.2027% |

At the layer output, aligning epsilon reduces source-relative L2 by `0.000101987` and reduces the squared source-relative error by `0.4799%`.  The post-norm stage itself is slightly worse (`+0.0516%`), then later MLP operations make the tiny final effect non-monotonic.  This is an epsilon-control result, not a direct full-tensor subtraction between two retained AQ4 outputs; the full tensors were deliberately discarded after streaming.

For scale, a deliberately generous linear repetition of the layer-output L2 reduction across 32 layers is `32 * 0.000101987 = 0.003263596`, only `0.5307%` of the observed final-model relative L2 `0.615`.  Thus the epsilon mismatch is negligible for the Phase 2 H8 conclusion and is not a basis for a production fix in this scoped diagnostic.

Raw identity-bound evidence: [runtime default comparison](epsilon-control/runtime-default-compare/comparison.json), [source-epsilon control comparison](epsilon-control/source-epsilon-compare/comparison.json), [runtime default AQ4 report](epsilon-control/runtime-default-aq4/aq4-report.json), and [source-epsilon AQ4 report](epsilon-control/source-epsilon-aq4/aq4-report.json).
