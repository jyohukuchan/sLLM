# Existing `AQ4_0` P3 manifest revalidation

After commit `49406705` extended the v2 execution contract to admit the
shape-closed `AQ4_0` grouped candidate, the unchanged active P3 manifest was
validated with `python3 tools/validate-served-model.py --manifest
/etc/ullm/served-models/active.json`.

`active-p3-validation.json` is the exact non-secret validator output.  It
confirms the active P3 SHA-256 remains
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`, has
`worker.execution: null`, and still binds the same gfx1201
`rdna4_aq4_resident` worker.  The extension did not add any selector to the
existing production manifest.
