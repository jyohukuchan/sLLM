# Recovery verification

- manifest SHA-256 before/after: `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`
- `ullm-openai.service`: `ActiveState=active`, `NRestarts=0`
- OpenWebUI bridge completion: HTTP 200, one-token content `rest`

The first post-start eight-token bridge probe reported `container_transport`.
The subsequent one-token bridge probe succeeded; this is the response evidence
used for recovery confirmation.
