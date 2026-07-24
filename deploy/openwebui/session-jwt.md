# OpenWebUI session JWT provisioning

This procedure provisions the frontend session credential used by the
OpenWebUI browser gates. It does not call OpenWebUI, perform a login, change an
account, or start a campaign.

## Pinned implementation contract

This deployment uses OpenWebUI `0.9.4` at upstream commit
`f51d2b026f1b0e7283b15f093412be8b67d24770`. The derived-image patch changes
only provider stream error handling, not authentication.

OpenWebUI's session implementation has the following contract:

- `WEBUI_SECRET_KEY` is the HMAC key and `HS256` is the only accepted
  algorithm.
- Login passes `{"id": user.id}` to `create_token`. The helper adds a UUIDv4
  `jti`, an integer UTC `iat`, and, when expiry is enabled, an integer UTC
  `exp`. It does not add `email`, `role`, `sub`, `iss`, `aud`, or `nbf`.
- Authentication verifies the signature and time claims, requires `id`, and
  looks that ID up in the current user database. Authorization uses the
  database role rather than a role claim.
- `JWT_EXPIRES_IN` is a persistent setting whose default is `4w`. The current
  container has no environment override or persisted `auth.jwt_expiry`, so its
  effective login lifetime is four weeks.
- The current container has no Redis configuration. Per-token `jti`
  revocation is therefore inactive; expiry is the effective revocation
  boundary for a directly minted token.

The image starts with `bash start.sh`. Its
`WEBUI_SECRET_KEY=$(cat "$KEY_FILE")` command substitution removes all trailing
LF bytes before exporting the key. The provisioning tool deliberately applies
that same normalization. Using the 65-byte mounted file, including its final
LF, as the HMAC key would produce an invalid signature.

The relevant upstream source is:

- [`create_token` and `decode_token`](https://github.com/open-webui/open-webui/blob/f51d2b026f1b0e7283b15f093412be8b67d24770/backend/open_webui/utils/auth.py#L200-L219)
- [`get_current_user`](https://github.com/open-webui/open-webui/blob/f51d2b026f1b0e7283b15f093412be8b67d24770/backend/open_webui/utils/auth.py#L297-L405)
- [`create_session_response`](https://github.com/open-webui/open-webui/blob/f51d2b026f1b0e7283b15f093412be8b67d24770/backend/open_webui/routers/auths.py#L100-L149)
- [`JWT_EXPIRES_IN`](https://github.com/open-webui/open-webui/blob/f51d2b026f1b0e7283b15f093412be8b67d24770/backend/open_webui/config.py#L370-L398)
- [`parse_duration`](https://github.com/open-webui/open-webui/blob/f51d2b026f1b0e7283b15f093412be8b67d24770/backend/open_webui/utils/misc.py#L727-L755)

## Selected method

Use `tools/openwebui-session-jwt.py` to mint an HS256 token directly from the
existing signing key and the exactly-one existing administrator in the
OpenWebUI SQLite database. A signed JWT has no separate server-side session
row. OpenWebUI therefore cannot distinguish this token from one returned by
its login helper when the signature, claims, time bounds, and database user
binding are valid.

This method avoids handling or retaining the administrator password and sends
no credential over HTTP. The tool:

- stable-reads bounded, single-link inputs with `O_NOFOLLOW`;
- never prints the signing key, token, user ID, or their hashes;
- rejects an ambiguous administrator selection;
- rejects no-expiry (`0` or `-1`) tokens;
- writes only to a metadata-checked private directory;
- refuses an existing output unless `--replace` is explicit;
- verifies the HS256 signature, exact claim set, UUIDv4 `jti`, time bounds, and
  current database role offline.

Use a short explicit lifetime rather than the normal four-week login lifetime.
The SQ8 full campaign's fixed maximum is six hours, so `24h` leaves operational
margin while bounding the non-revocable credential.

## Production file

The sealed final-campaign contract is:

```text
directory: /run/ullm-campaign-secrets
           uid=0 gid=1000 mode=0750
token:     /run/ullm-campaign-secrets/openwebui-session.jwt
           uid=0 gid=1000 mode=0640 nlink=1
```

This differs from a standalone caller-owned `0600` browser-gate file. Do not
substitute `/etc/ullm/openai-api-key` or
`/etc/ullm/openwebui-secret-key` for the session-token path.
Because `/run` is a tmpfs on this host, a reboot removes both the directory and
token; recreate them just in time with this procedure.

Create the private runtime directory, then mint a 24-hour token. These commands
do not restart or reconfigure OpenWebUI:

```bash
sudo install -d -m 0750 -o root -g homelab1 \
  /run/ullm-campaign-secrets

sudo tools/openwebui-session-jwt.py mint \
  --database /var/lib/docker/volumes/open-webui/_data/webui.db \
  --secret-key-file /etc/ullm/openwebui-secret-key \
  --output /run/ullm-campaign-secrets/openwebui-session.jwt \
  --expires-in 24h
```

The command emits only a non-secret JSON summary. If the output already exists,
inspect and verify it first. A deliberate renewal uses the same command with
`--replace`; the tool refuses to replace a target whose owner, group, mode, or
link count differs from the production contract.

Verify the result offline with enough validity for the next campaign:

```bash
sudo tools/openwebui-session-jwt.py verify \
  --database /var/lib/docker/volumes/open-webui/_data/webui.db \
  --secret-key-file /etc/ullm/openwebui-secret-key \
  --token-file /run/ullm-campaign-secrets/openwebui-session.jwt \
  --minimum-validity 7h
```

The successful summary reports `HS256`, the claim names
`exp,iat,id,jti`, integer issue/expiry metadata, remaining validity, a valid
signature, and a current `admin` database binding. It never reports the JWT or
the administrator ID.

Before a later campaign, repeat `verify` with a minimum validity longer than
the complete authorized window. If it fails or too little time remains, mint a
new short-lived token with explicit `--replace` and verify it again. Removing
the file after the campaign prevents later reads but, without Redis, does not
revoke copies made while it was valid.

## Login fallback

The browser-login flow is not needed for this pinned deployment. Do not search
for, create, or reset administrator credentials merely to obtain a token. If a
future OpenWebUI version adds server-side session state, changes the key or
algorithm, adds mandatory claims, or enables revocation checks that reject
directly minted tokens, stop and reassess the pinned source before using the
browser automation fallback. That fallback requires the real administrator
email and password supplied through the system's private secret mechanism; it
must not invent or change them.
