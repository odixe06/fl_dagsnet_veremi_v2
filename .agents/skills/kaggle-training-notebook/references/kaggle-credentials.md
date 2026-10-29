# Saved Kaggle credentials on this host

Credentials live outside Git in `~/.kaggle/` (directory 0700, credential files 0600).

- `accounts/<username>.mcp-token`: owner-supplied KGAT API token, also usable by the CLI/SDK.
- `accounts/<username>.credentials.json`: optional OAuth snapshot created after browser login.
- `credentials.json`: active CLI OAuth login; do not replace it with a KGAT token.
- `account-roster.json`: private username/email mapping; never copy email/token values into skills or docs.

Use env `nckh`. Run each account-scoped command through the repository's existing `kaggle_as.py`;
this preserves the active global login. `KAGGLE_API_TOKEN` may contain the saved token FILE PATH,
which the installed SDK reads. Do not export it globally or put the raw value on a command line.

A token-only account can run account-scoped CLI commands without browser login. `kaggle_account.py quota/health/plan` supports both token-only and OAuth accounts.
Global `use/ensure` still requires OAuth snapshots; prefer `kaggle_as.py` for process-scoped commands. The owner will perform CLI browser login
after environment setup. `list` inventories both token-only and OAuth accounts.

For a new OAuth account, use `kaggle_account.py add-account <username>`: save the current snapshot
before browser login, then save the new snapshot. The owner completes browser authorization.
Never invoke forced browser login unattended or silently overwrite another account's refresh token.

To verify a token-only identity, read its file in memory, build a Kaggle client with the token,
and call `security.oauth_client.introspect_token` with an `IntrospectTokenRequest` whose
`token` field is populated. Require `active=True` and a username matching the filename.
Suppress credential values and exception bodies; report only account, state and error class.
Quota/public metadata alone is not unique identity proof.

CLI is the configured channel for this checkout. Kaggle MCP, a Codex header helper,
`sync_kaggle_mcp.py`, `probe_kaggle_mcp.py` and Claude SessionStart hooks are not installed here;
do not run missing helpers or claim reconnect/authentication from historical notes.
If MCP is explicitly configured later, verify it independently using an account-scoped native read.

Read [multi-account.md](multi-account.md) for the OAuth workflow. Current project/account policy
and local setup take precedence over dated observations in that reference.
