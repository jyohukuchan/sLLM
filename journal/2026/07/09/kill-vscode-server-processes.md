# Kill vscode-server processes

- Date: 2026-07-09
- Work: Listed processes whose command line matched `vscode-server`, then sent `TERM`.
- Fallback: Sent `KILL` to any remaining matched processes after a short wait.
- Verification: `pgrep -af '[v]scode-server'` returned no remaining processes.
