---
name: sync-remote
description: Copy code and data to the remote GPU machine. Use before a remote run.
disable-model-invocation: true
---

1. Check that no run uses the files: `ssh maidai 'pgrep -af "scripts/|quant_suite|baselines"'`. If a run is active, stop and ask the user.
2. Show the changes: `make sync-remote-dry`.
3. Copy: `make sync-remote`. This copies `compress/`, `scripts/`, and `data/`. It never copies `outputs/`.
4. Check the copy: `ssh maidai 'cd ~/pretty_duck_gate && ls compress scripts | head'`.
5. If the host name does not resolve, tell the user. Do not guess another address.
