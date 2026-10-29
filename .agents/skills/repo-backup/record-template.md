# Backup record template

Copy these sections, in this order, into the repository's backup record. Keep each section short;
replace superseded state rather than appending. Write in the repository's documentation language.

```markdown
# Git / DVC backup record

Updated: <YYYY-MM-DD HH:MM, timezone> — <one-line summary of the last operation>.
Procedure: <link to the repo-backup skill>.

## 1. Destinations

- Git remote / branch / last pushed commit (verified with `git ls-remote`).
- DVC remote name + URL; bundle path (fixed name) + `git bundle verify` result.
- Cloud: files on the mounted drive = backed up; cloud upload state = <pending / unchecked / confirmed>.

## 2. Tracked artifacts

| Manifest (`.dvc`) | Directory MD5 | Files | Logical bytes | Local workspace |
|---|---|---:|---:|---|
| path/to/runs.dvc | `<md5>.dir` | n | bytes | present / deleted (date) / partial |

One row per manifest touched in recent operations; older manifests may be summarised in one line.
Mark partial local workspaces: they must never be re-added.

## 3. Checks that passed (reusable while inputs are unchanged)

| Check | Scope | Result |
|---|---|---|
| `dvc push <targets>` | selected manifests | exit 0, output excerpt |
| `dvc status -c <targets>` | selected manifests | in sync |
| `git push` / bundle verify | branch / all refs | commit SHA, PASS |

State coverage precisely (all objects vs sample; size vs checksum).

## 4. Local cleanup

- Requested by owner on <date>: <scope as requested>.
- Deleted paths (exact list); free space before/after; retained data and why.
- Pending manual steps (e.g. VHDX compaction), or "none".

## 5. Next session

- How to push new artifacts: explicit targets, what to skip.
- How to restore: `dvc pull <target>.dvc` examples; space needed.
- What invalidates the checks above.
- Known quirks of this remote (permissions, slow phases, misleading status categories).
```
