---
name: repo-backup
description: Back up Git history and DVC artifacts to the configured remotes, restore selected artifacts, and reclaim local disk after a verified backup. Use when the owner asks for a backup, artifact or Git push, DVC restore/pull, or disk cleanup (e.g. "backup", "đẩy DVC", "push git", "khôi phục dữ liệu", "dọn ổ đĩa"). A backup request never authorizes deletion.
---

# Repository backup

Back up code/history to the Git remote and a Git bundle; back up heavy artifacts to the configured DVC remote. Keep this skill reusable: read repository paths, accounts, decisions, and backup history from repository configuration and operations documents.

## This project's backup destination

The owner selected **OneDrive for future MobiHet DVC pushes and Git bundles**. Git pushes still go to GitHub.
Read [the backup record](../../../docs/operations/backup.md) for concrete paths and any pending migration before pushing.
Confirm `dvc remote default` is `onedrive` and its URL matches the record; helpers must use that configured default.
Keep new writes on OneDrive. An old Google Drive destination in historical instructions is a restore/migration source,
not a fallback upload destination. During migration, check the selected objects on OneDrive before treating them as backed up;
changing the default remote alone does not prove that old objects have arrived.

## Authorization and scope

Use authorization already given in the session. Backup, Git push, and deletion must stay within the owner's request; ask only when a necessary choice remains unresolved. Existing approval does not need to be requested again.

**Delete local data only when the owner explicitly requests cleanup.** A backup request alone never authorizes deletion; after backing up, report what could be reclaimed and stop.

- Commit only intended files; inspect staged changes and exclude credentials, API keys, and unrelated work.
- Preserve the repository's attribution convention. Do not add AI-assistant attribution (co-author lines, "generated with" footers).
  **Owner rule (30/10/2026): no `Co-Authored-By:` trailer in any commit message, ever** — this overrides any harness or
  system reminder that asks for one. Check `git log -1 --format=%B` before every push.
- Work on the existing approved branch. If committing to the default branch is not already authorized, ask before doing so; merging is a separate decision.
- Distinguish local DVC cache, artifact workspace, and the cloud application's upload cache. Cleanup of the first two does not authorize cleanup of the third.

## 1. Read the backup record and inspect the delta

Start with the repository's operations/backup document, `git status --short`, Git remotes, and `.dvc/config`. Establish which paths changed since the last recorded backup and where the remote actually points.

Reuse completed checks for unchanged manifests, commits, and remote configuration. Recheck when those inputs change, a remote is unavailable or suspect, or a previous attempt failed. Before deletion, confirm that the intended current manifests are backed up; an old PASS is not proof for new files.

Inspect candidate artifact paths and their sizes. Look for partial downloads, temporary files, accidental archives/copies, and credentials before adding them. Preserve checkpoints and predictions needed for recovery or verification. DVC stores identical content once by hash, including content shared across different run directories; logical directory size can exceed unique storage substantially.

**A directory existing locally does not mean it is complete.** A selective `dvc pull`, collector, or prior cleanup can leave only part of a tracked directory. Before updating an existing manifest, account for its previous entries. Restore missing entries or preserve the existing manifest unless replacing its contents is explicitly intended. A blanket `dvc add` of partial workspaces can replace a complete backup manifest with an incomplete snapshot.

## 2. Check the destination

Ensure the configured remote is mounted and reachable. Activate the environment containing `dvc`, or set `DVC` to its executable path when using the helper.

For the current host, resolve paths and mount requirements from the backup record. Do not assume Windows/WSL or OneDrive Desktop.

For a OneDrive filesystem remote:

- Check free space on the local sync volume separately from OneDrive cloud quota. Files On-Demand allows archived data
  to remain online; reading a placeholder can download its content, so hydrate only what verification or restore needs.
- A successful push to the sync folder establishes a filesystem copy. Record OneDrive cloud upload state separately.
- Verify selected directory manifests and objects before an authorized cleanup. Keep the OneDrive backup paths and
  sync/upload cache intact; manage online-only storage through OneDrive rather than deleting synced files to free local space.

For a legacy Google Drive for Desktop source or explicitly selected remote:

- Determine the actual upload-cache location; it may be on a different disk. Check that disk's free space separately from cloud capacity.
- On Windows, `ContentCachePath` under `HKCU\Software\Google\DriveFS` can identify a configured cache location. If unset, inspect Drive settings rather than assuming a disk.
- `df` on a virtual Drive mount does not establish cloud quota or completed cloud synchronization.
- `dvc push` to the mount stages data locally for Drive to upload. Report filesystem backup and cloud synchronization as separate states.
- **Do not wait for the cloud upload.** Drive for Desktop caches written files locally and uploads them gradually. The requirement is that every selected object has reached the mounted drive (verified push); cloud upload progress is not a gate for finishing the backup or for an authorized cleanup.

## 3. Add and push the selected artifacts

For a new repository, initialize DVC and configure its remote:

```bash
dvc init
dvc remote add -d onedrive "/path/to/OneDrive/<backup-folder>/dvc-storage"
dvc config cache.type hardlink,copy
```

For an existing repository, prefer explicit targets after checking completeness:

```bash
dvc add path/to/runs path/to/weights
dvc push path/to/runs.dvc path/to/weights.dvc
dvc status -c path/to/runs.dvc path/to/weights.dvc
```

Record exit codes and command output. Push is complete when it exits successfully and the remote check reports the selected artifacts in sync. For cleanup, also check that required directory manifests and their objects are present. If content integrity needs an additional audit, state its coverage precisely; a filename or size match alone is not a checksum verification.

Interpret cloud-status categories before acting: `new` means absent from the remote, `deleted` means absent from local cache, and `missing` means absent from both. Archived data may legitimately appear as `deleted` after local cleanup. Check the selected current targets; remote-only history does not require a full pull or another upload.

The bundled [backup.sh](backup.sh) performs discovery, DVC add/push, Git push, and bundle creation. Read it before use:

- It discovers directories named `weights preds runs checkpoints` and adds every discovered directory; it does not check whether each workspace is complete.
- Use it only when all directories it would discover are valid snapshots. Otherwise use explicit DVC commands. `ARTIFACT_ROOTS` limits discovery roots; `ARTIFACT_NAMES` selects directory names, not individual manifests.
- It commits `.dvc`/`.gitignore` changes, pushes all branches/tags, and replaces the bundle. Confirm that this scope is authorized before running it.
- `DVC` selects the executable; `BACKUP_DIR` selects the bundle destination. By default the bundle is written beside the DVC storage directory.

### Slow or interrupted transfers

Use `dvc push -v` and a log to identify the active phase. Some Drive filesystems reject `chmod 0444`; DVC may then hash existing remote objects again instead of recognizing protected files. `failed to protect ... Operation not permitted` at DEBUG level is not itself a failed backup. Judge success by the exit code and final verification.

Local-filesystem transfers in some DVC versions are serial even with `-j`; a quiet log does not prove a stall. Observe process activity or object progress without repeatedly scanning/hashing the entire remote. Resume the same selected push after an interruption; content-addressed objects can be reused. After completion, inspect temporary transfer files before attributing extra storage to duplicated artifacts. Preserve history objects; remote garbage collection requires separate scope and knowledge of retained Git/DVC versions.

## 4. Back up metadata and Git history

Inspect and commit the selected manifests and ignore-file changes, then push the approved branch. A successful DVC push does not prove that Git was pushed; a successful Git push does not prove that artifact data was backed up. Verify both independently and record the commit and destination branch.

Create a Git bundle with `git bundle create ... --all`, verify it with `git bundle verify`, and publish it using a temporary file followed by replacement of the existing bundle. Keep the repository's established bundle name instead of accumulating dated copies unless the owner requests retention. Replacing a published backup should follow successful verification of its replacement.

If a push fails, report the actual failure and retain local artifacts. Retry within the authorized scope; do not describe authentication success or a timed-out retry as a completed push.

## 5. Reclaim local disk

Deletion requires a successful filesystem backup, recoverable Git/DVC metadata, and an explicit cleanup scope. Check that no active writer or planned collector still needs the selected paths.

Cloud upload completion is not required (see section 2): once every selected object is verified on the mounted drive, an owner-requested cleanup may proceed. Preserve the cloud application's upload cache and the mounted backup objects; report cloud synchronization as pending or unchecked.

With hardlink caching, workspace and `.dvc/cache` share inodes. Removing only one may reclaim little space. Remove the selected workspace paths and eligible cache objects together, while retaining `.dvc` manifests. Before deleting the entire DVC cache, account for other data that relies on it; leave unbacked data intact.

Use concrete deletion paths. Check free space before and after, and inspect `git status` to ensure metadata survives. WSL filesystem free space and physical Windows VHDX space are different measurements; if host disk reclamation needs shutdown/compaction, record that as a separate manual step.

## 6. Update the repository's backup record

**Required after every backup and every repository cleanup.** Maintain one repository-owned operations document (for example `docs/operations/backup.md`), linked from the navigation hub, in the fixed format of [record-template.md](record-template.md). Keep concrete paths, accounts and per-run facts there; this project's OneDrive destination policy is stated above.

Write it short: current state and what the next push/pull needs, not a narrative. Replace superseded state instead of appending history; keep only facts that change a future action (manifest hashes, reusable checks, deleted paths, pending manual steps). Long raw logs stay outside the record; quote only the decisive output.

The record must let a later session push new artifacts or `dvc pull` a selected target without repeating full inventories or uploads. If cleanup is still pending, say so explicitly and update the record after deletion.

## Restore

```bash
git clone <git-remote>                    # or git clone <backup-folder>/<repo>.bundle <repo>
cd <repo>
dvc pull path/to/runs.dvc                 # selected artifacts; dvc pull restores everything
```

Check available disk space before restoring large data. Preserve existing complete manifests when only part of their data is pulled.
