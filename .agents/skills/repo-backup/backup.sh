#!/usr/bin/env bash
# Back up a repository so authorized local cleanup remains recoverable:
#   - code and Git history        -> origin + a Git bundle beside the DVC remote
#   - heavy artifact directories  -> the default DVC remote
# Run from anywhere in the repository: .agents/skills/repo-backup/backup.sh
#
# Repository settings come from .dvc/config or environment variables:
#   DVC             DVC executable (default: dvc from PATH)
#   ARTIFACT_ROOTS  space-separated discovery roots (default: .)
#   ARTIFACT_NAMES  directory names (default: "weights preds runs checkpoints")
#   BACKUP_DIR      bundle destination (default: parent of the default DVC remote URL)
#
# Restore: git clone <origin> (or git clone "$BACKUP_DIR/<repo>.bundle" <repo>); cd <repo> && dvc pull
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
REPO=$(basename "$PWD")

DVC=${DVC:-$(command -v dvc || true)}
[ -n "$DVC" ] || { echo "DVC not found: activate its environment or set DVC=/path/to/dvc" >&2; exit 1; }
REMOTE=$("$DVC" remote default)
[ -n "$REMOTE" ] || { echo "No default DVC remote configured (dvc remote add -d ...)" >&2; exit 1; }
URL=$("$DVC" config "remote.$REMOTE.url")
BACKUP_DIR=${BACKUP_DIR:-$(dirname "$URL")}
[ -d "$URL" ] || [ -d "$BACKUP_DIR" ] || { echo "Backup directory unavailable: $BACKUP_DIR; mount the configured cloud filesystem first; see docs/operations/backup.md" >&2; exit 1; }

# Refuse accidental writes into the local directory underneath an unmounted cloud mount.
if [[ "$URL" == "$HOME/Cloud/OneDrive/"* ]]; then
  mountpoint -q "$HOME/Cloud/OneDrive" || { echo "OneDrive rclone mount unavailable" >&2; exit 1; }
fi

# 1. Add discovered artifact directories; commit only DVC/ignore metadata changes.
read -ra roots <<< "${ARTIFACT_ROOTS:-.}"
read -ra names <<< "${ARTIFACT_NAMES:-weights preds runs checkpoints}"
match=(); for n in "${names[@]}"; do match+=(-o -name "$n"); done
mapfile -t dirs < <(find "${roots[@]}" \( -name .git -o -name .dvc -o -name node_modules \) -prune \
                      -o -type d \( "${match[@]:1}" \) -prune -print | sed 's|^\./||')
[ ${#dirs[@]} -eq 0 ] || "$DVC" add -q "${dirs[@]}"
git add -- '*.dvc' '*.gitignore'
git diff --cached --quiet || git commit -q -m "data: update DVC-tracked artifacts" -- '*.dvc' '*.gitignore'

# 2. Push artifact data to the DVC remote.
"$DVC" push

# 3. Push Git and create a bundle (temporary file, then replace the published name).
git push -q origin --all
git push -q origin --tags
mkdir -p "$BACKUP_DIR"
git bundle create -q "$BACKUP_DIR/$REPO.bundle.tmp" --all
mv -f "$BACKUP_DIR/$REPO.bundle.tmp" "$BACKUP_DIR/$REPO.bundle"

if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: uncommitted changes remain; they are NOT backed up:" >&2
  git status --short >&2
fi
echo "Filesystem backup complete. Cloud sync is separate; follow the owner's cleanup policy."
