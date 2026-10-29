# Method/build evidence layout

Use the owner's current repository layout. For a migration, inventory source/notebook hashes, DVC stage outputs,
writers and links first; agree whether to move physically or only route documents. The organizing request does not
grant backup, remote push or cleanup permissions.

Keep method dossiers in `docs/methods/<method>/`: a short README, detailed design with evidence/mechanism/gates, and
method lessons. One `docs/proposals.md` catalogs all proposals, tested status and scoped PASS/FAIL/pending.
Build descriptions and readable results belong under that method's docs; each description links the exact source,
controls, configs, probes, calibration, pilot, committed artifacts, metrics and logs.

Physical builds belong under `methods/<method>/builds/<build>/`, including probes and calibration, not a detached
global probes directory. Repeated builds of one method share its parent folder. Comparison builds have one physical
artifact owner and links from participating methods; do not duplicate checkpoints to make navigation look simpler.

Separate frozen source snapshots, public notebooks, method scripts, evidence/probes, artifacts and private credentials.
Put standalone logs under that build's `logs/`; an optional root log alias provides navigation. Logs inside a committed DVC directory
remain part of its immutable content unless an explicitly authorized new manifest is created.

Keep reusable infrastructure in `scripts/` and shared packages; move build-specific modules and helpers with their build.
Classify by actual exported helpers and consumers, not versioned filenames. Mixed immutable modules that provide shared
helpers remain shared until a separately authorized source change. Record canonical ownership and dependency paths.
If fingerprints enumerate the old package filenames, preserve relative compatibility aliases and verify the complete
filename-to-byte-hash map before/after. Test imports and the generator runtime layout; do not silently change resume identity.

Check imports and
`__file__.resolve().parents[N]` after moves: a symlink does not preserve the old resolved depth. Adapt operational
root discovery or reconstruct the original runtime tree, preserving scientific source bytes and fingerprints.

Move `.dvc` metadata with the build. Change only the output path when appropriate; preserve object hashes, sizes,
file counts and remote-only entries. Ignore compatibility aliases in DVC scanning to avoid duplicate stages; ignore
heavy/private paths in Git. Partial metadata restores are incomplete workspaces, not valid `dvc add` inputs.

Every owned logical folder has a short Markdown entrypoint. Do not add README files inside frozen DVC payloads or
credential folders just to satisfy that rule; their parent entrypoint and artifact link serve the same purpose.

Validation evidence should include frozen SHA checks, preserved DVC identities and remote directory-manifest MD5,
alias/link resolution, secret boundary, import/compile checks for moved helpers, a refreshed file index, and independent
Git HEAD/index checks. Workspace moves may leave unstaged Git changes; do not call that a committed migration.

Keep the host copies semantically synchronized. Codex adds `agents/openai.yaml`; Claude Code uses `SKILL.md` name/
description frontmatter and optional references. No accounts, live kernel IDs, quota snapshots or model HP belong here.

## Active work and archive

Use names that describe the mechanism or responsibility. Batch IDs such as V2/V3 are provenance, not necessarily
replacement versions. Keep active builds and reusable code in the working area; move completed builds, retired
tools and historical dossiers into a clearly indexed archive. Keep only transferable lessons in current lessons
documents, with links to the archived evidence. Do not change scientific status merely to classify a folder.

A closed experiment can still provide a frozen runtime dependency. Preserve that source and its immutable
filename/hash map; place compatibility facades outside the canonical working code. Check resolved source depth,
script namespace imports, standalone path reads, current and archive indexes, and ignored credentials/artifacts
in both trees. Do not add navigation README files to committed payloads.
