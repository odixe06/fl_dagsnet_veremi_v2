"""Build a private W&B-free Kaggle notebook that resumes MobiHet v4 from session 1."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OWNER = "minhiw"
SLUG = "mobihet-v4-weighted-nowb-s2"
RUN = "mobihet-v4-weighted-nowb"
PREVIOUS = "minhiw/mobihet-v4-weighted-nowb"


def main():
    subprocess.run([
        sys.executable, str(ROOT / "scripts/gen_notebook.py"),
        "--owner", OWNER, "--slug", SLUG, "--run", RUN, "--method", "mobihet",
        "--no-wandb", "--max-hours", "10", "--input-kernel", PREVIOUS,
    ], check=True)
    path = ROOT / "notebooks" / SLUG / f"{SLUG}.ipynb"
    nb = nbformat.read(path, as_version=4)
    nb.cells[0].source += (
        "\n\nResume session 2 from `minhiw/mobihet-v4-weighted-nowb/1`, round 25. "
        "Restore the exact executed source snapshot and both rank checkpoints before data decode. W&B disabled."
    )
    nb.cells[2].source = '''subprocess.run(["git", "clone", "-q", "https://github.com/odixe06/fl_dagsnet_veremi_v2.git",
                "/kaggle/temp/src"], check=True)
from pathlib import Path
import hashlib, json
snapshots = list(Path("/kaggle/input").glob("**/source_snapshot.json"))
manifests = list(Path("/kaggle/input").glob("**/source_manifest.json"))
assert len(snapshots) == len(manifests) == 1, (snapshots, manifests)
source = json.loads(snapshots[0].read_text())
hashes = json.loads(manifests[0].read_text())
assert source.keys() == hashes.keys()
for name, content in source.items():
    assert hashlib.sha256(content.encode()).hexdigest() == hashes[name], name
    p = Path("/kaggle/temp/src") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
Path("/kaggle/working/source_manifest.json").write_text(json.dumps(hashes, indent=2))
Path("/kaggle/working/source_snapshot.json").write_text(json.dumps(source))
print("restored exact session-1 source", len(source), "files", flush=True)
'''
    resume = f'''from pathlib import Path
import gc, json, os, shutil, torch
run_name = {RUN!r}
candidates = [p for p in Path("/kaggle/input").glob(f"**/{{run_name}}/state.pt")
              if p.parent.name == run_name]
assert len(candidates) == 1, candidates
src = candidates[0].parent
for rank, name in enumerate(("state.pt", "state_rank1.pt")):
    p = src / name
    assert p.is_file(), p
    state = torch.load(p, map_location="cpu", weights_only=True)
    assert state["round"] == 25 and state.get("method", "mobihet") == "mobihet", (name, state.get("round"))
    assert state["method_state"]["version"] == 25
    assert "gen_q" in state["method_state"] and "gen_g" in state["method_state"]
    for key, value in state["method_state"].items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), (name, key)
    del state
    gc.collect()
dst = Path("/kaggle/working") / run_name
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(src)
    out = dst / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    shutil.copyfile(p, part)
    assert part.stat().st_size == p.stat().st_size
    os.replace(part, out)
rows = [json.loads(s) for s in (dst / "rounds.jsonl").read_text().splitlines()]
assert [r["round"] for r in rows] == list(range(1, 26))
print("resume preflight PASS: round 25, both ranks, 25 metric rows", flush=True)
'''
    nb.cells.insert(4, nbformat.v4.new_code_cell(resume))
    # Generated launch cell follows data decode. It runs the same run directory and 50-round horizon.
    nb.cells[-2].source = nb.cells[-2].source.replace(
        "print(cmd)", 'assert budget > 8.4, f"insufficient budget after setup: {budget:.2f}h"\nprint(cmd, flush=True)'
    )
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert all("import wandb" not in c.source and "--wandb " not in c.source
               for c in nb.cells if c.cell_type == "code")
    nbformat.validate(nb)
    nbformat.write(nb, path)
    meta_path = path.parent / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == ["odixe0502/veremi-nextgen-v4-scaled-per-vehicle"]
    assert meta["kernel_sources"] == [PREVIOUS]
    print(path)
    print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
