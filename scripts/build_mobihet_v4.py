"""Freeze reviewed local source into a private, W&B-free Kaggle notebook."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", default="mobihet-v4-weighted-nowb")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", args.owner,
                    "--slug", args.slug, "--method", "mobihet", "--no-wandb", "--max-hours", "10"], check=True)
    path = ROOT / "notebooks" / args.slug / f"{args.slug}.ipynb"
    nb = nbformat.read(path, as_version=4)
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"), *ROOT.glob("tests/*.py"),
                    *ROOT.glob("knowledge/*.json"), ROOT / "scripts/run_fl.py"])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    hashes = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in source.items()}
    nb.cells[0].source += "\n\nLocal source snapshot below overrides base Git SHA. W&B disabled. " \
        "CPU resume test and two-GPU real-data smoke gate precede 50-round training. " \
        "Smoke uses 32 clients; final uses all 5,343. Test scores never choose hyperparameters."
    nb.cells[2].source = f'''from pathlib import Path
import hashlib, json
source = {source!r}
hashes = {hashes!r}
for name, content in source.items():
    p = Path("/kaggle/temp/src") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[name]
Path("/kaggle/working/source_manifest.json").write_text(json.dumps(hashes, indent=2))
Path("/kaggle/working/source_snapshot.json").write_text(json.dumps(source))
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_methods.py", "tests/test_mobihet_resume.py",
                "-k", "mobihet or next_round_resume"], cwd="/kaggle/temp/src", check=True,
               env={{**os.environ, "OMP_NUM_THREADS": "2", "PYTHONPATH": "/kaggle/temp/src"}})
'''
    smoke = '''os.environ.update(V4_NPY="/kaggle/temp/v4npy", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  TORCHINDUCTOR_COMPILE_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  WANDB_MODE="disabled")
import numpy as np
for split, rows in (("train", 37361273), ("test", 9347679)):
    ids = np.load(f"/kaggle/temp/v4npy/{split}_ids.npy")
    y = np.load(f"/kaggle/temp/v4npy/{split}_Y.npy", mmap_mode="r")
    assert np.array_equal(ids, np.arange(5343)) and len(y) == rows, (split, len(ids), len(y))
    assert y.min() == 0 and y.max() == 15
base = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/run_fl.py", "--method", "mobihet",
        "--out", "/kaggle/working/smoke", "--width", "1", "--rounds", "50", "--subset", "32", "--compile", "--cache", ""]
for stop in (1, 2):
    subprocess.run(base + ["--stop-after", str(stop)], cwd="/kaggle/temp/src", check=True)
rows = [json.loads(s) for s in Path("/kaggle/working/smoke/rounds.jsonl").read_text().splitlines()]
assert [r["round"] for r in rows] == [1, 2]
for r in rows:
    assert r["applied_steps"] > 0 and r["skipped_frac"] < 0.01, r
    for key, value in r.items():
        if isinstance(value, (float, int)):
            assert np.isfinite(value), (key, value)
for rank in (0, 1):
    p = Path("/kaggle/working/smoke") / ("state.pt" if rank == 0 else "state_rank1.pt")
    state = torch.load(p, map_location="cpu", weights_only=True)
    assert state["round"] == 2
    assert "gen_q" in state["method_state"] and "gen_g" in state["method_state"]
    del state
Path("/kaggle/working/SMOKE_PASS.json").write_text(json.dumps({"rounds": 2, "clients": 32, "resume": True}))
print("SMOKE PASS; start full-client protocol", flush=True)
'''
    nb.cells.insert(5, nbformat.v4.new_code_cell(smoke))
    nb.cells[-2].source = nb.cells[-2].source.replace('print(cmd)',
        'assert budget > 1.0, "insufficient session budget after smoke"\nprint(cmd, flush=True)')
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if index != 2:  # frozen shared runner retains optional W&B support, never enabled here
                assert "import wandb" not in cell.source and "--wandb " not in cell.source
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(path)
    print("source files:", len(source), "notebook sha256:", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
