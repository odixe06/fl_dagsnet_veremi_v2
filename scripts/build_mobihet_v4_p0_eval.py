"""Build a private, W&B-free r25 evaluation notebook for MobiHet's own-client decision rule."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OWNER = "minhiw"
SLUG = "mobihet-v4-p0-local-prior-r25"
PARENT = "minhiw/mobihet-v4-weighted-nowb"
PARENT_RUN = "mobihet-v4-weighted-nowb"
RUN = "mobihet-v4-p0-local-prior-r25"


def main():
    command = (f"scripts/run_fl.py --method mobihet --out /kaggle/working/{RUN} --width 1 "
               "--rounds 50 --eval-only --cache ''")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/gen_notebook.py"),
        "--owner", OWNER, "--slug", SLUG, "--method", "mobihet", "--no-wandb",
        "--max-hours", "2", "--input-kernel", PARENT, "--args", command,
    ], check=True)
    path = ROOT / "notebooks" / SLUG / f"{SLUG}.ipynb"
    nb = nbformat.read(path, as_version=4)
    paths = sorted([*ROOT.glob("core/*.py"), *ROOT.glob("methods/*.py"), *ROOT.glob("tests/*.py"),
                    *ROOT.glob("knowledge/*.json"), ROOT / "scripts/run_fl.py"])
    source = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    hashes = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in source.items()}
    nb.cells[0].source += (
        "\n\nP0 only: evaluate the completed r25 private checkpoint with its local-prior logits and retain the raw-logit "
        "control. No training, no W&B, no test-fitted coefficient. PASS requires client-F1 mean +0.005 and pooled "
        "macro-F1 no worse than -0.005."
    )
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
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_units.py", "tests/test_methods.py",
                "-k", "own_eval_applies_per_client_logit_bias or method_runs and mobihet"],
               cwd="/kaggle/temp/src", check=True,
               env={{**os.environ, "OMP_NUM_THREADS": "2", "PYTHONPATH": "/kaggle/temp/src"}})
'''
    restore = f'''from pathlib import Path
import json, shutil, torch
parent = [p for p in Path("/kaggle/input").glob("**/{PARENT_RUN}/state.pt")
          if p.parent.name == "{PARENT_RUN}"]
assert len(parent) == 1, parent
src = parent[0].parent
dst = Path("/kaggle/working/{RUN}")
dst.mkdir(parents=True, exist_ok=True)
for name in ("state.pt", "state_rank1.pt"):
    state = torch.load(src / name, map_location="cpu", weights_only=True)
    assert state["round"] == 25 and state.get("method", "mobihet") == "mobihet", (name, state.get("round"))
    assert state["method_state"]["version"] == 25
    assert all(torch.isfinite(v).all() for v in state["method_state"].values() if torch.is_tensor(v))
    shutil.copyfile(src / name, dst / name)
rows = [json.loads(line) for line in (src / "rounds.jsonl").read_text().splitlines()]
r25 = next(row for row in rows if row["round"] == 25)
Path("/kaggle/working/parent_r25.json").write_text(json.dumps(r25))
print("r25 checkpoint preflight PASS; both ranks", flush=True)
'''
    nb.cells.insert(4, nbformat.v4.new_code_cell(restore))
    gate = f'''from pathlib import Path
import json, math
parent = json.loads(Path("/kaggle/working/parent_r25.json").read_text())
result = json.loads(Path("/kaggle/working/{RUN}/eval_r025.json").read_text())
for metric, tol in (("accuracy", 1e-5), ("f1_macro", 1e-5), ("client_f1_macro_mean", 1e-5),
                    ("client_f1_macro_p10", 1e-3)):
    assert math.isclose(result[f"private/{{metric}}"], parent[f"private/{{metric}}"], rel_tol=0, abs_tol=tol), metric
gain_mean = result["private_local/client_f1_macro_mean"] - result["private/client_f1_macro_mean"]
gain_macro = result["private_local/f1_macro"] - result["private/f1_macro"]
gate = {{"round": 25, "gain_client_f1_mean": gain_mean, "gain_pooled_macro_f1": gain_macro,
        "pass": gain_mean >= 0.005 and gain_macro >= -0.005,
        "rule": "client mean gain >= 0.005 and pooled macro-F1 delta >= -0.005"}}
Path("/kaggle/working/P0_GATE.json").write_text(json.dumps(gate, indent=2))
print("P0 GATE", json.dumps(gate), flush=True)
'''
    nb.cells.insert(-1, nbformat.v4.new_code_cell(gate))
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if index != 2:  # frozen shared runner retains optional W&B support, never enabled here
                assert "import wandb" not in cell.source and "--wandb " not in cell.source
    nbformat.validate(nb)
    nbformat.write(nb, path)
    meta_path = path.parent / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == ["odixe0502/veremi-nextgen-v4-scaled-per-vehicle"]
    assert meta["kernel_sources"] == [PARENT]
    print(path)
    print("source files:", len(source), "sha256:", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
