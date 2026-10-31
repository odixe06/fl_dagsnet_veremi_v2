"""Build the private, W&B-free P1a identifiability notebook."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OWNER = "minhiw"
SLUG = "mobihet-v4-p1-identifiability"
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"


def main():
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    script = (ROOT / "scripts/p1_identifiability.py").read_text(encoding="utf-8")
    order = (ROOT / "knowledge/input_order.json").read_text(encoding="utf-8")
    ast.parse(script)
    source = {"scripts/p1_identifiability.py": script, "knowledge/input_order.json": order}
    manifest = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in source.items()}
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# MobiHet v4 P1a — identifiability 0 vs 13\n\n"
            "Fit only fold `train`; evaluate full fold `val`; official test is neither mounted separately nor read. "
            "HGB and 3×256 MLP use frozen 77 features. W&B disabled."
        ),
        nbformat.v4.new_code_cell('''import json, os, sys, time
from pathlib import Path
import torch
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(2))
print(torch.__version__, sys.version, [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
os.environ.update(WANDB_MODE="disabled", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
'''),
        nbformat.v4.new_code_cell(f'''import ast, hashlib
source = {source!r}
manifest = {manifest!r}
base = Path("/kaggle/temp/p1")
for name, body in source.items():
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name]
    if path.suffix == ".py": ast.parse(body)
Path("/kaggle/working/source_snapshot.json").write_text(json.dumps(source))
Path("/kaggle/working/source_manifest.json").write_text(json.dumps(manifest, indent=2))
print("source preflight PASS", manifest, flush=True)
'''),
        nbformat.v4.new_code_cell('''roots = [p.parent for p in Path("/kaggle/input").glob("**/scaler.json")
         if (p.parent / "train").is_dir() and (p.parent / "test").is_dir()]
assert len(roots) == 1, roots
root = roots[0]
print("scaled root", root, flush=True)
'''),
        nbformat.v4.new_code_cell('''out = Path("/kaggle/working/mobihet-v4-p1-identifiability")
cmd = [sys.executable, "/kaggle/temp/p1/scripts/p1_identifiability.py",
       "--root", str(root), "--features", "/kaggle/temp/p1/knowledge/input_order.json", "--out", str(out)]
print(cmd, flush=True)
subprocess = __import__("subprocess")
subprocess.run(cmd, check=True)
'''),
        nbformat.v4.new_code_cell('''report = json.loads(Path("/kaggle/working/mobihet-v4-p1-identifiability/p1_report.json").read_text())
assert report["protocol"]["fit_fold"] == "train" and report["protocol"]["eval_fold"] == "val"
assert report["protocol"]["official_test_read"] is False
assert report["dataset"]["rows_all_labels"] == 37361273
assert report["counts"]["train_13"] > 0 and report["counts"]["val_13"] > 0
print(json.dumps(report, indent=2), flush=True)
''')]
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook"}})
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert "wandb.login" not in cell.source and "--wandb" not in cell.source
    nbformat.validate(nb)
    out = ROOT / "notebooks" / SLUG
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{SLUG}.ipynb"
    nbformat.write(nb, path)
    metadata = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4", "docker_image": runtime["docker_image"],
                "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1))
    print(path)
    print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
