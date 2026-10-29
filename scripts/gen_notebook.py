"""Generate one private Kaggle notebook (2xT4) that runs scripts/run_fl.py (or another repo script) at a pinned commit.

    python scripts/gen_notebook.py --owner odixe0502 --slug fl2-fedavg --method fedavg --max-hours 11.5
    python .claude/skills/kaggle-training-notebook/scripts/embed_wandb_key.py notebooks/fl2-fedavg/fl2-fedavg.ipynb \
        --metadata notebooks/fl2-fedavg/kernel-metadata.json
    python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py odixe0502 -- kaggle kernels push -p notebooks/fl2-fedavg

The notebook clones github.com/odixe06/fl_dagsnet_veremi_v2 at --sha (default: local HEAD, must be pushed), reads the v4
dataset mount, resumes from a previous session's output when --resume-kernel is given, and writes everything under
/kaggle/working/<run>. notebooks/ is gitignored: embedded notebooks hold a credential.
"""
import argparse
import json
import os
import subprocess

import nbformat

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNTIME = json.load(open(os.path.join(ROOT, "knowledge", "runtime.json")))
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
REPO = "https://github.com/odixe06/fl_dagsnet_veremi_v2.git"
WANDB_PROJECT = "fl-dagsnet-veremi-v2"


def cells(a, sha):
    run = a.run or a.slug
    args = a.args or (f"scripts/run_fl.py --method {a.method} --out /kaggle/working/{run} --width {a.width} "
                      f"--rounds {a.rounds} --compile --wandb {WANDB_PROJECT} --max-hours BUDGET --cache ''" + (" --set " + " ".join(a.set) if a.set else ""))
    yield "markdown", f"# {a.slug}\n\nRepo `{REPO}` @ `{sha}` · dataset `{DATASET}` · `torchrun --nproc_per_node=2 {args}`"
    yield "code", """import time, os, subprocess, sys, glob, shutil
T0 = time.monotonic()
import torch
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}"
assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(n)), "expected Tesla T4"
print(torch.__version__, sys.version, [torch.cuda.get_device_name(i) for i in range(n)])"""
    yield "code", f"""subprocess.run(["git", "clone", "-q", "{REPO}", "/kaggle/temp/src"], check=True)
subprocess.run(["git", "-C", "/kaggle/temp/src", "checkout", "-q", "{sha}"], check=True)
print(subprocess.run(["git", "-C", "/kaggle/temp/src", "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)"""
    yield "code", """import wandb
# BEGIN INLINE WANDB CREDENTIAL — owner-authorized private notebook
raise SystemExit("W&B credential not embedded: run embed_wandb_key.py")
# END INLINE WANDB CREDENTIAL
print("W&B authenticated")"""
    resume = ""
    if a.resume_kernel:
        resume = f"""prev = [p for p in glob.glob("/kaggle/input/**/{run}/state.pt", recursive=True)]
assert prev, "resume requested but no {run}/state.pt in inputs"
os.makedirs("/kaggle/working/{run}", exist_ok=True)
for f in glob.glob(os.path.join(os.path.dirname(prev[0]), "*")):
    shutil.copy(f, "/kaggle/working/{run}/")
print("resumed from", prev[0])
"""
    yield "code", """t = time.monotonic()
sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
D.build_npy(D.find_root(), "/kaggle/temp/v4npy")    # one reader; every rank then memmaps (V4_NPY)
print(f"npy cache built in {time.monotonic() - t:.0f}s")"""
    yield "code", resume + f"""os.environ.update(TORCHINDUCTOR_COMPILE_THREADS="2", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  V4_NPY="/kaggle/temp/v4npy")
budget = {a.max_hours} - (time.monotonic() - T0) / 3600
cmd = "cd /kaggle/temp/src && torchrun --standalone --nproc_per_node=2 {args}".replace("BUDGET", f"{{budget - 0.4:.2f}}")
print(cmd)
rc = subprocess.run(cmd, shell=True).returncode
assert rc == 0, f"training exited {{rc}}"
"""
    yield "code", f"""for f in glob.glob("/kaggle/working/{run}/*.jsonl"):
    print(f, open(f).read()[-3000:])"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--method", default="fedavg")
    ap.add_argument("--run", default="", help="run dir name (default slug); keep it across resume sessions")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=11.5)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--args", default="", help="replace the run_fl.py command line (e.g. a centralized script)")
    ap.add_argument("--sha", default="")
    ap.add_argument("--resume-kernel", default="", help="<owner>/<slug> whose output holds <run>/state.pt")
    a = ap.parse_args()
    sha = a.sha or subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "markdown" else nbformat.v4.new_code_cell(s) for k, s in cells(a, sha)]
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True, "language": "python",
                   "sourceType": "notebook"}})
    out = os.path.join(ROOT, "notebooks", a.slug)
    os.makedirs(out, exist_ok=True)
    nbformat.write(nb, os.path.join(out, f"{a.slug}.ipynb"))
    meta = {"id": f"{a.owner}/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "docker_image": RUNTIME["docker_image"],
            "dataset_sources": [DATASET], "kernel_sources": [a.resume_kernel] if a.resume_kernel else [],
            "competition_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
    print(out, sha)


if __name__ == "__main__":
    main()
