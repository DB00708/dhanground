"""Run the GPU stages on Kaggle (free T4 GPUs, fp16) from this machine. No Hub token ever goes to Kaggle.

  python -m scripts.kaggle_run stage                  # upload inputs as a PRIVATE Kaggle dataset <kaggle-user>/dhanground-work
  python -m scripts.kaggle_run push smoke             # push + run the GPU notebook pinned to the current commit
  python -m scripts.kaggle_run push sft-seed0         # A1 s0 on GPU 0 and A2 s0 on GPU 1, in parallel (AlignTune)
  python -m scripts.kaggle_run status | logs <task>   # status; download the notebook's output into data/kaggle/<task>

Tiny Aya Fire is attached from Kaggle Models (cohere-labs/tiny-aya/transformers/fire/1) and loaded through
DHANGROUND_BASE_PATH; adapters come back as notebook output and are pushed to the Hub from this machine.
Needs KAGGLE_API_TOKEN (Kaggle CLI) in the environment (.env).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_URL = "https://github.com/DB00708/dhanground"
DATASET = "dhanground-work"
BASE_MODEL_SOURCE = "cohere-labs/tiny-aya/transformers/fire/1"
NOTEBOOK = "dhanground-gpu"
SHAPE = "NvidiaTeslaT4"
# A single-P100 copy (own notebook slug) runs the same steps one after another; it often gets a GPU sooner.
P100 = {"notebook": NOTEBOOK + "-p100", "shape": "NvidiaTeslaP100"}
SMOKE_ROWS = 400
STAGE = {"data/raw/train.jsonl": "smoke/train.jsonl",
         "data/arms/A1/train.jsonl": "arms/A1/train.jsonl",
         "data/arms/A2/train.jsonl": "arms/A2/train.jsonl"}
TASKS = {
    "smoke": {
        "steps": [["python", "-m", "scripts.smoke_train", "--rows", "data/smoke/train.jsonl",
                   "--n", "48", "--steps", "24", "--per-device", "2"]],
        "parallel": False,
    },
    "sft-seed0": {
        # Fail fast in the same session (one queue wait, one setup): no training if the fp16 smoke test says no-go.
        "pre": [["python", "-m", "scripts.smoke_train", "--rows", "data/smoke/train.jsonl",
                 "--n", "48", "--steps", "24", "--per-device", "2"]],
        "steps": [["python", "-m", "adapters.aligntune_sft", "--arm", "A1", "--seed", "0", "--backend", "aligntune"],
                  ["python", "-m", "adapters.aligntune_sft", "--arm", "A2", "--seed", "0", "--backend", "aligntune"]],
        "parallel": True,
    },
}

KERNEL_SCRIPT = r'''
import glob, json, os, shutil, subprocess, sys
def sh(cmd, **kw):
    print("+ " + (cmd if isinstance(cmd, str) else " ".join(cmd)), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)
print("python", sys.version, flush=True)
if sys.version_info < (3, 11):
    sys.exit("KAGGLE_PYTHON_TOO_OLD: aligntune 0.1.33 needs Python >= 3.11")
CFG = json.loads(__CFG__)
configs = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "tiny-aya" in p.lower()]
if not configs:
    sys.exit("NO_BASE_MODEL: Tiny Aya is not attached; inputs: %s" % os.listdir("/kaggle/input"))
os.environ.update({"DHANGROUND_PRECISION": "fp16",
                   "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
print("base model:", configs[0], flush=True)
root = "/kaggle/working/dhanground"
sh(["git", "clone", "-q", CFG["repo_url"], root]); os.chdir(root); sh(["git", "checkout", "-q", CFG["sha"]])
shutil.rmtree(".git", ignore_errors=True)  # keep the notebook output small
sys.path.insert(0, root)
from adapters.common import flatten_chat_template
base = "/kaggle/working/base"
print("chat template:", flatten_chat_template(configs[0], base), flush=True)
os.environ["DHANGROUND_BASE_PATH"] = base
sh([sys.executable, "-m", "pip", "install", "-q", "aligntune==0.1.33", "curatorkit==0.1.24", "pydantic", "PyYAML"])
sh([sys.executable, "-c", "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"])
work = [p for p in glob.glob("/kaggle/input/**/arms", recursive=True)] + glob.glob("/kaggle/input/**/smoke", recursive=True)
if not work:
    sys.exit("NO_DATA: dataset %s is not attached; inputs: %s" % (CFG["dataset"], os.listdir("/kaggle/input")))
for d in work:
    shutil.copytree(d, os.path.join("data", os.path.basename(d)), dirs_exist_ok=True)
os.environ["DHANGROUND_GIT_SHA"] = CFG["sha"][:7]
for s in CFG.get("pre", []):
    sh([sys.executable if a == "python" else a for a in s])
steps = [[sys.executable if a == "python" else a for a in s] for s in CFG["steps"]]
import torch
n_gpu = torch.cuda.device_count()
if CFG["parallel"] and n_gpu >= len(steps):
    procs = []
    for gpu, s in enumerate(steps):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = f"/kaggle/working/step{gpu}.log"
        print(f"GPU {gpu}: " + " ".join(s), flush=True)
        procs.append((s, subprocess.Popen(s, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
    codes = [(s, p.wait(), log) for s, p, log in procs]
    for s, code, log in codes:
        print(f"=== exit {code}: {' '.join(s)}\n" + open(log).read()[-6000:], flush=True)
    if any(c for _, c, _ in codes):
        sys.exit("SOME_STEP_FAILED")
else:
    for s in steps:
        sh(s)
print("KAGGLE_TASK_DONE", flush=True)
'''


def kaggle(*args: str) -> str:
    exe = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    r = subprocess.run([str(exe), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def kaggle_user() -> str:
    for line in kaggle("config", "view").splitlines():
        if "username" in line:
            return line.split(":", 1)[1].strip()
    raise SystemExit("could not read the Kaggle username from `kaggle config view`")


def git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True, text=True).stdout
    if "origin/main" not in pushed:
        raise SystemExit(f"commit {sha[:7]} is not on origin/main: push it first, the notebook clones the public repo")
    return sha


def build_kernel(task: str, sha: str, kuser: str, folder: Path, p100: bool = False) -> Path:
    t = TASKS[task]
    notebook, shape = (P100["notebook"], P100["shape"]) if p100 else (NOTEBOOK, SHAPE)
    cfg = {"repo_url": REPO_URL, "sha": sha, "dataset": f"{kuser}/{DATASET}", "parallel": t["parallel"],
           "pre": t.get("pre", []), "steps": t["steps"]}
    (folder / "run.py").write_text(KERNEL_SCRIPT.replace("__CFG__", repr(json.dumps(cfg))), encoding="utf-8")
    meta = {"id": f"{kuser}/{notebook}", "title": notebook, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": shape, "dataset_sources": [f"{kuser}/{DATASET}"],
            "competition_sources": [], "kernel_sources": [], "model_sources": [BASE_MODEL_SOURCE]}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def stage_dir(folder: Path, kuser: str) -> Path:
    for src, dst in STAGE.items():
        if not Path(src).exists():
            raise SystemExit(f"missing {src}: run the pipeline stage that writes it first")
        out = folder / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        if dst.startswith("smoke/"):
            lines = Path(src).read_text(encoding="utf-8").splitlines(keepends=True)[:SMOKE_ROWS]
            out.write_text("".join(lines), encoding="utf-8")
        else:
            shutil.copyfile(src, out)
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET, "id": f"{kuser}/{DATASET}", "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    return folder


def stage() -> None:
    kuser = kaggle_user()
    with tempfile.TemporaryDirectory() as d:
        stage_dir(Path(d), kuser)
        exists = DATASET in kaggle("datasets", "list", "--mine", "-s", DATASET)
        if exists:
            print(kaggle("datasets", "version", "-p", d, "-m", "update inputs", "-r", "zip"))
        else:
            print(kaggle("datasets", "create", "-p", d, "-r", "zip"))  # private unless --public
    for _ in range(60):
        status = kaggle("datasets", "status", f"{kuser}/{DATASET}").strip().lower()
        if "ready" in status:
            print(f"dataset {kuser}/{DATASET} is ready (private)")
            return
        time.sleep(10)
    raise SystemExit(f"dataset not ready after 10 minutes: {status}")


def push(task: str, no_run: bool = False, p100: bool = False) -> None:
    sha, kuser = git_sha(), kaggle_user()
    with tempfile.TemporaryDirectory() as d:
        build_kernel(task, sha, kuser, Path(d), p100)
        print(kaggle("kernels", "push", "-p", d, *(["--no-run"] if no_run else [])))
    nb, shape = (P100["notebook"], P100["shape"]) if p100 else (NOTEBOOK, SHAPE)
    print(f"notebook https://www.kaggle.com/code/{kuser}/{nb}: {task} at {sha[:7]} (fp16, {shape})")


def logs(task: str, p100: bool = False) -> None:
    nb = P100["notebook"] if p100 else NOTEBOOK
    out = Path("data/kaggle") / (task + ("-p100" if p100 else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", f"{kaggle_user()}/{nb}", "-p", str(out), "-o"))
    for f in sorted(out.glob("*.log")):
        print(f"--- {f.name}\n" + f.read_text(encoding="utf-8", errors="replace")[-4000:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["stage", "push", "status", "logs"])
    ap.add_argument("task", nargs="?", choices=sorted(TASKS))
    ap.add_argument("--no-run", action="store_true", help="save the notebook without running it")
    ap.add_argument("--p100", action="store_true", help="use the single-P100 notebook copy (steps run one after another)")
    a = ap.parse_args()
    if a.command == "stage":
        stage()
    elif a.command == "status":
        print(kaggle("kernels", "status", f"{kaggle_user()}/{P100['notebook'] if a.p100 else NOTEBOOK}"))
    elif not a.task:
        raise SystemExit(f"{a.command} needs a task: {sorted(TASKS)}")
    elif a.command == "push":
        push(a.task, a.no_run, a.p100)
    else:
        logs(a.task, a.p100)


if __name__ == "__main__":
    main()
