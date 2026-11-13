"""Run the GPU stages on Kaggle (free T4 GPUs, fp16) from this machine.

  python -m scripts.kaggle_run stage smoke            # upload the inputs a task needs to <hf-user>/dhanground-work (private)
  python -m scripts.kaggle_run push smoke             # push + run a Kaggle GPU kernel pinned to the current commit
  python -m scripts.kaggle_run push sft-seed0         # A1 s0 on GPU 0 and A2 s0 on GPU 1, in parallel (AlignTune)
  python -m scripts.kaggle_run status smoke | logs smoke

The kernel reads HF_TOKEN from a Kaggle *secret* (Add-ons > Secrets, attached to that notebook once in the
UI); tokens are never put in code, metadata or datasets. Adapters are pushed to <hf-user>/dhanground-<name>.
Needs KAGGLE_API_TOKEN (Kaggle CLI) and HF_TOKEN in the environment (.env).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/DB00708/dhanground"
WORK_REPO = "dhanground-work"
SHAPE = "NvidiaTeslaT4"
TASKS = {
    "smoke": {
        "stage": {"data/raw/train.jsonl": "smoke/train.jsonl"},
        "patterns": ["smoke/*"],
        "steps": [["python", "-m", "scripts.smoke_train", "--rows", "data/smoke/train.jsonl",
                   "--n", "48", "--steps", "24", "--per-device", "2"]],
        "parallel": False,
    },
    "sft-seed0": {
        "stage": {"data/arms/A1/train.jsonl": "arms/A1/train.jsonl", "data/arms/A2/train.jsonl": "arms/A2/train.jsonl"},
        "patterns": ["arms/A1/*", "arms/A2/*"],
        "steps": [["python", "-m", "adapters.aligntune_sft", "--arm", "A1", "--seed", "0", "--backend", "aligntune",
                   "--hub-org", "{hf_user}"],
                  ["python", "-m", "adapters.aligntune_sft", "--arm", "A2", "--seed", "0", "--backend", "aligntune",
                   "--hub-org", "{hf_user}"]],
        "parallel": True,
    },
}

KERNEL_SCRIPT = r'''
import os, subprocess, sys, json
def sh(cmd, **kw):
    print("+ " + (cmd if isinstance(cmd, str) else " ".join(cmd)), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)
print("python", sys.version, flush=True)
if sys.version_info < (3, 11):
    sys.exit("KAGGLE_PYTHON_TOO_OLD: aligntune 0.1.33 needs Python >= 3.11")
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    sys.exit("NO_HF_TOKEN_SECRET: attach a Kaggle secret named HF_TOKEN to this notebook (Add-ons > Secrets): %r" % e)
os.environ.update({"DHANGROUND_PRECISION": "fp16", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
CFG = json.loads(__CFG__)
root = "/kaggle/working/dhanground"
sh(["git", "clone", "-q", CFG["repo_url"], root]); os.chdir(root); sh(["git", "checkout", "-q", CFG["sha"]])
sh([sys.executable, "-m", "pip", "install", "-q", "aligntune==0.1.33", "curatorkit==0.1.24", "pydantic", "PyYAML"])
sh([sys.executable, "-c", "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"])
from huggingface_hub import snapshot_download
snapshot_download(CFG["work_repo"], repo_type="dataset", local_dir="data", allow_patterns=CFG["patterns"])
steps = [[sys.executable if a == "python" else a for a in s] for s in CFG["steps"]]
if CFG["parallel"]:
    procs = []
    for gpu, s in enumerate(steps):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = open(f"/kaggle/working/step{gpu}.log", "w")
        print(f"GPU {gpu}: " + " ".join(s), flush=True)
        procs.append((s, subprocess.Popen(s, env=env, stdout=log, stderr=subprocess.STDOUT), f"/kaggle/working/step{gpu}.log"))
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


def env_or_exit(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is not set (put it in .env and `set -a; . ./.env`)")
    return v


def hf_user() -> str:
    from huggingface_hub import whoami

    return whoami(token=env_or_exit("HF_TOKEN"))["name"]


def git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    pushed = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True, text=True).stdout
    if "origin/main" not in pushed:
        raise SystemExit(f"commit {sha[:7]} is not on origin/main: push it first, the kernel clones the public repo")
    return sha


def kernel_slug(task: str) -> str:
    return f"dhanground-{task}"


def build_kernel(task: str, sha: str, user: str, kaggle_user: str, folder: Path) -> Path:
    t = TASKS[task]
    cfg = {"repo_url": REPO_URL, "sha": sha, "work_repo": f"{user}/{WORK_REPO}", "patterns": t["patterns"],
           "parallel": t["parallel"], "steps": [[a.format(hf_user=user) for a in s] for s in t["steps"]]}
    script = KERNEL_SCRIPT.replace("__CFG__", repr(json.dumps(cfg)))
    (folder / "run.py").write_text(script, encoding="utf-8")
    meta = {"id": f"{kaggle_user}/{kernel_slug(task)}", "title": kernel_slug(task), "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True, "machine_shape": SHAPE,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


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


def stage(task: str) -> None:
    from huggingface_hub import create_repo, upload_file

    user, token = hf_user(), env_or_exit("HF_TOKEN")
    repo = f"{user}/{WORK_REPO}"
    create_repo(repo, repo_type="dataset", private=True, exist_ok=True, token=token)
    for src, dst in TASKS[task]["stage"].items():
        if not Path(src).exists():
            raise SystemExit(f"missing {src}: run the pipeline stage that writes it first")
        upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="dataset", token=token)
        print(f"staged {src} -> {repo}/{dst}")


def push(task: str) -> None:
    sha, user, kuser = git_sha(), hf_user(), kaggle_user()
    with tempfile.TemporaryDirectory() as d:
        build_kernel(task, sha, user, kuser, Path(d))
        print(kaggle("kernels", "push", "-p", d))
    print(f"kernel https://www.kaggle.com/code/{kuser}/{kernel_slug(task)} at {sha[:7]} (fp16, {SHAPE})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["stage", "push", "status", "logs"])
    ap.add_argument("task", choices=sorted(TASKS))
    a = ap.parse_args()
    if a.command == "stage":
        stage(a.task)
    elif a.command == "push":
        push(a.task)
    elif a.command == "status":
        print(kaggle("kernels", "status", f"{kaggle_user()}/{kernel_slug(a.task)}"))
    else:
        out = Path("data/kaggle") / a.task
        out.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{kaggle_user()}/{kernel_slug(a.task)}", "-p", str(out), "-o"))
        for f in sorted(out.glob("*.log")):
            print(f"--- {f.name}\n" + f.read_text(encoding="utf-8", errors="replace")[-4000:])


if __name__ == "__main__":
    main()
