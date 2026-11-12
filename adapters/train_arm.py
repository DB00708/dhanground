"""Train every seed of an SFT arm: the first seed through AlignTune, the rest through plain TRL.

AlignTune 0.1.33 cannot resume a run, so it trains only the seed that carries the CuratorKIT -> AlignTune ->
AuditKIT chain; the other seeds use --backend trl, which resumes from a Hub checkpoint. Both backends train on
the same rows, chat-template render, loss mask and hyperparameters (see adapters/aligntune_sft.py).

Usage: python -m adapters.train_arm --arm A2 --hub-org your-org [--seeds 0 1 2] [--aligntune-seeds 0] [--dry-run]
If one seed dies mid-run, resume just that seed with aligntune_sft --backend trl --resume-from-hub ...,
then rerun this command with --seeds set to the seeds still missing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def seed_plan(seeds: list[int], aligntune_seeds: list[int]) -> list[tuple[int, str]]:
    return [(s, "aligntune" if s in aligntune_seeds else "trl") for s in seeds]


def seed_command(arm: str, seed: int, backend: str, hub_org: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "adapters.aligntune_sft", "--arm", arm, "--seed", str(seed), "--backend", backend]
    return cmd + (["--hub-org", hub_org] if hub_org else [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A1", "A2"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--aligntune-seeds", type=int, nargs="*", default=[0],
                    help="seeds trained through AlignTune (cannot resume); the rest use --backend trl")
    ap.add_argument("--hub-org", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    for seed, backend in seed_plan(a.seeds, a.aligntune_seeds):
        cmd = seed_command(a.arm, seed, backend, a.hub_org)
        print(" ".join(cmd[1:]), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
