"""Arm A3: DPO on top of the A2 adapter, with A2 itself as the reference policy.

Runs through plain TRL (the design's labeled DPO fallback). The A2 adapter is loaded as the trainable
"default" adapter; TRL's DPOTrainer (trl 1.7.1, pinned by aligntune 0.1.33) then adds a frozen copy named
"ref" and computes reference log-probs with it. So A3 stays a LoRA adapter on the same base and hot-swaps
with A1 and A2 at serve time. (trl 1.7.1 has no model_adapter_name/ref_adapter_name DPO options.)

Usage: python -m adapters.aligntune_dpo --a2 checkpoints/A2-s0-<sha> --pairs data/arms/A3_pairs.jsonl [--hub-org org]
Resume after a dead session: add --resume-from-hub <org>/dhanground-A3-s0-<sha>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.common import (CheckpointSync, adapter_name, load_configs, name_from_repo, push_adapter,
                             restore_latest_checkpoint, resume_checkpoint)


def restore_policy(model, ckpt: Path) -> None:
    """Load the trained "default" adapter from a checkpoint before trainer.train(resume_from_checkpoint=...).

    A checkpoint holds "default" at its root and "ref" in ref/. transformers' resume loads only adapter
    subfolders when any exist, so without this the policy would silently restart from A2 while the
    optimizer and step count resume.
    """
    model.load_adapter(str(ckpt), adapter_name="default", is_trainable=True)
    model.set_adapter("default")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", required=True, help="A2 adapter path or Hub repo id")
    ap.add_argument("--pairs", type=Path, default=Path("data/arms/A3_pairs.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hub-org", default=None)
    ap.add_argument("--resume-from-hub", default=None, metavar="REPO_ID",
                    help="restore the latest checkpoint from this Hub repo and resume")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    arms, lora = load_configs()
    base, dpo = arms["base_model"], lora["dpo"]
    name = name_from_repo(a.resume_from_hub, "A3", a.seed) if a.resume_from_hub else adapter_name("A3", a.seed)
    out = Path("checkpoints") / name
    repo = a.resume_from_hub or (f"{a.hub_org}/dhanground-{name}" if a.hub_org else None)
    if a.resume_from_hub:
        restored = restore_latest_checkpoint(a.resume_from_hub, out)
        print(f"resuming from {restored}" if restored else f"no complete checkpoint in {a.resume_from_hub}; starting fresh")

    rows = [json.loads(l) for l in a.pairs.open(encoding="utf-8") if l.strip()]
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, a.a2, is_trainable=True)  # adapter "default"

    args = DPOConfig(
        output_dir=str(out), num_train_epochs=dpo["epochs"], learning_rate=dpo["lr"], beta=dpo["beta"],
        per_device_train_batch_size=2, gradient_accumulation_steps=8, bf16=True, seed=a.seed,
        max_length=lora["sft"]["max_len"] + 384, save_steps=dpo["checkpoint_every_steps"], logging_steps=5,
        report_to="none", gradient_checkpointing=True,
    )
    with CheckpointSync(out, repo):
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)  # adds "ref"
        ckpt = resume_checkpoint(out)
        if ckpt:
            restore_policy(trainer.model, ckpt)
        trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None)

    final = out / "final"
    trainer.model.save_pretrained(str(final), selected_adapters=["default"])  # "default" saves at the folder root
    (final / "dhanground.json").write_text(json.dumps({"arm": "A3", "seed": a.seed, "base": base, "init": a.a2,
                                                       "backend": "trl-dpo", "name": name}, indent=2))
    if repo:
        push_adapter(final, repo)
    print(f"adapter: {final}" + (f" (pushed to {repo})" if repo else ""))


if __name__ == "__main__":
    main()
