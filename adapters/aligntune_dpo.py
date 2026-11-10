"""Arm A3: DPO on top of the A2 adapter, with A2 itself as the reference policy.

Runs through plain TRL (the design's labeled DPO fallback): the A2 adapter is loaded twice,
"train" (trainable) and "reference" (frozen), so A3 stays a LoRA adapter on the same base and
hot-swaps with A1 and A2 at serve time.

Usage: python -m adapters.aligntune_dpo --a2 checkpoints/A2-s0-<sha> --pairs data/arms/A3_pairs.jsonl [--hub-org org]
Resume after a dead session: add --resume-from-hub <org>/dhanground-A3-s0-<sha>.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from adapters.common import CheckpointSync, adapter_name, load_configs, push_adapter, restore_latest_checkpoint


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
    name = adapter_name("A3", a.seed)
    out = Path("checkpoints") / name
    repo = a.resume_from_hub or (f"{a.hub_org}/dhanground-{name}" if a.hub_org else None)
    restored = restore_latest_checkpoint(a.resume_from_hub, out) if a.resume_from_hub else None
    if a.resume_from_hub:
        print(f"resuming from {restored}" if restored else f"no complete checkpoint in {a.resume_from_hub}; starting fresh")

    rows = [json.loads(l) for l in a.pairs.open(encoding="utf-8")]
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, a.a2, adapter_name="train", is_trainable=True)
    model.load_adapter(a.a2, adapter_name="reference")

    args = DPOConfig(
        output_dir=str(out), num_train_epochs=dpo["epochs"], learning_rate=dpo["lr"], beta=dpo["beta"],
        per_device_train_batch_size=2, gradient_accumulation_steps=8, bf16=True, seed=a.seed,
        max_length=lora["sft"]["max_len"] + 384, save_steps=dpo["checkpoint_every_steps"], logging_steps=5,
        report_to="none", model_adapter_name="train", ref_adapter_name="reference", gradient_checkpointing=True,
    )
    with CheckpointSync(out, repo, already_sent=restored):
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
        trainer.train(resume_from_checkpoint=any(out.glob("checkpoint-*")) or None)

    final = out / "final"
    model.save_pretrained(str(final), selected_adapters=["train"])
    nested = final / "train"
    if nested.exists():
        for f in nested.iterdir():
            shutil.move(str(f), final / f.name)
        nested.rmdir()
    (final / "dhanground.json").write_text(json.dumps({"arm": "A3", "seed": a.seed, "base": base, "init": a.a2,
                                                       "backend": "trl-dpo", "name": name}, indent=2))
    if repo:
        push_adapter(final, repo)
    print(f"adapter: {final}" + (f" (pushed to {repo})" if repo else ""))


if __name__ == "__main__":
    main()
