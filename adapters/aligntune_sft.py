"""SFT for arms A1 and A2 through AlignTune (TRL backend), LoRA r=16 in bf16.

Every row is pre-rendered with the base model's own chat template into one `text` field, the same
render serving uses, so train and serve see byte-identical prompts.
Fallback when AlignTune breaks on the template: --backend trl (plain TRL, completion-only loss).

Usage: python -m adapters.aligntune_sft --arm A2 --seed 0 [--hub-org your-org] [--backend aligntune|trl]
Resume (e.g. on Kaggle after HF credits run out): add --backend trl --resume-from-hub <org>/dhanground-A2-s0-<sha>.
AlignTune 0.1.33 cannot resume: its SFT train() builds the HF trainer internally and calls .train() with no
resume_from_checkpoint, so only the trl backend resumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.common import CheckpointSync, adapter_name, load_configs, push_adapter, restore_latest_checkpoint


def render_rows(tokenizer, src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / "train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for line in src.open(encoding="utf-8"):
            row = json.loads(line)
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return dst_dir


def train_aligntune(base: str, data_dir: Path, out: Path, lora: dict, seed: int) -> None:
    from aligntune.core.backend_factory import create_sft_trainer

    sft = lora["sft"]
    trainer = create_sft_trainer(
        model_name=base,
        dataset_name=str(data_dir),
        backend="trl",
        output_dir=str(out),
        num_epochs=sft["epochs"],
        batch_size=4,
        gradient_accumulation_steps=sft["effective_batch_size"] // 4,
        learning_rate=sft["lr"],
        lr_scheduler=sft["scheduler"],
        max_seq_length=sft["max_len"],
        use_peft=True,
        lora_r=lora["lora"]["r"],
        lora_alpha=lora["lora"]["alpha"],
        lora_dropout=lora["lora"]["dropout"],
        lora_target_modules=lora["lora"]["target_modules"],
        precision="bf16",
        seed=seed,
        save_steps=sft["checkpoint_every_steps"],
        dataset_text_field="text",
        text_column="text",
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer.train()
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(out))


def train_trl(base: str, rows_path: Path, out: Path, lora: dict, seed: int) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sft = lora["sft"]
    rows = [json.loads(l) for l in rows_path.open(encoding="utf-8")]
    ds = Dataset.from_list([{"prompt": r["messages"][:1], "completion": r["messages"][1:]} for r in rows])
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    args = SFTConfig(
        output_dir=str(out), num_train_epochs=sft["epochs"], per_device_train_batch_size=4,
        gradient_accumulation_steps=sft["effective_batch_size"] // 4, learning_rate=sft["lr"],
        lr_scheduler_type=sft["scheduler"], max_length=sft["max_len"], bf16=True, seed=seed,
        save_steps=sft["checkpoint_every_steps"], logging_steps=10, report_to="none",
        completion_only_loss=True, gradient_checkpointing=True,
    )
    peft_cfg = LoraConfig(r=lora["lora"]["r"], lora_alpha=lora["lora"]["alpha"], lora_dropout=lora["lora"]["dropout"],
                          target_modules=lora["lora"]["target_modules"], task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train(resume_from_checkpoint=any(out.glob("checkpoint-*")) or None)
    trainer.save_model(str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A1", "A2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["aligntune", "trl"], default="aligntune")
    ap.add_argument("--hub-org", default=None, help="push checkpoints and the final adapter here")
    ap.add_argument("--resume-from-hub", default=None, metavar="REPO_ID",
                    help="restore the latest checkpoint from this Hub repo and resume (trl backend only)")
    a = ap.parse_args()
    if a.resume_from_hub and a.backend == "aligntune":
        raise SystemExit("AlignTune 0.1.33 cannot resume from a checkpoint (its train() takes no "
                         "resume_from_checkpoint). Rerun with --backend trl --resume-from-hub ... to resume.")

    arms, lora = load_configs()
    base = arms["base_model"]
    name = adapter_name(a.arm, a.seed)
    out = Path("checkpoints") / name
    rows_path = Path("data/arms") / a.arm / "train.jsonl"
    repo = a.resume_from_hub or (f"{a.hub_org}/dhanground-{name}" if a.hub_org else None)
    restored = restore_latest_checkpoint(a.resume_from_hub, out) if a.resume_from_hub else None
    if a.resume_from_hub:
        print(f"resuming from {restored}" if restored else f"no complete checkpoint in {a.resume_from_hub}; starting fresh")
    elif a.backend == "aligntune" and repo:
        print("note: AlignTune runs cannot resume; use --backend trl if this GPU session may die")

    with CheckpointSync(out, repo, already_sent=restored):
        if a.backend == "aligntune":
            from transformers import AutoTokenizer

            data_dir = render_rows(AutoTokenizer.from_pretrained(base), rows_path, Path("data/arms") / a.arm / "rendered")
            train_aligntune(base, data_dir, out, lora, a.seed)
        else:
            train_trl(base, rows_path, out, lora, a.seed)
    (out / "dhanground.json").write_text(json.dumps({"arm": a.arm, "seed": a.seed, "base": base,
                                                     "backend": a.backend, "name": name}, indent=2))
    if repo:
        push_adapter(out, repo)
    print(f"adapter: {out}" + (f" (pushed to {repo})" if repo else ""))


if __name__ == "__main__":
    main()
