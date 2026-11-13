"""SFT for arms A1 and A2 through AlignTune (TRL backend), LoRA r=16 in bf16.

Both backends train on the same thing: each row's `messages` list (user turn + assistant JSON), rendered by
TRL's SFTTrainer with the base model's own chat template, `enable_thinking=False` in the template kwargs
(AlignTune adds that to every row, so the trl backend adds it too), and the same loss mask
(`sft.assistant_only_loss` in configs/lora.yaml). Before any GPU work, `check_render` asserts on the real
tokenizer that this training render starts with the exact prompt serving renders.

Usage: python -m adapters.aligntune_sft --arm A2 --seed 0 [--hub-org your-org] [--backend aligntune|trl]
All seeds of an arm (seed 0 via AlignTune, the rest via trl): python -m adapters.train_arm --arm A2 --hub-org org
Resume (e.g. on Kaggle after HF credits run out): --backend trl --resume-from-hub <org>/dhanground-A2-s1-<sha>.
AlignTune 0.1.33 cannot resume: its TRLSFTTrainer.train() builds the HF trainer itself and calls
self.trainer.train() with no resume_from_checkpoint, so only the trl backend resumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapters.common import (CheckpointSync, adapter_name, load_configs, name_from_repo, push_adapter,
                             restore_latest_checkpoint, resume_checkpoint)

# AlignTune's SystemPromptInjector puts this on every messages row; TRL forwards it to apply_chat_template.
TEMPLATE_KWARGS = {"enable_thinking": False}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def trl_rows(rows: list[dict]) -> list[dict]:
    """What both backends hand to TRL's SFTTrainer (AlignTune builds the same from write_aligntune_rows)."""
    return [{"messages": r["messages"], "chat_template_kwargs": dict(TEMPLATE_KWARGS)} for r in rows]


def write_aligntune_rows(rows: list[dict], dst_dir: Path) -> Path:
    """messages-only JSONL: AlignTune's 'sft' task keeps a messages column as is and adds TEMPLATE_KWARGS."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    with (dst_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    return dst_dir


def render_mismatches(tokenizer, rows: list[dict], chat_template: str | None = None) -> list[str]:
    """row_ids whose training render differs from serving.

    Training text = the template TRL trains with (patched for assistant masks when chat_template is given)
    over the full conversation with TEMPLATE_KWARGS. It must equal the plain template render and start with
    the prompt serving builds (user turn + generation prompt, serving/engine.py).
    """
    bad = []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        plain = tokenizer.apply_chat_template(msgs, tokenize=False)
        train = tokenizer.apply_chat_template(msgs, tokenize=False, chat_template=chat_template, **TEMPLATE_KWARGS)
        serve = tokenizer.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        if train != plain or not train.startswith(serve):
            bad.append(r.get("row_id", f"#{i}"))
    return bad


def check_render(tokenizer, rows: list[dict], assistant_only_loss: bool) -> None:
    template = None
    if assistant_only_loss:
        from trl.chat_template_utils import get_training_chat_template

        try:
            template = get_training_chat_template(tokenizer)
        except ValueError as e:
            raise SystemExit(f"{e}\nTRL cannot mask assistant tokens with this chat template. Set "
                             "sft.assistant_only_loss: false in configs/lora.yaml (both backends then train on "
                             "the full sequence) and note it in the run log.")
    bad = render_mismatches(tokenizer, rows, template)
    if bad:
        raise SystemExit(f"{len(bad)} rows render differently in training than in serving (first: {bad[0]}). "
                         "Fix the template or the rows before training.")


def grad_accum(sft: dict) -> int:
    n, b = sft["effective_batch_size"], sft["per_device_batch_size"]
    if n % b:
        raise SystemExit(f"sft.effective_batch_size {n} is not a multiple of per_device_batch_size {b}")
    return n // b


def set_loss_type(trainer, loss_type: str) -> None:
    """Give AlignTune the same loss_type as the trl backend.

    AlignTune 0.1.33 checks loss_type in {nll, dft} only in TrainingConfig.__post_init__; build_sft_config reads
    config.train.loss_type at train() time and passes it to trl's SFTConfig, which also accepts chunked_nll.
    """
    trainer.config.train.loss_type = loss_type


def train_aligntune(base: str, data_dir: Path, out: Path, lora: dict, seed: int, n_rows: int) -> None:
    from aligntune.core.backend_factory import create_sft_trainer

    sft = lora["sft"]
    trainer = create_sft_trainer(
        model_name=base,
        dataset_name=str(data_dir),
        backend="trl",
        task_type="sft",
        output_dir=str(out),
        num_epochs=sft["epochs"],
        batch_size=sft["per_device_batch_size"],
        gradient_accumulation_steps=grad_accum(sft),
        loss_type="nll",  # AlignTune's TrainingConfig accepts only nll/dft here; set_loss_type applies ours
        learning_rate=sft["lr"],
        lr_scheduler=sft["scheduler"],
        warmup_ratio=sft["warmup_ratio"],
        weight_decay=sft["weight_decay"],
        optimizer="adamw_torch",
        max_seq_length=sft["max_len"],
        assistant_only_loss=sft["assistant_only_loss"],
        use_peft=True,
        lora_r=lora["lora"]["r"],
        lora_alpha=lora["lora"]["alpha"],
        lora_dropout=lora["lora"]["dropout"],
        lora_target_modules=lora["lora"]["target_modules"],
        precision="bf16",
        seed=seed,
        save_steps=sft["checkpoint_every_steps"],
        report_to="none",
        gradient_checkpointing=True,
    )
    set_loss_type(trainer, sft["loss_type"])
    trainer.setup_model()
    trainer.setup_data()
    # CuratorKIT's schema gate runs inside setup_data; a silently dropped row would make the backends differ.
    got = len(trainer.train_dataset)
    if got != n_rows:
        raise SystemExit(f"AlignTune kept {got} of {n_rows} rows after its CuratorKIT gate; "
                         "fix the rows or train this seed with --backend trl")
    trainer.dataset = trainer.train_dataset
    trainer.train()
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(out))


def train_trl(base: str, rows: list[dict], out: Path, lora: dict, seed: int) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sft = lora["sft"]
    ds = Dataset.from_list(trl_rows(rows))
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    args = SFTConfig(  # mirrors what AlignTune 0.1.33's build_sft_config sets for train_aligntune above
        output_dir=str(out), num_train_epochs=sft["epochs"], per_device_train_batch_size=sft["per_device_batch_size"],
        gradient_accumulation_steps=grad_accum(sft), loss_type=sft["loss_type"], learning_rate=sft["lr"],
        lr_scheduler_type=sft["scheduler"], warmup_steps=sft["warmup_ratio"], weight_decay=sft["weight_decay"],
        optim="adamw_torch", max_length=sft["max_len"], bf16=True, seed=seed, padding_free=False,
        save_steps=sft["checkpoint_every_steps"], logging_steps=10, report_to="none",
        assistant_only_loss=sft["assistant_only_loss"], gradient_checkpointing=True,
    )
    peft_cfg = LoraConfig(r=lora["lora"]["r"], lora_alpha=lora["lora"]["alpha"], lora_dropout=lora["lora"]["dropout"],
                          target_modules=lora["lora"]["target_modules"], task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    ckpt = resume_checkpoint(out)
    print(f"resuming from {ckpt}" if ckpt else "starting from step 0", flush=True)
    trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None)
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
    # A resumed run keeps the original run's name (its git sha), whatever commit this box has checked out.
    name = name_from_repo(a.resume_from_hub, a.arm, a.seed) if a.resume_from_hub else adapter_name(a.arm, a.seed)
    out = Path("checkpoints") / name
    rows = load_rows(Path("data/arms") / a.arm / "train.jsonl")
    repo = a.resume_from_hub or (f"{a.hub_org}/dhanground-{name}" if a.hub_org else None)

    from transformers import AutoTokenizer

    check_render(AutoTokenizer.from_pretrained(base), rows, lora["sft"]["assistant_only_loss"])
    if a.resume_from_hub:
        restored = restore_latest_checkpoint(a.resume_from_hub, out)
        print(f"restored {restored} from {a.resume_from_hub}" if restored
              else f"no complete checkpoint in {a.resume_from_hub}")
    elif a.backend == "aligntune" and repo:
        print("note: AlignTune runs cannot resume; if this GPU session may die, use --backend trl")

    with CheckpointSync(out, repo):
        if a.backend == "aligntune":
            data_dir = write_aligntune_rows(rows, Path("data/arms") / a.arm / "aligntune")
            train_aligntune(base, data_dir, out, lora, a.seed, len(rows))
        else:
            train_trl(base, rows, out, lora, a.seed)
    (out / "dhanground.json").write_text(json.dumps({"arm": a.arm, "seed": a.seed, "base": base,
                                                     "backend": a.backend, "name": name}, indent=2))
    if repo:
        push_adapter(out, repo)
    print(f"adapter: {out}" + (f" (pushed to {repo})" if repo else ""))


if __name__ == "__main__":
    main()
