"""SFT smoke test on a GPU box (the design's first gate): loss falls, no NaN/inf, template pre-flight passes.

Trains the real LoRA setup (same rows render, loss mask, LoRA config and run-wide precision as
adapters/aligntune_sft.py) for a few steps on ~50 rows, and prints a JSON verdict with peak GPU memory so
the real runs can be sized. Exit code 0 = go, 1 = no-go.

Usage: python -m scripts.smoke_train --rows data/raw/train.jsonl --n 48 --steps 24 --per-device 2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from adapters.aligntune_sft import check_render, load_rows, trl_rows
from adapters.common import load_configs, precision, torch_dtype


def sft_rows(path: Path, n: int) -> list[dict]:
    """Rows with `messages`; raw rows are rendered with the same builder as the arm datasets."""
    rows = load_rows(path)
    if rows and "messages" not in rows[0]:
        from pipeline.build_arms import to_sft

        rows = [to_sft(r) for r in rows if r.get("answerable")]
    return rows[:n]


def verdict(losses: list[float], window: int = 4) -> dict:
    finite = bool(losses) and all(math.isfinite(x) for x in losses)
    k = max(1, min(window, len(losses) // 2))
    first, last = (sum(losses[:k]) / k, sum(losses[-k:]) / k) if finite else (None, None)
    falls = finite and len(losses) >= 2 and last < first
    return {"go": bool(finite and falls), "finite": finite, "loss_first": first, "loss_last": last,
            "n_logged": len(losses)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=Path("data/raw/train.jsonl"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--per-device", type=int, default=2)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    arms, lora = load_configs()
    base, sft, p = arms["base_model"], lora["sft"], precision(lora)
    rows = sft_rows(a.rows, a.n)
    tok = AutoTokenizer.from_pretrained(base)
    check_render(tok, rows, sft["assistant_only_loss"])

    losses: list[float] = []

    class Collect(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append(float(logs["loss"]))

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch_dtype(p))
    args = SFTConfig(
        output_dir="checkpoints/smoke", max_steps=a.steps, per_device_train_batch_size=a.per_device,
        gradient_accumulation_steps=1, learning_rate=sft["lr"], lr_scheduler_type="constant",
        loss_type=sft["loss_type"], optim="adamw_torch", max_length=sft["max_len"],
        bf16=p == "bf16", fp16=p == "fp16", seed=0, padding_free=False, logging_steps=1, save_strategy="no",
        report_to="none", assistant_only_loss=sft["assistant_only_loss"], gradient_checkpointing=True,
    )
    peft_cfg = LoraConfig(r=lora["lora"]["r"], lora_alpha=lora["lora"]["alpha"], lora_dropout=lora["lora"]["dropout"],
                          target_modules=lora["lora"]["target_modules"], task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(trl_rows(rows)),
                         processing_class=tok, peft_config=peft_cfg, callbacks=[Collect()])
    trainer.train()

    out = verdict(losses)
    out.update(precision=p, rows=len(rows), steps=a.steps, per_device=a.per_device, losses=losses,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    print("SMOKE_VERDICT " + json.dumps(out), flush=True)
    raise SystemExit(0 if out["go"] else 1)


if __name__ == "__main__":
    main()
