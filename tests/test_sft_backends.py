"""Both SFT backends must train on the same text and the same seed split (adapters/aligntune_sft.py)."""
import json
import types

from adapters.aligntune_dpo import restore_policy
from adapters.aligntune_sft import (TEMPLATE_KWARGS, load_rows, render_mismatches, trl_rows,
                                    write_aligntune_rows)
from adapters.train_arm import seed_command, seed_plan

ROWS = [
    {"row_id": "r1", "messages": [{"role": "user", "content": "q1 <passages>"}, {"role": "assistant", "content": "{}"}]},
    {"row_id": "r2", "messages": [{"role": "user", "content": "q2 ₹"}, {"role": "assistant", "content": "{\"a\": 1}"}]},
]


class FakeTok:
    """Cohere-like template; `thinking_changes_text` mimics a template that reads enable_thinking."""

    def __init__(self, thinking_changes_text=False, patched_differs=False):
        self.thinking_changes_text, self.patched_differs = thinking_changes_text, patched_differs

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, chat_template=None, **kw):
        out = "<BOS>" + "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in msgs)
        if add_generation_prompt:
            out += "<|assistant|>"
        if self.thinking_changes_text and kw.get("enable_thinking") is False:
            out = out.replace("<|assistant|>", "<|assistant|><nothink>")
        if self.patched_differs and chat_template is not None:
            out += " "
        return out


def test_backends_get_identical_messages(tmp_path):
    src = tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS), encoding="utf-8")
    rows = load_rows(src)
    d = write_aligntune_rows(rows, tmp_path / "aligntune")
    aligntune = [json.loads(l) for l in (d / "train.jsonl").open(encoding="utf-8")]
    assert list(aligntune[0]) == ["messages"]
    # AlignTune's SystemPromptInjector adds TEMPLATE_KWARGS to every messages row; trl_rows does the same.
    assert trl_rows(rows) == [{"messages": r["messages"], "chat_template_kwargs": TEMPLATE_KWARGS} for r in aligntune]


def test_render_check_passes_for_consistent_template():
    assert render_mismatches(FakeTok(), ROWS) == []
    assert render_mismatches(FakeTok(), ROWS, chat_template="patched") == []


def test_render_check_flags_enable_thinking_and_patched_template_drift():
    assert render_mismatches(FakeTok(thinking_changes_text=True), ROWS) == ["r1", "r2"]
    assert render_mismatches(FakeTok(patched_differs=True), ROWS, chat_template="patched") == ["r1", "r2"]


def test_seed_plan_first_seed_aligntune_rest_trl():
    assert seed_plan([0, 1, 2], [0]) == [(0, "aligntune"), (1, "trl"), (2, "trl")]
    assert seed_plan([1, 2], [0]) == [(1, "trl"), (2, "trl")]
    cmd = seed_command("A2", 1, "trl", "org")
    assert cmd[1:] == ["-m", "adapters.aligntune_sft", "--arm", "A2", "--seed", "1", "--backend", "trl",
                       "--hub-org", "org"]


def test_dpo_resume_reloads_policy_adapter(tmp_path):
    calls = []
    model = types.SimpleNamespace(load_adapter=lambda *a, **k: calls.append(("load", a, k)),
                                  set_adapter=lambda n: calls.append(("set", n)))
    restore_policy(model, tmp_path / "checkpoint-400")
    assert calls == [("load", (str(tmp_path / "checkpoint-400"),), {"adapter_name": "default", "is_trainable": True}),
                     ("set", "default")]


# --- seed provenance in eval results; shared loss and batch settings ---

import sys

import pytest
import yaml

from adapters.aligntune_sft import grad_accum
from evals import summarize
from evals.run_eval import adapter_provenance


def test_adapter_provenance_local_hub_and_base(monkeypatch, tmp_path):
    (tmp_path / "dhanground.json").write_text(json.dumps({"backend": "aligntune", "name": "A2-s0-abc"}))
    assert adapter_provenance("A2", str(tmp_path)) == {"backend": "aligntune", "adapter_name": "A2-s0-abc"}
    assert adapter_provenance("A0", None) == {"backend": "base", "adapter_name": None}

    hub_file = tmp_path / "hub.json"
    hub_file.write_text(json.dumps({"backend": "trl", "name": "A2-s1-abc"}))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=lambda repo, f: str(hub_file)))
    assert adapter_provenance("A2", "org/dhanground-A2-s1-abc") == {"backend": "trl", "adapter_name": "A2-s1-abc"}

    def missing(repo, f):
        raise OSError("404")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=missing))
    assert adapter_provenance("A2", "org/nope")["backend"] == "unknown"


def test_summary_reports_backend_per_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(summarize, "RESULTS", tmp_path)
    for seed, backend in [(0, "aligntune"), (1, "trl")]:
        (tmp_path / f"A2_s{seed}_test.json").write_text(json.dumps({"backend": backend}))
    assert summarize.seed_backends("A2", [0, 1, 2], "test") == {"0": "aligntune", "1": "trl", "2": "unknown"}


def test_lora_yaml_settings_valid_for_both_backends():
    sft = yaml.safe_load(open("configs/lora.yaml", encoding="utf-8"))["sft"]
    assert sft["loss_type"] in {"nll", "dft", "chunked_nll"}
    assert grad_accum(sft) * sft["per_device_batch_size"] == sft["effective_batch_size"]
    with pytest.raises(SystemExit):
        grad_accum({"effective_batch_size": 16, "per_device_batch_size": 3})


def test_aligntune_gets_same_loss_type_as_trl_backend():
    from adapters.aligntune_sft import set_loss_type

    train = types.SimpleNamespace(loss_type="nll", extra_params={"loss_type": "nll", "seed": 0})
    trainer = types.SimpleNamespace(config=types.SimpleNamespace(train=train))
    sft = yaml.safe_load(open("configs/lora.yaml", encoding="utf-8"))["sft"]
    set_loss_type(trainer, sft["loss_type"])
    assert train.loss_type == sft["loss_type"] == "chunked_nll"
    # extra_params override config.train in AlignTune's build_sft_config, so no loss_type may remain there
    assert "loss_type" not in train.extra_params and train.extra_params == {"seed": 0}
    src = open("adapters/aligntune_sft.py", encoding="utf-8").read()
    aligntune_call = src[src.index("create_sft_trainer("):src.index("set_loss_type(trainer")]
    assert "loss_type" not in aligntune_call  # passing it would land in extra_params


def test_run_eval_warns_when_adapter_is_for_another_seed():
    from evals.run_eval import seed_mismatch

    assert seed_mismatch("A2", 1, "A2-s1-abc") is None
    assert seed_mismatch("A0", 2, None) is None
    assert "A2-s0-abc" in seed_mismatch("A2", 1, "A2-s0-abc")
