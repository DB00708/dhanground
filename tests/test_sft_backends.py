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
