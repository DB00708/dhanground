import json

import pytest
import yaml

from evals.tune_tau import choose_tau, main, replace_tau

CONFIG = """k: 4
hybrid:
  bm25_weight: 0.5
  embedding_weight: 0.5
  embedding_model: null   # pick a multilingual embedding model on Nov 10
tau: null                 # set on dev so over-refusal stays under 10%
index_version: null       # e.g. rbi-2026-11-09, stamped on every answer
"""


def test_tau_gates_at_most_max_share_of_answerable():
    ans = [i / 100 for i in range(1, 101)]  # 0.01 .. 1.00
    res = choose_tau(ans, [0.0, 0.02, 0.5, 0.9], max_gated=0.05)
    assert res["tau"] == pytest.approx(0.06)  # 5 scores (0.01..0.05) below it; any higher gates 6
    assert res["answerable_gated"] == pytest.approx(0.05)
    assert res["unanswerable_caught"] == pytest.approx(0.5)
    assert [c["quantile"] for c in res["curve"]][0] == 0.0
    assert all(c["answerable_gated"] <= c["quantile"] + 1e-9 for c in res["curve"])


def test_all_answerable_high_catches_low_unanswerable():
    res = choose_tau([0.9, 0.91, 0.95], [0.1, 0.2], max_gated=0.05)
    assert res["tau"] == 0.9 and res["answerable_gated"] == 0.0 and res["unanswerable_caught"] == 1.0


def test_ties_never_exceed_budget():
    res = choose_tau([0.3] + [0.5] * 9, [0.4], max_gated=0.2)  # 2 allowed, but 0.5 ties
    assert res["tau"] == 0.5 and res["answerable_gated"] == pytest.approx(0.1)
    assert res["unanswerable_caught"] == 1.0


def test_empty_unanswerable_and_empty_answerable():
    res = choose_tau([0.2, 0.4], [], max_gated=0.05)
    assert res["tau"] == 0.2 and res["unanswerable_caught"] is None
    with pytest.raises(ValueError):
        choose_tau([], [0.1])


def test_replace_tau_keeps_comments_and_other_lines():
    out = replace_tau(CONFIG, 0.4321)
    old_lines, new_lines = CONFIG.splitlines(), out.splitlines()
    i = next(n for n, line in enumerate(old_lines) if line.startswith("tau:"))
    assert new_lines[i].startswith("tau: 0.4321 ") and new_lines[i].index("#") == old_lines[i].index("#")
    assert new_lines[:i] + new_lines[i + 1:] == old_lines[:i] + old_lines[i + 1:]
    assert yaml.safe_load(out)["tau"] == 0.4321
    long = replace_tau(CONFIG, 0.123456789012345678)
    assert "# set on dev" in long and yaml.safe_load(long)["tau"] == 0.123456789012345678
    assert replace_tau(CONFIG.replace("\n", "\r\n"), 0.5).count("\r\n") == CONFIG.count("\n")
    with pytest.raises(ValueError):
        replace_tau("k: 4\n", 0.5)


def test_cli_end_to_end(index_dir, tmp_path, capsys):
    dev = tmp_path / "dev.jsonl"
    rows = [{"question": "UPI account debited but beneficiary not credited, when is the reversal?", "answerable": True},
            {"question": "ATM cash not dispensed but account debited", "answerable": True},
            {"question": "How do I file a complaint with the RBI Ombudsman?", "answerable": True},
            {"question": "Best mutual fund returns kya hai?", "answerable": False}]
    dev.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    cfg = tmp_path / "retrieval.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    out = tmp_path / "tau.json"

    res = main(["--index", str(index_dir), "--dev", str(dev), "--config", str(cfg), "--out", str(out)])
    assert cfg.read_text(encoding="utf-8") == CONFIG  # no --write, no change
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["n_answerable"] == 3 and saved["n_unanswerable"] == 1
    assert saved["index_version"] == "v1" and saved["embedding_model"] is None and saved["gate"] == "bm25_raw"
    assert saved["tau"] > 0 and saved["unanswerable_caught"] == 1.0  # off-topic question has BM25 0
    assert "WARNING" in capsys.readouterr().err

    main(["--index", str(index_dir), "--dev", str(dev), "--config", str(cfg), "--out", str(out), "--write"])
    written = cfg.read_text(encoding="utf-8")
    assert yaml.safe_load(written)["tau"] == res["tau"]
    assert "# set on dev so over-refusal stays under 10%" in written and "# pick a multilingual" in written
