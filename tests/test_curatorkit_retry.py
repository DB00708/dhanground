from adapters import curatorkit_run
from adapters.curatorkit_run import row_key


def rows(n):
    return [{"question": f"q{i}", "answer": f"a{i}", "passages": [], "cited_pids": []} for i in range(n)]


def test_judge_errors_are_retried_but_low_scores_are_not(tmp_path, monkeypatch):
    rs = rows(4)
    keys = [row_key(r["question"], r["answer"]) for r in rs]
    calls = []

    def fake_once(batch, work_dir, *a):
        calls.append([r["question"] for r in batch])
        if len(calls) == 1:  # first pass: 0 passes, 1 low score, 2 and 3 judge errors
            return ({keys[0]: 0.9}, {keys[1]: "below_reward_threshold:0.30", keys[2]: "judge_error:JudgeParseError",
                                     keys[3]: "judge_error:JudgeParseError"})
        return {row_key(r["question"], r["answer"]): 0.8 for r in batch if r["question"] == "q2"}, \
               {row_key(r["question"], r["answer"]): "judge_error:JudgeParseError" for r in batch if r["question"] == "q3"}

    monkeypatch.setattr(curatorkit_run, "_run_once", fake_once)
    passed, rejected = curatorkit_run.run_gates(rs, tmp_path, "m", 0.7, 0.7)
    assert set(passed) == {keys[0], keys[2]}
    assert rejected[keys[1]].startswith("below_reward") and rejected[keys[3]].startswith("judge_error")
    assert calls[1] == ["q2", "q3"] and calls[2] == ["q3"] and len(calls) == 1 + curatorkit_run.JUDGE_RETRIES
