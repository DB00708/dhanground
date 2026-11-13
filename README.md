# DhanGround: Saaf Data, Sahi Jawab

Tiny Aya Fire trained to answer Hinglish banking questions only from public RBI rules, with a four-arm
ablation showing that curated data beats raw synthetic data at equal row count.
Lexsi AI Safety and Alignment Hackathon, Track 1. Chains CuratorKIT → AlignTune → AuditKIT.
Research demo, not financial or legal advice. Tiny Aya weights are non-commercial.

| Arm | Data | Objective | Isolates |
|---|---|---|---|
| A0 | none | none | starting point |
| A1 | 3,000 raw rows | SFT | effect of more data |
| A2 | 3,000 curated rows, same strata as A1 | SFT | curation dividend = A2 − A1 |
| A3 | A2 + mined pairs | DPO on A2 | preference gain = A3 − A2 |

## Layout

| Path | Component |
|---|---|
| `core/` | deterministic, unit-tested: template, schema + validators V1–V5, claimcheck, splits, PII, language, manifests |
| `pipeline/ingest.py` | 1 corpus ingester (HTML + PDF, rbi.org.in allowlist) |
| `intelligence/generate.py` | 2 teacher question generation (LiteLLM) |
| `pipeline/curate.py` + `adapters/curatorkit_run.py` | 3 curation gates (ours + CuratorKIT dedup / grounding / reward) |
| `pipeline/build_arms.py` | A1/A2 datasets, test review and freeze |
| `intelligence/mine_pairs.py` | 4 DPO pairs from A2's own failures |
| `adapters/aligntune_sft.py`, `adapters/train_arm.py`, `adapters/aligntune_dpo.py` | 5 trainer (AlignTune SFT seed 0, TRL SFT other seeds; TRL DPO with A2 as reference) |
| `evals/` + `adapters/auditkit_eval.py` | 6 evaluator, summary, ROI |
| `retrieval/` | 7 hybrid BM25 + multilingual embedding retriever |
| `serving/`, `apps/api/` | 8 answer service, `/v1` FE↔BE contract |
| `apps/space/` | 9 Gradio demo (mounted on the API server) |

## Runbook

Setup (CPU side):

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt && .venv/Scripts/python -m pytest
```

On the L4 box: `pip install -r requirements-gpu.txt && pip freeze > requirements.lock`, then `hf auth login`
(Tiny Aya is gated). Env: `TEACHER_MODEL`, `JUDGE_MODEL` (non-Cohere), provider key, `INDEX_DIR`.

**Corpus and split.** PDFs on rbidocs.rbi.org.in show a CAPTCHA to scripts: open each `url` with a `file`
in `configs/rbi_sources.yaml` in a browser and save it at that path. HTML pages fetch automatically.

```bash
python -m pipeline.ingest configs/rbi_sources.yaml corpus/
python -m pipeline.split corpus/ data/splits.json
python -m retrieval.build_index corpus/ index/ --version rbi-2026-11-10 --embedding-model intfloat/multilingual-e5-small
```

**Generate and curate.** Then hand-write the 100 adversarial test rows in `evals/adversarial_test.jsonl`
(six seeds are there) and hand-check the test rows.

```bash
python -m intelligence.generate --index index/rbi-2026-11-10
python -m pipeline.curate
python -m pipeline.build_arms arms
python -m pipeline.build_arms review          # fill keep=y/n in data/test/review.csv
python -m pipeline.build_arms freeze --index index/rbi-2026-11-10
```

**Train A1, A2 (3 seeds each), mine pairs, train A3.**

```bash
python -m adapters.train_arm --arm A1 --hub-org <org>    # seed 0 via AlignTune, seeds 1, 2 via --backend trl
python -m adapters.train_arm --arm A2 --hub-org <org>
python -m intelligence.mine_pairs --a2 checkpoints/A2-s0-<sha> --extra-curated-train 1500
python -m adapters.aligntune_dpo --a2 checkpoints/A2-s0-<sha> --hub-org <org>
```

Seed split: AlignTune 0.1.33 cannot resume (its SFT `train()` calls the HF trainer with no
`resume_from_checkpoint`), so it trains only seed 0 of A1 and A2, the run that carries the
CuratorKIT → AlignTune → AuditKIT chain. Seeds 1 and 2 use `--backend trl`, which can resume.
Both backends train on the same `messages` rows, the same chat-template render (with
`enable_thinking=False`, which AlignTune adds), the same loss mask (`sft.assistant_only_loss`) and the
same LoRA and optimizer settings (`configs/lora.yaml`). Before training, each run checks on the real
tokenizer that the training render starts with the exact prompt serving builds; it stops if not.
If it stops because TRL cannot mask assistant tokens for this template, set
`sft.assistant_only_loss: false` (both backends then use full-sequence loss) and say so in the report.
`run_eval` records each seed's backend (from the adapter's `dhanground.json`) and the summary
table has a "backend by seed" row, so the report shows the AlignTune/TRL split. `--dry-run` prints the per-seed commands. If AlignTune fails outright, pass `--aligntune-seeds` with no
value to train every seed with `--backend trl`.

If HF credits run out mid-run, resume on Kaggle from the last Hub checkpoint (saved every 200 steps):
`python -m adapters.aligntune_sft --arm <arm> --seed <seed> --backend trl --resume-from-hub <org>/dhanground-<arm>-s<seed>-<sha>`
(DPO: rerun the same command plus `--resume-from-hub <org>/dhanground-A3-s0-<sha>`). The resumed run keeps
the original run's name and repo. A dead AlignTune seed 0 cannot resume: rerun it from step 0, or resume it
with `--backend trl` and record in the report that seed 0 finished on TRL.
Then rerun `train_arm` with `--seeds` set to the seeds still missing.

**Set τ, evaluate, report.** Put adapter paths in `configs/serving.yaml`. Tune `tau` in
`configs/retrieval.yaml` on dev so over-refusal stays under 10%. Score each arm on test once.

```bash
python -m evals.run_eval --arm A2 --seed 0 --split dev
python -m evals.run_eval --arm A2 --seed 0 --split test --adapter <path>   # once per (arm, seed)
python -m adapters.auditkit_eval --arms A0 A1 A2 A3
python -m evals.summarize --split test
python -m evals.roi --measure --arm A3 --p-in <vendor $/1M in> --p-out <vendor $/1M out>
```

**Demo.** `python -m apps.api.build_cache --index index/rbi-2026-11-10` caches the 30 examples for all
arms, then `python -m apps.space.app` serves the API and UI on port 7860. On an HF Space set
`app_file: apps/space/app.py`. With `DEMO_MODE=cached` or no GPU, it serves labeled cached answers.

## Contracts

- Prompt: `core/template.py`, identical for train and serve; byte-exact test in `tests/test_template.py`.
- Model output: `{"answerable", "answer", "cited_pids", "lang"}`; backend validation V1–V5 in `core/schema.py`.
- API: `POST /v1/ask`, `POST /v1/compare`, `GET /v1/examples`, `GET /v1/health`; errors
  `{"request_id", "status": "error", "error": {"code", "message"}}` with BAD_INPUT, RATE_LIMITED,
  MODEL_UNAVAILABLE, INTERNAL.
