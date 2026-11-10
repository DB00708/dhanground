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
| `adapters/aligntune_sft.py`, `adapters/aligntune_dpo.py` | 5 trainer (AlignTune SFT; TRL DPO with A2 as reference) |
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
python -m adapters.aligntune_sft --arm A1 --seed 0 --hub-org <org>     # repeat seeds 1, 2; then A2
python -m intelligence.mine_pairs --a2 checkpoints/A2-s0-<sha> --extra-curated-train 1500
python -m adapters.aligntune_dpo --a2 checkpoints/A2-s0-<sha> --hub-org <org>
```

If AlignTune fails on the chat template, add `--backend trl` (same LoRA settings, completion-only loss).

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
