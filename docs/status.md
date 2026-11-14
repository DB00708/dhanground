# DhanGround build status

Deadline: Nov 14 AoE (Nov 13 submit, Nov 14 buffer). Last updated: 2026-11-10.
Status: Done = built and verified as stated in Evidence. Code done = written and unit-tested, not yet run on real
inputs. Blocked = waiting on something only the owner can provide. Open = not started.
Owner: main = the coordinating thread; Wn = coder thread n (reviewed by reviewer thread n); you = Divyansh.

## Work items

| ID | Phase | Item | Status | Owner | Evidence |
|---|---|---|---|---|---|
| B01 | Foundation | Repo skeleton, configs, venv, test harness | Done | main | 0167e9a |
| B02 | Foundation | Prompt template, byte-exact render test | Done | main | 0167e9a, tests/test_template.py |
| B03 | Foundation | Output schema + backend validators V1–V5 | Done | main | 0167e9a, tests/test_schema.py |
| B04 | Foundation | Deterministic claimcheck (T+N, ₹, Devanagari digits) | Done | main | 0167e9a, tests/test_claimcheck.py |
| B05 | Foundation | PII masking, language detection, stage manifests | Done | main | 0167e9a, tests/test_pii_lang_ingest.py |
| B06 | Data | Corpus ingester (HTML + PDF, rbi.org.in allowlist) | Done | main | 0167e9a; corpus: 29 docs, 2,975 passages |
| B07 | Data | 12 RBI PDFs (behind a CAPTCHA for scripts) | Done | you | saved by hand into sources/pdf on Nov 10 |
| B08 | Data | Topic-stratified document split | Done | main | 9b12a5e; test covers upi, atm, cards, kyc, ombudsman, fraud |
| B09 | Data | Consumer-content filter for generation (Credit Facilities) | Done | main | 9b12a5e, tests/test_relevance.py |
| B10 | Data | Hybrid retrieval (BM25 + multilingual embeddings) | Code done | main | BM25 index built; embedding index needs the GPU box |
| B11 | Data | Teacher question generation | Code done | main | not run: needs TEACHER_MODEL + API key |
| B12 | Data | Curation gates (ours + CuratorKIT) | Code done | main | not run: needs generated rows + JUDGE_MODEL |
| B13 | Data | A1/A2 arm builder, test review + freeze | Code done | main | not run: needs curated rows |
| B14 | Data | 100 hand-written adversarial test rows | In progress | W8 → you | 6 of 100; W8 drafts, you hand-check |
| B15 | Data | Hand-check of 500 test rows | Blocked | you | needs B12 output |
| B16 | Train | 50-row SFT smoke test on Tiny Aya (Nov 7 gate) | Blocked | you | needs L4 + gated Tiny Aya access |
| B17 | Train | AlignTune SFT adapter (A1, A2; TRL fallback) | Code done | main | untested on GPU |
| B18 | Train | DPO pair miner + A3 DPO (A2 as reference) | Code done | main | untested on GPU |
| B19 | Train | Train A1, A2 x 3 seeds; A3 | Blocked | you | needs B13 + GPU |
| B20 | Eval | Metrics, bootstrap CIs, slices, release gates, chart | Code done | main | tests/test_metrics_pipeline.py |
| B21 | Eval | AuditKIT general-capability benchmark | Code done | main | dataset config unverified |
| B22 | Eval | ROI with measured throughput | Code done | main | not run |
| B23 | Demo | Answer engine + FastAPI /v1 contract | Done | main | tests/test_api.py, tests/test_retrieval_engine.py (fake model) |
| B24 | Demo | Gradio Space mounted on the API | Code done | main | not run: gradio not installed locally |
| B25 | Edge cases | Unsupported language answered in English | Open | W1 | design edge case 2 |
| B26 | Edge cases | Typo-noise and multi-passage training rows | Open | W2 | design edge cases 3, 10 |
| B27 | Edge cases | Tau tuning tool on dev | Open | W3 | design edge case 8 |
| B28 | Edge cases | Demo warm-up ping | Open | W4 | design edge case 18 |
| B29 | Edge cases | Resume training from a Hub checkpoint | Open | W5 | design edge case 19 |
| B30 | Repro | Notebooks 01–07, one per stage | Open | W6 | design repo tree |
| B31 | Publish | HF publish script, test-set card, upstream-issues doc | Open | W7 | design: open artifacts |
| B32 | Data | Consumer filter for KYC and other Master Directions | Open | W9 | follow-up to B09 |
| B33 | Repro | requirements.lock pinned on the L4 box | Blocked | you | needs GPU box |
| B34 | Publish | Run publish: datasets, adapters, results public on HF | Blocked | you | needs B19–B22 |
| B35 | Publish | HF article + README results table | Open | main | needs results |
| B36 | Pitch | 10-slide deck with measured numbers | Open | main | needs results |

## Team (sessions started 2026-11-12)

Manager: this session. Flow: coder → its task reviewer (loop to APPROVE) → integration tester (Docker e2e, loop to PASS) → manager merges to main.

| Task | Item | Coder session | Reviewer session | Integration |
|---|---|---|---|---|
| T1 | Unsupported language → English (B25) | serve-demo | lang-gate-review | integration-e2e |
| T2 | Demo warm-up ping (B28) | serve-demo | warmup-review | integration-e2e |
| T3 | Typo + two-passage rows (B26) | data-gen | gen-variety-review | integration-e2e |
| T4 | Consumer filter, other directions (B32) | data-gen | consumer-filter-review | integration-e2e |
| T5 | Tau tuning tool (B27) | eval-set | tau-tune-review | integration-e2e |
| T6 | 100 adversarial test rows (B14) | eval-set | adversarial-set-review | integration-e2e |
| T7 | Resume from Hub checkpoint (B29) | train-resume | hub-resume-review | integration-e2e |
| T8 | Seed backend plan (AlignTune seed 0, TRL rest) | train-resume | seed-plan-review | integration-e2e |
| T9 | Stage notebooks (B30) | repro-publish | notebooks-review | integration-e2e |
| T10 | HF publish, test card, upstream doc (B31) | repro-publish | publish-review | integration-e2e |

Review and integration results are logged in tests/e2e/INTEGRATION_LOG.md on the integration branch.

### Review log

| Time (IST) | Task | Event | Branch @ sha | Notes |
|---|---|---|---|---|
| 2026-11-12 | T2 | Reviewer APPROVED | claude/serve-demo-d875aa @ 29561c5 | 3,000 simulated runs vs the server rate limiter: 0 × 429; 107 tests; 2 optional nits |
| 2026-11-12 | T1 | Reviewer APPROVED | claude/serve-demo-d875aa @ 93540fa | 2 rounds; fixed short Tamil answers with citations being read as English; T2 approval still valid (later commit touches only core/lang.py + tests) → sent to integration |
| 2026-11-12 | T3 | Reviewer APPROVED | claude/data-gen-0f329e @ 2d44de9 | 1 should-fix (review.csv showed only the first passage) + 2 nits fixed; decision: typo and two-passage questions also in dev/test, reported as their own slice |
| 2026-11-13 | T1+T2 | MERGED to main | main @ ea25de3 | with the e2e harness (498297d); 109 unit tests; native e2e 9/9 PASS in 82 s incl. warm-up across 4 arms; pushed to https://github.com/DB00708/dhanground (public) |
| 2026-11-13 | T7 | Reviewer CHANGES REQUESTED | claude/train-resume-d748e9 @ d73a895 | no blockers; 1 should-fix: a failed Hub upload kills the checkpoint-sync thread and can block the A3 final save; download layout, latest-checkpoint pick, trl 1.7.1 DPO changes, DPO restore and "AlignTune can't resume" all verified against library source; A3 resume untested on GPU |
| 2026-11-13 | T3+T4, T5+T6, T9+T10 | MERGED to main | 6c1eaa9, 9e71290, c1395e7 | all reviewer-approved (later commits on each branch touched only the sibling task); 216 tests; e2e 9/9 PASS |
| 2026-11-13 | T7+T8 | MERGED to main | 20bdf92 | T7 @ 0711d8b, T8 @ 992fdb9 (later commits T8-only) |
| 2026-11-13 | coupling fix | main | a858e7e | notebook 04 → adapters.train_arm + resume cell (T8 × T9); 239 tests; pushed |
| 2026-11-13 | real run | main | ebe3780 | Gemini teacher (3.6 Flash) / judge (3.1 Pro) verified; embedding index rbi-2026-11-13 (e5-small); generation running; output-token budgets raised for thinking models |
| 2026-11-13 | T11 | Budget plan | claude/hf-jobs-f8e090 @ 92b01c2 (in review) | HF Jobs need credit balance > 0 (not PRO); GPU Spaces need PRO, so the demo is a free CPU Space in cached mode. Full run ≈ $20–25 at $0.80/h. Priority if short: probe → stage → smoke → A1 s0, A2 s0 → A2 dev eval → mine → DPO (A3) → test evals s0 → seeds 1–2 |
| 2026-11-13 | T1–T10 | Integration PASS (Docker) | main @ a6e6480 (integration 37bb401) | 235 unit passed / 4 skipped; e2e 10/10 in 82 s incl. evals.summarize; image installs CPU torch then requirements.txt |
| 2026-11-13 | compute | Decision | — | HF rejects Indian cards; rules allow training anywhere (only artifacts on HF). Primary compute = Kaggle T4 ×2, fp16 run-wide (account divyanshbhatia007, token verified); T11 extended with a Kaggle backend + precision setting |
| 2026-11-13 | real run | Generation DONE | main @ ebe3780 | 6,100 raw rows (train 5,000 / dev 400 / test 700); 81% answerable, 19% unanswerable, 4% adversarial; 228 typo, 439 two-passage; 0 failed jobs after token fix; curation trial on dev running (Gemini 3.1 Pro judge) |
| 2026-11-14 | real run | Curation DONE | main @ 977db51 | judge = Gemini 3.5 Flash-Lite (thinking judges truncated JSON under load: 30–75% false rejects); train 5,000→4,133, dev 400→353, test 700→588; removed ~700 low-quality, 187 mislabelled unanswerable, 55 invented-number rows |
| 2026-11-14 | real run | Arms BUILT | 977db51 | A1 = A2 = 3,000 rows, identical strata; A1 has 531 rows that fail curation incl. 28 invented-number answers, A2 has 0; 17 cross-split duplicates dropped (test untouched); staged to private HF dataset divyansh-b/dhanground-work |
| 2026-11-12 | T5 | Reviewer APPROVED | claude/eval-set-cc3068 @ c427d83 | tau tuning only; 3 optional nits, no blockers |

## Edge-case catalog (design: 20 cases)

| # | Case | Mechanism | Status |
|---|---|---|---|
| 1 | Mixed script question | script detector, template rule | Done |
| 2 | Tamil or Bengali question | langid gate, English reply | Open (W1) |
| 3 | Romanized typos | hybrid retrieval, typo rows in train | Partial: retrieval only (W2) |
| 4 | Invented day count | claimcheck amber badge | Done |
| 5 | Right rule, wrong payment system | judge with gold answer | Done |
| 6 | Malformed JSON | one repair, then abstain | Done |
| 7 | Cites unknown pid | V2 drop, abstain if none | Done |
| 8 | Over-refusal | metric, tau tuned on dev, DPO negatives | Partial: tuning tool missing (W3) |
| 9 | Conflicting old vs amended rule | newer-first ranking, doc date | Done |
| 10 | Answer needs 2 passages | top-4, multi-cite train rows | Partial: no multi-cite rows (W2) |
| 11 | No passage above tau | retrieval gate | Done |
| 12 | Empty or overlong input | 400 BAD_INPUT | Done |
| 13 | Account number in question | FE + BE masking | Done |
| 14 | Prompt injection | tagged fields, adversarial rows | Done |
| 15 | Personal advice | complaint path, adversarial rows | Done |
| 16 | Duplicate questions across splits | hash overlap fails build | Done |
| 17 | Teacher leaks wrong fact | CuratorKIT grounding gate | Done |
| 18 | GPU cold start | cached answers, warm-up ping | Partial: ping missing (W4) |
| 19 | Credits run out mid-training | Hub checkpoints, resume | Partial: resume missing (W5) |
| 20 | Toolkit version drift | pinned versions | Partial: lock on GPU box (B33) |

## Change log

| Date | Commit | Change |
|---|---|---|
| 2026-11-10 | 0167e9a | All nine components built; 76 tests |
| 2026-11-10 | 9b12a5e | Topic-stratified split; consumer-content filter; 86 tests |
