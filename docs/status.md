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
