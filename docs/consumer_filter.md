# Consumer-content filter for question generation

Generation (`intelligence/generate.py`) asks the teacher for questions only from passages a retail customer
could plausibly ask about: rights, charges, documents, timelines, limits, complaints, fraud liability. Bank-internal
content (board and governance, reporting to RBI / FIU, internal controls, repeal lists) is skipped. Retrieval still
indexes every passage, so the model can still answer or abstain on those topics at inference time.

Rule used: **when unsure, keep.**

## How it works

`core/relevance.py` tags each passage with its chapter (chapters only move forward, contents pages are `toc`,
cross-references such as "Chapter XI – Acquisition Finance of these Directions" are ignored).
`configs/generation.yaml` then decides:

| Key | Effect |
|---|---|
| `exclude_chapters` | skip every passage in these chapters |
| `keep_spans` | keep a section inside an excluded chapter (by heading text, `from` up to `to`) |
| `exclude_spans` | skip a section inside a kept chapter, or in a document without chapter headings; wins over `keep_spans`; `to: null` runs to the end |
| `exclude_terms` | skip any passage that contains the substring (unused) |

Changes made for this filter:

- `exclude_spans` and `to: null` are new. Four documents (PPIs, Internal Ombudsman, Ombudsman Scheme, BBPS) have
  no chapter headings that the tagger can read, and two (KYC, Responsible Business Conduct) have bank-side sections
  inside chapters that are otherwise for customers.
- Cross-reference fix: the old check treated any heading followed by "of the" as a cross-reference, so
  md-13642 "Chapter II - Role of the Board" was missed. Now only "of these / the Directions" counts. md-13156 chapter
  tags and kept passages are identical before and after (checked passage by passage).
- `tests/test_relevance.py::test_configured_spans_resolve_on_the_real_index` checks that every configured span
  matches text in the built index (set `DHANGROUND_INDEX_DIR` or build `index/`; it is skipped otherwise).
  A span that does not match also raises `ValueError` at generation time.

Passage counts below are all passages of the document in that chapter (index `rbi-2026-11-10`), with the
paragraph range. "Kept" counts are before the `MIN_SOURCE_WORDS` length filter.

## Before / after generation passages per split

From `intelligence.generate.load_paragraphs` (after the contents-page and minimum-length filters), on
`index/rbi-2026-11-10` and `data/splits.json`. "Before" is commit 9b12a5e (only md-13156 filtered).

| Split | Before | After | Dropped |
|---|---:|---:|---:|
| train | 1815 | 1534 | 281 |
| dev | 224 | 140 | 84 |
| test | 463 | 418 | 45 |
| **total** | **2502** | **2092** | **410** |

| Document | Split | Before | After |
|---|---|---:|---:|
| md-13141 KYC | train | 395 | 193 |
| md-13140 Responsible Business Conduct | train | 612 | 577 |
| md-12156 PPIs | dev | 207 | 123 |
| md-13155 Credit / Debit Cards | train | 135 | 130 |
| md-13642 Digital Payment Security | test | 104 | 59 |
| md-13157 Interest on Deposits | train | 84 | 79 |
| md-13271 Internal Ombudsman | train | 54 | 29 |
| ios-2026 Integrated Ombudsman Scheme | train | 81 | 80 |
| md-12616 BBPS | train | 21 | 13 |
| md-13156 Credit Facilities (unchanged) | train | 198 | 198 |

All other documents are unchanged.

Note: the dev split loses 38% of its generation passages (PPI is its only long Master Direction). The 400-row dev target
needs 200 jobs at 2 questions per call, so each of the 140 passages is used about 1.4 times.

## md-13141 KYC (404 passages, train)

| Chapter | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| – | Contents | 3 (p1-3) | exclude | contents page |
| – | Introduction (FATF, PMLA) | 3 (p4-6) | keep | untagged preamble; harmless background |
| I | Preliminary (definitions: OVD, CDD, V-CIP, etc.) | 46 (p7-52) | keep | customers ask what counts as an OVD, what V-CIP is |
| II | General (KYC policy, board, ML/TF risk assessment) | 14 (p53-66) | exclude | board and policy framework |
| III | Customer Acceptance Policy | 7 (p67-73) | keep | when a bank may refuse or must not deny an account |
| IV | Risk Management (risk categorisation, STR instead of CDD) | 3 (p74-76) | keep | bank must not reveal the risk category to the customer (reviewer) |
| V | Customer Identification Procedure | 4 (p77-80) | keep | when identification is needed, reliance on third-party CDD |
| VI | Customer Due Diligence | 107 (p81-187) | keep 102 | documents, e-KYC, V-CIP, small accounts, re-KYC, PEPs, foreign students |
| VI § 46 | Client accounts via professional intermediaries | 2 (p180-181) | exclude | intermediary-side obligations |
| VI § 49 | Simplified KYC for Foreign Portfolio Investors | 3 (p185-187) | exclude | institutional investors |
| VII | Record Management | 5 (p188-192) | exclude | record keeping |
| VIII | Reporting to FIU-India | 6 (p193-198) | exclude | CTR / STR reporting |
| IX | International agreements (UNSC / UAPA lists, FATF jurisdictions) | 16 (p199-214) | exclude | sanctions screening by the bank |
| X | Other Instructions | 55 (p215-269) | keep 30 | see rows below |
| X § 62-63 | Secrecy obligations and sharing of customer information | 2 (p215-216) | keep | "can my bank share my data?" |
| X § 64 | Foreign Contribution (FCRA) | 1 (p217) | exclude | NGO-specific; **borderline** |
| X § 65 | CKYCR: KYC identifier, re-KYC via CKYCR | 10 (p218-227) | keep | customers get and use a KYC identifier |
| X § 66 | FATCA / CRS reporting | 4 (p228-231) | exclude | reporting to tax authorities |
| X § 67-70 | Cheque validity, money mules, account payee cheques, UCIC | 5 (p232-236) | keep | cheque validity period, account freeze for mule accounts |
| X § 72-73, 74(2)+ | New technologies, correspondent banking, wire-transfer bank duties | 17 (p237-242, p251-261) | exclude | bank-to-bank |
| X § 74(1) | Wire-transfer information: originator details for ₹50,000+ | 8 (p243-250) | keep | counter rule like the DD rule (reviewer) |
| X § 75-77 | Demand drafts ≥ ₹50,000, quoting PAN, third-party products | 5 (p262-266) | keep | DD and PAN rules customers meet at the counter |
| X § 78-79 | At-par cheques for co-op banks, hiring / training staff | 3 (p267-269) | exclude | internal |
| XI | Repeal and Other Provisions + annexes (UAPA and WMD Act orders) | 135 (p270-404) | exclude | repeal list and government orders to banks / registrars |

## md-13140 Responsible Business Conduct (619 passages, train)

| Chapter | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| – | Contents | 3 (p1-3) | exclude | contents page |
| – | Preamble | 1 (p4) | keep | untagged |
| I | Preliminary (definitions) | 17 (p5-21) | keep | definitions customers ask about |
| II A, B.1 | Role of Board: policy lists, board reviews | 6 (p22-27) | exclude | board governance |
| II B.2-B.3 | Deposit, cheque collection and compensation policies | 5 (p28-32) | keep | what the compensation policy must cover (erroneous debits, delayed collection) |
| II B.4-B.7 | Customer service committees (board, standing, branch) | 7 (p33-39) | exclude | internal committees |
| III | Customer Service in Branches | 27 (p40-66) | keep | branch hours, account transfer between branches |
| IV | Customer Guidance and Protection | 61 (p67-127) | keep | disclosures, limited / zero liability for fraud |
| V | Financial Inclusion and Accessibility | 44 (p128-171) | keep | BSBD accounts, senior citizens, Form 15G / H |
| VI | Payment and Remittance Services | 26 (p172-197) | keep | demand drafts, remittances |
| VII | Deposit Accounts and Other Liabilities | 194 (p198-391) | keep | passbooks, inoperative accounts, lockers, deceased claims |
| VIII A-I | Responsible Lending (fair practices, KFS, collateral release, microfinance) | 113 (p392-504) + p505 | keep | loan charges, timelines, compensation for late document release |
| VIII J | Export Credit | 18 (p506-523) | exclude | exporters, not retail (p505 kept: it ends the microfinance section) |
| VIII K | Recovery agents | 19 (p524-542) | keep | what recovery agents may not do, complaints |
| IX | Miscellaneous (standing instructions) | 8 (p543-550) | keep | customer instructions |
| X | Repeal and Savings | 4 (p552-555) | exclude | repeal list (p551 kept: §463 no incentives on banking products) |
| X annexes | Notice board, rates at a glance, deceased-claim forms, indemnity, locker inventory, KFS | 64 (p556-619) | keep | which documents a claim needs; **borderline** (many are blank form fields) |

## md-12156 Prepaid Payment Instruments (212 passages, dev; no chapter headings)

| Section | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| 1-2 | Introduction, definitions | 20 (p1-20) | keep | what a PPI / small PPI / full-KYC PPI is |
| 3-5 | Issuer eligibility, capital, authorisation of non-banks | 22 (p21-42) | exclude | issuer-side licensing |
| 6-11 | AML safeguards, issuance and loading, cross-border, PPI categories, interoperability | 56 (p43-98) | keep | KYC for wallets, load limits, gift cards, transit cards |
| 12 | Deployment of money collected (escrow) | 18 (p99-116) | exclude | issuer treasury |
| 13-17 | Validity and redemption, limits, security, grievance redress, limited liability | 52 (p117-168) | keep | expiry, refunds, fraud liability, complaints |
| 18-20 + annexes | IS audit, reporting, repeal list, director declaration, escrow and complaint report formats, acronyms | 44 (p169-212) | exclude | audit and reporting |

## md-13155 Credit and Debit Cards (137 passages, train)

| Chapter | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| – | Contents (p2 also starts Chapter I) | 2 (p1-2) | exclude | contents page |
| I | Preliminary, definitions (untagged: the heading sits on the contents page) | 13 (p3-15) | keep | billing cycle, MITC, total amount due |
| II | Conduct of Credit Card Business | 70 (p16-85) | keep | issue, billing, closure, interest, reporting to bureaus |
| III | Debit Cards | 4 (p86-89) | keep | debit card issue rules |
| IV | Form Factor | 1 (p90) | keep | wearables / tokens |
| V | Co-branding | 8 (p91-98) | keep | data sharing with co-brand partner; **borderline** |
| VI | General Guidelines for cards | 30 (p99-128) | keep | unsolicited cards, card controls, fraud |
| VII | Miscellaneous (MITC contents) | 3 (p129-131) | keep | what MITC / KFS must state |
| VIII | Repeal and Other Provisions | 6 (p132-137) | keep 1 | p132 holds the KFS table end (kept by span); the rest is repeal |

## md-13642 Digital Payment Security Controls (107 passages, test)

| Chapter | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| – | Contents, Hindi cover | 3 (p1-3) | exclude 2 / keep 1 | contents page; Hindi cover is untagged |
| I | Preliminary (definitions) | 8 (p4-11) | keep | definitions |
| II | Role of the Board | 1 (p12) | exclude | board policy (newly tagged by the cross-reference fix) |
| III A-C | Governance of security risk, generic controls, application security life cycle | 39 (p13-51) | exclude | bank IT controls |
| III D-E | Authentication framework, fraud risk management | 15 (p52-66) | keep | 2FA / AFA, transaction alerts, fraud monitoring |
| III F | Reconciliation | 1 (p67) | exclude | back office |
| III G | Customer protection, awareness, grievance redress | 9 (p68-76) | keep | customer-facing |
| IV | Internet banking security controls | 3 (p77-79) | keep | session time-outs, login alerts |
| V | Mobile payments app security | 11 (p80-90) | keep | device binding, app cooling period |
| VI | Card payment security | 13 (p91-103) | keep | card limits, on / off controls |
| VII | Repeal and Other Provisions | 4 (p104-107) | exclude | repeal |

## md-13157 Interest Rate on Deposits (89 passages, train)

| Chapter | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| – | Contents | 5 (p1-5) | exclude | contents page |
| I | Preliminary | 10 (p6-15) | keep | definitions |
| II | Interest rate framework | 7 (p16-22) | keep | how banks set rates |
| III | Domestic rupee deposits | 21 (p23-43) | keep | savings / FD interest, premature withdrawal, deceased depositors |
| IV | Rupee deposits of non-residents | 11 (p44-54) | keep | NRE / NRO interest |
| V | Foreign currency deposits | 20 (p55-74) | keep | FCNR(B) |
| VI | Prohibitions and exemptions | 10 (p75-84) | keep | what banks may not pay or offer |
| VII | Repeal and Other Provisions | 5 (p85-89) | exclude | repeal |

## md-13271 Internal Ombudsman (54 passages, train; no chapter headings parsed)

| Section | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| 1-4 | Preamble, applicability, definitions | 11 (p1-11) | keep | what a complaint / deficiency in service is |
| 5-11 | IO appointment, tenure, oversight, secretariat, internal audit | 15 (p12-26) | exclude | bank staffing and governance |
| 12 | Role and responsibilities of IO | 4 (p27-30) | keep | what the IO does for a complaint |
| 13 | Board oversight | 3 (p31-33) | exclude | board reporting |
| 14-15 | Complaint handling: auto escalation, 30-day timeline, what is outside scope | 14 (p34-47) | keep | customer timelines and rights |
| 16 + annex | Reporting to RBI, repeal, quarterly report format | 7 (p48-54) | exclude | reporting |

## ios-2026 Integrated Ombudsman Scheme (83 passages, train; no chapter headings)

| Section | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| 1-6 | Scope, definitions, ombudsman offices, CRPC | 16 (p1-16) | keep | who is covered, where complaints go |
| 7 | Staffing of ombudsman offices | 1 (p17) | exclude | RBI internal |
| 8-18 | Powers, grounds, filing, resolution, award, appeal, display duties | 46 (p18-63) | keep | the customer complaint process |
| 19 | Removal of difficulties | 1 (p64) | keep | user decision: keep all of § 19-20 |
| 20 | Repeal of the 2021 scheme, pending complaints | 2 (p65-66) | keep | which scheme covers complaints before July 1, 2026 (reviewer) |
| Annexes | Guidance for filing, complaint form | 17 (p67-83) | keep | how to file, compensation caps |

## md-12616 Bharat Bill Payment System (21 passages, train; no chapter headings)

| Section | Title | Passages | Decision | Reason |
|---|---|---:|---|---|
| Cover letter | Purpose, review | 4 (p1-4) | keep | background; harmless |
| 1 | Short title | 1 (p5) | keep | untagged |
| 2-3 | Applicability, authorisation of NBBL / BBPOUs | 2 (p6-7) | exclude | operator licensing |
| 4 | Definitions | 5 (p8-12) | keep | what BBPS, biller, COU are |
| 5-7 | Participation, roles, settlement and escrow | 5 (p13-15, p17-18) | exclude | operator / participant obligations |
| 6 c | COU duties: customer access to billers, dispute system | 1 (p16) | keep | customer-facing |
| 8 | Complaint management, failed transactions | 2 (p19-20) | keep | customer complaint and refund timelines |
| 9 | Supersession | 1 (p21) | exclude | repeal list |

## Borderline calls (for confirmation)

1. md-13141 IV Risk Management (3 passages): now kept (T4 reviewer).
2. md-13141 X § 74(1) wire-transfer information (8 passages): now kept; the rest of § 72-74 (17) stays excluded.
3. md-13141 X § 64 FCRA (1 passage): excluded; matters only to NGOs.
4. md-13140 X annexes (64 passages): kept. They include blank claim and indemnity forms that may give weak questions,
   but also the list of documents for a deceased-depositor claim.
5. md-13155 V Co-branding (8 passages): kept; mostly bank-partner arrangements, with some data-sharing rules.
6. ios-2026 § 19-20 (p64-66): kept (user decision); which scheme covers older complaints is a real customer question.
7. Dev split shrinks from 224 to 140 source passages (PPI only); see note above.
