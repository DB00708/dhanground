import json

import pytest

PASSAGES = [
    {"pid": "rbi-11693-p15", "doc_id": "rbi-11693", "para_id": "p15", "title": "TAT circular",
     "url": "https://www.rbi.org.in/x", "date": "2019-09-20", "topics": ["upi"], "sha256": "a",
     "text": "Unified Payments Interface (UPI): Account debited but the beneficiary account is not credited. "
             "Auto reversal by the Beneficiary bank latest on T + 1 day. ₹100/- per day if delay is beyond T + 1 day."},
    {"pid": "rbi-11693-p10", "doc_id": "rbi-11693", "para_id": "p10", "title": "TAT circular",
     "url": "https://www.rbi.org.in/x", "date": "2019-09-20", "topics": ["atm"], "sha256": "b",
     "text": "Automated Teller Machines (ATMs): Customer's account debited but cash not dispensed. "
             "Pro-active reversal within a maximum of T + 5 days. ₹100/- per day of delay beyond T + 5 days."},
    {"pid": "faq-153-p4", "doc_id": "faq-153", "para_id": "p4", "title": "Ombudsman FAQ",
     "url": "https://www.rbi.org.in/y", "date": "2026-07-01", "topics": ["ombudsman"], "sha256": "c",
     "text": "A complaint can be filed with the RBI Ombudsman if the bank does not reply within 30 days "
             "or the customer is not satisfied with the reply."},
]


@pytest.fixture
def index_dir(tmp_path):
    d = tmp_path / "index" / "v1"
    d.mkdir(parents=True)
    with (d / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in PASSAGES:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (d / "meta.json").write_text(json.dumps({"index_version": "v1", "embedding_model": None}))
    return d
