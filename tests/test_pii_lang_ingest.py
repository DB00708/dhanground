import pytest

from core import lang
from core.pii import MASK, for_log, mask
from pipeline.ingest import check_allowed, chunk_text


@pytest.mark.parametrize("text", [
    "mera account 123456789012 hai",
    "card 4111 1111 1111 1111 se payment",
    "aadhaar 1234-5678-9012",
    "मेरा खाता १२३४५६७८९०१ है",
])
def test_pii_masked(text):
    masked, n = mask(text)
    assert n == 1 and MASK in masked


@pytest.mark.parametrize("text", ["T + 1 day", "Rs 100 per day", "Rs 1,00,000", "call 14440"])
def test_non_pii_untouched(text):
    assert mask(text) == (text, 0)


def test_log_truncated():
    assert len(for_log("a" * 500)) == 200


@pytest.mark.parametrize("text,expected", [
    ("UPI payment fail hua par paisa kat gaya, kab wapas aayega?", lang.HI_LATN),
    ("paisa wpas kab aayega", lang.HI_LATN),
    ("यूपीआई से भेजा पैसा कब वापस आएगा?", lang.HI_DEVA),
    ("When will my failed ATM withdrawal be reversed?", lang.EN),
    ("UPI ka refund कब आएगा", lang.HI_LATN),
    ("என் பணம் எப்போது திரும்ப வரும்?", lang.OTHER),
])
def test_lang_detect(text, expected):
    assert lang.detect(text) == expected


def test_allowlist():
    check_allowed("https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=11693")
    for bad in ["http://www.rbi.org.in/x", "https://rbi.org.in.evil.com/x", "https://example.com/rbi"]:
        with pytest.raises(ValueError):
            check_allowed(bad)


def test_table_rows_carry_section_heading():
    text = "\n".join([
        "| Sl. no. | Description of the incident | Framework | |",
        "| 4 | Unified Payments Interface (UPI) | | |",
        "| a | Account debited but the beneficiary account is not credited. | Auto reversal latest on T + 1 day. | ₹100/- per day |",
        "| b | Merchant confirmation not received. | | |",
    ])
    chunks = chunk_text(text)
    assert chunks[0].startswith("Unified Payments Interface (UPI): Account debited")
    assert "T + 1" in chunks[0]
    assert chunks[1] == "Unified Payments Interface (UPI): Merchant confirmation not received."
