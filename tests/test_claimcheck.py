from core.claimcheck import FAIL, NA, PASS, check, extract_claims

UPI = "Unified Payments Interface (UPI): auto reversal (R) by the Beneficiary bank latest on T + 1 day. | ₹100/- per day if delay is beyond T + 1 day."
ATM = "ATMs: Pro-active reversal (R) of failed transaction within a maximum of T + 5 days. | ₹ 100/- per day of delay beyond T + 5 days."


def test_supported_answer_passes():
    r = check("Bank ko T + 1 din mein paisa wapas karna hai, warna Rs 100 per day [rbi-11693-p15]", [UPI])
    assert r.status == PASS


def test_invented_day_count_fails():
    r = check("Paisa T + 3 din mein wapas aayega [rbi-11693-p15]", [UPI])
    assert r.status == FAIL and r.unsupported == ["T+3"]


def test_wrong_system_rule_fails_against_cited_passage():
    r = check("UPI refund T+5 days mein aata hai [rbi-11693-p15]", [UPI])
    assert r.status == FAIL


def test_citation_ids_are_not_claims():
    assert extract_claims("Haan [rbi-11693-p15]") == set()
    assert check("Haan, bank refund karega [rbi-11693-p15]", [UPI]).status == NA


def test_amount_formats_normalize():
    passage = "Compensation of Rs. 1,00,000 or 2.50 percent."
    assert check("₹100000 milega", [passage]).status == PASS
    assert check("2.5% milega", [passage]).status == PASS


def test_devanagari_digits():
    assert check("T + १ दिन में पैसा वापस", [UPI]).status == PASS
    assert check("₹१०० प्रति दिन", [UPI]).status == PASS


def test_bare_day_count_supported_by_t_plus():
    assert check("5 din ke andar paisa wapas", [ATM]).status == PASS


def test_number_words():
    assert check("within five days", [ATM]).status == PASS
    assert check("within three days", [ATM]).status == FAIL


def test_injected_number_from_question_is_not_supported():
    r = check("Bank must refund in 1 hour", [ATM])
    assert r.status == FAIL and r.unsupported == ["1"]
