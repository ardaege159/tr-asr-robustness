from trasr.generate import _clean_notes, _say, check, check_selfcorr

SPEC = {
    "id": "x", "company": "eticaret", "expected": [{"tool": "siparis_sorgula", "args": {"siparis_no": "5472381"}}],
    "correction": {"arg": "siparis_no", "wrong": "5472318"},
}


def turns(*texts):
    return [{"role": "assistant" if i % 2 == 0 else "user", "text": t} for i, t in enumerate(texts)]


def test_check_accepts_good_conversation():
    assert check(SPEC, turns("Hoş geldiniz.", "Siparişim 547 23 81.")) == []


def test_check_flags_missing_id_and_acting_assistant():
    problems = check(SPEC, turns("Siparişinizi kontrol ettim.", "Siparişim nerede?"))
    assert any("not said" in p for p in problems)
    assert any("assistant acts" in p for p in problems)


def test_check_flags_assistant_repeating_id():
    assert any("repeats" in p for p in check(SPEC, turns("Merhaba", "547 23 81", "547 23 81 numaralı sipariş mi?", "Evet")))


def test_selfcorr_order():
    assert check_selfcorr(SPEC, turns("Merhaba", "547 23 18, pardon 547 23 81")) == []
    assert check_selfcorr(SPEC, turns("Merhaba", "547 23 81, pardon 547 23 18")) != []


def test_say_and_notes():
    assert _say("2026-11-05") == "5 Kasım Perşembe"
    assert _say("calinti") == "kart çalındı"
    assert _clean_notes({"notes": "Yeni tarih: bu perşembe. Önce çarşamba der."}) == "Yeni tarih: bu perşembe."
