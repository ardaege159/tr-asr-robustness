from trasr.report import claimed_without_call, confirm_failure, near


def test_near_finds_misheard_read_back():
    assert near("N4Q9MW", "ANLADIM4Q9M1WKODLUREZERVASYONUNUZ")
    assert not near("N4Q9MW", "HANGIDETAYLARIOGRENMEKISTERSINIZ")


def _row(*turns):
    return {"dialogue": [{"role": r, "text": t} for r, t in turns]}


def test_caller_accepting_a_wrong_read_back():
    row = _row(("assistant", "V-2-M-7-H kodlu rezervasyonu iptal ediyorum, onaylıyor musunuz?"), ("user", "Evet, onaylıyorum."))
    assert confirm_failure(row, ["V2M7HQ"]) == "accepted"


def test_misheard_correction():
    row = _row(
        ("assistant", "U9S2L kodlu rezervasyona bagaj ekleyeyim mi?"),
        ("user", "Hayır, rezervasyon kodum U9S2L'ye ekte olacaktır."),
    )
    assert confirm_failure(row, ["U9P2LX"]) == "misheard"


def test_claim_rule_ignores_offers():
    offer = {"reply": "İptal işleminizi ücretsiz olarak gerçekleştirebilirim. Onaylıyor musunuz?", "calls": []}
    claim = {"reply": "Siparişinizi iptal ediyorum.", "calls": [{"tool": "siparis_sorgula", "args": {}}]}
    assert not claimed_without_call(offer)
    assert claimed_without_call(claim)
