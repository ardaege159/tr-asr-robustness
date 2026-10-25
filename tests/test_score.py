import json
from pathlib import Path

import pytest

from trasr.score import arg_matches, norm_date, norm_id, score

SPECS = {
    s["id"]: s
    for s in map(json.loads, (Path(__file__).parents[1] / "data" / "specs.jsonl").read_text(encoding="utf-8").splitlines())
}


def call(tool, **args):
    return {"tool": tool, "args": args}


def test_specs_are_well_formed():
    for s in SPECS.values():
        assert s["company"] in {"eticaret", "banka", "seyahat"}
        c = s.get("correction")
        if c and s["expected"]:
            exp_args = {k: v for e in s["expected"] for k, v in e["args"].items()}
            assert c["arg"] in exp_args, s["id"]
            assert c["wrong"] != exp_args[c["arg"]], s["id"]


def test_correct_call():
    assert score(SPECS["ec-001"], [call("siparis_sorgula", siparis_no="5472381")]).outcome == "correct"


def test_id_normalisation():
    assert norm_id("k7m-2xa") == "K7M2XA"
    assert norm_id("dı9a3vn") == "DI9A3VN"
    assert arg_matches("pnr", "K7M2XA", "K7M 2XA")
    assert not arg_matches("pnr", "K7M2XA", "K7N2XA")


def test_wrong_digit_is_wrong_args():
    s = score(SPECS["ec-001"], [call("siparis_sorgula", siparis_no="5472388")])
    assert s.outcome == "wrong_args"


def test_took_first_only_in_selfcorr():
    calls = [call("siparis_sorgula", siparis_no="5472318")]
    assert score(SPECS["ec-001"], calls, "selfcorr").outcome == "took_first"
    assert score(SPECS["ec-001"], calls, "asr").outcome == "wrong_args"


def test_numbers_and_dates():
    spec = SPECS["bk-004"]
    ok = call("islem_itiraz", tutar_tl="2.450,00", tarih="31.10.2026", isyeri="teknosa")
    assert score(spec, [ok]).outcome == "correct"
    first = call("islem_itiraz", tutar_tl=2540, tarih="2026-10-31", isyeri="Teknosa")
    assert score(spec, [first], "selfcorr").outcome == "took_first"
    assert norm_date("2026/11/5") == "2026-11-05"
    assert norm_date("yarın") is None


def test_no_action_expected():
    spec = SPECS["ec-006"]
    assert score(spec, []).outcome == "correct"
    assert score(spec, [call("siparis_sorgula", siparis_no="4415807")]).outcome == "correct"
    assert score(spec, [call("iade_baslat", siparis_no="4415807", sebep="diger")]).outcome == "policy_breach"


def test_missing_and_wrong_tool():
    spec = SPECS["bk-005"]
    freeze = call("kart_dondur", kart_son4="7764", sebep="supheli_islem")
    assert score(spec, [freeze, call("insana_aktar", sebep="dolandırıcılık")]).outcome == "correct"
    assert score(spec, [freeze]).outcome == "missed"
    assert score(SPECS["bk-001"], [call("insana_aktar", sebep="kart kayıp")]).outcome == "wrong_tool"


def test_return_reason_is_checked():
    spec = SPECS["ec-005"]
    assert score(spec, [call("iade_baslat", siparis_no="2281937", sebep="yanlis_urun")]).outcome == "correct"
    assert score(spec, [call("iade_baslat", siparis_no="2281937", sebep="begenmedim")], "selfcorr").outcome == "took_first"


def test_extra_action_is_breach():
    spec = SPECS["bk-001"]
    calls = [call("kart_dondur", kart_son4="4821", sebep="kayip"), call("insana_aktar", sebep="x")]
    assert score(spec, calls).outcome == "policy_breach"


@pytest.mark.parametrize("got", ["Bağdat Cad. No 214 D 5 Kadıköy İstanbul", "bagdat caddesi no 214 daire 5 kadikoy istanbul"])
def test_fuzzy_address(got):
    spec = SPECS["ec-004"]
    assert score(spec, [call("adres_degistir", siparis_no="6634092", yeni_adres=got)]).outcome == "correct"
